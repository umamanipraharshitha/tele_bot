import os
import json
import time
import threading
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

import requests
from dotenv import load_dotenv
import google.generativeai as genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8080")
PORT = int(os.getenv("PORT", "8080"))

LOG_FILE = "run.jsonl"

# Store recent conversation history per Telegram chat
conversation_history = {}

# Limit history so it doesn't grow forever
MAX_HISTORY = 10


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if not BOT_TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN is not set in .env")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY is not set in .env")

genai.configure(api_key=GEMINI_API_KEY)


# ============================================================
# PUBLIC LOG SERVER
# ============================================================

class ReusableTCPServer(TCPServer):
    allow_reuse_address = True


class LogHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/run.jsonl":

            self.send_response(200)
            self.send_header("Content-Type", "application/jsonl")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                with open(LOG_FILE, "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                pass

        else:

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()

            self.wfile.write(
                b"Data Analyst Telegram Bot is running.<br>"
                b"Log: <a href='/run.jsonl'>/run.jsonl</a>"
            )

    def log_message(self, format, *args):
        # Keep HTTP access logs quiet
        pass


def start_http_server():

    try:
        server = ReusableTCPServer(
            ("0.0.0.0", PORT),
            LogHandler
        )

        print(f"HTTP server started on port {PORT}")
        print(f"Log URL: {PUBLIC_URL.rstrip('/')}/run.jsonl")

        server.serve_forever()

    except OSError as e:

        print(f"HTTP server error: {e}")
        print(
            "If running locally, another program may already "
            f"be using port {PORT}."
        )


# ============================================================
# LOGGING
# ============================================================

def log_run(question, response_json):

    entry = {
        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime()
        ),
        "question": question,
        "response": response_json
    }

    try:

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Logging error: {e}")


# ============================================================
# GEMINI
# ============================================================

def solve_question(chat_id, question_text):

    history = conversation_history.get(chat_id, [])

    history_text = ""

    for item in history:
        history_text += (
            f"USER: {item['user']}\n"
            f"ASSISTANT: {json.dumps(item['assistant'])}\n\n"
        )

    prompt = f"""
You are an expert data analyst.

Your job is to answer the user's data-analysis question.

The user may:
- provide data directly in the message
- provide CSV/JSON data
- provide a public URL to a dataset
- refer to information from earlier messages
- ask calculations, comparisons, aggregations, rankings, percentages,
  averages, totals, or other data-analysis questions.

Conversation history:
{history_text}

Current user question:
{question_text}

IMPORTANT OUTPUT RULES:

1. Return ONLY a valid JSON object.
2. Do NOT use Markdown.
3. Do NOT use ```json.
4. Do NOT add explanations outside the JSON.
5. The JSON you return is the VALUE that will go inside the outer
   "answer" field.
6. Therefore, DO NOT create another "answer" or "log_url" field.
7. Follow the exact answer shape requested by the user.

For example, if the user asks:

Reply with ONLY this JSON shape:
{{"state": "<state name>"}}

then return:

{{"state": "Assam"}}

If the user asks:

{{"value": <number>}}

return:

{{"value": 123}}

Do not return:

{{"answer": {{"state": "Assam"}}}}

Return only the requested inner JSON object.
"""

    model = genai.GenerativeModel("gemini-3.5-flash")

    for attempt in range(5):

        try:

            response = model.generate_content(prompt)

            text = response.text.strip()

            # Remove accidental Markdown fences
            if text.startswith("```"):

                lines = text.splitlines()

                if lines:
                    lines = lines[1:]

                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]

                text = "\n".join(lines).strip()

            answer = json.loads(text)

            # Remember conversation
            if chat_id not in conversation_history:
                conversation_history[chat_id] = []

            conversation_history[chat_id].append(
                {
                    "user": question_text,
                    "assistant": answer
                }
            )

            # Keep only recent history
            conversation_history[chat_id] = (
                conversation_history[chat_id][-MAX_HISTORY:]
            )

            return answer

        except json.JSONDecodeError as e:

            print(f"Gemini returned invalid JSON: {e}")

            return {
                "error": "Model returned invalid JSON"
            }

        except Exception as e:

            error_text = str(e)

            print(
                f"Gemini error "
                f"(attempt {attempt + 1}/5): {error_text}"
            )

            if (
                "429" in error_text
                or "quota" in error_text.lower()
                or "resourceexhausted" in error_text.lower()
            ):

                time.sleep(10 * (attempt + 1))
                continue

            return {
                "error": error_text
            }

    return {
        "error": "Gemini request failed after retries"
    }


# ============================================================
# TELEGRAM
# ============================================================

def send_message(chat_id, text):

    url = (
        f"https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": text
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:

            print(
                "Telegram send error:",
                response.status_code,
                response.text
            )

    except Exception as e:

        print(f"Error sending Telegram message: {e}")


# ============================================================
# TELEGRAM POLLING
# ============================================================

def poll_updates():

    offset = None

    print("Telegram polling loop started...")

    while True:

        try:

            url = (
                f"https://api.telegram.org/"
                f"bot{BOT_TOKEN}/getUpdates"
            )

            params = {
                "timeout": 30
            }

            if offset is not None:
                params["offset"] = offset

            response = requests.get(
                url,
                params=params,
                timeout=40
            )

            if response.status_code != 200:

                print(
                    "Telegram polling error:",
                    response.status_code,
                    response.text
                )

                time.sleep(5)
                continue

            data = response.json()

            if not data.get("ok"):

                print("Telegram API error:", data)
                time.sleep(5)
                continue

            updates = data.get("result", [])

            for update in updates:

                update_id = update.get("update_id")

                if update_id is not None:
                    offset = update_id + 1

                message = update.get("message")

                if not message:
                    continue

                chat = message.get("chat", {})
                chat_id = chat.get("id")

                text = message.get("text", "")

                if not chat_id or not text:
                    continue

                print()
                print("=" * 60)
                print("Received:")
                print(text)
                print("=" * 60)

                # ------------------------------------------------
                # Solve question
                # ------------------------------------------------

                answer_obj = solve_question(
                    chat_id,
                    text
                )

                # ------------------------------------------------
                # Public log URL
                # ------------------------------------------------

                log_url = (
                    f"{PUBLIC_URL.rstrip('/')}"
                    f"/run.jsonl"
                )

                # ------------------------------------------------
                # Required final JSON
                # ------------------------------------------------

                final_response = {
                    "answer": answer_obj,
                    "log_url": log_url
                }

                # Make sure it is exactly one JSON object
                final_text = json.dumps(
                    final_response,
                    ensure_ascii=False,
                    separators=(",", ":")
                )

                # ------------------------------------------------
                # Log
                # ------------------------------------------------

                log_run(
                    text,
                    final_response
                )

                # ------------------------------------------------
                # Send to Telegram
                # ------------------------------------------------

                send_message(
                    chat_id,
                    final_text
                )

                print("Sent:")
                print(final_text)

        except requests.RequestException as e:

            print(
                "Network error while polling Telegram:",
                e
            )

            time.sleep(5)

        except Exception as e:

            print(
                "Unexpected polling error:",
                type(e).__name__,
                e
            )

            time.sleep(5)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Data Analyst Telegram Bot")
    print("=" * 60)

    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN is missing.")
        raise SystemExit(1)

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is missing.")
        raise SystemExit(1)

    print("Telegram token: OK")
    print("Gemini API key: OK")
    print(f"Port: {PORT}")
    print(f"Public URL: {PUBLIC_URL}")
    print("=" * 60)

    # Start public HTTP server
    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True
    )

    http_thread.start()

    # Start Telegram polling
    poll_updates()
