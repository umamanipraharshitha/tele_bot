import os
import re
import io
import json
import time
import threading

import requests
import pandas as pd
from dotenv import load_dotenv
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from google import genai

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8080")
PORT = int(os.getenv("PORT", "8080"))

LOG_FILE = "run.jsonl"
MAX_HISTORY = 10
MAX_DATA_ROWS = 10000
REQUEST_TIMEOUT = 30

if not BOT_TOKEN:
    raise SystemExit("ERROR: TELEGRAM_BOT_TOKEN is missing from .env")

if not GEMINI_API_KEY:
    raise SystemExit("ERROR: GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=GEMINI_API_KEY)
conversation_history = {}


def add_history(chat_id, role, content):
    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": role, "content": content})
    conversation_history[chat_id] = history[-MAX_HISTORY:]


def get_history(chat_id):
    return conversation_history.get(chat_id, [])


def extract_urls(text):
    urls = re.findall(r'https?://[^\s<>"\']+', text)
    return [u.rstrip(".,;:!?)]}>") for u in urls]


def dataframe_summary(df):
    summary = {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing_values": {
            str(c): int(v) for c, v in df.isna().sum().items()
        },
    }

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        summary["numeric_summary"] = (
            numeric.describe().round(4).to_dict()
        )

    sample = df.head(20).where(pd.notnull(df.head(20)), None)
    summary["sample_rows"] = sample.to_dict(orient="records")
    return summary


def download_dataset(url):
    try:
        print(f"Downloading dataset: {url}")
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "DataAnalystTelegramBot/1.0"},
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        raw = response.content
        clean_url = url.lower().split("?")[0]

        if "csv" in content_type or clean_url.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw), nrows=MAX_DATA_ROWS)
            return {"url": url, "type": "csv", "data": df}

        if "json" in content_type or clean_url.endswith(".json"):
            obj = response.json()
            try:
                df = pd.json_normalize(obj).head(MAX_DATA_ROWS)
                return {"url": url, "type": "json", "data": df}
            except Exception:
                return {"url": url, "type": "json", "data": obj}

        try:
            df = pd.read_csv(io.BytesIO(raw), nrows=MAX_DATA_ROWS)
            return {"url": url, "type": "csv", "data": df}
        except Exception:
            pass

        try:
            obj = response.json()
            try:
                df = pd.json_normalize(obj).head(MAX_DATA_ROWS)
                return {"url": url, "type": "json", "data": df}
            except Exception:
                return {"url": url, "type": "json", "data": obj}
        except Exception:
            return {
                "url": url,
                "type": "text",
                "data": response.text[:100000],
            }

    except Exception as e:
        print(f"Dataset download failed: {e}")
        return {"url": url, "type": "error", "error": str(e)}


def prepare_data_context(question):
    urls = extract_urls(question)
    datasets = []

    for url in urls[:3]:
        loaded = download_dataset(url)

        if loaded["type"] == "error":
            datasets.append({
                "url": url,
                "status": "failed_to_download",
                "error": loaded.get("error", "unknown error"),
            })
            continue

        data = loaded["data"]

        if isinstance(data, pd.DataFrame):
            datasets.append({
                "url": url,
                "type": loaded["type"],
                "status": "loaded",
                "summary": dataframe_summary(data),
            })
        else:
            datasets.append({
                "url": url,
                "type": loaded["type"],
                "status": "loaded",
                "data": data,
            })

    return {"datasets": datasets}


def solve_question(chat_id, question):
    history_text = ""

    for item in get_history(chat_id):
        history_text += (
            f"{item['role'].upper()}: {item['content']}\n\n"
        )

    data_context = prepare_data_context(question)

    prompt = f"""
You are an expert data analyst.

Solve the data-analysis question below.

The user specifies the exact JSON shape they want. Follow that shape exactly.

You must return ONLY the JSON value that belongs inside the
outer "answer" field. Do not return "answer" or "log_url".

Example:
If the requested answer is {{"state": "<state name>"}}
return exactly something like:
{{"state": "Assam"}}

Rules:
1. Answer the CURRENT question.
2. Use previous messages for multi-turn questions.
3. Use supplied dataset information when available.
4. Perform calculations carefully.
5. Do not invent data.
6. Follow the exact requested JSON shape.
7. Return valid JSON only.
8. No Markdown.
9. No ```json fences.
10. No explanations outside the JSON.

PREVIOUS CONVERSATION:
{history_text}

CURRENT QUESTION:
{question}

PUBLIC DATASETS:
{json.dumps(data_context, ensure_ascii=False, default=str)}

Return ONLY the JSON answer.
"""

    for attempt in range(5):
        try:
            print(
                f"Calling Gemini 3.5 Flash "
                f"(attempt {attempt + 1}/5)"
            )

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
            )

            text = response.text.strip()

            if text.startswith("```"):
                lines = text.splitlines()
                if lines:
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

            answer = json.loads(text)

            add_history(chat_id, "user", question)
            add_history(
                chat_id,
                "assistant",
                json.dumps(answer, ensure_ascii=False),
            )

            return answer

        except json.JSONDecodeError:
            print("Gemini returned invalid JSON.")
            return {"error": "Model returned invalid JSON"}

        except Exception as e:
            error = str(e)
            print(f"Gemini error: {error}")

            if (
                "429" in error
                or "quota" in error.lower()
                or "resourceexhausted" in error.lower()
            ):
                wait_time = 10 * (attempt + 1)
                print(f"Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if attempt < 4:
                time.sleep(3)
                continue

            return {"error": error}

    return {"error": "Gemini request failed after retries"}


def log_run(chat_id, question, response):
    entry = {
        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "chat_id": chat_id,
        "question": question,
        "response": response,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(entry, ensure_ascii=False) + "\n"
        )


class ReusableTCPServer(TCPServer):
    allow_reuse_address = True


class LogHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/run.jsonl":
            self.send_response(200)
            self.send_header("Content-Type", "application/jsonl")
            self.send_header(
                "Access-Control-Allow-Origin", "*"
            )
            self.end_headers()

            try:
                with open(LOG_FILE, "rb") as file:
                    self.wfile.write(file.read())
            except FileNotFoundError:
                self.wfile.write(b"")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(
                b"Data Analyst Telegram Bot is running."
            )

    def log_message(self, format, *args):
        pass


def start_http_server():
    try:
        server = ReusableTCPServer(
            ("0.0.0.0", PORT),
            LogHandler,
        )

        print(f"HTTP server running on port {PORT}")
        print(
            "Log URL:",
            f"{PUBLIC_URL.rstrip('/')}/run.jsonl",
        )

        server.serve_forever()

    except OSError as e:
        print(f"HTTP server error: {e}")


def send_message(chat_id, text):
    url = (
        "https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": text},
            timeout=30,
        )

        if response.status_code != 200:
            print("Telegram send error:", response.text)

    except Exception as e:
        print("Telegram send exception:", e)


def poll_updates():
    offset = None

    print("Telegram polling started...")

    while True:
        try:
            url = (
                "https://api.telegram.org/"
                f"bot{BOT_TOKEN}/getUpdates"
            )

            params = {"timeout": 30}

            if offset is not None:
                params["offset"] = offset

            response = requests.get(
                url,
                params=params,
                timeout=40,
            )

            if response.status_code != 200:
                print(
                    "Telegram polling error:",
                    response.status_code,
                    response.text,
                )
                time.sleep(5)
                continue

            data = response.json()

            if not data.get("ok"):
                print("Telegram API error:", data)
                time.sleep(5)
                continue

            for update in data.get("result", []):
                update_id = update.get("update_id")

                if update_id is not None:
                    offset = update_id + 1

                message = update.get("message")
                if not message:
                    continue

                chat_id = message.get(
                    "chat", {}
                ).get("id")

                text = message.get("text", "")

                if not chat_id or not text:
                    continue

                print("\n" + "=" * 60)
                print("Received:", text)

                answer = solve_question(
                    chat_id,
                    text,
                )

                log_url = (
                    f"{PUBLIC_URL.rstrip('/')}/run.jsonl"
                )

                final_response = {
                    "answer": answer,
                    "log_url": log_url,
                }

                final_text = json.dumps(
                    final_response,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                log_run(
                    chat_id,
                    text,
                    final_response,
                )

                send_message(
                    chat_id,
                    final_text,
                )

                print("Sent:", final_text)

        except Exception as e:
            print(
                "Polling exception:",
                type(e).__name__,
                e,
            )
            time.sleep(5)


if __name__ == "__main__":
    print("=" * 60)
    print("DATA ANALYST TELEGRAM BOT")
    print("=" * 60)
    print(
        "Telegram token:",
        "OK" if BOT_TOKEN else "MISSING",
    )
    print(
        "Gemini API key:",
        "OK" if GEMINI_API_KEY else "MISSING",
    )
    print("Gemini model: gemini-3.5-flash")
    print("Port:", PORT)
    print("Public URL:", PUBLIC_URL)
    print("=" * 60)

    threading.Thread(
        target=start_http_server,
        daemon=True,
    ).start()

    poll_updates()
