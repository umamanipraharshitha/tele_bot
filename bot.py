import os
import json
import time
import requests
import threading
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import google.generativeai as genai

# Configuration
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PUBLIC_URL = os.environ.get("PUBLIC_URL", "https://your-bot-app.onrender.com")
PORT = int(os.environ.get("PORT", 8080))

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# HTTP Server to serve run.jsonl
class LogHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/run.jsonl':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json-seq')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                with open('run.jsonl', 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"")
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Bot is running. Log file is at <a href='/run.jsonl'>/run.jsonl</a>")

def start_http_server():
    server = TCPServer(('0.0.0.0', PORT), LogHandler)
    print(f"HTTP Server started on port {PORT}")
    server.serve_forever()

# Log a run to run.jsonl
def log_run(question, response_json):
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": question,
        "response": response_json
    }
    with open("run.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Call Gemini LLM to solve data analysis questions with retries on 429
def solve_question(question_text):
    prompt = f"""
You are a highly skilled Data Analyst.
You will receive a data-analysis question, possibly referencing public datasets like MOSPI or inline data.
Solve the question carefully and return ONLY the raw JSON object representing the answer.
Do not include markdown code block formatting (like ```json). Just the raw JSON.

Example output:
{{"state": "Assam"}}

Question:
{question_text}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    for attempt in range(5):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Clean potential markdown formatting
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            return json.loads(text)
        except Exception as e:
            err_str = str(e)
            print(f"Error in solve_question (attempt {attempt+1}): {err_str}")
            if "429" in err_str or "Quota exceeded" in err_str or "ResourceExhausted" in err_str:
                time.sleep(15)
                continue
            return {"error": err_str}
    return {"error": "Exhausted all retries due to quota limits"}

# Send message helper
def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending message: {e}")

# Telegram updates polling loop
def poll_updates():
    offset = None
    print("Telegram polling loop started...")
    while True:
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            params = {"timeout": 30}
            if offset:
                params["offset"] = offset
            r = requests.get(url, params=params)
            if r.status_code == 200:
                results = r.json().get("result", [])
                for update in results:
                    update_id = update.get("update_id")
                    offset = update_id + 1
                    
                    message = update.get("message")
                    if not message:
                        continue
                    
                    chat_id = message.get("chat").get("id")
                    text = message.get("text", "")
                    
                    if not text:
                        continue
                    
                    print(f"Received message: {text}")
                    
                    # Compute answer
                    answer_obj = solve_question(text)
                    
                    # Prepare log URL
                    log_url = f"{PUBLIC_URL.rstrip('/')}/run.jsonl"
                    
                    # Final response JSON
                    final_response = {
                        "answer": answer_obj,
                        "log_url": log_url
                    }
                    
                    # Log run
                    log_run(text, final_response)
                    
                    # Send response
                    send_message(chat_id, json.dumps(final_response))
            else:
                print(f"Error polling: {r.status_code} {r.text}")
        except Exception as e:
            print(f"Exception in polling: {e}")
        time.sleep(1)

if __name__ == "__main__":
    if not BOT_TOKEN:
        print("WARNING: TELEGRAM_BOT_TOKEN environment variable not set.")
    
    # Start HTTP Server in background
    threading.Thread(target=start_http_server, daemon=True).start()
    
    # Start polling loop
    poll_updates()
