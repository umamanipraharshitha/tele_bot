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


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# On Render, set PUBLIC_URL to your Render service URL.
# Example:
# https://your-bot-name.onrender.com
PUBLIC_URL = os.getenv(
    "PUBLIC_URL",
    "http://localhost:8080"
)

# Render provides PORT automatically.
PORT = int(
    os.getenv("PORT", "8080")
)

LOG_FILE = "run.jsonl"

# Maximum number of previous messages kept per Telegram chat.
MAX_HISTORY = 10

# Don't download enormous datasets into memory.
MAX_DATA_ROWS = 10000

REQUEST_TIMEOUT = 30


# ============================================================
# CHECK ENVIRONMENT
# ============================================================

if not BOT_TOKEN:
    raise SystemExit(
        "ERROR: TELEGRAM_BOT_TOKEN is missing from .env"
    )

if not GEMINI_API_KEY:
    raise SystemExit(
        "ERROR: GEMINI_API_KEY is missing from .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_history = {}


def add_history(chat_id, role, content):
    """
    Store conversation messages for multi-turn questions.
    """

    history = conversation_history.setdefault(
        chat_id,
        []
    )

    history.append({
        "role": role,
        "content": content
    })

    # Keep memory bounded.
    conversation_history[chat_id] = history[
        -MAX_HISTORY:
    ]


def get_history(chat_id):
    return conversation_history.get(
        chat_id,
        []
    )


# ============================================================
# URL EXTRACTION
# ============================================================

def extract_urls(text):
    """
    Find public HTTP/HTTPS URLs inside the question.
    """

    urls = re.findall(
        r'https?://[^\s<>"\']+',
        text
    )

    # Remove common punctuation accidentally attached
    # to URLs in natural-language questions.
    cleaned = []

    for url in urls:
        url = url.rstrip(
            ".,;:!?)]}>"
        )

        cleaned.append(url)

    return cleaned


# ============================================================
# DOWNLOAD PUBLIC DATA
# ============================================================

def download_dataset(url):
    """
    Download a public dataset.

    Supports:
    - CSV
    - JSON
    - plain text
    """

    try:

        print(
            f"Downloading dataset: {url}"
        )

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent":
                    "DataAnalystTelegramBot/1.0"
            }
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        raw = response.content

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        if (
            "csv" in content_type
            or url.lower()
            .split("?")[0]
            .endswith(".csv")
        ):

            df = pd.read_csv(
                io.BytesIO(raw),
                nrows=MAX_DATA_ROWS
            )

            return {
                "url": url,
                "type": "csv",
                "data": df
            }

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        if (
            "json" in content_type
            or url.lower()
            .split("?")[0]
            .endswith(".json")
        ):

            obj = response.json()

            try:

                df = pd.json_normalize(obj)

                if len(df) > MAX_DATA_ROWS:
                    df = df.head(
                        MAX_DATA_ROWS
                    )

                return {
                    "url": url,
                    "type": "json",
                    "data": df
                }

            except Exception:

                return {
                    "url": url,
                    "type": "json",
                    "data": obj
                }

        # ----------------------------------------------------
        # Try CSV even if server gave bad Content-Type
        # ----------------------------------------------------

        try:

            df = pd.read_csv(
                io.BytesIO(raw),
                nrows=MAX_DATA_ROWS
            )

            return {
                "url": url,
                "type": "csv",
                "data": df
            }

        except Exception:
            pass

        # ----------------------------------------------------
        # Try JSON
        # ----------------------------------------------------

        try:

            obj = response.json()

            try:

                df = pd.json_normalize(obj)

                if len(df) > MAX_DATA_ROWS:
                    df = df.head(
                        MAX_DATA_ROWS
                    )

                return {
                    "url": url,
                    "type": "json",
                    "data": df
                }

            except Exception:

                return {
                    "url": url,
                    "type": "json",
                    "data": obj
                }

        except Exception:
            pass

        # ----------------------------------------------------
        # Plain text / HTML
        # ----------------------------------------------------

        text = response.text

        return {
            "url": url,
            "type": "text",
            "data": text[:100000]
        }

    except Exception as e:

        print(
            f"Dataset download failed: {e}"
        )

        return {
            "url": url,
            "type": "error",
            "error": str(e)
        }


# ============================================================
# DATAFRAME SUMMARY
# ============================================================

def dataframe_summary(df):
    """
    Create a compact representation of a dataframe
    for Gemini.
    """

    summary = {
        "rows": int(len(df)),
        "columns": [
            str(column)
            for column in df.columns
        ],

        "dtypes": {
            str(column): str(dtype)
            for column, dtype
            in df.dtypes.items()
        }
    }

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    summary["missing_values"] = {
        str(column): int(value)
        for column, value
        in df.isna().sum().items()
    }

    # --------------------------------------------------------
    # Numeric statistics
    # --------------------------------------------------------

    numeric = df.select_dtypes(
        include="number"
    )

    if not numeric.empty:

        summary["numeric_summary"] = (
            numeric
            .describe()
            .round(4)
            .to_dict()
        )

    # --------------------------------------------------------
    # Sample rows
    # --------------------------------------------------------

    sample = df.head(20)

    sample = sample.where(
        pd.notnull(sample),
        None
    )

    summary["sample_rows"] = (
        sample.to_dict(
            orient="records"
        )
    )

    return summary


# ============================================================
# PREPARE DATA CONTEXT
# ============================================================

def prepare_data_context(question):

    urls = extract_urls(
        question
    )

    if not urls:

        return {
            "datasets": [],
            "note":
                "No public dataset URL detected."
        }

    datasets = []

    # Process at most 3 URLs.
    for url in urls[:3]:

        loaded = download_dataset(
            url
        )

        if loaded is None:

            datasets.append({
                "url": url,
                "status":
                    "failed_to_download"
            })

            continue

        if loaded["type"] == "error":

            datasets.append({
                "url": url,
                "status":
                    "failed_to_download",
                "error":
                    loaded.get(
                        "error",
                        "unknown error"
                    )
            })

            continue

        data = loaded["data"]

        # ----------------------------------------------------
        # DataFrame
        # ----------------------------------------------------

        if isinstance(
            data,
            pd.DataFrame
        ):

            datasets.append({
                "url": url,
                "type":
                    loaded["type"],
                "status":
                    "loaded",
                "summary":
                    dataframe_summary(data)
            })

        # ----------------------------------------------------
        # JSON / text
        # ----------------------------------------------------

        else:

            datasets.append({
                "url": url,
                "type":
                    loaded["type"],
                "status":
                    "loaded",
                "data":
                    data
            })

    return {
        "datasets": datasets
    }


# ============================================================
# GEMINI DATA ANALYSIS
# ============================================================

def solve_question(
    chat_id,
    question
):
    """
    Ask Gemini to solve the current data-analysis question.

    Gemini returns ONLY the value belonging inside
    the required outer "answer" field.
    """

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    history = get_history(
        chat_id
    )

    history_text = ""

    for item in history:

        history_text += (
            f"{item['role'].upper()}: "
            f"{item['content']}\n\n"
        )

    # --------------------------------------------------------
    # Download datasets
    # --------------------------------------------------------

    data_context = (
        prepare_data_context(
            question
        )
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an expert data analyst.

You are solving a data-analysis question sent through Telegram.

The Telegram bot MUST ultimately return exactly one JSON object:

{{
  "answer": <answer>,
  "log_url": "<public URL>"
}}

The Telegram wrapper creates the outer object.

Therefore, YOU MUST RETURN ONLY THE JSON VALUE THAT BELONGS
INSIDE THE "answer" FIELD.

For example, if the user asks for:

{{"state": "<state name>"}}

you must return:

{{"state": "Assam"}}

Do NOT return:

{{"answer": {{"state": "Assam"}}}}

Do NOT return a log_url.

------------------------------------------------------------
PREVIOUS CONVERSATION
------------------------------------------------------------

{history_text}

------------------------------------------------------------
CURRENT QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
PUBLIC DATASETS DOWNLOADED FROM THE QUESTION
------------------------------------------------------------

{json.dumps(
    data_context,
    ensure_ascii=False,
    default=str
)}

------------------------------------------------------------
RULES
------------------------------------------------------------

1. Answer the CURRENT question.

2. Use previous conversation messages if this is
   a multi-turn question.

3. If a public dataset was successfully downloaded,
   use the supplied dataset information rather than
   guessing.

4. Perform calculations carefully.

5. Do not invent dataset values.

6. Follow the exact JSON shape requested by the user.

7. If the user requests an object, return an object.

8. If the user requests an array, return an array.

9. If the user requests a number, return a JSON number.

10. If the user requests a string, return a JSON string.

11. Return valid JSON only.

12. Do not use Markdown.

13. Do not include ```json.

14. Do not include explanations outside the JSON.

15. Do not include "answer" or "log_url" in your response.

Return ONLY the JSON answer.
"""

    # ========================================================
    # GEMINI 3.5 FLASH
    # ========================================================

    for attempt in range(5):

        try:

            print(
                f"Calling Gemini 3.5 Flash "
                f"(attempt {attempt + 1}/5)"
            )

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )

            text = response.text.strip()

            # ------------------------------------------------
            # Remove accidental Markdown fences
            # ------------------------------------------------

            if text.startswith("```"):

                lines = text.splitlines()

                if lines:
                    lines = lines[1:]

                if (
                    lines
                    and
                    lines[-1].strip()
                    == "```"
                ):

                    lines = lines[:-1]

                text = "\n".join(
                    lines
                ).strip()

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            answer = json.loads(
                text
            )

            # ------------------------------------------------
            # Save conversation
            # ------------------------------------------------

            add_history(
                chat_id,
                "user",
                question
            )

            add_history(
                chat_id,
                "assistant",
                json.dumps(
                    answer,
                    ensure_ascii=False
                )
            )

            return answer

        except json.JSONDecodeError:

            print(
                "Gemini returned invalid JSON:"
            )

            try:
                print(
                    response.text
                )
            except Exception:
                pass

            return {
                "error":
                    "Model returned invalid JSON"
            }

        except Exception as e:

            error = str(e)

            print(
                f"Gemini error: {error}"
            )

            # Retry quota/rate-limit errors.
            if (
                "429" in error
                or
                "quota" in error.lower()
                or
                "resourceexhausted"
                in error.lower()
            ):

                wait_time = (
                    10 * (attempt + 1)
                )

                print(
                    f"Waiting {wait_time}s..."
                )

                time.sleep(
                    wait_time
                )

                continue

            return {
                "error": error
            }

    return {
        "error":
            "Gemini request failed after retries"
    }


# ============================================================
# JSONL LOGGING
# ============================================================

def log_run(
    chat_id,
    question,
    response
):
    """
    Append one JSON object per run.
    """

    entry = {
        "timestamp":
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime()
            ),

        "chat_id":
            chat_id,

        "question":
            question,

        "response":
            response
    }

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                entry,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# HTTP SERVER
# ============================================================

class ReusableTCPServer(TCPServer):

    allow_reuse_address = True


class LogHandler(
    SimpleHTTPRequestHandler
):

    def do_GET(self):

        # ----------------------------------------------------
        # Public JSONL log
        # ----------------------------------------------------

        if self.path == "/run.jsonl":

            self.send_response(
                200
            )

            self.send_header(
                "Content-Type",
                "application/jsonl"
            )

            self.send_header(
                "Access-Control-Allow-Origin",
                "*"
            )

            self.end_headers()

            try:

                with open(
                    LOG_FILE,
                    "rb"
                ) as file:

                    self.wfile.write(
                        file.read()
                    )

            except FileNotFoundError:

                self.wfile.write(
                    b""
                )

        # ----------------------------------------------------
        # Health check
        # ----------------------------------------------------

        else:

            self.send_response(
                200
            )

            self.send_header(
                "Content-Type",
                "text/plain"
            )

            self.end_headers()

            self.wfile.write(
                b"Data Analyst Telegram Bot is running."
            )

    def log_message(
        self,
        format,
        *args
    ):
        # Don't print HTTP access logs.
        pass


def start_http_server():

    try:

        server = ReusableTCPServer(
            ("0.0.0.0", PORT),
            LogHandler
        )

        print(
            f"HTTP server running on port {PORT}"
        )

        print(
            "Public log URL:",
            f"{PUBLIC_URL.rstrip('/')}/run.jsonl"
        )

        server.serve_forever()

    except OSError as e:

        print(
            f"HTTP server error: {e}"
        )


# ============================================================
# TELEGRAM API
# ============================================================
# TELEGRAM API
# ============================================================

def send_message(
    chat_id,
    text
):

    url = (
        "https://api.telegram.org/"
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
                response.text
            )

    except Exception as e:

        print(
            "Telegram send exception:",
            e
        )


# ============================================================
# TELEGRAM POLLING
# ============================================================

def poll_updates():

    offset = None

    print(
        "Telegram polling started..."
    )

    while True:

        try:

            url = (
                "https://api.telegram.org/"
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

            # ------------------------------------------------
            # HTTP error
            # ------------------------------------------------

            if response.status_code != 200:

                print(
                    "Telegram polling error:",
                    response.status_code,
                    response.text
                )

                time.sleep(
                    5
                )

                continue

            data = response.json()

            # ------------------------------------------------
            # API error
            # ------------------------------------------------

            if not data.get("ok"):

                print(
                    "Telegram API error:",
                    data
                )

                time.sleep(
                    5
                )

                continue

            # ------------------------------------------------
            # Process updates
            # ------------------------------------------------

            for update in data.get(
                "result",
                []
            ):

                update_id = update.get(
                    "update_id"
                )

                if update_id is not None:

                    offset = (
                        update_id + 1
                    )

                message = update.get(
                    "message"
                )

                if not message:
                    continue

                chat_id = (
                    message
                    .get("chat", {})
                    .get("id")
                )

                text = message.get(
                    "text",
                    ""
                )

                if not chat_id:
                    continue

                if not text:
                    continue

                print(
                    "\n"
                    + "=" * 60
                )

                print(
                    "Received:",
                    text
                )

                # ------------------------------------------------
                # Solve
                # ------------------------------------------------

                answer = solve_question(
                    chat_id,
                    text
                )

                # ------------------------------------------------
                # Public log URL
                # ------------------------------------------------

                log_url = (
                    f"{PUBLIC_URL.rstrip('/')}"
                    "/run.jsonl"
                )

                # ------------------------------------------------
                # REQUIRED FINAL RESPONSE
                # ------------------------------------------------

                final_response = {
                    "answer":
                        answer,

                    "log_url":
                        log_url
                }

                # Compact JSON is safest for Telegram.
                final_text = json.dumps(
                    final_response,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":"
                    )
                )

                # ------------------------------------------------
                # Log
                # ------------------------------------------------

                log_run(
                    chat_id,
                    text,
                    final_response
                )

                # ------------------------------------------------
                # Send
                # ------------------------------------------------

                send_message(
                    chat_id,
                    final_text
                )

                print(
                    "Sent:",
                    final_text
                )

        except Exception as e:

            print(
                "Polling exception:",
                type(e).__name__,
                e
            )

            time.sleep(
                5
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "DATA ANALYST TELEGRAM BOT"
    )
    print("=" * 60)

    print(
        "Telegram token:",
        "OK"
        if BOT_TOKEN
        else "MISSING"
    )

    print(
        "Gemini API key:",
        "OK"
        if GEMINI_API_KEY
        else "MISSING"
    )

    print(
        "Gemini model:",
        "gemini-3.5-flash"
    )

    print(
        "Port:",
        PORT
    )

    print(
        "Public URL:",
        PUBLIC_URL
    )

    print("=" * 60)

    # Start public log server.
    threading.Thread(
        target=start_http_server,
        daemon=True
    ).start()

    # Start Telegram bot.
    poll_updates()
