# Telegram Data Analyst Bot Setup

We have created the code for your bot in `bot.py`!

## Step 1: Create your Telegram Bot
1. Open Telegram and search for `@BotFather`.
2. Send the command `/newbot`.
3. Choose a name for your bot (e.g. `IITM Data Analyst Bot`).
4. Choose a username for your bot. It **MUST** end in `bot` (e.g. `aplora_data_bot` or `mharshi_data_bot`).
5. `@BotFather` will give you a **HTTP API Token** (e.g. `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`).

## Step 2: Push the code to GitHub
Push the `tds-telegram-bot` folder containing `bot.py` and `requirements.txt` to your public GitHub repository:
`https://github.com/umamanipraharshitha/aplora`

## Step 3: Run the Bot
You can run the bot on Render.com (free tier) or run it locally using `ngrok` or similar for the public URL.

### Running on Render (Recommended)
1. Sign up on [Render.com](https://render.com).
2. Create a new **Web Service** and link your GitHub repository `https://github.com/umamanipraharshitha/aplora`.
3. Set the following details:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r tds-telegram-bot/requirements.txt`
   - **Start Command**: `python tds-telegram-bot/bot.py`
4. Add the following **Environment Variables** in Render:
   - `TELEGRAM_BOT_TOKEN`: The token you got from @BotFather.
   - `GEMINI_API_KEY`: Your Gemini API key.
   - `PUBLIC_URL`: The URL Render gives you (e.g. `https://aplora-bot.onrender.com`).
5. Render will deploy the app and provide the public URL.

## Step 4: Register in the portal
Enter the following comma-separated values in your portal form:
`https://github.com/umamanipraharshitha/aplora, <YOUR_BOT_USERNAME>`
*(Replace `<YOUR_BOT_USERNAME>` with your bot's username, e.g. `aplora_data_bot`)*
