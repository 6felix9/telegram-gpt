<p align="center">
  <img src="assets/logo.png" alt="Telegram GPT Bot logo" width="160">
</p>

<h1 align="center">Telegram GPT Bot</h1>

<p align="center">
  An AI agent that lives directly in your Telegram.
</p>

A Telegram bot with persistent chat history, token-aware context trimming, image support, and a PostgreSQL/Neon backend. It's triggered by the keyword `chatgpt` or by directly mentioning the bot, supports multiple model providers, and persists the active model in the database so model switches survive restarts.

## How It Works

Add the bot to a group (or DM it) and it quietly reads along. It only speaks when you say `chatgpt` or @mention it.

Because it has been listening the whole time, you can just ask about what already happened:

```
Nathan   the reservation is at 7, i already told everyone
Cheryl   you said 8 in the group chat yesterday
Nathan   i did NOT
Amy      chatgpt who is right, nathan or cheryl

Bot      Cheryl. Yesterday at 4:12pm Nathan wrote "table's booked for 8".
         He changed it to 7 this morning without saying anything.
```

Other things people actually use it for:

```
You      chatgpt summarise the conversation

Bot      Three of you argued about dinner time, settled on 7pm, and Amy is
         bringing dessert. Nobody booked parking.
```

```
You      [replying to a photo of an ingredients label]
         chatgpt what's the second ingredient

Bot      Sunflower oil, right after wheat flour.
```

```
You      chatgpt what's the weather in tokyo this weekend

Bot      [searches the web] Rain Saturday, clearing Sunday, 18-22C.
```

It can also put things on a timer. There is no command — just ask:

```
You      chatgpt post a summary of the day's messages every weekday at 6pm

Bot      Scheduled #3 — every weekday at 6:00pm. First run Mon 22 Sep,
         6:00pm SGT. Will run: "Summarise the day's messages in this chat."
```

It tells you the id, when it fires next, and the exact prompt it saved — the
prompt is rewritten to stand on its own, since it runs later with nobody
around to ask. "chatgpt what's scheduled here" lists them, "chatgpt cancel 3"
removes one. Times are Singapore time, and a schedule fires at most once an
hour.

What it keeps track of:

- **Every text message** in the chat, whether or not it was addressed to the bot
- **Photos** — described on arrival, so you can ask about one long after it was posted
- **Voice notes** — transcribed automatically; they never trigger a reply, but you can ask about them
- **Old conversations** — once history gets long it's compacted into a rolling summary, so the bot keeps the gist forever without blowing the context window
- **Schedules** — stored in the database, so they survive restarts and redeploys; a run missed during downtime is skipped rather than fired late

## Run It Locally

You'll need Python 3.12+, a Telegram bot token from [@BotFather](https://t.me/BotFather), your Telegram user ID from [@userinfobot](https://t.me/userinfobot), an OpenAI API key, and a PostgreSQL / [Neon](https://neon.tech/) database.

```bash
git clone https://github.com/6felix9/telegram-gpt.git
cd telegram-gpt
cp .env.example .env
```

Fill in the four required values in `.env`:

```bash
TELEGRAM_BOT_TOKEN=...   # from @BotFather
AUTHORIZED_USER_ID=...   # from @userinfobot
OPENAI_API_KEY=...
DATABASE_URL=postgresql://...
```

Everything else is optional — see the comments in `.env.example`. Set `BOT_USERNAME` if you want @mention activation, and `XAI_API_KEY` / `GEMINI_API_KEY` if you want to use Grok or Gemini models.

Then start it:

```bash
./start.sh
```

`start.sh` creates a `venv/`, installs dependencies, applies database migrations, sets up the checkpointer tables, and runs the bot.

To do it by hand instead:

```bash
pip install -r requirements.txt
alembic upgrade head
python scripts/setup_checkpointer.py
python3 bot.py
```

You can also talk to the bot without Telegram at all:

```bash
python3 scripts/chat_cli.py --chat-id test          # private-chat mode
python3 scripts/chat_cli.py --chat-id test --group  # group-chat mode
```

## Deploy on Railway

1. Create a [Neon](https://neon.tech/) database and copy its connection string.
2. Create a Railway project from this repo. Railway builds the included `Dockerfile` automatically.
3. Set the service's variables — the same four required ones as above, plus any optional keys you want:

   ```
   TELEGRAM_BOT_TOKEN, AUTHORIZED_USER_ID, OPENAI_API_KEY, DATABASE_URL
   ```

4. Set the service's **Pre-deploy Command** so the schema is applied before the bot starts:

   ```
   alembic upgrade head && python scripts/setup_checkpointer.py && python scripts/cleanup_retention.py
   ```

   These are all idempotent. The last one prunes old messages, old images, and superseded checkpoint rows.

5. Deploy. Watch it come up with `railway logs`, then message your bot on Telegram.

This repo runs two Railway environments — `dev` (tracks the `dev` branch) and `production` (tracks `main`) — each with its own Telegram bot and Neon database branch. `.github/workflows/deploy-railway.yml` deploys on every push, so no manual `railway up` is needed.

## Commands

All commands are admin-only (`AUTHORIZED_USER_ID`). Granted users can chat with the bot but can't run commands.

```
/model                      show the active model and everything available
/model gemini-3.5-flash     switch model globally, persists across restarts

/personality                show the active group persona and the options
/personality sarcastic      switch the persona used in group chats

/clear                      wipe this chat's memory (summary + recent messages)
/stats                      message count, token usage, oldest message kept

/grant 123456789            let another user talk to the bot
/revoke 123456789           take that access away
/allowlist                  who currently has access
/openbot on 2h              temporarily let anyone talk to the bot

/version                    current bot version
/help                       this list, in Telegram
```

## Built With

- **Language:** Python 3.12+
- **Bot framework:** [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- **Agent orchestration:** [LangChain / LangGraph](https://github.com/langchain-ai/langgraph) — agent loop, tool calling, checkpointed conversation memory
- **Model providers:** OpenAI, xAI, Google Gemini
- **Database:** PostgreSQL / [Neon](https://neon.tech/)
- **Migrations:** Alembic
- **Token counting:** [tiktoken](https://github.com/openai/tiktoken)
- **Web search:** Tavily, falling back to DuckDuckGo
- **Deployment:** Docker, Railway
