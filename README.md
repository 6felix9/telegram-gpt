# Telegram GPT Bot

An AI-powered Telegram bot with support for multiple LLM providers (OpenAI, Gemini, Groq), featuring persistent conversation history and intelligent context management.

## Features

- **Multi-Provider Support**: Choose between OpenAI, Google Gemini, or Groq
- **Keyword Activation**: Bot responds only when "chatgpt" is mentioned
- **Persistent Conversation History**: SQLite-backed storage for reliable conversation tracking
- **Token-Aware Context Management**: Intelligent trimming to stay within model limits
- **Benchmark Harness**: CLI tool for testing provider performance
- **Docker-Ready**: Production deployment with Docker and docker-compose
- **Private Authorization**: Single-user access control with grant/revoke commands
- **Error Resilient**: Comprehensive error handling for all edge cases

## Architecture

- **Python 3.12+** with modern async/await patterns
- **SQLite** with WAL mode for concurrent access
- **tiktoken** for accurate token counting (OpenAI/Groq)
- **python-telegram-bot 21.7** for Telegram API
- **Pluggable LLM Providers**: OpenAI, Google Gemini, Groq
- **Provider Interface**: Extensible architecture for adding new providers

## Quick Start

### Prerequisites

- Python 3.12 or higher
- Telegram account
- API key for at least one provider:
  - OpenAI API key, or
  - Google Gemini API key, or
  - Groq API key

### Local Development

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```

4. **Edit `.env` with your credentials**
   - `TELEGRAM_BOT_TOKEN`: Get from [@BotFather](https://t.me/BotFather)
   - `PROVIDER`: Choose `openai`, `gemini`, or `groq`
   - API Key (choose one based on provider):
     - `OPENAI_API_KEY`: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
     - `GEMINI_API_KEY`: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
     - `GROQ_API_KEY`: Get from [Groq Console](https://console.groq.com/keys)
   - `AUTHORIZED_USER_ID`: Get from [@userinfobot](https://t.me/userinfobot)

5. **Run the bot**
   ```bash
   python bot.py
   ```

### Docker Deployment

1. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Build and run**
   ```bash
   docker-compose up -d
   ```

3. **View logs**
   ```bash
   docker-compose logs -f
   ```

4. **Stop the bot**
   ```bash
   docker-compose down
   ```

## Getting Credentials

### Telegram Bot Token

1. Open Telegram and search for [@BotFather](https://t.me/BotFather)
2. Send `/newbot` command
3. Follow the prompts to create your bot
4. Copy the token provided

### Your Telegram User ID

1. Open Telegram and search for [@userinfobot](https://t.me/userinfobot)
2. Send `/start` command
3. Copy your user ID (numeric value)

### LLM Provider API Keys

Choose at least one provider:

**OpenAI**
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Create a new API key
4. Copy the key immediately (you won't see it again)

**Google Gemini**
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key

**Groq**
1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign in or create an account
3. Create a new API key
4. Copy the key

## Usage

### Basic Conversation

Send a message containing "chatgpt" followed by your question:

```
chatgpt what is the weather like?
chatgpt tell me a joke
ChatGPT explain quantum computing
```

The keyword is case-insensitive and will be removed from your prompt.

### Commands

- `/clear` - Clear conversation history for current chat
- `/stats` - Show chat statistics (message count, tokens used, etc.)
- `/grant <user_id>` - Grant access to another user (main authorized user only)
- `/revoke <user_id>` - Revoke access from a user (main authorized user only)

### Authorization

Only the user specified in `AUTHORIZED_USER_ID` can use the bot. Other users will receive "Sorry, you have no access to me."

### Private Chats vs Groups

The bot works identically in both private chats and group chats:
- In groups, only messages containing "chatgpt" trigger the bot
- Authorization is per-user, not per-chat
- Each chat maintains its own conversation history

## Configuration

All configuration is done via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Required | Your bot token from BotFather |
| `PROVIDER` | `openai` | LLM provider: `openai`, `gemini`, or `groq` |
| `OPENAI_API_KEY` | - | OpenAI API key (required if PROVIDER=openai) |
| `GEMINI_API_KEY` | - | Gemini API key (required if PROVIDER=gemini) |
| `GROQ_API_KEY` | - | Groq API key (required if PROVIDER=groq) |
| `AUTHORIZED_USER_ID` | Required | Telegram user ID allowed to use bot |
| `MODEL` | Provider default | Model name (see Provider Models below) |
| `MAX_CONTEXT_TOKENS` | `16000` | Maximum tokens in conversation context |
| `TIMEOUT` | `60` | API request timeout in seconds |
| `DATABASE_PATH` | `data/messages.db` | SQLite database file path |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

If `MODEL` is blank, the provider’s default model is applied automatically (see Provider Models below).

## LLM Providers

### OpenAI

**Default Model**: `gpt-4o-mini`

**Supported Models**:
- `gpt-4o-mini` (recommended for cost/performance, 128K context)
- `gpt-4o` (most capable, 128K context)
- `gpt-3.5-turbo` (faster, cheaper, 16K context)
- `gpt-4-turbo` (previous generation flagship, 128K context)
- `gpt-4` (legacy, 8K context)

**Configuration Example**:
```bash
PROVIDER=openai
OPENAI_API_KEY=sk-...
MODEL=gpt-4o-mini
```

**Pricing**: ~$0.15/$0.60 per 1M tokens (input/output)

### Google Gemini

**Default Model**: `gemini-2.5-flash-preview-09-2025`

**Supported Models**:
- `gemini-2.5-flash-preview-09-2025` (latest, 1M context)
- `gemini-1.5-pro` (high capability, 2M context)
- `gemini-1.5-flash` (fast, 1M context)

**Configuration Example**:
```bash
PROVIDER=gemini
GEMINI_API_KEY=AIza...
MODEL=gemini-2.5-flash-preview-09-2025
```

**Pricing**: Free tier available, then ~$0.075/$0.30 per 1M tokens

### Groq

**Default Model**: `llama-3.3-70b-versatile`

**Supported Models**:
- `llama-3.3-70b-versatile` (recommended, 128K context)
- `llama-3.1-8b-instant` (fastest, 128K context)
- `mixtral-8x7b-32768` (32K context)

**Configuration Example**:
```bash
PROVIDER=groq
GROQ_API_KEY=gsk_...
MODEL=llama-3.3-70b-versatile
```

**Pricing**: Free tier available, very fast inference

## Benchmark Tool

Test provider performance with the included benchmark harness:

```bash
# Benchmark OpenAI
python benchmark.py --provider openai --model gpt-4o-mini --prompt "Explain quantum computing"

# Benchmark Gemini
python benchmark.py --provider gemini --prompt "Write a Python function to sort a list"

# Benchmark Groq
python benchmark.py --provider groq --model llama-3.3-70b-versatile --prompt "Tell me a joke"
```

The benchmark tool measures:
- End-to-end latency (ms)
- Token counts (input/output)
- Throughput (tokens/sec)
- Response preview

## Project Structure

```
telegram-gpt/
├── bot.py                  # Main entry point
├── config.py               # Configuration management
├── database.py             # SQLite message storage
├── handlers.py             # Telegram message handlers
├── llm_provider.py         # Abstract provider interface
├── llm_factory.py          # Provider factory
├── benchmark.py            # CLI benchmark harness
├── providers/              # LLM provider implementations
│   ├── openai_provider.py
│   ├── gemini_provider.py
│   └── groq_provider.py
├── utils/                  # Utility functions
│   └── context_trimmer.py  # Token budget management
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker orchestration
└── data/                   # Database storage (created at runtime)
```

## How It Works

1. **Message Reception**: Bot receives all messages but only processes those containing "chatgpt"
2. **Authorization**: Checks if sender's user ID matches `AUTHORIZED_USER_ID` or has been granted access
3. **Context Retrieval**: Fetches conversation history from SQLite database
4. **Token Management**: Provider-specific token counting and intelligent trimming to fit model's context window
5. **API Call**: Sends trimmed conversation to selected LLM provider
6. **Storage**: Saves both user message and assistant response to database with token counts
7. **Response**: Sends assistant's response back to Telegram

## Error Handling

The bot gracefully handles:
- Invalid API keys
- Network timeouts
- Rate limits
- Token limit exceeded
- Database corruption (with automatic backup)
- Concurrent access
- Invalid messages
- Unauthorized access

All errors are logged and user-friendly messages are returned.

## Troubleshooting

### Bot doesn't respond

- Check that your message contains "chatgpt" keyword
- Verify your Telegram user ID matches `AUTHORIZED_USER_ID`
- Check bot logs for errors: `docker-compose logs -f`

### "OpenAI API key is invalid"

- Verify `OPENAI_API_KEY` in `.env` is correct
- Ensure no extra spaces or quotes around the key

### "Rate limit exceeded"

- Wait a few moments before trying again
- Consider upgrading your OpenAI plan

### Database errors

- Check that `data/` directory exists and is writable
- Try clearing the database: `rm data/messages.db`

### Docker issues

- Ensure `.env` file exists: `ls -la .env`
- Check container logs: `docker-compose logs -f`
- Rebuild image: `docker-compose up --build -d`

## Development

### Running Tests

Manual testing checklist is provided in the plan. For automated testing, consider adding pytest.

### Logging

Set `LOG_LEVEL=DEBUG` in `.env` for detailed logs.

View logs:
- Local: Check console output
- Docker: `docker-compose logs -f`

### Database Inspection

```bash
sqlite3 data/messages.db

# View all messages
SELECT * FROM messages;

# Count messages per chat
SELECT chat_id, COUNT(*) FROM messages GROUP BY chat_id;

# Total tokens used
SELECT SUM(token_count) FROM messages;
```

### Database Cleanup

Remove conversation history from unauthorized users using the cleanup utility:

```bash
# Preview what will be deleted (safe, read-only)
python cleanup_db.py

# The script will:
# 1. Show authorized users
# 2. Preview unauthorized user history
# 3. Ask for confirmation
# 4. Create automatic backup
# 5. Perform cleanup
# 6. Show results
```

**What gets deleted:**
- All messages in private chats with unauthorized users (both user messages and bot responses)
- Individual messages from unauthorized users in group chats
- Keeps all history for authorized users (main user + granted users)

**Safety features:**
- Preview mode shows what will be deleted before confirmation
- Automatic database backup before deletion
- Requires explicit "yes" confirmation
- Detailed logging of all operations

**When to use:**
- After revoking access from users with `/revoke`
- Regular maintenance to clean up old unauthorized data
- Before deploying to production with fresh user list

**Backup restoration:**
If you need to restore from backup:
```bash
# Backups are created as data/messages.db.bak
cp data/messages.db.bak data/messages.db
```

## Security Considerations

- Keep `.env` file secure (never commit to git)
- Use environment-specific API keys
- Consider network restrictions for production
- Regular database backups recommended
- Monitor OpenAI API usage to avoid unexpected costs

## Performance

- SQLite WAL mode enables concurrent reads
- Token counting is done locally (no API calls)
- Database queries use indexes for speed
- Context trimming prevents excessive API costs

## License

This is a personal project. Use at your own discretion.

## Support

For issues and questions:
1. Check the Troubleshooting section above
2. Review logs for specific error messages
3. Verify all environment variables are set correctly

## Acknowledgments

- Built with [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- Powered by [OpenAI API](https://openai.com/)
- Token counting via [tiktoken](https://github.com/openai/tiktoken)
# telegram-gpt
