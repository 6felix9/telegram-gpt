# Telegram GPT Bot

An AI-powered Telegram bot using OpenAI's GPT models with persistent conversation history and intelligent context management.

## Features

- **Keyword Activation**: Bot responds only when "chatgpt" is mentioned
- **Persistent Conversation History**: PostgreSQL-backed storage with connection pooling for reliable concurrent access
- **Token-Aware Context Management**: Intelligent trimming to stay within model limits
- **Docker-Ready**: Production deployment with Docker and docker-compose
- **Private Authorization**: Single-user access control
- **Error Resilient**: Comprehensive error handling for all edge cases

## Architecture

- **Python 3.12+** with modern async/await patterns
- **PostgreSQL** (Neon) with connection pooling for concurrent access
- **tiktoken** for accurate token counting
- **python-telegram-bot 21.7** for Telegram API
- **OpenAI API** for GPT completions

## Quick Start

### Prerequisites

- Python 3.12 or higher
- Telegram account
- OpenAI API key

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
   - `OPENAI_API_KEY`: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
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

### OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Create a new API key
4. Copy the key immediately (you won't see it again)

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
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `AUTHORIZED_USER_ID` | Required | Telegram user ID allowed to use bot |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `MAX_CONTEXT_TOKENS` | `16000` | Maximum tokens in conversation context |
| `OPENAI_TIMEOUT` | `60` | API request timeout in seconds |
| `DATABASE_URL` | Required | PostgreSQL connection string (e.g., Neon DB) |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Model Options

Supported models:
- `gpt-4o-mini` (default, recommended for cost/performance)
- `gpt-4o` (most capable)
- `gpt-3.5-turbo` (faster, cheaper)
- `gpt-4-turbo` (previous generation flagship)

## Database Setup

### PostgreSQL / Neon DB

This bot uses PostgreSQL (or Neon DB for cloud hosting) for conversation storage.

#### Getting Started with Neon

1. **Sign up at [neon.tech](https://neon.tech)** (free tier available)
2. **Create a new project** and note your connection string
3. **Format your DATABASE_URL**:
   ```
   postgresql://username:password@host:port/database?sslmode=require&channel_binding=require
   ```
4. **Update `.env`** with the connection string
5. **Start the bot** - tables are created automatically on first run

## Project Structure

```
telegram-gpt-bot/s
├── bot.py              # Main entry point
├── config.py           # Configuration management
├── database.py         # PostgreSQL message storage
├── token_manager.py    # Token counting and trimming
├── openai_client.py    # OpenAI API wrapper
├── handlers.py         # Telegram message handlers
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .gitignore         # Git ignore rules
├── Dockerfile         # Docker image definition
├── docker-compose.yml # Docker orchestration
└── .dockerignore      # Docker build exclusions
```

## How It Works

1. **Message Reception**: Bot receives all messages but only processes those containing "chatgpt"
2. **Authorization**: Checks if sender's user ID matches `AUTHORIZED_USER_ID`
3. **Context Retrieval**: Fetches conversation history from PostgreSQL database
4. **Token Management**: Uses tiktoken to count tokens and trim history to fit model's context window
5. **API Call**: Sends trimmed conversation to OpenAI API
6. **Storage**: Saves both user message and assistant response to database
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

- Verify `DATABASE_URL` is set correctly in `.env`
- Check that Neon DB instance is accessible
- Ensure Neon connection limits haven't been exceeded
- Check logs for specific PostgreSQL error messages

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
# Connect to PostgreSQL database using psql or your preferred PostgreSQL client
# Use the DATABASE_URL from your .env file

# View all messages
SELECT * FROM messages;

# Count messages per chat
SELECT chat_id, COUNT(*) FROM messages GROUP BY chat_id;

# Total tokens used
SELECT SUM(token_count) FROM messages;
```

## Security Considerations

- Keep `.env` file secure (never commit to git)
- Use environment-specific API keys
- Consider network restrictions for production
- Regular database backups recommended
- Monitor OpenAI API usage to avoid unexpected costs

## Performance

- PostgreSQL connection pooling enables concurrent access
- Token counting is done locally (no API calls)
- Database queries use indexes for speed
- Context trimming prevents excessive API costs
- Keepalive settings prevent idle connection timeouts

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
