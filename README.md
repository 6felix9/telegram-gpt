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

2. **Build and push arm64 image (local build machine)**
   ```bash
   docker login
   docker buildx build --platform linux/arm64 -t felixlmao/telegram-gpt:latest --push .
   ```

3. **Run from the published image** (stop/remove old container first if it exists)
   ```bash
   # Stop and remove existing container (safe to run if none exists)
   docker stop telegram-gpt-bot || true
   docker rm telegram-gpt-bot || true

   docker pull felixlmao/telegram-gpt:latest
   docker run -d \
     --name telegram-gpt-bot \
     --restart unless-stopped \
     --env-file .env \
     -v "$(pwd)/data:/app/data" \
     felixlmao/telegram-gpt:latest
   ```

4. **View logs**
   ```bash
   docker logs -f telegram-gpt-bot
   ```

5. **Redeploy with a newer image**
   ```bash
   docker stop telegram-gpt-bot || true
   docker rm telegram-gpt-bot || true
   docker pull felixlmao/telegram-gpt:latest
   docker run -d \
     --name telegram-gpt-bot \
     --restart unless-stopped \
     --env-file .env \
     -v "$(pwd)/data:/app/data" \
     felixlmao/telegram-gpt:latest
   ```

### GitHub Actions Auto-Deploy to EC2 (CI/CD)

This repo includes a GitHub Actions workflow at `.github/workflows/deploy-ec2.yml` that:
1) builds and pushes a Docker image to Docker Hub, then
2) SSHes into your EC2 instance and restarts the container with the new image.

#### 1) Prepare your EC2 instance (one-time)

On the EC2 box, install Docker and make sure your SSH user can run `docker` without a password prompt.

Create an app directory and put your `.env` there:
```bash
sudo mkdir -p /opt/telegram-gpt/data
sudo nano /opt/telegram-gpt/.env
```

Make the directory writable by the container user (UID 1000 in the Dockerfile):
```bash
sudo chown -R 1000:1000 /opt/telegram-gpt
```

#### 2) Create an SSH key for GitHub Actions (one-time)

Generate a dedicated deploy key on your local machine:
```bash
ssh-keygen -t ed25519 -C "github-actions-ec2" -f ./ec2_deploy_key
```

Add `ec2_deploy_key.pub` to your EC2 user’s `~/.ssh/authorized_keys`.

#### 3) Add GitHub repository secrets (one-time)

In GitHub → your repo → Settings → Secrets and variables → Actions, add:

- `DOCKERHUB_USERNAME`: your Docker Hub username
- `DOCKERHUB_TOKEN`: a Docker Hub access token (recommended) or password
- `DOCKERHUB_REPO`: the Docker Hub repo name (e.g. `telegram-gpt`)
- `EC2_HOST`: EC2 public DNS or IP
- `EC2_USER`: SSH username on the EC2 box (e.g. `ubuntu`)
- `EC2_SSH_KEY`: contents of `ec2_deploy_key` (the private key)
- `EC2_APP_DIR` (optional): defaults to `/opt/telegram-gpt`

#### 4) Deploy

Push to the `main` branch (or run the workflow manually via the Actions tab). The workflow will:
- push the image tags `sha-<commit>` and `latest`
- redeploy the container named `telegram-gpt-bot` on your EC2 instance

Check status/logs:
```bash
docker ps
docker logs -f telegram-gpt-bot
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
- `/grant <user_id>` - Admin only, grant bot access to a user
- `/revoke <user_id>` - Admin only, revoke bot access from a user
- `/allowlist` - Admin only, show all authorized users
- `/version` - Show current bot version
- `/personality <name>` - Admin only, view or change active personality
- `/list_personality` - Admin only, list all available personalities

### Authorization

Two-tier authorization system:
- The primary user specified in `AUTHORIZED_USER_ID` has full admin access
- Additional users can be granted access via `/grant` command
- Other users will receive "Sorry, you have no access to me."

### Private Chats vs Groups

The bot works identically in both private chats and group chats:
- In groups, only messages containing "chatgpt" or @mentioning the bot trigger a response
- All group messages are stored for context (even without keyword)
- Authorization is per-user, not per-chat
- Each chat maintains its own conversation history

## CLI Chat Simulator

The project includes an interactive CLI tool for testing conversations without using Telegram.

### Usage

**Test mode (default, writes to database):**
```bash
python3 scripts/chat_cli.py --chat-id test
```

**Simulate real group chat (read-only, doesn't write to database):**
```bash
python3 scripts/chat_cli.py --chat-id 123456789 --group
```

**Test mode with group formatting:**
```bash
python3 scripts/chat_cli.py --chat-id test --group
```

### CLI Commands

- `/clear` - Clear conversation history (only works when `chat_id="test"`)
- `/stats` - Show chat statistics (message count, tokens used, etc.)
- `/exit` or `/quit` - Exit the CLI

### Modes

**TEST MODE** (`chat_id="test"`):
- All prompts and responses are saved to the Neon database
- You can use `/clear` to clear the conversation history
- Useful for testing and development

**READ-ONLY MODE** (any other `chat_id`):
- Fetches existing conversation history from the database
- Your prompts/responses are NOT saved (simulation only)
- `/clear` command is disabled
- Useful for testing how the bot would respond in a real chat without modifying the actual conversation history

## Configuration

All configuration is done via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Required | Your bot token from BotFather |
| `BOT_USERNAME` | Required | Your bot's Telegram username |
| `OPENAI_API_KEY` | Required | Your OpenAI API key (or xAI key for Grok) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model to use |
| `OPENAI_BASE_URL` | _(empty)_ | Optional base URL for xAI or other OpenAI-compatible APIs |
| `OPENAI_TIMEOUT` | `60` | API request timeout in seconds |
| `MAX_CONTEXT_TOKENS` | `16000` | Maximum tokens in conversation context |
| `RESERVE_TOKENS_TEXT` | `1000` | Tokens reserved for text response generation |
| `RESERVE_TOKENS_IMAGE` | `3000` | Tokens reserved for vision response generation |
| `MAX_GROUP_CONTEXT_MESSAGES` | `100` | Max messages stored per group chat |
| `AUTHORIZED_USER_ID` | Required | Telegram user ID for admin access |
| `DATABASE_URL` | Required | PostgreSQL connection string (e.g., Neon DB) |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Model Options

The bot uses the OpenAI Responses API and supports any model compatible with it:
- `gpt-4o-mini` (default, recommended for cost/performance)
- `gpt-4o` (most capable)
- `gpt-5-mini` (reasoning model, uses verbosity/effort instead of temperature)
- xAI Grok models via `OPENAI_BASE_URL=https://api.x.ai/v1`

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
telegram-gpt/
├── bot.py              # Main entry point
├── config.py           # Configuration management
├── database.py         # PostgreSQL message storage (Neon)
├── token_manager.py    # Token counting and trimming
├── openai_client.py    # OpenAI Responses API wrapper
├── handlers.py         # Telegram message handlers
├── prompt_builder.py   # System prompt construction and message formatting
├── scripts/
│   └── chat_cli.py     # CLI chat simulator for testing
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker orchestration
└── .dockerignore       # Docker build exclusions
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
- Concurrent database access (via connection pooling)
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
