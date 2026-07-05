# Dev/Prod Environment Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give this Telegram bot a real dev/prod split — a second fully-deployed Railway environment with its own Telegram bot and Neon database branch, plus versioned Alembic migrations replacing the current `CREATE TABLE IF NOT EXISTS` bootstrap in `database.py`.

**Architecture:** Two Railway environments (`production` on `main`, `dev` on a `dev` branch), each with its own Telegram bot token and its own Neon Postgres branch. Schema changes move from ad-hoc idempotent DDL run on every boot to versioned Alembic migration files run as an explicit `preDeployCommand` step before each deploy. The existing `production` database and the new `dev` Neon branch (a copy-on-write clone of production) both already have the current schema physically present, so the baseline migration is applied via `alembic stamp head` (record the revision without re-running DDL) rather than `alembic upgrade head`.

**Tech Stack:** Alembic (migration engine), SQLAlchemy (Alembic's required engine layer — no ORM models, migrations are raw SQL via `op.execute`), existing psycopg2/Railway/Neon stack unchanged.

## Global Constraints

- Target Python 3.12+, 4-space indentation, existing code style (per `CLAUDE.md`).
- No ORM models — migrations are raw SQL strings via `op.execute()`, consistent with the existing raw-psycopg2 style in `database.py`. Do not introduce SQLAlchemy Core/ORM query building elsewhere.
- Pin new dependencies: `alembic==1.14.1`, `SQLAlchemy==2.0.37` (or latest available 1.14.x / 2.0.x patch at install time).
- Never commit real bot tokens, Neon connection strings, or other secrets — placeholders only in `.env.example` and this plan.
- Per `CLAUDE.md`, pytest tests are pure logic only, no database required. Migration correctness is verified manually against real Neon branches (dev first, then prod), not via new pytest tests.
- Commit style: short imperative subject lines, one concern per commit (per `CLAUDE.md` commit guidelines).

---

## File Structure

**Create:**
- `alembic.ini` — Alembic config, `script_location = alembic`, no hardcoded DB URL.
- `alembic/env.py` — resolves `DATABASE_URL` from `config.py` (which already loads `.env`) instead of `alembic.ini`.
- `alembic/script.py.mako` — Alembic's standard revision template (needed so future `alembic revision` calls generate consistent files).
- `alembic/versions/0001_initial_schema.py` — baseline migration recreating the current live schema exactly.

**Modify:**
- `requirements.txt` — add `alembic`, `SQLAlchemy`.
- `database.py` — remove `_init_db()` (lines 38, 49–124); schema is now migration-managed, not created on boot.
- `start.sh` — run `alembic upgrade head` before launching `bot.py`.
- `.github/workflows/ci.yml` — include `alembic/` in the `py_compile` sanity check.
- `README.md` — Local Development steps, Database section, Validation section.
- `database.md` — Operational Notes: schema is Alembic-managed.
- `CLAUDE.md` — Build/Test commands, Database Schema section.
- `.env.example` — note that `DATABASE_URL` is also read by Alembic.

**No code changes (infrastructure-only, done via CLI/dashboard):**
- New Telegram bot for `dev` (via BotFather).
- New Neon branch `dev`.
- New Railway environment `dev`, new `dev` git branch, Railway variable + `preDeployCommand` wiring.

---

### Task 1: Add Alembic scaffold

**Files:**
- Modify: `requirements.txt`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/` (empty dir until Task 2)

**Interfaces:**
- Produces: `alembic upgrade head` / `alembic stamp head` / `alembic current` as the commands every later task (local dev, CI, Railway `preDeployCommand`) relies on.
- Consumes: `config.DATABASE_URL` from `config.py` (already loads `.env` via `load_dotenv()` at import time — no new env-loading code needed).

- [ ] **Step 1: Add dependencies**

Append to `requirements.txt`:

```
alembic==1.14.1
SQLAlchemy==2.0.37
```

- [ ] **Step 2: Install and verify**

Run: `pip install -r requirements.txt`
Expected: `alembic` and `SQLAlchemy` install cleanly, no dependency conflicts with `psycopg2-binary==2.9.9`.

- [ ] **Step 3: Create `alembic.ini`**

```ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console
qualname =

[logger_sqlalchemy]
level = WARNING
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

`prepend_sys_path = .` is what lets `alembic/env.py` import `config.py` from the repo root without manual `sys.path` hacks.

- [ ] **Step 4: Create `alembic/env.py`**

```python
"""Alembic environment: resolves DATABASE_URL from config.py instead of alembic.ini."""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from config import config as app_config

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def get_url() -> str:
    return app_config.DATABASE_URL


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

`target_metadata = None` is intentional — there are no ORM models. Every migration writes raw SQL directly, so Alembic's autogenerate diffing is not used.

- [ ] **Step 5: Create `alembic/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 6: Verify Alembic can see the config**

Run: `alembic current`
Expected: connects using your local `.env`'s `DATABASE_URL` and prints (no current revision yet — empty output is correct at this point, since no `alembic_version` table exists until Task 2/6).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt alembic.ini alembic/env.py alembic/script.py.mako
git commit -m "Add Alembic migration scaffold"
```

---

### Task 2: Baseline migration matching the current live schema

**Files:**
- Create: `alembic/versions/0001_initial_schema.py`

**Interfaces:**
- Consumes: nothing (first revision, `down_revision = None`).
- Produces: revision id `"0001"` — Task 6 stamps both the Neon `dev` branch and `production` at this exact revision.

This migration must byte-for-byte match what `database.py:_init_db()` currently creates (`messages`, `granted_users`, `personality`, `active_personality`, `active_model`, plus the `idx_chat_timestamp` index and the `active_personality` seed row), so that a genuinely fresh database (not cloned from production) ends up identical to today's schema.

- [ ] **Step 1: Write the migration**

```python
"""Initial schema: messages, granted_users, personality, active_personality, active_model

Revision ID: 0001
Revises:
Create Date: 2026-07-05

"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE messages (
            id SERIAL PRIMARY KEY,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id BIGINT,
            message_id BIGINT,
            token_count INTEGER DEFAULT 0,
            sender_name TEXT,
            sender_username TEXT,
            is_group_chat BOOLEAN DEFAULT FALSE
        )
    """)
    op.execute("""
        CREATE INDEX idx_chat_timestamp
        ON messages(chat_id, timestamp DESC)
    """)
    op.execute("""
        CREATE TABLE granted_users (
            user_id TEXT PRIMARY KEY,
            granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            first_name TEXT,
            username TEXT
        )
    """)
    op.execute("""
        CREATE TABLE personality (
            personality TEXT PRIMARY KEY,
            prompt TEXT NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE active_personality (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            personality TEXT NOT NULL DEFAULT 'default',
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    op.execute("""
        INSERT INTO active_personality (id, personality, updated_at)
        VALUES (1, 'default', CURRENT_TIMESTAMP)
    """)
    op.execute("""
        CREATE TABLE active_model (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            model TEXT NOT NULL DEFAULT 'gpt-4o-mini',
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS active_model")
    op.execute("DROP TABLE IF EXISTS active_personality")
    op.execute("DROP TABLE IF EXISTS personality")
    op.execute("DROP TABLE IF EXISTS granted_users")
    op.execute("DROP INDEX IF EXISTS idx_chat_timestamp")
    op.execute("DROP TABLE IF EXISTS messages")
```

- [ ] **Step 2: Verify against a scratch database**

Do NOT run this against production or the Neon dev branch yet (both already have this schema — see Task 6). Verify the migration is well-formed against a throwaway local Postgres instead:

```bash
docker run --rm -d --name alembic-scratch -e POSTGRES_PASSWORD=test -p 5433:5432 postgres:16
DATABASE_URL="postgresql://postgres:test@localhost:5433/postgres" alembic upgrade head
DATABASE_URL="postgresql://postgres:test@localhost:5433/postgres" alembic current
```

Expected: `alembic current` prints `0001 (head)`, and connecting with `psql` shows all five tables.

```bash
docker stop alembic-scratch
```

- [ ] **Step 3: Commit**

```bash
git add alembic/versions/0001_initial_schema.py
git commit -m "Add baseline migration for current schema"
```

---

### Task 3: Remove auto-schema-creation from `database.py`

**Files:**
- Modify: `database.py:38` (remove `self._init_db()` call)
- Modify: `database.py:49-124` (delete `_init_db` method entirely)
- Modify: `start.sh`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Database.__init__` no longer creates tables — callers (`bot.py`, `scripts/chat_cli.py`, tests) now rely on migrations having already been applied to whatever `DATABASE_URL` points at.

- [ ] **Step 1: Remove the `_init_db` call**

In `database.py`, delete line 38:
```python
        self._init_db()
```
so `__init__` ends right after the pool-creation `try/except` block (the `logger.info("Database connection pool initialized with keepalives")` block).

- [ ] **Step 2: Delete the `_init_db` method**

Delete `database.py:49-124` in full (the entire `def _init_db(self): ...` method, from the docstring through the closing `raise` of its `except` block).

- [ ] **Step 3: Verify nothing else calls `_init_db`**

Run: `grep -rn "_init_db" .`
Expected: no matches (confirms no other caller was relying on it).

- [ ] **Step 4: Wire migrations into local/dev startup**

In `start.sh`, replace:
```bash
echo "Starting bot with fresh instance..."
python bot.py
```
with:
```bash
echo "Applying database migrations..."
alembic upgrade head

echo "Starting bot with fresh instance..."
python bot.py
```

- [ ] **Step 5: Verify locally**

With a local `.env` pointing at a database already stamped at `0001` (do this after Task 6), run:
```bash
python3 -m py_compile database.py
python3 scripts/chat_cli.py --chat-id test
```
Expected: `chat_cli.py` starts and processes a message without any `_init_db`-related errors — confirms `Database.__init__` no longer depends on the removed method.

- [ ] **Step 6: Commit**

```bash
git add database.py start.sh
git commit -m "Remove auto schema creation, defer to Alembic migrations"
```

---

### Task 4: CI sanity check for migration files

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `alembic/env.py`, `alembic/versions/0001_initial_schema.py` from Tasks 1–2.
- Produces: a CI step that fails fast if a future migration file has a Python syntax error, before it ever reaches a real deploy.

- [ ] **Step 1: Update the compile step**

Change:
```yaml
      - name: Compile Python files
        run: python3 -m py_compile *.py
```
to:
```yaml
      - name: Compile Python files
        run: python3 -m py_compile *.py alembic/env.py alembic/versions/*.py
```

- [ ] **Step 2: Verify**

Run locally: `python3 -m py_compile *.py alembic/env.py alembic/versions/*.py`
Expected: exits 0, no output.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "Include Alembic migrations in CI compile check"
```

---

### Task 5: Update documentation

**Files:**
- Modify: `README.md:41-79` (Local Development), `README.md:236-249` (Database), `README.md:250-257` (Validation)
- Modify: `database.md` (Operational Notes section)
- Modify: `CLAUDE.md` (Build/Test commands list, Database Schema section)
- Modify: `.env.example`

- [ ] **Step 1: Update `README.md` Local Development steps**

After the existing step 3 (`Fill in .env`) and before step 4 (`Run the bot`), insert a new step:

```markdown
4. Apply database migrations:

```bash
alembic upgrade head
```

5. Run the bot:
```

Renumber the old step 4 (`Run the bot`) to step 5, and update the `start.sh` description line (`README.md:79`) to: `` `start.sh` creates or reuses `venv/`, installs dependencies, applies migrations, and starts the bot. ``

- [ ] **Step 2: Update `README.md` Database section (`README.md:236-249`)**

Replace:
```markdown
Tables are created automatically on startup if they do not exist.
```
with:
```markdown
Schema is managed with Alembic migrations in `alembic/versions/`. Run `alembic upgrade head` to apply pending migrations — this is done automatically by `start.sh` locally and by the Railway `preDeployCommand` in each environment.
```

- [ ] **Step 3: Update `README.md` Validation section (`README.md:250-257`)**

Add `alembic upgrade head` (against a local/dev database) to the checklist, and add:
```markdown
python3 -m py_compile alembic/env.py alembic/versions/*.py
```

- [ ] **Step 4: Update `database.md` Operational Notes**

Replace the line `The bot creates missing tables automatically on startup.` with: `Schema changes are applied via Alembic migrations (\`alembic upgrade head\`), not created automatically on boot.` Also resolve the pre-existing documented mismatch note (the `active_personality` default `'default'` vs. live `'normal'`) by adding: `The default seeded by \`alembic/versions/0001_initial_schema.py\` is \`'default'\`; the live production value was already changed to \`'normal'\` via \`/personality\` before Alembic was adopted, which is expected — the seed only applies to brand-new databases.`

- [ ] **Step 5: Update `CLAUDE.md`**

In the "Build, Test, and Development Commands" list, add after `Install test deps`:
```markdown
- Apply database migrations: `alembic upgrade head`
```
In the "Database Schema" section, add a line: `Schema is version-controlled via Alembic migrations in \`alembic/versions/\`, applied with \`alembic upgrade head\` (not created automatically on boot).`

- [ ] **Step 6: Update `.env.example`**

After the `DATABASE_URL` line, add a comment:
```
# Note: DATABASE_URL is also used by Alembic (alembic/env.py) to apply migrations.
```

- [ ] **Step 7: Commit**

```bash
git add README.md database.md CLAUDE.md .env.example
git commit -m "Document Alembic migration workflow"
```

---

### Task 6: Provision the Neon `dev` branch and stamp both databases

**No code changes — infrastructure provisioning.** Use the `neon-postgres` skill's CLI/API guidance for exact commands; outline below.

**Interfaces:**
- Consumes: revision `"0001"` from Task 2.
- Produces: a `dev` Neon branch connection string, consumed by Task 9 as the `dev` Railway environment's `DATABASE_URL`.

- [ ] **Step 1: Create the Neon branch**

```bash
neon branches create --project-id <your-project-id> --name dev --parent production
```
Expected: prints the new branch's id and a connection string (copy-on-write clone of `production` — includes the current live data and schema).

- [ ] **Step 2: Get the dev branch connection string**

```bash
neon connection-string dev --project-id <your-project-id> --database-name <your-db-name> --pooled
```
Expected: a `postgresql://...` URL. Save it — this becomes the `dev` Railway environment's `DATABASE_URL` in Task 9. Do not commit it anywhere.

- [ ] **Step 3: Stamp the dev branch at the baseline revision**

The dev branch already has all five tables (cloned from production), so running the migration for real would fail with `relation "messages" already exists`. Mark it as already being at `0001` without executing DDL:

```bash
DATABASE_URL="<dev-branch-connection-string>" alembic stamp head
DATABASE_URL="<dev-branch-connection-string>" alembic current
```
Expected: `alembic current` prints `0001 (head)`.

- [ ] **Step 4: Stamp production at the baseline revision**

Same reasoning applies to production — it already has the schema.

```bash
DATABASE_URL="<production-connection-string>" alembic stamp head
DATABASE_URL="<production-connection-string>" alembic current
```
Expected: `alembic current` prints `0001 (head)`.

From this point forward, any *new* migration (`0002_...` etc.) is applied for real with `alembic upgrade head` — tested against the dev branch first, then production — since both are now on record as being at `0001`.

---

### Task 7: Create the second Telegram bot for dev

**No code changes — manual step, must be done by the user in Telegram.**

- [ ] **Step 1: Create the bot**

Message `@BotFather` in Telegram:
```
/newbot
```
Follow the prompts to name it (e.g. `Your Bot Dev`) and choose a unique username ending in `bot` (e.g. `yourbot_dev_bot`). BotFather replies with the new bot's token.

- [ ] **Step 2: Record credentials**

Save the new token and username — they become `TELEGRAM_BOT_TOKEN` / `BOT_USERNAME` for the `dev` Railway environment in Task 9. Do not commit them anywhere.

- [ ] **Step 3: Authorize yourself on the dev bot**

Once the `dev` environment is live (Task 9), message the dev bot once. Since `AUTHORIZED_USER_ID` is the same Telegram user id as production, you'll bootstrap as the admin automatically (per the existing authorization model — no separate grant needed for the main admin).

---

### Task 8: Create the `dev` git branch

- [ ] **Step 1: Create and push**

```bash
git checkout -b dev
git push -u origin dev
```
Expected: `dev` branch now exists on the remote, currently identical to `main`.

- [ ] **Step 2: Confirm**

```bash
git branch -a | grep dev
```
Expected: shows both local `dev` and `remotes/origin/dev`.

---

### Task 9: Create the Railway `dev` environment

Use the `use-railway` skill's `configure.md`/`setup.md` guidance for exact flags; outline below.

**Interfaces:**
- Consumes: dev Neon connection string (Task 6), dev bot token/username (Task 7), `dev` git branch (Task 8).
- Produces: a second live deployment that behaves identically to production but talks to the dev bot and dev database.

- [ ] **Step 1: Duplicate environment config**

```bash
railway environment new dev --duplicate production
```
Expected: new `dev` environment with the same service config and variables as `production`.

- [ ] **Step 2: Override dev-specific variables**

```bash
railway variable set TELEGRAM_BOT_TOKEN="<dev-bot-token>" --service <bot-service> --environment dev
railway variable set BOT_USERNAME="<dev-bot-username>" --service <bot-service> --environment dev
railway variable set DATABASE_URL="<dev-branch-connection-string>" --service <bot-service> --environment dev
```

- [ ] **Step 3: Wire the dev environment to the `dev` branch**

```bash
railway environment edit --project <project-id> --environment dev --service-config <bot-service> source.branch "dev"
```

- [ ] **Step 4: Set `preDeployCommand` to run migrations, on both environments**

```bash
railway environment edit --project <project-id> --environment dev --service-config <bot-service> deploy.preDeployCommand "alembic upgrade head"
railway environment edit --project <project-id> --environment production --service-config <bot-service> deploy.preDeployCommand "alembic upgrade head"
```
This is what makes the migration workflow real going forward: every future deploy runs `alembic upgrade head` before the bot starts, in both environments.

- [ ] **Step 5: Verify config**

```bash
railway environment config --project <project-id> --environment dev --json
railway environment config --project <project-id> --environment production --json
```
Expected: both show `deploy.preDeployCommand: "alembic upgrade head"`; `dev`'s variables show the dev bot token and dev `DATABASE_URL`; `source.branch` for `dev` is `"dev"`.

- [ ] **Step 6: Trigger and verify the first dev deploy**

```bash
git checkout dev
git commit --allow-empty -m "Trigger initial dev environment deploy"
git push origin dev
railway deployment list --project <project-id> --environment dev --service <bot-service> --json
```
Poll until `status` is `SUCCESS` (per the `use-railway` skill's rule: never report a deploy successful without observing a terminal `SUCCESS`).

---

### Task 10: End-to-end verification

- [ ] **Step 1: Confirm dev bot responds**

Message the dev bot in Telegram (e.g. `chatgpt hello`). Expected: it responds, using the dev Neon branch (verify by checking `messages` row count on the dev branch increased, not production's).

- [ ] **Step 2: Confirm isolation**

```bash
DATABASE_URL="<production-connection-string>" python3 -c "
from database import Database
db = Database('<production-connection-string>')
print(db.get_stats('test'))
"
```
Expected: production's `test` chat stats are unchanged by the dev bot conversation above — confirms the two databases are genuinely isolated.

- [ ] **Step 3: Confirm the full promotion path**

Make a trivial change on `dev` (e.g. a comment), push to `dev`, confirm the dev deploy picks it up, then merge `dev` → `main` and confirm the production deploy picks it up separately. This is the loop going forward: build on `dev`, verify against the dev bot/branch, merge to `main` for production.

---

## Self-Review

**Spec coverage:**
- Second fully deployed dev environment → Tasks 7–9. ✓
- Isolated dev database via Neon branching → Task 6. ✓
- Versioned migration tool replacing `CREATE TABLE IF NOT EXISTS` → Tasks 1–3. ✓
- Deploy pipeline actually runs migrations (not just files sitting unused) → Task 9 Step 4 (`preDeployCommand`). ✓
- CI/doc consistency → Tasks 4–5. ✓
- End-to-end proof it works → Task 10. ✓

**Placeholder scan:** No TBD/"add appropriate"/"similar to Task N" patterns — every step has literal file content or literal commands with expected output.

**Type/interface consistency:** Revision id `"0001"` (Task 2) is the exact string used in every `alembic stamp head` / `alembic current` verification in Tasks 6 and 9. `DATABASE_URL` is the single env var name used consistently across `config.py`, `alembic/env.py`, `start.sh`, and Railway variable commands — no renamed variants introduced.
