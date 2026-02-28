# Neon DB – Telegram GPT

**Project:** Telegram GPT (`fancy-meadow-58588477`)  
**Region:** aws-ap-southeast-1  
**PostgreSQL:** 17

**Branches:** `production` (default), `development` (archived)

---

## Tables (public schema)

### 1. `messages` (800 kB total)

| Column          | Type                     | Nullable | Default           |
|-----------------|--------------------------|----------|-------------------|
| id              | integer                  | NOT NULL | nextval(...)      |
| chat_id         | text                     | NOT NULL | —                 |
| role            | text                     | NOT NULL | —                 |
| content         | text                     | NOT NULL | —                 |
| timestamp       | timestamp without time zone | NOT NULL | CURRENT_TIMESTAMP |
| user_id         | bigint                   | yes      | —                 |
| message_id      | bigint                   | yes      | —                 |
| token_count     | integer                  | yes      | 0                 |
| sender_name     | text                     | yes      | —                 |
| sender_username | text                     | yes      | —                 |
| is_group_chat   | boolean                  | yes      | false             |

- **Primary key:** `id`
- **Index:** `idx_chat_timestamp` on `(chat_id, timestamp DESC)`

---

### 2. `granted_users` (32 kB total)

| Column     | Type                     | Nullable | Default           |
|------------|--------------------------|----------|-------------------|
| user_id    | text                     | NOT NULL | —                 |
| granted_at | timestamp without time zone | NOT NULL | CURRENT_TIMESTAMP |

- **Primary key:** `user_id`

---

### 3. `personality` (80 kB total)

| Column      | Type | Nullable | Default |
|-------------|------|----------|---------|
| personality | text | NOT NULL | —       |
| prompt      | text | NOT NULL | —       |

- **Primary key:** `personality`

---

### 4. `active_personality` (32 kB total)

| Column      | Type                     | Nullable | Default           |
|-------------|--------------------------|----------|-------------------|
| id          | integer                  | NOT NULL | 1                 |
| personality | text                     | NOT NULL | 'normal'          |
| updated_at  | timestamp without time zone | NOT NULL | CURRENT_TIMESTAMP |

- **Primary key:** `id`
- **Check:** `id = 1` (single-row config)

---

## Summary

- **messages** – Conversation history with token counts and group chat metadata
- **granted_users** – Users granted bot access
- **personality** / **active_personality** – System prompts / personalities and current default

*Last updated from Neon MCP – Feb 28, 2026*
