# Code Review Issues

**Last Updated**: 2026-02-28
**Status**: Active issues tracked


## MEDIUM Priority Issues 🟡

### 1. Confusing OpenAI API Branching Logic

**Priority**: MEDIUM
**Location**: `openai_client.py:107-123`
**Type**: Code Clarity / Maintainability

**Problem**:
The code uses `startswith("gpt-5")` to branch between model-specific parameters:
```python
if self.model.startswith("gpt-5"):
    response = await asyncio.to_thread(
        self.client.responses.create,
        model=self.model,
        instructions=system_prompt,
        input=formatted_messages,
        text={ "verbosity": "low" },
        reasoning={ "effort": "low" },
    )
else:
    response = await asyncio.to_thread(
        self.client.responses.create,
        model=self.model,
        instructions=system_prompt,
        input=formatted_messages,
        temperature=0.7,
    )
```

**Issues**:
- Fragile string prefix checking (`startswith("gpt-5")`)
- Adding new model families requires updating if-else logic
- Different parameter sets (temperature vs verbosity/reasoning) split across branches

**Solution**:
Create a model parameter registry to cleanly map models to their API parameters, eliminating the branching logic.

---

### 12. API Format Conversion - Responses API Image Compatibility

**Priority**: MEDIUM
**Location**: `prompt_builder.py:139-158`
**Type**: API Integration

**Problem**:
- Image format conversion from Chat Completions format (`"type": "image_url"`) to Responses API format (`"type": "input_image"`) exists in `prompt_builder.py`
- The conversion logic looks correct but hasn't been thoroughly tested against the latest API spec
- Could cause silent failures with vision requests if the API changes

**Solution**:
- Test image processing with the current OpenAI Responses API
- Add a simple smoke test for image message formatting

---

## LOW Priority Issues 🔵

### 14. No Rate Limiting

**Priority**: LOW
**Location**: All handlers
**Type**: Cost Protection

**Problem**:
- No per-user rate limits implemented
- A granted user could theoretically spam the bot

**Practical Assessment**:
Low risk in practice - the bot requires authorization (admin + explicitly granted users only). The attack surface is limited to trusted users. Telegram itself also has rate limits on bot messages.

**Solution** (if needed):
Simple in-memory sliding window counter in handlers.

---

### 15. No Input Size Validation

**Priority**: LOW
**Location**: `handlers.py`
**Type**: Defensive Programming

**Problem**:
- No explicit message length validation before token counting
- No image size check before downloading

**Practical Assessment**:
Telegram enforces its own limits: text messages max ~4096 chars, images max ~20MB. The token manager and OpenAI API will reject oversized inputs. This is defense-in-depth, not a critical gap.

**Solution** (if needed):
Add simple length check for messages and file_size check for images.

---

### 23. Inconsistent Error Handling

**Priority**: LOW
**Location**: `handlers.py` (multiple locations)
**Type**: Code Quality

**Problem**:
Some exceptions are logged silently (e.g., failed group message storage), while others show user-facing error messages.

**Practical Assessment**:
The inconsistency is partially intentional - silently failing to store a background group message is reasonable (the user didn't ask for anything). User-facing operations correctly show errors. Could be more consistent but not causing real issues.

---

## Strategic Enhancements 🧠

### 16. Context Trimming Strategy

**Priority**: Strategic
**Location**: `token_manager.py`, `database.py`
**Type**: Intelligence Enhancement

Messages are trimmed purely by age (oldest first). No importance scoring, semantic relevance, or summarization. The bot loses information as conversations grow, but this is standard behavior for most chatbots.

**Potential improvements**:
1. Conversation summarization (compress old context instead of dropping)
2. Semantic search for relevant past messages (would require pgvector)

---

### 17. Long-Term Memory

**Priority**: Strategic
**Location**: Entire codebase
**Type**: Missing Feature

No persistence of user preferences, facts, or conversation history beyond the context window. This is the highest-impact enhancement for user experience.

**Potential approach**:
1. Extract facts from messages using GPT
2. Store in structured format per user
3. Inject relevant facts into system prompt

---

### 27. Semantic Search

**Priority**: Strategic
**Type**: Missing Feature

Implement pgvector for semantic message retrieval - find relevant messages by meaning, not just recency. Would require adding an `embedding` column to the messages table.

---

### 28. Conversation Summarization

**Priority**: Strategic
**Type**: Missing Feature

When context exceeds limit, summarize old messages instead of dropping them. Would preserve important information beyond token limits.

---

### 29. User Profile Building

**Priority**: Strategic
**Type**: Missing Feature

Extract and persist user facts automatically (preferences, characteristics, goals). Would enable personalized, context-aware responses.

---

## Recommended Action Plan

### Phase 1: Quick Wins
1. Fix model branching logic (Issue #1)
2. ✅ ~~Centralize prompt construction (Issue #2)~~ - **COMPLETED 2026-02-13**
3. ✅ ~~Query result caching (Issue #19)~~ - **COMPLETED (separate branch)**
4. Verify Responses API image compatibility (Issue #12)

### Phase 2: Intelligence Upgrade
1. Implement long-term memory (Issue #17)
2. Add conversation summarization (Issue #28)
3. Build user profile storage (Issue #29)
4. Explore semantic search with pgvector (Issue #27)

---

**Last Updated**: 2026-02-28
**Status**: Active issues - focused on practical improvements
