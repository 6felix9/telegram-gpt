# Code Review Issues

**Last Updated**: 2026-04-05  
**Status**: Active issues tracked

## Medium Priority

### 1. `active_personality` Bootstrap Drift

**Location**: `database.py`, live Neon schema  
**Type**: Schema / behavior consistency

**Problem**:

- The live database currently has `active_personality.personality DEFAULT 'normal'`
- Fresh bootstrap SQL in `database.py` creates the table with `DEFAULT 'default'`
- Different environments can therefore start with different sentinel values

**Impact**:

- Existing deployments and fresh databases do not start from exactly the same state
- Documentation and mental models drift because both `normal` and `default` are in use

**Suggested fix**:

- Standardize on one value and migrate the other, or
- Remove reliance on a sentinel name entirely and fall back whenever the active personality has no matching row

### 2. Responses/Image Formatting Still Needs Explicit Smoke Coverage

**Location**: `prompt_builder.py`, `handlers.py`  
**Type**: API integration confidence

**Problem**:

- Vision requests depend on converting internal message parts into provider-specific payloads
- The current formatting logic looks coherent, but there is still no automated smoke coverage for image requests

**Impact**:

- A future SDK or API-shape change could break image handling quietly

**Suggested fix**:

- Add a small integration smoke test or local validation script for image request formatting

## Low Priority

### 3. No Per-User Rate Limiting

**Location**: `handlers.py`  
**Type**: Cost protection

**Problem**:

- Authorized users can send requests without any application-level throttle

**Assessment**:

- Low risk because access is restricted to the admin plus explicitly granted users

**Suggested fix**:

- Add a simple in-memory sliding window or token bucket per user

### 4. No Explicit Input Size Guardrails

**Location**: `handlers.py`  
**Type**: Defensive programming

**Problem**:

- The bot relies on Telegram, token trimming, and provider APIs to reject oversized payloads
- There is no explicit text length or image file-size guardrail before processing

**Assessment**:

- Mostly defense-in-depth, not an immediate production bug

**Suggested fix**:

- Add cheap preflight checks for message length and photo size

### 5. Inconsistent Error Exposure

**Location**: `handlers.py`  
**Type**: Code quality

**Problem**:

- Some failures are intentionally silent or log-only
- Others surface user-facing errors
- The split is reasonable in places, but the policy is implicit rather than documented

**Suggested fix**:

- Document which failures should stay silent vs user-visible, then align handlers to that rule

## Completed Since The Previous Review

### Model Registry Cleanup

The older `startswith("gpt-5")` branching issue in `openai_client.py` is no longer active. The code now uses `MODEL_REGISTRY` and `ModelConfig`, which is the correct direction.

## Strategic Enhancements

### 6. Conversation Summarization

Messages are still trimmed mainly by recency. Summarizing older context before dropping it would preserve more useful history.

### 7. Long-Term Memory

The bot does not yet extract and persist user preferences or facts outside the immediate context window.

### 8. Semantic Retrieval

There is no semantic search over historical messages. If this becomes important, `pgvector` would be the natural next step.
