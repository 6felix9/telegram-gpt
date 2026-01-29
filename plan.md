# Implementation Plan: Add Replied-to Message Context

## Goal
Include the context of the replied-to message in the prompt sent to the LLM when a user replies to a text message in Telegram. This ensures the LLM understands what the user is referring to.

## Proposed Changes

### 1. Modify `handlers.py`

#### `extract_reply_context(message)`
- **Functionality**: Extracts text content and sender name from a replied-to message.
- **Location**: Add this helper function at the module level or inside `handlers.py`.
- **Implementation Logic**:
    ```python
    def extract_reply_context(message) -> str:
        """
        Extracts context from the message being replied to.
        Returns a formatted string or empty string if no reply/content.
        """
        # Check if this is a reply
        if not message.reply_to_message:
            return ""
            
        reply = message.reply_to_message
        
        # Extract content (prioritize text, then caption for media)
        content = reply.text or reply.caption or ""
        
        # If no text content, ignore
        if not content:
            return ""
            
        # Get sender name
        sender = reply.from_user.first_name if reply.from_user else "Unknown"
        
        # Format the context
        return f"\n\n[Context - Replying to {sender}]: \"{content}\""
    ```

#### `message_handler`
- **Functionality**: Update the main message handler to inject the extracted context into the prompt.
- **Location**: inside `async def message_handler(...)`
- **Changes**:
    1. Call `extract_reply_context(message)` after extracting the initial prompt.
    2. If context is returned, prepend it to the existing `prompt`.
    
    ```python
    # ... inside message_handler ...
    
    # 2. Check for activation keyword
    has_keyword, prompt = extract_keyword(message.text)
    
    # ... existing group logic ...
    
    # NEW: Extract reply context if it exists
    reply_context = extract_reply_context(message)
    if reply_context:
        # Prepend context to the prompt
        # If prompt is empty (just keyword), this effectively becomes the prompt
        prompt = f"{reply_context}\n\n{prompt}".strip()
        
    # ... existing authorization and processing checks ...
    ```

## Verification Plan
1. **Manual Verification**:
    - **Scenario 1**: Reply to a message saying "Hello" with "@chatgpt translate this".
        - **Expected**: Bot prompts LLM with `[Context - Replying to User]: "Hello"\n\ntranslate this`.
    - **Scenario 2**: Reply to a message without `@chatgpt` (in private chat).
        - **Expected**: Bot prompts LLM with context included if keywords aren't required, or standard behavior.
    - **Scenario 3**: Normal message without reply.
        - **Expected**: No context added.
