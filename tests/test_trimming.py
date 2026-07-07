"""Token counting and the pre-model trimming middleware."""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import agent


def test_count_tokens_nonzero():
    assert agent.count_tokens("hello world") > 0
    assert agent.count_tokens("") == 0


def test_keeps_recent_and_drops_old_when_over_budget():
    # Many large messages; only the newest should survive a tiny budget.
    big = "word " * 500
    messages = [HumanMessage(content=big) for _ in range(10)]
    kept = agent.trim_messages(messages, 200, 50, 300)
    assert kept[-1] is messages[-1]
    assert len(kept) < len(messages)


def test_never_starts_with_orphan_tool_message():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content="result", tool_call_id="1"),
        HumanMessage(content="hi"),
    ]
    # Force a budget that would cut the AIMessage but keep the ToolMessage.
    max_context = (
        agent.count_message_tokens(messages[1])
        + agent.count_message_tokens(messages[2])
        + 5
    )
    kept = agent.trim_messages(messages, max_context, 0, 300)
    assert not (kept and isinstance(kept[0], ToolMessage))
