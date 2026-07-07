"""Token counting and the pre-model trimming middleware."""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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
    # Make the AIMessage large enough that, once the ToolMessage and
    # HumanMessage are already kept, it no longer fits the remaining budget.
    # That forces the age-based trim to drop the AIMessage, leaving a leading
    # orphaned ToolMessage that the drop-loop must then remove.
    big = "word " * 200
    messages = [
        AIMessage(content=big, tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content="result", tool_call_id="1"),
        HumanMessage(content="hi"),
    ]
    # Budget fits only the ToolMessage + HumanMessage, not the much larger AIMessage.
    max_context = (
        agent.count_message_tokens(messages[1])
        + agent.count_message_tokens(messages[2])
        + 5
    )
    kept = agent.trim_messages(messages, max_context, 0, 300)
    # The AIMessage was trimmed by age, so the orphan-drop loop must fire,
    # leaving exactly the HumanMessage behind.
    assert not (kept and isinstance(kept[0], ToolMessage))
    assert kept == [messages[2]]


def test_lone_most_recent_tool_message_is_never_dropped():
    # The newest (and only surviving) message is itself a ToolMessage.
    # trim_messages must not strip it even though the orphan-drop loop
    # would otherwise treat it as a leading orphaned ToolMessage.
    messages = [ToolMessage(content="result", tool_call_id="1")]
    kept = agent.trim_messages(messages, 100000, 0, 300)
    assert kept != []
    assert kept[-1] is messages[-1]
