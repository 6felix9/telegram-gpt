"""Direct coverage for token_budget.py, independent of agent.py's re-export."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import token_budget


def test_count_tokens_nonzero_for_text():
    assert token_budget.count_tokens("hello world") > 0
    assert token_budget.count_tokens("") == 0


def test_trim_messages_keeps_last_message_even_over_budget():
    messages = [HumanMessage(content="a" * 50)]
    kept = token_budget.trim_messages(messages, max_context_tokens=1, reserve=0)
    assert kept == messages


def test_trim_messages_drops_orphaned_leading_tool_message():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "f", "args": {}, "id": "1"}]),
        ToolMessage(content="result", tool_call_id="1"),
        HumanMessage(content="final"),
    ]
    kept = token_budget.trim_messages(messages, max_context_tokens=100000, reserve=0)
    assert not isinstance(kept[0], ToolMessage)


def test_trim_messages_always_keeps_summary_under_pressure():
    summary = HumanMessage(
        content="earlier conversation summary",
        additional_kwargs={"lc_source": "summarization"},
    )
    big = "word " * 200
    messages = [
        summary,
        HumanMessage(content=big),
        HumanMessage(content=big),
        HumanMessage(content="latest"),
    ]
    # Budget fits summary + last only; the two large middle messages must be dropped.
    max_context = (
        token_budget.count_message_tokens(summary)
        + token_budget.count_message_tokens(messages[-1])
        + 5
    )
    kept = token_budget.trim_messages(messages, max_context_tokens=max_context, reserve=0)
    assert kept[0] is summary
    assert kept[-1] is messages[-1]
    assert summary in kept
    assert len(kept) == 2


def test_trim_messages_keeps_summary_and_last_even_over_budget():
    summary = HumanMessage(
        content="a" * 80,
        additional_kwargs={"lc_source": "summarization"},
    )
    latest = HumanMessage(content="b" * 80)
    messages = [summary, latest]
    kept = token_budget.trim_messages(messages, max_context_tokens=1, reserve=0)
    assert kept == [summary, latest]
