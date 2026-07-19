import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from conversation_summary import (
    _PendingSummaryAuditRecord,
    ResilientSummarizationMiddleware,
    SummaryAuditRecord,
    sanitize_summary_messages,
)


class _FakeSummaryChat(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _runtime(thread_id="chat-1"):
    return SimpleNamespace(context=SimpleNamespace(thread_id=thread_id))


def _count_messages(messages):
    return sum(len(str(message.content)) + 4 for message in messages)


def test_sanitize_replaces_data_url_without_mutating_original():
    original = HumanMessage(
        content=[
            {"type": "text", "text": "A receipt"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,SECRET"},
            },
        ]
    )

    sanitized = sanitize_summary_messages([original])

    assert sanitized[0] is not original
    assert sanitized[0].content == [
        {"type": "text", "text": "A receipt"},
        {"type": "text", "text": "[image omitted]"},
    ]
    assert "SECRET" in str(original.content)
    assert "SECRET" not in str(sanitized[0].content)


def test_sanitize_leaves_plain_text_message_unchanged():
    original = HumanMessage(content="plain text")
    assert sanitize_summary_messages([original]) == [original]


def _middleware(summary_text="durable summary"):
    model = _FakeSummaryChat(messages=iter([AIMessage(content=summary_text)]))
    return ResilientSummarizationMiddleware(
        model=model,
        summary_model_name="gpt-4.1-mini",
        trigger=("messages", 4),
        keep=("messages", 1),
        token_counter=_count_messages,
        trim_tokens_to_summarize=10000,
    )


def test_before_model_replaces_old_messages_with_summary_and_recent_suffix():
    middleware = _middleware()
    newest = HumanMessage(id="4", content="newest")
    state = {
        "messages": [
            HumanMessage(id="1", content="old question"),
            AIMessage(id="2", content="old answer"),
            HumanMessage(id="3", content="recent question"),
            newest,
        ]
    }

    update = middleware.before_model(state, _runtime())

    assert update is not None
    assert isinstance(update["messages"][0], RemoveMessage)
    summary = update["messages"][1]
    assert summary.additional_kwargs["lc_source"] == "summarization"
    assert "durable summary" in summary.content
    assert update["messages"][-1] is newest


def test_tool_call_and_result_are_not_split_at_cutoff():
    middleware = ResilientSummarizationMiddleware(
        model=_FakeSummaryChat(messages=iter([AIMessage(content="tool summary")])),
        summary_model_name="gpt-4.1-mini",
        trigger=("messages", 5),
        keep=("messages", 2),
        token_counter=_count_messages,
        trim_tokens_to_summarize=10000,
    )
    messages = [
        HumanMessage(id="1", content="old"),
        AIMessage(
            id="2",
            content="",
            tool_calls=[{"name": "fetch_url", "args": {"url": "https://x.test"}, "id": "c1"}],
        ),
        ToolMessage(id="3", content="result", tool_call_id="c1"),
        HumanMessage(id="4", content="follow-up"),
        HumanMessage(id="5", content="newest"),
    ]

    update = middleware.before_model({"messages": messages}, _runtime())
    kept = update["messages"][2:]

    assert not (kept and isinstance(kept[0], ToolMessage))


def test_summary_exception_fails_open_without_state_update(monkeypatch):
    middleware = _middleware()
    monkeypatch.setattr(
        middleware,
        "_create_summary",
        Mock(side_effect=TimeoutError("provider timeout")),
    )
    messages = [
        HumanMessage(id=None, content="message 0"),
        HumanMessage(id="1", content="message 1"),
        HumanMessage(id=None, content=[{"type": "text", "text": "message 2"}]),
        HumanMessage(id="3", content="message 3"),
    ]
    state = {"messages": messages}
    original_list = state["messages"]
    original_message_objs = list(state["messages"])
    original_message_ids = [message.id for message in state["messages"]]
    original_message_contents = [
        copy.deepcopy(message.content) for message in state["messages"]
    ]

    assert middleware.before_model(state, _runtime()) is None

    assert state["messages"] is original_list
    assert len(state["messages"]) == len(original_message_objs)
    for index, message in enumerate(original_message_objs):
        assert state["messages"][index] is message
    assert [message.id for message in state["messages"]] == original_message_ids
    assert [
        copy.deepcopy(message.content) for message in state["messages"]
    ] == original_message_contents


def test_error_sentinel_fails_open():
    middleware = _middleware("Error generating summary: provider timeout")
    state = {
        "messages": [
            HumanMessage(id=str(index), content=f"message {index}")
            for index in range(4)
        ]
    }
    assert middleware.before_model(state, _runtime()) is None


def test_empty_summary_fails_open():
    middleware = _middleware("")
    state = {
        "messages": [
            HumanMessage(id=str(index), content=f"message {index}")
            for index in range(4)
        ]
    }
    assert middleware.before_model(state, _runtime()) is None


def test_async_summary_exception_fails_open(monkeypatch):
    middleware = _middleware()
    monkeypatch.setattr(
        middleware,
        "_acreate_summary",
        AsyncMock(side_effect=TimeoutError("provider timeout")),
    )
    messages = [
        HumanMessage(id=None, content="message 0"),
        HumanMessage(id="1", content="message 1"),
        HumanMessage(id=None, content=[{"type": "text", "text": "message 2"}]),
        HumanMessage(id="3", content="message 3"),
    ]
    state = {"messages": messages}
    original_list = state["messages"]
    original_message_objs = list(state["messages"])
    original_message_ids = [message.id for message in state["messages"]]
    original_message_contents = [
        copy.deepcopy(message.content) for message in state["messages"]
    ]

    result = asyncio.run(middleware.abefore_model(state, _runtime()))

    assert result is None
    assert state["messages"] is original_list
    assert len(state["messages"]) == len(original_message_objs)
    for index, message in enumerate(original_message_objs):
        assert state["messages"][index] is message
    assert [message.id for message in state["messages"]] == original_message_ids
    assert [
        copy.deepcopy(message.content) for message in state["messages"]
    ] == original_message_contents


def test_before_model_stages_audit_record_without_calling_callback():
    callback = Mock()
    runtime = _runtime("chat-9")
    runtime.context.pending_summary_records = []
    middleware = ResilientSummarizationMiddleware(
        model=_FakeSummaryChat(messages=iter([AIMessage(content="durable summary")])),
        summary_model_name="gpt-4.1-mini",
        trigger=("messages", 4),
        keep=("messages", 1),
        token_counter=_count_messages,
        trim_tokens_to_summarize=10000,
        on_summary=callback,
    )
    state = {
        "messages": [
            HumanMessage(id="1", content="old question"),
            AIMessage(id="2", content="old answer"),
            HumanMessage(id="3", content="recent question"),
            HumanMessage(id="4", content="newest"),
        ]
    }

    update = middleware.before_model(state, runtime)

    assert update is not None
    callback.assert_not_called()
    assert len(runtime.context.pending_summary_records) == 1
    pending = runtime.context.pending_summary_records[0]
    assert isinstance(pending, _PendingSummaryAuditRecord)
    assert pending.summary_message_id
    record = pending.record
    assert isinstance(record, SummaryAuditRecord)
    assert record.chat_id == "chat-9"
    assert record.summary_model == "gpt-4.1-mini"
    assert "durable summary" in record.summary_text
    assert record.before_message_count == 4
    assert record.after_message_count == 2


def test_second_before_model_pass_in_one_runtime_stages_only_one_summary():
    runtime = _runtime("chat-9")
    runtime.context.pending_summary_records = []
    runtime.context.summary_compacted = False
    middleware = ResilientSummarizationMiddleware(
        model=_FakeSummaryChat(messages=iter([
            AIMessage(content="first summary"),
            AIMessage(content="second summary"),
        ])),
        summary_model_name="gpt-4.1-mini",
        trigger=("messages", 4),
        keep=("messages", 1),
        token_counter=_count_messages,
        trim_tokens_to_summarize=10000,
        on_summary=Mock(),
    )
    state = {
        "messages": [
            HumanMessage(id="1", content="old question"),
            AIMessage(id="2", content="old answer"),
            HumanMessage(id="3", content="recent question"),
            HumanMessage(id="4", content="newest"),
        ]
    }

    first_update = middleware.before_model(state, runtime)
    second_update = middleware.before_model(state, runtime)

    assert first_update is not None
    assert second_update is None
    assert len(runtime.context.pending_summary_records) == 1
    pending = runtime.context.pending_summary_records[0]
    assert isinstance(pending, _PendingSummaryAuditRecord)
    assert pending.summary_message_id
    assert "first summary" in pending.record.summary_text


def test_before_model_stages_no_record_when_skipped_or_failed_open():
    callback = Mock()
    below_threshold = ResilientSummarizationMiddleware(
        model=_FakeSummaryChat(messages=iter([AIMessage(content="unused")])),
        summary_model_name="gpt-4.1-mini",
        trigger=("messages", 100),
        keep=("messages", 1),
        token_counter=_count_messages,
        trim_tokens_to_summarize=10000,
        on_summary=callback,
    )
    skipped_runtime = _runtime()
    skipped_runtime.context.pending_summary_records = []
    below_threshold.before_model(
        {"messages": [HumanMessage(id="1", content="hi")]}, skipped_runtime
    )

    failed_open = _middleware("Error generating summary: provider timeout")
    failed_open.on_summary = callback
    failed_runtime = _runtime()
    failed_runtime.context.pending_summary_records = []
    failed_open.before_model(
        {
            "messages": [
                HumanMessage(id=str(index), content=f"message {index}")
                for index in range(4)
            ]
        },
        failed_runtime,
    )

    callback.assert_not_called()
    assert skipped_runtime.context.pending_summary_records == []
    assert failed_runtime.context.pending_summary_records == []

