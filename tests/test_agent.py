"""Agent: fake-model tool invocation, key-missing handling, error mapping."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agent as agent_mod
from prompt_builder import PromptBuilder
from conversation_summary import SummaryAuditRecord, _PendingSummaryAuditRecord

# GenericFakeChatModel (langchain-core 1.4.8) does not implement bind_tools, but
# create_agent binds the tool set to the model at compile time. A no-op
# bind_tools that returns self lets the fake replay its queued messages
# (including tool_calls) through the real compiled graph without a live API.
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class _FakeChat(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class _SummaryFakeChat(_FakeChat):
    calls: int = 0
    fail: bool = False

    def invoke(self, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise TimeoutError("summary unavailable")
        return super().invoke(*args, **kwargs)


class _Cfg:
    OPENAI_API_KEY = "o"
    XAI_API_KEY = ""            # xAI key intentionally missing
    GEMINI_API_KEY = ""
    TAVILY_API_KEY = ""
    MODEL_TIMEOUT = 60
    MAX_CONTEXT_TOKENS = 16000
    MAX_OUTPUT_TOKENS = 2048
    SUMMARY_MODEL = "gpt-4.1-mini"
    SUMMARY_TRIGGER_TOKENS = 10000
    SUMMARY_KEEP_TOKENS = 4000
    SUMMARY_CONTEXT_TOKENS = 14000


def _prompt_builder():
    return PromptBuilder(default_private_prompt="PRIVATE", default_group_prompt="GROUP")


def _agent_with_fake(fake_model, summary_model=None, config=_Cfg, db=None):
    """Build an Agent with fake reply and summary models over real graph state."""
    if summary_model is None:
        summary_model = _FakeChat(messages=iter([]))
    a = agent_mod.Agent(
        config=config,
        prompt_builder=_prompt_builder(),
        checkpointer=InMemorySaver(),
        model_name="gpt-5.4",
        summary_model=summary_model,
        db=db,
    )
    a._compile(fake_model)
    return a


def test_run_returns_final_text():
    fake = _FakeChat(messages=iter([AIMessage(content="hi there")]))
    a = _agent_with_fake(fake)
    out = asyncio.run(a.run("chat-1", HumanMessage(content="hello"), is_group=False))
    assert out == "hi there"


def test_run_flattens_block_list_content():
    # Gemini 3.x models return AIMessage.content as a list of content blocks
    # (e.g. [{"type": "text", "text": "..."}]) instead of a plain string.
    fake = _FakeChat(messages=iter([
        AIMessage(content=[{"type": "text", "text": "hi there"}])
    ]))
    a = _agent_with_fake(fake)
    out = asyncio.run(a.run("chat-2", HumanMessage(content="hello"), is_group=False))
    assert out == "hi there"


def test_missing_provider_key_raises_completion_error():
    a = agent_mod.Agent(
        config=_Cfg, prompt_builder=_prompt_builder(),
        checkpointer=InMemorySaver(), model_name="gpt-5.4",
    )
    a.set_model("grok-4-1-fast-reasoning")  # xAI key blank -> uncompiled
    with pytest.raises(agent_mod.CompletionError) as exc:
        asyncio.run(a.run("chat-1", HumanMessage(content="hi"), is_group=False))
    assert "xAI" in exc.value.user_message


def test_agent_invokes_a_tool_then_answers(monkeypatch):
    # First model turn asks to call fetch_url; second turn answers. fetch_url is
    # the real tool from tools.py, so stub its underlying httpx.get to avoid a
    # live network call (would otherwise hit https://example.com for real).
    class _FakeResponse:
        text = "fake page content"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("tools.httpx.get", lambda *a, **k: _FakeResponse())

    fake = _FakeChat(messages=iter([
        AIMessage(content="", tool_calls=[
            {"name": "fetch_url", "args": {"url": "https://example.com"}, "id": "c1"}]),
        AIMessage(content="done"),
    ]))
    a = _agent_with_fake(fake)
    out = asyncio.run(a.run("chat-tool", HumanMessage(content="read example.com"),
                            is_group=False))
    assert out == "done"


class _SmallSummaryCfg(_Cfg):
    SUMMARY_TRIGGER_TOKENS = 40
    SUMMARY_KEEP_TOKENS = 16
    SUMMARY_CONTEXT_TOKENS = 150
    MAX_CONTEXT_TOKENS = 200
    MAX_OUTPUT_TOKENS = 50


def test_triggered_run_persists_summary_and_recent_messages():
    summary_model = _FakeChat(
        messages=iter([AIMessage(content="Alice prefers window seats.")])
    )
    reply_model = _FakeChat(messages=iter([AIMessage(content="noted")]))
    a = _agent_with_fake(reply_model, summary_model, _SmallSummaryCfg)
    for index in range(4):
        a.append_context_message(
            "summary-chat",
            HumanMessage(content=f"[Alice]: old context {index} " + "word " * 8),
        )

    out = asyncio.run(
        a.run("summary-chat", HumanMessage(content="chatgpt remember that"), True)
    )
    state = a._graph.get_state(a._config_for("summary-chat"))
    summaries = [
        message
        for message in state.values["messages"]
        if message.additional_kwargs.get("lc_source") == "summarization"
    ]

    assert out == "noted"
    assert len(summaries) == 1
    assert "window seats" in summaries[0].content
    assert state.values["messages"][-1].content == "noted"


def test_later_compaction_replaces_previous_summary():
    summary_model = _FakeChat(
        messages=iter([
            AIMessage(content="first rolling summary"),
            AIMessage(content="second rolling summary"),
        ])
    )
    reply_model = _FakeChat(
        messages=iter([AIMessage(content="reply one"), AIMessage(content="reply two")])
    )
    a = _agent_with_fake(reply_model, summary_model, _SmallSummaryCfg)
    for index in range(4):
        a.append_context_message(
            "rolling-chat",
            HumanMessage(content=f"first batch {index} " + "word " * 8),
        )
    asyncio.run(a.run("rolling-chat", HumanMessage(content="first trigger"), False))
    for index in range(4):
        a.append_context_message(
            "rolling-chat",
            HumanMessage(content=f"second batch {index} " + "word " * 8),
        )
    asyncio.run(a.run("rolling-chat", HumanMessage(content="second trigger"), False))

    state = a._graph.get_state(a._config_for("rolling-chat"))
    summaries = [
        message
        for message in state.values["messages"]
        if message.additional_kwargs.get("lc_source") == "summarization"
    ]
    assert len(summaries) == 1
    assert "second rolling summary" in summaries[0].content


def test_passive_append_does_not_invoke_summary_model():
    summary_model = _SummaryFakeChat(
        messages=iter([AIMessage(content="must not be consumed")])
    )
    a = _agent_with_fake(
        _FakeChat(messages=iter([])), summary_model, _SmallSummaryCfg
    )

    for index in range(5):
        a.append_context_message(
            "passive-chat",
            HumanMessage(content=f"passive {index} " + "word " * 8),
        )

    assert summary_model.calls == 0


def test_summary_failure_does_not_block_reply():
    summary_model = _SummaryFakeChat(messages=iter([]), fail=True)
    a = _agent_with_fake(
        _FakeChat(messages=iter([AIMessage(content="fallback reply")])),
        summary_model,
        _SmallSummaryCfg,
    )
    for index in range(4):
        a.append_context_message(
            "failure-chat",
            HumanMessage(content=f"context {index} " + "word " * 8),
        )

    out = asyncio.run(
        a.run("failure-chat", HumanMessage(content="trigger"), False)
    )

    assert out == "fallback reply"


def test_clear_thread_removes_summary_and_recent_state():
    a = _agent_with_fake(
        _FakeChat(messages=iter([AIMessage(content="reply")])),
        _FakeChat(messages=iter([AIMessage(content="summary")])),
        _SmallSummaryCfg,
    )
    for index in range(4):
        a.append_context_message(
            "clear-chat",
            HumanMessage(content=f"context {index} " + "word " * 8),
        )
    asyncio.run(a.run("clear-chat", HumanMessage(content="trigger"), False))

    a.clear_thread("clear-chat")
    state = a._graph.get_state(a._config_for("clear-chat"))

    assert not state.values


def test_set_model_does_not_replace_dedicated_summary_model(monkeypatch):
    summary_model = _FakeChat(messages=iter([]))
    a = _agent_with_fake(
        _FakeChat(messages=iter([])),
        summary_model,
    )
    monkeypatch.setattr(
        agent_mod,
        "init_chat_model",
        lambda *args, **kwargs: _FakeChat(messages=iter([])),
    )

    a.set_model("gpt-5.4-mini")

    assert a._summary_model is summary_model
    assert a._summary_middleware.model is summary_model


def test_successful_summary_records_audit_after_checkpoint_update():
    summary_model = _FakeChat(
        messages=iter([AIMessage(content="Alice prefers window seats.")])
    )
    reply_model = _FakeChat(messages=iter([AIMessage(content="noted")]))
    fake_db = Mock()
    a = _agent_with_fake(reply_model, summary_model, _SmallSummaryCfg, db=fake_db)
    for index in range(4):
        a.append_context_message(
            "audit-chat",
            HumanMessage(content=f"[Alice]: old context {index} " + "word " * 8),
        )

    def assert_summary_was_checkpointed(**kwargs):
        state = a._graph.get_state(a._config_for("audit-chat"))
        summaries = [
            message
            for message in state.values["messages"]
            if message.additional_kwargs.get("lc_source") == "summarization"
        ]
        assert len(summaries) == 1
        assert summaries[0].content == kwargs["summary_text"]

    fake_db.record_conversation_summary.side_effect = assert_summary_was_checkpointed
    out = asyncio.run(
        a.run("audit-chat", HumanMessage(content="chatgpt remember that"), True)
    )

    assert out == "noted"
    fake_db.record_conversation_summary.assert_called_once()
    kwargs = fake_db.record_conversation_summary.call_args.kwargs
    assert kwargs["chat_id"] == "audit-chat"
    assert kwargs["summary_model"] == _SmallSummaryCfg.SUMMARY_MODEL
    assert "window seats" in kwargs["summary_text"]


def test_identical_summary_text_does_not_confirm_wrong_summary_id():
    fake_db = Mock()
    a = _agent_with_fake(
        _FakeChat(messages=iter([])),
        _FakeChat(messages=iter([])),
        _SmallSummaryCfg,
        db=fake_db,
    )
    record = SummaryAuditRecord(
        chat_id="checkpoint-failure-chat",
        summary_text="uncheckpointed summary",
        summary_model=_SmallSummaryCfg.SUMMARY_MODEL,
        before_message_count=4,
        after_message_count=2,
        before_tokens=100,
        after_tokens=20,
    )
    context = agent_mod.AgentContext(
        thread_id="checkpoint-failure-chat",
        pending_summary_records=[
            _PendingSummaryAuditRecord(
                summary_message_id="expected-summary-id",
                record=record,
            )
        ],
    )
    a._graph = SimpleNamespace(
        get_state=lambda *args, **kwargs: SimpleNamespace(
            values={
                "messages": [
                    AIMessage(
                        id="wrong-summary-id",
                        content="uncheckpointed summary",
                        additional_kwargs={"lc_source": "summarization"},
                    )
                ]
            }
        ),
    )

    a._persist_checkpointed_summary_records(
        "checkpoint-failure-chat", context, None
    )
    fake_db.record_conversation_summary.assert_not_called()
    assert context.pending_summary_records == []


def test_successful_invocation_confirms_summary_from_returned_messages():
    fake_db = Mock()
    a = _agent_with_fake(
        _FakeChat(messages=iter([])),
        _FakeChat(messages=iter([])),
        _SmallSummaryCfg,
        db=fake_db,
    )
    record = SummaryAuditRecord(
        chat_id="successful-chat",
        summary_text="durable summary",
        summary_model=_SmallSummaryCfg.SUMMARY_MODEL,
        before_message_count=4,
        after_message_count=2,
        before_tokens=100,
        after_tokens=20,
    )

    def invoke(*args, **kwargs):
        kwargs["context"].pending_summary_records.append(
            _PendingSummaryAuditRecord("summary-id", record)
        )
        return {
            "messages": [
                AIMessage(
                    id="summary-id",
                    content="durable summary",
                    additional_kwargs={"lc_source": "summarization"},
                ),
                AIMessage(content="reply"),
            ]
        }

    a._graph = SimpleNamespace(
        invoke=invoke,
        get_state=Mock(side_effect=RuntimeError("must not inspect checkpoint")),
    )

    out = asyncio.run(
        a.run("successful-chat", HumanMessage(content="trigger"), False)
    )

    assert out == "reply"
    fake_db.record_conversation_summary.assert_called_once()


def test_checkpoint_inspection_failure_after_reply_error_does_not_audit():
    fake_db = Mock()
    a = _agent_with_fake(
        _FakeChat(messages=iter([])),
        _FakeChat(messages=iter([])),
        _SmallSummaryCfg,
        db=fake_db,
    )
    record = SummaryAuditRecord(
        chat_id="checkpoint-error-chat",
        summary_text="durable summary",
        summary_model=_SmallSummaryCfg.SUMMARY_MODEL,
        before_message_count=4,
        after_message_count=2,
        before_tokens=100,
        after_tokens=20,
    )

    def fail_after_staging(*args, **kwargs):
        kwargs["context"].pending_summary_records.append(
            _PendingSummaryAuditRecord("summary-id", record)
        )
        raise RuntimeError("reply failed")

    a._graph = SimpleNamespace(
        invoke=fail_after_staging,
        get_state=Mock(side_effect=RuntimeError("checkpoint unavailable")),
    )

    with pytest.raises(agent_mod.CompletionError):
        asyncio.run(
            a.run("checkpoint-error-chat", HumanMessage(content="trigger"), False)
        )

    fake_db.record_conversation_summary.assert_not_called()


def test_reply_failure_records_checkpointed_summary_once():
    class _FailingReplyChat(_FakeChat):
        def invoke(self, *args, **kwargs):
            raise RuntimeError("reply failed")

    summary_model = _FakeChat(messages=iter([AIMessage(content="durable summary")]))
    fake_db = Mock()
    a = _agent_with_fake(
        _FailingReplyChat(messages=iter([])),
        summary_model,
        _SmallSummaryCfg,
        db=fake_db,
    )
    for index in range(4):
        a.append_context_message(
            "reply-failure-chat",
            HumanMessage(content=f"context {index} " + "word " * 8),
        )

    with pytest.raises(agent_mod.CompletionError):
        asyncio.run(
            a.run("reply-failure-chat", HumanMessage(content="trigger"), False)
        )

    fake_db.record_conversation_summary.assert_called_once()


def test_audit_write_failure_does_not_block_reply():
    summary_model = _FakeChat(
        messages=iter([AIMessage(content="Bob prefers aisle seats.")])
    )
    reply_model = _FakeChat(messages=iter([AIMessage(content="ok")]))
    fake_db = Mock()
    fake_db.record_conversation_summary.side_effect = RuntimeError("db down")
    a = _agent_with_fake(reply_model, summary_model, _SmallSummaryCfg, db=fake_db)
    for index in range(4):
        a.append_context_message(
            "audit-failure-chat",
            HumanMessage(content=f"[Bob]: old context {index} " + "word " * 8),
        )

    out = asyncio.run(
        a.run("audit-failure-chat", HumanMessage(content="chatgpt remember that"), True)
    )

    assert out == "ok"
