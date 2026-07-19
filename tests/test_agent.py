"""Agent: fake-model tool invocation, key-missing handling, error mapping."""
import asyncio
from unittest.mock import Mock

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import agent as agent_mod
from prompt_builder import PromptBuilder

# GenericFakeChatModel (langchain-core 1.4.8) does not implement bind_tools, but
# create_agent binds the tool set to the model at compile time. A no-op
# bind_tools that returns self lets the fake replay its queued messages
# (including tool_calls) through the real compiled graph without a live API.
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class _FakeChat(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class _Cfg:
    OPENAI_API_KEY = "o"
    XAI_API_KEY = ""            # xAI key intentionally missing
    GEMINI_API_KEY = ""
    TAVILY_API_KEY = ""
    MODEL_TIMEOUT = 60
    MAX_CONTEXT_TOKENS = 16000
    MAX_OUTPUT_TOKENS = 2048


def _prompt_builder():
    return PromptBuilder(default_private_prompt="PRIVATE", default_group_prompt="GROUP")


def _agent_with_fake(fake_model):
    """Build an Agent, then swap in a fake compiled graph over a fake model."""
    a = agent_mod.Agent(
        config=_Cfg,
        prompt_builder=_prompt_builder(),
        checkpointer=InMemorySaver(),
        model_name="gpt-5.4",
    )
    a._compile(fake_model)  # test hook: recompile against an injected model
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


def test_checkpoint_pruning_removes_oldest_messages_to_eighty_percent():
    messages = [HumanMessage(id=str(i), content=f"message {i}") for i in range(501)]

    removals = agent_mod.checkpoint_messages_to_remove(messages)

    assert [message.id for message in removals] == [str(i) for i in range(101)]


def test_checkpoint_pruning_does_nothing_at_limit():
    messages = [HumanMessage(id=str(i), content=f"message {i}") for i in range(500)]

    assert agent_mod.checkpoint_messages_to_remove(messages) == []


def test_checkpoint_pruning_drops_leading_orphaned_tool_messages():
    messages = [
        HumanMessage(id="0", content="old"),
        AIMessage(
            id="1", content="", tool_calls=[
                {"name": "fetch_url", "args": {"url": "https://example.com"}, "id": "c1"}
            ],
        ),
        ToolMessage(id="2", content="result", tool_call_id="c1"),
        HumanMessage(id="3", content="newer"),
        AIMessage(id="4", content="reply"),
        HumanMessage(id="5", content="newest"),
    ]

    removals = agent_mod.checkpoint_messages_to_remove(messages, 5, 4)

    assert [message.id for message in removals] == ["0", "1", "2"]


def test_append_context_message_prunes_checkpoint_to_low_watermark(monkeypatch):
    a = _agent_with_fake(_FakeChat(messages=iter([])))
    monkeypatch.setattr(agent_mod, "MAX_CHECKPOINT_MESSAGES", 5)
    monkeypatch.setattr(agent_mod, "CHECKPOINT_PRUNE_TARGET_MESSAGES", 4)

    for index in range(6):
        a.append_context_message("group", HumanMessage(content=f"message {index}"))

    state = a._graph.get_state(a._config_for("group"))
    assert [message.content for message in state.values["messages"]] == [
        "message 2", "message 3", "message 4", "message 5"
    ]


def test_run_prunes_checkpoint_to_low_watermark(monkeypatch):
    a = _agent_with_fake(_FakeChat(messages=iter([AIMessage(content="reply")])))
    monkeypatch.setattr(agent_mod, "MAX_CHECKPOINT_MESSAGES", 3)
    monkeypatch.setattr(agent_mod, "CHECKPOINT_PRUNE_TARGET_MESSAGES", 2)
    for index in range(3):
        a.append_context_message("chat", HumanMessage(content=f"context {index}"))

    out = asyncio.run(a.run("chat", HumanMessage(content="question"), False))

    assert out == "reply"
    state = a._graph.get_state(a._config_for("chat"))
    assert [message.content for message in state.values["messages"]] == [
        "question", "reply"
    ]


def test_run_returns_response_when_checkpoint_pruning_fails(monkeypatch):
    a = _agent_with_fake(_FakeChat(messages=iter([AIMessage(content="hi there")])))
    monkeypatch.setattr(agent_mod, "MAX_CHECKPOINT_MESSAGES", 1)
    monkeypatch.setattr(agent_mod, "CHECKPOINT_PRUNE_TARGET_MESSAGES", 1)
    monkeypatch.setattr(
        a._graph, "update_state", Mock(side_effect=RuntimeError("checkpoint unavailable"))
    )

    out = asyncio.run(a.run("chat-prune-error", HumanMessage(content="hello"), False))

    assert out == "hi there"
    a._graph.update_state.assert_called_once()
