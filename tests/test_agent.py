"""Agent: fake-model tool invocation, key-missing handling, error mapping."""
import asyncio
import pytest

from langchain_core.messages import AIMessage, HumanMessage
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
    OPENAI_TIMEOUT = 60
    MAX_CONTEXT_TOKENS = 16000
    RESERVE_TOKENS_TEXT = 2000
    RESERVE_TOKENS_IMAGE = 3000


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
