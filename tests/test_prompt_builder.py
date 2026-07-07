"""PromptBuilder: system prompt resolution + LangChain message construction."""
from langchain_core.messages import HumanMessage
from prompt_builder import PromptBuilder


def _pb(active=None, prompt_for=None):
    return PromptBuilder(
        default_private_prompt="PRIVATE",
        default_group_prompt="GROUP",
        get_active_personality=(lambda: active) if active else None,
        get_personality_prompt=(lambda name: prompt_for) if prompt_for is not None else None,
    )


def test_private_uses_default_private_prompt():
    out = _pb().build_system_prompt(is_group=False)
    assert "PRIVATE" in out


def test_group_uses_active_personality_prompt():
    out = _pb(active="villain", prompt_for="BE EVIL").build_system_prompt(is_group=True)
    assert "BE EVIL" in out


def test_group_falls_back_to_default_when_personality_missing():
    out = _pb(active="ghost", prompt_for=None).build_system_prompt(is_group=True)
    assert "GROUP" in out


def test_to_lc_human_message_plain_text():
    msg = _pb().to_lc_human_message(text="hello", is_group=False)
    assert isinstance(msg, HumanMessage)
    assert msg.content == "hello"


def test_group_message_gets_sender_prefix():
    msg = _pb().to_lc_human_message(text="hi", is_group=True, sender_name="Alice")
    assert msg.content == "[Alice]: hi"


def test_image_message_has_text_and_image_blocks():
    msg = _pb().to_lc_human_message(
        text="what is this?", image_data_url="data:image/jpeg;base64,AAAA"
    )
    types = [b["type"] for b in msg.content]
    assert types == ["text", "image_url"]
    assert msg.content[1]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"
