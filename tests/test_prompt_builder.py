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


def test_to_lc_human_message_sets_id_when_provided():
    msg = _pb().to_lc_human_message(text="hi", message_id="abc-123")
    assert msg.id == "abc-123"


def test_to_lc_human_message_no_id_by_default():
    msg = _pb().to_lc_human_message(text="hi")
    assert msg.id is None


def test_tools_section_renders_bound_tools_with_usage():
    out = _pb().build_system_prompt(
        is_group=False, tool_names=["web_search", "fetch_url", "get_image"]
    )
    assert "## Tools" in out
    for name in ("web_search", "fetch_url", "get_image"):
        assert f"- {name} —" in out


def test_unknown_tool_is_listed_without_usage_text():
    out = _pb().build_system_prompt(is_group=False, tool_names=["mystery_tool"])
    assert "- mystery_tool" in out


def test_no_tools_section_when_no_tools_bound():
    out = _pb().build_system_prompt(is_group=False)
    assert "## Tools" not in out
    assert "## Conventions" in out


def test_system_prompt_has_no_per_call_data():
    """Volatile data lives in the context message so this stays cacheable."""
    out = _pb().build_system_prompt(is_group=False, tool_names=["web_search"])
    assert "Date/time" not in out
    assert "Current context" not in out


def test_markdown_rule_present_for_both_private_and_group():
    assert "Markdown" in _pb().build_system_prompt(is_group=False)
    assert "Markdown" in _pb().build_system_prompt(is_group=True)


def test_markdown_rule_survives_custom_personality():
    out = _pb(active="villain", prompt_for="BE EVIL").build_system_prompt(is_group=True)
    assert "BE EVIL" in out
    assert "Markdown" in out
    assert out.index("BE EVIL") < out.index("Markdown")


def test_group_prefix_convention_is_group_only():
    assert "[Name]: content" in _pb().build_system_prompt(is_group=True)
    assert "[Name]: content" not in _pb().build_system_prompt(is_group=False)


def test_context_message_has_date_and_reply():
    msg = _pb().build_context_message(reply_context=("Alice", "the sketch"))
    assert "## Current context" in msg.content
    assert "Date/time:" in msg.content
    assert 'Replying to a previous message from Alice: "the sketch"' in msg.content


def test_context_message_omits_reply_line_when_absent():
    msg = _pb().build_context_message()
    assert "Date/time:" in msg.content
    assert "Replying to" not in msg.content
