"""Pure compaction logic: sanitization, keep-suffix selection, plan()."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

from conversation_summary import (
    SUMMARY_HEADING,
    CompactionPlan,
    ConversationCompactor,
    SummaryAuditRecord,
    SummaryGenerationError,
    sanitize_summary_messages,
    select_keep_suffix,
)

NO_HISTORY_PLACEHOLDER = "No previous conversation history."
TOO_LONG_PLACEHOLDER = "Previous conversation was too long to summarize."


class _FakeModel:
    """Minimal stand-in for a chat model: replays queued replies, counts calls."""

    def __init__(self, replies=("durable summary",), error=None):
        self.replies = list(replies)
        self.error = error
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.replies.pop(0))


def _compactor(model=None, trigger_tokens=100, max_summary_output=40, on_summary=None):
    return ConversationCompactor(
        model=model if model is not None else _FakeModel(),
        summary_model_name="gpt-5.6-luna",
        trigger_tokens=trigger_tokens,
        max_summary_output=max_summary_output,
        on_summary=on_summary,
    )


def _big(text, repeat=60):
    """A message body large enough to push state over a small test trigger.

    At repeat=60 one message costs roughly 66 tokens, so two of them clear the
    100-token trigger the plan() tests use while the short "recent" messages
    stay well inside the 60-token post-summary keep budget.
    """
    return f"{text} " + ("word " * repeat)


# --- sanitization -----------------------------------------------------------

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


# --- keep-suffix selection --------------------------------------------------

def test_keep_suffix_starts_at_last_human_message():
    messages = [
        HumanMessage(content="old question"),
        AIMessage(content="old answer"),
        HumanMessage(content="recent question"),
        AIMessage(content="recent answer"),
    ]

    keep = select_keep_suffix(messages, trigger_tokens=1000, max_summary_output=100)

    assert [message.content for message in keep] == [
        "recent question",
        "recent answer",
    ]


def test_keep_suffix_includes_trailing_tool_messages():
    messages = [
        HumanMessage(content="old question"),
        HumanMessage(content="recent question"),
        AIMessage(
            content="",
            tool_calls=[{"name": "fetch_url", "args": {"url": "https://x.test"}, "id": "c1"}],
        ),
        ToolMessage(content="page text", tool_call_id="c1"),
        AIMessage(content="recent answer"),
    ]

    keep = select_keep_suffix(messages, trigger_tokens=1000, max_summary_output=100)

    assert len(keep) == 4
    assert isinstance(keep[0], HumanMessage)
    assert keep[0].content == "recent question"


def test_keep_suffix_never_starts_with_an_orphaned_tool_message():
    messages = [
        HumanMessage(content="question"),
        AIMessage(
            content="",
            tool_calls=[{"name": "fetch_url", "args": {"url": "https://x.test"}, "id": "c1"}],
        ),
        ToolMessage(content="page text", tool_call_id="c1"),
    ]

    keep = select_keep_suffix(messages, trigger_tokens=1000, max_summary_output=100)

    assert not isinstance(keep[0], ToolMessage)


def test_keep_suffix_is_empty_without_a_human_message():
    messages = [AIMessage(content="only an answer")]
    assert select_keep_suffix(messages, trigger_tokens=1000, max_summary_output=100) == []


def test_keep_suffix_is_dropped_when_it_exceeds_the_post_summary_budget():
    # Budget is trigger - max_summary_output = 60; this single message is larger.
    messages = [
        HumanMessage(content="old"),
        HumanMessage(content=_big("enormous", repeat=100)),
    ]

    assert select_keep_suffix(messages, trigger_tokens=100, max_summary_output=40) == []


def test_keep_suffix_ignores_summary_human_message():
    messages = [
        HumanMessage(content=f"{SUMMARY_HEADING}\n\nAlice likes window seats."),
        AIMessage(content="Got it."),
    ]
    assert select_keep_suffix(messages, trigger_tokens=1000, max_summary_output=100) == []


# --- plan() -----------------------------------------------------------------

def test_plan_returns_none_below_threshold():
    model = _FakeModel()
    compactor = _compactor(model, trigger_tokens=100_000)

    assert compactor.plan("chat-1", [HumanMessage(content="hi")]) is None
    assert model.calls == []


def test_plan_ignores_summary_tokens_in_trigger_check():
    # Large summary + small chat message: total tokens > trigger, but chat tokens < trigger.
    model = _FakeModel()
    compactor = _compactor(model, trigger_tokens=100, max_summary_output=40)
    messages = [
        HumanMessage(content=f"{SUMMARY_HEADING}\n\n" + ("word " * 150)),
        HumanMessage(content="short question"),
    ]

    assert compactor.plan("chat-1", messages) is None
    assert model.calls == []


def test_plan_triggers_when_chat_tokens_alone_cross_threshold():
    # Large summary + large chat messages: chat tokens > trigger.
    model = _FakeModel(["Updated summary."])
    compactor = _compactor(model, trigger_tokens=100, max_summary_output=40)
    messages = [
        HumanMessage(content=f"{SUMMARY_HEADING}\n\n" + ("word " * 50)),
        HumanMessage(content=_big("new question")),
        AIMessage(content=_big("new answer")),
    ]

    plan = compactor.plan("chat-1", messages)
    assert isinstance(plan, CompactionPlan)
    assert len(model.calls) == 1
    assert plan.record.summary_text == "Updated summary."
    assert plan.record.before_message_count == 3


def test_plan_replaces_state_with_summary_and_last_exchange():
    model = _FakeModel(["Alice prefers window seats."])
    compactor = _compactor(model, trigger_tokens=100, max_summary_output=40)
    messages = [
        HumanMessage(content=_big("old question")),
        AIMessage(content=_big("old answer")),
        HumanMessage(content="recent question"),
        AIMessage(content="recent answer"),
    ]

    plan = compactor.plan("chat-1", messages)

    assert isinstance(plan, CompactionPlan)
    assert len(plan.messages) == 3
    assert plan.messages[0].content == (
        f"{SUMMARY_HEADING}\n\nAlice prefers window seats."
    )
    assert [message.content for message in plan.messages[1:]] == [
        "recent question",
        "recent answer",
    ]


def test_plan_summarizes_every_message_including_the_kept_suffix():
    model = _FakeModel(["rolling summary"])
    compactor = _compactor(model, trigger_tokens=100, max_summary_output=40)
    messages = [
        HumanMessage(content=_big("old question")),
        AIMessage(content=_big("old answer")),
        HumanMessage(content="recent question"),
    ]

    compactor.plan("chat-1", messages)

    assert len(model.calls) == 1
    prompt = model.calls[0]
    assert "old question" in prompt
    assert "old answer" in prompt
    assert "recent question" in prompt


def test_plan_sanitizes_images_out_of_the_summary_input():
    model = _FakeModel(["summary"])
    compactor = _compactor(model, trigger_tokens=10, max_summary_output=4)
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "A receipt"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,SECRET"},
                },
            ]
        ),
        HumanMessage(content="what does it say"),
    ]

    compactor.plan("chat-1", messages)

    assert "SECRET" not in model.calls[0]
    assert "A receipt" in model.calls[0]
    assert "SECRET" in str(messages[0].content)  # original state untouched


def test_plan_builds_the_audit_record():
    model = _FakeModel(["Alice prefers window seats."])
    compactor = _compactor(model, trigger_tokens=100, max_summary_output=40)
    messages = [
        HumanMessage(content=_big("old question")),
        AIMessage(content=_big("old answer")),
        HumanMessage(content="recent question"),
    ]

    record = compactor.plan("chat-9", messages).record

    assert isinstance(record, SummaryAuditRecord)
    assert record.chat_id == "chat-9"
    assert record.summary_model == "gpt-5.6-luna"
    assert record.summary_text == "Alice prefers window seats."
    assert record.before_message_count == 3
    assert record.after_message_count == 2
    assert record.before_tokens > record.after_tokens


def test_plan_raises_on_model_error():
    compactor = _compactor(
        _FakeModel(error=TimeoutError("provider timeout")),
        trigger_tokens=10,
        max_summary_output=4,
    )
    with pytest.raises(TimeoutError):
        compactor.plan("chat-1", [HumanMessage(content=_big("a"))])


@pytest.mark.parametrize(
    "unusable",
    [
        "",
        "   ",
        "Error generating summary: provider timeout",
        NO_HISTORY_PLACEHOLDER,
        TOO_LONG_PLACEHOLDER,
    ],
)
def test_plan_raises_on_unusable_summary(unusable):
    compactor = _compactor(
        _FakeModel([unusable]), trigger_tokens=10, max_summary_output=4
    )
    with pytest.raises(SummaryGenerationError):
        compactor.plan("chat-1", [HumanMessage(content=_big("a"))])


def test_plan_converts_stop_iteration_into_a_summary_error():
    # StopIteration must never escape: plan() runs in asyncio.to_thread, where a
    # StopIteration on the future would hang the await instead of failing open.
    compactor = _compactor(
        _FakeModel(error=StopIteration()), trigger_tokens=10, max_summary_output=4
    )
    with pytest.raises(SummaryGenerationError):
        compactor.plan("chat-1", [HumanMessage(content=_big("a"))])


def test_validate_summary_allows_legitimate_text_with_similar_words():
    text = (
        "Previous conversation covered travel plans; no previous conversation "
        "history was discarded because it was too long to summarize in full."
    )
    assert ConversationCompactor._validate_summary(text) == text


def test_plan_flattens_block_list_summary_content():
    class _BlockListModel(_FakeModel):
        def invoke(self, prompt):
            self.calls.append(prompt)
            return AIMessage(content=[{"type": "text", "text": "block summary"}])

    compactor = _compactor(_BlockListModel(), trigger_tokens=10, max_summary_output=4)

    plan = compactor.plan("chat-1", [HumanMessage(content=_big("a"))])

    assert plan.record.summary_text == "block summary"
