"""Image summary helper + get_image tool: chat-scoped retrieval, multimodal
return shape, and text-flattening of the vision model reply."""
from dataclasses import dataclass
from types import SimpleNamespace

from langchain_core.messages import AIMessage

import image_store


@dataclass
class _Rec:
    id: int
    chat_id: str
    mime_type: str
    caption: str | None
    summary: str
    image_bytes: bytes


class _FakeModel:
    def __init__(self, reply):
        self._reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self._reply)


def test_make_image_summary_returns_text():
    model = _FakeModel("A tabby cat on a sofa.")
    out = image_store.make_image_summary(model, "data:image/jpeg;base64,AAAA")
    assert out == "A tabby cat on a sofa."
    # The image data-URL was sent as an image_url block.
    sent = model.calls[0][0].content
    assert any(b.get("type") == "image_url" for b in sent)


def test_build_image_blocks_returns_multimodal_for_matching_chat():
    db = SimpleNamespace(
        get_image=lambda chat_id, image_id: _Rec(
            image_id, chat_id, "image/jpeg", "a cat", "summary", b"\x00\x01")
    )
    result = image_store.build_image_blocks(db, "123", 42)
    types = [b["type"] for b in result]
    assert types == ["text", "image"]
    assert result[0]["text"] == "Image #42 (a cat):"
    assert result[1]["mime_type"] == "image/jpeg"
    assert result[1]["base64"] == "AAE="  # base64 of b"\x00\x01"


def test_build_image_blocks_not_found_returns_text_only():
    db = SimpleNamespace(get_image=lambda chat_id, image_id: None)
    result = image_store.build_image_blocks(db, "123", 999)
    assert result == [{"type": "text", "text": "Image #999 not found."}]


def test_build_image_blocks_none_chat_returns_unavailable():
    db = SimpleNamespace(get_image=lambda chat_id, image_id: None)
    result = image_store.build_image_blocks(db, None, 5)
    assert result == [{"type": "text", "text": "Image not available."}]


def test_build_image_tool_is_named_get_image():
    tool = image_store.build_image_tool(SimpleNamespace(get_image=lambda *a: None))
    assert tool.name == "get_image"
