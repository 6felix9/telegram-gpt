"""Vision summary helper and the get_image agent tool.

Kept separate from tools.py (web search / fetch) so image concerns stay
in one focused module. build_image_tool binds a Database so the tool can
scope retrieval to the calling chat via runtime context."""
from __future__ import annotations

import base64
import logging

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage

from token_budget import _message_text

logger = logging.getLogger(__name__)

IMAGE_SUMMARY_PROMPT = (
    "Describe this image in 2-4 sentences. Note key objects, any visible text, "
    "layout, and notable details someone might ask about later. Be concise and factual."
)


def make_image_summary(model, image_data_url: str) -> str:
    """Run the vision model on one image and return its flattened text reply."""
    message = HumanMessage(content=[
        {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ])
    response = model.invoke([message])
    return _message_text(response).strip()


def build_image_blocks(db, chat_id, image_id: int) -> list[dict]:
    """Pure, directly-testable core of get_image: chat-scoped lookup -> blocks."""
    if chat_id is None:
        return [{"type": "text", "text": "Image not available."}]
    record = db.get_image(chat_id, image_id)
    if record is None:
        return [{"type": "text", "text": f"Image #{image_id} not found."}]
    b64 = base64.b64encode(record.image_bytes).decode("utf-8")
    caption = f" ({record.caption})" if record.caption else ""
    return [
        {"type": "text", "text": f"Image #{image_id}{caption}:"},
        {"type": "image", "base64": b64, "mime_type": record.mime_type},
    ]


def build_image_tool(db):
    """Return a get_image tool bound to db, scoped to the calling chat."""

    @tool
    def get_image(image_id: int, runtime: ToolRuntime) -> list[dict]:
        """Retrieve a previously shared image so you can see the full picture.

        Use this when an [image #N] marker's text description is not enough
        to answer a question about that image.

        Args:
            image_id: The numeric id from an [image #N] marker.
        """
        context = getattr(runtime, "context", None)
        chat_id = getattr(context, "thread_id", None)
        return build_image_blocks(db, chat_id, image_id)

    return get_image
