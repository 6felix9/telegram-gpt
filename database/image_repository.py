"""Durable image blob persistence, keyed by chat for scoped retrieval."""
import logging
from dataclasses import dataclass

from .db_connection import ConnectionManager

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    id: int
    chat_id: str
    mime_type: str
    caption: str | None
    summary: str
    image_bytes: bytes


class ImageRepository:
    """CRUD for the `images` table. Retrieval is always chat-scoped."""

    def __init__(self, conn: ConnectionManager):
        self._conn = conn

    def save_image(
        self,
        chat_id: str,
        message_id: int | None,
        mime_type: str,
        caption: str | None,
        summary: str,
        image_bytes: bytes,
    ) -> int:
        """Persist one image and return its stable id."""
        chat_id = str(chat_id)
        with self._conn.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO images
                    (chat_id, message_id, mime_type, caption, summary, image_bytes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (chat_id, message_id, mime_type, caption, summary, image_bytes),
                )
                row_id = cur.fetchone()[0]
        logger.info("Saved image %s for chat %s", row_id, chat_id)
        return row_id

    def get_image(self, chat_id: str, image_id: int) -> ImageRecord | None:
        """Fetch one image by id, scoped to chat_id. Returns None if the id
        does not exist OR belongs to a different chat (the isolation boundary)."""
        chat_id = str(chat_id)
        with self._conn.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chat_id, message_id, mime_type, caption, summary, image_bytes
                    FROM images
                    WHERE id = %s AND chat_id = %s
                    """,
                    (image_id, chat_id),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return ImageRecord(
            id=image_id,
            chat_id=row[0],
            mime_type=row[2],
            caption=row[3],
            summary=row[4],
            image_bytes=bytes(row[5]),
        )
