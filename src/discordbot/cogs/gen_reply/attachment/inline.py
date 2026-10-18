"""Inline attachment renderer for whenever attachments cannot go by Gemini Files URI."""

from datetime import UTC, datetime, timedelta
from collections import OrderedDict

import logfire
from nextcord import Attachment, StickerItem
from pydantic import PrivateAttr
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.images import to_data_uri
from discordbot.typings.media import RenderedAttachment
from discordbot.typings.context_budgets import MAX_INLINE_ATTACHMENT_BYTES
from discordbot.cogs.gen_reply.attachment.base import AttachmentRenderer, loggable_cache_key
from discordbot.cogs.gen_reply.attachment.loaders import (
    attachment_mime,
    load_image_bytes,
    load_attachment_bytes,
    resolve_source_filename,
)


def _inline_expiry() -> datetime:
    """Returns the 12-hour cache expiry for an inlined attachment part."""
    return datetime.now(tz=UTC) + timedelta(hours=12)


class InlineRenderer(AttachmentRenderer):
    """Inlines attachments as base64 / text parts.

    Selected for any non-Gemini answer model, none of which can resolve a Gemini Files URI,
    for a Gemini one when no Gemini key is configured to upload with, and for every provider
    while `file_api_enabled` is off. Every render fetches the source and embeds it directly in
    the request, so there is no upload handle to track and `allow_dead_cache` is ignored; all it
    remembers is which files it has read and cannot carry. Images inline as `input_image` base64,
    PDFs as base64 `input_file`, UTF-8 files as `input_text`, and anything else is dropped, as is
    an attachment past `MAX_INLINE_ATTACHMENT_BYTES`.
    """

    dropped_modalities = frozenset({"video", "audio"})

    # Files whose bytes turned out to be neither a PDF nor UTF-8, which only the bytes can tell (a
    # Big5 `.txt` is `text/plain` like a UTF-8 one). `carries` refuses them from then on, so the
    # route marker, the history media budget and the render cache leave them out after the first
    # read. Bounded like the render cache.
    _unreadable: OrderedDict[int | str, None] = PrivateAttr(default_factory=OrderedDict)

    def carries(self, content_type: str, cache_key: int | str, size: int | None) -> bool:
        """Returns whether `content_type`, `size`, and `cache_key` are eligible for inline rendering.

        Refuses fonts, Android packages, files exceeding `MAX_INLINE_ATTACHMENT_BYTES`, and files
        previously recorded in `_unreadable`.

        Args:
            content_type (str): Resolved lowercase MIME type of the source.
            cache_key (int | str): Per-source identifier checked against `_unreadable`.
            size (int | None): Byte size reported by Discord, or `None` when unknown.
        """
        if content_type.startswith("font/") or (
            content_type == "application/vnd.android.package-archive"
        ):
            return False
        if size is not None and size > MAX_INLINE_ATTACHMENT_BYTES:
            return False
        return cache_key not in self._unreadable

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> RenderedAttachment | None:
        """Returns `source` downscaled and inlined as a base64 `input_image` part, or `None`.

        Args:
            source (Attachment | StickerItem | str): Image attachment, sticker, or URL to inline.
            cache_key (int | str): Per-source identifier for logging.
            allow_dead_cache (bool): Ignored; inline renders do not use the dead-source cache.
        """
        loaded = await self._load_source_bytes(
            cache_key=cache_key,
            filename=resolve_source_filename(source=source, url_fallback="image.png"),
            load_data=lambda: load_image_bytes(source=source),
            allow_dead_cache=False,
        )
        if loaded is None:
            return None
        image_part = ResponseInputImageParam(
            type="input_image",
            image_url=to_data_uri(data=loaded.data, mime_type=loaded.mime_type),
            detail="auto",
        )
        return RenderedAttachment(part=image_part, expires_at=_inline_expiry())

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns `attachment` inlined as a PDF `input_file` or UTF-8 `input_text` part, or `None`.

        Args:
            attachment (Attachment): Discord file attachment to inline.
            cache_key (int | str): Per-source identifier recorded in `_unreadable` if non-UTF-8.
            allow_dead_cache (bool): Ignored; inline renders do not use the dead-source cache.
        """
        mime_type = attachment_mime(attachment=attachment)
        if not mime_type:
            logfire.warn(
                "skipping attachment with unknown MIME type",
                filename=attachment.filename,
                url=attachment.url,
            )
            return None
        if mime_type.partition("/")[0] in self.dropped_modalities:
            # The modality gate already keeps these out, so only a direct caller gets here;
            # `_inline_file_part` would drop the clip too, but only after downloading all of it.
            logfire.warn(
                "dropping video / audio attachment the inline renderer cannot carry",
                filename=attachment.filename,
                mime_type=mime_type,
            )
            return None
        loaded = await self._load_source_bytes(
            cache_key=cache_key,
            filename=attachment.filename,
            load_data=lambda: load_attachment_bytes(attachment=attachment),
            allow_dead_cache=False,
        )
        if loaded is None:
            return None
        return self._inline_file_part(
            filename=attachment.filename,
            data=loaded.data,
            mime_type=mime_type,
            cache_key=cache_key,
        )

    def _inline_file_part(
        self, filename: str, data: bytes, mime_type: str, cache_key: int | str
    ) -> RenderedAttachment | None:
        """Returns `data` as a base64 PDF `input_file` or UTF-8 `input_text` part, recording unreadable files."""
        if mime_type == "application/pdf":
            pdf_part = ResponseInputFileParam(
                type="input_file",
                filename=filename,
                file_data=to_data_uri(data=data, mime_type=mime_type),
            )
            return RenderedAttachment(part=pdf_part, expires_at=_inline_expiry())
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            logfire.warn(
                "dropping non-text, non-PDF attachment the inline renderer cannot carry",
                filename=filename,
                mime_type=mime_type,
                cache_key=loggable_cache_key(cache_key=cache_key),
            )
            self._unreadable[cache_key] = None
            if len(self._unreadable) > 128:
                self._unreadable.popitem(last=False)
            return None
        text_part = ResponseInputTextParam(
            type="input_text", text=f"[attached file: {filename}]\n{text}"
        )
        return RenderedAttachment(part=text_part, expires_at=_inline_expiry())
