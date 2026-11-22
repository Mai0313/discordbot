"""Inline attachment renderer for whenever attachments cannot go by Gemini Files URI."""

from typing import TYPE_CHECKING, cast
from datetime import UTC, datetime, timedelta

import logfire
from pydantic import PrivateAttr
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.lru import BoundedLRU
from discordbot.typings.media import LoadedMedia, RenderedAttachment
from discordbot.utils.llm_transcript import defuse_separator_fences
from discordbot.cogs.gen_reply.images import to_data_uri
from discordbot.typings.context_budgets import MAX_INLINE_ATTACHMENT_BYTES
from discordbot.cogs.gen_reply.attachment.base import (
    FileBytesLoader,
    AttachmentSource,
    AttachmentRenderer,
    media_semaphore,
    loggable_cache_key,
)
from discordbot.cogs.gen_reply.attachment.loaders import load_image_bytes, load_attachment_bytes

if TYPE_CHECKING:
    from nextcord import Attachment


def _inline_expiry() -> datetime:
    """Returns the 12-hour cache expiry for an inlined attachment part."""
    return datetime.now(tz=UTC) + timedelta(hours=12)


class InlineRenderer(AttachmentRenderer):
    """Inlines attachments as base64 or UTF-8 text parts, ignoring `allow_dead_cache`.

    Images inline as base64 `input_image`, PDFs as base64 `input_file`, and UTF-8 files as
    `input_text`. Audio, video, files over `MAX_INLINE_ATTACHMENT_BYTES`, and non-UTF-8 binary
    files are dropped and recorded so `carries` rejects them on later turns.
    """

    dropped_modalities = frozenset({"video", "audio"})

    # Per-instance LRU (max 128) of unreadable non-PDF/non-UTF-8 keys refused by `carries`.
    _unreadable: BoundedLRU[int | str, None] = PrivateAttr(default_factory=BoundedLRU)

    def carries(self, source: AttachmentSource) -> bool:
        """Returns whether `source` is eligible for inline rendering.

        Refuses fonts, Android packages, files exceeding `MAX_INLINE_ATTACHMENT_BYTES`, and files
        previously recorded in `_unreadable`.

        Args:
            source (AttachmentSource): Classified attachment metadata to check.
        """
        if source.content_type.startswith("font/") or (
            source.content_type == "application/vnd.android.package-archive"
        ):
            return False
        if source.size is not None and source.size > MAX_INLINE_ATTACHMENT_BYTES:
            return False
        return source.cache_key not in self._unreadable

    async def _load_source_bytes(
        self,
        cache_key: int | str,
        filename: str,
        load_data: FileBytesLoader,
        allow_dead_cache: bool,
    ) -> LoadedMedia | None:
        """Fetches one source's bytes under the process-wide media concurrency semaphore."""
        async with media_semaphore.get():
            return await super()._load_source_bytes(
                cache_key=cache_key,
                filename=filename,
                load_data=load_data,
                allow_dead_cache=allow_dead_cache,
            )

    async def render_image(
        self, source: AttachmentSource, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns `source` downscaled and inlined as a base64 `input_image` part, or `None`.

        Args:
            source (AttachmentSource): Classified image attachment, sticker, or URL to inline.
            allow_dead_cache (bool): Ignored; inline renders do not use the dead-source cache.
        """
        del allow_dead_cache
        loaded = await self._load_source_bytes(
            cache_key=source.cache_key,
            filename=source.filename,
            load_data=lambda: load_image_bytes(source=source.handle),
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
        self, source: AttachmentSource, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns `source` inlined as a PDF `input_file` or UTF-8 `input_text` part, or `None`.

        Args:
            source (AttachmentSource): Classified Discord file attachment to inline.
            allow_dead_cache (bool): Ignored; inline renders do not use the dead-source cache.
        """
        del allow_dead_cache
        if source.content_type.partition("/")[0] in self.dropped_modalities:
            # The modality gate already keeps these out, so only a direct caller gets here;
            # `_inline_file_part` would drop the clip too, but only after downloading all of it.
            logfire.warn(
                "dropping video / audio attachment the inline renderer cannot carry",
                filename=source.filename,
                mime_type=source.content_type,
            )
            return None
        # Only attachments are classified as `kind="file"`; stickers and embeds are images.
        attachment = cast("Attachment", source.handle)
        loaded = await self._load_source_bytes(
            cache_key=source.cache_key,
            filename=source.filename,
            load_data=lambda: load_attachment_bytes(attachment=attachment),
            allow_dead_cache=False,
        )
        if loaded is None:
            return None
        return self._inline_file_part(
            filename=source.filename,
            data=loaded.data,
            mime_type=source.content_type,
            cache_key=source.cache_key,
        )

    def _inline_file_part(
        self, filename: str, data: bytes, mime_type: str, cache_key: int | str
    ) -> RenderedAttachment | None:
        """Returns a base64 PDF or UTF-8 text part, recording non-UTF-8 keys in `_unreadable`."""
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
                _exc_info=True,
            )
            self._unreadable.set(key=cache_key, value=None)
            return None
        text_part = ResponseInputTextParam(
            type="input_text",
            text=defuse_separator_fences(text=f"[attached file: {filename}]\n{text}"),
        )
        return RenderedAttachment(part=text_part, expires_at=_inline_expiry())
