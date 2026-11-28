"""Anthropic Files API attachment renderer for Claude answer models.

Uploads supported images and PDFs directly via `anthropic.AsyncAnthropic` with a 48h TTL and
references each file ID as an `input_file` part carrying LiteLLM's `format` MIME key (LiteLLM
rejects `input_image` with `file_id` and routes unrecognized MIME types to `container_upload`).
Non-PDF text and code files inline as UTF-8 `input_text` because LiteLLM drops filenames on
Anthropic document blocks and rejects text MIME subtypes other than `text/plain`.
"""

import io
import time
from typing import TYPE_CHECKING, cast
import asyncio
from datetime import UTC, datetime, timedelta
from collections.abc import Callable

import logfire
from pydantic import Field, PrivateAttr
from anthropic import AsyncAnthropic
from openai.types.responses.response_input_text_param import ResponseInputTextParam

from discordbot.utils.lru import BoundedLRU
from discordbot.typings.media import UploadedFile, RenderedAttachment, FormattedInputFileParam
from discordbot.typings.timeouts import ATTACHMENT_UPLOAD_TIMEOUT_SECONDS
from discordbot.utils.llm_transcript import defuse_separator_fences
from discordbot.typings.context_budgets import MAX_INLINE_ATTACHMENT_BYTES
from discordbot.cogs.gen_reply.attachment.base import (
    UploadKind,
    AttachmentSource,
    FileUploadRenderer,
    media_semaphore,
    loggable_cache_key,
)
from discordbot.cogs.gen_reply.attachment.loaders import load_image_bytes, load_attachment_bytes

if TYPE_CHECKING:
    from nextcord import Attachment

# Matches Gemini's 48h auto-expiry so Anthropic files clean up server-side without a sweep job.
ANTHROPIC_FILE_TTL_SECONDS = 172_800
# Fallback local cache window when `uploaded.expires_at` is omitted.
ANTHROPIC_FILE_CACHE_TTL = timedelta(hours=47)
_INLINE_TEXT_CACHE_TTL = timedelta(hours=12)

# Anthropic Messages API image allowlist and hard per-image / per-PDF ceilings.
ANTHROPIC_IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
ANTHROPIC_IMAGE_MAX_BYTES = 10 * 1024 * 1024
ANTHROPIC_PDF_MAX_BYTES = 32 * 1024 * 1024


def _anthropic_file_part(file_id: str, filename: str, mime_type: str) -> FormattedInputFileParam:
    """Returns an `input_file` part with LiteLLM's `format` MIME key for Anthropic."""
    return {"type": "input_file", "file_id": file_id, "filename": filename, "format": mime_type}


class AnthropicFileUploader(FileUploadRenderer):
    """Uploads images and PDFs to the Anthropic Files API and inlines UTF-8 text files.

    Uses the deployment's direct Anthropic client so uploaded file IDs belong to the same
    Anthropic workspace as the proxy's Claude deployments. Audio, video, unsupported image MIME
    types, oversized files, and non-UTF-8 binaries are dropped.
    """

    dropped_modalities = frozenset({"video", "audio"})

    anthropic_client: Callable[[], AsyncAnthropic | None] = Field(
        ...,
        description=(
            "Hands back the deployment's direct Anthropic client, or None when no key is "
            "configured. A keyless deployment inlines instead of building this uploader, so a "
            "None here only drops the upload with the missing key named."
        ),
    )
    # Per-instance LRU (max 128) of unreadable non-PDF/non-UTF-8 keys refused by `carries`.
    _unreadable: BoundedLRU[int | str, None] = PrivateAttr(default_factory=BoundedLRU)

    def carries(self, source: AttachmentSource) -> bool:
        """Returns whether `source` is eligible for Anthropic rendering.

        Refuses fonts, Android packages, PDFs past `ANTHROPIC_PDF_MAX_BYTES`, non-PDF files past
        `MAX_INLINE_ATTACHMENT_BYTES`, and keys previously recorded in `_unreadable`. Images are
        checked against `ANTHROPIC_IMAGE_MAX_BYTES` after downscaling.

        Args:
            source (AttachmentSource): Classified attachment metadata to check.
        """
        if source.content_type.startswith("font/") or (
            source.content_type == "application/vnd.android.package-archive"
        ):
            return False
        if source.size is not None:
            if source.content_type == "application/pdf" and source.size > ANTHROPIC_PDF_MAX_BYTES:
                return False
            if (
                source.kind == "file"
                and source.content_type != "application/pdf"
                and source.size > MAX_INLINE_ATTACHMENT_BYTES
            ):
                return False
        return source.cache_key not in self._unreadable

    async def render_image(
        self, source: AttachmentSource, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns `source` uploaded to the Anthropic Files API as an `input_file` part, or `None`.

        Args:
            source (AttachmentSource): Classified image attachment, sticker, path, or URL to upload.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        if allow_dead_cache and self._is_known_dead(cache_key=source.cache_key):
            return None
        wait_started = time.monotonic()
        async with media_semaphore.get():
            logfire.debug(
                "media slot acquired",
                cache_key=loggable_cache_key(cache_key=source.cache_key),
                wait_seconds=time.monotonic() - wait_started,
            )
            loaded = await self._load_source_bytes(
                cache_key=source.cache_key,
                filename=source.filename,
                load_data=lambda: load_image_bytes(source=source.handle),
                allow_dead_cache=allow_dead_cache,
            )
            if loaded is None:
                return None
            if loaded.mime_type not in ANTHROPIC_IMAGE_MIMES:
                logfire.warn(
                    "dropping image with unsupported Anthropic MIME type",
                    filename=source.filename,
                    mime_type=loaded.mime_type,
                    cache_key=loggable_cache_key(cache_key=source.cache_key),
                )
                return None
            if len(loaded.data) > ANTHROPIC_IMAGE_MAX_BYTES:
                logfire.warn(
                    "dropping image exceeding Anthropic size limit",
                    filename=source.filename,
                    mime_type=loaded.mime_type,
                    bytes=len(loaded.data),
                    max_bytes=ANTHROPIC_IMAGE_MAX_BYTES,
                )
                return None
            uploaded = await self._upload_file(
                cache_key=source.cache_key,
                filename=source.filename,
                data=loaded.data,
                content_type=loaded.mime_type,
                kind="image",
            )
        if uploaded is None:
            return None
        return RenderedAttachment(
            part=_anthropic_file_part(
                file_id=uploaded.uri, filename=source.filename, mime_type=loaded.mime_type
            ),
            expires_at=uploaded.expires_at,
        )

    async def render_file(
        self, source: AttachmentSource, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Uploads PDFs to the Anthropic Files API and inlines UTF-8 text files as `input_text`.

        Args:
            source (AttachmentSource): Classified Discord file attachment to render.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        if source.content_type.partition("/")[0] in self.dropped_modalities:
            logfire.warn(
                "dropping video / audio attachment the Anthropic renderer cannot carry",
                filename=source.filename,
                mime_type=source.content_type,
            )
            return None
        if allow_dead_cache and self._is_known_dead(cache_key=source.cache_key):
            return None
        # Only attachments are classified as `kind="file"`; stickers and embeds are images.
        attachment = cast("Attachment", source.handle)
        wait_started = time.monotonic()
        async with media_semaphore.get():
            logfire.debug(
                "media slot acquired",
                cache_key=loggable_cache_key(cache_key=source.cache_key),
                wait_seconds=time.monotonic() - wait_started,
            )
            loaded = await self._load_source_bytes(
                cache_key=source.cache_key,
                filename=source.filename,
                load_data=lambda: load_attachment_bytes(attachment=attachment),
                allow_dead_cache=allow_dead_cache,
            )
            if loaded is None:
                return None
            mime_type = loaded.mime_type or source.content_type
            if mime_type == "application/pdf":
                return await self._render_pdf(source=source, data=loaded.data, mime_type=mime_type)
        return self._inline_text_part(
            filename=source.filename,
            data=loaded.data,
            mime_type=mime_type,
            cache_key=source.cache_key,
        )

    async def _render_pdf(
        self, source: AttachmentSource, data: bytes, mime_type: str
    ) -> RenderedAttachment | None:
        """Uploads a PDF within `ANTHROPIC_PDF_MAX_BYTES` and returns its `input_file` part."""
        if len(data) > ANTHROPIC_PDF_MAX_BYTES:
            logfire.warn(
                "dropping PDF exceeding Anthropic size limit",
                filename=source.filename,
                bytes=len(data),
                max_bytes=ANTHROPIC_PDF_MAX_BYTES,
            )
            return None
        uploaded = await self._upload_file(
            cache_key=source.cache_key,
            filename=source.filename,
            data=data,
            content_type=mime_type,
            kind="file",
        )
        if uploaded is None:
            return None
        return RenderedAttachment(
            part=_anthropic_file_part(
                file_id=uploaded.uri, filename=source.filename, mime_type=mime_type
            ),
            expires_at=uploaded.expires_at,
        )

    def _inline_text_part(
        self, filename: str, data: bytes, mime_type: str, cache_key: int | str
    ) -> RenderedAttachment | None:
        """Inlines a UTF-8 file as `input_text`, recording non-UTF-8 keys in `_unreadable`."""
        if len(data) > MAX_INLINE_ATTACHMENT_BYTES:
            logfire.warn(
                "dropping text attachment exceeding inline size limit",
                filename=filename,
                mime_type=mime_type,
                bytes=len(data),
                max_bytes=MAX_INLINE_ATTACHMENT_BYTES,
            )
            return None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            logfire.warn(
                "dropping non-text, non-PDF attachment the Anthropic renderer cannot carry",
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
        return RenderedAttachment(
            part=text_part, expires_at=datetime.now(tz=UTC) + _INLINE_TEXT_CACHE_TTL
        )

    async def _upload_file(
        self, cache_key: int | str, filename: str, data: bytes, content_type: str, kind: UploadKind
    ) -> UploadedFile | None:
        """Uploads `data` to the GA Anthropic Files API with a 48h TTL, or returns `None`."""
        del cache_key, kind
        started = time.monotonic()
        logfire.debug(
            "anthropic upload start", filename=filename, content_type=content_type, bytes=len(data)
        )
        client = self.anthropic_client()
        if client is None:
            logfire.error(
                "anthropic Files API key missing; dropping attachment", filename=filename
            )
            return None
        try:
            async with asyncio.timeout(delay=ATTACHMENT_UPLOAD_TIMEOUT_SECONDS):
                uploaded = await client.files.upload(
                    file=(filename, io.BytesIO(data), content_type),
                    expires_in_seconds=ANTHROPIC_FILE_TTL_SECONDS,
                )
        except Exception as exc:
            # Broad on purpose: SDK, network and auth failures all degrade the same way here.
            logfire.warn(
                "failed to upload attachment to Anthropic Files API",
                filename=filename,
                content_type=content_type,
                bytes=len(data),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        if not uploaded.id:
            logfire.warn("upload returned no file id; dropping", filename=filename)
            return None
        expires_at = uploaded.expires_at or (datetime.now(tz=UTC) + ANTHROPIC_FILE_CACHE_TTL)
        logfire.debug(
            "anthropic upload done",
            filename=filename,
            file_id=uploaded.id,
            elapsed_seconds=time.monotonic() - started,
        )
        return UploadedFile(uri=uploaded.id, expires_at=expires_at)
