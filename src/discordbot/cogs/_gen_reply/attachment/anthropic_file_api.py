"""Anthropic Files API attachment renderer for Claude answer models.

Uploads attachment bytes through the Anthropic SDK (a direct side-channel, not the LiteLLM
proxy) and references the returned file id in Responses API content parts. Kept disabled in
`select.py` until the proxy's reference-translation path for Claude is verified; today
non-Gemini answer models inline instead (`InlineRenderer`).

Simpler than the Gemini uploader: Anthropic files are usable the moment `beta.files.upload`
returns (no PROCESSING/ACTIVE poll) and persist until deleted (no provider expiry), so there
is no pending-upload re-poll machinery and the render cache uses a fixed synthetic TTL. That
persistence cuts both ways: the synthetic TTL only evicts the local render cache, never the
remote file, and Anthropic files never auto-expire (unlike Gemini's ~48h), so enabling this
renderer needs a deletion / periodic-sweep strategy or active channels accumulate uploads
toward the org storage quota. Evicting-then-deleting is not enough on its own: a file can
still be referenced in scrollback after its cache entry is gone, so the cleanup policy is an
enable-time design decision, not something this disabled scaffold settles.
"""

import io
from typing import TYPE_CHECKING
import asyncio
from datetime import UTC, datetime, timedelta
from functools import cached_property
from collections import OrderedDict

import logfire
from nextcord import Attachment, StickerItem
from pydantic import Field, PrivateAttr, SkipValidation
from anthropic import AsyncAnthropic
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.cogs._gen_reply.attachment.base import RenderedPart, AttachmentRenderer
from discordbot.cogs._gen_reply.attachment.loaders import (
    attachment_mime,
    load_image_bytes,
    load_attachment_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

type FileBytesLoader = Callable[[], Awaitable[tuple[bytes, str]]]

DEAD_SOURCE_TTL = timedelta(minutes=30)
MEDIA_CONCURRENCY = 8
# Anthropic Files API entries persist until explicitly deleted (no provider expiry), so the
# per-message render cache reuses an upload for a fixed window as a cheap CDN-rehost safety net.
ANTHROPIC_FILE_CACHE_TTL = timedelta(hours=12)


class AnthropicFileUploader(AttachmentRenderer):
    """Uploads attachments to the Anthropic Files API and references them by file id.

    Attributes:
        config: Runtime LLM config supplying the Anthropic Files API key for the upload client.
    """

    config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Runtime LLM config supplying the Anthropic Files API upload client key.",
    )
    _dead_sources: OrderedDict[int | str, datetime] = PrivateAttr(default_factory=OrderedDict)
    _media_semaphore: SkipValidation[asyncio.Semaphore] = PrivateAttr(
        default_factory=lambda: asyncio.Semaphore(MEDIA_CONCURRENCY)
    )

    @cached_property
    def anthropic_client(self) -> AsyncAnthropic:
        """The Anthropic client for direct Files API uploads, built lazily on first use.

        Built inside this module rather than via a shared `utils/llm.py` factory while the
        renderer is still disabled in `select.py`: that keeps the `anthropic` import out of the
        live request-path module graph (`utils/llm.py` is imported there) until the Claude path
        is wired on. A missing key does not fail construction; it surfaces at the upload call.

        Returns:
            An Anthropic client reused across uploads.
        """
        return AsyncAnthropic(api_key=self.config.anthropic_api_key)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> tuple[RenderedPart, datetime] | None:
        if isinstance(source, str):
            source_name = "image.jpg"
        else:
            source_name = (
                getattr(source, "filename", None) or f"{getattr(source, 'name', 'sticker')}.png"
            )
        uploaded = await self._resolve_file_upload(
            cache_key=cache_key,
            filename=source_name,
            load_data=lambda: load_image_bytes(source=source),
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        file_id, expires_at = uploaded
        part = ResponseInputImageParam(type="input_image", file_id=file_id, detail="auto")
        return part, expires_at

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> tuple[RenderedPart, datetime] | None:
        mime_type = attachment_mime(attachment=attachment)
        if not mime_type:
            logfire.warn(
                f"Skipping attachment with unknown MIME type: {attachment.filename} ({attachment.url})"
            )
            return None
        uploaded = await self._resolve_file_upload(
            cache_key=cache_key,
            filename=attachment.filename,
            load_data=lambda: load_attachment_bytes(attachment=attachment),
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        file_id, expires_at = uploaded
        part = ResponseInputFileParam(
            type="input_file", file_id=file_id, filename=attachment.filename
        )
        return part, expires_at

    def _is_known_dead(self, cache_key: int | str) -> bool:
        """Whether a source's fetch failed recently enough to skip re-fetching it."""
        dead_at = self._dead_sources.get(cache_key)
        if dead_at is None:
            return False
        if datetime.now(tz=UTC) - dead_at < DEAD_SOURCE_TTL:
            self._dead_sources.move_to_end(cache_key)
            return True
        self._dead_sources.pop(cache_key, None)
        return False

    def _mark_dead(self, cache_key: int | str) -> None:
        """Records a source's fetch failure so history scrollback stays cheap."""
        self._dead_sources[cache_key] = datetime.now(tz=UTC)
        self._dead_sources.move_to_end(cache_key)
        if len(self._dead_sources) > 128:
            self._dead_sources.popitem(last=False)

    async def _resolve_file_upload(
        self,
        cache_key: int | str,
        filename: str,
        load_data: "FileBytesLoader",
        allow_dead_cache: bool = False,
    ) -> tuple[str, datetime] | None:
        """Returns an uploaded Anthropic file id and its synthetic cache expiry."""
        if allow_dead_cache and self._is_known_dead(cache_key=cache_key):
            return None
        async with self._media_semaphore:
            try:
                data, content_type = await load_data()
            except Exception:
                logfire.warn(f"Failed to load attachment bytes for upload: {filename}")
                if allow_dead_cache:
                    self._mark_dead(cache_key=cache_key)
                return None
            return await self._upload_file(filename=filename, data=data, content_type=content_type)

    async def _upload_file(
        self, filename: str, data: bytes, content_type: str
    ) -> tuple[str, datetime] | None:
        """Uploads bytes to the Anthropic Files API and returns `(file_id, expires_at)`.

        The SDK sets the required `files-api-2025-04-14` beta header for `beta.files`
        automatically. Anthropic files have no provider expiry, so the cache window is a
        fixed synthetic TTL.
        """
        try:
            uploaded = await self.anthropic_client.beta.files.upload(
                file=(filename, io.BytesIO(data), content_type)
            )
        except Exception:
            logfire.warn(f"Failed to upload attachment to Anthropic Files API: {filename}")
            return None
        if not uploaded.id:
            logfire.warn(f"Anthropic file upload returned no file id; dropping: {filename}")
            return None
        return uploaded.id, datetime.now(tz=UTC) + ANTHROPIC_FILE_CACHE_TTL
