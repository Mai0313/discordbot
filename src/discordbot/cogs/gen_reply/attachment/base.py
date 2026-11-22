"""The attachment renderer strategy interface, and the Files API upload renderer built on it."""

import time
from typing import TYPE_CHECKING, Literal, ClassVar
from datetime import UTC, datetime, timedelta
from collections import OrderedDict

import logfire
from nextcord import Attachment, StickerItem
from pydantic import BaseModel, ConfigDict, PrivateAttr
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.media import LoadedMedia, UploadedFile, RenderedAttachment
from discordbot.utils.asyncio_locks import LoopLocalSemaphore
from discordbot.cogs.gen_reply.attachment.loaders import (
    attachment_mime,
    load_image_bytes,
    load_attachment_bytes,
    resolve_source_filename,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

# Lazy loader so renderers adopting an existing upload avoid re-downloading bytes.
type FileBytesLoader = Callable[[], Awaitable[LoadedMedia]]

# Distinguishes image vs generic file uploads for providers that handle them differently.
type UploadKind = Literal["image", "file"]

# Window to skip failing scrollback sources before retrying once.
DEAD_SOURCE_TTL = timedelta(minutes=30)
# Process-wide cap on concurrent media fetch + upload tasks across in-flight turns.
MEDIA_CONCURRENCY = 8

# Module-level so the concurrency cap holds across all renderer instances.
media_semaphore = LoopLocalSemaphore(capacity_provider=lambda: MEDIA_CONCURRENCY)


def loggable_cache_key(cache_key: int | str) -> int | str:
    """Returns `cache_key` with any URL query string stripped so signed CDN tokens are not logged.

    Args:
        cache_key (int | str): Attachment/sticker ID or embed-image source URL.
    """
    if isinstance(cache_key, str):
        return cache_key.split("?", 1)[0]
    return cache_key


class AttachmentRenderer(BaseModel):
    """Strategy that turns one Discord attachment source into a Responses API content part.

    `render_image` and `render_file` return a `RenderedAttachment` with its cache expiry, or
    `None` when the source is unsupported or fails. `cache_key` and `allow_dead_cache` drive
    the per-instance dead-source cache.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Modalities dropped regardless of answer-model support; subtracted by the modality gate.
    dropped_modalities: ClassVar[frozenset[str]] = frozenset()

    # Per-instance LRU (max 128) of `cache_key -> first_failure_at` for dead-source skipping.
    _dead_sources: OrderedDict[int | str, datetime] = PrivateAttr(default_factory=OrderedDict)

    def carries(self, content_type: str, cache_key: int | str, size: int | None) -> bool:
        """Returns whether this renderer can carry the unfetched source.

        Checked alongside `dropped_modalities` before downloading so the route marker, history
        media budget, and render cache skip sources the answer cannot use.

        Args:
            content_type (str): Resolved lowercase MIME type of the source.
            cache_key (int | str): Per-source identifier.
            size (int | None): Byte size reported by Discord, or `None` when unknown.
        """
        del content_type, cache_key, size
        return True

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> RenderedAttachment | None:
        """Returns an image source rendered to a Responses API content part, or `None` if dropped.

        Args:
            source (Attachment | StickerItem | str): Image attachment, sticker, or URL to render.
            cache_key (int | str): Per-source identifier for upload and dead-source caching.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        raise NotImplementedError

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns a non-image attachment rendered to a Responses API content part, or `None`.

        Args:
            attachment (Attachment): Discord file attachment to render.
            cache_key (int | str): Per-source identifier for upload and dead-source caching.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        raise NotImplementedError

    def _is_known_dead(self, cache_key: int | str) -> bool:
        """Returns whether `cache_key` failed a fetch within `DEAD_SOURCE_TTL`, evicting expired entries."""
        dead_at = self._dead_sources.get(cache_key)
        if dead_at is None:
            return False
        if datetime.now(tz=UTC) - dead_at < DEAD_SOURCE_TTL:
            self._dead_sources.move_to_end(cache_key)
            return True
        self._dead_sources.pop(cache_key, None)
        return False

    def _mark_dead(self, cache_key: int | str) -> None:
        """Records a source's fetch failure so it is skipped for DEAD_SOURCE_TTL."""
        self._dead_sources[cache_key] = datetime.now(tz=UTC)
        self._dead_sources.move_to_end(cache_key)
        if len(self._dead_sources) > 128:
            self._dead_sources.popitem(last=False)

    async def _load_source_bytes(
        self,
        cache_key: int | str,
        filename: str,
        load_data: "FileBytesLoader",
        allow_dead_cache: bool,
    ) -> LoadedMedia | None:
        """Fetches one source's bytes and MIME type, or returns `None` on failure.

        Swallows loader errors so one broken attachment does not fail sibling renders gathered
        without `return_exceptions`, and marks `cache_key` dead when `allow_dead_cache` is true.
        """
        try:
            loaded = await load_data()
        except Exception as exc:
            logfire.warn(
                "failed to load attachment bytes; dropping it",
                filename=filename,
                cache_key=loggable_cache_key(cache_key=cache_key),
                allow_dead_cache=allow_dead_cache,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            if allow_dead_cache:
                self._mark_dead(cache_key=cache_key)
            return None
        return loaded


class FileUploadRenderer(AttachmentRenderer):
    """Renderer that uploads each source to a provider's Files API and references its handle.

    Subclasses implement `_upload_file`, and may override `render_image` or `_resolve_file_upload`
    to customize part shape or short-circuit before downloading bytes.
    """

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> RenderedAttachment | None:
        """Returns `source` uploaded to the provider's Files API as an `input_image` part, or `None`.

        Args:
            source (Attachment | StickerItem | str): Image attachment, sticker, or URL to upload.
            cache_key (int | str): Per-source identifier for dead-source caching.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        source_name = resolve_source_filename(source=source, url_fallback="image.jpg")
        uploaded = await self._resolve_file_upload(
            cache_key=cache_key,
            filename=source_name,
            load_data=lambda: load_image_bytes(source=source),
            kind="image",
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        part = ResponseInputImageParam(type="input_image", file_id=uploaded.uri, detail="auto")
        return RenderedAttachment(part=part, expires_at=uploaded.expires_at)

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns `attachment` uploaded to the provider's Files API as an `input_file` part, or `None`.

        Args:
            attachment (Attachment): Discord file attachment to upload.
            cache_key (int | str): Per-source identifier for upload and dead-source caching.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        mime_type = attachment_mime(attachment=attachment)
        if not mime_type:
            logfire.warn(
                "skipping attachment with unknown MIME type",
                filename=attachment.filename,
                url=loggable_cache_key(cache_key=attachment.url),
            )
            return None
        uploaded = await self._resolve_file_upload(
            cache_key=cache_key,
            filename=attachment.filename,
            load_data=lambda: load_attachment_bytes(attachment=attachment),
            kind="file",
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        part = ResponseInputFileParam(
            type="input_file", file_id=uploaded.uri, filename=attachment.filename
        )
        return RenderedAttachment(part=part, expires_at=uploaded.expires_at)

    async def _resolve_file_upload(
        self,
        cache_key: int | str,
        filename: str,
        load_data: "FileBytesLoader",
        kind: UploadKind,
        allow_dead_cache: bool = False,
    ) -> UploadedFile | None:
        """Uploads one source under a single media slot spanning download and upload, or `None`."""
        # Only history scrollback opts in so transient errors never poison a fresh attachment.
        if allow_dead_cache and self._is_known_dead(cache_key=cache_key):
            return None
        wait_started = time.monotonic()
        async with media_semaphore.get():
            logfire.debug(
                "media slot acquired",
                cache_key=loggable_cache_key(cache_key=cache_key),
                wait_seconds=time.monotonic() - wait_started,
            )
            loaded = await self._load_source_bytes(
                cache_key=cache_key,
                filename=filename,
                load_data=load_data,
                allow_dead_cache=allow_dead_cache,
            )
            if loaded is None:
                return None
            return await self._upload_file(
                cache_key=cache_key,
                filename=filename,
                data=loaded.data,
                content_type=loaded.mime_type,
                kind=kind,
            )

    async def _upload_file(
        self, cache_key: int | str, filename: str, data: bytes, content_type: str, kind: UploadKind
    ) -> UploadedFile | None:
        """Uploads one source's bytes and returns the handle, or `None` on failure."""
        raise NotImplementedError
