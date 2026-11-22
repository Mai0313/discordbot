"""The attachment renderer strategy interface, and the Files API upload renderer built on it."""

import time
from typing import TYPE_CHECKING, Literal, ClassVar, cast
from datetime import UTC, datetime, timedelta

import logfire
from nextcord import Attachment, StickerItem
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.lru import BoundedLRU
from discordbot.typings.media import LoadedMedia, UploadedFile, RenderedAttachment
from discordbot.utils.asyncio_locks import LoopLocalSemaphore
from discordbot.cogs.gen_reply.attachment.loaders import load_image_bytes, load_attachment_bytes

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


class AttachmentSource(BaseModel):
    """One renderable attachment source classified from message metadata.

    Collected once per message and shared by every render of it. Carries only metadata (no
    bytes, no network) so it is safe to build on the route critical path.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    handle: SkipValidation[Attachment | StickerItem | str] = Field(
        ..., description="The attachment, sticker, or image URL the loaders consume."
    )
    kind: Literal["image", "file"] = Field(
        ..., description="Whether the source renders as an image or a generic file."
    )
    is_sticker: bool = Field(
        default=False,
        description=(
            "Whether the source is a Discord sticker. Separate from `kind`, which stays the "
            "render path a sticker shares with every other image; this is what the text-only "
            "marker names it."
        ),
    )
    content_type: str = Field(
        ...,
        description=(
            "Resolved MIME type; empty when the sent type is blank or parameter-only, or when "
            "none is sent and the filename is unguessable."
        ),
    )
    cache_key: int | str = Field(
        ...,
        description="Stable identity (attachment/sticker id or chosen embed URL) for the cache.",
    )
    filename: str = Field(
        ...,
        description=(
            "Resolved upload filename with its extension (attachment filename, `<sticker>.png`, "
            "or a fallback for embed URLs)."
        ),
    )
    size: int | None = Field(
        default=None,
        description=(
            "Bytes Discord reports for an attachment; None for a sticker or an embed image, "
            "which report none."
        ),
    )

    @property
    def marker_kind(self) -> str:
        """The name an `[attachment: ...]` marker gives this source: a sticker as such, else its kind."""
        return "sticker" if self.is_sticker else self.kind


class AttachmentRenderer(BaseModel):
    """Strategy that turns one classified `AttachmentSource` into a Responses API content part.

    `render_image` and `render_file` return a `RenderedAttachment` with its cache expiry, or
    `None` when the source is unsupported or fails. `source.cache_key` and `allow_dead_cache`
    drive the per-instance dead-source cache.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Modalities dropped regardless of answer-model support; subtracted by the modality gate.
    dropped_modalities: ClassVar[frozenset[str]] = frozenset()

    # Per-instance LRU (max 128) of `cache_key -> first_failure_at` for dead-source skipping.
    _dead_sources: BoundedLRU[int | str, datetime] = PrivateAttr(default_factory=BoundedLRU)

    def carries(self, source: AttachmentSource) -> bool:
        """Returns whether this renderer can carry the unfetched `source`.

        Checked alongside `dropped_modalities` before downloading so the route marker, history
        media budget, and render cache skip sources the answer cannot use.

        Args:
            source (AttachmentSource): Classified attachment metadata to check.
        """
        del source
        return True

    async def render_image(
        self, source: AttachmentSource, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns an image source rendered to a Responses API content part, or `None` if dropped.

        Args:
            source (AttachmentSource): Classified image attachment, sticker, or URL to render.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        raise NotImplementedError

    async def render_file(
        self, source: AttachmentSource, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns a non-image attachment rendered to a Responses API content part, or `None`.

        Args:
            source (AttachmentSource): Classified Discord file attachment to render.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        raise NotImplementedError

    def _is_known_dead(self, cache_key: int | str) -> bool:
        """Returns whether `cache_key` failed a fetch within `DEAD_SOURCE_TTL`, evicting expired entries."""
        dead_at = self._dead_sources.peek(key=cache_key)
        if dead_at is None:
            return False
        if datetime.now(tz=UTC) - dead_at < DEAD_SOURCE_TTL:
            self._dead_sources.touch(key=cache_key)
            return True
        self._dead_sources.pop(key=cache_key)
        return False

    def _mark_dead(self, cache_key: int | str) -> None:
        """Records a source's fetch failure so it is skipped for DEAD_SOURCE_TTL."""
        self._dead_sources.set(key=cache_key, value=datetime.now(tz=UTC))

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
        self, source: AttachmentSource, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns `source` uploaded to the provider's Files API as an `input_image` part, or `None`.

        Args:
            source (AttachmentSource): Classified image attachment, sticker, or URL to upload.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        uploaded = await self._resolve_file_upload(
            cache_key=source.cache_key,
            filename=source.filename,
            load_data=lambda: load_image_bytes(source=source.handle),
            kind="image",
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        part = ResponseInputImageParam(type="input_image", file_id=uploaded.uri, detail="auto")
        return RenderedAttachment(part=part, expires_at=uploaded.expires_at)

    async def render_file(
        self, source: AttachmentSource, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns `source` uploaded to the provider's Files API as an `input_file` part, or `None`.

        Args:
            source (AttachmentSource): Classified Discord file attachment to upload.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        # Only attachments are classified as `kind="file"`; stickers and embeds are images.
        attachment = cast("Attachment", source.handle)
        uploaded = await self._resolve_file_upload(
            cache_key=source.cache_key,
            filename=source.filename,
            load_data=lambda: load_attachment_bytes(attachment=attachment),
            kind="file",
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        part = ResponseInputFileParam(
            type="input_file", file_id=uploaded.uri, filename=source.filename
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
