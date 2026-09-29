"""The attachment renderer strategy interface, and the Files API upload renderer built on it."""

from typing import TYPE_CHECKING, Literal
from datetime import UTC, datetime, timedelta
from collections import OrderedDict

import logfire
from nextcord import Attachment, StickerItem
from pydantic import BaseModel, ConfigDict, PrivateAttr
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.typings.media import LoadedMedia, UploadedFile, RenderedAttachment
from discordbot.utils.asyncio_locks import LoopLocalSemaphore
from discordbot.cogs.gen_reply.attachment.loaders import attachment_mime, load_attachment_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

# Lazily fetches a source's bytes and mime type. Awaited only when an upload is actually needed,
# so a renderer that can adopt an already-uploaded file never re-downloads the source.
type FileBytesLoader = Callable[[], Awaitable[LoadedMedia]]

# Which render asked for an upload, for a provider that declares an image upload differently
# from any other file.
type UploadKind = Literal["image", "file"]

# A source whose byte fetch fails (typically an expired Discord/Threads CDN url that sits in
# history scrollback) is skipped for this long so it is not re-fetched and re-warned on every
# reply; after the window it is retried once so a transient blip self-heals.
DEAD_SOURCE_TTL = timedelta(minutes=30)
# Bounds concurrent media fetch + Files-API upload work across all in-flight pipelines. Above
# the typical per-message attachment count so a single request stays fully parallel, while two
# concurrent pipelines cannot launch dozens of simultaneous uploads and starve each other (the
# source of the worst observed render tail).
MEDIA_CONCURRENCY = 8

# Module-level rather than per renderer, so the cap holds however many renderers exist: a
# per-instance semaphore would multiply it by their count, restoring exactly the starvation the
# number above was measured against. Loop-local because a module-level `asyncio.Semaphore` binds
# to the first loop that waits on it and every test runs a fresh one (`utils/asyncio_locks.py`
# has the mechanism).
media_semaphore = LoopLocalSemaphore(capacity_provider=lambda: MEDIA_CONCURRENCY)


def loggable_cache_key(cache_key: int | str) -> int | str:
    """A log-safe form of an attachment cache key.

    Attachment / sticker keys are ids (safe to log). An embed-image key is its source URL,
    which can carry a signed CDN token in the query string; drop the query so a log keeps a
    stable, correlatable identifier without leaking the token.
    """
    if isinstance(cache_key, str):
        return cache_key.split("?", 1)[0]
    return cache_key


class AttachmentRenderer(BaseModel):
    """Strategy that turns one Discord attachment source into a Responses API content part.

    Each implementation owns one way to make an attachment readable by the answer model
    (Gemini Files-API upload, or per-type inline base64), so the answer model's provider is
    swapped by injecting a different renderer into `MessageInputBuilder`, not by branching
    inside it. Both methods return the rendered part plus the cache expiry the per-message
    render cache reuses it until, or None when the source is dropped (unsupported / failed).
    `cache_key` and `allow_dead_cache` drive the dead-source cache below (and the Gemini
    uploader's own re-poll cache); a stateless renderer inherits the cache attributes for
    interface parity but never uses them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Sources whose byte fetch failed, keyed by cache_key -> first-failure time. Held here rather
    # than on each uploader so the Files-API uploaders cannot drift (the dict itself is per
    # instance); a hit within DEAD_SOURCE_TTL skips the fetch fast, past it the entry is dropped
    # and the source retried once. Bounded at 128 entries. A stateless renderer (InlineRenderer)
    # inherits but never touches it.
    _dead_sources: OrderedDict[int | str, datetime] = PrivateAttr(default_factory=OrderedDict)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> RenderedAttachment | None:
        """Renders an image source (attachment, sticker, or URL) to a content part."""
        raise NotImplementedError

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Renders a non-image file attachment to a content part."""
        raise NotImplementedError

    def _is_known_dead(self, cache_key: int | str) -> bool:
        """Whether a source's fetch failed recently enough to skip re-fetching it.

        Past DEAD_SOURCE_TTL the marker is dropped so the source is retried once, letting a
        transient blip self-heal while an expired CDN url stays cheap.
        """
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
        *,
        cache_key: int | str,
        filename: str,
        load_data: "FileBytesLoader",
        allow_dead_cache: bool,
    ) -> LoadedMedia | None:
        """Fetches one source's bytes and mime type, or None when the fetch failed.

        Call it INSIDE the media slot: the fetch is half of what that slot bounds, and holding
        the slot across the download and the upload alike is what stops concurrent pipelines
        buffering dozens of files while they queue for an upload.

        The except is broad because `load_data` is caller-supplied and spans a CDN fetch plus a
        PIL decode; any failure must degrade to dropping this one attachment rather than blanking
        the message it belongs to. A history render (`allow_dead_cache`) additionally marks the
        source dead, so an expired CDN url is not re-fetched on every later reply.

        The await happens INSIDE that guard rather than at the caller, so a loader that fails is
        this one attachment's problem: `input.py` gathers the renders without `return_exceptions`,
        so an escaping error would lose the whole message's attachments rather than this one.
        """
        try:
            loaded = await load_data()
        except Exception as exc:
            logfire.warn(
                "failed to load attachment bytes for upload",
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
    """A renderer that uploads each source to a provider's Files API and references its handle.

    A provider supplies `_upload_file`, plus its own `render_image`, since how an uploaded image
    is referenced differs per provider. The file render and the upload resolution are shared: a
    history source known dead is skipped, and one media slot spans the download and the upload.
    A provider with more to decide before uploading overrides `_resolve_file_upload` whole.
    """

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        mime_type = attachment_mime(attachment=attachment)
        if not mime_type:
            logfire.warn(
                "skipping attachment with unknown MIME type",
                filename=attachment.filename,
                url=attachment.url,
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
        """Returns an uploaded file's handle and expiry, or None when the source is dropped.

        One media slot spans the whole download plus upload, so concurrent pipelines cannot
        launch dozens of CDN downloads at once and buffer all their bytes while they wait for
        an upload.
        """
        if allow_dead_cache and self._is_known_dead(cache_key=cache_key):
            return None
        async with media_semaphore.get():
            loaded = await self._load_source_bytes(
                cache_key=cache_key,
                filename=filename,
                load_data=load_data,
                allow_dead_cache=allow_dead_cache,
            )
            if loaded is None:
                return None
            return await self._upload_file(
                filename=filename, data=loaded.data, content_type=loaded.mime_type, kind=kind
            )

    async def _upload_file(
        self, filename: str, data: bytes, content_type: str, kind: UploadKind
    ) -> UploadedFile | None:
        """Uploads one source's bytes, returning the handle or None when the upload failed."""
        raise NotImplementedError
