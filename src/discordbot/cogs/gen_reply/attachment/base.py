"""The attachment renderer strategy interface and its shared rendered-part type."""

from typing import TYPE_CHECKING
from datetime import UTC, datetime, timedelta
from collections import OrderedDict

import logfire
from nextcord import Attachment, StickerItem
from pydantic import BaseModel, ConfigDict, PrivateAttr
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.asyncio_locks import LoopLocalSemaphore

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

# Lazily fetches a source's bytes and mime type. Awaited only when an upload is actually needed,
# so a renderer that can adopt an already-uploaded file never re-downloads the source.
type FileBytesLoader = Callable[[], Awaitable[tuple[bytes, str]]]

# A rendered attachment content part. The Gemini answer model reads a Files-API handle
# (input_file with a file URI); non-Gemini answer models cannot resolve that URI, so their
# attachments are inlined per type instead: images as input_image base64, PDFs as input_file
# base64 file_data, and text/code files as input_text.
type RenderedPart = ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam

# A source whose byte fetch fails (typically an expired Discord/Threads CDN url that sits in
# history scrollback) is skipped for this long so it is not re-fetched and re-warned on every
# reply; after the window it is retried once so a transient blip self-heals.
DEAD_SOURCE_TTL = timedelta(minutes=30)
# Bounds concurrent media fetch + Files-API upload work across all in-flight pipelines. Above
# the typical per-message attachment count so a single request stays fully parallel, while two
# concurrent pipelines cannot launch dozens of simultaneous uploads and starve each other (the
# source of the worst observed render tail).
MEDIA_CONCURRENCY = 8

# Module-level rather than per renderer, because there is one renderer per Gemini key now and
# a per-instance semaphore would multiply the cap by the key count — restoring exactly the
# starvation the number above was measured against. Loop-local because a module-level
# `asyncio.Semaphore` binds to the first loop that waits on it and every test runs a fresh one
# (`utils/asyncio_locks.py` has the mechanism).
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
    ) -> tuple[RenderedPart, datetime] | None:
        """Renders an image source (attachment, sticker, or URL) to a content part."""
        raise NotImplementedError

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> tuple[RenderedPart, datetime] | None:
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
    ) -> tuple[bytes, str] | None:
        """Fetches one source's bytes and mime type, or None when the fetch failed.

        Call it INSIDE the media slot: the fetch is half of what that slot bounds, and holding
        the slot across the download and the upload alike is what stops concurrent pipelines
        buffering dozens of files while they queue for an upload.

        The except is broad because `load_data` is caller-supplied and spans a CDN fetch plus a
        PIL decode; any failure must degrade to dropping this one attachment rather than blanking
        the message it belongs to. A history render (`allow_dead_cache`) additionally marks the
        source dead, so an expired CDN url is not re-fetched on every later reply.

        The unpack happens INSIDE that guard rather than at the caller, so a loader that answers
        the wrong shape is the same kind of failure as one that raises. Returning the pair
        unpacked would let a `None` short-circuit with no warning and no dead-source marking, and
        would let a wrong-arity tuple raise into `input.py`'s attachment `gather`, which has no
        `return_exceptions` and would lose the whole message's attachments rather than this one.
        """
        try:
            data, content_type = await load_data()
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
        return data, content_type
