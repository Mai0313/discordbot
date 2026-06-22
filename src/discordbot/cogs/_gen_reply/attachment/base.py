"""The attachment renderer strategy interface and its shared rendered-part type."""

import asyncio
from datetime import UTC, datetime, timedelta
from collections import OrderedDict

from nextcord import Attachment, StickerItem
from pydantic import BaseModel, ConfigDict, PrivateAttr, SkipValidation
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

# A rendered attachment content part. The Gemini answer model reads a Files-API handle
# (input_file with a file URI); non-Gemini answer models cannot resolve that URI, so their
# attachments are inlined per type instead: images as input_image base64, PDFs as input_file
# base64 file_data, and text/code files as input_text.
type RenderedPart = ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam

# A source whose byte fetch fails (typically an expired Discord/Threads CDN url that sits in
# history scrollback) is skipped for this long so it is not re-fetched and re-warned on every
# reply; after the window it is retried once so a transient blip self-heals.
DEAD_SOURCE_TTL = timedelta(minutes=30)
# Bounds concurrent media fetch + Files-API upload work across all in-flight pipelines (the
# input builder is a shared singleton). Above the typical per-message attachment count so a
# single request stays fully parallel, while two concurrent pipelines cannot launch dozens of
# simultaneous uploads and starve each other (the source of the worst observed render tail).
MEDIA_CONCURRENCY = 8


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
    `cache_key` and `allow_dead_cache` drive the shared dead-source cache below (and the Gemini
    uploader's per-class re-poll cache); a stateless renderer inherits the cache attributes for
    interface parity but never uses them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Sources whose byte fetch failed, keyed by cache_key -> first-failure time. Shared by every
    # uploading renderer so the Files-API uploaders cannot drift; a hit within DEAD_SOURCE_TTL
    # skips the fetch fast, past it the entry is dropped and the source retried once. Bounded at
    # 128 entries. A stateless renderer (InlineRenderer) inherits but never touches it.
    _dead_sources: OrderedDict[int | str, datetime] = PrivateAttr(default_factory=OrderedDict)
    # Caps concurrent media fetch + upload work; see MEDIA_CONCURRENCY. Created in-loop on first
    # access (during message handling), so it binds to the running event loop.
    _media_semaphore: SkipValidation[asyncio.Semaphore] = PrivateAttr(
        default_factory=lambda: asyncio.Semaphore(MEDIA_CONCURRENCY)
    )

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
