"""The attachment renderer strategy interface, its rendered-part type, and their shared state.

`MessageInputBuilder` (`input.py`) collects a message's attachment sources and hands each one to
an `AttachmentRenderer`; this module is that seam. It owns the `RenderedPart` union every
renderer must produce, the two constants they share (`DEAD_SOURCE_TTL`, `MEDIA_CONCURRENCY`), the
base class carrying the dead-source cache and the media semaphore, and `loggable_cache_key`,
which strips a signed CDN token out of a key before it reaches a log.

It sits here rather than in `input.py` because the answer model's provider decides how an
attachment is made readable, not how it is found: the builder stays one code path and `select.py`
injects the matching implementation (`gemini_file_api.py` uploads to the Files API and references
the uri, `inline.py` embeds base64 / text, with the OpenAI / Anthropic / Grok uploaders
scaffolded beside them). It sits here rather than in each renderer because the skip-a-dead-source
window and the concurrency bound have to be one behavior across all of them, and because the
renderer instance is process-wide (the cog's `input_builder` is a `cached_property`), so those
two caches are what keeps concurrent pipelines from starving each other on media I/O.
"""

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

    Args:
        cache_key (int | str): The source's cache key, an attachment / sticker id or an
            embed-image URL.

    Returns:
        The id unchanged, or the URL with its query string removed.
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
    uploader's own pending-upload re-poll cache); a stateless renderer inherits the cache
    attributes for interface parity but never uses them.
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
        """Renders an image source (attachment, sticker, or URL) to a content part.

        An implementation degrades to None instead of raising: `MessageInputBuilder` gathers
        every source's render without `return_exceptions`, so one escaping error would blank a
        whole message's attachments rather than cost the one part.

        Args:
            source (Attachment | StickerItem | str): The image to render, as a Discord
                attachment, a sticker, or a bare image URL taken from an embed.
            cache_key (int | str): Stable identity of the source (attachment / sticker id, or
                the embed URL), keying the dead-source and pending-upload caches.
            allow_dead_cache (bool): Whether this render may skip a source that failed
                recently, and record a fresh failure. Opt-in for history scrollback only, where
                an expired CDN url re-fails every turn; a current or reference render always
                retries so one transient failure is not poisoned for the next reply.

        Returns:
            The rendered part and the time its handle stops being reusable, or None when the
            source was dropped (unsupported type, failed fetch, or an upload still processing).
        """
        raise NotImplementedError

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> tuple[RenderedPart, datetime] | None:
        """Renders a non-image file attachment to a content part.

        Same degrade-to-None contract as `render_image`. Only a real Discord attachment reaches
        here: the builder classifies stickers and embed images as image sources.

        Args:
            attachment (Attachment): The non-image file to render.
            cache_key (int | str): Stable identity of the source (its attachment id), keying
                the dead-source and pending-upload caches.
            allow_dead_cache (bool): Whether this render may skip a source that failed
                recently, and record a fresh failure. Opt-in for history scrollback only.

        Returns:
            The rendered part and the time its handle stops being reusable, or None when the
            attachment was dropped (unknown or unsupported MIME type, failed download, or an
            upload still processing).
        """
        raise NotImplementedError

    def _is_known_dead(self, cache_key: int | str) -> bool:
        """Whether a source's fetch failed recently enough to skip re-fetching it.

        Past DEAD_SOURCE_TTL the marker is dropped so the source is retried once, letting a
        transient blip self-heal while an expired CDN url stays cheap. A hit also refreshes the
        entry's LRU position, so a source referenced every reply is not evicted by newer ones.

        Args:
            cache_key (int | str): Stable identity of the source to check.

        Returns:
            True while the source's last failure is inside DEAD_SOURCE_TTL, False otherwise.
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
        """Records a source's fetch failure so it is skipped for DEAD_SOURCE_TTL.

        The oldest entry is evicted past 128, so a long-lived process cannot grow the cache
        without bound; re-marking a source overwrites its timestamp and restarts the window.

        Args:
            cache_key (int | str): Stable identity of the source whose fetch failed.
        """
        self._dead_sources[cache_key] = datetime.now(tz=UTC)
        self._dead_sources.move_to_end(cache_key)
        if len(self._dead_sources) > 128:
            self._dead_sources.popitem(last=False)
