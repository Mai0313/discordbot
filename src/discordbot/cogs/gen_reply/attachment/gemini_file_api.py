"""Gemini Files API attachment renderer: direct-SDK upload, activation poll, re-poll cache.

Owns the mechanical side-channel that turns attachment bytes into an ACTIVE Gemini file URI
referenced as an `input_file` part: the direct-SDK upload, the activation poll, and the re-poll
that adopts a slow upload on a later reference instead of paying for it twice. Kept separate
from `input.py` so the upload state machine does not tangle with source-to-part rendering.

`build_attachment_handler` (`select.py`) picks this renderer whenever the answer model is a
Gemini one, because only Gemini resolves a Files uri; every other provider inlines instead
(`inline.py`). The upload goes direct to Google rather than through the LiteLLM proxy, and so
forgoes proxy-side cost tracking, because only the direct SDK exposes the `state` this module
has to poll before the uri is referenced.

Two contracts hold across everything below. Only an ACTIVE uri is ever handed back, since the
answer request has no per-attachment retry and a not-yet-ready uri 400s the whole reply. And
every failure (missing key, dead CDN url, rejected upload, a file that never activates)
degrades to None, dropping that one attachment: `input.py` gathers the renders without
`return_exceptions`, so an escaping error would blank the entire message instead.
"""

import io
import time
from typing import TYPE_CHECKING
import asyncio
from datetime import UTC, datetime, timedelta
from functools import cached_property
from collections import OrderedDict

from google import genai
import logfire
from nextcord import Attachment, StickerItem
from pydantic import Field, BaseModel, PrivateAttr
from google.genai.types import FileState
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.gen_reply.attachment.base import (
    RenderedPart,
    AttachmentRenderer,
    loggable_cache_key,
)
from discordbot.cogs.gen_reply.attachment.loaders import (
    attachment_mime,
    load_image_bytes,
    load_attachment_bytes,
    resolve_source_filename,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

# Lazily fetches a source's bytes and mime type. Awaited only when a fresh Gemini upload is
# needed, so adopting an already-uploaded pending file never re-downloads the source.
type FileBytesLoader = Callable[[], Awaitable[tuple[bytes, str]]]


class PendingUpload(BaseModel):
    """A Gemini Files upload still PROCESSING when the activation poll bound elapsed.

    Cached per attachment source so a slow upload (typically large video/media that
    keeps cooking server-side past the bound) is re-polled on the next reference to
    that source instead of re-uploaded from scratch. The answer never references a
    pending uri; it is adopted only once a later `files.get` reports ACTIVE.

    Attributes:
        name: The Gemini file resource name (`files/<id>`) used to re-poll its state.
        uri: The full file uri the answer references once the file becomes ACTIVE.
        expires_at: Provider-reported expiry; a pending entry past it is discarded.
    """

    name: str = Field(..., description="The Gemini file resource name used to re-poll its state.")
    uri: str = Field(
        ..., description="The full file uri the answer references once the file is ACTIVE."
    )
    expires_at: datetime = Field(
        ..., description="Provider-reported expiry; a pending entry past it is discarded."
    )


class GeminiFileUploader(AttachmentRenderer):
    """Uploads attachments to the Gemini Files API and references them by URI.

    Built once by `build_attachment_handler` and held by the cog's cached input builder, so its
    state spans every in-flight pipeline: the pending re-poll cache below, plus the dead-source
    cache and the media semaphore inherited from `AttachmentRenderer`. Both render methods go
    through `_resolve_file_upload`, which is what makes the ACTIVE-only and degrade-to-None
    contracts in the module docstring hold for either entry point.

    Attributes:
        config: Runtime LLM config supplying the Gemini Files API key for the lazily
            built upload client.
    """

    config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Runtime LLM config supplying the Gemini Files API key for the upload client.",
    )
    # Uploads that timed out while still PROCESSING, keyed by attachment source cache_key
    # (attachment/sticker id or embed url). The next reference to that source re-polls the
    # same file (usually ACTIVE by then) instead of re-uploading. Kept until the file's
    # provider expiry; bounded like the render cache.
    _pending_uploads: OrderedDict[int | str, PendingUpload] = PrivateAttr(
        default_factory=OrderedDict
    )

    @cached_property
    def gemini_client(self) -> genai.Client:
        """The Gemini client for direct Files API uploads, built lazily on first use.

        The client uploads attachments directly (not through the LiteLLM proxy) so each
        upload can be polled to an ACTIVE `state` before it is referenced. Built here, not
        at the cog: this uploader is only constructed on the Gemini answer-model path, so a
        non-Gemini deployment never builds it. An empty `GEMINI_API_KEY` raises here, and
        because construction is lazy that surfaces at the upload call, where `_upload_file`
        catches it and drops the attachment while the text reply still goes out.

        Returns:
            A Gemini client reused across uploads.
        """
        return genai.Client(api_key=self.config.gemini_api_key)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> tuple[RenderedPart, datetime] | None:
        """Uploads an image source and references the resulting file uri as an `input_file` part.

        A URL or sticker carries no filename, so one is synthesized; the bytes are fetched and
        downscaled by `load_image_bytes` only if a fresh upload turns out to be needed.

        Args:
            source (Attachment | StickerItem | str): The image attachment, sticker, or embed
                image URL to upload.
            cache_key (int | str): Identifies this source across replies (attachment / sticker
                id, or the embed url), keying both the pending re-poll and dead-source caches.
            allow_dead_cache (bool): Whether a recent fetch failure for this source may skip the
                fetch outright; opt-in for history scrollback only.

        Returns:
            The `input_file` part plus the uri's provider-reported expiry, or None when the
            source was dropped (fetch failed, or the upload has not reached ACTIVE yet).
        """
        source_name = resolve_source_filename(source=source, url_fallback="image.png")
        uploaded = await self._resolve_file_upload(
            cache_key=cache_key,
            filename=source_name,
            load_data=lambda: load_image_bytes(source=source),
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        file_id, expires_at = uploaded
        # The input_file filename is cosmetic (the LiteLLM bridge drops it); the route's
        # attachment marker is derived from message metadata, not from this part.
        part = ResponseInputFileParam(type="input_file", file_id=file_id, filename=source_name)
        return part, expires_at

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> tuple[RenderedPart, datetime] | None:
        """Uploads a non-image attachment and references its file uri as an `input_file` part.

        An attachment whose MIME type cannot be resolved is dropped before any fetch, since the
        upload has to declare one.

        Args:
            attachment (Attachment): The non-image attachment to upload.
            cache_key (int | str): Identifies this source across replies (the attachment id),
                keying both the pending re-poll and dead-source caches.
            allow_dead_cache (bool): Whether a recent fetch failure for this source may skip the
                fetch outright; opt-in for history scrollback only.

        Returns:
            The `input_file` part plus the uri's provider-reported expiry, or None when the
            attachment was dropped (unknown MIME type, fetch failed, or not yet ACTIVE).
        """
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
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        file_id, expires_at = uploaded
        part = ResponseInputFileParam(
            type="input_file", file_id=file_id, filename=attachment.filename
        )
        return part, expires_at

    async def _repoll_pending_upload(
        self, cache_key: int | str
    ) -> tuple[bool, tuple[str, datetime] | None]:
        """Re-polls a prior pending upload once, without re-downloading the source.

        Exactly one poll per reference: a file still PROCESSING keeps its cache entry (moved to
        the LRU tail) for the next one. An expired entry, a poll failure, or any terminal
        non-ACTIVE state drops the entry so the caller re-uploads from scratch.

        Args:
            cache_key (int | str): The source whose pending upload to re-poll.

        Returns:
            `(handled, result)`: `handled=True` means stop and use `result` (an ACTIVE
            `(uri, expiry)`, or None if it is still PROCESSING); `handled=False` means there is
            no usable pending entry, so the caller should fall through to a fresh upload.
        """
        pending = self._pending_uploads.get(cache_key)
        if pending is None:
            return False, None
        if datetime.now(tz=UTC) >= pending.expires_at:
            self._pending_uploads.pop(cache_key, None)
            return False, None
        try:
            uploaded = await self.gemini_client.aio.files.get(name=pending.name)
        except Exception as exc:
            # Broad on purpose: this is a best-effort side-channel, and the caller's renders are
            # gathered without `return_exceptions`, so an escaping error would blank the whole
            # message instead of costing one re-upload.
            logfire.warn(
                "gemini pending upload repoll failed; falling back to a fresh upload",
                cache_key=loggable_cache_key(cache_key=cache_key),
                name=pending.name,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            self._pending_uploads.pop(cache_key, None)
            return False, None
        logfire.debug(
            "gemini pending upload repoll",
            cache_key=loggable_cache_key(cache_key=cache_key),
            state=str(uploaded.state),
            adopted=uploaded.state == FileState.ACTIVE,
        )
        if uploaded.state == FileState.ACTIVE:
            self._pending_uploads.pop(cache_key, None)
            return True, (pending.uri, pending.expires_at)
        if uploaded.state == FileState.PROCESSING:
            # Still cooking; keep it and retry on the next reference.
            self._pending_uploads.move_to_end(cache_key)
            return True, None
        # Terminal non-active state: drop it and let the caller re-upload.
        self._pending_uploads.pop(cache_key, None)
        return False, None

    async def _resolve_file_upload(
        self,
        cache_key: int | str,
        filename: str,
        load_data: "FileBytesLoader",
        allow_dead_cache: bool = False,
    ) -> tuple[str, datetime] | None:
        """Returns an ACTIVE file (uri, expiry), re-polling a prior pending upload first.

        A source whose first upload timed out while still PROCESSING is cached as a
        `PendingUpload` keyed on its `cache_key`. The next reference re-polls that same
        file once (it has usually finished cooking in the background by then) instead of
        re-uploading from scratch, so a large-but-processable attachment becomes usable on
        a later reply rather than being re-uploaded and re-dropped every time. Only an
        ACTIVE file is ever returned, so the answer never references a not-yet-ready uri.

        `load_data` fetches the source bytes (and their mime type) and is awaited only
        when a fresh upload is actually needed: adopting a now-ACTIVE pending upload, or
        dropping one still PROCESSING, never re-downloads the source. So a borderline file
        keeps being adopted even after its Discord CDN url has expired and a re-download
        would fail.

        The download, the upload and the activation poll all run inside one media-semaphore
        slot, and a `load_data` failure is swallowed (marking the source dead only when
        `allow_dead_cache` is set) rather than raised.

        Args:
            cache_key (int | str): Identifies this source across replies, keying both the
                pending re-poll and dead-source caches.
            filename (str): Display name sent with the upload, and the name logged on failure.
            load_data (FileBytesLoader): Lazily fetches the source's bytes and MIME type.
            allow_dead_cache (bool): Whether a recent fetch failure for this source may skip the
                fetch outright, and whether a fresh failure records one.

        Returns:
            The ACTIVE file's `(uri, expiry)`, or None when the source was dropped or its
            upload is still PROCESSING (cached for the next reference).
        """
        handled, adopted = await self._repoll_pending_upload(cache_key=cache_key)
        if handled:
            return adopted
        # The dead-source skip is for history scrollback only (an expired CDN url that
        # re-fails every turn); current/reference renders never opt in, so one transient
        # failure on a just-posted attachment is not poisoned for the next reply.
        if allow_dead_cache and self._is_known_dead(cache_key=cache_key):
            return None
        # One media slot spans the whole download + upload (+ activation poll) for every
        # attachment type, so concurrent pipelines cannot launch dozens of CDN downloads or
        # uploads at once and buffer all their bytes while waiting for an upload slot.
        wait_started = time.monotonic()
        async with self._media_semaphore:
            logfire.debug(
                "gemini media slot acquired",
                cache_key=loggable_cache_key(cache_key=cache_key),
                wait_seconds=time.monotonic() - wait_started,
            )
            try:
                data, content_type = await load_data()
            except Exception as exc:
                # Broad on purpose: `load_data` is caller-supplied and spans a CDN fetch plus a
                # PIL decode, and any failure must degrade to dropping this one attachment.
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
            result = await self._upload_file(
                filename=filename, data=data, content_type=content_type
            )
        if isinstance(result, PendingUpload):
            self._pending_uploads[cache_key] = result
            self._pending_uploads.move_to_end(cache_key)
            if len(self._pending_uploads) > 128:
                self._pending_uploads.popitem(last=False)
            return None
        return result

    async def _upload_file(  # noqa: PLR0911 -- one best-effort upload with several distinct degrade-to-None paths
        self, filename: str, data: bytes, content_type: str
    ) -> tuple[str, datetime] | PendingUpload | None:
        """Uploads bytes to the Gemini Files API, polling to ACTIVE within the bound.

        Sending attachments by file URI instead of inlined base64 keeps oversized
        payloads under Gemini's ~10MB per-part `inline_data` cap. The upload goes
        through the Gemini SDK directly (not the LiteLLM proxy) so the file can be
        polled to an ACTIVE `state` before it is referenced; the proxy's file resource
        only ever reports a deprecated `uploaded` status, which is why a fresh upload
        used immediately intermittently 400s with "not in an ACTIVE state".

        The answer request still references the file through the proxy, by the full
        `uri` (`https://.../files/<id>`): the proxy resolves that to a `fileData.fileUri`
        part, while the bare `files/<id>` name fails its mime-type lookup. The upload +
        activation poll runs in the background while the route and memory selection calls
        resolve, so small files (instant ACTIVE) add no latency and only large / video
        uploads spend any of that overlap window waiting.

        Args:
            filename (str): Display name sent with the upload, and the name logged on failure.
            data (bytes): The already-fetched source bytes to upload.
            content_type (str): The MIME type declared to the Files API.

        Returns:
            The ACTIVE file's `(uri, expiry)`; a `PendingUpload` for one still PROCESSING at
            the bound, which the caller caches to re-poll on the next reference; or None on a
            terminal non-active state or any failure. The expiry is the provider's own
            `expiration_time` (Gemini files live ~48h), so the caches reuse the handle until it
            really expires instead of guessing a fixed TTL.
        """
        activation_timeout_seconds = 15.0
        poll_interval_seconds = 0.5
        started = time.monotonic()
        logfire.debug(
            "gemini upload start", filename=filename, content_type=content_type, bytes=len(data)
        )
        # The caller (`_resolve_file_upload`) holds the media semaphore across this whole
        # call, so the activation poll counts against the concurrency cap on purpose.
        try:
            # Resolved outside the upload call so a missing key is not mistaken for an SDK
            # rejection: the lazy build raises ValueError when no key resolves at all.
            client = self.gemini_client
        except ValueError as exc:
            logfire.error(
                "gemini Files API key missing; dropping attachment",
                filename=filename,
                _exc_info=exc,
            )
            return None
        try:
            uploaded = await client.aio.files.upload(
                file=io.BytesIO(data), config={"mime_type": content_type, "display_name": filename}
            )
        except Exception as exc:
            # Broad on purpose: the SDK and its transport raise no single stable type, and this
            # is the best-effort attachment boundary.
            logfire.warn(
                "gemini Files API upload failed",
                filename=filename,
                content_type=content_type,
                bytes=len(data),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        # The SDK types name/uri as Optional; in practice both are assigned at upload
        # time. Capture the stable resource name once (guarded) so the poll loop and
        # PendingUpload reuse it, and degrade explicitly if the provider ever omits it.
        file_name = uploaded.name
        if file_name is None:
            logfire.warn("upload returned no resource name; dropping", filename=filename)
            return None
        deadline = time.monotonic() + activation_timeout_seconds
        while uploaded.state == FileState.PROCESSING:
            if time.monotonic() >= deadline:
                logfire.warn(
                    "attachment still processing; will retry on next reference", filename=filename
                )
                if uploaded.uri is None:
                    logfire.warn("pending upload has no uri; dropping", filename=filename)
                    return None
                # Hand back the in-flight upload so the caller can re-poll it later
                # instead of re-uploading the same bytes from scratch.
                expires_at = uploaded.expiration_time or (
                    datetime.now(tz=UTC) + timedelta(hours=47)
                )
                return PendingUpload(name=file_name, uri=uploaded.uri, expires_at=expires_at)
            await asyncio.sleep(poll_interval_seconds)
            try:
                uploaded = await self.gemini_client.aio.files.get(name=file_name)
            except Exception as exc:
                # Broad on purpose: the poll is the same best-effort boundary as the upload.
                logfire.warn(
                    "gemini activation poll failed",
                    filename=filename,
                    file_name=file_name,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
                return None
        if uploaded.state != FileState.ACTIVE:
            logfire.warn(
                "attachment failed processing", filename=filename, state=str(uploaded.state)
            )
            return None
        file_uri = uploaded.uri
        if file_uri is None:
            logfire.warn("active upload has no uri; dropping", filename=filename)
            return None
        # Fall back to a conservative 47h (under the ~48h lifetime) if the provider omits
        # the expiry, so a missing field never pins an unbounded cache entry.
        expires_at = uploaded.expiration_time or (datetime.now(tz=UTC) + timedelta(hours=47))
        logfire.debug(
            "gemini upload done",
            filename=filename,
            file_uri=file_uri,
            elapsed_seconds=time.monotonic() - started,
            state="active",
        )
        return file_uri, expires_at
