"""Gemini Files API attachment renderer: direct-SDK upload, activation poll, re-poll cache.

Owns the mechanical side-channel that turns attachment bytes into an ACTIVE Gemini file URI
referenced as an `input_file` part: the direct-SDK upload, the activation poll, the pending
re-poll, and the per-source dead-source / pending / concurrency caches. Kept separate from
`input.py` so the upload state machine does not tangle with source-to-part rendering.
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
from pydantic import Field, BaseModel, PrivateAttr, SkipValidation
from google.genai.types import FileState
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.utils.llm import create_gemini_client
from discordbot.typings.llm import LLMConfig
from discordbot.cogs._gen_reply.attachment.base import RenderedPart, AttachmentRenderer
from discordbot.cogs._gen_reply.attachment.loaders import (
    attachment_mime,
    load_image_bytes,
    load_attachment_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

# Lazily fetches a source's bytes and mime type. Awaited only when a fresh Gemini upload is
# needed, so adopting an already-uploaded pending file never re-downloads the source.
type FileBytesLoader = Callable[[], Awaitable[tuple[bytes, str]]]

# A source whose byte fetch fails (typically an expired Discord/Threads CDN url that sits in
# history scrollback) is skipped for this long so it is not re-fetched and re-warned on every
# reply; after the window it is retried once so a transient blip self-heals.
DEAD_SOURCE_TTL = timedelta(minutes=30)
# Bounds concurrent media fetch + Files-API upload work across all in-flight pipelines (the
# input builder is a shared singleton). Above the typical per-message attachment count so a
# single request stays fully parallel, while two concurrent pipelines cannot launch dozens of
# simultaneous uploads and starve each other (the source of the worst observed render tail).
MEDIA_CONCURRENCY = 8


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
    # Sources whose byte fetch failed, keyed by cache_key -> first-failure time. A hit within
    # DEAD_SOURCE_TTL skips the fetch fast (no network, no per-turn re-warn); past the TTL the
    # entry is dropped and the source retried once. Bounded like the caches above.
    _dead_sources: OrderedDict[int | str, datetime] = PrivateAttr(default_factory=OrderedDict)
    # Caps concurrent media fetch/upload work; see MEDIA_CONCURRENCY. Created in-loop on first
    # uploader access (during message handling), so it binds to the running event loop.
    _media_semaphore: SkipValidation[asyncio.Semaphore] = PrivateAttr(
        default_factory=lambda: asyncio.Semaphore(MEDIA_CONCURRENCY)
    )

    @cached_property
    def gemini_client(self) -> genai.Client:
        """The Gemini client for direct Files API uploads, built lazily on first use.

        The client uploads attachments directly (not through the LiteLLM proxy) so each
        upload can be polled to an ACTIVE `state` before it is referenced. Built here, not
        at the cog: this uploader is only constructed on the Gemini answer-model path, so a
        non-Gemini deployment never builds it. A missing `GEMINI_API_KEY` does not fail
        construction; the failure surfaces at the upload call, where `_upload_file` catches
        it and drops the attachment while the text reply still goes out.

        Returns:
            A Gemini client reused across uploads.
        """
        return create_gemini_client(config=self.config)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> tuple[RenderedPart, datetime] | None:
        if isinstance(source, str):
            source_name = "image"
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
        # The input_file filename is cosmetic (the LiteLLM bridge drops it); the route's
        # attachment marker is derived from message metadata, not from this part.
        part = ResponseInputFileParam(type="input_file", file_id=file_id, filename=source_name)
        return part, expires_at

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> tuple[RenderedPart, datetime] | None:
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

    async def _repoll_pending_upload(
        self, cache_key: int | str
    ) -> tuple[bool, tuple[str, datetime] | None]:
        """Re-polls a prior pending upload once, without re-downloading the source.

        Returns `(handled, result)`: `handled=True` means stop and use `result` (an ACTIVE
        `(uri, expiry)`, or `None` if it is still PROCESSING); `handled=False` means there is
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
        except Exception:
            self._pending_uploads.pop(cache_key, None)
            return False, None
        logfire.debug(
            "gemini pending upload repoll",
            cache_key=cache_key,
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
                cache_key=cache_key,
                wait_seconds=time.monotonic() - wait_started,
            )
            try:
                data, content_type = await load_data()
            except Exception:
                logfire.warn(
                    "failed to load attachment bytes for upload",
                    filename=filename,
                    cache_key=cache_key,
                    allow_dead_cache=allow_dead_cache,
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
        uploads spend any of that overlap window waiting. A file still PROCESSING at the
        bound returns a `PendingUpload` (the caller caches it to re-poll on the next
        reference); a terminal non-active state or any failure returns None.

        Returns the provider-reported `expiration_time` alongside the URI so the cache
        can reuse the handle until it actually expires (Gemini files live ~48h) instead
        of guessing a fixed TTL.
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
            uploaded = await self.gemini_client.aio.files.upload(
                file=io.BytesIO(data), config={"mime_type": content_type, "display_name": filename}
            )
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
                        "attachment still processing; will retry on next reference",
                        filename=filename,
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
                uploaded = await self.gemini_client.aio.files.get(name=file_name)
            if uploaded.state != FileState.ACTIVE:
                logfire.warn(
                    "attachment failed processing", filename=filename, state=str(uploaded.state)
                )
                return None
        except Exception:
            logfire.warn("failed to upload attachment to Files API", filename=filename)
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
