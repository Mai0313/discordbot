"""Gemini Files API attachment renderer: activation bound, pending re-poll cache.

Turns attachment bytes into an ACTIVE Gemini file URI referenced as an `input_file` part. The
upload and the activation poll themselves are `files_api.py`'s; this module owns what an
attachment does with them: the activation bound, and the per-source pending re-poll cache
that adopts an upload which finished processing after that bound. Kept separate from
`input.py` so the upload state machine does not tangle with source-to-part rendering.
"""

import time
from datetime import UTC, datetime, timedelta
from collections import OrderedDict
from collections.abc import Callable

from google import genai
import logfire
from nextcord import Attachment, StickerItem
from pydantic import Field, BaseModel, PrivateAttr
from google.genai.types import File, FileState
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.typings.media import UploadedFile, RenderedAttachment
from discordbot.typings.timeouts import ATTACHMENT_ACTIVATION_TIMEOUT_SECONDS
from discordbot.cogs.gen_reply.files_api import upload_file, poll_while_processing
from discordbot.cogs.gen_reply.attachment.base import (
    UploadKind,
    FileBytesLoader,
    FileUploadRenderer,
    media_semaphore,
    loggable_cache_key,
)
from discordbot.cogs.gen_reply.attachment.loaders import load_image_bytes, resolve_source_filename


class PendingUpload(BaseModel):
    """A Gemini Files upload still PROCESSING when the activation poll bound elapsed.

    Cached per attachment source so a slow upload (typically large video/media that
    keeps cooking server-side past the bound) is re-polled on the next reference to
    that source instead of re-uploaded from scratch. The answer never references a
    pending uri; it is adopted only once a later `files.get` reports ACTIVE.
    """

    name: str = Field(
        ..., description="The Gemini file resource name (`files/<id>`) used to re-poll its state."
    )
    uri: str = Field(
        ..., description="The full file uri the answer references once the file is ACTIVE."
    )
    expires_at: datetime = Field(
        ..., description="Provider-reported expiry; a pending entry past it is discarded."
    )


class PendingUploadRepoll(BaseModel):
    """What re-polling an in-flight upload concluded.

    Two fields rather than one optional, because "stop, there is nothing yet" and "carry on,
    there was never anything to adopt" are different answers that both carry no file.
    """

    handled: bool = Field(
        ..., description="True to stop here; False to fall through to a fresh upload."
    )
    uploaded: UploadedFile | None = Field(
        default=None, description="The adopted file, or None while it is still processing."
    )


def _expiry_of(*, uploaded: File) -> datetime:
    """The file's provider-reported expiry, or a conservative 47h when the provider omits it.

    47h sits under the ~48h a Gemini file lives, so a missing field never pins an unbounded
    cache entry.
    """
    return uploaded.expiration_time or (datetime.now(tz=UTC) + timedelta(hours=47))


class GeminiFileUploader(FileUploadRenderer):
    """Uploads attachments to the Gemini Files API and references them by URI.

    Uploads through the deployment's own direct client rather than one built from a key of
    its own, because a file is readable only by the project that uploaded it: a uri uploaded
    under any other key fails the whole answer rather than dropping the attachment.

    Overrides `_resolve_file_upload` whole, for the pending re-poll, so its upload is
    `_upload_or_pend` rather than `_upload_file`: a file still PROCESSING at the activation
    bound comes back as a `PendingUpload` to re-poll later, not as a failure.
    """

    gemini_client: Callable[[], genai.Client | None] = Field(
        ...,
        description=(
            "Hands back the deployment's direct Gemini client, or None when no key is "
            "configured. Read at each upload rather than once, so a keyless deployment builds "
            "no client and every upload is dropped with the missing key named."
        ),
    )
    # Uploads that timed out while still PROCESSING, keyed by attachment source cache_key
    # (attachment/sticker id or embed url). The next reference to that source re-polls the
    # same file (usually ACTIVE by then) instead of re-uploading. Kept until the file's
    # provider expiry; bounded like the render cache.
    _pending_uploads: OrderedDict[int | str, PendingUpload] = PrivateAttr(
        default_factory=OrderedDict
    )

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> RenderedAttachment | None:
        source_name = resolve_source_filename(source=source, url_fallback="image.png")
        uploaded = await self._resolve_file_upload(
            cache_key=cache_key,
            filename=source_name,
            load_data=lambda: load_image_bytes(source=source),
            kind="image",
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        # The input_file filename is cosmetic (the LiteLLM bridge drops it); the route's
        # attachment marker is derived from message metadata, not from this part.
        part = ResponseInputFileParam(
            type="input_file", file_id=uploaded.uri, filename=source_name
        )
        return RenderedAttachment(part=part, expires_at=uploaded.expires_at)

    async def _repoll_pending_upload(self, cache_key: int | str) -> PendingUploadRepoll:
        """Re-polls a prior pending upload once, without re-downloading the source."""
        pending = self._pending_uploads.get(cache_key)
        # A pending entry exists only after an upload with a client, so a missing one here
        # means there is nothing to re-poll.
        client = self.gemini_client()
        if pending is None or client is None:
            return PendingUploadRepoll(handled=False)
        if datetime.now(tz=UTC) >= pending.expires_at:
            self._pending_uploads.pop(cache_key, None)
            return PendingUploadRepoll(handled=False)
        try:
            uploaded = await client.aio.files.get(name=pending.name)
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
            return PendingUploadRepoll(handled=False)
        logfire.debug(
            "gemini pending upload repoll",
            cache_key=loggable_cache_key(cache_key=cache_key),
            state=str(uploaded.state),
            adopted=uploaded.state == FileState.ACTIVE,
        )
        if uploaded.state == FileState.ACTIVE:
            self._pending_uploads.pop(cache_key, None)
            return PendingUploadRepoll(
                handled=True, uploaded=UploadedFile(uri=pending.uri, expires_at=pending.expires_at)
            )
        if uploaded.state == FileState.PROCESSING:
            # Still cooking; keep it and retry on the next reference.
            self._pending_uploads.move_to_end(cache_key)
            return PendingUploadRepoll(handled=True)
        # Terminal non-active state: drop it and let the caller re-upload.
        self._pending_uploads.pop(cache_key, None)
        return PendingUploadRepoll(handled=False)

    async def _resolve_file_upload(
        self,
        cache_key: int | str,
        filename: str,
        load_data: "FileBytesLoader",
        kind: UploadKind,
        allow_dead_cache: bool = False,
    ) -> UploadedFile | None:
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
        would fail. `kind` changes nothing here: Gemini uploads an image like any other file.
        """
        del kind
        repoll = await self._repoll_pending_upload(cache_key=cache_key)
        if repoll.handled:
            return repoll.uploaded
        # The dead-source skip is for history scrollback only (an expired CDN url that
        # re-fails every turn); current/reference renders never opt in, so one transient
        # failure on a just-posted attachment is not poisoned for the next reply.
        if allow_dead_cache and self._is_known_dead(cache_key=cache_key):
            return None
        # One media slot spans the whole download + upload (+ activation poll) for every
        # attachment type, so concurrent pipelines cannot launch dozens of CDN downloads or
        # uploads at once and buffer all their bytes while waiting for an upload slot.
        wait_started = time.monotonic()
        async with media_semaphore.get():
            logfire.debug(
                "gemini media slot acquired",
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
            result = await self._upload_or_pend(
                filename=filename, data=loaded.data, content_type=loaded.mime_type
            )
        if isinstance(result, PendingUpload):
            self._pending_uploads[cache_key] = result
            self._pending_uploads.move_to_end(cache_key)
            if len(self._pending_uploads) > 128:
                self._pending_uploads.popitem(last=False)
            return None
        return result

    async def _upload_or_pend(  # noqa: PLR0911 -- one best-effort upload with several distinct degrade-to-None paths
        self, filename: str, data: bytes, content_type: str
    ) -> UploadedFile | PendingUpload | None:
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
        activation poll runs in the background while the route call resolves, so small files (instant ACTIVE) add no latency and only large / video
        uploads spend any of that overlap window waiting. A file still PROCESSING at the
        bound returns a `PendingUpload` (the caller caches it to re-poll on the next
        reference); a terminal non-active state or any failure returns None.

        Returns the provider-reported `expiration_time` alongside the URI so the cache
        can reuse the handle until it actually expires (Gemini files live ~48h) instead
        of guessing a fixed TTL.
        """
        started = time.monotonic()
        logfire.debug(
            "gemini upload start", filename=filename, content_type=content_type, bytes=len(data)
        )
        # The caller (`_resolve_file_upload`) holds the media semaphore across this whole
        # call, so the activation poll counts against the concurrency cap on purpose.
        client = self.gemini_client()
        if client is None:
            logfire.error("gemini Files API key missing; dropping attachment", filename=filename)
            return None
        try:
            uploaded = await upload_file(
                client=client, source=data, mime_type=content_type, display_name=filename
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
        try:
            uploaded = await poll_while_processing(
                client=client,
                uploaded=uploaded,
                name=file_name,
                poll_interval_seconds=0.5,
                timeout_seconds=ATTACHMENT_ACTIVATION_TIMEOUT_SECONDS,
            )
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
        if uploaded.state == FileState.PROCESSING:
            logfire.warn(
                "attachment still processing; will retry on next reference", filename=filename
            )
            if uploaded.uri is None:
                logfire.warn("pending upload has no uri; dropping", filename=filename)
                return None
            # Hand back the in-flight upload so the caller can re-poll it later
            # instead of re-uploading the same bytes from scratch.
            return PendingUpload(
                name=file_name, uri=uploaded.uri, expires_at=_expiry_of(uploaded=uploaded)
            )
        if uploaded.state != FileState.ACTIVE:
            logfire.warn(
                "attachment failed processing", filename=filename, state=str(uploaded.state)
            )
            return None
        file_uri = uploaded.uri
        if file_uri is None:
            logfire.warn("active upload has no uri; dropping", filename=filename)
            return None
        expires_at = _expiry_of(uploaded=uploaded)
        logfire.debug(
            "gemini upload done",
            filename=filename,
            file_uri=file_uri,
            elapsed_seconds=time.monotonic() - started,
            state="active",
        )
        return UploadedFile(uri=file_uri, expires_at=expires_at)
