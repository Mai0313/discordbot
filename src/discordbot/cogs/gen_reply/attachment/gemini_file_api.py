"""Gemini Files API attachment renderer: activation bound and pending re-poll cache.

Uploads attachment bytes directly via the Gemini SDK (`files_api.py`) and polls each file to
`ACTIVE` before returning a full `https://.../files/<id>` URI (required by the proxy to emit
`fileData.fileUri` rather than failing MIME lookup on a bare `files/<id>` name). Handles are
cached until `expiration_time` (~48h); files still `PROCESSING` when the activation timeout
elapses are cached per source and re-polled on later turns instead of re-uploaded.
"""

import time
import asyncio
from datetime import UTC, datetime, timedelta
from collections.abc import Callable

from google import genai
import logfire
from pydantic import Field, BaseModel, PrivateAttr
from google.genai.types import File, FileState
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.utils.lru import BoundedLRU
from discordbot.typings.media import UploadedFile, RenderedAttachment
from discordbot.typings.timeouts import (
    FILES_API_READ_TIMEOUT_SECONDS,
    ATTACHMENT_UPLOAD_TIMEOUT_SECONDS,
    ATTACHMENT_ACTIVATION_TIMEOUT_SECONDS,
)
from discordbot.cogs.gen_reply.files_api import upload_file, poll_while_processing
from discordbot.cogs.gen_reply.attachment.base import (
    UploadKind,
    FileBytesLoader,
    AttachmentSource,
    FileUploadRenderer,
    loggable_cache_key,
)
from discordbot.cogs.gen_reply.attachment.loaders import load_image_bytes


class PendingUpload(BaseModel):
    """A Gemini Files upload still `PROCESSING` when the activation poll bound elapsed.

    Cached per attachment source so a slow upload is re-polled on the next reference instead of
    re-uploaded from scratch. Adopted only once a later `files.get` reports `ACTIVE`.
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
    """Outcome of re-polling an in-flight `PendingUpload`."""

    handled: bool = Field(
        ..., description="True to stop here; False to fall through to a fresh upload."
    )
    uploaded: UploadedFile | None = Field(
        default=None, description="The adopted file, or None while it is still processing."
    )


def _expiry_of(uploaded: File) -> datetime:
    """Returns `uploaded.expiration_time`, or 47 hours from now when omitted."""
    return uploaded.expiration_time or (datetime.now(tz=UTC) + timedelta(hours=47))


class GeminiFileUploader(FileUploadRenderer):
    """Uploads attachments to the Gemini Files API and references them by URI.

    Uses the deployment's own direct Gemini client so the project that uploads a file is the
    same project that reads it during the answer call. Re-polls any cached `PendingUpload`
    before downloading or uploading a source again.
    """

    gemini_client: Callable[[], genai.Client | None] = Field(
        ...,
        description=(
            "Hands back the deployment's direct Gemini client, or None when no key is "
            "configured. A keyless deployment inlines instead of building this uploader, so a "
            "None here only drops the upload with the missing key named."
        ),
    )
    # Uploads that timed out while still `PROCESSING`, keyed by attachment source `cache_key`.
    # Re-polled on the next reference and kept until provider expiry; bounded at 128 entries.
    _pending_uploads: BoundedLRU[int | str, PendingUpload] = PrivateAttr(
        default_factory=BoundedLRU
    )

    async def render_image(
        self, source: AttachmentSource, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns `source` uploaded to the Gemini Files API as an `input_file` part, or `None`.

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
        # LiteLLM drops `filename`, but the Interactions answer path classifies parts by extension.
        part = ResponseInputFileParam(
            type="input_file", file_id=uploaded.uri, filename=source.filename
        )
        return RenderedAttachment(part=part, expires_at=uploaded.expires_at)

    async def _repoll_pending_upload(self, cache_key: int | str) -> PendingUploadRepoll:
        """Re-polls a prior pending upload once, without re-downloading the source."""
        pending = self._pending_uploads.peek(key=cache_key)
        client = self.gemini_client()
        if pending is None or client is None:
            return PendingUploadRepoll(handled=False)
        if datetime.now(tz=UTC) >= pending.expires_at:
            self._pending_uploads.pop(key=cache_key)
            return PendingUploadRepoll(handled=False)
        try:
            async with asyncio.timeout(delay=FILES_API_READ_TIMEOUT_SECONDS):
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
            self._pending_uploads.pop(key=cache_key)
            return PendingUploadRepoll(handled=False)
        logfire.debug(
            "gemini pending upload repoll",
            cache_key=loggable_cache_key(cache_key=cache_key),
            state=str(uploaded.state),
            adopted=uploaded.state == FileState.ACTIVE,
        )
        if uploaded.state == FileState.ACTIVE:
            self._pending_uploads.pop(key=cache_key)
            return PendingUploadRepoll(
                handled=True, uploaded=UploadedFile(uri=pending.uri, expires_at=pending.expires_at)
            )
        if uploaded.state == FileState.PROCESSING:
            self._pending_uploads.touch(key=cache_key)
            return PendingUploadRepoll(handled=True)
        self._pending_uploads.pop(key=cache_key)
        return PendingUploadRepoll(handled=False)

    async def _resolve_file_upload(
        self,
        cache_key: int | str,
        filename: str,
        load_data: "FileBytesLoader",
        kind: UploadKind,
        allow_dead_cache: bool = False,
    ) -> UploadedFile | None:
        """Returns an `ACTIVE` upload, re-polling any cached `PendingUpload` before fetching."""
        repoll = await self._repoll_pending_upload(cache_key=cache_key)
        if repoll.handled:
            return repoll.uploaded
        return await super()._resolve_file_upload(
            cache_key=cache_key,
            filename=filename,
            load_data=load_data,
            kind=kind,
            allow_dead_cache=allow_dead_cache,
        )

    async def _upload_file(
        self, cache_key: int | str, filename: str, data: bytes, content_type: str, kind: UploadKind
    ) -> UploadedFile | None:
        """Uploads `data`, caching a `PendingUpload` if still `PROCESSING` at the bound."""
        del kind
        result = await self._upload_or_pend(
            filename=filename, data=data, content_type=content_type
        )
        if isinstance(result, PendingUpload):
            self._pending_uploads.set(key=cache_key, value=result)
            return None
        return result

    async def _upload_or_pend(  # noqa: PLR0911 -- one best-effort upload with several distinct degrade-to-None paths
        self, filename: str, data: bytes, content_type: str
    ) -> UploadedFile | PendingUpload | None:
        """Uploads `data` and polls to `ACTIVE`, returning `PendingUpload` on timeout or `None`."""
        started = time.monotonic()
        logfire.debug(
            "gemini upload start", filename=filename, content_type=content_type, bytes=len(data)
        )
        # The shared `_resolve_file_upload` holds the media semaphore across this whole call,
        # so the activation poll counts against the concurrency cap on purpose.
        client = self.gemini_client()
        if client is None:
            logfire.error("gemini Files API key missing; dropping attachment", filename=filename)
            return None
        try:
            async with asyncio.timeout(delay=ATTACHMENT_UPLOAD_TIMEOUT_SECONDS):
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
                read_timeout_seconds=FILES_API_READ_TIMEOUT_SECONDS,
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
