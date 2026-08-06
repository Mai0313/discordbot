"""Anthropic Files API attachment renderer for Claude answer models.

Uploads attachment bytes through the Anthropic SDK (a direct side-channel, not the LiteLLM
proxy) and references the returned file id in Responses API content parts. Kept disabled in
`select.py` until the proxy's reference-translation path for Claude is verified; today
non-Gemini answer models inline instead (`InlineRenderer`).

Simpler than the Gemini uploader: Anthropic files are usable the moment `beta.files.upload`
returns (no PROCESSING/ACTIVE poll) and persist until deleted (no provider expiry), so there
is no pending-upload re-poll machinery and the render cache uses a fixed synthetic TTL. That
persistence cuts both ways: the synthetic TTL only evicts the local render cache, never the
remote file, and Anthropic files never auto-expire (unlike Gemini's ~48h), so enabling this
renderer needs a deletion / periodic-sweep strategy or active channels accumulate uploads
toward the org storage quota. Evicting-then-deleting is not enough on its own: a file can
still be referenced in scrollback after its cache entry is gone, so the cleanup policy is an
enable-time design decision, not something this disabled scaffold settles.

Reference-path prerequisites before uncommenting the `select.py` branch (the answer request,
not this renderer, builds the message that cites the file): the answer request must send
`anthropic-beta: files-api-2025-04-14` when it references an uploaded file, since LiteLLM does
not auto-add that header for file references; LiteLLM's Anthropic mapper needs the file format,
not a bare `file_id`, to emit a `document` block rather than a code-execution `container_upload`,
so PDF / text references must preserve their MIME; and Anthropic document blocks only cover
PDF / plain text / images, so keep the type narrowing `InlineRenderer` does today (UTF-8 files
as `input_text`, the rest dropped) instead of uploading every MIME and referencing it blindly.
"""

import io
import time
from typing import TYPE_CHECKING
from datetime import UTC, datetime, timedelta
from functools import cached_property

import logfire
from nextcord import Attachment, StickerItem
from pydantic import Field
from anthropic import AsyncAnthropic
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

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

type FileBytesLoader = Callable[[], Awaitable[tuple[bytes, str]]]

# Anthropic Files API entries persist until explicitly deleted (no provider expiry), so the
# per-message render cache reuses an upload for a fixed window as a cheap CDN-rehost safety net.
ANTHROPIC_FILE_CACHE_TTL = timedelta(hours=12)


class AnthropicFileUploader(AttachmentRenderer):
    """Uploads attachments to the Anthropic Files API and references them by file id.

    Attributes:
        config: Runtime LLM config supplying the Anthropic Files API key for the upload client.
    """

    config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Runtime LLM config supplying the Anthropic Files API upload client key.",
    )

    @cached_property
    def anthropic_client(self) -> AsyncAnthropic:
        """The Anthropic client for direct Files API uploads, built lazily on first use.

        Built inside this module rather than via a shared `utils/llm.py` factory while the
        renderer is still disabled in `select.py`: that keeps the `anthropic` import out of the
        live request-path module graph (`utils/llm.py` is imported there) until the Claude path
        is wired on. A missing key does not fail construction; it surfaces at the upload call.

        Returns:
            An Anthropic client reused across uploads.
        """
        return AsyncAnthropic(api_key=self.config.anthropic_api_key)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> tuple[RenderedPart, datetime] | None:
        """Uploads an image source and cites the resulting file id as an `input_image` part.

        A URL or embed image carries no filename, so the upload is named `image.jpg`; a
        filename-only classifier downstream then reads it as an image rather than a document.
        A source that could not be fetched or uploaded is dropped rather than raised, so the
        rest of the message still renders.

        Args:
            source (Attachment | StickerItem | str): The image attachment, sticker, or image url.
            cache_key (int | str): Identifies this source in the shared dead-source cache.
            allow_dead_cache (bool): Whether a recently failed source may be skipped without
                re-fetching; set for history scrollback only.

        Returns:
            The `input_image` part plus its synthetic cache expiry, or None when the source
            was dropped.
        """
        source_name = resolve_source_filename(source=source, url_fallback="image.jpg")
        uploaded = await self._resolve_file_upload(
            cache_key=cache_key,
            filename=source_name,
            load_data=lambda: load_image_bytes(source=source),
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        file_id, expires_at = uploaded
        part = ResponseInputImageParam(type="input_image", file_id=file_id, detail="auto")
        return part, expires_at

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> tuple[RenderedPart, datetime] | None:
        """Uploads a file attachment and cites the resulting file id as an `input_file` part.

        An attachment whose MIME type cannot be guessed is dropped before any upload, since that
        same guess is what types the Files API entry and the reference path needs a real format
        rather than a bare id (see the module docstring). A fetch or upload failure drops the
        attachment the same way.

        Args:
            attachment (Attachment): The non-image attachment to upload.
            cache_key (int | str): Identifies this source in the shared dead-source cache.
            allow_dead_cache (bool): Whether a recently failed source may be skipped without
                re-fetching; set for history scrollback only.

        Returns:
            The `input_file` part plus its synthetic cache expiry, or None when the attachment
            was dropped.
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

    async def _resolve_file_upload(
        self,
        cache_key: int | str,
        filename: str,
        load_data: "FileBytesLoader",
        allow_dead_cache: bool = False,
    ) -> tuple[str, datetime] | None:
        """Fetches a source's bytes and uploads them, under the shared media slot.

        One media slot spans the whole fetch + upload, so concurrent pipelines cannot launch
        dozens of CDN downloads and buffer all their bytes at once. `load_data` is awaited
        inside that slot, and a failure there drops this one attachment; with
        `allow_dead_cache` set it also marks the source dead, so an expired CDN url sitting in
        scrollback is not re-fetched and re-warned on every reply.

        Args:
            cache_key (int | str): Identifies this source in the shared dead-source cache.
            filename (str): The upload's filename, also carried into the failure logs.
            load_data (FileBytesLoader): Lazily fetches the source bytes and their MIME type.
            allow_dead_cache (bool): Whether the dead-source cache may be read and written for
                this source; set for history scrollback only.

        Returns:
            `(file_id, expires_at)`, or None when the bytes could not be loaded or the upload
            failed.
        """
        if allow_dead_cache and self._is_known_dead(cache_key=cache_key):
            return None
        async with self._media_semaphore:
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
            return await self._upload_file(filename=filename, data=data, content_type=content_type)

    async def _upload_file(
        self, filename: str, data: bytes, content_type: str
    ) -> tuple[str, datetime] | None:
        """Uploads bytes to the Anthropic Files API, direct to Anthropic rather than the proxy.

        The SDK sets `files-api-2025-04-14` for this upload call automatically; the separate
        answer request that later references the file must send that beta header itself (see
        the module docstring). Anthropic files have no provider expiry to report back, so the
        cache window handed to the caller is a fixed synthetic TTL instead.

        Best-effort: an SDK, network or auth failure (including a missing key, which the lazy
        client build defers to here) drops this one attachment and the text reply still goes
        out.

        Args:
            filename (str): The name the Files API entry is created under.
            data (bytes): The attachment payload to upload.
            content_type (str): The payload's MIME type, sent as the upload's content type.

        Returns:
            `(file_id, expires_at)` where `expires_at` is the synthetic local cache expiry, or
            None when the upload failed or returned no id.
        """
        started = time.monotonic()
        logfire.debug(
            "anthropic upload start", filename=filename, content_type=content_type, bytes=len(data)
        )
        try:
            uploaded = await self.anthropic_client.beta.files.upload(
                file=(filename, io.BytesIO(data), content_type)
            )
        except Exception as exc:
            # Broad on purpose: SDK, network and auth failures all degrade the same way here.
            logfire.warn(
                "failed to upload attachment to Anthropic Files API",
                filename=filename,
                content_type=content_type,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        if not uploaded.id:
            logfire.warn("upload returned no file id; dropping", filename=filename)
            return None
        logfire.debug(
            "anthropic upload done",
            filename=filename,
            file_id=uploaded.id,
            elapsed_seconds=time.monotonic() - started,
        )
        return uploaded.id, datetime.now(tz=UTC) + ANTHROPIC_FILE_CACHE_TTL
