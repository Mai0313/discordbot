"""Anthropic Files API attachment renderer for Claude answer models (disabled scaffold).

Uploads attachment bytes directly via the Anthropic SDK and references the file id in Responses
API parts. Kept disabled in `select.py` until the proxy's Claude file-reference path is verified.

Enable-time prerequisites:
- Files persist indefinitely with no provider auto-expiry; add a remote deletion/sweep policy
  (the synthetic `ANTHROPIC_FILE_CACHE_TTL` only evicts the local render cache).
- Answer requests referencing uploaded files must send `anthropic-beta: files-api-2025-04-14`
  (LiteLLM does not add it automatically).
- Preserve MIME types on PDF/text references so LiteLLM emits a `document` block rather than
  `container_upload`, and keep `InlineRenderer`'s type narrowing (PDF, plain text, images).
"""

import io
import time
from datetime import UTC, datetime, timedelta
from functools import cached_property

import logfire
from nextcord import Attachment, StickerItem
from pydantic import Field
from anthropic import AsyncAnthropic
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.media import UploadedFile, RenderedAttachment
from discordbot.cogs.gen_reply.attachment.base import UploadKind, FileUploadRenderer
from discordbot.cogs.gen_reply.attachment.loaders import load_image_bytes, resolve_source_filename

# Synthetic local cache window; Anthropic files have no provider auto-expiry.
ANTHROPIC_FILE_CACHE_TTL = timedelta(hours=12)


class AnthropicFileUploader(FileUploadRenderer):
    """Uploads attachments to the Anthropic Files API and references them by file id."""

    config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Runtime LLM config supplying the Anthropic Files API upload client key.",
    )

    @cached_property
    def anthropic_client(self) -> AsyncAnthropic:
        """Returns the Anthropic client for direct Files API uploads, built lazily on first use.

        A missing key does not fail construction; it surfaces at the upload call.
        """
        return AsyncAnthropic(api_key=self.config.anthropic_api_key)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> RenderedAttachment | None:
        """Returns `source` uploaded to the Anthropic Files API as an `input_image` part, or `None`.

        Args:
            source (Attachment | StickerItem | str): Image attachment, sticker, or URL to upload.
            cache_key (int | str): Per-source identifier for dead-source caching.
            allow_dead_cache (bool): Whether to skip and record sources whose fetch failed.
        """
        source_name = resolve_source_filename(source=source, url_fallback="image.jpg")
        uploaded = await self._resolve_file_upload(
            cache_key=cache_key,
            filename=source_name,
            load_data=lambda: load_image_bytes(source=source),
            kind="image",
            allow_dead_cache=allow_dead_cache,
        )
        if uploaded is None:
            return None
        part = ResponseInputImageParam(type="input_image", file_id=uploaded.uri, detail="auto")
        return RenderedAttachment(part=part, expires_at=uploaded.expires_at)

    async def _upload_file(
        self, cache_key: int | str, filename: str, data: bytes, content_type: str, kind: UploadKind
    ) -> UploadedFile | None:
        """Uploads bytes to the Anthropic Files API and returns the handle with a synthetic TTL."""
        del cache_key, kind
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
        return UploadedFile(
            uri=uploaded.id, expires_at=datetime.now(tz=UTC) + ANTHROPIC_FILE_CACHE_TTL
        )
