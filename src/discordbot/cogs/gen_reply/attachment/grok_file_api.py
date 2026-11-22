"""xAI Files API attachment renderer for Grok answer models (disabled scaffold).

Uploads file attachments directly via `xai-sdk` (gRPC; LiteLLM 400s `create_file` for xAI) and
references the file id in Responses API parts. Kept disabled in `select.py` until verified against
a live Grok model. Images delegate to `InlineRenderer` because xAI has no image file-id form.

Enable-time prerequisites (unverified live):
- File attachments add xAI's `attachment_search` tool, requiring an agentic tool-calling Grok
  model and `n == 1`; single-file cap is 48 MB.
- Keep `InlineRenderer`'s type narrowing (text/markdown/code/CSV/JSON/PDF); xAI types uploads by
  filename alone (no MIME field on the proto).
- Close the `AsyncClient` gRPC channel on shutdown once enabled.
- Avoid LiteLLM's chat-completions bridge, which rewrites `input_file` into chat `file` blocks.
"""

import time
import asyncio
from datetime import UTC, datetime, timedelta
from functools import cached_property

import logfire
from xai_sdk import AsyncClient
from pydantic import Field

from discordbot.typings.llm import LLMConfig
from discordbot.typings.media import UploadedFile, RenderedAttachment
from discordbot.typings.timeouts import GROK_FILE_UPLOAD_TIMEOUT_SECONDS
from discordbot.cogs.gen_reply.attachment.base import (
    UploadKind,
    AttachmentSource,
    FileUploadRenderer,
)
from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer

# Upload TTL in seconds (1h min, 30d max); xAI files persist forever without one.
GROK_FILE_EXPIRY_SECONDS = 2_592_000


class GrokFileUploader(FileUploadRenderer):
    """Uploads file attachments to the xAI Files API and references them by file id."""

    config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Runtime LLM config supplying the xAI Files API upload client key.",
    )
    image_renderer: InlineRenderer = Field(
        default_factory=InlineRenderer,
        description=(
            "Renderer images fall back to, since xAI takes no image file id and accepts them "
            "only inline or by public URL."
        ),
    )

    @cached_property
    def xai_client(self) -> AsyncClient:
        """Returns the direct xAI SDK client for Files API uploads, built lazily on first use.

        Raises `ValueError` on construction when `xai_api_key` is empty.
        """
        return AsyncClient(api_key=self.config.xai_api_key)

    async def render_image(
        self, source: AttachmentSource, allow_dead_cache: bool = False
    ) -> RenderedAttachment | None:
        """Returns `source` rendered inline via `image_renderer`, since xAI takes no image file ID.

        Args:
            source (AttachmentSource): Classified image attachment, sticker, or URL to render.
            allow_dead_cache (bool): Passed through to `image_renderer`.
        """
        return await self.image_renderer.render_image(
            source=source, allow_dead_cache=allow_dead_cache
        )

    async def _upload_file(
        self, cache_key: int | str, filename: str, data: bytes, content_type: str, kind: UploadKind
    ) -> UploadedFile | None:
        """Uploads bytes to the xAI Files API and returns the handle, or `None` on failure.

        `content_type` is logged only; `asyncio.timeout` is the sole deadline for streaming upload.
        """
        del cache_key, kind
        started = time.monotonic()
        logfire.debug(
            "xai upload start", filename=filename, content_type=content_type, bytes=len(data)
        )
        try:
            # Resolved outside the upload call so a missing key is not reported as an upload
            # failure: an empty key makes the client constructor raise.
            client = self.xai_client
        except ValueError as exc:
            logfire.error(
                "xAI Files API key missing; dropping attachment", filename=filename, _exc_info=exc
            )
            return None
        try:
            async with asyncio.timeout(delay=GROK_FILE_UPLOAD_TIMEOUT_SECONDS):
                uploaded = await client.files.upload(
                    file=data, filename=filename, expires_after=GROK_FILE_EXPIRY_SECONDS
                )
        except Exception as exc:
            # Broad on purpose: the SDK surfaces auth/quota, size/type rejection and transport
            # errors as unrelated types, and the deadline above adds TimeoutError; any of them
            # just drops this one attachment.
            logfire.warn(
                "failed to upload attachment to xAI Files API",
                filename=filename,
                content_type=content_type,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        if not uploaded.id:
            logfire.warn("upload returned no file id; dropping", filename=filename)
            return None
        # `expires_at` is a protobuf Timestamp, so an omitted one reads as the epoch rather than
        # as None; `HasField` is what tells the two apart.
        if uploaded.HasField("expires_at"):
            expires_at = uploaded.expires_at.ToDatetime(tzinfo=UTC)
        else:
            expires_at = datetime.now(tz=UTC) + timedelta(seconds=GROK_FILE_EXPIRY_SECONDS)
        logfire.debug(
            "xai upload done",
            filename=filename,
            file_id=uploaded.id,
            elapsed_seconds=time.monotonic() - started,
        )
        return UploadedFile(uri=uploaded.id, expires_at=expires_at)
