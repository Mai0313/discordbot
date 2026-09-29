"""OpenAI Files API attachment renderer for OpenAI answer models.

Uploads attachment bytes through the OpenAI SDK and references the returned file id in
Responses API content parts. Kept disabled in `select.py` until the OpenAI model path is
ready to rely on uploaded files instead of inline parts.
"""

import io
import time
from typing import Literal
from datetime import UTC, datetime, timedelta
from functools import cached_property

from openai import AsyncOpenAI
import logfire
from nextcord import Attachment, StickerItem
from pydantic import Field
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.media import UploadedFile, RenderedAttachment
from discordbot.cogs.gen_reply.attachment.base import UploadKind, FileUploadRenderer
from discordbot.cogs.gen_reply.attachment.loaders import load_image_bytes, resolve_source_filename

type OpenAIFilePurpose = Literal["user_data", "vision"]

OPENAI_FILE_EXPIRY_SECONDS = 2_592_000


class OpenAIFileUploader(FileUploadRenderer):
    """Uploads attachments to OpenAI Files API and references them by file id."""

    model_name: str = Field(description="Selected answer model name for LiteLLM file routing.")
    config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Runtime LLM config supplying the OpenAI-compatible file upload client.",
    )

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The OpenAI-compatible client used for Files API uploads."""
        return AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> RenderedAttachment | None:
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
        self, filename: str, data: bytes, content_type: str, kind: UploadKind
    ) -> UploadedFile | None:
        """Uploads bytes to OpenAI Files API and returns the uploaded handle."""
        purpose: OpenAIFilePurpose = "vision" if kind == "image" else "user_data"
        started = time.monotonic()
        logfire.debug(
            "openai upload start", filename=filename, content_type=content_type, bytes=len(data)
        )
        try:
            uploaded = await self.client.files.create(
                file=(filename, io.BytesIO(data), content_type),
                purpose=purpose,
                expires_after={"anchor": "created_at", "seconds": OPENAI_FILE_EXPIRY_SECONDS},
                extra_body={"model": self.model_name},
            )
        except Exception as exc:
            # Broad on purpose: the SDK surfaces auth/quota, mime/purpose rejection and transport
            # errors as unrelated types; any of them just drops this one attachment.
            logfire.warn(
                "failed to upload attachment to OpenAI Files API",
                filename=filename,
                content_type=content_type,
                purpose=purpose,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        if uploaded.status == "error":
            logfire.warn("OpenAI file upload failed processing", filename=filename)
            return None
        if not uploaded.id:
            logfire.warn("upload returned no file id; dropping", filename=filename)
            return None
        if uploaded.expires_at is None:
            expires_at = datetime.now(tz=UTC) + timedelta(seconds=OPENAI_FILE_EXPIRY_SECONDS)
        else:
            expires_at = datetime.fromtimestamp(uploaded.expires_at, tz=UTC)
        logfire.debug(
            "openai upload done",
            filename=filename,
            file_id=uploaded.id,
            elapsed_seconds=time.monotonic() - started,
        )
        return UploadedFile(uri=uploaded.id, expires_at=expires_at)
