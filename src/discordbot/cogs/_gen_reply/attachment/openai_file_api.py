"""OpenAI Files API attachment renderer for OpenAI answer models.

Uploads attachment bytes through the OpenAI SDK and references the returned file id in
Responses API content parts. Kept disabled in `select.py` until the OpenAI model path is
ready to rely on uploaded files instead of inline parts.
"""

import io
from typing import TYPE_CHECKING, Literal
import asyncio
from datetime import UTC, datetime, timedelta
from functools import cached_property
from collections import OrderedDict

from openai import AsyncOpenAI
import logfire
from nextcord import Attachment, StickerItem
from pydantic import Field, PrivateAttr, SkipValidation
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.llm import create_litellm_client
from discordbot.typings.llm import LLMConfig
from discordbot.cogs._gen_reply.attachment.base import RenderedPart, AttachmentRenderer
from discordbot.cogs._gen_reply.attachment.loaders import (
    attachment_mime,
    load_image_bytes,
    load_attachment_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

type FileBytesLoader = Callable[[], Awaitable[tuple[bytes, str]]]
type OpenAIFilePurpose = Literal["user_data", "vision"]

DEAD_SOURCE_TTL = timedelta(minutes=30)
MEDIA_CONCURRENCY = 8
OPENAI_FILE_EXPIRY_SECONDS = 2_592_000


class OpenAIFileUploader(AttachmentRenderer):
    """Uploads attachments to OpenAI Files API and references them by file id.

    Attributes:
        config: Runtime LLM config supplying the OpenAI-compatible client settings.
        model_name: Selected answer model name used by LiteLLM to route file uploads.
    """

    model_name: str = Field(description="Selected answer model name for LiteLLM file routing.")
    config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Runtime LLM config supplying the OpenAI-compatible file upload client.",
    )
    _dead_sources: OrderedDict[int | str, datetime] = PrivateAttr(default_factory=OrderedDict)
    _media_semaphore: SkipValidation[asyncio.Semaphore] = PrivateAttr(
        default_factory=lambda: asyncio.Semaphore(MEDIA_CONCURRENCY)
    )

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The OpenAI-compatible client used for Files API uploads."""
        return create_litellm_client(config=self.config)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> tuple[RenderedPart, datetime] | None:
        if isinstance(source, str):
            source_name = "image.jpg"
        else:
            source_name = (
                getattr(source, "filename", None) or f"{getattr(source, 'name', 'sticker')}.png"
            )
        uploaded = await self._resolve_file_upload(
            cache_key=cache_key,
            filename=source_name,
            load_data=lambda: load_image_bytes(source=source),
            purpose="vision",
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
        mime_type = attachment_mime(attachment=attachment)
        if not mime_type:
            logfire.warn(
                f"Skipping attachment with unknown MIME type: {attachment.filename} ({attachment.url})"
            )
            return None
        uploaded = await self._resolve_file_upload(
            cache_key=cache_key,
            filename=attachment.filename,
            load_data=lambda: load_attachment_bytes(attachment=attachment),
            purpose="user_data",
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
        """Whether a source's fetch failed recently enough to skip re-fetching it."""
        dead_at = self._dead_sources.get(cache_key)
        if dead_at is None:
            return False
        if datetime.now(tz=UTC) - dead_at < DEAD_SOURCE_TTL:
            self._dead_sources.move_to_end(cache_key)
            return True
        self._dead_sources.pop(cache_key, None)
        return False

    def _mark_dead(self, cache_key: int | str) -> None:
        """Records a source's fetch failure so history scrollback stays cheap."""
        self._dead_sources[cache_key] = datetime.now(tz=UTC)
        self._dead_sources.move_to_end(cache_key)
        if len(self._dead_sources) > 128:
            self._dead_sources.popitem(last=False)

    async def _resolve_file_upload(
        self,
        cache_key: int | str,
        filename: str,
        load_data: "FileBytesLoader",
        purpose: OpenAIFilePurpose,
        allow_dead_cache: bool = False,
    ) -> tuple[str, datetime] | None:
        """Returns an uploaded OpenAI file id and its cache expiry."""
        if allow_dead_cache and self._is_known_dead(cache_key=cache_key):
            return None
        async with self._media_semaphore:
            try:
                data, content_type = await load_data()
            except Exception:
                logfire.warn(f"Failed to load attachment bytes for upload: {filename}")
                if allow_dead_cache:
                    self._mark_dead(cache_key=cache_key)
                return None
            return await self._upload_file(
                filename=filename, data=data, content_type=content_type, purpose=purpose
            )

    async def _upload_file(
        self, filename: str, data: bytes, content_type: str, purpose: OpenAIFilePurpose
    ) -> tuple[str, datetime] | None:
        """Uploads bytes to OpenAI Files API and returns `(file_id, expires_at)`."""
        try:
            uploaded = await self.client.files.create(
                file=(filename, io.BytesIO(data), content_type),
                purpose=purpose,
                expires_after={"anchor": "created_at", "seconds": OPENAI_FILE_EXPIRY_SECONDS},
                extra_body={"model": self.model_name},
            )
        except Exception:
            logfire.warn(f"Failed to upload attachment to OpenAI Files API: {filename}")
            return None
        if uploaded.status == "error":
            logfire.warn(f"OpenAI file upload failed processing: {filename}")
            return None
        if not uploaded.id:
            logfire.warn(f"OpenAI file upload returned no file id; dropping: {filename}")
            return None
        if uploaded.expires_at is None:
            expires_at = datetime.now(tz=UTC) + timedelta(seconds=OPENAI_FILE_EXPIRY_SECONDS)
        else:
            expires_at = datetime.fromtimestamp(uploaded.expires_at, tz=UTC)
        return uploaded.id, expires_at
