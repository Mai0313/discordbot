"""OpenAI Files API attachment renderer for OpenAI answer models.

Uploads attachment bytes through the OpenAI SDK and references the returned file id in
Responses API content parts. Kept disabled in `select.py` until the OpenAI model path is
ready to rely on uploaded files instead of inline parts; today every non-Gemini answer model
inlines its attachments instead (`InlineRenderer`).

Alone among the Files-API uploaders here this one rides the LiteLLM proxy rather than a direct
side-channel, so it carries no provider key of its own: the client is built from `LLMConfig`'s
`base_url` / `api_key`, the same endpoint the answer request that later cites the file goes to.
That is what makes `extra_body={"model": ...}` load-bearing rather than decorative. The proxy's
`/v1/files` endpoint reads a `model` form field (the OpenAI SDK folds `extra_body` into the
multipart form) to pick which deployment's credentials the upload is made with, then encodes
that model back into the file id it hands out, so the id resolves only through the same proxy.
Sent without it, the upload falls through to whichever provider the proxy's own `files_settings`
names as the default.

Simpler than the Gemini uploader: there is no activation poll and no pending re-poll machinery,
because OpenAI exposes no ACTIVE state to wait for. The `status` the SDK still returns is
documented as deprecated, so an explicit `error` is refused and nothing else is waited on, and
the dead-source cache and the media semaphore are inherited from `AttachmentRenderer` unchanged.
"""

import io
import time
from typing import TYPE_CHECKING, Literal
from datetime import UTC, datetime, timedelta
from functools import cached_property

from openai import AsyncOpenAI
import logfire
from nextcord import Attachment, StickerItem
from pydantic import Field
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

# Lazily fetches a source's bytes and mime type. Awaited only past the dead-source check, so a
# source known to be unfetchable costs no CDN round trip.
type FileBytesLoader = Callable[[], Awaitable[tuple[bytes, str]]]
# The two Files API purposes this renderer uses. `vision` is the purpose OpenAI documents for an
# image later referenced by file id; every other attachment goes up as `user_data`.
type OpenAIFilePurpose = Literal["user_data", "vision"]

# TTL sent with every upload. An OpenAI file that is not a batch file is kept until it is
# deleted, so without this the uploads accumulate against the org's storage forever; 30 days is
# the maximum `expires_after.seconds` accepts.
OPENAI_FILE_EXPIRY_SECONDS = 2_592_000


class OpenAIFileUploader(AttachmentRenderer):
    """Uploads attachments to OpenAI Files API and references them by file id.

    Attributes:
        model_name: Selected answer model name, sent with every upload so the proxy routes it
            to that deployment's credentials.
        config: Runtime LLM config supplying the proxy base url and key the upload client uses.
    """

    model_name: str = Field(description="Selected answer model name for LiteLLM file routing.")
    config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Runtime LLM config supplying the OpenAI-compatible file upload client.",
    )

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The OpenAI-compatible client used for Files API uploads, built lazily on first use.

        Points at the LiteLLM proxy, not OpenAI directly, so the upload and the answer request
        that later cites the file share one endpoint and one credential. An empty
        `OPENAI_API_KEY` raises here, and because construction is lazy that surfaces at the
        upload call, where `_upload_file`'s broad catch logs it as an upload failure and drops
        the one attachment while the text reply still goes out.

        Returns:
            An OpenAI-compatible client reused across uploads.
        """
        return AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> tuple[RenderedPart, datetime] | None:
        """Uploads an image source and cites the resulting file id as an `input_image` part.

        Uploaded under the `vision` purpose, since that is what an image referenced by file id
        is documented to need. A URL or embed image carries no filename, so the upload is named
        `image.jpg`; a filename-only classifier downstream then reads it as an image rather than
        a document. A source that could not be fetched or uploaded is dropped rather than
        raised, so the rest of the message still renders.

        Args:
            source (Attachment | StickerItem | str): The image attachment, sticker, or image url.
            cache_key (int | str): Identifies this source in the shared dead-source cache.
            allow_dead_cache (bool): Whether a recently failed source may be skipped without
                re-fetching; set for history scrollback only.

        Returns:
            The `input_image` part plus the uploaded file's expiry, or None when the source was
            dropped.
        """
        source_name = resolve_source_filename(source=source, url_fallback="image.jpg")
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
        """Uploads a file attachment and cites the resulting file id as an `input_file` part.

        Uploaded under the `user_data` purpose. An attachment whose MIME type cannot be guessed
        is dropped before any upload, since the Files API entry is created with that content
        type and an untyped one is useless to the answer request. A fetch or upload failure
        drops the attachment the same way.

        Args:
            attachment (Attachment): The non-image attachment to upload.
            cache_key (int | str): Identifies this source in the shared dead-source cache.
            allow_dead_cache (bool): Whether a recently failed source may be skipped without
                re-fetching; set for history scrollback only.

        Returns:
            The `input_file` part plus the uploaded file's expiry, or None when the attachment
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

    async def _resolve_file_upload(
        self,
        cache_key: int | str,
        filename: str,
        load_data: "FileBytesLoader",
        purpose: OpenAIFilePurpose,
        allow_dead_cache: bool = False,
    ) -> tuple[str, datetime] | None:
        """Fetches a source's bytes and uploads them, returning the file id and its expiry.

        One media slot covers the whole fetch plus upload, so concurrent pipelines cannot launch
        dozens of CDN downloads at once and buffer all their bytes while waiting for an upload
        slot. The dead-source skip is for history scrollback only (an expired CDN url that
        re-fails every turn); current and reference renders never opt in, so one transient
        failure on a just-posted attachment is not poisoned for the next reply.

        Args:
            cache_key (int | str): Identifies this source in the shared dead-source cache.
            filename (str): Name the uploaded file is created under.
            load_data (FileBytesLoader): Fetches the source bytes and their MIME type.
            purpose (OpenAIFilePurpose): Files API purpose the upload is created with.
            allow_dead_cache (bool): Whether a recently failed source may be skipped without
                re-fetching, and whether a fresh failure marks it dead.

        Returns:
            The `(file_id, expires_at)` pair, or None when the source was skipped, could not be
            fetched, or failed to upload.
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
            return await self._upload_file(
                filename=filename, data=data, content_type=content_type, purpose=purpose
            )

    async def _upload_file(
        self, filename: str, data: bytes, content_type: str, purpose: OpenAIFilePurpose
    ) -> tuple[str, datetime] | None:
        """Uploads bytes to OpenAI Files API and returns `(file_id, expires_at)`.

        The upload rides the proxy and so carries the routing `model` and the TTL described in
        the module docstring. The returned file is usable immediately, so there is no activation
        poll; the deprecated `status` is read only to refuse an explicit `error`. The
        provider-reported expiry wins where it is given, and the TTL just sent stands in where
        it is not, so a missing field never pins an unbounded render-cache entry.

        Args:
            filename (str): Name the uploaded file is created under.
            data (bytes): The already-fetched source bytes.
            content_type (str): MIME type the Files API entry is created with.
            purpose (OpenAIFilePurpose): Files API purpose the upload is created with.

        Returns:
            The `(file_id, expires_at)` pair, or None when the upload failed, was rejected, or
            came back without an id.
        """
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
        return uploaded.id, expires_at
