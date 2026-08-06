"""xAI Files API attachment renderer for Grok answer models.

Uploads file attachments through the OpenAI SDK pointed straight at xAI's own host (a direct
side-channel, not the LiteLLM proxy) and references the returned file id in Responses API
content parts. Kept disabled in `select.py` until the reference path is verified against a live
Grok answer model; today non-Gemini answer models inline instead (`InlineRenderer`).

Unlike the OpenAI scaffold, the upload cannot ride the proxy: LiteLLM routes `create_file` only
for openai / hosted_vllm / azure / vertex_ai / gemini / bedrock / manus / anthropic and 400s an
xai target, so this uploader carries its own `XAI_API_KEY` and forgoes proxy-side cost tracking,
like the Gemini and Anthropic ones. The reference half does survive the proxy: a Grok model
dispatches on xAI's native `/v1/responses`, whose request transform passes `input` through
untouched, so an `input_file` file id reaches xAI verbatim.

Images are deliberately not uploaded. xAI's Files API covers text / markdown / code / CSV /
JSON / PDF, while image understanding documents only `input_image` carrying a base64 data URI
or a public URL (20 MiB, jpg/png) and has no file-id form; minting a Files public URL instead
would put a private Discord attachment on an unauthenticated CDN. So `render_image` delegates
to `InlineRenderer` and only `render_file` uploads.

Prerequisites before uncommenting the `select.py` branch, none of them verified against a live
model: attaching a file implicitly adds xAI's server-side `attachment_search` tool and needs a
model with agentic tool calling (grok-4.20 / grok-4.5 per the docs), which also rules out
`n > 1`; the accepted MIME set is documented only as "many text-based formats" plus PDF, so
keep the type narrowing `InlineRenderer` does today rather than uploading every MIME and
referencing it blindly; a single file is capped at 48 MB; and an uploaded file lives forever
unless a TTL is sent, which is why every upload here carries one (see the constant below).
One hazard to keep in view: forcing LiteLLM's chat-completions bridge rewrites the part into a
chat `file` block, where an http(s)-looking file id is fetched and base64-inlined instead.
"""

import io
import time
from typing import TYPE_CHECKING
from datetime import UTC, datetime, timedelta
from functools import cached_property

from openai import AsyncOpenAI, OpenAIError
import logfire
from nextcord import Attachment, StickerItem
from pydantic import Field
from openai.types.responses.response_input_file_param import ResponseInputFileParam

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.gen_reply.attachment.base import (
    RenderedPart,
    AttachmentRenderer,
    loggable_cache_key,
)
from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs.gen_reply.attachment.loaders import attachment_mime, load_attachment_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

type FileBytesLoader = Callable[[], Awaitable[tuple[bytes, str]]]

# xAI's own API host. The upload goes here directly because the proxy cannot route a file
# upload to xai; the answer request that later cites the file still rides the proxy.
XAI_API_BASE = "https://api.x.ai/v1"
# TTL sent with every upload. An xAI file is kept until it is deleted, so without this the
# uploads accumulate against the team's storage forever; 30 days is the documented maximum.
GROK_FILE_EXPIRY_SECONDS = 2_592_000


class GrokFileUploader(AttachmentRenderer):
    """Uploads file attachments to the xAI Files API and references them by file id.

    Attributes:
        config: Runtime LLM config supplying the xAI Files API key for the upload client.
        image_renderer: Renderer images fall back to, since xAI takes no image file id.
    """

    config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Runtime LLM config supplying the xAI Files API upload client key.",
    )
    image_renderer: InlineRenderer = Field(
        default_factory=InlineRenderer,
        description="Renderer used for images, which xAI accepts only inline or by public URL.",
    )

    @cached_property
    def xai_client(self) -> AsyncOpenAI:
        """The OpenAI-compatible client for direct xAI Files API uploads, built lazily.

        Points at xAI's own host rather than the proxy base url, since LiteLLM refuses to route
        a file upload to xai. Built inside this module rather than via a shared factory while
        the renderer is still disabled in `select.py`, like the Anthropic uploader's client.
        Unlike that one, an empty key fails here rather than at the request: `AsyncOpenAI`
        raises `OpenAIError` at construction, so `_upload_file` resolves the client before the
        upload call to keep a missing key from being logged as an upload failure.

        Returns:
            An OpenAI-compatible client reused across uploads.
        """
        return AsyncOpenAI(base_url=XAI_API_BASE, api_key=self.config.xai_api_key)

    async def render_image(
        self,
        source: Attachment | StickerItem | str,
        cache_key: int | str,
        allow_dead_cache: bool = False,
    ) -> tuple[RenderedPart, datetime] | None:
        """Renders an image inline, since xAI resolves no file id for image input.

        Args:
            source (Attachment | StickerItem | str): The image, sticker, or URL to inline.
            cache_key (int | str): Dead-source cache key; ignored, since the inline renderer is
                stateless.
            allow_dead_cache (bool): Whether stale-source skipping applies; ignored for the same
                reason.

        Returns:
            The inlined image part plus its cache expiry, or None when the source could not be
            loaded.
        """
        return await self.image_renderer.render_image(
            source=source, cache_key=cache_key, allow_dead_cache=allow_dead_cache
        )

    async def render_file(
        self, attachment: Attachment, cache_key: int | str, allow_dead_cache: bool = False
    ) -> tuple[RenderedPart, datetime] | None:
        """Uploads a file attachment and cites it as an `input_file` carrying its xAI file id.

        An attachment with no resolvable MIME type is dropped before the fetch, since the
        multipart upload carries that type and xAI's accepted set is narrow.

        Args:
            attachment (Attachment): The non-image file attachment to upload.
            cache_key (int | str): Key a failed byte fetch is recorded under, so scrollback does
                not re-fetch an expired CDN url on every reply.
            allow_dead_cache (bool): Whether that dead-source cache is consulted and updated; set
                for history scrollback only.

        Returns:
            The `input_file` part plus the file's expiry, or None when the MIME type is unknown
            or the fetch or upload failed.
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
        """Loads an attachment's bytes and uploads them, returning the file id and its expiry.

        Holds the shared media semaphore across both the fetch and the upload, so concurrent
        pipelines cannot launch dozens of simultaneous uploads. A byte-fetch failure drops this
        one attachment and, under `allow_dead_cache`, marks the source dead so scrollback stops
        re-fetching it for `DEAD_SOURCE_TTL`.

        Args:
            cache_key (int | str): Key the dead-source cache is read and recorded under.
            filename (str): Name sent with the upload and reported in the failure logs.
            load_data (FileBytesLoader): Fetches the bytes plus their content type.
            allow_dead_cache (bool): Whether the dead-source cache is consulted and updated.

        Returns:
            The xAI file id and its expiry, or None when the fetch or the upload failed.
        """
        if allow_dead_cache and self._is_known_dead(cache_key=cache_key):
            return None
        async with self._media_semaphore:
            try:
                data, content_type = await load_data()
            except Exception as exc:
                # Broad on purpose: `load_data` is caller-supplied and spans a CDN fetch, and
                # any failure must degrade to dropping this one attachment.
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
        """Uploads bytes to the xAI Files API and returns `(file_id, expires_at)`.

        `purpose` is accepted only for OpenAI SDK compatibility and xAI never reads it. The TTL
        keeps OpenAI's `{anchor, seconds}` shape, which xAI documents alongside a bare number of
        seconds; either way the SDK sends it as a form field ahead of the file part, the order
        xAI requires and rejects with a 400 when it is reversed.

        Args:
            filename (str): Name sent with the multipart file part.
            data (bytes): The attachment's bytes.
            content_type (str): MIME type sent alongside those bytes.

        Returns:
            The xAI file id and its expiry, synthesized from the TTL when the response omits one,
            or None when the API key is missing, the upload failed, or no id came back.
        """
        started = time.monotonic()
        logfire.debug(
            "xai upload start", filename=filename, content_type=content_type, bytes=len(data)
        )
        try:
            # Resolved outside the upload call so a missing key is not reported as an upload
            # failure: an empty key makes the client constructor raise.
            client = self.xai_client
        except OpenAIError as exc:
            logfire.error(
                "xAI Files API key missing; dropping attachment", filename=filename, _exc_info=exc
            )
            return None
        try:
            uploaded = await client.files.create(
                file=(filename, io.BytesIO(data), content_type),
                purpose="assistants",
                expires_after={"anchor": "created_at", "seconds": GROK_FILE_EXPIRY_SECONDS},
            )
        except Exception as exc:
            # Broad on purpose: the SDK surfaces auth/quota, size/type rejection and transport
            # errors as unrelated types; any of them just drops this one attachment.
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
        if uploaded.expires_at is None:
            expires_at = datetime.now(tz=UTC) + timedelta(seconds=GROK_FILE_EXPIRY_SECONDS)
        else:
            expires_at = datetime.fromtimestamp(uploaded.expires_at, tz=UTC)
        logfire.debug(
            "xai upload done",
            filename=filename,
            file_id=uploaded.id,
            elapsed_seconds=time.monotonic() - started,
        )
        return uploaded.id, expires_at
