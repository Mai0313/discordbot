"""Builds Responses API input messages from Discord messages."""

import re
import time
from typing import TYPE_CHECKING, Literal, cast
import asyncio
from datetime import datetime
from mimetypes import guess_type
from collections import OrderedDict

from openai import AsyncOpenAI, NotFoundError
import logfire
from nextcord import Embed, Message, Attachment, StickerItem
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.ext import commands
from openai.types import FileObject
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam

from discordbot.utils.images import get_image_data, shrink_image_bytes
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.model_pricing import get_supported_modalities

if TYPE_CHECKING:
    from collections.abc import Sequence, Coroutine

# Strips the usage_footer appended by `streaming.ResponseStreamer.stream` from
# bot-authored messages before feeding them back as `role=assistant` history.
# Without this, the model performs in-context learning on its own past footers
# and starts hallucinating fake "-# model · ⬆ ... ⬇ ... · $... · ..." lines into
# fresh replies. Anchored on the `\n\n-# ` separator plus the ⬆/⬇ token-count
# icons, which never appear together in user-authored content. The optional
# trailing `\n-# ...` line matches the second subtext line that credits looked-up
# memory owners, so the whole footer is stripped as one unit.
USAGE_FOOTER_RE = re.compile(r"\n\n-#[^\n]*⬆[^\n]*⬇[^\n]*(?:\n-#[^\n]*)?$")

# A display name (or legacy username) containing an `[id: ...]`-shaped string
# could forge the sender-identity prefix this module prepends, which the reply
# persona prompt and the memory extraction prompt both treat as the trusted
# authorship signal. Neutralize the lookalike before rendering.
_ID_PREFIX_LOOKALIKE_RE = re.compile(r"\[\s*id\s*:", flags=re.IGNORECASE)

# Gemini processes a freshly uploaded file to ACTIVE asynchronously; referencing the
# file id before then 400s with "not in an ACTIVE state". The poll below caps the wait
# so a stuck file cannot wedge a reply, but the wait overlaps the route and memory
# selection calls so a healthy file is usually ACTIVE before the answer needs it.
FILE_ACTIVATION_TIMEOUT_SECONDS = 20.0


def sanitize_identity(value: str) -> str:
    """Neutralizes authorship-prefix lookalikes in user-controlled identity fields."""
    return _ID_PREFIX_LOOKALIKE_RE.sub("[id-", value)


def _file_status_is_ready(status: str | None) -> bool:
    """Whether a Files API status means the file is usable in a request.

    Tolerates both the OpenAI SDK literal (`processed`) and the raw Gemini state
    (`ACTIVE`) the LiteLLM proxy might surface, since that mapping is unverified.
    """
    return (status or "").lower() in {"processed", "active"}


def _file_status_is_failed(status: str | None) -> bool:
    """Whether a Files API status means the upload terminally failed."""
    return (status or "").lower() in {"error", "failed"}


def render_author_identity(display_name: str, username: str, user_id: int) -> str:
    """Renders the single-line author identity stamped into memory files.

    Whitespace runs (including any newline that slips past Discord's name
    rules) collapse to single spaces so the identity can never break the
    one-line header formats the memory store relies on.
    """
    safe_display = " ".join(sanitize_identity(value=display_name).split())
    safe_username = " ".join(sanitize_identity(value=username).split())
    return f"{safe_display} ({safe_username}) [id: {user_id}]"


def render_server_identity(server_name: str, server_id: int) -> str:
    """Renders the single-line server identity stamped into per-server memory files.

    Mirrors `render_author_identity`: the guild name is user-controlled, so it
    is sanitized against `[id:` lookalikes and collapsed to one line before the
    `[id: <server_id>]` suffix the memory store's identity regex expects.
    """
    safe_name = " ".join(sanitize_identity(value=server_name).split())
    return f"{safe_name} [id: {server_id}]"


class AttachmentSource(BaseModel):
    """One renderable attachment source classified from message metadata.

    Collected once per message and shared by the text-only marker render, the
    Files-API upload, the per-message render cache key, and the IMAGE route's
    raw-bytes path. Carries only metadata (no bytes, no network) so it is safe to
    build on the route critical path.

    Attributes:
        handle: The attachment, sticker, or image URL the loaders consume.
        kind: Whether the source renders as an image or a generic file.
        content_type: Resolved MIME type, empty only for unguessable sources.
        cache_key: Stable identity (attachment/sticker id or chosen embed URL).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    handle: SkipValidation[Attachment | StickerItem | str] = Field(
        description="The attachment, sticker, or image URL the loaders consume."
    )
    kind: Literal["image", "file"] = Field(
        description="Whether the source renders as an image or a generic file."
    )
    content_type: str = Field(
        description="Resolved MIME type, empty only for unguessable sources."
    )
    cache_key: int | str = Field(
        description="Stable identity (attachment/sticker id or chosen embed URL) for the cache."
    )


class MessageInputBuilder(BaseModel):
    """Converts Discord messages into Responses API input parts.

    Attributes:
        bot: The Discord bot instance, used to detect the bot's own messages.
        runtime_models: Catalog whose slow model gates attachment modalities.
        client: LiteLLM-proxy client used to upload attachments to the Files API.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: SkipValidation[commands.Bot]
    runtime_models: RuntimeModelCatalog
    client: SkipValidation[AsyncOpenAI]
    # Rendered attachment parts per message, so replying repeatedly in the same channel
    # does not re-upload the same history attachments every time. Keyed on the exact
    # sources rendered (attachment + sticker ids, embed image/thumbnail URLs) plus edit
    # time, so an edit or a late embed unfurl that swaps a URL without changing the
    # source count still re-renders. Each entry pairs a monotonic render time with its
    # Files-API parts so a stale handle is re-uploaded before the file's 48h lifetime.
    _attachment_cache: OrderedDict[
        tuple[int, datetime | None, tuple[object, ...]], tuple[float, list[ResponseInputFileParam]]
    ] = PrivateAttr(default_factory=OrderedDict)

    async def get_user_prompt(self, content: str) -> str:
        """Removes bot mention syntax from image/video generation prompts."""
        if self.bot.user:
            bot_id = re.escape(str(self.bot.user.id))
            content = re.sub(rf"<@!?{bot_id}>", "", content)
        return content.strip()

    def has_bot_mention(self, content: str) -> bool:
        """Returns whether the content mentions the bot directly."""
        if not self.bot.user:
            return False
        bot_id = re.escape(str(self.bot.user.id))
        return re.search(rf"<@!?{bot_id}>", content) is not None

    @staticmethod
    def extract_embed_text(embeds: list[Embed]) -> str:
        """Joins author / title / description / fields / footer text from embeds."""
        embed_parts: list[str] = []
        for embed in embeds:
            parts: list[str] = []
            if embed.author and embed.author.name:
                parts.append(f"Author: {embed.author.name}")
            if embed.title:
                parts.append(f"Title: {embed.title}")
            if embed.description:
                parts.append(embed.description)
            for field in embed.fields:
                parts.append(f"{field.name}: {field.value}")
            if embed.footer and embed.footer.text:
                parts.append(f"Footer: {embed.footer.text}")
            if parts:
                embed_parts.append("\n".join(parts))
        return "\n\n".join(embed_parts)

    async def get_cleaned_content(self, message: Message) -> str:
        """Returns the textual content of a message without the author prefix."""
        content = message.content.strip()
        if content and self.bot.user and message.author.id == self.bot.user.id:
            content = USAGE_FOOTER_RE.sub("", content)
        if not content and message.embeds:
            content = self.extract_embed_text(embeds=list(message.embeds))
        if not content and message.is_system():
            content = message.system_content
        return content

    def collect_attachment_sources(self, message: Message) -> list[AttachmentSource]:
        """Classifies every renderable attachment source on a message from metadata.

        One metadata-only pass shared by the text-only marker render, the Files-API
        upload, the render cache key, and the IMAGE route; does no network or upload
        work so it is safe to call on the route critical path. Embeds prefer Discord's
        `proxy_url` (media.discordapp.net) over the origin URL, since sources like the
        Threads CDN expire and reject requests without specific headers.
        """
        sources: list[AttachmentSource] = []
        for attachment in message.attachments:
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            sources.append(
                AttachmentSource(
                    handle=attachment,
                    kind="image" if content_type.startswith("image/") else "file",
                    content_type=content_type,
                    cache_key=attachment.id,
                )
            )
        for sticker in message.stickers:
            sources.append(
                AttachmentSource(
                    handle=sticker,
                    kind="image",
                    content_type=guess_type(sticker.url)[0] or "image/png",
                    cache_key=sticker.id,
                )
            )
        for embed in message.embeds:
            if embed.image and (url := embed.image.proxy_url or embed.image.url):
                sources.append(
                    AttachmentSource(
                        handle=url,
                        kind="image",
                        content_type=guess_type(url)[0] or "image/png",
                        cache_key=url,
                    )
                )
            if embed.thumbnail and (url := embed.thumbnail.proxy_url or embed.thumbnail.url):
                sources.append(
                    AttachmentSource(
                        handle=url,
                        kind="image",
                        content_type=guess_type(url)[0] or "image/png",
                        cache_key=url,
                    )
                )
        return sources

    def _supported_sources(self, sources: list[AttachmentSource]) -> list[AttachmentSource]:
        """Drops sources whose required modality the slow model cannot accept.

        Gating once on the shared source list keeps the text-only marker render and the
        Files-API upload render in agreement: the route never marks an attachment the
        answer would silently drop, and vice versa.
        """
        if not sources:
            return []
        model_name = self.runtime_models.slow_model.name
        modalities = get_supported_modalities(model_name=model_name)
        supported: list[AttachmentSource] = []
        for source in sources:
            required = self.required_modality(content_type=source.content_type)
            if required not in modalities:
                logfire.warn(
                    f"Skipping {required} attachment for {model_name}: {source.cache_key}"
                )
                continue
            supported.append(source)
        return supported

    async def _upload_file(
        self, filename: str, data: bytes, content_type: str
    ) -> FileObject | None:
        """Uploads bytes to the Files API and returns the created file object.

        Sending attachments by `file_id` instead of inlined base64 keeps oversized
        payloads under Gemini's ~10MB per-part `inline_data` cap. `target_model_names`
        names the upload-only `file_model` deployment, not the reply model, so LiteLLM
        uses it only to pick the Files API credential and never runs inference on it.
        The caller polls the returned object to ACTIVE before using its id.
        """
        try:
            return await self.client.files.create(
                file=(filename, data, content_type),
                purpose="user_data",
                extra_body={"target_model_names": self.runtime_models.file_model.name},
            )
        except Exception:
            logfire.warn(f"Failed to upload attachment to Files API: {filename}")
            return None

    async def _upload_and_activate(
        self, filename: str, data: bytes, content_type: str
    ) -> str | None:
        """Uploads bytes and returns a file id only once the file is usable in a request.

        Gemini processes a freshly uploaded file to ACTIVE asynchronously; referencing
        the id before then 400s. This polls the file to ready, capped by
        `FILE_ACTIVATION_TIMEOUT_SECONDS`, so the wait overlaps the route and memory
        selection calls instead of being paid serially. A file that errors or never
        activates within the cap is dropped (returns None) so the reply answers without
        it rather than failing outright. If the proxy does not implement `files.retrieve`
        the id is returned best-effort, matching the pre-poll behavior.
        """
        uploaded = await self._upload_file(filename=filename, data=data, content_type=content_type)
        if uploaded is None:
            return None
        deadline = time.monotonic() + FILE_ACTIVATION_TIMEOUT_SECONDS
        delay = 0.4
        # The create response already carries a status, so the first iteration handles a
        # file that is ACTIVE on upload (no poll); later iterations poll files.retrieve.
        while True:
            if _file_status_is_ready(status=uploaded.status):
                return uploaded.id
            if _file_status_is_failed(status=uploaded.status):
                logfire.warn(f"Files API reported a failed upload: {filename}")
                return None
            if time.monotonic() >= deadline:
                logfire.warn(
                    f"File never reached ACTIVE within {FILE_ACTIVATION_TIMEOUT_SECONDS}s; "
                    f"dropping attachment: {filename}"
                )
                return None
            await asyncio.sleep(delay=min(delay, 2.0))
            delay = delay * 2
            try:
                uploaded = await self.client.files.retrieve(file_id=uploaded.id)
            except NotFoundError:
                # The proxy does not expose files.retrieve at all (404); fall back to the
                # best-effort id, matching pre-poll behavior, rather than polling to the cap.
                logfire.warn(
                    f"files.retrieve unavailable; using file id without activation poll: {filename}"
                )
                return uploaded.id
            except Exception:
                # A transient poll failure (timeout / 5xx / 429) while the file may still be
                # processing: keep the last status and retry until the deadline instead of
                # handing back an id that is not ACTIVE yet.
                logfire.warn(f"files.retrieve poll failed; retrying until cap: {filename}")

    async def _load_image_bytes(self, source: Attachment | StickerItem | str) -> tuple[bytes, str]:
        """Fetches and downscales an image source to upload-ready bytes and MIME type.

        URL sources fetch over the network and attachments decode/re-encode, so the
        blocking work runs off the event loop. Raises on any fetch/decode failure.
        """
        if isinstance(source, str):
            file_bytes = await asyncio.to_thread(get_image_data, image_file=source, use_b64=False)
            return file_bytes, "image/jpeg"
        if isinstance(source, Attachment):
            content_type = source.content_type or guess_type(source.filename)[0] or "image/png"
        else:
            content_type = guess_type(source.url)[0] or "image/png"
        file_bytes = await source.read()
        return await asyncio.to_thread(
            shrink_image_bytes, payload=file_bytes, content_type=content_type
        )

    async def image_to_part(
        self, source: Attachment | StickerItem | str
    ) -> ResponseInputFileParam | None:
        """Converts an image source to an uploaded `input_file` content part."""
        try:
            file_bytes, content_type = await self._load_image_bytes(source=source)
        except Exception:
            logfire.warn("Failed to convert this image")
            return None
        if isinstance(source, str):
            source_name = "image"
        else:
            source_name = (
                getattr(source, "filename", None) or f"{getattr(source, 'name', 'sticker')}.png"
            )
        file_id = await self._upload_and_activate(
            filename=source_name, data=file_bytes, content_type=content_type
        )
        if file_id is None:
            return None
        # The input_file filename is cosmetic (the LiteLLM bridge drops it); the route's
        # attachment marker is derived from message metadata, not from this part.
        return ResponseInputFileParam(type="input_file", file_id=file_id, filename=source_name)

    async def attachment_to_part(self, attachment: Attachment) -> ResponseInputFileParam | None:
        """Converts a file attachment to an uploaded `input_file` content part."""
        content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
        mime_type = content_type.split(";")[0].strip()
        if not mime_type:
            logfire.warn(
                f"Skipping attachment with unknown MIME type: {attachment.filename} ({attachment.url})"
            )
            return None
        try:
            file_bytes = await attachment.read()
        except Exception:
            logfire.warn(f"Failed to download this attachment: {attachment.url}")
            return None
        file_id = await self._upload_and_activate(
            filename=attachment.filename, data=file_bytes, content_type=mime_type
        )
        if file_id is None:
            return None
        return ResponseInputFileParam(
            type="input_file", file_id=file_id, filename=attachment.filename
        )

    async def get_image_source_bytes(self, message: Message) -> list[bytes]:
        """Returns downscaled bytes of a message's image sources for the IMAGE route.

        Image editing feeds raw pixels to `images.edit`, so it loads bytes directly
        rather than reusing the Files-API handles `get_attachment_parts` produces. Only
        image sources are collected; non-image files are not editable as images. The
        IMAGE route runs on the image model, so the slow model's modality gate is not
        applied here.
        """
        tasks: list[Coroutine[object, object, tuple[bytes, str]]] = []
        for source in self.collect_attachment_sources(message=message):
            if source.kind == "image":
                tasks.append(self._load_image_bytes(source=source.handle))
        loaded = await asyncio.gather(*tasks, return_exceptions=True)
        return [item[0] for item in loaded if isinstance(item, tuple)]

    @staticmethod
    def required_modality(content_type: str) -> Literal["image", "video", "audio", "unknown"]:
        """Maps a MIME type to the input modality the model must accept.

        Documents (PDF / Office / text / code) fall through to `image` as a
        proxy: LiteLLM only reports text/image/audio/video, and image-capable
        models in practice also accept `input_file`. Known binaries (archives,
        executables, octet-stream) are checked first and return `unknown` so
        they are dropped before reaching the API.
        """
        unsupported_binary_mimes = frozenset({
            "application/octet-stream",
            "application/zip",
            "application/x-zip-compressed",
            "application/x-rar-compressed",
            "application/vnd.rar",
            "application/x-7z-compressed",
            "application/x-tar",
            "application/gzip",
            "application/x-gzip",
            "application/x-bzip",
            "application/x-bzip2",
            "application/x-xz",
            "application/java-archive",
            "application/x-msdownload",
            "application/x-dosexec",
            "application/x-executable",
            "application/x-mach-binary",
            "application/x-sharedlib",
            "application/wasm",
        })
        if content_type in unsupported_binary_mimes:
            return "unknown"
        if content_type.startswith("video/"):
            return "video"
        if content_type.startswith("audio/"):
            return "audio"
        if content_type.startswith("image/"):
            return "image"
        return "image"

    async def _render_attachment_parts(
        self, sources: list[AttachmentSource]
    ) -> list[ResponseInputFileParam | None]:
        """Renders every supported source to an uploaded part; failures stay as None.

        Each source uploads to the Files API and polls to ACTIVE; the uploads run
        concurrently so a message with several attachments pays roughly one upload's
        latency, not the sum.
        """
        tasks: list[Coroutine[object, object, ResponseInputFileParam | None]] = []
        for source in sources:
            if source.kind == "image":
                tasks.append(self.image_to_part(source=source.handle))
            else:
                # Only attachments are ever classified as files; stickers and embeds are images.
                tasks.append(self.attachment_to_part(attachment=cast("Attachment", source.handle)))
        return list(await asyncio.gather(*tasks))

    async def get_attachment_parts(
        self, message: Message, sources: list[AttachmentSource] | None = None
    ) -> list[ResponseInputFileParam]:
        """Extracts attachment content parts from a message, with a per-message cache.

        Pass the pre-collected supported `sources` to avoid re-collecting; when omitted
        they are collected and gated here so direct callers keep working.
        """
        if sources is None:
            sources = self._supported_sources(
                sources=self.collect_attachment_sources(message=message)
            )
        if not sources:
            return []
        # Key on the exact sources rendered plus the edit time, so a late embed unfurl
        # or an edit that swaps a source URL re-renders even when the count is unchanged.
        source_keys = tuple(source.cache_key for source in sources)
        cache_key = (message.id, message.edited_at, source_keys)
        # Files API handles live ~48h; re-render before then so a long-lived cache entry
        # never hands back an expired file_id that the answer request would reject.
        cache_ttl_seconds = 36 * 3600
        cached = self._attachment_cache.get(cache_key)
        if cached is not None and time.monotonic() - cached[0] <= cache_ttl_seconds:
            self._attachment_cache.move_to_end(cache_key)
            # Hand out per-part copies so no caller ever holds the cached dicts; the
            # values are immutable strings, so the copies stay cheap.
            return [part.copy() for part in cached[1]]

        content_parts = await self._render_attachment_parts(sources=sources)
        resolved = [part for part in content_parts if part is not None]
        # A None part means a download/convert/activation failed; skip caching so the
        # next reply retries instead of pinning a degraded or never-activated render.
        if None not in content_parts:
            self._attachment_cache[cache_key] = (
                time.monotonic(),
                [part.copy() for part in resolved],
            )
            if len(self._attachment_cache) > 128:
                self._attachment_cache.popitem(last=False)
        return resolved

    def _assemble_input_message(
        self,
        message: Message,
        content: str,
        parts: "Sequence[ResponseInputTextParam | ResponseInputFileParam]",
        has_attachments: bool,
    ) -> EasyInputMessageParam:
        """Assembles one input message, sharing role and prefix rules across renders.

        `has_attachments` is decided from the message's sources, not from `parts`, so
        the text-only render (markers) and the full render (uploaded files) agree on
        role and message shape even when every upload is dropped.
        """
        is_bot = bool(self.bot.user and message.author.id == self.bot.user.id)

        # Bot's own history without attachments → role=assistant carries identity, so the
        # sender-prefix is dropped here. Without this, the model sees its own past replies
        # prefixed with `Bot (bot) [id: ...]:` and learns to mimic that header.
        if is_bot and not has_attachments:
            return EasyInputMessageParam(role="assistant", content=content)

        prefixed = (
            f"{sanitize_identity(value=message.author.display_name)} "
            f"({sanitize_identity(value=message.author.name)}) "
            f"[id: {message.author.id}]: {content}"
        )

        # No attachments → use EasyInputMessageParam's string-content shorthand. The SDK
        # serializes it as `input_text` for role=user, which satisfies the strict rule
        # about content-part types per role.
        if not has_attachments:
            return EasyInputMessageParam(role="user", content=prefixed)

        # Has attachments → must use a content list. role=assistant cannot carry
        # `input_file` (only output_text/refusal), so bot replies that include generated
        # images fall back to role=user; the author prefix above preserves bot identity.
        return EasyInputMessageParam(
            role="user", content=[ResponseInputTextParam(text=prefixed, type="input_text"), *parts]
        )

    async def render_text_only(
        self, message: Message, sources: list[AttachmentSource]
    ) -> EasyInputMessageParam:
        """Renders a message as cleaned text plus `[attachment: kind]` markers.

        Pure metadata plus the already-cheap cleaned content; performs no upload, so the
        route and memory-selection calls never wait on the Files API. Mirrors
        `process_single_message`'s role and prefix rules so the route sees the same shape
        the answer will, minus the payload bytes.
        """
        content = await self.get_cleaned_content(message=message)
        markers: list[ResponseInputTextParam] = [
            ResponseInputTextParam(text=f"[attachment: {source.kind}]", type="input_text")
            for source in sources
        ]
        return self._assemble_input_message(
            message=message, content=content, parts=markers, has_attachments=bool(sources)
        )

    async def process_single_message_text_only(self, message: Message) -> EasyInputMessageParam:
        """Renders a message for the route and memory-selection calls without uploading."""
        try:
            sources = self._supported_sources(
                sources=self.collect_attachment_sources(message=message)
            )
            return await self.render_text_only(message=message, sources=sources)
        except Exception:
            # The route awaits this before dispatching, so a cold-start modality lookup
            # (or render) failure must degrade to empty text like process_single_message
            # does, not abort the whole reply through the generic error path.
            logfire.warn(f"Failed to render message {message.id} for routing", _exc_info=True)
            return EasyInputMessageParam(role="user", content="")

    async def process_single_message(self, message: Message) -> EasyInputMessageParam:
        """Processes a single Discord message into a Responses API input message."""
        try:
            content = await self.get_cleaned_content(message=message)
            sources = self._supported_sources(
                sources=self.collect_attachment_sources(message=message)
            )
            attachment_parts = await self.get_attachment_parts(message=message, sources=sources)
            return self._assemble_input_message(
                message=message,
                content=content,
                parts=attachment_parts,
                has_attachments=bool(sources),
            )
        except Exception:
            logfire.warn(f"Failed to process message {message.id}", _exc_info=True)
            return EasyInputMessageParam(role="user", content="")
