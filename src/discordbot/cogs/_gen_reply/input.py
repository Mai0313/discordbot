"""Builds Responses API input messages from Discord messages."""

import re
from typing import TYPE_CHECKING, Literal, cast
import asyncio
from datetime import datetime
from mimetypes import guess_type
from collections import OrderedDict

from openai import AsyncOpenAI
import logfire
from nextcord import Embed, Message, Attachment, StickerItem
from pydantic import BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.ext import commands
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam

from discordbot.utils.images import get_image_data, shrink_image_bytes
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.model_pricing import get_supported_modalities

if TYPE_CHECKING:
    from collections.abc import Coroutine

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


def sanitize_identity(value: str) -> str:
    """Neutralizes authorship-prefix lookalikes in user-controlled identity fields."""
    return _ID_PREFIX_LOOKALIKE_RE.sub("[id-", value)


def strip_attachment_parts(messages: list[EasyInputMessageParam]) -> list[EasyInputMessageParam]:
    """Returns copies of input messages with attachment parts reduced to text markers.

    The route and memory-selection preflight calls only need the conversation text plus
    a hint that an attachment exists; re-uploading the full image/file payloads to those
    fast calls just adds latency. The answer request keeps the original parts untouched.
    """
    stripped: list[EasyInputMessageParam] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            stripped.append(message)
            continue
        parts: list[ResponseInputTextParam] = []
        for part in content:
            part_type = part.get("type")
            if part_type == "input_text":
                parts.append(cast("ResponseInputTextParam", part))
            elif part_type == "input_image":
                parts.append(ResponseInputTextParam(text="[attachment: image]", type="input_text"))
            elif part_type == "input_file":
                parts.append(ResponseInputTextParam(text="[attachment: file]", type="input_text"))
        stripped.append(EasyInputMessageParam(role=message["role"], content=parts))
    return stripped


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
    # source count still re-renders. Holds Files-API handles, valid for the file's 48h
    # lifetime, so cache reuse never outlives the underlying upload.
    _attachment_cache: OrderedDict[
        tuple[int, datetime | None, tuple[object, ...]], list[ResponseInputFileParam]
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

    async def _upload_file(self, filename: str, data: bytes, content_type: str) -> str | None:
        """Uploads bytes to the provider Files API and returns a managed file id.

        Referencing attachments by `file_id` instead of inlining their base64 keeps
        oversized payloads under the provider's per-part `inline_data` cap (Gemini
        rejects parts over ~10MB): the proxy resolves the id to a `fileData.fileUri`
        rather than carrying the bytes in the request. `target_model_names` routes the
        upload to the same provider that answers, so the handle is usable there.
        """
        try:
            uploaded = await self.client.files.create(
                file=(filename, data, content_type),
                purpose="user_data",
                extra_body={"target_model_names": self.runtime_models.slow_model.name},
            )
        except Exception:
            logfire.warn(f"Failed to upload attachment to Files API: {filename}")
            return None
        return uploaded.id

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
            filename = "image"
        else:
            filename = (
                getattr(source, "filename", None) or f"{getattr(source, 'name', 'sticker')}.png"
            )
        file_id = await self._upload_file(
            filename=filename, data=file_bytes, content_type=content_type
        )
        if file_id is None:
            return None
        return ResponseInputFileParam(type="input_file", file_id=file_id, filename=filename)

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
        file_id = await self._upload_file(
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
        rather than reusing the Files-API handles `get_attachment_parts` produces.
        Only image attachments, stickers, and embed images are collected; non-image
        files are not editable as images and are skipped.
        """
        sources: list[Attachment | StickerItem | str] = []
        for attachment in message.attachments:
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            if content_type.startswith("image/"):
                sources.append(attachment)
        sources.extend(message.stickers)
        for embed in message.embeds:
            if embed.image and (url := embed.image.proxy_url or embed.image.url):
                sources.append(url)
            if embed.thumbnail and (url := embed.thumbnail.proxy_url or embed.thumbnail.url):
                sources.append(url)
        tasks: list[Coroutine[object, object, tuple[bytes, str]]] = []
        for source in sources:
            tasks.append(self._load_image_bytes(source=source))
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
        self, message: Message
    ) -> list[ResponseInputFileParam | None]:
        """Renders every attachment source on a message; failures stay as None.

        Each source uploads to the Files API; the uploads run concurrently so a message
        with several attachments pays roughly one upload's latency, not the sum.
        """
        model_name = self.runtime_models.slow_model.name
        modalities = get_supported_modalities(model_name=model_name)
        tasks: list[Coroutine[object, object, ResponseInputFileParam | None]] = []

        for attachment in message.attachments:
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            required = self.required_modality(content_type=content_type)
            if required not in modalities:
                logfire.warn(
                    f"Skipping {required} attachment for {model_name}: {attachment.filename}"
                )
                continue
            if content_type.startswith("image/"):
                tasks.append(self.image_to_part(source=attachment))
            else:
                tasks.append(self.attachment_to_part(attachment=attachment))

        if "image" in modalities:
            for sticker in message.stickers:
                tasks.append(self.image_to_part(source=sticker))

            # Prefer Discord's proxy_url (media.discordapp.net) over the original URL, since sources like Threads CDN expire and reject requests without specific headers.
            for embed in message.embeds:
                if embed.image and (url := embed.image.proxy_url or embed.image.url):
                    tasks.append(self.image_to_part(source=url))
                if embed.thumbnail and (url := embed.thumbnail.proxy_url or embed.thumbnail.url):
                    tasks.append(self.image_to_part(source=url))

        return list(await asyncio.gather(*tasks))

    async def get_attachment_parts(self, message: Message) -> list[ResponseInputFileParam]:
        """Extracts attachment content parts from a message, with a per-message cache."""
        if not (message.attachments or message.stickers or message.embeds):
            return []
        # Identify the exact sources `_render_attachment_parts` reads: attachment and
        # sticker ids plus each embed's chosen image/thumbnail URL. A late unfurl or an
        # embed URL swap changes this even when the source counts stay the same.
        sources: tuple[object, ...] = (
            tuple(attachment.id for attachment in message.attachments),
            tuple(sticker.id for sticker in message.stickers),
            tuple(
                (
                    embed.image.proxy_url or embed.image.url if embed.image else None,
                    embed.thumbnail.proxy_url or embed.thumbnail.url if embed.thumbnail else None,
                )
                for embed in message.embeds
            ),
        )
        cache_key = (message.id, message.edited_at, sources)
        cached = self._attachment_cache.get(cache_key)
        if cached is not None:
            self._attachment_cache.move_to_end(cache_key)
            # Hand out per-part copies so no caller ever holds the cached dicts; the
            # values are immutable strings, so the copies stay cheap.
            return [part.copy() for part in cached]

        content_parts = await self._render_attachment_parts(message=message)
        resolved = [part for part in content_parts if part is not None]
        # A None part means a download/convert failed; skip caching so the next reply
        # retries instead of pinning the degraded render.
        if None not in content_parts:
            self._attachment_cache[cache_key] = [part.copy() for part in resolved]
            if len(self._attachment_cache) > 128:
                self._attachment_cache.popitem(last=False)
        return resolved

    async def process_single_message(self, message: Message) -> EasyInputMessageParam:
        """Processes a single Discord message into a Responses API input message."""
        try:
            content = await self.get_cleaned_content(message=message)
            attachment_parts = await self.get_attachment_parts(message=message)
            is_bot = bool(self.bot.user and message.author.id == self.bot.user.id)

            # Bot's own history without attachments → role=assistant carries identity,
            # so the sender-prefix is dropped here. Without this, the model sees its
            # own past replies prefixed with `Bot (bot) [id: ...]:` and learns to mimic
            # that header, which leaks into output despite the prompt-level guard.
            if is_bot and not attachment_parts:
                return EasyInputMessageParam(role="assistant", content=content)

            prefixed = (
                f"{sanitize_identity(value=message.author.display_name)} "
                f"({sanitize_identity(value=message.author.name)}) "
                f"[id: {message.author.id}]: {content}"
            )

            # No attachments → use EasyInputMessageParam's string-content shorthand.
            # The SDK serializes it as `input_text` for role=user, which satisfies
            # GPT-5.4's strict rule about content-part types per role.
            if not attachment_parts:
                return EasyInputMessageParam(role="user", content=prefixed)

            # Has attachments → must use a content list with input_text/input_file.
            # role=assistant cannot carry `input_file` (only output_text/refusal),
            # so bot replies that include generated images (from _handle_image_reply)
            # fall back to role=user; the author prefix above preserves bot identity.
            return EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(text=prefixed, type="input_text"),
                    *attachment_parts,
                ],
            )
        except Exception:
            logfire.warn(f"Failed to process message {message.id}", _exc_info=True)
            return EasyInputMessageParam(role="user", content="")
