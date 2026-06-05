"""Builds Responses API input messages from Discord messages."""

import re
import base64
from typing import Literal
import asyncio
from mimetypes import guess_type

import logfire
from nextcord import Embed, Message, Attachment, StickerItem
from pydantic import BaseModel, ConfigDict, SkipValidation
from nextcord.ext import commands
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.images import get_image_data, convert_base64_to_data_uri
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.model_pricing import get_supported_modalities

# Strips the usage_footer appended by `streaming.ResponseStreamer.stream` from
# bot-authored messages before feeding them back as `role=assistant` history.
# Without this, the model performs in-context learning on its own past footers
# and starts hallucinating fake "-# model · ⬆ ... ⬇ ... · $... · ..." lines into
# fresh replies. Anchored on the `\n\n-# ` separator plus the ⬆/⬇ token-count
# icons, which never appear together in user-authored content.
USAGE_FOOTER_RE = re.compile(r"\n\n-#[^\n]*⬆[^\n]*⬇[^\n]*$")

# A display name (or legacy username) containing an `[id: ...]`-shaped string
# could forge the sender-identity prefix this module prepends, which the reply
# persona prompt and the memory extraction prompt both treat as the trusted
# authorship signal. Neutralize the lookalike before rendering.
_ID_PREFIX_LOOKALIKE_RE = re.compile(r"\[\s*id\s*:", flags=re.IGNORECASE)


def sanitize_identity(value: str) -> str:
    """Neutralizes authorship-prefix lookalikes in user-controlled identity fields."""
    return _ID_PREFIX_LOOKALIKE_RE.sub("[id-", value)


class MessageInputBuilder(BaseModel):
    """Converts Discord messages into Responses API input parts.

    Attributes:
        bot: The Discord bot instance, used to detect the bot's own messages.
        runtime_models: Catalog whose slow model gates attachment modalities.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: SkipValidation[commands.Bot]
    runtime_models: RuntimeModelCatalog

    async def get_user_prompt(self, content: str) -> str:
        """Removes the bot mention from the content and strips whitespace."""
        if self.bot.user:
            content = content.replace(f"<@{self.bot.user.id}>", "")
        return content.strip()

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
        content = await self.get_user_prompt(content=message.content)
        if content and self.bot.user and message.author.id == self.bot.user.id:
            content = USAGE_FOOTER_RE.sub("", content)
        if not content and message.embeds:
            content = self.extract_embed_text(embeds=list(message.embeds))
        if not content and message.is_system():
            content = message.system_content
        return content

    async def image_to_part(
        self, source: Attachment | StickerItem | str
    ) -> ResponseInputImageParam | None:
        """Converts an image source to a content part for the API."""
        try:
            if isinstance(source, str):
                # A URL source fetches over the network inside get_image_data, so
                # keep that blocking work off the event loop.
                b64_data = await asyncio.to_thread(get_image_data, image_file=source)
                data_uri = convert_base64_to_data_uri(base64_image=b64_data)
                return ResponseInputImageParam(
                    image_url=data_uri, detail="auto", type="input_image"
                )
            if isinstance(source, Attachment):
                content_type = source.content_type or guess_type(source.filename)[0] or "image/png"
            else:
                content_type = guess_type(source.url)[0] or "image/png"
            file_bytes = await source.read()
            b64_data = base64.b64encode(file_bytes).decode("utf-8")
            data_uri = f"data:{content_type};base64,{b64_data}"
            return ResponseInputImageParam(image_url=data_uri, detail="auto", type="input_image")
        except Exception:
            logfire.warn("Failed to convert this image")
            return None

    async def attachment_to_part(self, attachment: Attachment) -> ResponseInputFileParam | None:
        """Converts a file attachment to a content part for the API."""
        try:
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            file_bytes = await attachment.read()
            b64_data = base64.b64encode(file_bytes).decode()
            mime_type = content_type.split(";")[0].strip()
            if not mime_type:
                logfire.warn(
                    f"Skipping attachment with unknown MIME type: {attachment.filename} ({attachment.url})"
                )
                return None
            data_uri = f"data:{mime_type};base64,{b64_data}"
            return ResponseInputFileParam(
                filename=attachment.filename, file_data=data_uri, type="input_file"
            )
        except Exception:
            logfire.warn(f"Failed to download this attachment: {attachment.url}")
            return None

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

    async def get_attachment_parts(
        self, message: Message
    ) -> list[ResponseInputImageParam | ResponseInputFileParam]:
        """Extracts attachment content parts from a message."""
        slow_model = self.runtime_models.slow_model
        modalities = get_supported_modalities(model_name=slow_model.name)
        content_parts: list[ResponseInputImageParam | ResponseInputFileParam | None] = []

        for attachment in message.attachments:
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            required = self.required_modality(content_type=content_type)
            if required in modalities:
                if content_type.startswith("image/"):
                    content_parts.append(await self.image_to_part(source=attachment))
                else:
                    content_parts.append(await self.attachment_to_part(attachment=attachment))
            else:
                logfire.warn(
                    f"Skipping {required} attachment for {slow_model.name}: {attachment.filename}"
                )

        if "image" in modalities:
            for sticker in message.stickers:
                content_parts.append(await self.image_to_part(source=sticker))

            # Prefer Discord's proxy_url (media.discordapp.net) over the original URL, since sources like Threads CDN expire and reject requests without specific headers.
            for embed in message.embeds:
                if embed.image and (url := embed.image.proxy_url or embed.image.url):
                    content_parts.append(await self.image_to_part(source=url))
                if embed.thumbnail and (url := embed.thumbnail.proxy_url or embed.thumbnail.url):
                    content_parts.append(await self.image_to_part(source=url))

        content_parts = [part for part in content_parts if part is not None]
        return content_parts

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

            # Has attachments → must use a content list with input_text/input_image.
            # role=assistant cannot carry `input_image` (only output_text/refusal),
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
