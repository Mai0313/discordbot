"""Discord message ingestion and LLM streaming helpers.

Two BaseModel-backed service classes own the Discord-side work that used to
live as private helpers on ``ReplyGeneratorCogs``:

- :class:`DiscordMessageOps` turns Discord messages, embeds, stickers, and
  attachments into Responses API input parts, and manages the bot's
  status-emoji reactions on a user's message.
- :class:`DiscordStreamOps` consumes an OpenAI Responses stream, writes the
  preview / final Discord messages, computes token cost, awards the chat
  reward, and adds the web-search reaction when applicable.
"""

from io import BytesIO
import re
import base64
import contextlib
from typing import Literal
from mimetypes import guess_type

from PIL import Image
import logfire
from nextcord import Embed, Message, Attachment, StickerItem
from pydantic import BaseModel, ConfigDict, SkipValidation
from nextcord.ext import commands
from openai import AsyncStream
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.images import get_pil_image, convert_base64_to_data_uri
from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.model_pricing import get_token_rates, get_supported_modalities
from discordbot.cogs._economy.database import credit_with_repayment
from discordbot.cogs._economy.presentation import currency_text

# Gemini occasionally wraps Discord mention syntax in backticks (inline code),
# which stops Discord from rendering the actual mention. Strip those wrappers
# before sending; matches user (<@id>, <@!id>), role (<@&id>) and channel (<#id>) mentions.
_CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")

# Strips the usage_footer appended by `DiscordStreamOps.handle_streaming` from
# bot-authored messages before feeding them back as `role=assistant` history.
# Without this, the model performs in-context learning on its own past footers
# and starts hallucinating fake "-# model · ⬆ ... ⬇ ... · $... · ..." lines into
# fresh replies. Anchored on the `\n\n-# ` separator plus the ⬆/⬇ token-count
# icons, which never appear together in user-authored content.
_USAGE_FOOTER_RE = re.compile(r"\n\n-#[^\n]*⬆[^\n]*⬇[^\n]*$")
_DISCORD_MESSAGE_LIMIT = 2000


class DiscordMessageOps(BaseModel):
    """Converts Discord messages and attachments into Responses API inputs.

    Attributes:
        bot: Discord bot whose user identity feeds prompt cleanup and the
            assistant role decision in ``process_single_message``.
        runtime_models: Catalog used to gate attachment modalities against the
            active slow model. Optional because callers that only need
            reaction handling (e.g. ``parse_threads``) can omit it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: SkipValidation[commands.Bot]
    runtime_models: RuntimeModelCatalog | None = None

    async def handle_reaction(
        self, message: Message, emoji: str, previous: str | None = None
    ) -> str:
        """Adds ``emoji`` to ``message`` and removes ``previous`` first if given.

        Returns:
            The emoji that was just applied, so callers can chain
            ``current_emoji = await handle_reaction(...)``.
        """
        if previous and self.bot.user:
            with contextlib.suppress(Exception):
                await message.remove_reaction(emoji=previous, member=self.bot.user)
        with contextlib.suppress(Exception):
            await message.add_reaction(emoji=emoji)
        return emoji

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
            content = _USAGE_FOOTER_RE.sub("", content)
        if not content and message.embeds:
            content = self.extract_embed_text(embeds=list(message.embeds))
        if not content and message.is_system():
            content = message.system_content
        return content

    async def _image_to_part(
        self, source: Attachment | StickerItem | str
    ) -> ResponseInputImageParam | None:
        """Converts an image source to a content part for the API."""
        url = source if isinstance(source, str) else source.url
        try:
            if isinstance(source, str):
                downloaded = get_pil_image(image_file=source)
            else:
                downloaded = Image.open(BytesIO(await source.read()))
            downloaded.thumbnail(size=(1568, 1568))
            if downloaded.mode != "RGB":
                downloaded = downloaded.convert("RGB")
            buffer = BytesIO()
            downloaded.save(buffer, format="JPEG", quality=85, optimize=True)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            converted = convert_base64_to_data_uri(b64)
            return ResponseInputImageParam(image_url=converted, detail="auto", type="input_image")
        except Exception:
            logfire.warn(f"Failed to convert this image: {url}")
            return None

    async def _attachment_to_part(self, attachment: Attachment) -> ResponseInputFileParam | None:
        """Converts a file attachment to a content part for the API."""
        try:
            file_bytes = await attachment.read()
            b64_data = base64.b64encode(file_bytes).decode()
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
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
    def _required_modality(content_type: str) -> Literal["image", "video", "audio"]:
        """Maps a MIME type to the input modality the model must accept.

        Document-style files (PDF / text / json / ...) collapse to `image`
        because LiteLLM's `supported_modalities` exposes no separate document
        bucket, and every multimodal model that takes `input_image` also
        accepts inline `input_file` payloads.
        """
        if content_type.startswith("video/"):
            return "video"
        if content_type.startswith("audio/"):
            return "audio"
        return "image"

    async def get_attachment_parts(
        self, message: Message
    ) -> list[ResponseInputImageParam | ResponseInputFileParam]:
        """Extracts attachment content parts from a message."""
        if self.runtime_models is None:
            raise ValueError("DiscordMessageOps.get_attachment_parts requires runtime_models")
        slow_model = self.runtime_models.slow_model
        modalities = get_supported_modalities(model_name=slow_model.name)
        content_parts: list[ResponseInputImageParam | ResponseInputFileParam | None] = []

        for attachment in message.attachments:
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            required = self._required_modality(content_type=content_type)
            if required in modalities:
                if content_type.startswith("image/"):
                    content_parts.append(await self._image_to_part(source=attachment))
                else:
                    content_parts.append(await self._attachment_to_part(attachment=attachment))
            else:
                logfire.warn(
                    f"Skipping {required} attachment for {slow_model.name}: {attachment.filename}"
                )

        if "image" in modalities:
            for sticker in message.stickers:
                content_parts.append(await self._image_to_part(source=sticker))

            # Prefer Discord's proxy_url (media.discordapp.net) over the original URL, since sources like Threads CDN expire and reject requests without specific headers.
            for embed in message.embeds:
                if embed.image and (url := embed.image.proxy_url or embed.image.url):
                    content_parts.append(await self._image_to_part(source=url))
                if embed.thumbnail and (url := embed.thumbnail.proxy_url or embed.thumbnail.url):
                    content_parts.append(await self._image_to_part(source=url))

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
                f"{message.author.display_name} ({message.author.name}) "
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


class DiscordStreamOps(BaseModel):
    """Renders OpenAI Responses streams into Discord messages with economy footer.

    Attributes:
        bot: Discord bot, kept for symmetry with :class:`DiscordMessageOps` and
            future expansion.
        msg_ops: Sibling ops used to delegate the web-search reaction so the
            reaction primitive lives in one place.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bot: SkipValidation[commands.Bot]
    msg_ops: DiscordMessageOps

    @staticmethod
    def _split_reply_for_discord(content: str, footer: str) -> tuple[str, list[str]]:
        """Splits a completed reply into one parent message plus follow-up chunks."""
        if len(f"{content}{footer}") <= _DISCORD_MESSAGE_LIMIT:
            return f"{content}{footer}", []

        tail_capacity = _DISCORD_MESSAGE_LIMIT - len(footer)
        if tail_capacity <= 0:
            raise ValueError("Usage footer is too long for Discord message content")

        parent_content = content[:_DISCORD_MESSAGE_LIMIT]
        remaining = content[_DISCORD_MESSAGE_LIMIT:]
        follow_up_chunks: list[str] = []

        while len(remaining) > _DISCORD_MESSAGE_LIMIT:
            follow_up_chunks.append(remaining[:_DISCORD_MESSAGE_LIMIT])
            remaining = remaining[_DISCORD_MESSAGE_LIMIT:]

        if len(remaining) <= tail_capacity:
            follow_up_chunks.append(f"{remaining}{footer}")
        else:
            follow_up_chunks.append(remaining[:tail_capacity])
            follow_up_chunks.append(f"{remaining[tail_capacity:]}{footer}")
        return parent_content, follow_up_chunks

    async def _write_streaming_preview(
        self, message: Message, reply: Message | None, content: str, displayed_content: str
    ) -> tuple[Message | None, str]:
        """Writes at most one Discord message worth of streaming preview text."""
        preview = content[:_DISCORD_MESSAGE_LIMIT]
        if preview == displayed_content:
            return reply, displayed_content
        if reply is None:
            reply = await message.reply(content=preview)
        else:
            await reply.edit(content=preview)
        return reply, preview

    async def _finalize_streaming_reply(
        self, message: Message, reply: Message | None, content: str, footer: str
    ) -> Message:
        """Writes the final reply, continuing overflow as follow-up replies in the same channel."""
        parent_content, follow_up_chunks = self._split_reply_for_discord(
            content=content, footer=footer
        )
        if reply is None:
            reply = await message.reply(content=parent_content)
        else:
            await reply.edit(content=parent_content)
        previous = reply
        for chunk in follow_up_chunks:
            previous = await previous.reply(content=chunk)
        return reply

    async def handle_streaming(  # noqa: C901 -- dispatches on multiple Responses API stream event types
        self, responses: AsyncStream[ResponseStreamEvent], message: Message
    ) -> str:
        """Handles streaming responses from the API and updates the Discord message."""
        stored_content = ""
        counted_content = 0
        reply: Message | None = None
        displayed_content = ""
        content_started = False
        model_name = ""
        input_tokens = 0
        output_tokens = 0
        used_web_search = False

        async for response in responses:
            if response.type == "response.completed":
                model_name = response.response.model
                if response.response.usage:
                    input_tokens = response.response.usage.input_tokens
                    output_tokens = response.response.usage.output_tokens
            elif response.type in {
                "response.web_search_call.in_progress",
                "response.web_search_call.searching",
                "response.web_search_call.completed",
                "response.output_text.annotation.added",
            }:
                used_web_search = True
            elif response.type == "response.output_text.delta":
                delta = response.delta
                if not content_started:
                    delta = delta.lstrip("\n")
                    if not delta:
                        continue
                    content_started = True
                stored_content += delta
                counted_content += len(delta)

                if counted_content >= 30:
                    reply, displayed_content = await self._write_streaming_preview(
                        message=message,
                        reply=reply,
                        content=stored_content,
                        displayed_content=displayed_content,
                    )
                    counted_content = 0

        input_rate, output_rate = get_token_rates(model_name=model_name)
        cost = input_rate * input_tokens + output_rate * output_tokens

        # Award chat points equal to total tokens used. We await this (rather than fire-and-forget)
        # so the resulting balance can land in the footer.
        # On DB failure, it returns None and the footer falls back to the delta-only format.
        total_tokens = input_tokens + output_tokens
        avatar_url = await guild_avatar_url(
            user=message.author, guild=getattr(message, "guild", None)
        )
        result = await credit_with_repayment(
            user_id=message.author.id,
            name=message.author.name,
            avatar_url=avatar_url,
            amount=total_tokens,
        )

        stored_content = _CODED_MENTION_RE.sub(r"\1", stored_content)
        if result.new_balance is not None:
            balance_text = f"{currency_text(amount=result.new_balance, compact=True)} ({currency_text(amount=total_tokens, signed=True, compact=True)})"
        else:
            balance_text = currency_text(amount=total_tokens, signed=True, compact=True)
        usage_footer = f"\n\n-# {model_name} · ⬆ {input_tokens:,} ⬇ {output_tokens:,} · ${cost:.8f} · {balance_text}"

        # Final update to ensure complete message is displayed.
        await self._finalize_streaming_reply(
            message=message, reply=reply, content=stored_content, footer=usage_footer
        )
        stored_content += usage_footer

        if used_web_search:
            await self.msg_ops.handle_reaction(message=message, emoji="🌐")

        return stored_content
