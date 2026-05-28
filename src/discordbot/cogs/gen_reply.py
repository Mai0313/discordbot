"""Cog that routes Discord messages through the AI reply pipeline."""

from io import BytesIO
import re
import base64
from typing import TYPE_CHECKING, Literal, cast
import asyncio
from functools import cached_property
from mimetypes import guess_type
import contextlib

from openai import AsyncOpenAI, AsyncStream
import logfire
from nextcord import File, Embed, Message, Attachment, StickerItem
from pydantic import ValidationError
from nextcord.ext import commands
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.utils.images import get_image_data, convert_base64_to_data_uri
from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.models import RouteDecision, RuntimeModelCatalog
from discordbot.utils.model_pricing import get_token_rates, get_supported_modalities
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.cogs._economy.database import credit_with_repayment
from discordbot.cogs._gen_reply.prompts import (
    IMAGE_PROMPT,
    REPLY_PROMPT,
    ROUTE_PROMPT,
    SUMMARY_PROMPT,
)
from discordbot.cogs._economy.presentation import currency_text
from discordbot.cogs._gen_reply.exceptions import extract_friendly_error

if TYPE_CHECKING:
    from collections.abc import Awaitable

# Gemini occasionally wraps Discord mention syntax in backticks (inline code),
# which stops Discord from rendering the actual mention. Strip those wrappers
# before sending; matches user (<@id>, <@!id>), role (<@&id>) and channel (<#id>) mentions.
_CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")

# Strips the usage_footer appended by `_handle_streaming` from bot-authored
# messages before feeding them back as `role=assistant` history. Without this,
# the model performs in-context learning on its own past footers and starts
# hallucinating fake "-# model · ⬆ ... ⬇ ... · $... · ..." lines into fresh
# replies. Anchored on the `\n\n-# ` separator plus the ⬆/⬇ token-count icons,
# which never appear together in user-authored content.
_USAGE_FOOTER_RE = re.compile(r"\n\n-#[^\n]*⬆[^\n]*⬇[^\n]*$")
_DISCORD_MESSAGE_LIMIT = 2000


class ReplyGeneratorCogs(commands.Cog):
    """Generates AI replies for Discord messages.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: The LLM client configuration loaded for reply generation.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the ReplyGeneratorCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()
        self.runtime_models = RuntimeModelCatalog()

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The cached AsyncOpenAI client instance.

        Returns:
            A configured AsyncOpenAI client reused across reply requests.
        """
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return client

    async def _get_user_prompt(self, content: str) -> str:
        """Removes the bot mention from the content and strips whitespace."""
        if self.bot.user:
            content = content.replace(f"<@{self.bot.user.id}>", "")
        return content.strip()

    @staticmethod
    def _extract_embed_text(embeds: list[Embed]) -> str:
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

    async def _get_cleaned_content(self, message: Message) -> str:
        """Returns the textual content of a message without the author prefix."""
        content = await self._get_user_prompt(content=message.content)
        if content and self.bot.user and message.author.id == self.bot.user.id:
            content = _USAGE_FOOTER_RE.sub("", content)
        if not content and message.embeds:
            content = self._extract_embed_text(embeds=list(message.embeds))
        if not content and message.is_system():
            content = message.system_content
        return content

    async def _image_to_part(
        self, source: Attachment | StickerItem | str
    ) -> ResponseInputImageParam | None:
        """Converts an image source to a content part for the API."""
        try:
            if isinstance(source, str):
                b64_data = get_image_data(image_file=source)
                data_uri = convert_base64_to_data_uri(base64_image=b64_data)
                return ResponseInputImageParam(
                    image_url=data_uri, detail="low", type="input_image"
                )
            if isinstance(source, Attachment):
                content_type = source.content_type or guess_type(source.filename)[0] or "image/png"
            else:
                content_type = guess_type(source.url)[0] or "image/png"
            file_bytes = await source.read()
            b64_data = base64.b64encode(file_bytes).decode("utf-8")
            data_uri = f"data:{content_type};base64,{b64_data}"
            return ResponseInputImageParam(image_url=data_uri, detail="low", type="input_image")
        except Exception:
            logfire.warn("Failed to convert this image")
            return None

    async def _attachment_to_part(self, attachment: Attachment) -> ResponseInputFileParam | None:
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
    def _required_modality(content_type: str) -> Literal["image", "video", "audio", "unknown"]:
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

    async def _get_attachment_parts(
        self, message: Message
    ) -> list[ResponseInputImageParam | ResponseInputFileParam]:
        """Extracts attachment content parts from a message."""
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

    async def _process_single_message(self, message: Message) -> EasyInputMessageParam:
        """Processes a single Discord message into a Responses API input message."""
        try:
            content = await self._get_cleaned_content(message=message)
            attachment_parts = await self._get_attachment_parts(message=message)
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

    async def _get_history_message(
        self, message: Message, limit: int
    ) -> list[EasyInputMessageParam]:
        """Retrieves and processes channel history as context."""
        messages: list[EasyInputMessageParam] = []
        hist_messages: list[Message] = []
        async for m in message.channel.history(limit=limit, before=message, oldest_first=True):
            hist_messages.append(m)

        if hist_messages:
            tasks: list[Awaitable[EasyInputMessageParam]] = []
            for hist_msg in hist_messages:
                task = self._process_single_message(message=hist_msg)
                tasks.append(task)
            processed: list[EasyInputMessageParam] = await asyncio.gather(*tasks)

            messages.append(
                EasyInputMessageParam(
                    role="system",
                    content=[
                        ResponseInputTextParam(
                            text="==== Chat History that might be helpful for answering. ====",
                            type="input_text",
                        )
                    ],
                )
            )
            messages.extend(processed)

        return messages

    async def _get_reference_message(self, message: Message) -> list[EasyInputMessageParam]:
        """Walks the reference chain up to depth 3 and renders each link as context."""
        chain: list[Message] = []
        visited: set[int] = {message.id}
        current = message
        while (
            len(chain) < 3
            and current.reference
            and isinstance(current.reference.resolved, Message)
            and current.reference.resolved.id not in visited
        ):
            ref = current.reference.resolved
            visited.add(ref.id)
            chain.append(ref)
            current = ref

        if not chain:
            return []

        tasks: list[Awaitable[EasyInputMessageParam]] = []
        for ref in chain:
            task = self._process_single_message(message=ref)
            tasks.append(task)
        processed: list[EasyInputMessageParam] = await asyncio.gather(*tasks)

        messages: list[EasyInputMessageParam] = []
        for ref, processed_ref in zip(reversed(chain), reversed(processed), strict=True):
            messages.append(
                EasyInputMessageParam(
                    role="system",
                    content=[
                        ResponseInputTextParam(
                            text=(
                                f"==== Reference Message from {ref.author.display_name} "
                                f"({ref.author.name}) [id: {ref.author.id}] that might be helpful "
                                "for answering. ===="
                            ),
                            type="input_text",
                        )
                    ],
                )
            )
            messages.append(processed_ref)
        return messages

    async def _get_current_message(self, message: Message) -> list[EasyInputMessageParam]:
        """Processes the current message that needs to be answered."""
        messages: list[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role="system",
                content=[
                    ResponseInputTextParam(
                        text=f"==== Current Message that needs to be answered from {message.author.display_name} ({message.author.name}) [id: {message.author.id}]. ====",
                        type="input_text",
                    )
                ],
            )
        ]
        current_msg = await self._process_single_message(message=message)
        messages.append(current_msg)
        return messages

    async def _handle_video_reply(self, message: Message, user_prompt: str) -> None:
        """Handles video generation requests."""
        video_model = self.runtime_models.video_model
        video = await self.client.videos.create(
            model=video_model.name,
            prompt=user_prompt,
            extra_headers={"x-litellm-end-user-id": message.author.name},
        )
        while video.status not in ("completed", "failed"):
            await asyncio.sleep(5)
            video = await self.client.videos.retrieve(
                video_id=video.id, extra_headers={"x-litellm-end-user-id": message.author.name}
            )
        if video.status != "completed":
            raise RuntimeError(f"Video generation failed: {video.error}")
        video_content = await self.client.videos.download_content(
            video_id=video.id, extra_headers={"x-litellm-end-user-id": message.author.name}
        )
        video_file = File(fp=BytesIO(video_content.content), filename="generated.mp4")
        await message.reply(content=f"{message.author.mention}", file=video_file)

    async def _handle_image_reply(self, message: Message, user_prompt: str) -> None:
        """Handles image generation or editing requests."""
        image_model = self.runtime_models.image_model
        if message.reference and isinstance(message.reference.resolved, Message):
            own_parts, ref_parts = await asyncio.gather(
                self._get_attachment_parts(message=message),
                self._get_attachment_parts(message=message.reference.resolved),
            )
            attachment_parts = own_parts + ref_parts
        else:
            attachment_parts = await self._get_attachment_parts(message=message)

        data_uris: list[str] = []
        for part in attachment_parts:
            if part.get("type") == "input_image" and (image_url := part.get("image_url")):
                data_uris.append(image_url)

        if data_uris:
            image_bytes_list: list[bytes] = []
            for uri in data_uris:
                image_data = get_image_data(image_file=uri, use_b64=False)
                image_bytes_list.append(image_data)
            result = await self.client.images.edit(
                image=image_bytes_list,
                prompt=user_prompt,
                model=image_model.name,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )
        else:
            result = await self.client.images.generate(
                prompt=user_prompt,
                model=image_model.name,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )

        if not result.data:
            raise ValueError("Image operation returned no results")
        image_b64 = result.data[0].b64_json
        if image_b64 is None:
            raise ValueError("Image operation returned no b64_json")
        image_url = convert_base64_to_data_uri(image_b64)
        image_description_input: list[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(
                        text="Describe this generated image briefly for the Discord reply.",
                        type="input_text",
                    ),
                    ResponseInputImageParam(
                        image_url=image_url, detail="auto", type="input_image"
                    ),
                ],
            )
        ]
        fast_model = self.runtime_models.fast_model
        image_responses = await self.client.responses.create(
            model=fast_model.name,
            instructions=IMAGE_PROMPT,
            input=cast("ResponseInputParam", image_description_input),
            reasoning=fast_model.reasoning,
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"mock_testing_fallbacks": False},
        )
        image_description = (image_responses.output_text or "").strip()
        image_bytes = BytesIO(base64.b64decode(image_b64))
        image_file = File(fp=image_bytes, filename="generated.png")
        final_content = f"{message.author.mention} {image_description}"
        await message.reply(content=final_content, file=image_file)

    async def _handle_reaction(
        self, message: Message, emoji: str, previous: str | None = None
    ) -> str:
        """Handles adding and removing reactions on a message."""
        if previous and self.bot.user:
            with contextlib.suppress(Exception):
                await message.remove_reaction(emoji=previous, member=self.bot.user)
        with contextlib.suppress(Exception):
            await message.add_reaction(emoji=emoji)
        return emoji

    async def _route_message(self, message: Message) -> Literal["IMAGE", "QA", "SUMMARY", "VIDEO"]:
        """Routes the message to the appropriate handler."""
        message_list: list[EasyInputMessageParam] = []

        reference_messages, current_message = await asyncio.gather(
            self._get_reference_message(message=message),
            self._get_current_message(message=message),
        )
        message_list.extend(reference_messages)
        message_list.extend(current_message)

        try:
            fast_model = self.runtime_models.fast_model
            responses = await self.client.responses.parse(
                model=fast_model.name,
                instructions=ROUTE_PROMPT,
                input=cast("ResponseInputParam", message_list),
                text_format=RouteDecision,
                reasoning=fast_model.reasoning,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
                extra_body={"mock_testing_fallbacks": False},
            )
            if responses.output_parsed is None:
                return "QA"
            return responses.output_parsed.decision
        except ValidationError:
            # The model returned no text output (e.g. safety filter, empty response);
            # model_validate_json(None) raises ValidationError before we can inspect output_parsed.
            logfire.warn("RouteDecision parse failed, model returned no text; defaulting to QA")
            return "QA"

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

    async def _handle_streaming(  # noqa: C901 -- dispatches on multiple Responses API stream event types
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
            await self._handle_reaction(message=message, emoji="🌐")

        return stored_content

    async def _handle_message_reply(
        self, message: Message, system_prompt: str, history_limit: int
    ) -> None:
        """Handles generating text replies using history and context."""
        message_list: list[EasyInputMessageParam] = []

        hist_messages, reference_messages, current_message = await asyncio.gather(
            self._get_history_message(message=message, limit=history_limit),
            self._get_reference_message(message=message),
            self._get_current_message(message=message),
        )
        message_list.extend(hist_messages)
        message_list.extend(reference_messages)
        message_list.extend(current_message)

        slow_model = self.runtime_models.slow_model
        responses = await self.client.responses.create(
            model=slow_model.name,
            instructions=system_prompt,
            input=cast("ResponseInputParam", message_list),
            reasoning=slow_model.reasoning,
            tools=slow_model.tools,
            stream=True,
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"mock_testing_fallbacks": False},
        )

        await self._handle_streaming(responses=responses, message=message)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and handles AI reply generation.

        Args:
            message: The message that was sent.
        """
        # Ignore messages from bots.
        if message.author.bot:
            return

        # In DMs, always respond. In guilds, only respond when explicitly mentioned.
        # A Discord reply-notification puts the bot in message.mentions without
        # writing <@ID> into the content, so we check content to avoid
        # triggering on messages that merely reply to a functional bot post
        # (e.g. a Threads embed or a video download result).
        is_dm = message.guild is None
        if not is_dm and (not self.bot.user or f"<@{self.bot.user.id}>" not in message.content):
            return

        user_prompt = await self._get_user_prompt(content=message.content)
        has_attachment = bool(message.attachments or message.stickers)

        if not user_prompt and not has_attachment:
            await self._handle_reaction(message=message, emoji="❓")
            await message.reply(content="?")
            return

        try:
            current_emoji = await self._handle_reaction(message=message, emoji="🔀")
            route = await self._route_message(message=message)
            if route == "IMAGE":
                current_emoji = await self._handle_reaction(
                    message=message, emoji="🎨", previous=current_emoji
                )
                await self._handle_image_reply(message=message, user_prompt=user_prompt)
            elif route == "VIDEO":
                current_emoji = await self._handle_reaction(
                    message=message, emoji="🎬", previous=current_emoji
                )
                await self._handle_video_reply(message=message, user_prompt=user_prompt)
            elif route == "SUMMARY":
                current_emoji = await self._handle_reaction(
                    message=message, emoji="📖", previous=current_emoji
                )
                await self._handle_message_reply(
                    message=message, system_prompt=SUMMARY_PROMPT, history_limit=50
                )
            else:
                current_emoji = await self._handle_reaction(
                    message=message, emoji="❓", previous=current_emoji
                )
                await self._handle_message_reply(
                    message=message, system_prompt=REPLY_PROMPT, history_limit=30
                )
            await self._handle_reaction(message=message, emoji="🆗", previous=current_emoji)
        except Exception as e:
            logfire.error("Failed to generate reply", user_id=message.author.name, _exc_info=True)
            with contextlib.suppress(Exception):
                await self._handle_reaction(message=message, emoji="❌")
                error_embed = Embed(
                    title="Something went wrong",
                    description=f"```\n{extract_friendly_error(exc=e)}\n```",
                    color=0xED4245,
                )
                error_embed.set_footer(text=type(e).__name__)
                await message.reply(
                    content=None,
                    embed=error_embed,
                    **embed_spacer_payload(embeds=[error_embed], is_edit=False, target=message),
                )


def setup(bot: commands.Bot) -> None:
    """Adds the ReplyGeneratorCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
