from io import BytesIO
import base64
from typing import Any, Literal
import asyncio
from functools import cached_property
from mimetypes import guess_type
import contextlib

from PIL import Image
from openai import AsyncOpenAI, AsyncStream
from litellm import model_cost
import logfire
from nextcord import File, Embed, Message, Attachment, StickerItem
from nextcord.ext import commands
from autogen.agentchat.contrib.img_utils import (
    get_pil_image,
    get_image_data,
    convert_base64_to_data_uri,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

from discordbot.typings.llm import LLMConfig

from ._gen_reply.prompts import IMAGE_PROMPT, REPLY_PROMPT, ROUTE_PROMPT, SUMMARY_PROMPT

DEFAULT_FAST_MODEL = "gemini-3-flash-preview"
DEFAULT_SLOW_MODEL = "gemini-3.1-pro-preview"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_VIDEO_MODEL = "veo-3.1-fast-generate-preview"


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config = LLMConfig()

    @cached_property
    def client(self) -> AsyncOpenAI:
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return client

    def get_tools(self, model: str) -> list[ChatCompletionToolUnionParam]:
        if "gemini" in model:
            return [{"googleSearch": {}}, {"urlContext": {}}]
        if "claude" in model:
            return [
                {"type": "web_search_20260209", "name": "web_search"},
                {"type": "web_fetch_20260209", "name": "web_fetch"},
            ]
        return [{"type": "web_search"}]

    async def _get_user_prompt(self, content: str) -> str:
        if self.bot.user:
            content = content.replace(f"<@{self.bot.user.id}>", "")
        return content.strip()

    async def _get_cleaned_content(self, message: Message) -> str:
        content = await self._get_user_prompt(content=message.content)
        if not content and message.embeds:
            embed_parts: list[str] = []
            for embed in message.embeds:
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
            content = "\n\n".join(embed_parts)
        content = f"{message.author.display_name} ({message.author.name}) [id: {message.author.id}]: {content}"
        return content

    async def _image_to_part(self, source: Attachment | StickerItem | str) -> dict[str, Any]:
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
            return {"type": "image_url", "image_url": {"url": converted}}
        except Exception:
            logfire.warn(f"Failed to convert image, keeping original URL: {url}")
            return {"type": "text", "text": f"Attachment URL: {url}"}

    async def _video_attachment_to_part(self, attachment: Attachment) -> dict[str, Any]:
        try:
            video_bytes = await attachment.read()
            b64_data = base64.b64encode(video_bytes).decode()
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            mime_type = content_type.split(";")[0].strip()
            data_uri = f"data:{mime_type};base64,{b64_data}"
            return {
                "type": "file",
                "file": {"filename": attachment.filename, "file_data": data_uri},
            }
        except Exception:
            logfire.warn(f"Failed to download video attachment: {attachment.url}")
            return {"type": "text", "text": f"Attachment URL: {attachment.url}"}

    async def _get_attachments(self, message: Message) -> list[dict[str, Any]]:
        content_parts: list[dict[str, Any]] = []

        for attachment in message.attachments:
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            if content_type.startswith("video/"):
                # Temporarily skip video attachments in content parts since they can be large and cause issues.
                # content_parts.append(await self._video_attachment_to_part(attachment=attachment))
                pass
            else:
                content_parts.append(await self._image_to_part(source=attachment))

        for sticker in message.stickers:
            content_parts.append(await self._image_to_part(source=sticker))

        # Prefer Discord's proxy_url (media.discordapp.net) over the original URL,
        # since sources like Threads CDN expire and reject requests without specific headers.
        for embed in message.embeds:
            if embed.image and (url := embed.image.proxy_url or embed.image.url):
                content_parts.append(await self._image_to_part(source=url))
            if embed.thumbnail and (url := embed.thumbnail.proxy_url or embed.thumbnail.url):
                content_parts.append(await self._image_to_part(source=url))

        return content_parts

    async def _process_single_message(self, message: Message) -> dict[str, Any]:
        content = await self._get_cleaned_content(message=message)
        role = "assistant" if self.bot.user and message.author.id == self.bot.user.id else "user"

        # Build content parts - start with text
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": content}]

        # Include all attachments (images, videos, stickers, embed images)
        attachment_parts = await self._get_attachments(message=message)
        content_parts.extend(attachment_parts)
        return {"role": role, "content": content_parts}

    async def _get_history_message(self, message: Message, limit: int) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        hist_messages: list[Message] = []
        async for m in message.channel.history(limit=limit, before=message, oldest_first=True):
            hist_messages.append(m)

        if hist_messages:
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "==== Chat History that might be helpful for answering. ====",
                    }
                ],
            })

            for hist_msg in hist_messages:
                hist_message = await self._process_single_message(message=hist_msg)
                messages.append(hist_message)
        return messages

    async def _get_reference_message(self, message: Message) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if message.reference and isinstance(message.reference.resolved, Message):
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"==== Reference Message from {message.author.display_name} ({message.author.name}) [id: {message.author.id}] that might be helpful for answering. ====",
                    }
                ],
            })
            reference_msg = await self._process_single_message(message=message.reference.resolved)
            messages.append(reference_msg)
        return messages

    async def _get_current_message(self, message: Message) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"==== Current Message that needs to be answered from {message.author.display_name} ({message.author.name}) [id: {message.author.id}]. ====",
                    }
                ],
            }
        ]
        current_msg = await self._process_single_message(message=message)
        messages.append(current_msg)
        return messages

    async def _handle_video_generation(self, message: Message, user_prompt: str) -> None:
        video = await self.client.videos.create(
            model=DEFAULT_VIDEO_MODEL,
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
        attachment_parts = await self._get_attachments(message=message)
        if message.reference and isinstance(message.reference.resolved, Message):
            ref_attachment_parts = await self._get_attachments(message=message.reference.resolved)
            attachment_parts.extend(ref_attachment_parts)

        data_uris = [
            part["image_url"]["url"]
            for part in attachment_parts
            if part.get("type") == "image_url"
        ]

        if data_uris:
            image_bytes_list: list[bytes] = []
            for uri in data_uris:
                image_data = get_image_data(image_file=uri, use_b64=False)
                image_bytes_list.append(image_data)
            result = await self.client.images.edit(
                image=image_bytes_list,
                prompt=user_prompt,
                model=DEFAULT_IMAGE_MODEL,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )
        else:
            result = await self.client.images.generate(
                prompt=user_prompt,
                model=DEFAULT_IMAGE_MODEL,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )

        if not result.data:
            raise ValueError("Image operation returned no results")
        image_url = convert_base64_to_data_uri(result.data[0].b64_json)
        image_responses = await self.client.chat.completions.create(
            model=DEFAULT_FAST_MODEL,
            messages=[
                {"role": "system", "content": IMAGE_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this generated image briefly for the Discord reply.",
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            reasoning_effort="none",
            service_tier="priority",
            extra_headers={"x-litellm-end-user-id": message.author.name},
        )
        image_description = (image_responses.choices[0].message.content or "").strip()
        image_bytes = BytesIO(base64.b64decode(result.data[0].b64_json))
        image_file = File(fp=image_bytes, filename="generated.png")

        await message.reply(
            content=f"{message.author.mention} {image_description}", file=image_file
        )

    async def _handle_reaction(
        self, message: Message, emoji: str, previous_emoji: str | None = None
    ) -> None:
        if previous_emoji and self.bot.user:
            with contextlib.suppress(Exception):
                await message.remove_reaction(emoji=previous_emoji, member=self.bot.user)
        with contextlib.suppress(Exception):
            await message.add_reaction(emoji=emoji)

    async def _route_message(self, message: Message) -> Literal["IMAGE", "QA", "SUMMARY", "VIDEO"]:
        message_list: list[dict[str, Any]] = [{"role": "system", "content": ROUTE_PROMPT}]

        reference_messages = await self._get_reference_message(message=message)
        message_list.extend(reference_messages)

        current_message = await self._get_current_message(message=message)
        message_list.extend(current_message)

        response = await self.client.chat.completions.create(
            model=DEFAULT_FAST_MODEL,
            messages=message_list,
            reasoning_effort="none",
            service_tier="priority",
            extra_headers={"x-litellm-end-user-id": message.author.name},
        )
        decision = (response.choices[0].message.content or "").strip().upper()
        if decision.startswith("IMAGE"):
            return "IMAGE"
        if decision.startswith("VIDEO"):
            return "VIDEO"
        if decision.startswith("SUMMARY"):
            return "SUMMARY"
        return "QA"

    @staticmethod
    def _calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
        info = model_cost.get(model_name) or {}
        default_input_rate = info.get("input_cost_per_token", 0)
        input_rate = info.get("input_cost_per_token_priority", default_input_rate)
        default_output_rate = info.get("output_cost_per_token", 0)
        output_rate = info.get("output_cost_per_token_priority", default_output_rate)
        return float(input_rate) * input_tokens + float(output_rate) * output_tokens

    async def _handle_streaming(
        self, stream: AsyncStream[ChatCompletionChunk], message: Message
    ) -> str:
        stored_content = ""
        counted_content = 0
        reply: Message | None = None
        content_started = False
        model_name = ""
        input_tokens = 0
        output_tokens = 0

        async for chunk in stream:
            if not model_name and chunk.model:
                model_name = chunk.model
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                if not content_started:
                    delta = delta.lstrip("\n")
                    if not delta:
                        continue
                    content_started = True
                stored_content += delta
                counted_content += len(delta)

                if counted_content >= 30:
                    if reply is None:
                        reply = await message.reply(content=stored_content)
                    else:
                        await reply.edit(content=stored_content)
                    counted_content = 0

        cost = self._calculate_cost(
            model_name=model_name, input_tokens=input_tokens, output_tokens=output_tokens
        )

        usage_footer = f"\n> **{model_name}**\n⬆ {input_tokens:,} ⬇ {output_tokens:,} ${cost:.8f}"
        stored_content += usage_footer

        # Final update to ensure complete message is displayed
        if reply is None:
            await message.reply(content=stored_content)
        else:
            with contextlib.suppress(Exception):
                await reply.edit(content=stored_content)

        return stored_content

    async def _handle_message_reply(
        self, message: Message, system_prompt: str, history_limit: int
    ) -> None:
        message_list: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]

        hist_messages = await self._get_history_message(message=message, limit=history_limit)
        message_list.extend(hist_messages)

        reference_messages = await self._get_reference_message(message=message)
        message_list.extend(reference_messages)

        current_message = await self._get_current_message(message=message)
        message_list.extend(current_message)

        tools = self.get_tools(model=DEFAULT_SLOW_MODEL)
        stream: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(
            model=DEFAULT_SLOW_MODEL,
            messages=message_list,
            reasoning_effort="high",
            tools=tools,
            stream=True,
            stream_options={"include_usage": True},
            service_tier="priority",
            extra_headers={"x-litellm-end-user-id": message.author.name},
        )

        await self._handle_streaming(stream=stream, message=message)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
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

        current_emoji = "🤔"
        await self._handle_reaction(message=message, emoji=current_emoji)
        user_prompt = await self._get_user_prompt(content=message.content)
        has_attachment = bool(message.attachments or message.stickers)

        if not user_prompt and not has_attachment:
            await self._handle_reaction(message=message, emoji="🆗", previous_emoji=current_emoji)
            await message.reply(content="?")
            return

        try:
            await self._handle_reaction(message=message, emoji="🔀", previous_emoji=current_emoji)
            current_emoji = "🔀"
            route = await self._route_message(message=message)
            if route == "IMAGE":
                await self._handle_reaction(
                    message=message, emoji="🎨", previous_emoji=current_emoji
                )
                current_emoji = "🎨"
                await self._handle_image_reply(message=message, user_prompt=user_prompt)
            elif route == "VIDEO":
                await self._handle_reaction(
                    message=message, emoji="🎬", previous_emoji=current_emoji
                )
                current_emoji = "🎬"
                await self._handle_video_generation(message=message, user_prompt=user_prompt)
            elif route == "SUMMARY":
                await self._handle_reaction(
                    message=message, emoji="📖", previous_emoji=current_emoji
                )
                current_emoji = "📖"
                await self._handle_message_reply(
                    message=message, system_prompt=SUMMARY_PROMPT, history_limit=100
                )
            else:
                await self._handle_reaction(
                    message=message, emoji="❓", previous_emoji=current_emoji
                )
                current_emoji = "❓"
                await self._handle_message_reply(
                    message=message, system_prompt=REPLY_PROMPT, history_limit=30
                )
            await self._handle_reaction(message=message, emoji="🆗", previous_emoji=current_emoji)
        except Exception as e:
            logfire.error(f"Failed to generate reply: {e}", _exc_info=True)
            with contextlib.suppress(Exception):
                await self._handle_reaction(
                    message=message, emoji="❌", previous_emoji=current_emoji
                )
                error_embed = Embed(
                    title="Something went wrong", description=f"```\n{e}\n```", color=0xED4245
                )
                error_embed.set_footer(text=type(e).__name__)
                await message.reply(content=None, embed=error_embed)


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
