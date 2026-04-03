from io import BytesIO
import time
import base64
from typing import Any, Literal
import asyncio
from mimetypes import guess_type
import contextlib

from openai import AsyncOpenAI, AsyncStream
import logfire
from nextcord import File, Embed, Message, Interaction
from nextcord.ext import commands
from autogen.agentchat.contrib.img_utils import (
    get_pil_image,
    get_image_data,
    pil_to_data_uri,
    convert_base64_to_data_uri,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

from discordbot.typings.llm import LLMConfig

from ._gen_reply.prompts import (
    ROUTE_PROMPT,
    get_system_prompt,
    get_summary_prompt,
    get_image_description_prompt,
)

DEFAULT_FAST_MODEL = "gemini-3-flash-preview"
DEFAULT_SLOW_MODEL = "gemini-3.1-pro-preview"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_VIDEO_MODEL = "veo-3.1-fast-generate-preview"
VIDEO_COOLDOWN = 10  # minutes
TOOLS: list[ChatCompletionToolUnionParam] = [
    {"googleSearch": {}},
    {"urlContext": {}},
    # {"codeExecution": {}},
]


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config = LLMConfig()
        self._video_cooldowns: dict[int, float] = {}

    @property
    def client(self) -> AsyncOpenAI:
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return client

    async def _get_user_prompt(self, message: Message) -> str:
        content = message.content
        for mention in message.mentions:
            content = content.replace(f"<@{mention.id}>", "").strip()
        return content.strip()

    async def _get_cleaned_content(self, message: Message) -> str:
        content = await self._get_user_prompt(message=message)
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
        content = f"{message.author.name}: {content}"
        return content

    async def _get_attachments(self, message: Message) -> list[dict[str, Any]]:
        content_parts: list[dict[str, Any]] = []

        # Process Discord attachments (have content_type metadata)
        for attachment in message.attachments:
            content_type = attachment.content_type or guess_type(attachment.filename)[0] or ""
            if content_type.startswith("video/"):
                try:
                    video_bytes = await attachment.read()
                    b64_data = base64.b64encode(video_bytes).decode()
                    mime_type = content_type.split(";")[0].strip()
                    data_uri = f"data:{mime_type};base64,{b64_data}"
                    content_parts.append({
                        "type": "file",
                        "file": {"filename": attachment.filename, "file_data": data_uri},
                    })
                except Exception:
                    logfire.warn("Failed to download video attachment, keeping original URL")
                    content_parts.append({
                        "type": "text",
                        "text": f"Attachment URL: {attachment.url}",
                    })
            else:
                try:
                    downloaded = get_pil_image(image_file=attachment.url)
                    converted = pil_to_data_uri(image=downloaded)
                    content_parts.append({"type": "image_url", "image_url": {"url": converted}})
                except Exception:
                    logfire.warn("Failed to convert attachment to image, keeping original URL")
                    content_parts.append({
                        "type": "text",
                        "text": f"Attachment URL: {attachment.url}",
                    })

        # Process stickers
        for sticker in message.stickers:
            try:
                downloaded = get_pil_image(image_file=sticker.url)
                converted = pil_to_data_uri(image=downloaded)
                content_parts.append({"type": "image_url", "image_url": {"url": converted}})
            except Exception:
                logfire.warn("Failed to convert sticker to image, keeping original URL")
                content_parts.append({"type": "text", "text": f"Attachment URL: {sticker.url}"})

        # Process embed images
        for embed in message.embeds:
            for url in filter(
                None,
                [
                    embed.image.url if embed.image else None,
                    embed.thumbnail.url if embed.thumbnail else None,
                ],
            ):
                try:
                    downloaded = get_pil_image(image_file=url)
                    converted = pil_to_data_uri(image=downloaded)
                    content_parts.append({"type": "image_url", "image_url": {"url": converted}})
                except Exception:
                    logfire.warn("Failed to convert embed image, keeping original URL")
                    content_parts.append({"type": "text", "text": f"Attachment URL: {url}"})

        return content_parts

    async def _process_single_message(self, message: Message) -> dict[str, Any]:
        content = await self._get_cleaned_content(message=message)
        role = "assistant" if message.author.bot else "user"

        # Build content parts - start with text
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": content}]

        # Only include images for current/recent messages, not historical ones
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
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "==== End of Chat History. ===="}],
            })
        return messages

    async def _get_reference_message(self, message: Message) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if message.reference and isinstance(message.reference.resolved, Message):
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"==== Reference Message from {message.reference.resolved.author.name} that might be helpful for answering. ====",
                    }
                ],
            })
            reference_msg = await self._process_single_message(message=message.reference.resolved)
            messages.append(reference_msg)
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "==== End of Reference Message. ===="}],
            })
        return messages

    async def _get_current_message(self, message: Message) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt()}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"==== Current Message that needs to be answered {message.author.name}. ====",
                    }
                ],
            },
        ]
        current_msg = await self._process_single_message(message=message)
        messages.append(current_msg)
        return messages

    async def _route_message(self, message: Message) -> Literal["IMAGE", "QA", "SUMMARY", "VIDEO"]:
        current_message = await self._get_current_message(message=message)
        response = await self.client.chat.completions.create(
            model=DEFAULT_FAST_MODEL,
            messages=[{"role": "system", "content": ROUTE_PROMPT}, *current_message],
            reasoning_effort="none",
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"metadata": {"tags": [message.author.name]}},
        )
        decision = (response.choices[0].message.content or "").strip().upper()
        if decision.startswith("IMAGE"):
            return "IMAGE"
        if decision.startswith("VIDEO"):
            return "VIDEO"
        if decision.startswith("SUMMARY"):
            return "SUMMARY"
        return "QA"

    async def _handle_video_generation(
        self, message: Message, reply_message: Message, user_prompt: str
    ) -> None:
        user_id = message.author.id
        if message.author.name != "mai9999":
            last_used = self._video_cooldowns.get(user_id, 0)
            cooldown_seconds = VIDEO_COOLDOWN * 60
            if time.time() - last_used < cooldown_seconds:
                remaining = int((cooldown_seconds - (time.time() - last_used)) / 60)
                minutes = remaining // 60
                with contextlib.suppress(Exception):
                    await reply_message.delete()
                await message.reply(
                    content=f"{message.author.mention} 影片生成每小時限用一次，還需等待 {minutes} 分鐘"
                )
                return

        # create_kwargs: dict[str, Any] = {"model": DEFAULT_VIDEO_MODEL, "prompt": user_prompt}
        # data_uris = await self._get_attachments(message=message)
        # if message.reference and isinstance(message.reference.resolved, Message):
        #     data_uris.extend(await self._get_attachments(message=message.reference.resolved))

        # if data_uris:
        #     create_kwargs["input_reference"] = get_image_data(image_file=data_uris[0], use_b64=False)

        video = await self.client.videos.create(
            model=DEFAULT_VIDEO_MODEL,
            prompt=user_prompt,
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"metadata": {"tags": [message.author.name]}},
        )
        while video.status not in ("completed", "failed"):
            await asyncio.sleep(5)
            await reply_message.edit(content=":hourglass:")
            video = await self.client.videos.retrieve(
                video_id=video.id,
                extra_headers={"x-litellm-end-user-id": message.author.name},
                extra_body={"metadata": {"tags": [message.author.name]}},
            )
        if video.status != "completed":
            raise RuntimeError(f"Video generation failed: {video.error}")
        video_content = await self.client.videos.download_content(
            video_id=video.id,
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"metadata": {"tags": [message.author.name]}},
        )
        video_file = File(BytesIO(video_content.content), filename="generated.mp4")
        with contextlib.suppress(Exception):
            await reply_message.delete()
        await message.reply(content=f"{message.author.mention}", file=video_file)
        self._video_cooldowns[user_id] = time.time()

    async def _handle_image_reply(
        self, message: Message, reply_message: Message, user_prompt: str
    ) -> None:
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
                extra_body={"metadata": {"tags": [message.author.name]}},
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
                extra_body={"metadata": {"tags": [message.author.name]}},
            )

        if not result.data:
            raise ValueError("Image operation returned no results")
        image_url = convert_base64_to_data_uri(result.data[0].b64_json)
        image_responses = await self.client.chat.completions.create(
            model=DEFAULT_FAST_MODEL,
            messages=[
                {"role": "system", "content": get_image_description_prompt()},
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
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"metadata": {"tags": [message.author.name]}},
        )
        image_description = (image_responses.choices[0].message.content or "").strip()
        image_bytes = BytesIO(base64.b64decode(result.data[0].b64_json))
        image_file = File(image_bytes, filename="generated.png")

        with contextlib.suppress(Exception):
            await reply_message.delete()
        await message.reply(
            content=f"{message.author.mention} {image_description}", file=image_file
        )

    async def _handle_streaming(  # noqa: PLR0912
        self,
        stream: AsyncStream[ChatCompletionChunk],
        message: Message,
        reply_message: Interaction | Message,
    ) -> str:
        # Get LLM response using the message chain
        stored_content = f"{message.author.mention} "
        counted_content = 0
        new_reply: Message | None = None
        content_started = False

        async for chunk in stream:
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
                    if new_reply is None:
                        # First content update: delete status message and send a new reply
                        with contextlib.suppress(Exception):
                            if isinstance(reply_message, Interaction):
                                await reply_message.delete_original_message()
                            else:
                                await reply_message.delete()
                        new_reply = await message.reply(content=stored_content)
                    else:
                        await new_reply.edit(content=stored_content)
                    counted_content = 0

        # Final update to ensure complete message is displayed
        if new_reply is None:
            with contextlib.suppress(Exception):
                if isinstance(reply_message, Interaction):
                    await reply_message.delete_original_message()
                else:
                    await reply_message.delete()
            await message.reply(content=stored_content)
        else:
            with contextlib.suppress(Exception):
                await new_reply.edit(content=stored_content)

        return stored_content

    async def _handle_message_reply(
        self, message: Message, reply_message: Interaction | Message
    ) -> None:
        message_list: list[dict[str, Any]] = []

        hist_messages = await self._get_history_message(message=message, limit=15)
        message_list.extend(hist_messages)

        reference_messages = await self._get_reference_message(message=message)
        message_list.extend(reference_messages)

        current_message = await self._get_current_message(message=message)
        message_list.extend(current_message)

        stream: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(
            model=DEFAULT_SLOW_MODEL,
            messages=message_list,
            reasoning_effort="medium",
            tools=TOOLS,
            stream=True,
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"metadata": {"tags": [message.author.name]}},
        )

        await self._handle_streaming(stream=stream, message=message, reply_message=reply_message)

    async def _handle_summary_reply(self, message: Message, reply_message: Message) -> None:
        hist_messages = await self._get_history_message(message=message, limit=50)
        message_list: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": get_summary_prompt()}]}
        ]
        message_list.extend(hist_messages)
        message_list.append({
            "role": "user",
            "content": [{"type": "text", "text": "請總結以上的聊天記錄。"}],
        })
        stream: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(
            model=DEFAULT_SLOW_MODEL,
            messages=message_list,
            reasoning_effort="medium",
            tools=TOOLS,
            stream=True,
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"metadata": {"tags": [message.author.name]}},
        )

        await self._handle_streaming(stream=stream, message=message, reply_message=reply_message)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        # Ignore messages from bots.
        if message.author.bot:
            return

        # Only respond when the bot is explicitly mentioned in the message text.
        # A Discord reply-notification puts the bot in message.mentions without
        # writing <@ID> into the content, so we check content to avoid
        # triggering on messages that merely reply to a functional bot post
        # (e.g. a Threads embed or a video download result).
        if not self.bot.user or f"<@{self.bot.user.id}>" not in message.content:
            return

        reply_message = await message.reply(":thinking:")
        user_prompt = await self._get_user_prompt(message=message)
        has_attachment = bool(message.attachments or message.stickers)

        if not user_prompt and not has_attachment:
            with contextlib.suppress(Exception):
                await reply_message.delete()
            await message.reply(content="?")
            return

        try:
            # Build current message only (for routing and image generation)
            await reply_message.edit(content=":twisted_rightwards_arrows:")
            route = await self._route_message(message=message)
            if route == "IMAGE":
                await reply_message.edit(content=":art:")
                await self._handle_image_reply(
                    message=message, reply_message=reply_message, user_prompt=user_prompt
                )
            elif route == "VIDEO":
                await reply_message.edit(content=":movie_camera:")
                await self._handle_video_generation(
                    message=message, reply_message=reply_message, user_prompt=user_prompt
                )
            elif route == "SUMMARY":
                await reply_message.edit(content=":book:")
                await self._handle_summary_reply(message=message, reply_message=reply_message)
            else:
                await reply_message.edit(content=":question:")
                await self._handle_message_reply(message=message, reply_message=reply_message)
        except Exception as e:
            logfire.error(f"Failed to generate reply: {e}", _exc_info=True)
            with contextlib.suppress(Exception):
                error_embed = Embed(
                    title="Something went wrong", description=f"```\n{e}\n```", color=0xED4245
                )
                error_embed.set_footer(text=type(e).__name__)
                await reply_message.delete()
                await message.reply(content=None, embed=error_embed)


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
