from io import BytesIO
import time
import base64
from typing import Any, Literal
import asyncio
import contextlib

from openai import AsyncOpenAI
import logfire
from nextcord import File, Embed, Message, Interaction
from nextcord.ext import commands
from autogen.agentchat.contrib.img_utils import (
    get_pil_image,
    get_image_data,
    pil_to_data_uri,
    convert_base64_to_data_uri,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

from discordbot.typings.llm import LLMConfig

from ._gen_reply.prompts import (
    ROUTE_PROMPT,
    SYSTEM_PROMPT,
    HISTORY_PROMPT,
    SUMMARY_PROMPT,
    IMAGE_DESCRIPTION_PROMPT,
)

DEFAULT_FAST_MODEL = "gemini-3-flash-preview"
DEFAULT_SLOW_MODEL = "gemini-3.1-pro-preview"
# DEFAULT_SLOW_MODEL = "gemini-3-flash-preview"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_VIDEO_MODEL = "veo-3.1-fast-generate-preview"
VIDEO_COOLDOWN = 10  # minutes
REASONING_EFFORT: ReasoningEffort = "none"
TOOLS: list[ChatCompletionToolUnionParam] = [
    {"googleSearch": {}},
    {"urlContext": {}},
    {"codeExecution": {}},
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

    async def _get_attachments(self, message: Message) -> list[str]:
        attachments: list[str] = []
        if message.attachments:
            attachments = [attachment.url for attachment in message.attachments]
        if message.stickers:
            attachments.extend([sticker.url for sticker in message.stickers])
        for embed in message.embeds:
            if embed.image and embed.image.url:
                attachments.append(embed.image.url)
            if embed.thumbnail and embed.thumbnail.url:
                attachments.append(embed.thumbnail.url)

        converted_attachments: list[str] = []
        for attachment in attachments:
            try:
                downloaded_attachment = get_pil_image(image_file=attachment)
                converted_attachment = pil_to_data_uri(image=downloaded_attachment)
                converted_attachments.append(converted_attachment)
            except Exception:
                logfire.warn("Filed to download the attachment")

        return converted_attachments

    async def _process_single_message(self, message: Message) -> dict[str, Any]:
        content = await self._get_cleaned_content(message=message)
        role = "assistant" if message.author.bot else "user"

        # Build content parts - start with text
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": content}]

        # Only include images for current/recent messages, not historical ones
        attachments = await self._get_attachments(message=message)
        for attachment in attachments:
            content_parts.append({"type": "image_url", "image_url": {"url": attachment}})
        return {"role": role, "content": content_parts}

    async def _get_history_message_ai(self, message: Message, limit: int) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        hist_messages: list[Message] = []
        async for m in message.channel.history(limit=limit, before=message, oldest_first=True):
            hist_messages.append(m)

        if hist_messages:
            # Build raw history for summarization
            raw_history: list[dict[str, Any]] = []
            for hist_msg in hist_messages:
                hist_message = await self._process_single_message(message=hist_msg)
                raw_history.append(hist_message)

            # Use LLM to summarize history into a clean conversation log
            summary_messages: list[dict[str, Any]] = [
                {"role": "system", "content": HISTORY_PROMPT},
                *raw_history,
                {
                    "role": "user",
                    "content": "Please summarize the above chat history into a clean conversation log.",
                },
            ]
            response = await self.client.chat.completions.create(
                model=DEFAULT_FAST_MODEL,
                messages=summary_messages,
                reasoning_effort=REASONING_EFFORT,
                tools=TOOLS,
            )
            summary = (response.choices[0].message.content or "").strip()

            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"==== Chat History Summary ====\n{summary}\n==== End of Chat History ====",
                    }
                ],
            })
        return messages

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
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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

    async def _route_message(
        self, current_message: list[ChatCompletionMessageParam]
    ) -> Literal["IMAGE", "QA", "SUMMARY", "EDIT", "VIDEO"]:
        response = await self.client.chat.completions.create(
            model=DEFAULT_FAST_MODEL,
            messages=[{"role": "system", "content": ROUTE_PROMPT}, *current_message],
            reasoning_effort=REASONING_EFFORT,
        )
        decision = (response.choices[0].message.content or "").strip().upper()
        if decision.startswith("EDIT"):
            return "EDIT"
        if decision.startswith("IMAGE"):
            return "IMAGE"
        if decision.startswith("VIDEO"):
            return "VIDEO"
        if decision.startswith("SUMMARY"):
            return "SUMMARY"
        return "QA"

    async def _handle_streaming(
        self,
        model: str,
        message: Message,
        reply_message: Interaction | Message,
        message_list: list[ChatCompletionMessageParam],
    ) -> str:
        # Get LLM response using the message chain
        stream = await self.client.chat.completions.create(
            model=model,
            messages=message_list,
            reasoning_effort=REASONING_EFFORT,
            tools=TOOLS,
            stream=True,
        )
        stored_content = f"{message.author.mention} "
        counted_content = 0
        new_reply: Message | None = None

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                stored_content += chunk.choices[0].delta.content
                counted_content += len(chunk.choices[0].delta.content)

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

        video = await self.client.videos.create(model=DEFAULT_VIDEO_MODEL, prompt=user_prompt)
        while video.status not in ("completed", "failed"):
            await asyncio.sleep(5)
            await reply_message.edit(content=":hourglass:")
            video = await self.client.videos.retrieve(video.id)
        if video.status != "completed":
            raise RuntimeError(f"Video generation failed: {video.error}")
        video_content = await self.client.videos.download_content(video.id)
        video_file = File(BytesIO(video_content.content), filename="generated.mp4")
        with contextlib.suppress(Exception):
            await reply_message.delete()
        await message.reply(content=f"{message.author.mention}", file=video_file)
        self._video_cooldowns[user_id] = time.time()

    async def _describe_generated_image(self, image_base64: str) -> str:
        image_url = convert_base64_to_data_uri(image_base64)
        response = await self.client.chat.completions.create(
            model=DEFAULT_FAST_MODEL,
            messages=[
                {"role": "system", "content": IMAGE_DESCRIPTION_PROMPT},
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
            reasoning_effort=REASONING_EFFORT,
        )
        description = (response.choices[0].message.content or "").strip()
        return description

    async def _handle_image_generation(
        self, message: Message, reply_message: Message, user_prompt: str
    ) -> None:
        image_result = await self.client.images.generate(
            model=DEFAULT_IMAGE_MODEL, prompt=user_prompt, n=1, size="auto"
        )
        if not image_result.data:
            raise ValueError("Image generation returned no results")
        image_base64 = image_result.data[0].b64_json
        image_description = await self._describe_generated_image(image_base64=image_base64)
        image_bytes = base64.b64decode(image_base64)
        image_file = File(BytesIO(image_bytes), filename="generated.png")

        with contextlib.suppress(Exception):
            await reply_message.delete()
        await message.reply(
            content=f"{message.author.mention} {image_description}", file=image_file
        )

    async def _handle_image_edit(
        self, message: Message, reply_message: Message, user_prompt: str
    ) -> None:
        data_uris = await self._get_attachments(message=message)
        if message.reference and isinstance(message.reference.resolved, Message):
            data_uris.extend(await self._get_attachments(message=message.reference.resolved))

        if not data_uris:
            await self._handle_image_generation(
                message=message, reply_message=reply_message, user_prompt=user_prompt
            )
            return

        image_bytes_list: list[bytes] = [
            get_image_data(image_file=uri, use_b64=False) for uri in data_uris
        ]
        image_input: bytes | list[bytes] = (
            image_bytes_list[0] if len(image_bytes_list) == 1 else image_bytes_list
        )
        edit_result = await self.client.images.edit(
            image=image_input, prompt=user_prompt, model=DEFAULT_IMAGE_MODEL, n=1, size="auto"
        )
        if not edit_result.data:
            raise ValueError("Image edit returned no results")
        image_base64 = edit_result.data[0].b64_json
        image_description = await self._describe_generated_image(image_base64=image_base64)
        edited_bytes = base64.b64decode(image_base64)
        image_file = File(BytesIO(edited_bytes), filename="edited.png")

        with contextlib.suppress(Exception):
            await reply_message.delete()
        await message.reply(
            content=f"{message.author.mention} {image_description}", file=image_file
        )

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

        await self._handle_streaming(
            model=DEFAULT_SLOW_MODEL,
            message=message,
            reply_message=reply_message,
            message_list=message_list,
        )

    async def _handle_summary(self, message: Message, reply_message: Message) -> None:
        hist_messages = await self._get_history_message(message=message, limit=50)
        message_list: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": SUMMARY_PROMPT}]}
        ]
        message_list.extend(hist_messages)
        message_list.append({
            "role": "user",
            "content": [{"type": "text", "text": "請總結以上的聊天記錄。"}],
        })
        await self._handle_streaming(
            model=DEFAULT_SLOW_MODEL,
            message=message,
            reply_message=reply_message,
            message_list=message_list,
        )

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

        # Start typing indicator
        async with message.channel.typing():
            # Send initial thinking message
            try:
                # Build current message only (for routing and image generation)
                current_message = await self._get_current_message(message=message)
                await reply_message.edit(content=":twisted_rightwards_arrows:")
                route = await self._route_message(current_message=current_message)
                if route == "IMAGE":
                    await reply_message.edit(content=":art:")
                    await self._handle_image_generation(
                        message=message, reply_message=reply_message, user_prompt=user_prompt
                    )
                elif route == "VIDEO":
                    await reply_message.edit(content=":movie_camera:")
                    await self._handle_video_generation(
                        message=message, reply_message=reply_message, user_prompt=user_prompt
                    )
                elif route == "EDIT":
                    await reply_message.edit(content=":paintbrush:")
                    await self._handle_image_edit(
                        message=message, reply_message=reply_message, user_prompt=user_prompt
                    )
                elif route == "SUMMARY":
                    await reply_message.edit(content=":book:")
                    await self._handle_summary(message=message, reply_message=reply_message)
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
