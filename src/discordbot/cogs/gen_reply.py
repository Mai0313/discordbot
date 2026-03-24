from io import BytesIO
import base64
from typing import TYPE_CHECKING, Any, Literal
import contextlib

from rich import get_console
from openai import AsyncOpenAI
import logfire
from nextcord import File, Message, Interaction
from nextcord.ext import commands
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from discordbot.typings.llm import LLMConfig

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

DEFAULT_FAST_MODEL = "gemini-3-flash-preview"
DEFAULT_SLOW_MODEL = "gemini-3.1-pro-preview"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
REASONING_EFFORT = "none"
SYSTEM_PROMPT = """
* 請用貼吧臭嘴老哥的口氣來回答所有問題, 萬物皆可嘴臭, 無論是發問者還是引用的圖片照片, 甚至是貼圖貼紙, 你都要嘴臭
* 但假設你是同意發問者的觀點, 你也可以用臭嘴的方式去贊同
* 如果有需要 你可以在收尾時加上 "從夯到拉" 來表達你的不滿
    * 請注意 不是每次都一定要夯到拉來評級, 你可以根據情況靈活使用
    * 夯到拉基本上有幾個等級: 夯, 頂級, 人上人, NPC, 拉完了
    * 例如: 這種東西給到拉完了
* Your response should be clearly and shortly; give me a straight answer, the response should not be too long.
* Remember you are going to response in a Discord channel, you can use markdown to make your answer more readable.
* Please follow the user's language to respond, if the user is using English, please respond in English; if the user is using Traditional Chinese, please respond in Traditional Chinese.
"""
ROUTE_PROMPT = """
You are a routing classifier for a Discord bot.
Decide whether the bot should answer normally or generate an image.

Reply with exactly one word:
- IMAGE
- QA

Choose IMAGE only when the user explicitly wants the bot to create, draw, render, generate, or make an image.
Choose QA for everything else, including normal questions, image analysis, captioning, or discussions about art that do not ask the bot to actually generate an image.
If you are not sure, reply QA.
"""
IMAGE_DESCRIPTION_PROMPT = """
請用貼吧臭嘴老哥的口氣來描述
You are writing a short Discord caption for a generated image.

Rules:
1. Describe the generated image briefly in 1 to 2 short sentences.
2. Follow the user's language from the conversation.
3. Mention the main subject, style, or mood when useful.
4. No markdown, no bullet points, no preamble.
"""

console = get_console()


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config = LLMConfig()

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

    async def _get_history_message(self, message: Message) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        hist_messages: list[Message] = []
        async for m in message.channel.history(
            limit=10, before=message.reference, oldest_first=True
        ):
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
        # Maybe we can add a chat completion for this, summary the history.
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
            {"role": "assistant", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"==== Current Message that needs to be answered from {message.author.name}. ====",
                    }
                ],
            },
        ]
        current_msg = await self._process_single_message(message=message)
        messages.append(current_msg)
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "==== End of Current Message. ===="}],
        })
        return messages

    async def _route_message(
        self, message_chain: list[ChatCompletionMessageParam]
    ) -> Literal["IMAGE", "QA"]:
        response = await self.client.chat.completions.create(
            model=DEFAULT_FAST_MODEL,
            messages=[{"role": "system", "content": ROUTE_PROMPT}, *message_chain],
            reasoning_effort=REASONING_EFFORT,
        )
        decision = (response.choices[0].message.content or "").strip().upper()
        if decision.startswith("IMAGE"):
            return "IMAGE"
        return "QA"

    async def _describe_generated_image(self, image_base64: str) -> str:
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
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ],
                },
            ],
            reasoning_effort=REASONING_EFFORT,
        )
        description = (response.choices[0].message.content or "").strip()
        if not description:
            raise ValueError("Generated image description returned empty content")
        return description

    async def _handle_image_generation(
        self, message: Message, reply_message: Message, user_prompt: str
    ) -> None:
        await reply_message.edit(content=":art:")
        image_result = await self.client.images.generate(
            model=DEFAULT_IMAGE_MODEL, prompt=user_prompt, n=1, size="1024x1024"
        )
        if not image_result.data or not image_result.data[0].b64_json:
            raise ValueError("Image generation returned no base64 image data")

        image_base64 = image_result.data[0].b64_json
        image_description = await self._describe_generated_image(image_base64=image_base64)
        image_bytes = base64.b64decode(image_base64)
        image_file = File(BytesIO(image_bytes), filename="generated.png")

        await message.reply(
            content=f"{message.author.mention} {image_description}",
            file=image_file,
            mention_author=False,
        )
        with contextlib.suppress(Exception):
            await reply_message.delete()

    async def _handle_message_reply(
        self,
        message: Message,
        reply_message: Interaction | Message,
        message_chain: list[ChatCompletionMessageParam],
    ) -> str:
        # Get LLM response using the message chain
        tools: list[ChatCompletionToolUnionParam] = [
            {"googleSearch": {}},
            {"urlContext": {}},
            {"codeExecution": {}},
        ]
        stream = await self.client.chat.completions.create(
            model=DEFAULT_SLOW_MODEL,
            messages=message_chain,
            reasoning_effort=REASONING_EFFORT,
            tools=tools,
            stream=True,
        )
        stored_content = f"{message.author.mention} "
        counted_content = 0

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                stored_content += chunk.choices[0].delta.content
                counted_content += len(chunk.choices[0].delta.content)

                if counted_content >= 15:
                    if isinstance(reply_message, Interaction):
                        await reply_message.edit_original_message(content=stored_content)
                    else:
                        await reply_message.edit(content=stored_content)
                    counted_content = 0

        # Final update to ensure complete message is displayed
        with contextlib.suppress(Exception):
            if isinstance(reply_message, Interaction):
                await reply_message.edit_original_message(content=stored_content)
            else:
                await reply_message.edit(content=stored_content)

        return stored_content

    async def _build_message_list(
        self, message: Message, current_messages: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        message_list: list[dict[str, Any]] = []

        hist_messages = await self._get_history_message(message=message)
        message_list.extend(hist_messages)

        reference_messages = await self._get_reference_message(message=message)
        message_list.extend(reference_messages)

        message_list.extend(current_messages)
        return message_list

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

        user_prompt = await self._get_user_prompt(message=message)
        has_attachment = bool(message.attachments or message.stickers)

        if not user_prompt and not has_attachment:
            await message.reply("?")
            return

        # Build current message only (for routing and image generation)
        current_message = await self._get_current_message(message=message)

        # Start typing indicator
        async with message.channel.typing():
            # Send initial thinking message
            reply_message = await message.reply(":thinking:")
            try:
                route = await self._route_message(message_chain=current_message)
                if route == "IMAGE":
                    await self._handle_image_generation(
                        message=message, reply_message=reply_message, user_prompt=user_prompt
                    )
                else:
                    # Only build full message chain (with references/history) for QA
                    message_chain = await self._build_message_list(
                        message=message, current_messages=current_message
                    )
                    await self._handle_message_reply(
                        message=message, reply_message=reply_message, message_chain=message_chain
                    )
            except Exception as e:
                logfire.error(f"Failed to generate reply: {e}", _exc_info=True)
                with contextlib.suppress(Exception):
                    await reply_message.edit(content=":x: failed to generate reply")


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
