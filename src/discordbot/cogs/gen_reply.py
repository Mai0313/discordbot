from typing import TYPE_CHECKING, Any
import contextlib

from rich import get_console
from openai import AsyncOpenAI, AsyncStream
import logfire
from nextcord import User, Message, Interaction
from nextcord.ext import commands
from openai.types.chat import ChatCompletionChunk
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from discordbot.typings.llm import LLMConfig

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

DEFAULT_MODEL = "gemini-3.1-pro-preview"
SYSTEM_PROMPT = """
1. Your response should be clearly and shortly; give me a straight answer.
2. The response should not be too long.
3. Remember you are going to response in a Discord channel, you can use markdown to make your answer more readable.
4. If the question contains images, you can also give your answer based on the image content.
5. If you don't know the answer, just say you don't know. Don't try to make up an answer.
6. Please follow the user's language to respond, if the user is using English, please respond in English; if the user is using Traditional Chinese, please respond in Traditional Chinese.
"""

console = get_console()


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.llm_sdk = LLMConfig()

    async def _get_cleaned_content(self, message: Message) -> str:
        content = message.content
        for mention in message.mentions:
            content = content.replace(f"<@{mention.id}>", "").strip()
        if isinstance(message.author, User):
            author_name = message.author.name
        else:
            author_name = message.author.nick if message.author.nick else message.author.name
        content = f"{author_name}: {content}"
        return content

    async def _get_attachments(self, message: Message) -> list[str]:
        attachments: list[str] = []
        if message.attachments:
            attachments = [attachment.url for attachment in message.attachments]
        if message.stickers:
            attachments.extend([sticker.url for sticker in message.stickers])

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
                        "text": "The following messages are the recent conversation history",
                    }
                ],
            })

            for hist_msg in hist_messages:
                hist_message = await self._process_single_message(message=hist_msg)
                messages.append(hist_message)
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
                        "text": f"The following message is the referenced message from {message.reference.resolved.author.name}",
                    }
                ],
            })
            reference_msg = await self._process_single_message(message=message.reference.resolved)
            messages.append(reference_msg)
        return messages

    async def _get_current_message(self, message: Message) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"The following message is the new message from {message.author.name}",
                    }
                ],
            },
        ]
        current_msg = await self._process_single_message(message=message)
        messages.append(current_msg)
        return messages

    async def _build_message_list(self, message: Message) -> list[ChatCompletionMessageParam]:
        message_list: list[dict[str, Any]] = []

        reference_messages = await self._get_reference_message(message=message)
        message_list.extend(reference_messages)

        # Temp disabled, LLM perform bad on long context.
        # hist_messages = await self._get_history_message(message=message)
        # message_list.extend(hist_messages)

        current_messages = await self._get_current_message(message=message)
        message_list.extend(current_messages)
        return message_list

    async def _handle_streaming(
        self,
        target: Interaction | Message,
        stream: AsyncStream[ChatCompletionChunk],
        user_mention: str,
    ) -> str:
        stored_content = f"{user_mention} "
        counted_content = 0

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                stored_content += chunk.choices[0].delta.content
                counted_content += len(chunk.choices[0].delta.content)

                if counted_content >= 15:
                    if isinstance(target, Interaction):
                        await target.edit_original_message(content=stored_content)
                    else:
                        await target.edit(content=stored_content)
                    counted_content = 0

        # Final update to ensure complete message is displayed
        with contextlib.suppress(Exception):
            if isinstance(target, Interaction):
                await target.edit_original_message(content=stored_content)
            else:
                await target.edit(content=stored_content)

        return stored_content

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        # Ignore messages from bots and skip if not mentioned
        if message.author.bot or self.bot.user not in message.mentions:
            return

        # Build the message chain (includes current message and any references)
        message_chain = await self._build_message_list(message=message)

        # Check if the current message has any actual content
        current_message_content = message_chain[-1]["content"]
        has_text = any(
            part.get("type") == "text" and part.get("text").strip()
            for part in current_message_content
        )

        if not has_text:
            await message.reply("?")
            return

        # Start typing indicator
        async with message.channel.typing():
            # Send initial thinking message
            reply_message = await message.reply(":thinking:")

            # Get LLM response using the message chain
            client = AsyncOpenAI(base_url=self.llm_sdk.base_url, api_key=self.llm_sdk.api_key)
            tools: list[ChatCompletionToolUnionParam] = [
                {"googleSearch": {}},
                {"urlContext": {}},
                {"codeExecution": {}},
            ]
            stream = await client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=message_chain,
                reasoning_effort="none",
                tools=tools,
                stream=True,
            )

            # Handle streaming response
            await self._handle_streaming(
                target=reply_message, stream=stream, user_mention=message.author.mention
            )


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
