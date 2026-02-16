from typing import Any
import contextlib

from openai import AsyncStream
import logfire
from nextcord import User, Message, Interaction
from nextcord.ext import commands
from openai.types.chat import ChatCompletionChunk
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri

from discordbot.utils.llm import LLMSDK

MODEL_CHOICES = {"gemini-3-pro-preview": "gemini-3-pro-preview"}
DEFAULT_MODEL = "gemini-3-pro-preview"
HISTORY_LIMIT = 0  # 歷史訊息數量限制
SYSTEM_PROMPT = """
1. Your response should be clearly and shortly; give me a straight answer.
2. The response should not be too long.
3. Remember you are going to response in a Discord channel, you can use markdown to make your answer more readable.
4. If the question contains images, you can also give your answer based on the image content.
5. If you don't know the answer, just say you don't know. Don't try to make up an answer.
6. Please follow the user's language to respond, if the user is using English, please respond in English; if the user is using Traditional Chinese, please respond in Traditional Chinese.
"""


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the ReplyGeneratorCogs.

        Args:
            bot (commands.Bot): The bot instance.
        """
        self.bot = bot
        self.llm_sdk: LLMSDK = LLMSDK(model=DEFAULT_MODEL)

    async def _get_attachments(self, message: Message) -> list[str]:
        """Get all attachments from a message and convert to base64 data URIs.

        Args:
            message: The Discord message to extract attachments from.
            validate_urls: Whether to validate that URLs are accessible (HEAD request).

        Returns:
            A list of base64 data URI strings.
        """
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

    async def _get_cleaned_content(self, message: Message) -> str:
        """Clean message content by replacing user mentions with readable names.

        Args:
            message: The Discord message to clean.

        Returns:
            The cleaned message content.
        """
        content = message.content
        for mention in message.mentions:
            content = content.replace(f"<@{mention.id}>", "").strip()
        if isinstance(message.author, User):
            author_name = message.author.name
        else:
            author_name = message.author.nick if message.author.nick else message.author.name
        content = f"{author_name}: {content}"
        return content

    async def _process_single_message(
        self, message: Message, role: str = "user", include_images: bool = True
    ) -> dict[str, Any]:
        """Process a single message into LLM message format.

        This is the unified method to process any Discord message into the format
        expected by LLM APIs. All message processing should go through this method
        to ensure consistency.

        Args:
            message: The Discord message to process.
            role: The role of the message sender (default: "user").
            include_images: Whether to include images in the message. For historical messages, this should be False to avoid 404 errors.

        Returns:
            A dictionary in LLM message format with role and content.
        """
        content = await self._get_cleaned_content(message=message)

        # Build content parts - start with text
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": content}]

        # Only include images for current/recent messages, not historical ones
        if include_images:
            attachments = await self._get_attachments(message=message)
            for attachment in attachments:
                content_parts.append({"type": "image_url", "image_url": {"url": attachment}})

        return {"role": role, "content": content_parts}

    async def _build_message_chain(self, message: Message) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        # 1. 處理引用的訊息（如果有的話）
        referenced_message = None
        if message.reference and isinstance(message.reference.resolved, Message):
            referenced_message = message.reference.resolved

        # 2. 獲取歷史記錄
        # 如果有引用訊息，從引用訊息之前開始獲取歷史記錄
        if referenced_message:
            hist_messages = await message.channel.history(
                limit=HISTORY_LIMIT, before=referenced_message
            ).flatten()
            hist_messages.reverse()
        else:
            # 否則維持原來的邏輯
            hist_messages = await message.channel.history(limit=HISTORY_LIMIT).flatten()
            hist_messages.reverse()

        if hist_messages:
            # Add separator for history
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The following messages are the recent conversation history",
                    }
                ],
            })

            # Add historical messages in chronological order (oldest first)
            # Skip images for historical messages to avoid 404 errors
            for hist_msg in hist_messages:
                # Determine role based on author
                role = "assistant" if hist_msg.author.bot else "user"
                hist_message = await self._process_single_message(
                    hist_msg, role=role, include_images=False
                )
                messages.append(hist_message)

        # Add separator to indicate end of history
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The following messages are the current conversation"}
            ],
        })

        # 3. 處理引用的訊息（如果有的話）
        if referenced_message:
            # Add separator to indicate referenced message
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"The following message is the referenced message from {referenced_message.author.name}",
                    }
                ],
            })

            # Process referenced message
            if referenced_message.author.bot:
                # Bot's previous response
                reference_msg = await self._process_single_message(
                    message=referenced_message, role="assistant"
                )
            else:
                # Another user's message
                reference_msg = await self._process_single_message(
                    message=referenced_message, role="user"
                )

            messages.append(reference_msg)

        # 4. 添加當前輸入訊息
        # Add separator to indicate current user's reply (only if there was a referenced message)
        if referenced_message:
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"The following message is the new message from {message.author.name}",
                    }
                ],
            })

        # 5. 加入一些基本指導
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        })

        # Add current message
        current_msg = await self._process_single_message(message, role="user")
        messages.append(current_msg)
        return messages

    async def _handle_streaming(
        self,
        target: Interaction | Message,
        stream: AsyncStream[ChatCompletionChunk],
        user_mention: str,
    ) -> str:
        """Handle streaming LLM response for both Interaction and Message.

        Args:
            target: Either an Interaction or Message object to edit
            stream: The streaming response from LLM
            user_mention: The user mention string to prefix the response with

        Returns:
            The complete accumulated response text
        """
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
        """Handle messages that mention the bot.

        This listener allows users to chat with the bot by mentioning it, without needing to use slash commands.

        Args:
            message (Message): The message object.
        """
        # Ignore messages from bots and skip if not mentioned
        if message.author.bot or self.bot.user not in message.mentions:
            return

        # Build the message chain (includes current message and any references)
        message_chain = await self._build_message_chain(message=message)

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
            try:
                # Send initial thinking message
                reply_message = await message.reply(":thinking:")

                # Get LLM response using the message chain
                stream: ChatCompletionChunk = await self.llm_sdk.client.chat.completions.create(
                    model=DEFAULT_MODEL, messages=message_chain, stream=True
                )

                # Handle streaming response
                await self._handle_streaming(
                    target=reply_message, stream=stream, user_mention=message.author.mention
                )

            except Exception as e:
                logfire.error("Error in on_message mention handler", _exc_info=True)
                await message.reply(f"{e}")


async def setup(bot: commands.Bot) -> None:
    """Register the reply generation cog with the bot.

    Args:
        bot (commands.Bot): The bot instance to which the cog will be added.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
