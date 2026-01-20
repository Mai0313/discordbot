from typing import Any
import contextlib

from openai import AsyncStream
import logfire
from nextcord import Message, Interaction
from nextcord.ext import commands
from openai.types.chat import ChatCompletionChunk

from discordbot.sdk.llm import LLMSDK

available_models = ["openrouter/x-ai/grok-4.1-fast"]
MODEL_CHOICES = {"grok-4.1-fast": "openrouter/x-ai/grok-4.1-fast"}
DEFAULT_MODEL = available_models[0]


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

        Returns:
            A list of base64 data URI strings.
        """
        attachments: list[str] = []
        if message.attachments:
            attachments = [attachment.url for attachment in message.attachments]
        if message.stickers:
            attachments.extend([sticker.url for sticker in message.stickers])
        return attachments

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
        content = f"{message.author.name}: {content}"
        return content

    async def _process_single_message(
        self, message: Message, role: str = "user"
    ) -> dict[str, Any]:
        """Process a single message into LLM message format.

        This is the unified method to process any Discord message into the format
        expected by LLM APIs. All message processing should go through this method
        to ensure consistency.

        Args:
            message: The Discord message to process.
            role: The role of the message sender (default: "user").

        Returns:
            A dictionary in LLM message format with role and content.
        """
        content = await self._get_cleaned_content(message=message)
        attachments = await self._get_attachments(message=message)

        # Build content parts
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
        for attachment in attachments:
            content_parts.append({"type": "image_url", "image_url": {"url": attachment}})

        return {"role": role, "content": content_parts}

    async def _build_message_chain(self, message: Message) -> list[dict[str, Any]]:
        """Build a chain of messages including references and current message.

        This method handles the message chain construction, including:
        - Referenced messages (if any)
        - Current message
        - (Future) Historical messages from conversation context

        Args:
            message: The current Discord message.

        Returns:
            A list of LLM messages in chronological order.
        """
        messages: list[dict[str, Any]] = []

        # Handle referenced message
        if message.reference and isinstance(message.reference.resolved, Message):
            ref_msg = message.reference.resolved

            # Add separator to indicate referenced message
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"The following message is the referenced message from {ref_msg.author.name}"}
                ],
            })

            # Process referenced message
            if ref_msg.author.bot:
                # Bot's previous response
                reference_msg = await self._process_single_message(ref_msg, role="assistant")
            else:
                # Another user's message
                reference_msg = await self._process_single_message(ref_msg, role="user")

            messages.append(reference_msg)

            # Add separator to indicate current user's reply
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"The following message is the new message from {message.author.name}"}],
            })

            # Then add current message
            current_msg = await self._process_single_message(message, role="user")
            messages.append(current_msg)
        else:
            # No reference, just process current message
            current_msg = await self._process_single_message(message, role="user")
            messages.append(current_msg)

        # TODO: Add historical messages from conversation context here in the future
        # When adding history, make sure to:
        # - Determine correct role for each message (user vs assistant)
        # - Maintain chronological order
        # - Handle message deduplication if reference is in history

        print(messages)

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
                # Get LLM response using the message chain
                stream: ChatCompletionChunk = await self.llm_sdk.client.chat.completions.create(
                    model=DEFAULT_MODEL, messages=message_chain, stream=True
                )

                # Send initial thinking message
                reply_message = await message.reply(":thinking:")

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
