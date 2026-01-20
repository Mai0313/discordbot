from typing import Any
import contextlib

from openai import AsyncStream
import logfire
import nextcord
from nextcord import Locale, Interaction
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
        # 儲存每個用戶的對話紀錄
        # key: user_id, value: list of message dicts
        self.user_memory: dict[int, list[dict[str, Any]]] = {}
        self.llm_sdk: LLMSDK = LLMSDK(model=DEFAULT_MODEL)

    async def _get_attachment_list(
        self, messages: list[nextcord.Message] | None = None
    ) -> list[str]:
        """Retrieve all attachments from a message.

        This function extracts image attachment URLs, embed descriptions, and converts sticker images to base64 encoded strings. If the message is None, an empty list is returned.

        Args:
            messages (Optional[list[nextcord.Message]]): The message from which to extract attachments.

        Returns:
            List[str]: A list containing the attachment URLs and base64 encoded sticker images.
        """
        if messages is None:
            messages = []
        attachments: list[str] = []
        for message in messages:
            if message.attachments:
                _attach = [attachment.url for attachment in message.attachments if attachment.url]
                if _attach:
                    attachments.extend(_attach)
            if message.embeds:
                _attach = [embed.description for embed in message.embeds if embed.description]
                if _attach:
                    attachments.extend(_attach)
            if message.stickers:
                _attach = [sticker.url for sticker in message.stickers]
                if _attach:
                    attachments.extend(_attach)
        return attachments

    async def _handle_streaming(
        self,
        target: Interaction | nextcord.Message,
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
    async def on_message(self, message: nextcord.Message) -> None:
        """Handle messages that mention the bot.

        This listener allows users to chat with the bot by mentioning it, without needing to use slash commands.

        Args:
            message (nextcord.Message): The message object.
        """
        # Ignore messages from bots and skip if not mentioned
        if message.author.bot or self.bot.user not in message.mentions:
            return

        # Extract message content without mentions
        content = message.content
        for mention in message.mentions:
            content = content.replace(f"<@{mention.id}>", "")
        content = content.strip()

        # If content is empty or only whitespace, reply with "?"
        if not content:
            await message.reply("?")
            return

        if message.author.id not in self.user_memory:
            self.user_memory[message.author.id] = []

        # Start typing indicator
        async with message.channel.typing():
            try:
                attachments = await self._get_attachment_list([message])
                completion_content = await self.llm_sdk.prepare_completion_content(
                    prompt=content, attachments=attachments
                )

                # Add user message to memory
                self.user_memory[message.author.id].append({
                    "role": "user",
                    "content": completion_content,
                })

                stream = await self.llm_sdk.client.chat.completions.create(
                    model=DEFAULT_MODEL, messages=self.user_memory[message.author.id], stream=True
                )

                # Send initial thinking message
                reply_message = await message.reply("🤔 思考中...")

                # Handle streaming response
                response_text = await self._handle_streaming(
                    target=reply_message, stream=stream, user_mention=message.author.mention
                )

                # Add AI response to memory
                if response_text:
                    self.user_memory[message.author.id].append({
                        "role": "assistant",
                        "content": response_text,
                    })

            except Exception as e:
                logfire.error("Error in on_message mention handler", _exc_info=True)
                await message.reply(f"{e}")

    @nextcord.slash_command(
        name="clear_memory",
        description="Clear your conversation memory with the bot.",
        name_localizations={Locale.zh_TW: "清除記憶", Locale.ja: "メモリをクリア"},
        description_localizations={
            Locale.zh_TW: "清除你與機器人的對話記憶。",
            Locale.ja: "ボットとの会話メモリをクリアします。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def clear_memory(self, interaction: Interaction) -> None:
        """清除用戶的對話記憶。

        Args:
            interaction (Interaction): The interaction object for the command.
        """
        user_id = interaction.user.id
        had_memory = self.user_memory.pop(user_id, None) is not None

        if had_memory:
            await interaction.response.send_message(
                content="對話記憶已清除! 下次對話將重新開始。", ephemeral=True
            )
        else:
            await interaction.response.send_message(
                content="你目前沒有對話記憶需要清除。", ephemeral=True
            )


async def setup(bot: commands.Bot) -> None:
    """Register the reply generation cog with the bot.

    Args:
        bot (commands.Bot): The bot instance to which the cog will be added.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
