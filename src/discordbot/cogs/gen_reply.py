from openai import BadRequestError
import logfire
import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.sdk.llm import LLMSDK

available_models = ["openai/gpt-5-mini", "openai/gpt-5-nano", "claude-3-5-haiku-20241022"]
MODEL_CHOICES = {available_model: available_model for available_model in available_models}


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the ReplyGeneratorCogs.

        Args:
            bot (commands.Bot): The bot instance.
        """
        self.bot = bot
        # 儲存每個用戶的上一個 response ID，用於對話記憶
        self.user_last_response_id: dict[int, str] = {}

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
                _attach = [f"data:image/png;base64,{sticker.url}" for sticker in message.stickers]
                if _attach:
                    attachments.extend(_attach)
        return attachments

    @nextcord.slash_command(
        name="oai",
        description="Generate a reply based on the given prompt.",
        name_localizations={Locale.zh_TW: "生成文字", Locale.ja: "テキストを生成"},
        description_localizations={
            Locale.zh_TW: "根據提供的提示生成回覆。",
            Locale.ja: "指定されたプロンプトに基づいて応答を生成します。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def oai(
        self,
        interaction: Interaction,
        prompt: str = SlashOption(
            description="Enter your prompt.",
            description_localizations={
                Locale.zh_TW: "請輸入提示詞。",
                Locale.ja: "プロンプトを入力してください。",
            },
        ),
        model: str = SlashOption(
            description="Choose a model (default: GPT-5).",
            description_localizations={
                Locale.zh_TW: "選擇模型 (預設為 GPT-5)",
                Locale.ja: "モデルを選択してください（デフォルトは GPT-5）",
            },
            choices=MODEL_CHOICES,
            required=False,
            default=available_models[0],
        ),
        image: nextcord.Attachment | None = SlashOption(  # noqa: B008
            description="(Optional) Upload an image.",
            description_localizations={
                Locale.zh_TW: "（可選）上傳一張圖片。",
                Locale.ja: "（オプション）画像をアップロードしてください。",
            },
            required=False,
        ),
    ) -> None:
        """Generate a reply based on the user's prompt.

        If the model 'o1' is selected along with an image, an error message is returned since 'o1' does not support image input.
        The function can either generate a complete response or stream the response in real-time based on the stream parameter.

        Args:
            interaction (Interaction): The interaction object for the command.
            prompt (str): The prompt text provided by the user.
            model (str): The selected model, defaults to "gpt-5" if not specified.
            image (Optional[nextcord.Attachment]): An optional image attachment uploaded by the user.
        """
        await interaction.response.defer()
        attachments = []
        if model not in ["o1", "o1-mini"] and image:
            attachments.append(image.url)

        await interaction.followup.send(content="Thinking...")

        try:
            llm_sdk = LLMSDK(model=model)
            content = await llm_sdk.prepare_response_content(prompt=prompt, image_urls=attachments)
            try:
                # 獲取用戶的最新 response ID
                previous_response_id = self.user_last_response_id.get(interaction.user.id, None)
                responses = await llm_sdk.client.responses.create(
                    model=model,
                    tools=[{"type": "web_search_preview"}],
                    input=[{"role": "user", "content": content}],
                    previous_response_id=previous_response_id,
                )
            except BadRequestError:
                # 如果 API 回傳錯誤（response ID 無效），清理該用戶記錄並重新嘗試
                self.user_last_response_id.pop(interaction.user.id, None)
                responses = await llm_sdk.client.responses.create(
                    model=model,
                    tools=[{"type": "web_search_preview"}],
                    input=[{"role": "user", "content": content}],
                )

            # 儲存新的 response ID
            self.user_last_response_id[interaction.user.id] = responses.id

            await interaction.edit_original_message(
                content=f"{interaction.user.mention}\n{responses.output_text}"
            )

        except Exception as e:
            await interaction.edit_original_message(content=f"{e}")
            logfire.error("Error in oai", _exc_info=True)

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
        had_memory = self.user_last_response_id.pop(user_id, None) is not None

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
