import logfire
import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands

from src.sdk.llm import LLMSDK

available_models = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    # "o4-mini",
    # "o3",
    # "o3-mini",
    # "o1",
    # "o1-preview",
    # "o1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    # "chatgpt-4o-latest",
]
MODEL_CHOICES = {available_model: available_model for available_model in available_models}


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the ReplyGeneratorCogs.

        Args:
            bot (commands.Bot): The bot instance.
        """
        self.bot = bot

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
            description="Choose a model (default: GPT-4.1).",
            description_localizations={
                Locale.zh_TW: "選擇模型 (預設為 GPT-4.1)",
                Locale.ja: "モデルを選択してください（デフォルトは GPT-4.1）",
            },
            choices=MODEL_CHOICES,
            required=False,
            default="gpt-4.1",
        ),
        stream: bool = SlashOption(
            description="Enable streaming response (default: False).",
            description_localizations={
                Locale.zh_TW: "啟用串流回應 (預設為 False)",
                Locale.ja: "ストリーミング応答を有効にする（デフォルト: False）",
            },
            required=False,
            default=False,
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
            model (str): The selected model, defaults to "gpt-4.1" if not specified.
            stream (bool): Whether to stream the response in real-time, defaults to False.
            image (Optional[nextcord.Attachment]): An optional image attachment uploaded by the user.
        """
        await interaction.response.defer()
        attachments = []
        if model not in ["o1", "o1-mini"] and image:
            attachments.append(image.url)

        await interaction.followup.send(content="思考中...")

        try:
            llm_sdk = LLMSDK(model=model)

            if stream:
                # Streaming response
                accumulated_text = f"{interaction.user.mention}\n"
                async for res in llm_sdk.get_oai_reply_stream(
                    prompt=prompt, image_urls=attachments
                ):
                    if res.choices[0].delta.content:
                        accumulated_text += res.choices[0].delta.content
                        await interaction.edit_original_message(content=accumulated_text)
            else:
                # Non-streaming response
                response = await llm_sdk.get_oai_reply(prompt=prompt, image_urls=attachments)
                final_content = (
                    f"{interaction.user.mention}\n{response.choices[0].message.content}"
                )
                await interaction.edit_original_message(content=final_content)

        except Exception:
            error_message = "Error processing the message."
            await interaction.edit_original_message(content=error_message)
            logfire.error("Error in oai", _exc_info=True)


async def setup(bot: commands.Bot) -> None:
    """Register the reply generation cog with the bot.

    Args:
        bot (commands.Bot): The bot instance to which the cog will be added.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
