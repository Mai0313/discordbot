import base64
from typing import Optional

import aiohttp
import logfire
import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands

from src.sdk.llm import LLMSDK

MODEL_CHOICES = {
    "GPT-4o": "gpt-4o",
    "GPT-4o-mini": "gpt-4o-mini",
    "GPT-4-Turbo": "gpt-4-turbo",
    "o3-mini": "o3-mini",
    "o1": "o1",
    "o1-mini": "o1-mini",
}


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the ReplyGeneratorCogs.

        Args:
            bot (commands.Bot): The bot instance.
        """
        self.bot = bot

    async def _get_attachment_list(self, message: Optional[nextcord.Message]) -> list[str]:
        """Retrieve all attachments from a message.

        This function extracts image attachment URLs, embed descriptions, and converts sticker images
        to base64 encoded strings. If the message is None, an empty list is returned.

        Args:
            message (Optional[nextcord.Message]): The message from which to extract attachments.

        Returns:
            List[str]: A list containing the attachment URLs and base64 encoded sticker images.
        """
        image_urls: list[str] = []
        embed_list: list[str] = []
        sticker_list: list[str] = []
        if not message:
            return []
        if message.attachments:
            image_urls = [attachment.url for attachment in message.attachments]
        if message.embeds:
            embed_list = [embed.description for embed in message.embeds if embed.description]
        if message.stickers:
            async with aiohttp.ClientSession() as session:
                for sticker in message.stickers:
                    async with session.get(sticker.url) as response:
                        if response.status == 200:
                            sticker_data = await response.read()
                            base64_image = base64.b64encode(sticker_data).decode("utf-8")
                            sticker_list.append(f"data:image/png;base64,{base64_image}")
        attachments = [*image_urls, *embed_list, *sticker_list]
        return attachments

    @nextcord.slash_command(
        name="oai",
        description="Generate a reply based on the given prompt.",
        name_localizations={
            Locale.zh_TW: "生成文字",
            Locale.zh_CN: "生成文字",
            Locale.ja: "テキストを生成",
        },
        description_localizations={
            Locale.zh_TW: "根據提供的提示生成回覆。",
            Locale.zh_CN: "根据提供的提示生成回复。",
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
                Locale.zh_CN: "请输入提示词。",
                Locale.ja: "プロンプトを入力してください。",
            },
        ),
        model: str = SlashOption(
            description="Choose a model (default: GPT-4o).",
            description_localizations={
                Locale.zh_TW: "選擇模型 (預設為 GPT-4o)",
                Locale.zh_CN: "选择模型（默认: GPT-4o）",
                Locale.ja: "モデルを選択してください（デフォルトは GPT-4o）",
            },
            choices=MODEL_CHOICES,
            required=False,
            default="gpt-4o",
        ),
        image: Optional[nextcord.Attachment] = SlashOption(  # noqa: B008
            description="(Optional) Upload an image.",
            description_localizations={
                Locale.zh_TW: "（可選）上傳一張圖片。",
                Locale.zh_CN: "（可选）上传一张图片。",
                Locale.ja: "（オプション）画像をアップロードしてください。",
            },
            required=False,
        ),
    ) -> None:
        """Generate a reply based on the user's prompt.

        If the model 'o1' is selected along with an image, an error message is returned since
        'o1' does not support image input. Otherwise, the function retrieves attachments from the message,
        calls the LLM SDK to generate a reply, and updates the original message with the generated content.

        Args:
            interaction (Interaction): The interaction object for the command.
            prompt (str): The prompt text provided by the user.
            model (str): The selected model, defaults to "gpt-4o" if not specified.
            image (Optional[nextcord.Attachment]): An optional image attachment uploaded by the user.

        Returns:
            None
        """
        if model == "o1" and image:
            await interaction.response.send_message("❌ o1 模型不支援圖片輸入。")
            return

        llm_sdk = LLMSDK(llm_model=model)
        attachments = await self._get_attachment_list(interaction.message)
        if image:
            attachments.append(image.url)

        init_message = (
            "⚠️ 你選擇的 o1 模型速度較慢，請稍候..." if model == "o1" else "Generating..."
        )
        await interaction.response.send_message(init_message)

        try:
            response = await llm_sdk.get_oai_reply(prompt=prompt, image_urls=attachments)
            final_content = f"{interaction.user.mention} {response.choices[0].message.content}"
            await interaction.edit_original_message(content=final_content)
        except Exception as e:
            await interaction.edit_original_message(content=f"Error processing the message: {e!s}")

    @nextcord.slash_command(
        name="oais",
        description="Generate a reply based on the prompt and show progress in real-time.",
        name_localizations={
            Locale.zh_TW: "實時生成文字",
            Locale.zh_CN: "实时生成文字",
            Locale.ja: "テキストを生成",
        },
        description_localizations={
            Locale.zh_TW: "根據提示詞即時生成回覆，並在生成過程中顯示進度。",
            Locale.zh_CN: "根据提示词实时生成回复，并在生成过程中显示进度。",
            Locale.ja: "指定されたプロンプトに基づいてリアルタイムで応答を生成し、進捗を表示します。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def oais(
        self,
        interaction: Interaction,
        prompt: str = SlashOption(
            description="Enter your prompt.",
            description_localizations={
                Locale.zh_TW: "請輸入提示詞。",
                Locale.zh_CN: "请输入提示词。",
                Locale.ja: "プロンプトを入力してください。",
            },
        ),
        model: str = SlashOption(
            description="Choose a model (default: GPT-4o).",
            description_localizations={
                Locale.zh_TW: "選擇模型 (預設為 GPT-4o)",
                Locale.zh_CN: "选择模型（默认: GPT-4o）",
                Locale.ja: "モデルを選択してください（デフォルトは GPT-4o）",
            },
            choices=MODEL_CHOICES,
            required=False,
            default="gpt-4o",
        ),
        image: Optional[nextcord.Attachment] = SlashOption(  # noqa: B008
            description="(Optional) Upload an image.",
            description_localizations={
                Locale.zh_TW: "（可選）上傳一張圖片。",
                Locale.zh_CN: "（可选）上传一张图片。",
                Locale.ja: "（オプション）画像をアップロードしてください。",
            },
            required=False,
        ),
    ) -> None:
        """Generate a reply in real-time based on the user's prompt.

        If the selected model is 'o1' or 'o1-mini', which do not support real-time responses, an error message is returned.
        Otherwise, the function retrieves attachments and continuously updates the reply message with the generated content.

        Args:
            interaction (Interaction): The interaction object for the command.
            prompt (str): The prompt text provided by the user.
            model (str): The selected model, defaults to "gpt-4o" if not specified.
            image (Optional[nextcord.Attachment]): An optional image attachment uploaded by the user.

        Returns:
            None
        """
        if model in ["o1", "o1-mini"]:
            await interaction.response.send_message(
                "❌ The selected model does not support real-time response. Please choose another model or use standard generation."
            )
            return

        llm_sdk = LLMSDK(llm_model=model)
        attachments = await self._get_attachment_list(interaction.message)
        if image:
            attachments.append(image.url)
        message = await interaction.response.send_message(content="Generating...")
        accumulated_text = f"{interaction.user.mention}\n"

        try:
            async for res in llm_sdk.get_oai_reply_stream(prompt=prompt, image_urls=attachments):
                if (
                    hasattr(res, "choices")
                    and len(res.choices) > 0
                    and res.choices[0].delta.content
                ):
                    accumulated_text += res.choices[0].delta.content
                    await message.edit(content=accumulated_text)

        except Exception as e:
            await message.edit(
                content=f"{interaction.user.mention} Unable to generate a valid reply, please try another prompt."
            )
            logfire.error(f"Error in oais: {e}")


async def setup(bot: commands.Bot) -> None:
    """Register the reply generation cog with the bot.

    Args:
        bot (commands.Bot): The bot instance to which the cog will be added.

    Returns:
        None
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
