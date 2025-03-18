import base64

import aiohttp
import logfire
import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands

from src.sdk.llm import LLMSDK


class ReplyGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def _get_attachment_list(self, message: nextcord.Message) -> list[str]:
        image_urls, embed_list, sticker_list = [], [], []
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
        name_localizations={Locale.zh_TW: "生成文字", Locale.ja: "テキストを生成"},
        description_localizations={
            Locale.zh_TW: "根據提供的提示詞生成回應。",
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
                Locale.zh_TW: "輸入提示詞。",
                Locale.ja: "プロンプトを入力してください。",
            },
        ),
        model: str = SlashOption(
            description="Choose a model (預設為 GPT-4o)",
            description_localizations={
                Locale.zh_TW: "選擇一個模型 (預設為 GPT-4o)",
                Locale.ja: "モデルを選択してください（デフォルトは GPT-4o）",
            },
            choices={
                "GPT-4o": "gpt-4o",
                "GPT-4o-mini": "gpt-4o-mini",
                "GPT-4-Turbo": "gpt-4-turbo",
                "o3-mini": "o3-mini",
                "o1": "o1",
                "o1-mini": "o1-mini",
            },
            required=False,
            default="gpt-4o",
        ),
        image: nextcord.Attachment = SlashOption(  # noqa: B008
            description="(Optional) Upload an image.",
            description_localizations={
                Locale.zh_TW: "（可選）上傳一張圖片。",
                Locale.ja: "（オプション）画像をアップロードしてください。",
            },
            required=False,
        ),
    ) -> None:
        # 若選擇 o1 模型且同時上傳圖片，立即回覆錯誤訊息並結束
        if model == "o1" and image:
            await interaction.response.send_message("❌ o1 模型不支援圖片輸入。")
            return

        llm_sdk = LLMSDK(llm_model=model)
        attachments = await self._get_attachment_list(interaction.message)
        if image:
            attachments.append(image.url)

        # 根據模型類型決定初始回覆內容
        init_message = "⚠️ 你選擇的 o1 模型速度較慢，請稍候..." if model == "o1" else "生成中..."
        await interaction.response.send_message(init_message)

        try:
            response = await llm_sdk.get_oai_reply(prompt=prompt, image_urls=attachments)
            final_content = f"{interaction.user.mention} {response.choices[0].message.content}"
            await interaction.edit_original_message(content=final_content)
        except Exception as e:
            await interaction.edit_original_message(content=f"處理訊息時發生錯誤: {e!s}")

    @nextcord.slash_command(
        name="oais",
        description="Generate a reply based on the given prompt and show progress.",
        name_localizations={Locale.zh_TW: "實時生成文字", Locale.ja: "テキストを生成"},
        description_localizations={
            Locale.zh_TW: "此指令將根據提供的提示實時生成回覆。",
            Locale.ja: "指定されたプロンプトに基づいて応答を生成し、進行状況を表示します。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def oais(
        self,
        interaction: Interaction,
        prompt: str = SlashOption(
            description="Enter your prompt",
            description_localizations={
                Locale.zh_TW: "輸入提示詞",
                Locale.ja: "プロンプトを入力してください",
            },
        ),
        model: str = SlashOption(
            description="Choose a model (預設為 gpt-4o)",
            description_localizations={
                Locale.zh_TW: "選擇一個模型 (預設為 gpt-4o)",
                Locale.ja: "モデルを選択してください（デフォルトは gpt-4o）",
            },
            choices={
                "GPT-4o": "gpt-4o",
                "GPT-4o-mini": "gpt-4o-mini",
                "GPT-4-Turbo": "gpt-4-turbo",
                "o3-mini": "o3-mini",
                "o1": "o1",
                "o1-mini": "o1-mini",
            },
            required=False,
            default="gpt-4o",
        ),
        image: nextcord.Attachment = SlashOption(  # noqa: B008
            description="(Optional) Upload an image.",
            description_localizations={
                Locale.zh_TW: "（可選）上傳一張圖片。",
                Locale.ja: "（オプション）画像をアップロードしてください。",
            },
            required=False,
        ),
    ) -> None:
        # 如果使用者選擇了 o1 或 o1-mini，則這兩個模型不支援串流
        if model in ["o1", "o1-mini"]:
            await interaction.response.send_message(
                "❌ 所選模型不支援實時回應，請選擇其他模型或使用普通生成。"
            )
            return

        llm_sdk = LLMSDK(llm_model=model)
        # 取得 message 附件並合併參數提供的圖片
        attachments = await self._get_attachment_list(interaction.message)
        if image:
            attachments.append(image.url)
        message = await interaction.response.send_message(content="生成中...")
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
                content=f"{interaction.user.mention} 無法生成有效回應，請嘗試其他提示詞。"
            )
            logfire.error(f"Error in oais: {e}")


# 註冊 Cog
async def setup(bot: commands.Bot) -> None:
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
