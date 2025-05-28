import logfire
import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands

from src.sdk.llm import LLMSDK


class ImageGeneratorCogs(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(
        name="graph",
        description="Generate an image based on the given prompt.",
        name_localizations={Locale.zh_TW: "生成圖片", Locale.ja: "画像を生成"},
        description_localizations={
            Locale.zh_TW: "根據提供的提示詞生成圖片。",
            Locale.ja: "指定されたプロンプトに基づいて画像を生成します。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def graph(
        self,
        interaction: Interaction,
        prompt: str = SlashOption(
            description="Enter your prompt.",
            description_localizations={
                Locale.zh_TW: "輸入提示詞。",
                Locale.ja: "プロンプトを入力してください。",
            },
        ),
    ) -> None:
        await interaction.response.defer()
        await interaction.followup.send(content="圖片生成中...")
        try:
            llm_sdk = LLMSDK(model="dall-e-3")
            image = await llm_sdk.client.images.generate(
                model=llm_sdk.model, prompt=prompt, n=1, size="1024x1024"
            )
            embed = nextcord.Embed(
                title="生成的圖片", description=f"提示詞: {prompt}", color=nextcord.Color.blue()
            )
            embed.set_image(url=image.data[0].url)
            embed.set_footer(text="圖片由 OpenAI 生成")
            await interaction.edit_original_message(embed=embed)
        except Exception as e:
            await interaction.edit_original_message(content=f"生成圖片時發生錯誤: {e!s}")
            logfire.error("Error generating image", _exc_info=True)


# 註冊 Cog
async def setup(bot: commands.Bot) -> None:
    bot.add_cog(ImageGeneratorCogs(bot), override=True)
