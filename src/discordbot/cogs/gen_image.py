import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands


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
        await interaction.edit_original_message(content="請直接使用 /生成 來產生圖片")


# 註冊 Cog
async def setup(bot: commands.Bot) -> None:
    bot.add_cog(ImageGeneratorCogs(bot), override=True)
