import os

import nextcord
from nextcord import Locale, Interaction, SlashOption
from nextcord.ext import commands

from src.sdk.llm import LLMSDK

os.environ["ANONYMIZED_TELEMETRY"] = "false"


class WebSearchCogs(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(
        name="search",
        description="Search the web based on the given prompt.",
        name_localizations={Locale.zh_TW: "網路搜尋", Locale.ja: "ウェブ検索"},
        description_localizations={
            Locale.zh_TW: "根據提供的提示詞進行網路搜尋。",
            Locale.ja: "指定されたプロンプトに基づいてウェブ検索を行います。",
        },
        dm_permission=True,
        nsfw=False,
    )
    async def search(
        self,
        interaction: Interaction,
        prompt: str = SlashOption(
            description="Enter your search query.",
            description_localizations={
                Locale.zh_TW: "輸入你的搜尋內容。",
                Locale.ja: "検索クエリを入力してください。",
            },
        ),
    ) -> None:
        await interaction.response.defer()
        await interaction.followup.send(content="搜尋中...")
        try:
            llm_sdk = LLMSDK()
            response = await llm_sdk.get_search_result(prompt=prompt)
            await interaction.edit_original_message(content=response.choices[0].message.content)
        except Exception as e:
            await interaction.edit_original_message(content=f"搜尋時發生錯誤: {e!s}")


# 註冊 Cog
async def setup(bot: commands.Bot) -> None:
    bot.add_cog(WebSearchCogs(bot), override=True)
