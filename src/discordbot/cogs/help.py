import nextcord
from nextcord import Embed, Locale, Interaction
from nextcord.ext import commands

_HELP_CONTENT = {
    "default": {
        "title": "Bot Guide",
        "description": "Here's everything I can do for you!",
        "ai_chat": (
            "**AI Chat**\n"
            "Mention me or send a DM to start a conversation.\n"
            "- Send text to ask questions\n"
            "- Attach images for analysis\n"
            "- Ask me to generate images or videos\n"
            "- Send a long article / URL and ask for a summary"
        ),
        "threads": (
            "**Threads Parser**\n"
            "Paste a Threads.net link and I'll automatically "
            "extract the content with media for you."
        ),
        "video": (
            "**Video Download** — `/download_video`\n"
            "Download videos from YouTube, Facebook, Instagram, X, TikTok, and more."
        ),
        "maplestory": (
            "**MapleStory Artale** — `/maple_*`\n"
            "Search game data:\n"
            "`/maple_monster` · `/maple_equip` · `/maple_scroll` · "
            "`/maple_npc` · `/maple_quest` · `/maple_map` · "
            "`/maple_item` · `/maple_stats`"
        ),
        "ping": "**Ping** — `/ping`\nCheck the bot's response latency.",
    },
    Locale.zh_TW: {
        "title": "機器人使用指南",
        "description": "以下是我能為你做的所有事情!",
        "ai_chat": (
            "**AI 對話**\n"
            "tag 我或私訊我即可開始對話。\n"
            "- 傳送文字來問問題\n"
            "- 附加圖片進行分析\n"
            "- 請我生成圖片或影片\n"
            "- 傳送長文 / 網址請我做摘要"
        ),
        "threads": ("**Threads 解析**\n貼上 Threads.net 的連結，我會自動擷取內容與媒體。"),
        "video": (
            "**影片下載** — `/download_video`\n"
            "支援從 YouTube、Facebook、Instagram、X、TikTok 等平台下載影片。"
        ),
        "maplestory": (
            "**MapleStory Artale** — `/maple_*`\n"
            "查詢遊戲資料：\n"
            "`/maple_monster` · `/maple_equip` · `/maple_scroll` · "
            "`/maple_npc` · `/maple_quest` · `/maple_map` · "
            "`/maple_item` · `/maple_stats`"
        ),
        "ping": "**延遲測試** — `/ping`\n檢查機器人的回應延遲。",
    },
    Locale.ja: {
        "title": "ボット利用ガイド",
        "description": "私ができることをご紹介します!",
        "ai_chat": (
            "**AI チャット**\n"
            "メンションまたはDMで会話を開始できます。\n"
            "- テキストで質問\n"
            "- 画像を添付して分析\n"
            "- 画像や動画の生成をリクエスト\n"
            "- 長文 / URLを送って要約をリクエスト"
        ),
        "threads": (
            "**Threads パーサー**\n"
            "Threads.net のリンクを貼ると、自動的にコンテンツとメディアを取得します。"
        ),
        "video": (
            "**動画ダウンロード** — `/download_video`\n"
            "YouTube、Facebook、Instagram、X、TikTok などから動画をダウンロードします。"
        ),
        "maplestory": (
            "**MapleStory Artale** — `/maple_*`\n"
            "ゲームデータ検索：\n"
            "`/maple_monster` · `/maple_equip` · `/maple_scroll` · "
            "`/maple_npc` · `/maple_quest` · `/maple_map` · "
            "`/maple_item` · `/maple_stats`"
        ),
        "ping": "**Ping** — `/ping`\nボットの応答遅延を確認します。",
    },
}

_SECTIONS = ("ai_chat", "threads", "video", "maplestory", "ping")


class HelpCogs(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @nextcord.slash_command(
        name="help",
        description="Show a guide on how to use this bot.",
        name_localizations={Locale.zh_TW: "使用說明", Locale.ja: "ヘルプ"},
        description_localizations={
            Locale.zh_TW: "顯示機器人的使用指南。",
            Locale.ja: "ボットの使い方ガイドを表示します。",
        },
        nsfw=False,
    )
    async def help(self, interaction: Interaction) -> None:
        await interaction.response.defer(ephemeral=True)

        locale = interaction.locale
        content = _HELP_CONTENT.get(locale, _HELP_CONTENT["default"])

        embed = Embed(
            title=content["title"],
            description=content["description"],
            color=0x5865F2,
            timestamp=nextcord.utils.utcnow(),
        )

        for section in _SECTIONS:
            embed.add_field(name="\u200b", value=content[section], inline=False)

        embed.set_footer(
            text=f"Requested by {interaction.user.display_name}",
            icon_url=interaction.user.display_avatar.url,
        )

        await interaction.followup.send(embed=embed)


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(HelpCogs(bot), override=True)
