import nextcord
from nextcord import Embed, Locale, Interaction
from nextcord.ext import commands

from discordbot.cogs._economy.presentation import CURRENCY_NAME

_HELP_CONTENT = {
    "default": {
        "title": "Bot Guide",
        "description": "Here's everything I can do for you!",
        "ai_chat": (
            "**AI Chat**\n"
            "Mention me or send a DM to start a conversation.\n"
            "- Send text to ask questions\n"
            "- Attach images or supported files for analysis\n"
            "- Ask me to generate images or videos\n"
            "- Send a long article / URL and ask for a summary\n"
            f"- Earn {CURRENCY_NAME} from streaming AI replies"
        ),
        "threads": (
            "**Threads Parser**\n"
            "Paste a Threads.net or Threads.com link and I'll automatically "
            "extract the content with media for you. "
            "Reply links also pull in the original post and any intermediate replies "
            "as context, with a grey gradient stripe so each layer is easy to tell apart. "
            "If the result is too large for Discord, I'll mark it with a warning reaction."
        ),
        "video": (
            "**Video Download** — `/download_video`\n"
            "Download videos from YouTube, Facebook, Instagram, X, TikTok, and more. "
            "Use the optional quality setting; oversized files retry once at low quality."
        ),
        "maplestory": (
            "**MapleStory Artale** — `/maple_*`\n"
            "Search game data:\n"
            "`/maple_monster` · `/maple_equip` · `/maple_scroll` · "
            "`/maple_npc` · `/maple_quest` · `/maple_map` · "
            "`/maple_item` · `/maple_stats`"
        ),
        "points": (
            f"**{CURRENCY_NAME}**\n"
            f"Earn {CURRENCY_NAME} from AI chat replies, then use them across servers.\n"
            "`/balance` check your balance · `/leaderboard` show the global top 10\n"
            f"`/give` transfer {CURRENCY_NAME} · `/house` show the dealer's running P&L\n"
            "`/balance`, `/leaderboard`, and `/house` results clean themselves up after 3 minutes."
        ),
        "games": (
            "**Games**\n"
            "`/dice` roll three dice against the dealer.\n"
            "`/dragon_gate` shoot one card strictly between two gate cards.\n"
            "`/blackjack` play one round of 21. Bets are withdrawn up front, "
            "over-bets auto all-in, idle hands auto-stand after 180 seconds, "
            "and final game messages are cleaned up after 3 minutes."
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
            "- 附加圖片或支援的檔案進行分析\n"
            "- 請我生成圖片或影片\n"
            "- 傳送長文 / 網址請我做摘要\n"
            f"- AI 串流回覆會獎勵{CURRENCY_NAME}"
        ),
        "threads": (
            "**Threads 解析**\n"
            "貼上 Threads.net 或 Threads.com 的連結，我會自動擷取內容與媒體。"
            "如果是回覆的連結，我也會把原始貼文與中間每一層回覆一起帶出來當作上下文，"
            "並用灰階漸層色帶區分層級。"
            "如果內容超過 Discord 限制，會用 warning reaction 標記。"
        ),
        "video": (
            "**影片下載** — `/download_video`\n"
            "支援從 YouTube、Facebook、Instagram、X、TikTok 等平台下載影片。"
            "可以選 quality；檔案太大時會自動 retry 一次低畫質。"
        ),
        "maplestory": (
            "**MapleStory Artale** — `/maple_*`\n"
            "查詢遊戲資料：\n"
            "`/maple_monster` · `/maple_equip` · `/maple_scroll` · "
            "`/maple_npc` · `/maple_quest` · `/maple_map` · "
            "`/maple_item` · `/maple_stats`"
        ),
        "points": (
            f"**{CURRENCY_NAME}**\n"
            f"AI chat 回覆會累積{CURRENCY_NAME}，{CURRENCY_NAME}跨 server 共用。\n"
            "`/balance` 查餘額 · `/leaderboard` 看 global 前 10 名\n"
            "`/give` 轉虛擬歡樂豆 · `/house` 看莊家累積 P&L\n"
            "`/balance`、`/leaderboard`、`/house` 查詢結果會在 3 分鐘後自動清掉。"
        ),
        "games": (
            "**小遊戲**\n"
            "`/dice` 用三顆骰子跟莊家比大小。\n"
            "`/dragon_gate` 射一張牌嚴格落在兩張門牌中間。\n"
            "`/blackjack` 跟莊家玩一局 21 點。bet 會先扣，超過餘額會自動 all-in，"
            "不操作 180 秒會自動 stand 結算，final game message 會在 3 分鐘後清掉。"
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
            "- 画像や対応ファイルを添付して分析\n"
            "- 画像や動画の生成をリクエスト\n"
            "- 長文 / URLを送って要約をリクエスト\n"
            f"- AIのストリーミング返信で{CURRENCY_NAME}を獲得"
        ),
        "threads": (
            "**Threads パーサー**\n"
            "Threads.net または Threads.com のリンクを貼ると、"
            "自動的にコンテンツとメディアを取得します。"
            "返信へのリンクの場合は、元の投稿と途中の返信もすべて文脈として展開し、"
            "グレースケールのグラデーションで各レイヤーを区別します。"
            "Discord の制限を超える場合は warning reaction で知らせます。"
        ),
        "video": (
            "**動画ダウンロード** — `/download_video`\n"
            "YouTube、Facebook、Instagram、X、TikTok などから動画をダウンロードします。"
            "quality を選択でき、ファイルが大きすぎる場合は一度 low quality で retry します。"
        ),
        "maplestory": (
            "**MapleStory Artale** — `/maple_*`\n"
            "ゲームデータ検索：\n"
            "`/maple_monster` · `/maple_equip` · `/maple_scroll` · "
            "`/maple_npc` · `/maple_quest` · `/maple_map` · "
            "`/maple_item` · `/maple_stats`"
        ),
        "points": (
            f"**{CURRENCY_NAME}**\n"
            f"AI チャット返信で{CURRENCY_NAME}を獲得し、サーバーをまたいで使えます。\n"
            "`/balance` 残高確認 · `/leaderboard` グローバルトップ10\n"
            f"`/give` {CURRENCY_NAME}送付 · `/house` ディーラーの累計損益\n"
            "`/balance`、`/leaderboard`、`/house` の結果は3分後に自動削除されます。"
        ),
        "games": (
            "**ゲーム**\n"
            "`/dice` 3個のサイコロでディーラーと勝負します。\n"
            "`/dragon_gate` 2枚のゲートカードの間に厳密に1枚を通します。\n"
            "`/blackjack` 21を1ラウンド遊びます。ベットは先に差し引かれ、"
            "残高超過は自動 all-in、180秒操作がない場合は自動 stand で精算され、"
            "final game message は3分後に削除されます。"
        ),
        "ping": "**Ping** — `/ping`\nボットの応答遅延を確認します。",
    },
}

_SECTIONS = ("ai_chat", "threads", "video", "maplestory", "points", "games", "ping")


class HelpCogs(commands.Cog):
    """Provides the localized help slash command.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the HelpCogs instance.

        Args:
            bot: The Discord bot instance.
        """
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
        """Shows a guide on how to use this bot.

        Args:
            interaction: The interaction that triggered the command.
        """
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


def setup(bot: commands.Bot) -> None:
    """Adds the HelpCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(HelpCogs(bot), override=True)
