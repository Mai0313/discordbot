import nextcord
from nextcord import Embed, Locale, Interaction
from nextcord.ext import commands

from discordbot.typings.economy import (
    VIP_PURCHASE_COST,
    BASE_CHECKIN_REWARD_AMOUNT,
    BASE_MESSAGE_REWARD_AMOUNT,
)
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
            f"- AI replies add token-based {CURRENCY_NAME} bonuses"
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
            f"Every message earns {BASE_MESSAGE_REWARD_AMOUNT:,} {CURRENCY_NAME}; "
            "AI chat replies add token-based bonuses.\n"
            "`/balance` check your balance · `/leaderboard` show the global top 10 · "
            "`/loss_leaderboard` show today's biggest casino losers\n"
            f"`/give` transfer {CURRENCY_NAME} · `/house` show the dealer's running P&L\n"
            "`/borrow` take out a loan (cap scales with your Discord account age, "
            "auto-expires at Asia/Taipei midnight); `/repay` pay debt from your balance.\n"
            "Every income event after a loan auto-applies 50% toward principal.\n"
            "`/balance`, `/leaderboard`, `/loss_leaderboard`, `/house`, `/borrow`, "
            "and `/repay` results clean themselves up after 3 minutes."
        ),
        "checkin": (
            "**Daily Check-in** — `/checkin`\n"
            f"Claim {BASE_CHECKIN_REWARD_AMOUNT:,} {CURRENCY_NAME} once per day (Asia/Taipei). "
            "Consecutive days within a 7-day cycle add a streak bonus; the reply is "
            "ephemeral so only you see it."
        ),
        "vip": (
            "**VIP** — `/vip`\n"
            f"Buy permanent VIP for {VIP_PURCHASE_COST:,} {CURRENCY_NAME}. VIP gives 1.5x "
            "blackjack payouts, 2x check-in points, and 2x borrow cap."
        ),
        "games": (
            "**Games**\n"
            "`/blackjack` play one round of 21. Bets settle only when the round ends, "
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
            "tag 我或私訊我即可開始對話\n"
            "- 傳送文字來問問題\n"
            "- 附加圖片或支援的檔案進行分析\n"
            "- 請我生成圖片或影片\n"
            "- 傳送長文 / 網址請我做摘要\n"
            f"- AI 回覆會追加 token 計算的{CURRENCY_NAME} bonus"
        ),
        "threads": (
            "**Threads 解析**\n"
            "貼上 Threads.net 或 Threads.com 的連結，我會自動擷取內容與媒體"
            "如果是回覆的連結，我也會把原始貼文與中間每一層回覆一起帶出來當作上下文，"
            "並用灰階漸層色帶區分層級"
            "如果內容超過 Discord 限制，會用 warning reaction 標記"
        ),
        "video": (
            "**影片下載** — `/download_video`\n"
            "支援從 YouTube、Facebook、Instagram、X、TikTok 等平台下載影片"
            "可以選 quality；檔案太大時會自動 retry 一次低畫質"
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
            f"每則訊息會獲得 {BASE_MESSAGE_REWARD_AMOUNT:,} {CURRENCY_NAME}, "
            f"AI chat 回覆另外追加 token bonus, {CURRENCY_NAME}跨 server 共用\n"
            "`/balance` 查餘額 (含欠款狀態) · `/leaderboard` 看 global 前 10 名 · "
            "`/loss_leaderboard` 看今日輸最多前 10 名\n"
            "`/give` 轉虛擬歡樂豆 · `/house` 看莊家累積 P&L\n"
            "`/borrow` 依 Discord 帳號年齡借款 (每天 0:00 Asia/Taipei 自動清零); "
            "`/repay` 從餘額還款\n"
            "借款後賺到的點數會自動 50% 用來還本金\n"
            "`/balance`、`/leaderboard`、`/loss_leaderboard`、`/house`、"
            "`/borrow`、`/repay` 結果 3 分鐘後自動清掉"
        ),
        "checkin": (
            "**每日簽到** — `/checkin`\n"
            f"每天可以領 {BASE_CHECKIN_REWARD_AMOUNT:,} {CURRENCY_NAME} (Asia/Taipei), "
            "連續七天為一個 cycle, 每天加成. 訊息是 ephemeral 只有自己看得到"
        ),
        "vip": (
            "**VIP** — `/vip`\n"
            f"花 {VIP_PURCHASE_COST:,} {CURRENCY_NAME}購買永久 VIP\n"
            "VIP 賭場贏錢 1.5x, 簽到 2x 點數, 貸款額度 2x"
        ),
        "games": (
            "**小遊戲**\n"
            "`/blackjack` 跟莊家玩一局 21 點，bet 會在結算時處理，超過餘額會自動 all-in，"
            "不操作 180 秒會自動 stand 結算，final game message 會在 3 分鐘後清掉"
        ),
        "ping": "**延遲測試** — `/ping`\n檢查機器人的回應延遲",
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
            f"- AI返信ではtokenベースの{CURRENCY_NAME}ボーナスも獲得"
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
            f"すべてのメッセージで{BASE_MESSAGE_REWARD_AMOUNT:,}{CURRENCY_NAME}を獲得し、"
            "AI チャット返信ではtokenベースのボーナスも入ります。\n"
            "`/balance` 残高と借入状況を確認 · `/leaderboard` グローバルトップ10 · "
            "`/loss_leaderboard` 本日の負け額トップ10\n"
            f"`/give` {CURRENCY_NAME}送付 · `/house` ディーラーの累計損益\n"
            "`/borrow` Discord アカウント年齢に応じて借入 (毎日0:00 Asia/Taipei 自動リセット); "
            "`/repay` 残高から返済。\n"
            "借入後の獲得点数は50%が自動的に元本返済に充当されます。\n"
            "`/balance`、`/leaderboard`、`/loss_leaderboard`、`/house`、"
            "`/borrow`、`/repay` の結果は3分後に自動削除されます。"
        ),
        "checkin": (
            "**デイリーチェックイン** — `/checkin`\n"
            f"毎日{BASE_CHECKIN_REWARD_AMOUNT:,}{CURRENCY_NAME}を受け取れます (Asia/Taipei)。"
            "7日サイクルで連続日数ボーナスが付き、返信は ephemeral で本人のみ閲覧可能。"
        ),
        "vip": (
            "**VIP** — `/vip`\n"
            f"{VIP_PURCHASE_COST:,}{CURRENCY_NAME}で永久 VIP を購入。"
            "VIP はブラックジャック配当1.5x、チェックイン2x、借入上限2x。"
        ),
        "games": (
            "**ゲーム**\n"
            "`/blackjack` 21を1ラウンド遊びます。ベットはラウンド終了時に精算され、"
            "残高超過は自動 all-in、180秒操作がない場合は自動 stand で精算され、"
            "final game message は3分後に削除されます。"
        ),
        "ping": "**Ping** — `/ping`\nボットの応答遅延を確認します。",
    },
}

_SECTIONS = (
    "ai_chat",
    "threads",
    "video",
    "maplestory",
    "points",
    "checkin",
    "vip",
    "games",
    "ping",
)


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
            Locale.zh_TW: "顯示機器人的使用指南",
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
