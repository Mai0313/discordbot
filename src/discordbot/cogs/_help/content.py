"""Localized help content and the data models that drive the help view."""

from nextcord import Locale
from pydantic import Field, BaseModel, field_validator

from discordbot.cogs._economy.presentation import CURRENCY_NAME

OVERVIEW_VALUE = "overview"
CATEGORY_ORDER = ("ai", "games", "economy", "stocks", "tools")


class HelpSection(BaseModel):
    """One help category shown as a select option and a detail embed.

    Attributes:
        emoji: Leading emoji for the select option and detail title.
        label: Category name used as the select option label and embed title.
        summary: One-line index/select-option description (kept under 100 chars).
        detail: Full command list rendered in the category's detail embed.
    """

    emoji: str = Field(description="Leading emoji for the select option and detail title.")
    label: str = Field(
        description="Category name used as the select option label and embed title."
    )
    summary: str = Field(
        description="One-line index/select-option description (kept under 100 chars)."
    )
    detail: str = Field(description="Full command list rendered in the category's detail embed.")


class HelpGuide(BaseModel):
    """Localized help content for one locale.

    Attributes:
        title: Overview embed title.
        intro: Leading line on the overview embed.
        select_placeholder: Placeholder shown on the category select menu.
        overview_label: Select option label that returns to the overview.
        sections: Category key to its content, keyed by `CATEGORY_ORDER`.
    """

    title: str = Field(description="Overview embed title.")
    intro: str = Field(description="Leading line on the overview embed.")
    select_placeholder: str = Field(description="Placeholder shown on the category select menu.")
    overview_label: str = Field(description="Select option label that returns to the overview.")
    sections: dict[str, HelpSection] = Field(
        description="Category key to its content, keyed by `CATEGORY_ORDER`."
    )

    @field_validator("sections")
    @classmethod
    def _sections_match_category_order(
        cls, value: dict[str, HelpSection]
    ) -> dict[str, HelpSection]:
        """Ensures a locale defines exactly the categories declared in CATEGORY_ORDER."""
        if set(value) != set(CATEGORY_ORDER):
            msg = (
                f"help sections {sorted(value)} must match CATEGORY_ORDER {sorted(CATEGORY_ORDER)}"
            )
            raise ValueError(msg)
        return value


HELP_CONTENT: dict[Locale | str, HelpGuide] = {
    "default": HelpGuide(
        title="🤖 Bot Guide",
        intro="Pick a category from the menu below to see its full commands.",
        select_placeholder="Choose a category…",
        overview_label="Overview",
        sections={
            "ai": HelpSection(
                emoji="💬",
                label="AI Chat",
                summary="Mention or DM me to chat, analyze, and generate",
                detail=(
                    "Mention me or DM me to get started.\n"
                    "• Chat, answer questions, and summarize articles\n"
                    "• Analyze attached images and files\n"
                    "• Generate images and short videos on request"
                ),
            ),
            "games": HelpSection(
                emoji="🎰",
                label="Casino Games",
                summary="Blackjack and Dragon Gate tables",
                detail=(
                    "**`/games blackjack`**\n"
                    "Start a Blackjack table. Five-card non-bust hands win, and five-or-more-card 21 keeps its system-funded bonus. The shoe carries over between rounds in the same channel and reshuffles once it runs low. `bet` accepts comma-formatted numbers, and `0` means all in. A single bet is capped at 1,000,000.\n\n"
                    "**`/games blackjack_history [member] [count]`**\n"
                    "Publicly post a player's recent Blackjack rounds as a shared table: hands, bets, dealer hands, and results. Defaults to yourself and the last 10 rounds (up to 50).\n\n"
                    "**`/games dragon_gate`**\n"
                    "Start a Dragon Gate jackpot table."
                ),
            ),
            "economy": HelpSection(
                emoji="💰",
                label=f"{CURRENCY_NAME} / Economy",
                summary="Balance, transfers, check-in, loans, boards",
                detail=(
                    "**Daily & status**\n"
                    "`/balance [member]` — balance, loans, holdings, net worth, VIP\n"
                    "`/checkin` — daily reward and streak bonus\n"
                    "`/vip` — buy or check VIP (boosts check-in and Blackjack rewards)\n\n"
                    "**Transfers & boards**\n"
                    "`/give` — transfer to members or bots (a 5% transfer tax is burned)\n"
                    "`/leaderboard` — balance board\n"
                    "`/loss_leaderboard` — today's loss board\n"
                    "`/casino` — casino system ledger\n"
                    "`/pocat` — bot player wallet\n\n"
                    "**Loans**\n"
                    "`/credit` — personal loans\n"
                    "`/central_bank` — central-bank loans\n\n"
                    "**Admin**\n"
                    "`/admin` — admin balance adjustments (comma-formatted amounts)"
                ),
            ),
            "stocks": HelpSection(
                emoji="📈",
                label="Simulated Stocks",
                summary="Trade and track virtual companies",
                detail=(
                    "**`/stock`**\n"
                    "Open the market board UI: prices, market-context news, top-holder and recent-trade summaries, positions, and 7D charts. Buy and short within supply caps (long capped at 49% of float), then sell or cover to exit. Share counts show in lots when possible."
                ),
            ),
            "tools": HelpSection(
                emoji="🧰",
                label="Tools",
                summary="Video, MapleStory, Threads, ping",
                detail=(
                    "`/download_video` — download videos from supported platforms (optional quality)\n"
                    "`/maplestory` — search monsters, equipment, scrolls, NPCs, quests, maps, items, and stats\n"
                    "`/ping` — check the bot's response latency\n\n"
                    "**Threads parser**\n"
                    "Paste a Threads.net or Threads.com URL and I'll extract posts, replies, and media."
                ),
            ),
        },
    ),
    Locale.zh_TW: HelpGuide(
        title="🤖 機器人使用指南",
        intro="從下方選單挑一個分類，看完整指令。",
        select_placeholder="選擇分類…",
        overview_label="總覽",
        sections={
            "ai": HelpSection(
                emoji="💬",
                label="AI 對話",
                summary="tag 或私訊我聊天、分析、生成",
                detail=(
                    "tag 我或私訊我就能開始。\n"
                    "• 聊天、回答問題、摘要文章\n"
                    "• 分析附上的圖片和檔案\n"
                    "• 依需求生成圖片或短影片"
                ),
            ),
            "games": HelpSection(
                emoji="🎰",
                label="賭場遊戲",
                summary="Blackjack 與射龍門",
                detail=(
                    "**`/games blackjack`**\n"
                    "開 21 點桌，五張未爆直接贏，五張或以上 21 保留 system-funded bonus；同一頻道的牌靴會跨局延續，剩牌不足時才重新洗牌；`bet` 可輸入含逗號的數字，`0` 就是 all in；單注上限 100 萬。\n\n"
                    "**`/games blackjack_history [member] [count]`**\n"
                    "公開貼出某位玩家近期的 21 點對局表格：手牌、下注、莊家手牌與結果；預設查自己、近 10 場（最多 50）。\n\n"
                    "**`/games dragon_gate`**\n"
                    "開射龍門 jackpot 桌。"
                ),
            ),
            "economy": HelpSection(
                emoji="💰",
                label=f"{CURRENCY_NAME} / 經濟",
                summary="餘額、轉帳、簽到、借貸、排行榜",
                detail=(
                    "**查詢與日常**\n"
                    "`/balance [member]` — 餘額、借貸、持股、淨資產、VIP\n"
                    "`/checkin` — 每日簽到與 streak bonus\n"
                    "`/vip` — 購買或查看 VIP（加成 check-in 與 Blackjack reward）\n\n"
                    "**轉帳與排行**\n"
                    "`/give` — 轉帳給成員或 bot（收取 5% 轉帳稅並銷毀）\n"
                    "`/leaderboard` — 餘額排行榜\n"
                    "`/loss_leaderboard` — 今日輸錢榜\n"
                    "`/casino` — 賭場系統 ledger\n"
                    "`/pocat` — 機器人玩家錢包\n\n"
                    "**借貸**\n"
                    "`/credit` — 個人借貸\n"
                    "`/central_bank` — 央行借貸\n\n"
                    "**管理**\n"
                    "`/admin` — admin 餘額調整（amount 支援逗號格式）"
                ),
            ),
            "stocks": HelpSection(
                emoji="📈",
                label="模擬股市",
                summary="股票交易與行情",
                detail=(
                    "**`/stock`**\n"
                    "開啟 market board UI：價格、market context news、top-holder 與 recent-trade summary、position、7D chart。可在 supply cap 內 buy / short（long 受單人 49% float 限制），或 sell / cover 出場；股數可用時以張顯示。"
                ),
            ),
            "tools": HelpSection(
                emoji="🧰",
                label="實用工具",
                summary="影片下載、楓之谷、Threads、ping",
                detail=(
                    "`/download_video` — 從支援的平台下載影片（可選 quality）\n"
                    "`/maplestory` — 查怪物、裝備、卷軸、NPC、任務、地圖、物品和 stats\n"
                    "`/ping` — 檢查 bot response latency\n\n"
                    "**Threads 解析**\n"
                    "貼上 Threads.net 或 Threads.com URL，我會擷取貼文、回覆和媒體。"
                ),
            ),
        },
    ),
    Locale.ja: HelpGuide(
        title="🤖 ボット利用ガイド",
        intro="下のメニューからカテゴリを選ぶと、詳しいコマンドを表示します。",
        select_placeholder="カテゴリを選択…",
        overview_label="概要",
        sections={
            "ai": HelpSection(
                emoji="💬",
                label="AI チャット",
                summary="メンションやDMで会話・分析・生成",
                detail=(
                    "メンションまたはDMで始められます。\n"
                    "• 会話、質問への回答、記事の要約\n"
                    "• 添付画像やファイルの分析\n"
                    "• リクエストに応じた画像・短い動画の生成"
                ),
            ),
            "games": HelpSection(
                emoji="🎰",
                label="ゲーム",
                summary="Blackjack と Dragon Gate",
                detail=(
                    "**`/games blackjack`**\n"
                    "Blackjack テーブルを開始。5枚で bust していなければ勝ち、5枚以上の 21 は system-funded bonus を維持します。シューは同じチャンネルでラウンドをまたいで引き継がれ、残りが少なくなるとシャッフルし直します。`bet` はカンマ付き数字に対応し、`0` は all in です。1ベットの上限は100万です。\n\n"
                    "**`/games blackjack_history [member] [count]`**\n"
                    "プレイヤーの最近のブラックジャックの対局を共有テーブルで公開投稿：手札、賭け金、ディーラーの手札、結果。既定は自分・直近 10 件（最大 50）。\n\n"
                    "**`/games dragon_gate`**\n"
                    "Dragon Gate jackpot テーブルを開始します。"
                ),
            ),
            "economy": HelpSection(
                emoji="💰",
                label=f"{CURRENCY_NAME} / Economy",
                summary="残高・送金・チェックイン・ローン・ランキング",
                detail=(
                    "**日常・ステータス**\n"
                    "`/balance [member]` — balance、loan、holdings、net worth、VIP\n"
                    "`/checkin` — daily reward と streak bonus\n"
                    "`/vip` — VIP の購入・確認（check-in と Blackjack reward を強化）\n\n"
                    "**送金・ランキング**\n"
                    "`/give` — メンバーや bot への送金（5% の送金税を徴収して焼却）\n"
                    "`/leaderboard` — 残高ランキング\n"
                    "`/loss_leaderboard` — 本日の loss ランキング\n"
                    "`/casino` — casino system ledger\n"
                    "`/pocat` — bot player wallet\n\n"
                    "**ローン**\n"
                    "`/credit` — 個人ローン\n"
                    "`/central_bank` — 中央銀行ローン\n\n"
                    "**管理**\n"
                    "`/admin` — admin による残高調整（カンマ付き金額対応）"
                ),
            ),
            "stocks": HelpSection(
                emoji="📈",
                label="シミュレーション株式",
                summary="仮想銘柄の取引と相場",
                detail=(
                    "**`/stock`**\n"
                    "market board UI を開きます：price、market context news、top-holder / recent-trade summary、position、7D chart。supply cap 内で buy / short（long は float の 49% 上限）、sell / cover で決済できます。Share counts は可能なら lots 表示になります。"
                ),
            ),
            "tools": HelpSection(
                emoji="🧰",
                label="ツール",
                summary="動画DL・MapleStory・Threads・ping",
                detail=(
                    "`/download_video` — 対応サイトから動画をダウンロード（quality 選択可）\n"
                    "`/maplestory` — monster、equip、scroll、NPC、quest、map、item、stats を検索\n"
                    "`/ping` — bot の応答遅延を確認\n\n"
                    "**Threads パーサー**\n"
                    "Threads.net または Threads.com の URL を貼ると、投稿・返信・メディアを取得します。"
                ),
            ),
        },
    ),
}


def resolve_guide(locale: Locale | str) -> HelpGuide:
    """Returns the help guide for a locale, falling back to the default copy."""
    guide = HELP_CONTENT.get(locale)
    return guide if guide is not None else HELP_CONTENT["default"]
