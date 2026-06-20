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

    emoji: str = Field(..., description="Leading emoji for the select option and detail title.")
    label: str = Field(
        ..., description="Category name used as the select option label and embed title."
    )
    summary: str = Field(
        ..., description="One-line index/select-option description (kept under 100 chars)."
    )
    detail: str = Field(
        ..., description="Full command list rendered in the category's detail embed."
    )


class HelpGuide(BaseModel):
    """Localized help content for one locale.

    Attributes:
        title: Overview embed title.
        intro: Leading line on the overview embed.
        select_placeholder: Placeholder shown on the category select menu.
        overview_label: Select option label that returns to the overview.
        sections: Category key to its content, keyed by `CATEGORY_ORDER`.
    """

    title: str = Field(..., description="Overview embed title.")
    intro: str = Field(..., description="Leading line on the overview embed.")
    select_placeholder: str = Field(
        ..., description="Placeholder shown on the category select menu."
    )
    overview_label: str = Field(
        ..., description="Select option label that returns to the overview."
    )
    sections: dict[str, HelpSection] = Field(
        ..., description="Category key to its content, keyed by `CATEGORY_ORDER`."
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
        intro="Pick a category below for its full commands; to use me directly, just mention or DM me.",
        select_placeholder="Choose a category…",
        overview_label="Overview",
        sections={
            "ai": HelpSection(
                emoji="💬",
                label="AI Chat",
                summary="Mention or DM me to chat, read images and files, and generate",
                detail=(
                    "Mention me or DM me to get started: chat, answer questions, summarize "
                    "articles, read your images and files, watch linked YouTube videos, and "
                    "generate images, or short videos that can use your attached images or a "
                    "referenced video as reference. I slowly learn your preferences "
                    "and may use them in a reply, marked with a 🧠 note.\n\n"
                    "Memory\n"
                    "`/memory show` — see what I remember about you\n"
                    "`/memory regenerate` — rebuild my memory of you in the background\n"
                    "`/memory server show` — see what I remember about this server"
                ),
            ),
            "games": HelpSection(
                emoji="🎰",
                label="Casino Games",
                summary="Blackjack, Dragon Gate, and Fishing",
                detail=(
                    "Games\n"
                    "`/games blackjack` — open a Blackjack table; a five-card non-bust hand wins\n"
                    "`/games dragon_gate` — open a Dragon Gate table with a shared jackpot\n"
                    "`/games fishing` — open the fishing panel: buy a rod and bait, then cast\n\n"
                    "History\n"
                    "`/games blackjack_history [member] [count]` — recent Blackjack rounds"
                ),
            ),
            "economy": HelpSection(
                emoji="💰",
                label=f"{CURRENCY_NAME} / Economy",
                summary="Balance, transfers, check-in, loans, and boards",
                detail=(
                    "Daily\n"
                    "`/balance` — balance, holdings, and net worth\n"
                    "`/checkin` — daily reward\n"
                    "`/vip` — buy VIP (boosts check-in and Blackjack)\n\n"
                    "Transfers & boards\n"
                    "`/give` — send to someone (5% transfer tax)\n"
                    "`/leaderboard` — wealth board\n"
                    "`/loss_leaderboard` — today's biggest losses\n"
                    "`/casino` — casino system profit and loss\n"
                    "`/pocat` — the bot player's wallet\n\n"
                    "Loans\n"
                    "`/credit` — personal loans: borrow, repay, or collect\n"
                    "`/central_bank` — central-bank loans: borrow or repay\n\n"
                    "Admin\n"
                    "`/admin` — admin balance adjustments"
                ),
            ),
            "stocks": HelpSection(
                emoji="📈",
                label="Simulated Stocks",
                summary="Trade and track virtual companies",
                detail=(
                    "`/stock` — open the market panel: prices, news, your positions, and charts. "
                    "Buy, short, sell, or cover right from the panel."
                ),
            ),
            "tools": HelpSection(
                emoji="🧰",
                label="Tools",
                summary="Video download, MapleStory lookup, Threads, ping",
                detail=(
                    "`/download_video` — download a video from supported platforms\n"
                    "`/deep_research` — kick off a long, cited research report in a thread\n"
                    "`/maplestory` — look up MapleStory monsters, equipment, scrolls, NPCs, "
                    "quests, maps, and items\n"
                    "`/ping` — check the bot's response latency\n\n"
                    "Threads parser\n"
                    "Paste a Threads link and I'll pull the posts, replies, and media."
                ),
            ),
        },
    ),
    Locale.zh_TW: HelpGuide(
        title="🤖 機器人使用指南",
        intro="從下方選單選一個分類看完整指令;想直接用,tag 我或私訊我就行。",
        select_placeholder="選擇分類…",
        overview_label="總覽",
        sections={
            "ai": HelpSection(
                emoji="💬",
                label="AI 對話",
                summary="tag 或私訊我就能聊天、看圖看檔、生成圖片",
                detail=(
                    "tag 我或私訊我就能開始:聊天、回答問題、摘要文章、分析你附上的圖片和檔案、"
                    "看你貼的 YouTube 影片,也能依需求生成圖片,或用你附加的圖片或引用的影片當參考生成短影片。我會慢慢記住你的偏好,回覆時可能參考並用 🧠 標註。\n\n"
                    "記憶\n"
                    "`/memory show` — 看我記得你什麼\n"
                    "`/memory regenerate` — 在背景重建我對你的記憶\n"
                    "`/memory server show` — 看我記得這個伺服器什麼"
                ),
            ),
            "games": HelpSection(
                emoji="🎰",
                label="賭場遊戲",
                summary="Blackjack、射龍門、釣魚",
                detail=(
                    "遊戲\n"
                    "`/games blackjack` — 開 21 點賭桌,五張未爆直接贏\n"
                    "`/games dragon_gate` — 開射龍門,共享 jackpot 彩池\n"
                    "`/games fishing` — 開釣魚面板:買竿買餌,拋竿釣魚\n\n"
                    "紀錄\n"
                    "`/games blackjack_history [member] [count]` — 查近期 21 點對局"
                ),
            ),
            "economy": HelpSection(
                emoji="💰",
                label=f"{CURRENCY_NAME} / 經濟",
                summary="餘額、轉帳、簽到、借貸、排行榜",
                detail=(
                    "日常\n"
                    "`/balance` — 查餘額、持股、淨資產\n"
                    "`/checkin` — 每日簽到領獎勵\n"
                    "`/vip` — 購買 VIP(簽到與 Blackjack 加成)\n\n"
                    "轉帳與排行\n"
                    "`/give` — 轉給其他人(扣 5% 稅)\n"
                    "`/leaderboard` — 財富排行榜\n"
                    "`/loss_leaderboard` — 今日輸最多榜\n"
                    "`/casino` — 賭場系統盈虧\n"
                    "`/pocat` — 機器人玩家的錢包\n\n"
                    "借貸\n"
                    "`/credit` — 個人借貸:跟其他人借、還、催收\n"
                    "`/central_bank` — 央行借貸:跟系統借、還\n\n"
                    "管理\n"
                    "`/admin` — 管理員餘額調整"
                ),
            ),
            "stocks": HelpSection(
                emoji="📈",
                label="模擬股市",
                summary="買賣與追蹤虛擬公司",
                detail=(
                    "`/stock` — 開啟股市面板:看股價、新聞、持股與走勢圖,"
                    "在面板上買進、放空、賣出或回補。"
                ),
            ),
            "tools": HelpSection(
                emoji="🧰",
                label="實用工具",
                summary="影片下載、楓之谷查詢、Threads、ping",
                detail=(
                    "`/download_video` — 從支援的平台下載影片\n"
                    "`/deep_research` — 開一條 thread 進行帶引用的深度研究\n"
                    "`/maplestory` — 查楓之谷的怪物、裝備、卷軸、NPC、任務、地圖、物品\n"
                    "`/ping` — 看機器人的回應延遲\n\n"
                    "Threads 解析\n"
                    "貼上 Threads 連結,我會自動擷取貼文、回覆和媒體。"
                ),
            ),
        },
    ),
    Locale.ja: HelpGuide(
        title="🤖 ボット利用ガイド",
        intro="下のメニューからカテゴリを選ぶと詳しいコマンドを表示します。直接使うにはメンションかDMでどうぞ。",
        select_placeholder="カテゴリを選択…",
        overview_label="概要",
        sections={
            "ai": HelpSection(
                emoji="💬",
                label="AI チャット",
                summary="メンションやDMで会話・画像/ファイル読取・生成",
                detail=(
                    "メンションまたはDMで始められます:会話、質問への回答、記事の要約、"
                    "添付画像やファイルの読み取り、貼られた YouTube 動画の視聴、"
                    "リクエストに応じた画像生成や、添付画像・参照動画を参考にした短い動画の生成。"
                    "あなたの好みを少しずつ覚え、返信時に参照して 🧠 で示すことがあります。\n\n"
                    "メモリー\n"
                    "`/memory show` — 覚えている内容を表示\n"
                    "`/memory regenerate` — バックグラウンドで記憶を作り直す\n"
                    "`/memory server show` — このサーバーについて覚えている内容を表示"
                ),
            ),
            "games": HelpSection(
                emoji="🎰",
                label="ゲーム",
                summary="Blackjack、Dragon Gate、釣り",
                detail=(
                    "ゲーム\n"
                    "`/games blackjack` — Blackjack テーブルを開く(5枚で未バーストなら勝ち)\n"
                    "`/games dragon_gate` — 共有ジャックポットの Dragon Gate を開く\n"
                    "`/games fishing` — 釣りパネルを開く:竿と餌を買って釣る\n\n"
                    "履歴\n"
                    "`/games blackjack_history [member] [count]` — 最近の Blackjack 対局"
                ),
            ),
            "economy": HelpSection(
                emoji="💰",
                label=f"{CURRENCY_NAME} / Economy",
                summary="残高・送金・チェックイン・ローン・ランキング",
                detail=(
                    "日常\n"
                    "`/balance` — 残高・保有・純資産\n"
                    "`/checkin` — 毎日のチェックイン報酬\n"
                    "`/vip` — VIP を購入(チェックインと Blackjack を強化)\n\n"
                    "送金・ランキング\n"
                    "`/give` — 誰かに送金(5% の送金税)\n"
                    "`/leaderboard` — 資産ランキング\n"
                    "`/loss_leaderboard` — 本日の負けランキング\n"
                    "`/casino` — カジノ全体の損益\n"
                    "`/pocat` — ボットプレイヤーの財布\n\n"
                    "ローン\n"
                    "`/credit` — 個人ローン:借りる・返す・取り立て\n"
                    "`/central_bank` — 中央銀行ローン:借りる・返す\n\n"
                    "管理\n"
                    "`/admin` — 管理者による残高調整"
                ),
            ),
            "stocks": HelpSection(
                emoji="📈",
                label="シミュレーション株式",
                summary="仮想銘柄の取引と相場確認",
                detail=(
                    "`/stock` — 株式パネルを開く:株価・ニュース・保有・チャートを確認し、"
                    "パネルから買い・空売り・売り・買い戻しができます。"
                ),
            ),
            "tools": HelpSection(
                emoji="🧰",
                label="ツール",
                summary="動画DL・MapleStory検索・Threads・ping",
                detail=(
                    "`/download_video` — 対応サイトから動画をダウンロード\n"
                    "`/deep_research` — スレッドで引用付きのディープリサーチを実行\n"
                    "`/maplestory` — MapleStory のモンスター・装備・巻物・NPC・"
                    "クエスト・マップ・アイテムを検索\n"
                    "`/ping` — ボットの応答遅延を確認\n\n"
                    "Threads パーサー\n"
                    "Threads のリンクを貼ると、投稿・返信・メディアを取得します。"
                ),
            ),
        },
    ),
}


def resolve_guide(locale: Locale | str) -> HelpGuide:
    """Returns the help guide for a locale, falling back to the default copy."""
    guide = HELP_CONTENT.get(locale)
    return guide if guide is not None else HELP_CONTENT["default"]
