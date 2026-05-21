"""Localized help command content and embed builders."""

import nextcord
from nextcord import Embed, Locale, Interaction
from nextcord.ext import commands

from discordbot.typings.economy import (
    VIP_PURCHASE_COST,
    BASE_CHECKIN_REWARD_AMOUNT,
    BASE_MESSAGE_REWARD_AMOUNT,
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
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
            "**MapleStory Artale** — `/maplestory`\n"
            "Search game data:\n"
            "`/maplestory monster` · `/maplestory equip` · `/maplestory scroll` · "
            "`/maplestory npc` · `/maplestory quest` · `/maplestory map` · "
            "`/maplestory item` · `/maplestory stats`"
        ),
        "points": (
            f"**{CURRENCY_NAME}**\n"
            f"Every message earns {BASE_MESSAGE_REWARD_AMOUNT:,} {CURRENCY_NAME}; "
            "AI chat replies add token-based bonuses.\n"
            "`/balance` check your balance · `/leaderboard` show the global top 10 · "
            "`/loss_leaderboard` show today's accumulated casino losses\n"
            f"`/give` transfer {CURRENCY_NAME} · "
            "`/house` show the Blackjack dealer ledger P&L · "
            "`/admin` run admin-only tax adjustments\n"
            f"`/credit` handles personal credit status, borrow requests with approval/rejection/cancel buttons, repayment, and collection; requests auto-reject after {LOAN_PROPOSAL_TIMEOUT_SECONDS} seconds\n"
            f"`/central_bank` handles central-bank status, borrow requests with approval/rejection/cancel buttons, repayment, and collection; requests auto-reject after {LOAN_PROPOSAL_TIMEOUT_SECONDS} seconds.\n"
            "`/portfolio` shows cash, debt, and net worth.\n"
            "`/balance`, `/credit status`, `/central_bank status`, `/portfolio`, and `/vip` replies are private where personal data is involved. "
            "`/give`, loan requests and settlements, successful `/admin` tax adjustments, `/leaderboard`, "
            "`/loss_leaderboard`, and `/house` stay public and "
            "clean themselves up after 3 minutes."
        ),
        "checkin": (
            "**Daily Check-in** — `/checkin`\n"
            f"Claim {BASE_CHECKIN_REWARD_AMOUNT:,} {CURRENCY_NAME} once per day (Asia/Taipei). "
            "Consecutive days within a 7-day cycle add a streak bonus; the reply is "
            "ephemeral so only you see it."
        ),
        "vip": (
            "**VIP** — `/vip`\n"
            f"Buy permanent VIP for {VIP_PURCHASE_COST:,} {CURRENCY_NAME}. VIP gives 2x "
            "daily check-in rewards and 1.5x Blackjack winning payouts. "
            "`/vip`, `/balance`, and `/checkin` show the base number and the "
            "VIP-boosted number; Blackjack final results show the VIP bonus when it applies."
        ),
        "games": (
            "**Games** — `/games`\n"
            "`/games blackjack` opens a 21 lobby. Other players can join before the owner starts, "
            "single-player starts are still allowed, the table stake follows the owner's "
            "effective wager, and idle hands auto-stand after 180 seconds.\n"
            "Player actions: **Hit / Stand / Double Down / Split / Surrender**. Double doubles "
            "the wager and draws one card before auto-standing. Action buttons only appear "
            "when they are currently legal. Split is available on same-value pairs, so "
            "10/J/Q/K can split with each other; **Double after Split is disabled** (No DAS). Splitting "
            "Aces takes one card per hand and 21 on a split hand pays 1:1 (not natural 3:2). "
            "A hand that reaches exactly five cards totaling 21 auto-stands as five-card 21, "
            "keeps normal Blackjack settlement, and receives an extra 1x bet system bonus. "
            "If the dealer also totals 21, the main hand pushes but the five-card bonus still pays. "
            "Late Surrender loses the rounded-up half of the wager.\n"
            "When the dealer up-card is **A** the round pauses for an insurance side bet "
            "(½ of the original wager, requires a positive half-bet and uncommitted balance, pays 2:1 if the dealer "
            "peeks Blackjack). When the dealer up-card is **A or 10-value** the dealer peeks "
            "the hole card; a natural "
            "Blackjack settles the round immediately without offering player actions.\n"
            "During player actions the dealer hand is shown as `🂠 <up-card>`; the hole card "
            "is revealed only at final settlement or when peek confirms Blackjack.\n"
            "After every hand stands / busts, the dealer hits on 16 or below; at 17+ "
            "the AI dealer decides hit / stand, and the final embed labels automatic draws "
            "and AI decisions in the dealer action path.\n"
            "`/games dragon_gate` opens an In-Between table over a **global jackpot pool** "
            "shared across every table. The ante is fixed at 5,000 (into the pool), the "
            "minimum bet is 10,000, the maximum bet is the entire pool, and every bet "
            "settles into the player row and the pool the instant it lands. Adjacent "
            "non-pair pillars are redealt without a bet. Players can "
            "only lose down to balance 0; players who hit 0 automatically leave the table. "
            "leave mid-table via the Leave button; if their running delta is positive at "
            "leave / timeout, that surplus is refunded into the pool (逆贏不拿). The "
            "table ends when someone wins the whole pool, all players have left or hit 0, or no "
            "one has interacted for 180 seconds. Whole-pool wins auto-reseed the jackpot "
            "to 100,000 without affecting `/house`.\n"
            "Final game messages are cleaned up after 3 minutes."
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
            "**MapleStory Artale** — `/maplestory`（`/楓之谷`）\n"
            "查詢遊戲資料：\n"
            "`/maplestory monster` 怪物 · `/maplestory equip` 裝備 · `/maplestory scroll` 卷軸 · "
            "`/maplestory npc` · `/maplestory quest` 任務 · `/maplestory map` 地圖 · "
            "`/maplestory item` 物品 · `/maplestory stats` 統計"
        ),
        "points": (
            f"**{CURRENCY_NAME}**\n"
            f"每則訊息會獲得 {BASE_MESSAGE_REWARD_AMOUNT:,} {CURRENCY_NAME}, "
            f"AI chat 回覆另外追加 token bonus, {CURRENCY_NAME}跨 server 共用\n"
            "`/balance` 查餘額、債務與淨資產 · `/leaderboard` 看全域前 10 名 · "
            "`/loss_leaderboard` 看今日輸局累計前 10 名\n"
            "`/give` 轉帳 · `/house` 看 Blackjack 莊家累積 P&L · `/admin` 管理員退稅/收稅\n"
            f"`/credit` 處理個人信貸狀態、借款申請、按鈕批准或拒絕、取消、還款與催收；申請 {LOAN_PROPOSAL_TIMEOUT_SECONDS} 秒後自動拒絕\n"
            f"`/central_bank` 處理央行狀態、借款申請、按鈕批准或拒絕、取消、還款與催收；申請 {LOAN_PROPOSAL_TIMEOUT_SECONDS} 秒後自動拒絕\n"
            "`/portfolio` 查現金、債務與淨資產\n"
            "`/balance`、`/credit status`、`/central_bank status`、`/portfolio`、`/vip` 涉及個人資料時會私密回覆\n"
            "`/give`、信貸與央行借款的申請/還款/催收、`/admin` 成功結果、`/leaderboard`、`/loss_leaderboard`、"
            "`/house` 維持公開, 3 分鐘後自動清掉"
        ),
        "checkin": (
            "**每日簽到** — `/checkin`\n"
            f"每天可以領 {BASE_CHECKIN_REWARD_AMOUNT:,} {CURRENCY_NAME} (Asia/Taipei), "
            "連續七天為一個 cycle, 每天加成. 訊息是 ephemeral 只有自己看得到"
        ),
        "vip": (
            "**VIP** — `/vip`\n"
            f"花 {VIP_PURCHASE_COST:,} {CURRENCY_NAME}購買永久 VIP\n"
            "VIP 會讓每日簽到 2x、Blackjack 贏局 payout 1.5x\n"
            "`/vip`、`/balance`、`/checkin` 會顯示原本數字與 VIP加成後數字, "
            "Blackjack final result 也會在套用時顯示 VIP加成"
        ),
        "games": (
            "**小遊戲** — `/games`（`/小遊戲`）\n"
            "`/games blackjack` 會開一個 21 點 lobby，其他玩家可以先加入，只有房主能開始，"
            "單人也可以直接開始，房主超過餘額的 bet 會用實際餘額當 table stake，"
            "後續玩家預設跟這個金額，"
            "不操作 180 秒會自動 stand\n"
            "玩家動作：**Hit / Stand / Double / Split / Surrender**\n"
            "只有目前合法的 action button 會出現。Double 加倍下注後只再抽一張即強制停牌；"
            "Split 可分同 value pair，所以 10/J/Q/K 可互分，"
            "**Split 後禁止 Double**（No DAS），Split Aces 兩手各只能再拿一張，"
            "且 21 點算一般 win（不是 natural Blackjack 3:2）；"
            "單手剛好第 5 張湊成 21 會自動 Stand 成過五關，主局照一般 Blackjack 結算，"
            "另外由 system 支付 1x bet bonus。若莊家也 21，主局平手但過五關 bonus 照給；"
            "Late Surrender 會輸掉進位後的一半本金\n"
            "莊家明牌 **A** 時會先進入保險階段（保險注 = 原注一半，必須是正數且需要剩餘可用餘額，"
            "莊家 BJ 賠 2:1）；"
            "明牌 **A 或 10 點** 莊家會 peek 暗牌，若 peek 出 Blackjack 立即結算，"
            "玩家無法做動作\n"
            "玩家操作期間莊家手牌會顯示成 `🂠 <明牌>`，暗牌只會在 final settlement "
            "或 peek 確認 Blackjack 時攤開\n"
            "所有玩家結束後莊家 16 點以下固定 hit；17 點以上交給 AI 莊家決定 hit / stand；"
            "final result 的 dealer action path 會標出自動抽牌與 AI 決定\n"
            "`/games dragon_gate` 開一桌射龍門, 共用一個**全域累計彩金池**, 所有桌都看到同一池"
            "入場費固定 5,000 點(進彩金池), 最低下注 10,000, 上限就是當下彩金池\n"
            "每次下注後玩家餘額與彩金池同步即時結算, 不再等桌結束\n"
            "輸錢最多只會扣到餘額 0, 歸零玩家會自動離桌, 其他玩家繼續玩\n"
            "相鄰且不同點的門柱沒有龍門, 會直接重發, 不會下注\n"
            "玩家可隨時按「離桌」中途退出, 不影響其他玩家繼續玩\n"
            "離桌或 180 秒無互動超時時, 若該玩家當下淨贏 > 0, 該部分會逆向退回彩金池(逆贏不拿)"
            "整桌結束的條件是彩金池被全池贏走, 所有玩家都離桌或歸零, 或 180 秒無人互動\n"
            "全池被贏走時系統會自動補回 100,000, 不算在 `/house`\n"
            "final game message 會在 3 分鐘後清掉"
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
            "**MapleStory Artale** — `/maplestory`\n"
            "ゲームデータ検索：\n"
            "`/maplestory monster` · `/maplestory equip` · `/maplestory scroll` · "
            "`/maplestory npc` · `/maplestory quest` · `/maplestory map` · "
            "`/maplestory item` · `/maplestory stats`"
        ),
        "points": (
            f"**{CURRENCY_NAME}**\n"
            f"すべてのメッセージで{BASE_MESSAGE_REWARD_AMOUNT:,}{CURRENCY_NAME}を獲得し、"
            "AI チャット返信ではtokenベースのボーナスも入ります。\n"
            "`/balance` 残高と借入状況を確認 · `/leaderboard` グローバルトップ10 · "
            "`/loss_leaderboard` 本日の累計負け額トップ10\n"
            f"`/give` {CURRENCY_NAME}送付 · `/house` Blackjack dealer ledger P&L\n"
            "`/admin` admin-only tax adjustments\n"
            f"`/credit` personal credit status、borrow request buttons、repayment、collection。requests は {LOAN_PROPOSAL_TIMEOUT_SECONDS} 秒後に auto-reject。\n"
            f"`/central_bank` central-bank status、borrow request buttons、repayment、collection、capacity。requests は {LOAN_PROPOSAL_TIMEOUT_SECONDS} 秒後に auto-reject。\n"
            "`/portfolio` cash、debt、net worth。\n"
            "`/balance`、`/credit status`、`/central_bank status`、`/portfolio`、`/vip` は個人情報を含む場合 private reply です。"
            "`/give`、loan requests and settlements、successful `/admin` tax adjustments、`/leaderboard`、"
            "`/loss_leaderboard`、`/house` は公開のまま3分後に自動削除されます。"
        ),
        "checkin": (
            "**デイリーチェックイン** — `/checkin`\n"
            f"毎日{BASE_CHECKIN_REWARD_AMOUNT:,}{CURRENCY_NAME}を受け取れます (Asia/Taipei)。"
            "7日サイクルで連続日数ボーナスが付き、返信は ephemeral で本人のみ閲覧可能。"
        ),
        "vip": (
            "**VIP** — `/vip`\n"
            f"{VIP_PURCHASE_COST:,}{CURRENCY_NAME}で永久 VIP を購入。"
            "VIP は check-in 2x、Blackjack 勝利 payout 1.5x。"
            "`/vip`、`/balance`、`/checkin` は通常値と VIP 後の値を表示し、"
            "Blackjack final result も適用時に VIP bonus を表示します。"
        ),
        "games": (
            "**ゲーム** — `/games`\n"
            "`/games blackjack` 21の lobby を開きます。他のプレイヤーは owner が開始する前に参加でき、"
            "1人でも開始できます。owner の有効ベットが table stake になり、"
            "参加者はその金額を既定で賭けます。"
            "180秒操作がない場合は自動 stand。\n"
            "プレイヤー action: **Hit / Stand / Double / Split / Surrender**。"
            "現在合法な action button だけ表示されます。"
            "Double はベットを倍にして1枚だけ引いて自動 stand。Split は同 value pair で可、"
            "10/J/Q/K は相互に split できます。"
            "**Split 後は Double 不可** (No DAS)、Split Aces は各手1枚だけで finished、"
            "21 になっても natural BJ 扱いではなく 1:1。"
            "1手が5枚ちょうどで21になると自動 Stand の five-card 21 になり、"
            "通常の Blackjack settlement に加えて system から 1x bet bonus が支払われます。"
            "dealer も21なら main hand は push ですが five-card bonus は支払われます。"
            "Late Surrender は切り上げ半額を失います。\n"
            "dealer up-card が **A** のとき insurance phase が入り "
            "(原 bet の半額、positive half-bet と未committed balance が必要、dealer BJ なら 2:1)、"
            "up-card が **A または 10 点** のときは dealer が hole card を peek し、"
            "natural Blackjack なら即時清算します。\n"
            "player action 中の dealer hand は `🂠 <up-card>` と表示され、hole card は "
            "final settlement または peek で Blackjack 確認時だけ reveal されます。\n"
            "全 action 後、dealer は 16 以下で hit、17 以上は AI dealer が hit / stand を決め、"
            "final result の dealer action path は自動 draw と AI decision を区別して表示します。\n"
            "`/games dragon_gate` は全 table で共有する**グローバルジャックポット**を巡る "
            "In-Between table を開きます。anteは固定 5,000 (pool へ)、最低 bet は 10,000、"
            "上限は pool の全額、各 bet は player 残高と pool に即時反映されます。\n"
            "loss は残高 0 までに clamp され、0 になった player は自動で退場します。"
            "隣り合う non-pair の柱は gate なしとして bet せず引き直します。"
            "「離桌」ボタンで途中退場可能で他のプレイヤーは継続。退場 / 180 秒の無操作で "
            "running delta が正なら、その分は pool へ返戻されます (逆贏不拿)。"
            "table は pool 全額勝利 / 全員退場または残高 0 / 180 秒の無操作で終了します。"
            "pool 全額勝利時は system が jackpot を 100,000 へ自動補充し、`/house` には影響しません。\n"
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
