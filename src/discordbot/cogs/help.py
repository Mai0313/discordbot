"""Localized help command content and embed builders."""

from datetime import datetime

import nextcord
from nextcord import Embed, Locale, Interaction
from nextcord.ext import commands

from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.cogs._economy.presentation import CURRENCY_NAME

_EMBED_FIELD_VALUE_LIMIT = 1024
_EMBED_FIELD_COUNT_LIMIT = 25
_EMBED_TOTAL_LENGTH_LIMIT = 6000
_MESSAGE_EMBED_COUNT_LIMIT = 10
_HELP_FIELD_NAME = "\u200b"

_HELP_CONTENT = {
    "default": {
        "title": "Bot Guide",
        "description": "A quick index of available commands.",
        "ai_chat": (
            "**AI Chat**\n"
            "Mention me or DM me for chat, image/file analysis, summaries, and media generation."
        ),
        "threads": (
            "**Threads Parser**\n"
            "Paste a Threads.net or Threads.com URL to extract posts, replies, and media."
        ),
        "video": (
            "**Video Download**\n"
            "`/download_video` downloads videos from supported platforms with an optional quality setting."
        ),
        "maplestory": (
            "**MapleStory Artale**\n"
            "`/maplestory` searches monsters, equipment, scrolls, NPCs, quests, maps, items, and stats."
        ),
        "points": (
            f"**{CURRENCY_NAME} / Economy**\n"
            "`/balance [member]` balance, loans, stock holdings, net worth, and VIP · `/give` transfer to members or bots · `/checkin` daily reward\n"
            "`/leaderboard` balance board · `/loss_leaderboard` loss board · `/casino` casino system ledger · `/pocat` bot player wallet\n"
            "`/credit` personal loans · `/central_bank` central-bank loans · `/admin` admin adjustments with comma-formatted amounts"
        ),
        "checkin": ("**Daily Check-in**\n`/checkin` claims the daily reward and streak bonus."),
        "stocks": (
            "**Simulated Stock Market**\n"
            "`/stock` opens the market board UI for prices, market-context news, compact top-holder and recent-trade summaries, positions, charts, supply-capped buy and short entries, 49%-capped long holdings, plus sell and cover exits. Share counts display in lots when possible."
        ),
        "vip": (
            "**VIP**\n"
            "`/vip` buys and checks VIP status for boosted check-in and Blackjack rewards."
        ),
        "games": (
            "**Games**\n"
            "`/games blackjack` starts a Blackjack table; five-card non-bust hands win, and five-or-more-card 21 keeps its system-funded bonus. `bet` accepts comma-formatted numbers, and `0` means all in. `/games dragon_gate` starts a Dragon Gate jackpot table."
        ),
        "ping": "**Ping**\n`/ping` checks the bot's response latency.",
    },
    Locale.zh_TW: {
        "title": "機器人使用指南",
        "description": "這裡只列出主要指令和用途。",
        "ai_chat": (
            "**AI 對話**\n"
            "tag 我或私訊我，可以聊天、分析圖片/檔案、摘要文章，也可以請我生成圖片或影片。"
        ),
        "threads": (
            "**Threads 解析**\n貼上 Threads.net 或 Threads.com URL，我會擷取貼文、回覆和媒體。"
        ),
        "video": ("**影片下載**\n`/download_video` 從支援的平台下載影片，也可以選 quality。"),
        "maplestory": (
            "**MapleStory Artale**\n"
            "`/maplestory` 查怪物、裝備、卷軸、NPC、任務、地圖、物品和 stats。"
        ),
        "points": (
            f"**{CURRENCY_NAME} / Economy**\n"
            "`/balance [member]` 餘額、借貸、stock holdings、淨資產和 VIP · `/give` 可轉給成員或 bot · `/checkin` 每日簽到\n"
            "`/leaderboard` 排行榜 board · `/loss_leaderboard` 今日輸錢榜 board · `/casino` 賭場系統 ledger · `/pocat` 機器人玩家錢包\n"
            "`/credit` 個人借貸 · `/central_bank` 央行借貸 · `/admin` admin 調整，amount 可輸入逗號格式"
        ),
        "checkin": ("**每日簽到**\n`/checkin` 領每日獎勵和 streak bonus。"),
        "stocks": (
            "**模擬股市**\n"
            "`/stock` 開啟 market board UI，可以看價格、參考 market context 的 news、compact top-holder / recent-trade summary、position、chart，也能建立受 supply cap 與單人 49% long holding cap 限制的 buy/short，或執行 sell/cover 出場；股數顯示會在可用時改用張。"
        ),
        "vip": ("**VIP**\n`/vip` 購買或查看 VIP 狀態，VIP 會加成 check-in 和 Blackjack reward。"),
        "games": (
            "**小遊戲**\n`/games blackjack` 開 21 點桌，五張未爆直接贏，五張或以上 21 保留 system-funded bonus；`bet` 可以輸入含逗號的數字，`0` 就是 all in。`/games dragon_gate` 開射龍門 jackpot 桌。"
        ),
        "ping": "**延遲測試**\n`/ping` 檢查 bot response latency。",
    },
    Locale.ja: {
        "title": "ボット利用ガイド",
        "description": "利用できる主なコマンドの一覧です。",
        "ai_chat": (
            "**AI チャット**\n"
            "メンションまたはDMで、会話、画像/ファイル分析、要約、メディア生成ができます。"
        ),
        "threads": (
            "**Threads パーサー**\n"
            "Threads.net または Threads.com の URL から投稿、返信、メディアを取得します。"
        ),
        "video": (
            "**動画ダウンロード**\n"
            "`/download_video` は対応サイトから動画をダウンロードし、quality も選べます。"
        ),
        "maplestory": (
            "**MapleStory Artale**\n"
            "`/maplestory` で monster、equip、scroll、NPC、quest、map、item、stats を検索できます。"
        ),
        "points": (
            f"**{CURRENCY_NAME} / Economy**\n"
            "`/balance [member]` balance、loan、stock holdings、net worth、VIP · `/give` transfer to members or bots · `/checkin` daily reward\n"
            "`/leaderboard` ranking board · `/loss_leaderboard` loss board · `/casino` casino system ledger · `/pocat` bot player wallet\n"
            "`/credit` personal loans · `/central_bank` central-bank loans · `/admin` admin adjustments with comma-formatted amounts"
        ),
        "checkin": (
            "**デイリーチェックイン**\n`/checkin` で daily reward と streak bonus を受け取れます。"
        ),
        "stocks": (
            "**シミュレーション株式市場**\n"
            "`/stock` で market board UI を開き、price、market context 付き news、compact top-holder / recent-trade summary、position、chart、supply cap と単一 user 49% long holding cap 付きの buy/short entry、sell/cover exit を扱えます。Share counts は可能な場合 lots 表示になります。"
        ),
        "vip": (
            "**VIP**\n"
            "`/vip` で VIP の購入と状態確認ができます。check-in と Blackjack reward が強化されます。"
        ),
        "games": (
            "**ゲーム**\n"
            "`/games blackjack` は 5枚で bust していなければ勝ち、5枚以上 21 は system-funded bonus を維持します。`bet` にカンマ付き数字を入力でき、`0` は all in です。`/games dragon_gate` は Dragon Gate jackpot table を開きます。"
        ),
        "ping": "**Ping**\n`/ping` で bot の応答遅延を確認します。",
    },
}

_SECTIONS = (
    "ai_chat",
    "threads",
    "video",
    "maplestory",
    "points",
    "checkin",
    "stocks",
    "vip",
    "games",
    "ping",
)


def _split_field_value(value: str, limit: int = _EMBED_FIELD_VALUE_LIMIT) -> list[str]:
    """Splits an embed field value without dropping any content."""
    if len(value) <= limit:
        return [value]

    chunks: list[str] = []
    remaining = value
    while len(remaining) > limit:
        newline_index = remaining.rfind("\n", 1, limit)
        space_index = remaining.rfind(" ", 1, limit)
        split_at = max(newline_index, space_index)

        if split_at == -1:
            chunks.append(remaining[:limit])
            remaining = remaining[limit:]
            continue

        chunk_end = split_at + 1
        chunks.append(remaining[:chunk_end])
        remaining = remaining[chunk_end:]

    if remaining:
        chunks.append(remaining)

    return chunks


def _new_help_embed(
    content: dict[str, str],
    requester_name: str,
    requester_avatar_url: str,
    timestamp: datetime,
    include_header: bool,
) -> Embed:
    embed = Embed(color=0x5865F2, timestamp=timestamp)
    if include_header:
        embed.title = content["title"]
        embed.description = content["description"]

    embed.set_footer(text=f"Requested by {requester_name}", icon_url=requester_avatar_url)
    return embed


def _build_help_embeds(
    locale: Locale | str, requester_name: str, requester_avatar_url: str
) -> list[Embed]:
    """Builds Discord-limit-safe help embeds for the requested locale."""
    content = _HELP_CONTENT.get(locale, _HELP_CONTENT["default"])
    timestamp = nextcord.utils.utcnow()
    embeds: list[Embed] = []
    embed = _new_help_embed(
        content=content,
        requester_name=requester_name,
        requester_avatar_url=requester_avatar_url,
        timestamp=timestamp,
        include_header=True,
    )

    for section in _SECTIONS:
        for value in _split_field_value(value=content[section]):
            if (
                len(embed.fields) >= _EMBED_FIELD_COUNT_LIMIT
                or len(embed) + len(_HELP_FIELD_NAME) + len(value) > _EMBED_TOTAL_LENGTH_LIMIT
            ):
                embeds.append(embed)
                embed = _new_help_embed(
                    content=content,
                    requester_name=requester_name,
                    requester_avatar_url=requester_avatar_url,
                    timestamp=timestamp,
                    include_header=False,
                )
            embed.add_field(name=_HELP_FIELD_NAME, value=value, inline=False)

    embeds.append(embed)
    return embeds


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

        embeds = _build_help_embeds(
            locale=interaction.locale,
            requester_name=interaction.user.display_name,
            requester_avatar_url=interaction.user.display_avatar.url,
        )

        for index in range(0, len(embeds), _MESSAGE_EMBED_COUNT_LIMIT):
            batch = embeds[index : index + _MESSAGE_EMBED_COUNT_LIMIT]
            await interaction.followup.send(
                embeds=batch,
                ephemeral=True,
                **embed_spacer_payload(embeds=batch, is_edit=False, target=interaction),
            )


def setup(bot: commands.Bot) -> None:
    """Adds the HelpCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(HelpCogs(bot), override=True)
