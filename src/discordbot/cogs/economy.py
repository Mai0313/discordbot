"""Slash commands that surface point balances, the leaderboard, and transfers."""

import nextcord
from nextcord import Embed, Locale, Member, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.cogs._games.cleanup import schedule_game_message_delete
from discordbot.cogs._economy.database import top_n, transfer, get_account, get_balance
from discordbot.cogs._economy.presentation import CURRENCY_NAME, currency_text

_BALANCE_COLOR = 0x57F287
_LEADERBOARD_COLOR = 0xFEE75C
_TRANSFER_COLOR = 0x5865F2
_HOUSE_COLOR = 0xEB459E
_ERROR_COLOR = 0xED4245

_LEADERBOARD_LIMIT = 10
_LEADERBOARD_MEDALS = ["🥇", "🥈", "🥉"]


async def _send_expiring_followup(*, interaction: Interaction, embed: Embed) -> None:
    """Sends a game-related economy embed and schedules its cleanup."""
    message = await interaction.followup.send(embed=embed, wait=True)
    schedule_game_message_delete(message=message)


class EconomyCogs(commands.Cog):
    """Player-facing point balance commands.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initialises the EconomyCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot

    @nextcord.slash_command(
        name="balance",
        description=f"Check your current {CURRENCY_NAME} balance.",
        name_localizations={Locale.zh_TW: "餘額", Locale.ja: "残高"},
        description_localizations={
            Locale.zh_TW: f"查詢你目前的{CURRENCY_NAME}餘額。",
            Locale.ja: f"現在の{CURRENCY_NAME}残高を確認します。",
        },
        nsfw=False,
    )
    async def balance(self, interaction: Interaction) -> None:
        """Replies with the caller's current balance.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return
        amount = await get_balance(user_id=interaction.user.id)
        embed = Embed(
            title=f":coin: {CURRENCY_NAME}餘額",
            description=f"{interaction.user.mention} 目前持有 **{currency_text(amount=amount)}**。",
            color=_BALANCE_COLOR,
        )
        embed.set_thumbnail(url=interaction.user.display_avatar.url)
        embed.set_footer(
            text=f"跟機器人聊天可以累積{CURRENCY_NAME}, 輸入 /dice、/blackjack 或 /dragon_gate 來下注。"
        )
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="leaderboard",
        description=f"Show the global top {CURRENCY_NAME} holders.",
        name_localizations={Locale.zh_TW: "排行榜", Locale.ja: "リーダーボード"},
        description_localizations={
            Locale.zh_TW: f"顯示 global {CURRENCY_NAME}前 10 名。",
            Locale.ja: f"グローバル{CURRENCY_NAME}トップ10を表示します。",
        },
        nsfw=False,
    )
    async def leaderboard(self, interaction: Interaction) -> None:
        """Replies with the top 10 point holders.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        # Exclude the bot's own house-ledger row so the casino's house P&L
        # never crowds out real players on the leaderboard.
        exclude_user_ids = (self.bot.user.id,) if self.bot.user else ()
        rows = await top_n(limit=_LEADERBOARD_LIMIT, exclude_user_ids=exclude_user_ids)
        if not rows:
            embed = Embed(
                title=f":trophy: {CURRENCY_NAME}排行榜",
                description=f"目前還沒有人有{CURRENCY_NAME}。",
                color=_LEADERBOARD_COLOR,
            )
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        lines: list[str] = []
        for index, (_, name, amount) in enumerate(iterable=rows):
            prefix = (
                _LEADERBOARD_MEDALS[index] if index < len(_LEADERBOARD_MEDALS) else f"#{index + 1}"
            )
            lines.append(f"{prefix} **{name}** — {currency_text(amount=amount)}")

        embed = Embed(
            title=f":trophy: {CURRENCY_NAME}排行榜",
            description="\n".join(lines),
            color=_LEADERBOARD_COLOR,
        )
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="give",
        description=f"Transfer your {CURRENCY_NAME} to another member.",
        name_localizations={Locale.zh_TW: "轉虛擬歡樂豆", Locale.ja: "虛擬歡樂豆送付"},
        description_localizations={
            Locale.zh_TW: f"把你的{CURRENCY_NAME}轉給其他成員。",
            Locale.ja: f"他のメンバーに{CURRENCY_NAME}を送ります。",
        },
        nsfw=False,
    )
    async def give(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description=f"The member to receive the {CURRENCY_NAME}.",
            name_localizations={Locale.zh_TW: "對象", Locale.ja: "受取人"},
            description_localizations={
                Locale.zh_TW: f"要接收{CURRENCY_NAME}的成員。",
                Locale.ja: f"{CURRENCY_NAME}を受け取るメンバー。",
            },
            required=True,
        ),
        amount: int = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to transfer (must be positive).",
            name_localizations={Locale.zh_TW: "虛擬歡樂豆", Locale.ja: "虛擬歡樂豆"},
            description_localizations={
                Locale.zh_TW: f"要轉的{CURRENCY_NAME} (必須大於 0)。",
                Locale.ja: f"送る{CURRENCY_NAME} (1以上)。",
            },
            required=True,
            min_value=1,
        ),
    ) -> None:
        """Transfers points from the caller to ``member``.

        Args:
            interaction: The interaction that triggered the command.
            member: The recipient.
            amount: How many points to transfer.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return

        if member.bot:
            await interaction.followup.send(
                embed=Embed(
                    title=":x: 轉虛擬歡樂豆失敗",
                    description=f"不能把{CURRENCY_NAME}轉給機器人。",
                    color=_ERROR_COLOR,
                )
            )
            return
        if member.id == interaction.user.id:
            await interaction.followup.send(
                embed=Embed(
                    title=":x: 轉虛擬歡樂豆失敗", description="不能轉給自己。", color=_ERROR_COLOR
                )
            )
            return

        transfer_result = await transfer(
            sender_id=interaction.user.id,
            sender_name=interaction.user.name,
            receiver_id=member.id,
            receiver_name=member.name,
            amount=amount,
        )
        if transfer_result is None:
            balance_now = await get_balance(user_id=interaction.user.id)
            await interaction.followup.send(
                embed=Embed(
                    title=":x: 轉虛擬歡樂豆失敗",
                    description=(
                        f"餘額不足, 你目前只有 **{currency_text(amount=balance_now)}**, "
                        f"想轉 **{currency_text(amount=amount)}**。"
                    ),
                    color=_ERROR_COLOR,
                )
            )
            return

        embed = Embed(
            title=":handshake: 轉虛擬歡樂豆成功",
            description=(
                f"{interaction.user.mention} → {member.mention}: "
                f"**{currency_text(amount=amount)}**\n"
                f"你剩下 **{currency_text(amount=transfer_result.sender_balance)}**, "
                f"{member.display_name} 現在有 "
                f"**{currency_text(amount=transfer_result.receiver_balance)}**。"
            ),
            color=_TRANSFER_COLOR,
        )
        await interaction.followup.send(embed=embed)

    @nextcord.slash_command(
        name="house",
        description="Show the dealer's running win/loss across every game.",
        name_localizations={Locale.zh_TW: "莊家戰績", Locale.ja: "ディーラー戦績"},
        description_localizations={
            Locale.zh_TW: "顯示莊家在所有遊戲累積的輸贏 (跨伺服器)。",
            Locale.ja: "ディーラーの全サーバー累計の勝敗を表示します。",
        },
        nsfw=False,
    )
    async def house(self, interaction: Interaction) -> None:
        """Shows the bot's accumulated dealer P&L across `/dice` and `/blackjack`.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        if self.bot.user is None:
            await _send_expiring_followup(
                interaction=interaction,
                embed=Embed(
                    title=":x: 無法查詢",
                    description="目前無法取得機器人身份。",
                    color=_ERROR_COLOR,
                ),
            )
            return

        account = await get_account(user_id=self.bot.user.id)
        # No row yet means nobody's played a round; show a fresh-house view
        # rather than treating it as an error.
        name = self.bot.user.display_name
        if account is None:
            balance, total_earned, total_spent = 0, 0, 0
        else:
            _, balance, total_earned, total_spent = account
            name = name or account[0]

        if balance > 0:
            verdict = f"莊家目前淨贏 **{currency_text(amount=balance)}**。"
        elif balance < 0:
            verdict = f"莊家目前淨虧 **{currency_text(amount=abs(balance))}**。"
        else:
            verdict = "莊家目前剛好打平。"

        embed = Embed(
            title=f":game_die: {name} - 莊家戰績", description=verdict, color=_HOUSE_COLOR
        )
        embed.add_field(
            name="莊家從玩家身上贏到", value=currency_text(amount=total_earned), inline=True
        )
        embed.add_field(name="莊家賠給玩家", value=currency_text(amount=total_spent), inline=True)
        embed.set_footer(text="跨伺服器累積; 莊家資金無上限, 餘額可為負。")
        await _send_expiring_followup(interaction=interaction, embed=embed)


def setup(bot: commands.Bot) -> None:
    """Adds the EconomyCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(EconomyCogs(bot), override=True)
