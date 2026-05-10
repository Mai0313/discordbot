"""Slash commands that surface point balances, the leaderboard, and peer transfers."""

import nextcord
from nextcord import Embed, Locale, Member, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.cogs._economy.database import top_n, transfer, get_account, get_balance

_BALANCE_COLOR = 0x57F287
_LEADERBOARD_COLOR = 0xFEE75C
_TRANSFER_COLOR = 0x5865F2
_HOUSE_COLOR = 0xEB459E
_ERROR_COLOR = 0xED4245

_LEADERBOARD_LIMIT = 10
_LEADERBOARD_MEDALS = ["🥇", "🥈", "🥉"]


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
        description="Check your current point balance.",
        name_localizations={Locale.zh_TW: "餘額", Locale.ja: "残高"},
        description_localizations={
            Locale.zh_TW: "查詢你目前的點數餘額。",
            Locale.ja: "現在のポイント残高を確認します。",
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
            title=":coin: 點數餘額",
            description=f"{interaction.user.mention} 目前持有 **{amount:,}** 點。",
            color=_BALANCE_COLOR,
        )
        embed.set_footer(text="跟機器人聊天可以累積點數, 輸入 /dice 或 /blackjack 來下注。")
        await interaction.followup.send(embed=embed)

    @nextcord.slash_command(
        name="leaderboard",
        description="Show the top point holders on this server.",
        name_localizations={Locale.zh_TW: "排行榜", Locale.ja: "リーダーボード"},
        description_localizations={
            Locale.zh_TW: "顯示伺服器內點數前 10 名。",
            Locale.ja: "サーバーのポイントトップ10を表示します。",
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
                title=":trophy: 點數排行榜",
                description="目前還沒有人有點數。",
                color=_LEADERBOARD_COLOR,
            )
            await interaction.followup.send(embed=embed)
            return

        lines: list[str] = []
        for index, (_, name, amount) in enumerate(iterable=rows):
            prefix = (
                _LEADERBOARD_MEDALS[index] if index < len(_LEADERBOARD_MEDALS) else f"#{index + 1}"
            )
            lines.append(f"{prefix} **{name}** — {amount:,} 點")

        embed = Embed(
            title=":trophy: 點數排行榜", description="\n".join(lines), color=_LEADERBOARD_COLOR
        )
        await interaction.followup.send(embed=embed)

    @nextcord.slash_command(
        name="give",
        description="Transfer your points to another member.",
        name_localizations={Locale.zh_TW: "轉點", Locale.ja: "ポイント送付"},
        description_localizations={
            Locale.zh_TW: "把你的點數轉給其他成員。",
            Locale.ja: "他のメンバーにポイントを送ります。",
        },
        nsfw=False,
    )
    async def give(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="The member to receive the points.",
            name_localizations={Locale.zh_TW: "對象", Locale.ja: "受取人"},
            description_localizations={
                Locale.zh_TW: "要接收點數的成員。",
                Locale.ja: "ポイントを受け取るメンバー。",
            },
            required=True,
        ),
        amount: int = SlashOption(
            name="amount",
            description="How many points to transfer (must be positive).",
            name_localizations={Locale.zh_TW: "點數", Locale.ja: "ポイント数"},
            description_localizations={
                Locale.zh_TW: "要轉的點數 (必須大於 0)。",
                Locale.ja: "送るポイント数 (1以上)。",
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
                    title=":x: 轉點失敗", description="不能把點數轉給機器人。", color=_ERROR_COLOR
                )
            )
            return
        if member.id == interaction.user.id:
            await interaction.followup.send(
                embed=Embed(title=":x: 轉點失敗", description="不能轉給自己。", color=_ERROR_COLOR)
            )
            return

        ok = await transfer(
            sender_id=interaction.user.id,
            sender_name=interaction.user.name,
            receiver_id=member.id,
            receiver_name=member.name,
            amount=amount,
        )
        if not ok:
            balance_now = await get_balance(user_id=interaction.user.id)
            await interaction.followup.send(
                embed=Embed(
                    title=":x: 轉點失敗",
                    description=(
                        f"餘額不足, 你目前只有 **{balance_now:,}** 點, 想轉 **{amount:,}** 點。"
                    ),
                    color=_ERROR_COLOR,
                )
            )
            return

        sender_balance = await get_balance(user_id=interaction.user.id)
        receiver_balance = await get_balance(user_id=member.id)
        embed = Embed(
            title=":handshake: 轉點成功",
            description=(
                f"{interaction.user.mention} → {member.mention}: **{amount:,}** 點\n"
                f"你剩下 **{sender_balance:,}** 點, {member.display_name} 現在有 "
                f"**{receiver_balance:,}** 點。"
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
            await interaction.followup.send(
                embed=Embed(
                    title=":x: 無法查詢",
                    description="目前無法取得機器人身份。",
                    color=_ERROR_COLOR,
                )
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
            verdict = f"莊家目前淨贏 **{balance:,}** 點。"
        elif balance < 0:
            verdict = f"莊家目前淨虧 **{abs(balance):,}** 點。"
        else:
            verdict = "莊家目前剛好打平。"

        embed = Embed(
            title=f":game_die: {name} - 莊家戰績", description=verdict, color=_HOUSE_COLOR
        )
        embed.add_field(name="莊家從玩家身上贏到", value=f"{total_earned:,} 點", inline=True)
        embed.add_field(name="莊家賠給玩家", value=f"{total_spent:,} 點", inline=True)
        embed.set_footer(text="跨伺服器累積; 莊家資金無上限, 餘額可為負。")
        await interaction.followup.send(embed=embed)


def setup(bot: commands.Bot) -> None:
    """Adds the EconomyCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(EconomyCogs(bot), override=True)
