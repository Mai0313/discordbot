"""Slash commands that surface point balances, the leaderboard, transfers, and loans."""

from datetime import UTC, datetime

import nextcord
from nextcord import Embed, Locale, Member, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.cogs._games.cleanup import schedule_game_message_delete
from discordbot.cogs._economy.database import (
    repay,
    top_n,
    borrow,
    transfer,
    get_account,
    get_balance,
    credit_limit,
    accrual_delta,
    get_loan_view,
)
from discordbot.cogs._economy.presentation import CURRENCY_NAME, currency_text

_BALANCE_COLOR = 0x57F287
_LEADERBOARD_COLOR = 0xFEE75C
_TRANSFER_COLOR = 0x5865F2
_HOUSE_COLOR = 0xEB459E
_BORROW_COLOR = 0xF1C40F
_REPAY_COLOR = 0x2ECC71
_ERROR_COLOR = 0xED4245


async def _send_expiring_followup(*, interaction: Interaction, embed: Embed) -> None:
    """Sends a game-related economy embed and schedules its cleanup."""
    message = await interaction.followup.send(embed=embed, wait=True)
    schedule_game_message_delete(message=message)


class EconomyCogs(commands.Cog):
    """Player-facing point balance and loan commands.

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
        description=f"Check your current {CURRENCY_NAME} balance and loan status.",
        name_localizations={Locale.zh_TW: "餘額", Locale.ja: "残高"},
        description_localizations={
            Locale.zh_TW: f"查詢你目前的{CURRENCY_NAME}餘額與欠款狀態",
            Locale.ja: f"現在の{CURRENCY_NAME}残高と借入状況を確認します。",
        },
        nsfw=False,
    )
    async def balance(self, interaction: Interaction) -> None:
        """Replies with the caller's current balance and any outstanding loan.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user
        amount = await get_balance(user_id=user.id)
        loan = await get_loan_view(user_id=user.id)
        limit = credit_limit(user=user)
        age_days = (datetime.now(tz=UTC) - user.created_at).days

        embed = Embed(color=_BALANCE_COLOR, description=f"💰 **{currency_text(amount=amount)}**")
        embed.set_author(name=f"{user.display_name} 的錢包", icon_url=user.display_avatar.url)
        embed.set_thumbnail(url=user.display_avatar.url)

        has_debt = loan is not None and (loan.principal > 0 or loan.interest_stored > 0)
        if has_debt and loan is not None:
            pending_interest = 0
            if loan.last_accrual_at is not None and loan.principal > 0:
                pending_interest = accrual_delta(
                    principal=loan.principal,
                    last_accrual_at=loan.last_accrual_at,
                    now=datetime.now(tz=UTC),
                )
            effective_interest = loan.interest_stored + pending_interest
            embed.add_field(name="未還本金", value=f"`{loan.principal:,}`", inline=True)
            embed.add_field(name="累積利息", value=f"`{effective_interest:,}`", inline=True)
            embed.add_field(name="自動還款", value="收入 50% 抵債", inline=True)

        embed.set_footer(text=f"帳號 {age_days} 天 · 借款上限 {limit:,}")
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="leaderboard",
        description=f"Show the global top {CURRENCY_NAME} holders.",
        name_localizations={Locale.zh_TW: "排行榜", Locale.ja: "リーダーボード"},
        description_localizations={
            Locale.zh_TW: f"顯示 global {CURRENCY_NAME}前 10 名",
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
        rows = await top_n(limit=10, exclude_user_ids=exclude_user_ids)
        if not rows:
            embed = Embed(
                title=f"🏆 {CURRENCY_NAME}排行榜",
                description=f"目前還沒有人有{CURRENCY_NAME}, /dice 或 /blackjack 開賭看看",
                color=_LEADERBOARD_COLOR,
            )
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        medals = ("🥇", "🥈", "🥉")
        champion_name = rows[0][1]
        champion_balance = rows[0][2]
        champion_avatar_url = rows[0][3]
        lines: list[str] = []
        for index, row in enumerate(iterable=rows):
            name, amount = row[1], row[2]
            if index < 3:
                lines.append(f"{medals[index]} **{name}** · `{amount:,}`")
            else:
                lines.append(f"{name} · {amount:,}")

        embed = Embed(title=f"🏆 {CURRENCY_NAME}排行榜", color=_LEADERBOARD_COLOR)
        embed.set_author(
            name=f"👑 本期霸主 · {champion_name}", icon_url=champion_avatar_url or None
        )
        if champion_avatar_url:
            embed.set_thumbnail(url=champion_avatar_url)
        embed.description = (
            f"持有 **{currency_text(amount=champion_balance)}**\n"
            "━━━━━━━━━━━━━━━━━━\n" + "\n".join(lines)
        )
        embed.set_footer(text=f"共 {len(rows)} 位玩家 · /balance 查自己")
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="give",
        description=f"Transfer your {CURRENCY_NAME} to another member.",
        name_localizations={Locale.zh_TW: "轉虛擬歡樂豆", Locale.ja: "虛擬歡樂豆送付"},
        description_localizations={
            Locale.zh_TW: f"把你的{CURRENCY_NAME}轉給其他成員",
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
                Locale.zh_TW: f"要接收{CURRENCY_NAME}的成員",
                Locale.ja: f"{CURRENCY_NAME}を受け取るメンバー。",
            },
            required=True,
        ),
        amount: int = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to transfer (must be positive).",
            name_localizations={Locale.zh_TW: "虛擬歡樂豆", Locale.ja: "虛擬歡樂豆"},
            description_localizations={
                Locale.zh_TW: f"要轉的{CURRENCY_NAME} (必須大於 0)",
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

        sender = interaction.user

        if member.bot:
            embed = Embed(
                title="❌ 轉帳失敗",
                description=f"不能把{CURRENCY_NAME}轉給機器人",
                color=_ERROR_COLOR,
            )
            embed.set_author(name=sender.display_name, icon_url=sender.display_avatar.url)
            await interaction.followup.send(embed=embed)
            return
        if member.id == sender.id:
            embed = Embed(title="❌ 轉帳失敗", description="不能轉給自己", color=_ERROR_COLOR)
            embed.set_author(name=sender.display_name, icon_url=sender.display_avatar.url)
            await interaction.followup.send(embed=embed)
            return

        transfer_result = await transfer(
            sender_id=sender.id,
            sender_name=sender.name,
            sender_avatar_url=sender.display_avatar.url,
            receiver_id=member.id,
            receiver_name=member.name,
            receiver_avatar_url=member.display_avatar.url,
            amount=amount,
        )
        if transfer_result is None:
            balance_now = await get_balance(user_id=sender.id)
            embed = Embed(
                title="❌ 轉帳失敗",
                description=(
                    f"餘額只有 **{currency_text(amount=balance_now)}**, "
                    f"想轉 **{currency_text(amount=amount)}**"
                ),
                color=_ERROR_COLOR,
            )
            embed.set_author(name=sender.display_name, icon_url=sender.display_avatar.url)
            await interaction.followup.send(embed=embed)
            return

        embed = Embed(
            title="💸 轉帳成功",
            description=(
                f"{sender.mention} → {member.mention}\n金額 **{currency_text(amount=amount)}**"
            ),
            color=_TRANSFER_COLOR,
        )
        embed.set_author(name=sender.display_name, icon_url=sender.display_avatar.url)
        embed.set_thumbnail(url=member.display_avatar.url)
        embed.add_field(
            name=f"{sender.display_name} 的餘額",
            value=f"`{transfer_result.sender_balance:,}`",
            inline=True,
        )
        embed.add_field(
            name=f"{member.display_name} 的餘額",
            value=f"`{transfer_result.receiver_balance:,}`",
            inline=True,
        )
        await interaction.followup.send(embed=embed)

    @nextcord.slash_command(
        name="house",
        description="Show the dealer's running win/loss across every game.",
        name_localizations={Locale.zh_TW: "莊家戰績", Locale.ja: "ディーラー戦績"},
        description_localizations={
            Locale.zh_TW: "顯示莊家在所有遊戲累積的輸贏 (跨伺服器)",
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
            embed = Embed(
                title="❌ 無法查詢", description="目前無法取得機器人身份", color=_ERROR_COLOR
            )
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        bot_user = self.bot.user
        account = await get_account(user_id=bot_user.id)
        # No row yet means nobody's played a round; show a fresh-house view
        # rather than treating it as an error.
        name = bot_user.display_name
        if account is None:
            balance, total_earned, total_spent = 0, 0, 0
        else:
            _, balance, total_earned, total_spent = account
            name = name or account[0]

        if balance > 0:
            verdict = f"📈 淨贏 **{currency_text(amount=balance)}**"
            color = _BALANCE_COLOR
        elif balance < 0:
            verdict = f"📉 淨虧 **{currency_text(amount=abs(balance))}**"
            color = _ERROR_COLOR
        else:
            verdict = "⚖️ 打平"
            color = _HOUSE_COLOR

        embed = Embed(title="🎰 莊家戰績", description=verdict, color=color)
        embed.set_author(name=f"{name} 的賭場", icon_url=bot_user.display_avatar.url)
        embed.set_thumbnail(url=bot_user.display_avatar.url)
        embed.add_field(name="贏到", value=f"`{total_earned:,}`", inline=True)
        embed.add_field(name="賠出", value=f"`{total_spent:,}`", inline=True)
        embed.set_footer(text="跨伺服器累積 · 莊家資金無上限")
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="borrow",
        description=f"Borrow {CURRENCY_NAME} against your Discord account age.",
        name_localizations={Locale.zh_TW: "貸款", Locale.ja: "借入"},
        description_localizations={
            Locale.zh_TW: (
                f"用 Discord 帳號年齡換取{CURRENCY_NAME}借款 "
                "(日利息 1%, 之後賺到的點數會自動 50% 抵債)"
            ),
            Locale.ja: (
                f"Discord アカウント年齢に応じて{CURRENCY_NAME}を借入します "
                "(日利1%, 以降の獲得分は50%自動返済)。"
            ),
        },
        nsfw=False,
    )
    async def borrow_loan(
        self,
        interaction: Interaction,
        amount: int = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to borrow (must be positive).",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要借入的{CURRENCY_NAME} (必須大於 0)",
                Locale.ja: f"借入する{CURRENCY_NAME} (1以上)。",
            },
            required=True,
            min_value=1,
        ),
    ) -> None:
        """Borrows ``amount`` points against the caller's credit limit.

        Args:
            interaction: The interaction that triggered the command.
            amount: How many points to borrow.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user
        limit = credit_limit(user=user)
        result = await borrow(
            user_id=user.id,
            name=user.name,
            avatar_url=user.display_avatar.url,
            amount=amount,
            credit_limit_value=limit,
        )
        if result is None:
            loan = await get_loan_view(user_id=user.id)
            current_debt = (loan.principal + loan.interest_stored) if loan else 0
            remaining = max(limit - current_debt, 0)
            embed = Embed(
                title="❌ 借款失敗",
                description=f"超過借款上限 **{currency_text(amount=limit)}**",
                color=_ERROR_COLOR,
            )
            embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
            embed.add_field(name="目前欠款", value=f"`{current_debt:,}`", inline=True)
            embed.add_field(name="尚可借", value=f"`{remaining:,}`", inline=True)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="💴 借款成功",
            description=f"撥款 **{currency_text(amount=amount)}** 入帳",
            color=_BORROW_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
        embed.set_thumbnail(url=user.display_avatar.url)
        embed.add_field(name="目前餘額", value=f"`{result.new_balance:,}`", inline=True)
        embed.add_field(name="未還本金", value=f"`{result.principal:,}`", inline=True)
        embed.set_footer(text=f"日利息 1% · 收入 50% 自動抵債 · 上限 {limit:,}")
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="repay",
        description=f"Repay your outstanding {CURRENCY_NAME} loan from your balance.",
        name_localizations={Locale.zh_TW: "還款", Locale.ja: "返済"},
        description_localizations={
            Locale.zh_TW: f"從餘額扣款以償還{CURRENCY_NAME}欠款 (利息優先, 本金其次)",
            Locale.ja: f"残高から{CURRENCY_NAME}の借入を返済します (利息優先, 本金次)。",
        },
        nsfw=False,
    )
    async def repay_loan(
        self,
        interaction: Interaction,
        amount: int = SlashOption(
            name="amount",
            description=f"Maximum {CURRENCY_NAME} to apply against the debt.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要還款的最高{CURRENCY_NAME} (自動 clamp 到欠款額)",
                Locale.ja: f"返済する{CURRENCY_NAME}の上限 (借入額にクランプ)。",
            },
            required=True,
            min_value=1,
        ),
    ) -> None:
        """Pays down outstanding interest then principal from the caller's balance.

        Args:
            interaction: The interaction that triggered the command.
            amount: Maximum amount to repay; clamped to ``min(amount, balance, debt)``.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user

        result = await repay(
            user_id=user.id, name=user.name, avatar_url=user.display_avatar.url, amount=amount
        )
        if result is None:
            balance_now = await get_balance(user_id=user.id)
            loan = await get_loan_view(user_id=user.id)
            debt = (loan.principal + loan.interest_stored) if loan else 0
            if debt == 0:
                reason = "目前沒有欠款"
            elif balance_now == 0:
                reason = f"餘額為 0, 無法還款 (欠 **{currency_text(amount=debt)}**)"
            else:
                reason = "還款失敗, 請稍後再試"
            embed = Embed(title="❌ 還款失敗", description=reason, color=_ERROR_COLOR)
            embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        effective = result.interest_repaid + result.principal_repaid
        embed = Embed(
            title="🧾 還款成功",
            description=f"扣款 **{currency_text(amount=effective)}**",
            color=_REPAY_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
        embed.set_thumbnail(url=user.display_avatar.url)
        embed.add_field(name="利息", value=f"`{result.interest_repaid:,}`", inline=True)
        embed.add_field(name="本金", value=f"`{result.principal_repaid:,}`", inline=True)
        embed.add_field(name="剩餘欠款", value=f"`{result.remaining_debt:,}`", inline=True)
        embed.set_footer(text=f"餘額 {result.new_balance:,}")
        await _send_expiring_followup(interaction=interaction, embed=embed)


def setup(bot: commands.Bot) -> None:
    """Adds the EconomyCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(EconomyCogs(bot), override=True)
