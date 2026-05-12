"""Slash commands that surface point balances, leaderboards, transfers, and loans."""

from datetime import UTC, datetime

import nextcord
from nextcord import Embed, Locale, Member, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.typings.economy import VIP_PURCHASE_COST, BASE_CHECKIN_REWARD_AMOUNT
from discordbot.cogs._games.cleanup import schedule_game_message_delete
from discordbot.cogs._economy.database import (
    repay,
    top_n,
    borrow,
    buy_vip,
    checkin,
    get_vip,
    transfer,
    top_losers,
    get_account,
    get_balance,
    credit_limit,
    get_loan_view,
)
from discordbot.cogs._economy.presentation import (
    CURRENCY_NAME,
    amount_code,
    bold_currency,
    currency_text,
)

_BALANCE_COLOR = 0x57F287
_LEADERBOARD_COLOR = 0xFEE75C
_LOSS_LEADERBOARD_COLOR = 0xE67E22
_TRANSFER_COLOR = 0x5865F2
_HOUSE_COLOR = 0xEB459E
_BORROW_COLOR = 0xF1C40F
_REPAY_COLOR = 0x2ECC71
_CHECKIN_COLOR = 0x9B59B6
_VIP_COLOR = 0xF1C40F
_ERROR_COLOR = 0xED4245


def _set_optional_thumbnail(*, embed: Embed, avatar_url: str) -> None:
    """Sets an embed thumbnail when an avatar URL is available."""
    if avatar_url:
        embed.set_thumbnail(url=avatar_url)


def _rank_line(*, position: int, name: str, balance: int) -> str:
    """Formats one leaderboard row."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    rank = medals.get(position, f"`#{position}`")
    return f"{rank} **{name}**  {amount_code(amount=balance)} {CURRENCY_NAME}"


def _loss_rank_line(*, position: int, name: str, loss: int) -> str:
    """Formats one loss-leaderboard row."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    rank = medals.get(position, f"`#{position}`")
    return f"{rank} **{name}**  輸 {amount_code(amount=loss)} {CURRENCY_NAME}"


async def _send_expiring_followup(*, interaction: Interaction, embed: Embed) -> None:
    """Sends a game-related economy embed and schedules its cleanup."""
    message = await interaction.followup.send(embed=embed, wait=True)
    schedule_game_message_delete(message=message)


class EconomyCogs(commands.Cog):
    """Player-facing point balance, leaderboards, loans, VIP, and check-in commands.

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
        is_vip = await get_vip(user_id=user.id)
        limit = credit_limit(user=user, is_vip=is_vip)
        age_days = (datetime.now(tz=UTC) - user.created_at).days

        embed = Embed(
            title="💰 錢包", color=_BALANCE_COLOR, description=f"## {currency_text(amount=amount)}"
        )
        embed.set_author(name=f"{user.display_name} 的錢包", icon_url=user.display_avatar.url)
        _set_optional_thumbnail(embed=embed, avatar_url=user.display_avatar.url)

        has_debt = loan is not None and loan.principal > 0
        if has_debt and loan is not None:
            embed.add_field(
                name="今日欠款",
                value=(
                    f"本金 {amount_code(amount=loan.principal)}\n"
                    "-# 收入 50% 自動還款 | 每天 0:00 自動清零"
                ),
                inline=False,
            )

        vip_badge = " · 👑 VIP" if is_vip else ""
        embed.set_footer(
            text=f"帳號 {age_days} 天 | 借款上限 {currency_text(amount=limit)}{vip_badge}"
        )
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
                title=f"🏆 {CURRENCY_NAME} Top 10",
                description="### 尚未開張\n/blackjack 開局就會上榜",
                color=_LEADERBOARD_COLOR,
            )
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        _, champion_name, champion_balance, champion_avatar_url = rows[0]

        embed = Embed(
            title=f"🏆 {CURRENCY_NAME} Top 10",
            description=(f"## 🥇 {champion_name}\n{bold_currency(amount=champion_balance)}"),
            color=_LEADERBOARD_COLOR,
        )
        embed.set_author(name="目前第一名", icon_url=champion_avatar_url or None)
        _set_optional_thumbnail(embed=embed, avatar_url=champion_avatar_url)
        if len(rows) > 1:
            others = "\n".join(
                _rank_line(position=position, name=row[1], balance=row[2])
                for position, row in enumerate(iterable=rows[1:], start=2)
            )
            embed.add_field(name="其他玩家", value=others, inline=False)
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="loss_leaderboard",
        description=f"Show today's biggest {CURRENCY_NAME} losers in the casino.",
        name_localizations={Locale.zh_TW: "輸錢榜", Locale.ja: "負け額ランキング"},
        description_localizations={
            Locale.zh_TW: f"顯示今日輸最多{CURRENCY_NAME}的前 10 名 (每天 0:00 重置)",
            Locale.ja: f"本日{CURRENCY_NAME}を最も失った上位10名 (毎日 0:00 リセット)。",
        },
        nsfw=False,
    )
    async def loss_leaderboard(self, interaction: Interaction) -> None:
        """Replies with the top 10 net casino losers for the current day.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        exclude_user_ids = (self.bot.user.id,) if self.bot.user else ()
        rows = await top_losers(limit=10, exclude_user_ids=exclude_user_ids)
        if not rows:
            embed = Embed(
                title=f"💸 今日輸錢榜 {CURRENCY_NAME}",
                description="### 今天還沒有人輸錢\n/blackjack 開局就可能進榜",
                color=_LOSS_LEADERBOARD_COLOR,
            )
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        _, champion_name, champion_loss, champion_avatar_url = rows[0]

        embed = Embed(
            title=f"💸 今日輸錢榜 {CURRENCY_NAME}",
            description=f"## 🥇 {champion_name}\n輸 {bold_currency(amount=champion_loss)}",
            color=_LOSS_LEADERBOARD_COLOR,
        )
        embed.set_author(name="今日輸最多", icon_url=champion_avatar_url or None)
        _set_optional_thumbnail(embed=embed, avatar_url=champion_avatar_url)
        if len(rows) > 1:
            others = "\n".join(
                _loss_rank_line(position=position, name=row[1], loss=row[2])
                for position, row in enumerate(iterable=rows[1:], start=2)
            )
            embed.add_field(name="其他賠錢人", value=others, inline=False)
        embed.set_footer(text="每天 0:00 (Asia/Taipei) 重置")
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
                title="轉帳失敗", description="### 不能轉給 bot\n請選一般成員", color=_ERROR_COLOR
            )
            embed.set_author(name=sender.display_name, icon_url=sender.display_avatar.url)
            await interaction.followup.send(embed=embed)
            return
        if member.id == sender.id:
            embed = Embed(title="轉帳失敗", description="### 不能轉給自己", color=_ERROR_COLOR)
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
                title="轉帳失敗",
                description=(
                    f"### 餘額不足\n"
                    f"目前 {bold_currency(amount=balance_now)}\n"
                    f"想轉 {bold_currency(amount=amount)}"
                ),
                color=_ERROR_COLOR,
            )
            embed.set_author(name=sender.display_name, icon_url=sender.display_avatar.url)
            await interaction.followup.send(embed=embed)
            return

        embed = Embed(
            title="💸 轉帳完成",
            description=f"### {currency_text(amount=amount)}\n{sender.mention} → {member.mention}",
            color=_TRANSFER_COLOR,
        )
        embed.set_author(name=sender.display_name, icon_url=sender.display_avatar.url)
        _set_optional_thumbnail(embed=embed, avatar_url=member.display_avatar.url)
        embed.add_field(
            name="轉帳後餘額",
            value=(
                f"**{sender.display_name}** {amount_code(amount=transfer_result.sender_balance)}\n"
                f"**{member.display_name}** {amount_code(amount=transfer_result.receiver_balance)}"
            ),
            inline=False,
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
        """Shows the bot's accumulated dealer P&L across `/blackjack`.

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
            verdict = f"📈 淨贏 {bold_currency(amount=balance)}"
            color = _BALANCE_COLOR
        elif balance < 0:
            verdict = f"📉 淨虧 {bold_currency(amount=abs(balance))}"
            color = _ERROR_COLOR
        else:
            verdict = "⚖️ 打平"
            color = _HOUSE_COLOR

        embed = Embed(title="🎰 莊家戰績", description=f"## {verdict}", color=color)
        embed.set_author(name=f"{name} 的賭場", icon_url=bot_user.display_avatar.url)
        _set_optional_thumbnail(embed=embed, avatar_url=bot_user.display_avatar.url)
        embed.add_field(
            name="流水",
            value=(
                f"贏到 {amount_code(amount=total_earned)}\n賠出 {amount_code(amount=total_spent)}"
            ),
            inline=False,
        )
        embed.set_footer(text="跨伺服器累積 | 莊家資金無上限")
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="borrow",
        description=f"Borrow {CURRENCY_NAME} that auto-expires at midnight (Asia/Taipei).",
        name_localizations={Locale.zh_TW: "貸款", Locale.ja: "借入"},
        description_localizations={
            Locale.zh_TW: (
                f"借入{CURRENCY_NAME}, 每天 0:00 自動清零 (額度依 Discord 帳號年齡, VIP 2x)"
            ),
            Locale.ja: (
                f"{CURRENCY_NAME}を借入します (毎日0:00自動リセット, 上限はアカウント年齢に依存, VIPは2倍)。"
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
        """Borrows ``amount`` points against the caller's daily credit window.

        Args:
            interaction: The interaction that triggered the command.
            amount: How many points to borrow.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user
        is_vip = await get_vip(user_id=user.id)
        limit = credit_limit(user=user, is_vip=is_vip)
        result = await borrow(
            user_id=user.id,
            name=user.name,
            avatar_url=user.display_avatar.url,
            amount=amount,
            credit_limit_value=limit,
        )
        if result is None:
            loan = await get_loan_view(user_id=user.id)
            current_debt = loan.principal if loan else 0
            remaining = max(limit - current_debt, 0)
            embed = Embed(
                title="借款失敗",
                description=(
                    f"### 剩餘額度 {currency_text(amount=remaining)}\n"
                    f"申請後會超過今日上限 {bold_currency(amount=limit)}"
                ),
                color=_ERROR_COLOR,
            )
            embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
            _set_optional_thumbnail(embed=embed, avatar_url=user.display_avatar.url)
            embed.add_field(name="目前欠款", value=amount_code(amount=current_debt), inline=False)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="💴 借款完成",
            description=f"### {currency_text(amount=amount, signed=True)} 入帳",
            color=_BORROW_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
        _set_optional_thumbnail(embed=embed, avatar_url=user.display_avatar.url)
        embed.add_field(
            name="借款後",
            value=(
                f"餘額 {amount_code(amount=result.new_balance)}\n"
                f"本金 {amount_code(amount=result.principal)}"
            ),
            inline=False,
        )
        embed.set_footer(
            text=(
                f"每天 0:00 (Asia/Taipei) 自動清零 | 收入 50% 自動還款 | "
                f"上限 {currency_text(amount=limit)}"
            )
        )
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="repay",
        description=f"Repay your outstanding {CURRENCY_NAME} loan from your balance.",
        name_localizations={Locale.zh_TW: "還款", Locale.ja: "返済"},
        description_localizations={
            Locale.zh_TW: f"從餘額扣款以償還{CURRENCY_NAME}欠款",
            Locale.ja: f"残高から{CURRENCY_NAME}の借入を返済します。",
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
        """Pays down outstanding principal from the caller's balance.

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
            debt = loan.principal if loan else 0
            if debt == 0:
                reason = "目前沒有欠款"
            elif balance_now == 0:
                reason = f"餘額為 0, 無法還款\n欠 {bold_currency(amount=debt)}"
            else:
                reason = "還款失敗, 請稍後再試"
            embed = Embed(title="還款失敗", description=f"### {reason}", color=_ERROR_COLOR)
            embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
            _set_optional_thumbnail(embed=embed, avatar_url=user.display_avatar.url)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="🧾 還款完成",
            description=f"### {currency_text(amount=-result.principal_repaid, signed=True)} 扣款",
            color=_REPAY_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
        _set_optional_thumbnail(embed=embed, avatar_url=user.display_avatar.url)
        embed.add_field(
            name="本次還款",
            value=f"本金 {amount_code(amount=result.principal_repaid)}",
            inline=True,
        )
        embed.add_field(
            name="剩餘",
            value=(
                f"欠款 {amount_code(amount=result.remaining_debt)}\n"
                f"餘額 {amount_code(amount=result.new_balance)}"
            ),
            inline=True,
        )
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="checkin",
        description=f"Daily {CURRENCY_NAME} check-in with a 7-day streak bonus.",
        name_localizations={Locale.zh_TW: "簽到", Locale.ja: "デイリーチェックイン"},
        description_localizations={
            Locale.zh_TW: f"每日簽到領 {BASE_CHECKIN_REWARD_AMOUNT:,} {CURRENCY_NAME}, 連續 7 天加成",
            Locale.ja: (f"毎日{BASE_CHECKIN_REWARD_AMOUNT:,}{CURRENCY_NAME}, 7日連続でボーナス。"),
        },
        nsfw=False,
    )
    async def checkin_command(self, interaction: Interaction) -> None:
        """Claims today's check-in reward; ephemeral so only the caller sees it.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        user = interaction.user
        result = await checkin(user_id=user.id, name=user.name, avatar_url=user.display_avatar.url)
        if result is None:
            embed = Embed(
                title="今天已經簽到過了",
                description="### 0:00 (Asia/Taipei) 後再回來簽吧",
                color=_ERROR_COLOR,
            )
            embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
            await interaction.followup.send(embed=embed, ephemeral=True)
            return

        vip_badge = " · 👑 VIP 2x" if result.is_vip else ""
        embed = Embed(
            title="📅 每日簽到",
            description=f"## {currency_text(amount=result.amount, signed=True)} 入帳",
            color=_CHECKIN_COLOR,
        )
        embed.set_author(name=f"{user.display_name} 的簽到", icon_url=user.display_avatar.url)
        _set_optional_thumbnail(embed=embed, avatar_url=user.display_avatar.url)
        embed.add_field(name="連續簽到", value=f"第 {result.streak} / 7 天", inline=True)
        embed.add_field(name="目前餘額", value=amount_code(amount=result.new_balance), inline=True)
        embed.set_footer(text=f"連續 7 天為一個 cycle | 每天 0:00 (Asia/Taipei) 重置{vip_badge}")
        await interaction.followup.send(embed=embed, ephemeral=True)

    @nextcord.slash_command(
        name="vip",
        description=f"Buy permanent VIP for {VIP_PURCHASE_COST:,} {CURRENCY_NAME}.",
        name_localizations={Locale.zh_TW: "購買vip", Locale.ja: "vip購入"},
        description_localizations={
            Locale.zh_TW: f"花 {VIP_PURCHASE_COST:,} {CURRENCY_NAME}購買永久 VIP",
            Locale.ja: f"{VIP_PURCHASE_COST:,}{CURRENCY_NAME}で永久 VIP を購入します。",
        },
        nsfw=False,
    )
    async def vip_command(self, interaction: Interaction) -> None:
        """Buys the permanent VIP perk for a one-time fixed cost.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user
        already_vip = await get_vip(user_id=user.id)
        if already_vip:
            embed = Embed(
                title="已經是 VIP",
                description="### 你已經擁有永久 VIP 了, 不用再買一次",
                color=_VIP_COLOR,
            )
            embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        result = await buy_vip(user_id=user.id, name=user.name, avatar_url=user.display_avatar.url)
        if result is None:
            balance_now = await get_balance(user_id=user.id)
            embed = Embed(
                title="VIP 購買失敗",
                description=(
                    f"### 餘額不足\n"
                    f"目前 {bold_currency(amount=balance_now)}\n"
                    f"需要 {bold_currency(amount=VIP_PURCHASE_COST)}"
                ),
                color=_ERROR_COLOR,
            )
            embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="👑 升級 VIP 成功",
            description=f"### {currency_text(amount=-result.cost, signed=True)} 扣款",
            color=_VIP_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
        _set_optional_thumbnail(embed=embed, avatar_url=user.display_avatar.url)
        embed.add_field(
            name="VIP 永久權益",
            value=("- 賭場贏錢賠率 **1.5x**\n- 每日簽到點數 **2x**\n- 貸款額度 **2x**"),
            inline=False,
        )
        embed.add_field(
            name="目前餘額", value=amount_code(amount=result.new_balance), inline=False
        )
        await _send_expiring_followup(interaction=interaction, embed=embed)


def setup(bot: commands.Bot) -> None:
    """Adds the EconomyCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(EconomyCogs(bot), override=True)
