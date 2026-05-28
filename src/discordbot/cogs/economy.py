"""Slash commands that surface point balances, leaderboards, transfers, and loans."""

from io import BytesIO
from datetime import UTC, datetime
import contextlib

import nextcord
from nextcord import Embed, Locale, Member, ButtonStyle, Interaction, SlashOption
from nextcord.ui import View, Button
from nextcord.ext import commands

from discordbot.typings.stock import StockPortfolioView, StockPortfolioHolding
from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.config import EconomyConfig
from discordbot.typings.economy import (
    VIP_PURCHASE_COST,
    BASE_CHECKIN_REWARD_AMOUNT,
    DEFAULT_LOAN_MONTHLY_RATE_BPS,
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    LoanLenderType,
)
from discordbot.utils.number_text import share_quantity_text
from discordbot.cogs._stock.market import format_price
from discordbot.cogs._economy.boards import (
    LOSS_LEADERBOARD_BOARD_FILENAME,
    BALANCE_LEADERBOARD_BOARD_FILENAME,
    build_loss_leaderboard_board_image,
    build_balance_leaderboard_board_image,
)
from discordbot.cogs._stock.database import get_stock_portfolio
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import schedule_public_message_delete
from discordbot.cogs._economy.database import (
    top_n,
    buy_vip,
    checkin,
    get_vip,
    transfer,
    get_admin,
    top_losers,
    get_account,
    get_balance,
    get_portfolio,
    adjust_balance,
    checkin_reward,
    get_casino_ledger,
    get_central_banker,
    call_personal_loans,
    list_loan_contracts,
    accept_loan_proposal,
    cancel_loan_proposal,
    reject_loan_proposal,
    repay_personal_loans,
    call_central_bank_loans,
    get_central_bank_status,
    repay_central_bank_loans,
    apply_vip_blackjack_bonus,
    monthly_rate_bps_to_percent,
    monthly_rate_percent_to_bps,
    create_personal_loan_request,
    reject_expired_loan_proposal,
    create_central_bank_loan_request,
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
_ADMIN_COLOR = 0x3498DB
_CASINO_COLOR = 0xEB459E
_BORROW_COLOR = 0xF1C40F
_REPAY_COLOR = 0x2ECC71
_CENTRAL_BANK_COLOR = 0x1ABC9C
_CHECKIN_COLOR = 0x9B59B6
_VIP_COLOR = 0xF1C40F
_ERROR_COLOR = 0xED4245
_STOCK_POSITION_LINE_LIMIT = 5
_STOCK_POSITION_NAME_LIMIT = 20


def _vip_perk_lines(checkin_streak: int = 1) -> str:
    """Formats VIP perks with the base number and the boosted number."""
    base_checkin = checkin_reward(streak=checkin_streak, is_vip=False)
    vip_checkin = checkin_reward(streak=checkin_streak, is_vip=True)
    sample_win = 10_000
    boosted_win = apply_vip_blackjack_bonus(delta=sample_win, is_vip=True)
    checkin_label = "簽到基礎" if checkin_streak == 1 else f"第 {checkin_streak} 天簽到"
    return (
        f"{checkin_label} {amount_code(amount=base_checkin, compact=True)} → "
        f"{amount_code(amount=vip_checkin, compact=True)}\n"
        f"Blackjack 贏局例 {amount_code(amount=sample_win, signed=True, compact=True)} → "
        f"{amount_code(amount=boosted_win, signed=True, compact=True)}"
    )


def _set_optional_thumbnail(embed: Embed, avatar_url: str) -> None:
    """Sets an embed thumbnail when an avatar URL is available."""
    if avatar_url:
        embed.set_thumbnail(url=avatar_url)


def _parse_positive_amount(raw_amount: str | None) -> int | None:
    """Parses user-entered positive amount text with optional comma separators."""
    normalized = (raw_amount or "").replace(",", "").strip()
    if not normalized.isdecimal():
        return None
    try:
        amount = int(normalized)
    except ValueError:
        return None
    if amount <= 0:
        return None
    return amount


def _stock_position_lines(stock_portfolio: StockPortfolioView) -> str:
    """Formats stock holdings for the private balance embed."""
    if not stock_portfolio.holdings:
        return "目前沒有股票部位"
    lines = [
        _stock_position_line(holding=holding)
        for holding in stock_portfolio.holdings[:_STOCK_POSITION_LINE_LIMIT]
    ]
    remaining = len(stock_portfolio.holdings) - _STOCK_POSITION_LINE_LIMIT
    if remaining > 0:
        lines.append(f"還有 `{remaining:,}` 檔未列出")
    return "\n".join(lines)


def _stock_position_line(holding: StockPortfolioHolding) -> str:
    """Formats one stock holding into a compact balance line."""
    position_parts: list[str] = []
    if holding.long_shares > 0:
        position_parts.append(
            f"持股 `{share_quantity_text(shares=holding.long_shares)}` / 市值 "
            f"{amount_code(amount=holding.long_market_value, compact=True)}"
        )
    if holding.short_shares > 0:
        short_equity = (
            holding.short_collateral + holding.short_entry_value - holding.short_cover_cost
        )
        position_parts.append(
            f"做空 `{share_quantity_text(shares=holding.short_shares)}` / 淨值 "
            f"{amount_code(amount=short_equity, compact=True)}"
        )
    position_text = " · ".join(position_parts) if position_parts else "無部位"
    name = _stock_position_name(name=holding.name)
    return (
        f"`{holding.symbol}` {name} · 股價 `{format_price(price_cents=holding.price_cents)}` · "
        f"{position_text} · 未實現 "
        f"{amount_code(amount=holding.unrealized_pnl, signed=True, compact=True)}"
    )


def _stock_position_name(name: str) -> str:
    """Keeps long company names from filling the stock field."""
    if len(name) <= _STOCK_POSITION_NAME_LIMIT:
        return name
    return f"{name[:_STOCK_POSITION_NAME_LIMIT]}..."


def _debt_summary_text(*, principal: int, interest: int) -> str:
    """Formats outstanding loan principal and interest."""
    if principal <= 0 and interest <= 0:
        return "無未還債務"
    return (
        f"本金 {amount_code(amount=principal, compact=True)}\n"
        f"利息 {amount_code(amount=interest, compact=True)}"
    )


def _vip_status_text(is_vip: bool) -> str:
    """Formats VIP status for the private balance embed."""
    if not is_vip:
        return "一般會員"
    return f"👑 VIP\n{_vip_perk_lines()}"


async def _send_expiring_followup(
    interaction: Interaction,
    embed: Embed,
    view: View | None = None,
    file: nextcord.File | None = None,
) -> None:
    """Sends a game-related economy embed and schedules its cleanup."""
    extra_files = [file] if file is not None else None
    kwargs: dict[str, object] = {
        "embed": embed,
        "wait": True,
        **embed_spacer_payload(
            embeds=[embed], is_edit=False, target=interaction, extra_files=extra_files
        ),
    }
    if view is not None:
        kwargs["view"] = view
    message = await interaction.followup.send(**kwargs)
    user_name = interaction.user.name if interaction.user is not None else None
    schedule_public_message_delete(message=message, user_name=user_name)


async def _send_loan_request_followup(interaction: Interaction, embed: Embed, view: View) -> None:
    """Sends a loan request message that owns its cleanup after a terminal state."""
    message = await interaction.followup.send(
        embed=embed,
        view=view,
        wait=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )
    view.message = message


async def _send_private_followup(interaction: Interaction, embed: Embed) -> None:
    """Sends a personal economy embed visible only to the caller."""
    await interaction.followup.send(
        embed=embed,
        ephemeral=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )


async def _send_ephemeral_response(interaction: Interaction, embed: Embed) -> None:
    """Sends an ephemeral economy embed as the initial interaction response."""
    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )


async def _edit_response_embed(interaction: Interaction, embed: Embed) -> None:
    """Edits the interaction's public message embed and clears its controls."""
    await interaction.response.edit_message(
        embed=embed,
        view=None,
        **embed_spacer_payload(embeds=[embed], is_edit=True, target=interaction),
    )


def _rate_text(monthly_rate_bps: int) -> str:
    """Formats a monthly loan rate."""
    return f"每月 {monthly_rate_bps_to_percent(monthly_rate_bps=monthly_rate_bps):g}%"


def _loan_terms_text(amount: int, monthly_rate_bps: int) -> str:
    """Formats the loan terms shown before acceptance."""
    return (
        f"本金 {amount_code(amount=amount, compact=True)}\n"
        f"利率 `{_rate_text(monthly_rate_bps=monthly_rate_bps)}`\n"
        "利息採單利，依經過天數按比例計算\n"
        "還款會先抵利息，再抵本金；貸方可催收"
    )


def _payment_summary_text(  # noqa: PLR0913 -- summary needs all visible repayment fields
    paid_amount: int,
    interest_paid: int,
    principal_paid: int,
    remaining_principal: int,
    remaining_interest: int,
    borrower_balance: int,
) -> str:
    """Formats one repayment or collection result."""
    return (
        f"本次扣款 {amount_code(amount=paid_amount, compact=True)}\n"
        f"償還利息 {amount_code(amount=interest_paid, compact=True)}\n"
        f"償還本金 {amount_code(amount=principal_paid, compact=True)}\n"
        f"剩餘本金 {amount_code(amount=remaining_principal, compact=True)}\n"
        f"剩餘利息 {amount_code(amount=remaining_interest, compact=True)}\n"
        f"借方餘額 {amount_code(amount=borrower_balance, compact=True)}"
    )


def _credit_request_footer() -> str:
    """Formats the personal credit request button hint."""
    return (
        f"貸方可用下方按鈕批准或拒絕，發起者可取消，{LOAN_PROPOSAL_TIMEOUT_SECONDS} 秒後自動拒絕"
    )


def _central_bank_request_footer() -> str:
    """Formats the central-bank request button hint."""
    return f"央行成員可用下方按鈕批准或拒絕，發起者可取消，{LOAN_PROPOSAL_TIMEOUT_SECONDS} 秒後自動拒絕"


class CentralBankLoanDecisionView(View):
    """Button controls for deciding a public central-bank loan request."""

    def __init__(
        self,
        bot: commands.Bot,
        proposal_id: int,
        creator_id: int,
        allow_self_approval: bool = False,
    ) -> None:
        """Initializes a decision view for one proposal."""
        super().__init__(timeout=LOAN_PROPOSAL_TIMEOUT_SECONDS)
        self.bot = bot
        self.proposal_id = proposal_id
        self.creator_id = creator_id
        self.allow_self_approval = allow_self_approval
        self.message: nextcord.Message | None = None

    def _schedule_cleanup(self, interaction: Interaction | None = None) -> None:
        """Schedules the public request message for cleanup after a terminal state."""
        message = self.message or getattr(interaction, "message", None)
        if message is None:
            return
        user_name = None
        if interaction is not None and interaction.user is not None:
            user_name = interaction.user.name
        schedule_public_message_delete(message=message, user_name=user_name)

    async def on_timeout(self) -> None:
        """Rejects a stale central-bank request and cleans up its message."""
        proposal = await reject_expired_loan_proposal(proposal_id=self.proposal_id)
        if proposal is None or self.message is None:
            return
        self.stop()
        embed = Embed(
            title="🏛️ 央行申請已逾時",
            description="### 申請已逾時，自動拒絕",
            color=_CENTRAL_BANK_COLOR,
        )
        with contextlib.suppress(Exception):
            await self.message.edit(
                embed=embed,
                view=None,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=self.message),
            )
        self._schedule_cleanup()

    async def _send_permission_denied(self, interaction: Interaction) -> None:
        """Replies privately when a non-banker clicks a decision button."""
        embed = Embed(
            title="權限不足",
            description="### 只有央行成員可以處理央行借款申請",
            color=_ERROR_COLOR,
        )
        await _send_ephemeral_response(interaction=interaction, embed=embed)

    async def _is_central_banker(self, interaction: Interaction) -> bool:
        """Returns whether the clicking user can decide central-bank proposals."""
        if interaction.user is None:
            return False
        return await get_central_banker(user_id=interaction.user.id)

    def _central_bank_exclude_user_ids(self) -> tuple[int, ...]:
        """Returns bot-owned account IDs excluded from central-bank capacity."""
        return (self.bot.user.id,) if self.bot.user is not None else ()

    @nextcord.ui.button(
        label="批准",
        emoji="✅",
        style=ButtonStyle.success,
        custom_id="central_bank:approve",
        row=0,
    )
    async def approve(self, _button: Button, interaction: Interaction) -> None:
        """Approves the central-bank request when clicked by a central banker."""
        if interaction.user is None:
            return
        if not await self._is_central_banker(interaction=interaction):
            await self._send_permission_denied(interaction=interaction)
            return

        banker_avatar_url = await guild_avatar_url(
            user=interaction.user, guild=getattr(interaction, "guild", None)
        )
        result = await accept_loan_proposal(
            proposal_id=self.proposal_id,
            actor_id=interaction.user.id,
            actor_name=interaction.user.name,
            actor_avatar_url=banker_avatar_url,
            is_central_banker=True,
            central_bank_exclude_user_ids=self._central_bank_exclude_user_ids(),
            allow_central_bank_self_approval=self.allow_self_approval,
        )
        if result is None:
            embed = Embed(
                title="批准失敗",
                description="### 申請不存在、已處理、自我批准未開放，或央行額度不足",
                color=_ERROR_COLOR,
            )
            await _send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="🏛️ 央行借款已批准",
            description=(
                f"### {currency_text(amount=result.contract.principal_remaining, compact=True)} 已入帳"
            ),
            color=_CENTRAL_BANK_COLOR,
        )
        embed.add_field(name="批准者", value=interaction.user.mention, inline=True)
        embed.add_field(
            name="央行剩餘額度",
            value=amount_code(amount=result.central_bank_available_credit or 0, compact=True),
            inline=True,
        )
        self.stop()
        await _edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)

    @nextcord.ui.button(
        label="拒絕", emoji="✖️", style=ButtonStyle.danger, custom_id="central_bank:reject", row=0
    )
    async def reject(self, _button: Button, interaction: Interaction) -> None:
        """Rejects the central-bank request when clicked by a central banker."""
        if interaction.user is None:
            return
        if not await self._is_central_banker(interaction=interaction):
            await self._send_permission_denied(interaction=interaction)
            return

        proposal = await reject_loan_proposal(
            proposal_id=self.proposal_id, actor_id=interaction.user.id, is_central_banker=True
        )
        if proposal is None:
            embed = Embed(
                title="拒絕失敗",
                description="### 申請不存在、已處理，或你沒有權限拒絕",
                color=_ERROR_COLOR,
            )
            await _send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="🏛️ 央行申請已拒絕",
            description=f"### 央行借款申請已關閉\n處理人 {interaction.user.mention}",
            color=_CENTRAL_BANK_COLOR,
        )
        self.stop()
        await _edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)

    @nextcord.ui.button(
        label="取消",
        emoji="🚫",
        style=ButtonStyle.secondary,
        custom_id="central_bank:cancel",
        row=0,
    )
    async def cancel(self, _button: Button, interaction: Interaction) -> None:
        """Cancels the central-bank request when clicked by its creator."""
        if interaction.user is None:
            return
        if interaction.user.id != self.creator_id:
            embed = Embed(
                title="權限不足",
                description="### 只有申請發起者可以取消央行借款申請",
                color=_ERROR_COLOR,
            )
            await _send_ephemeral_response(interaction=interaction, embed=embed)
            return

        proposal = await cancel_loan_proposal(
            proposal_id=self.proposal_id, actor_id=interaction.user.id
        )
        if proposal is None:
            embed = Embed(
                title="取消失敗",
                description="### 申請不存在、已處理，或你不是發起者",
                color=_ERROR_COLOR,
            )
            await _send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="🏛️ 央行申請已取消",
            description=f"### 央行借款申請已關閉\n發起者 {interaction.user.mention}",
            color=_CENTRAL_BANK_COLOR,
        )
        self.stop()
        await _edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)


class CreditLoanDecisionView(View):
    """Button controls for deciding a public personal credit request."""

    def __init__(self, proposal_id: int, lender_id: int, creator_id: int) -> None:
        """Initializes a decision view for one personal credit proposal."""
        super().__init__(timeout=LOAN_PROPOSAL_TIMEOUT_SECONDS)
        self.proposal_id = proposal_id
        self.lender_id = lender_id
        self.creator_id = creator_id
        self.message: nextcord.Message | None = None

    def _schedule_cleanup(self, interaction: Interaction | None = None) -> None:
        """Schedules the public request message for cleanup after a terminal state."""
        message = self.message or getattr(interaction, "message", None)
        if message is None:
            return
        user_name = None
        if interaction is not None and interaction.user is not None:
            user_name = interaction.user.name
        schedule_public_message_delete(message=message, user_name=user_name)

    async def on_timeout(self) -> None:
        """Rejects a stale personal credit request and cleans up its message."""
        proposal = await reject_expired_loan_proposal(proposal_id=self.proposal_id)
        if proposal is None or self.message is None:
            return
        self.stop()
        embed = Embed(
            title="信貸申請已逾時", description="### 申請已逾時，自動拒絕", color=_REPAY_COLOR
        )
        with contextlib.suppress(Exception):
            await self.message.edit(
                embed=embed,
                view=None,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=self.message),
            )
        self._schedule_cleanup()

    async def _send_permission_denied(self, interaction: Interaction, description: str) -> None:
        """Replies privately when a user clicks a button they cannot use."""
        embed = Embed(title="權限不足", description=description, color=_ERROR_COLOR)
        await _send_ephemeral_response(interaction=interaction, embed=embed)

    async def _require_lender(self, interaction: Interaction) -> bool:
        """Returns whether the clicking user is the requested lender."""
        if interaction.user is None:
            return False
        if interaction.user.id == self.lender_id:
            return True
        await self._send_permission_denied(
            interaction=interaction, description="### 只有指定貸方可以處理這筆信貸申請"
        )
        return False

    @nextcord.ui.button(
        label="批准", emoji="✅", style=ButtonStyle.success, custom_id="credit:approve", row=0
    )
    async def approve(self, _button: Button, interaction: Interaction) -> None:
        """Approves the personal credit request when clicked by the lender."""
        if interaction.user is None or not await self._require_lender(interaction=interaction):
            return

        lender_avatar_url = await guild_avatar_url(
            user=interaction.user, guild=getattr(interaction, "guild", None)
        )
        result = await accept_loan_proposal(
            proposal_id=self.proposal_id,
            actor_id=interaction.user.id,
            actor_name=interaction.user.name,
            actor_avatar_url=lender_avatar_url,
        )
        if result is None:
            embed = Embed(
                title="批准失敗",
                description="### 申請不存在、已處理、不是指定貸方，或貸方餘額不足",
                color=_ERROR_COLOR,
            )
            await _send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="✅ 信貸已批准",
            description=f"### {currency_text(amount=result.contract.principal_remaining, compact=True)} 已入帳",
            color=_BORROW_COLOR,
        )
        embed.add_field(name="批准者", value=interaction.user.mention, inline=True)
        embed.add_field(
            name="借方餘額",
            value=amount_code(amount=result.borrower_balance, compact=True),
            inline=True,
        )
        self.stop()
        await _edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)

    @nextcord.ui.button(
        label="拒絕", emoji="✖️", style=ButtonStyle.danger, custom_id="credit:reject", row=0
    )
    async def reject(self, _button: Button, interaction: Interaction) -> None:
        """Rejects the personal credit request when clicked by the lender."""
        if interaction.user is None or not await self._require_lender(interaction=interaction):
            return

        proposal = await reject_loan_proposal(
            proposal_id=self.proposal_id, actor_id=interaction.user.id
        )
        if proposal is None:
            embed = Embed(
                title="拒絕失敗",
                description="### 申請不存在、已處理，或你不是指定貸方",
                color=_ERROR_COLOR,
            )
            await _send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="信貸申請已拒絕",
            description=f"### 信貸申請已關閉\n處理人 {interaction.user.mention}",
            color=_REPAY_COLOR,
        )
        self.stop()
        await _edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)

    @nextcord.ui.button(
        label="取消", emoji="🚫", style=ButtonStyle.secondary, custom_id="credit:cancel", row=0
    )
    async def cancel(self, _button: Button, interaction: Interaction) -> None:
        """Cancels the personal credit request when clicked by its creator."""
        if interaction.user is None:
            return
        if interaction.user.id != self.creator_id:
            await self._send_permission_denied(
                interaction=interaction, description="### 只有申請發起者可以取消這筆信貸申請"
            )
            return

        proposal = await cancel_loan_proposal(
            proposal_id=self.proposal_id, actor_id=interaction.user.id
        )
        if proposal is None:
            embed = Embed(
                title="取消失敗",
                description="### 申請不存在、已處理，或你不是發起者",
                color=_ERROR_COLOR,
            )
            await _send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="信貸申請已取消",
            description=f"### 信貸申請已關閉\n發起者 {interaction.user.mention}",
            color=_REPAY_COLOR,
        )
        self.stop()
        await _edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)


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
        name="admin",
        description=f"Run admin-only {CURRENCY_NAME} maintenance operations.",
        name_localizations={Locale.zh_TW: "管理員", Locale.ja: "管理者"},
        description_localizations={
            Locale.zh_TW: f"執行管理員限定的{CURRENCY_NAME}維護操作",
            Locale.ja: f"管理者専用の{CURRENCY_NAME}メンテナンス操作を実行します。",
        },
        nsfw=False,
    )
    async def admin(self, interaction: Interaction) -> None:
        """Slash command group for economy admin operations."""

    @admin.subcommand(
        name="refund_tax",
        description=f"Admin-only: credit {CURRENCY_NAME} to a member or bot.",
        name_localizations={Locale.zh_TW: "退稅", Locale.ja: "税還付"},
        description_localizations={
            Locale.zh_TW: f"管理員限定：無條件增加某位成員或 bot 的{CURRENCY_NAME}",
            Locale.ja: f"管理者専用：メンバーまたは bot に{CURRENCY_NAME}を付与します。",
        },
    )
    async def admin_refund_tax(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description=f"The member or bot to receive the {CURRENCY_NAME}.",
            name_localizations={Locale.zh_TW: "對象", Locale.ja: "対象"},
            description_localizations={
                Locale.zh_TW: f"要增加{CURRENCY_NAME}的成員或 bot",
                Locale.ja: f"{CURRENCY_NAME}を受け取るメンバーまたは bot。",
            },
            required=True,
        ),
        amount: str = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to add. Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要增加的{CURRENCY_NAME}，可加逗號",
                Locale.ja: f"追加する{CURRENCY_NAME}。カンマ可。",
            },
            required=True,
            min_length=1,
        ),
    ) -> None:
        """Credits points to a member through the manual-adjustment audit path."""
        parsed_amount = _parse_positive_amount(raw_amount=amount)
        if parsed_amount is None:
            await _send_ephemeral_response(
                interaction=interaction, embed=self._invalid_admin_amount_embed(title="退稅失敗")
            )
            return
        await self._run_admin_adjustment(
            interaction=interaction,
            member=member,
            action="refund_tax",
            title="退稅完成",
            delta=parsed_amount,
        )

    @admin.subcommand(
        name="collect_tax",
        description=f"Admin-only: debit {CURRENCY_NAME} from a member or bot.",
        name_localizations={Locale.zh_TW: "收稅", Locale.ja: "徴税"},
        description_localizations={
            Locale.zh_TW: f"管理員限定：無條件扣除某位成員或 bot 的{CURRENCY_NAME}",
            Locale.ja: f"管理者専用：メンバーまたは bot から{CURRENCY_NAME}を徴収します。",
        },
    )
    async def admin_collect_tax(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description=f"The member or bot to debit the {CURRENCY_NAME} from.",
            name_localizations={Locale.zh_TW: "對象", Locale.ja: "対象"},
            description_localizations={
                Locale.zh_TW: f"要扣除{CURRENCY_NAME}的成員或 bot",
                Locale.ja: f"{CURRENCY_NAME}を徴収するメンバーまたは bot。",
            },
            required=True,
        ),
        amount: str = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to debit. Commas are allowed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要扣除的{CURRENCY_NAME}，可加逗號",
                Locale.ja: f"徴収する{CURRENCY_NAME}。カンマ可。",
            },
            required=True,
            min_length=1,
        ),
    ) -> None:
        """Debits points from a member through the manual-adjustment audit path."""
        parsed_amount = _parse_positive_amount(raw_amount=amount)
        if parsed_amount is None:
            await _send_ephemeral_response(
                interaction=interaction, embed=self._invalid_admin_amount_embed(title="收稅失敗")
            )
            return
        await self._run_admin_adjustment(
            interaction=interaction,
            member=member,
            action="collect_tax",
            title="收稅完成",
            delta=-parsed_amount,
        )

    @staticmethod
    def _invalid_admin_amount_embed(title: str) -> Embed:
        """Builds the validation embed for malformed admin adjustment amounts."""
        return Embed(
            title=title,
            description="### 金額格式錯誤\n請輸入正整數，可以加逗號，例如 `1,000`。",
            color=_ERROR_COLOR,
        )

    async def _run_admin_adjustment(
        self, interaction: Interaction, member: Member, action: str, title: str, delta: int
    ) -> None:
        """Runs a gated admin balance adjustment and publishes successful results."""
        if interaction.user is None:
            return
        actor = interaction.user
        guild = getattr(interaction, "guild", None)
        actor_avatar_url = await guild_avatar_url(user=actor, guild=guild)
        if not await get_admin(user_id=actor.id):
            await interaction.response.defer(ephemeral=True)
            embed = Embed(
                title="權限不足",
                description="### 只有 economy admin 可以執行這個操作",
                color=_ERROR_COLOR,
            )
            embed.set_author(name=actor.display_name, icon_url=actor_avatar_url)
            await _send_private_followup(interaction=interaction, embed=embed)
            return
        await interaction.response.defer()
        member_avatar_url = await guild_avatar_url(user=member, guild=guild)
        result = await adjust_balance(
            user_id=member.id,
            name=member.name,
            delta=delta,
            allow_negative=False,
            avatar_url=member_avatar_url,
        )
        embed = Embed(
            title=title,
            description=(
                f"### {member.mention}\n"
                f"{currency_text(amount=result.applied_delta, signed=True, compact=True)}"
            ),
            color=_ADMIN_COLOR,
        )
        embed.set_author(name=actor.display_name, icon_url=actor_avatar_url)
        _set_optional_thumbnail(embed=embed, avatar_url=member_avatar_url)
        embed.add_field(
            name="操作結果",
            value=(
                f"申請 {amount_code(amount=delta, signed=True, compact=True)}\n"
                f"實際 {amount_code(amount=result.applied_delta, signed=True, compact=True)}\n"
                f"餘額 {amount_code(amount=result.new_balance, compact=True)}"
            ),
            inline=False,
        )
        if action == "collect_tax" and result.applied_delta != delta:
            embed.set_footer(text="收稅最多扣到餘額 0")
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="balance",
        description=f"Check a member's {CURRENCY_NAME} balance, loans, stocks, and VIP status.",
        name_localizations={Locale.zh_TW: "餘額", Locale.ja: "残高"},
        description_localizations={
            Locale.zh_TW: f"查詢成員的{CURRENCY_NAME}餘額、借貸、股票與 VIP 狀態",
            Locale.ja: f"member の{CURRENCY_NAME}残高、loan、stock、VIP 状態を確認します。",
        },
        nsfw=False,
    )
    async def balance(
        self,
        interaction: Interaction,
        member: Member | None = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="Member to inspect; defaults to yourself.",
            name_localizations={Locale.zh_TW: "成員", Locale.ja: "メンバー"},
            description_localizations={
                Locale.zh_TW: "要查看的成員；預設是自己",
                Locale.ja: "表示する member。省略時は自分。",
            },
            required=False,
            default=None,
        ),
    ) -> None:
        """Replies with a member's balance, loans, stocks, and VIP status.

        Args:
            interaction: The interaction that triggered the command.
            member: Optional member to inspect.
        """
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        target = member or interaction.user
        portfolio = await get_portfolio(user_id=target.id)
        stock_portfolio = await get_stock_portfolio(user_id=target.id)
        is_vip = await get_vip(user_id=target.id)
        age_days = (datetime.now(tz=UTC) - target.created_at).days
        net_worth = portfolio.net_worth + stock_portfolio.equity_value

        embed = Embed(
            title="💰 財務總覽",
            color=_BALANCE_COLOR,
            description=(
                f"## {target.display_name}\n淨資產 {bold_currency(amount=net_worth, compact=True)}"
            ),
        )
        embed.set_author(
            name=f"{target.display_name} 的財務總覽", icon_url=target.display_avatar.url
        )
        _set_optional_thumbnail(embed=embed, avatar_url=target.display_avatar.url)

        embed.add_field(
            name="現金", value=amount_code(amount=portfolio.balance, compact=True), inline=True
        )
        embed.add_field(
            name="股票淨值",
            value=(
                f"估值 {amount_code(amount=stock_portfolio.equity_value, compact=True)}\n"
                f"未實現 "
                f"{amount_code(amount=stock_portfolio.unrealized_pnl, signed=True, compact=True)}\n"
                f"已實現 "
                f"{amount_code(amount=stock_portfolio.realized_pnl, signed=True, compact=True)}"
            ),
            inline=True,
        )
        embed.add_field(
            name="債務",
            value=_debt_summary_text(
                principal=portfolio.debt_principal, interest=portfolio.debt_interest
            ),
            inline=True,
        )
        embed.add_field(
            name="股票部位",
            value=_stock_position_lines(stock_portfolio=stock_portfolio),
            inline=False,
        )
        embed.add_field(name="會員狀態", value=_vip_status_text(is_vip=is_vip), inline=False)

        vip_badge = " · 👑 VIP" if is_vip else ""
        embed.set_footer(text=f"帳號 {age_days} 天{vip_badge}")
        await _send_private_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="leaderboard",
        description=f"Show the global top {CURRENCY_NAME} holders.",
        name_localizations={Locale.zh_TW: "排行榜", Locale.ja: "リーダーボード"},
        description_localizations={
            Locale.zh_TW: f"顯示全域 {CURRENCY_NAME}前 10 名",
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
        rows = await top_n(limit=10)
        if not rows:
            embed = Embed(
                title=f"🏆 {CURRENCY_NAME} Top 10",
                description="### 尚未開張\n/games blackjack 或 /games dragon_gate 開局就會上榜",
                color=_LEADERBOARD_COLOR,
            )
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        champion = rows[0]
        board = build_balance_leaderboard_board_image(rows=rows)
        embed = Embed(
            title=f"🏆 {CURRENCY_NAME} Top 10",
            description="### 公開排行榜\n依可用餘額排序。",
            color=_LEADERBOARD_COLOR,
        )
        embed.set_author(name="目前第一名", icon_url=champion.avatar_url or None)
        _set_optional_thumbnail(embed=embed, avatar_url=champion.avatar_url)
        embed.set_image(url=f"attachment://{BALANCE_LEADERBOARD_BOARD_FILENAME}")
        await _send_expiring_followup(
            interaction=interaction,
            embed=embed,
            file=nextcord.File(fp=BytesIO(board), filename=BALANCE_LEADERBOARD_BOARD_FILENAME),
        )

    @nextcord.slash_command(
        name="loss_leaderboard",
        description=f"Show today's accumulated {CURRENCY_NAME} casino losses.",
        name_localizations={Locale.zh_TW: "輸錢榜", Locale.ja: "負け額ランキング"},
        description_localizations={
            Locale.zh_TW: f"顯示今日累計輸掉{CURRENCY_NAME}的前 10 名 (每天 0:00 重置)",
            Locale.ja: f"本日累計で失った{CURRENCY_NAME}の上位10名 (毎日 0:00 リセット)。",
        },
        nsfw=False,
    )
    async def loss_leaderboard(self, interaction: Interaction) -> None:
        """Replies with the top 10 gross casino losses for the current day.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer()
        rows = await top_losers(limit=10)
        if not rows:
            embed = Embed(
                title=f"💸 今日輸局累計 {CURRENCY_NAME}",
                description="### 今天還沒有人輸錢\n/games blackjack 或 /games dragon_gate 開局就可能進榜",
                color=_LOSS_LEADERBOARD_COLOR,
            )
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        champion = rows[0]
        board = build_loss_leaderboard_board_image(rows=rows)
        embed = Embed(
            title=f"💸 今日輸局累計 {CURRENCY_NAME}",
            description="### 今日累計輸排序\n以 gross loss 排名，贏回來不抵扣。",
            color=_LOSS_LEADERBOARD_COLOR,
        )
        embed.set_author(name="今日累計輸最多", icon_url=champion.avatar_url or None)
        _set_optional_thumbnail(embed=embed, avatar_url=champion.avatar_url)
        embed.set_image(url=f"attachment://{LOSS_LEADERBOARD_BOARD_FILENAME}")
        embed.set_footer(text="今日實際輸掉累計 | 贏回來不抵扣 | 每天 0:00 (Asia/Taipei) 重置")
        await _send_expiring_followup(
            interaction=interaction,
            embed=embed,
            file=nextcord.File(fp=BytesIO(board), filename=LOSS_LEADERBOARD_BOARD_FILENAME),
        )

    @nextcord.slash_command(
        name="give",
        description=f"Transfer your {CURRENCY_NAME} to another member or bot.",
        name_localizations={Locale.zh_TW: "轉帳", Locale.ja: "虛擬歡樂豆送付"},
        description_localizations={
            Locale.zh_TW: f"把你的{CURRENCY_NAME}轉給其他成員或 bot",
            Locale.ja: f"他のメンバーまたは bot に{CURRENCY_NAME}を送ります。",
        },
        nsfw=False,
    )
    async def give(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description=f"The member or bot to receive the {CURRENCY_NAME}.",
            name_localizations={Locale.zh_TW: "對象", Locale.ja: "受取人"},
            description_localizations={
                Locale.zh_TW: f"要接收{CURRENCY_NAME}的成員或 bot",
                Locale.ja: f"{CURRENCY_NAME}を受け取るメンバーまたは bot。",
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
        """Transfers points from the caller to `member`.

        Args:
            interaction: The interaction that triggered the command.
            member: The recipient.
            amount: How many points to transfer.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return

        sender = interaction.user
        guild = getattr(interaction, "guild", None)
        sender_avatar_url = await guild_avatar_url(user=sender, guild=guild)

        if member.id == sender.id:
            embed = Embed(title="轉帳失敗", description="### 不能轉給自己", color=_ERROR_COLOR)
            embed.set_author(name=sender.display_name, icon_url=sender_avatar_url)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        receiver_avatar_url = await guild_avatar_url(user=member, guild=guild)
        transfer_result = await transfer(
            sender_id=sender.id,
            sender_name=sender.name,
            sender_avatar_url=sender_avatar_url,
            receiver_id=member.id,
            receiver_name=member.name,
            receiver_avatar_url=receiver_avatar_url,
            amount=amount,
        )
        if transfer_result is None:
            balance_now = await get_balance(user_id=sender.id)
            embed = Embed(
                title="轉帳失敗",
                description=(
                    f"### 餘額不足\n"
                    f"目前 {bold_currency(amount=balance_now, compact=True)}\n"
                    f"想轉 {bold_currency(amount=amount, compact=True)}"
                ),
                color=_ERROR_COLOR,
            )
            embed.set_author(name=sender.display_name, icon_url=sender_avatar_url)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="💸 轉帳完成",
            description=f"### {currency_text(amount=amount, compact=True)}\n{sender.mention} → {member.mention}",
            color=_TRANSFER_COLOR,
        )
        embed.set_author(name=sender.display_name, icon_url=sender_avatar_url)
        _set_optional_thumbnail(embed=embed, avatar_url=receiver_avatar_url)
        embed.add_field(
            name="轉帳後餘額",
            value=(
                f"**{sender.display_name}** "
                f"{amount_code(amount=transfer_result.sender_balance, compact=True)}\n"
                f"**{member.display_name}** "
                f"{amount_code(amount=transfer_result.receiver_balance, compact=True)}"
            ),
            inline=False,
        )
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="casino",
        description="Show the casino system's cumulative profit and loss.",
        name_localizations={Locale.zh_TW: "賭場", Locale.ja: "カジノ"},
        description_localizations={
            Locale.zh_TW: "顯示賭場系統累積 P&L (跨伺服器)",
            Locale.ja: "カジノシステムの累計 P&L を表示します。",
        },
        nsfw=False,
    )
    async def casino(self, interaction: Interaction) -> None:
        """Shows the casino system's accumulated P&L (was `/house`)."""
        await interaction.response.defer()
        snapshot = await get_casino_ledger()
        balance = snapshot.balance
        total_earned = snapshot.total_earned
        total_spent = snapshot.total_spent

        if balance > 0:
            verdict = rf"\+ {bold_currency(amount=balance, compact=True)}"
            color = _BALANCE_COLOR
        elif balance < 0:
            verdict = rf"\- {bold_currency(amount=abs(balance), compact=True)}"
            color = _ERROR_COLOR
        else:
            verdict = "⚖️ 打平"
            color = _CASINO_COLOR

        embed = Embed(title="🎰 賭場戰績", description=f"## {verdict}", color=color)
        embed.set_author(name="賭場系統")
        embed.add_field(
            name="流水",
            value=(
                f"贏到 {amount_code(amount=total_earned, compact=True)}\n"
                f"賠出 {amount_code(amount=total_spent, compact=True)}"
            ),
            inline=False,
        )
        embed.set_footer(text="跨伺服器累積 | 賭場資金無上限")
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="pocat",
        description="Show the bot player's own wallet (shortcut for /balance @bot).",
        name_localizations={Locale.zh_TW: "破貓", Locale.ja: "ポキャット"},
        description_localizations={
            Locale.zh_TW: "顯示機器人玩家自己的錢包 (等同 /balance @bot)",
            Locale.ja: "ボットプレイヤー自身の財布を表示します (/balance @bot のショートカット)。",
        },
        nsfw=False,
    )
    async def pocat(self, interaction: Interaction) -> None:
        """Shows the bot player's `user_wallet` balance and gross flows."""
        await interaction.response.defer()
        if self.bot.user is None:
            embed = Embed(
                title="❌ 無法查詢", description="目前無法取得機器人身份", color=_ERROR_COLOR
            )
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        bot_user = self.bot.user
        account = await get_account(user_id=bot_user.id)
        name = bot_user.display_name
        if account is None:
            balance, total_earned, total_spent = 0, 0, 0
        else:
            balance = account.balance
            total_earned = account.total_earned
            total_spent = account.total_spent
            name = name or account.name

        if balance > 0:
            verdict = rf"{bold_currency(amount=balance, compact=True)}"
            color = _BALANCE_COLOR
        elif balance < 0:
            verdict = rf"\- {bold_currency(amount=abs(balance), compact=True)}"
            color = _ERROR_COLOR
        else:
            verdict = "餘額 0"
            color = _CASINO_COLOR

        embed = Embed(title="🐱 破貓戰績", description=f"## {verdict}", color=color)
        embed.set_author(name=name, icon_url=bot_user.display_avatar.url)
        _set_optional_thumbnail(embed=embed, avatar_url=bot_user.display_avatar.url)
        embed.add_field(
            name="流水",
            value=(
                f"贏到 {amount_code(amount=total_earned, compact=True)}\n"
                f"賠出 {amount_code(amount=total_spent, compact=True)}"
            ),
            inline=False,
        )
        embed.set_footer(text="bot 玩家錢包")
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="credit",
        description="Personal credit operations.",
        name_localizations={Locale.zh_TW: "信貸", Locale.ja: "信用"},
        description_localizations={
            Locale.zh_TW: "個人信貸操作",
            Locale.ja: "personal credit 操作。",
        },
        nsfw=False,
    )
    async def credit(self, interaction: Interaction) -> None:
        """Slash command group for personal credit operations."""

    @credit.subcommand(
        name="borrow",
        description=f"Request a personal {CURRENCY_NAME} loan from another member.",
        name_localizations={Locale.zh_TW: "借款", Locale.ja: "借入"},
        description_localizations={
            Locale.zh_TW: f"向指定成員提出{CURRENCY_NAME}借款申請",
            Locale.ja: f"指定メンバーに{CURRENCY_NAME}借入リクエストを送ります。",
        },
    )
    async def credit_borrow(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="The member you want to borrow from.",
            name_localizations={Locale.zh_TW: "貸方", Locale.ja: "貸し手"},
            description_localizations={
                Locale.zh_TW: "要向誰借款",
                Locale.ja: "借入先のメンバー。",
            },
            required=True,
        ),
        amount: int = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to request.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要借入的{CURRENCY_NAME}",
                Locale.ja: f"借入する{CURRENCY_NAME}。",
            },
            required=True,
            min_value=1,
        ),
        monthly_rate_percent: float = SlashOption(
            name="monthly_rate_percent",
            description="Monthly simple-interest rate percent.",
            name_localizations={Locale.zh_TW: "月利率", Locale.ja: "月利率"},
            description_localizations={
                Locale.zh_TW: "每月單利百分比",
                Locale.ja: "月次 simple interest rate percent。",
            },
            required=False,
            default=DEFAULT_LOAN_MONTHLY_RATE_BPS / 100,
            min_value=0,
            max_value=100,
        ),
    ) -> None:
        """Creates a personal loan request for the target lender.

        Args:
            interaction: The interaction that triggered the command.
            member: The requested lender.
            amount: How many points to borrow.
            monthly_rate_percent: Monthly simple-interest rate.
        """
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user
        guild = getattr(interaction, "guild", None)
        user_avatar_url = await guild_avatar_url(user=user, guild=guild)
        lender_avatar_url = await guild_avatar_url(user=member, guild=guild)
        if member.bot:
            embed = Embed(title="借款失敗", description="### 不能向 bot 借款", color=_ERROR_COLOR)
            embed.set_author(name=user.display_name, icon_url=user_avatar_url)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return
        if member.id == user.id:
            embed = Embed(title="借款失敗", description="### 不能向自己借款", color=_ERROR_COLOR)
            embed.set_author(name=user.display_name, icon_url=user_avatar_url)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        monthly_rate_bps = monthly_rate_percent_to_bps(monthly_rate_percent=monthly_rate_percent)
        proposal = await create_personal_loan_request(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            lender_id=member.id,
            lender_name=member.name,
            lender_avatar_url=lender_avatar_url,
            amount=amount,
            monthly_rate_bps=monthly_rate_bps,
        )
        if proposal is None:
            embed = Embed(title="借款失敗", description="### 無法建立借款申請", color=_ERROR_COLOR)
            embed.set_author(name=user.display_name, icon_url=user_avatar_url)
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="💴 信貸申請已建立",
            description=(
                f"### {user.mention} → {member.mention}\n"
                f"{currency_text(amount=amount, compact=True)}"
            ),
            color=_BORROW_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user_avatar_url)
        _set_optional_thumbnail(embed=embed, avatar_url=lender_avatar_url)
        embed.add_field(
            name="條款",
            value=_loan_terms_text(amount=amount, monthly_rate_bps=monthly_rate_bps),
            inline=False,
        )
        embed.set_footer(text=_credit_request_footer())
        await _send_loan_request_followup(
            interaction=interaction,
            embed=embed,
            view=CreditLoanDecisionView(
                proposal_id=proposal.proposal_id, lender_id=member.id, creator_id=user.id
            ),
        )

    @credit.subcommand(
        name="repay",
        description=f"Repay a personal {CURRENCY_NAME} loan to a member.",
        name_localizations={Locale.zh_TW: "還款", Locale.ja: "返済"},
        description_localizations={
            Locale.zh_TW: "還款給指定貸方",
            Locale.ja: "指定 lender へ personal loan を返済します。",
        },
    )
    async def credit_repay(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="The lender to repay.",
            name_localizations={Locale.zh_TW: "貸方", Locale.ja: "貸し手"},
            description_localizations={Locale.zh_TW: "要還款給誰", Locale.ja: "返済先の lender。"},
            required=True,
        ),
        amount: int = SlashOption(
            name="amount",
            description=f"Maximum {CURRENCY_NAME} to apply against personal debt.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要還款的最高{CURRENCY_NAME}",
                Locale.ja: f"返済する{CURRENCY_NAME}の上限。",
            },
            required=True,
            min_value=1,
        ),
    ) -> None:
        """Pays down active personal loans owed to `member`.

        Args:
            interaction: The interaction that triggered the command.
            member: The personal lender.
            amount: Maximum amount to repay.
        """
        if interaction.user is None:
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(
            user=user, guild=getattr(interaction, "guild", None)
        )

        result = await repay_personal_loans(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            lender_id=member.id,
            amount=amount,
        )
        if result is None:
            reason = f"沒有可還給 {member.display_name} 的有效個人借款"
            await interaction.response.defer(ephemeral=True)
            embed = Embed(title="還款失敗", description=f"### {reason}", color=_ERROR_COLOR)
            embed.set_author(name=user.display_name, icon_url=user_avatar_url)
            _set_optional_thumbnail(embed=embed, avatar_url=user_avatar_url)
            await _send_private_followup(interaction=interaction, embed=embed)
            return

        await interaction.response.defer()
        embed = Embed(
            title="🧾 信貸還款完成",
            description=(
                f"### {currency_text(amount=-result.paid_amount, signed=True, compact=True)} 扣款"
            ),
            color=_REPAY_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user_avatar_url)
        _set_optional_thumbnail(embed=embed, avatar_url=user_avatar_url)
        embed.add_field(
            name=f"還給 {member.display_name}",
            value=_payment_summary_text(
                paid_amount=result.paid_amount,
                interest_paid=result.interest_paid,
                principal_paid=result.principal_paid,
                remaining_principal=result.remaining_principal,
                remaining_interest=result.remaining_interest,
                borrower_balance=result.borrower_balance,
            ),
            inline=False,
        )
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @credit.subcommand(
        name="call",
        description=f"Forcibly collect a personal {CURRENCY_NAME} loan.",
        name_localizations={Locale.zh_TW: "催收", Locale.ja: "回収"},
        description_localizations={
            Locale.zh_TW: "從借方可用餘額強制回收個人借款",
            Locale.ja: "借り手の利用可能残高から personal loan を回収します。",
        },
    )
    async def credit_call(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="The borrower to collect from.",
            name_localizations={Locale.zh_TW: "借方", Locale.ja: "借り手"},
            description_localizations={
                Locale.zh_TW: "要向誰強制回收",
                Locale.ja: "回収対象の borrower。",
            },
            required=True,
        ),
        amount: int = SlashOption(
            name="amount",
            description="Maximum amount to collect; omit or 0 means all owed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: "最多回收多少；0 代表嘗試全收",
                Locale.ja: "回収上限。0 は全額。",
            },
            required=False,
            default=0,
            min_value=0,
        ),
    ) -> None:
        """Forcibly collects a personal loan from a borrower."""
        if interaction.user is None:
            return
        user = interaction.user
        borrower_avatar_url = await guild_avatar_url(
            user=member, guild=getattr(interaction, "guild", None)
        )
        result = await call_personal_loans(
            lender_id=user.id,
            borrower_id=member.id,
            borrower_name=member.name,
            borrower_avatar_url=borrower_avatar_url,
            amount=amount or None,
        )
        if result is None:
            await interaction.response.defer(ephemeral=True)
            embed = Embed(
                title="催收失敗",
                description=f"### {member.display_name} 沒有欠你有效個人借款，或目前無可扣餘額",
                color=_ERROR_COLOR,
            )
            embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
            await _send_private_followup(interaction=interaction, embed=embed)
            return
        await interaction.response.defer()
        embed = Embed(
            title="📣 信貸催收完成",
            description=(
                f"### 從 {member.mention} 回收 "
                f"{currency_text(amount=result.paid_amount, compact=True)}"
            ),
            color=_REPAY_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
        embed.add_field(
            name="回收明細",
            value=_payment_summary_text(
                paid_amount=result.paid_amount,
                interest_paid=result.interest_paid,
                principal_paid=result.principal_paid,
                remaining_principal=result.remaining_principal,
                remaining_interest=result.remaining_interest,
                borrower_balance=result.borrower_balance,
            ),
            inline=False,
        )
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @credit.subcommand(
        name="status",
        description="Show your active personal loan contracts.",
        name_localizations={Locale.zh_TW: "狀態", Locale.ja: "状態"},
        description_localizations={
            Locale.zh_TW: "查看你的有效個人信貸",
            Locale.ja: "active personal loan contracts を表示します。",
        },
    )
    async def credit_status(self, interaction: Interaction) -> None:
        """Shows the caller's active personal credit contracts."""
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        contracts = [
            contract
            for contract in await list_loan_contracts(user_id=interaction.user.id)
            if contract.lender_type == LoanLenderType.USER
        ]
        if not contracts:
            embed = Embed(
                title="信貸狀態", description="### 目前沒有有效信貸", color=_BORROW_COLOR
            )
            await _send_private_followup(interaction=interaction, embed=embed)
            return
        lines = [
            (
                f"{'欠 ' + contract.lender_name if contract.borrower_id == interaction.user.id else contract.borrower_name + ' 欠你'} "
                f"本金 {amount_code(amount=contract.principal_remaining, compact=True)} · "
                f"利息 {amount_code(amount=contract.interest_due, compact=True)} · "
                f"{_rate_text(monthly_rate_bps=contract.monthly_rate_bps)}"
            )
            for contract in contracts[:10]
        ]
        embed = Embed(title="信貸狀態", description="\n".join(lines), color=_BORROW_COLOR)
        await _send_private_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="central_bank",
        description="Central bank lending operations.",
        name_localizations={Locale.zh_TW: "中央銀行", Locale.ja: "中央銀行"},
        description_localizations={
            Locale.zh_TW: "中央銀行借款操作",
            Locale.ja: "中央銀行 loan 操作。",
        },
        nsfw=False,
    )
    async def central_bank(self, interaction: Interaction) -> None:
        """Slash command group for central bank operations."""

    @central_bank.subcommand(
        name="borrow",
        description=f"Request a central bank {CURRENCY_NAME} loan.",
        name_localizations={Locale.zh_TW: "借款", Locale.ja: "借入"},
        description_localizations={
            Locale.zh_TW: f"向中央銀行提出{CURRENCY_NAME}借款申請",
            Locale.ja: f"中央銀行に{CURRENCY_NAME}借入 request を送ります。",
        },
    )
    async def central_bank_borrow(
        self,
        interaction: Interaction,
        amount: int = SlashOption(
            name="amount",
            description=f"How much {CURRENCY_NAME} to request.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: f"要向中央銀行借的{CURRENCY_NAME}",
                Locale.ja: f"中央銀行から借入する{CURRENCY_NAME}。",
            },
            required=True,
            min_value=1,
        ),
        monthly_rate_percent: float = SlashOption(
            name="monthly_rate_percent",
            description="Monthly simple-interest rate percent.",
            name_localizations={Locale.zh_TW: "月利率", Locale.ja: "月利率"},
            description_localizations={
                Locale.zh_TW: "每月單利百分比",
                Locale.ja: "月次 simple interest rate percent。",
            },
            required=False,
            default=DEFAULT_LOAN_MONTHLY_RATE_BPS / 100,
            min_value=0,
            max_value=100,
        ),
    ) -> None:
        """Creates a central bank loan request."""
        await interaction.response.defer()
        if interaction.user is None:
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(
            user=user, guild=getattr(interaction, "guild", None)
        )
        monthly_rate_bps = monthly_rate_percent_to_bps(monthly_rate_percent=monthly_rate_percent)
        proposal = await create_central_bank_loan_request(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            amount=amount,
            monthly_rate_bps=monthly_rate_bps,
        )
        if proposal is None:
            embed = Embed(
                title="央行借款失敗", description="### 無法建立央行借款申請", color=_ERROR_COLOR
            )
            await _send_expiring_followup(interaction=interaction, embed=embed)
            return
        embed = Embed(
            title="🏛️ 央行借款申請已建立",
            description=f"### {user.mention}\n{currency_text(amount=amount, compact=True)}",
            color=_CENTRAL_BANK_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user_avatar_url)
        embed.add_field(
            name="條款",
            value=_loan_terms_text(amount=amount, monthly_rate_bps=monthly_rate_bps),
            inline=False,
        )
        embed.set_footer(text=_central_bank_request_footer())
        await _send_loan_request_followup(
            interaction=interaction,
            embed=embed,
            view=CentralBankLoanDecisionView(
                bot=self.bot,
                proposal_id=proposal.proposal_id,
                creator_id=user.id,
                allow_self_approval=EconomyConfig().allow_central_bank_self_approval,
            ),
        )

    @central_bank.subcommand(
        name="repay",
        description="Repay your central bank loan.",
        name_localizations={Locale.zh_TW: "還款", Locale.ja: "返済"},
        description_localizations={
            Locale.zh_TW: "償還自己的央行借款",
            Locale.ja: "自分の central bank loan を返済します。",
        },
    )
    async def central_bank_repay(
        self,
        interaction: Interaction,
        amount: int = SlashOption(
            name="amount",
            description="Maximum amount to repay.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={Locale.zh_TW: "最多還款多少", Locale.ja: "返済上限。"},
            required=True,
            min_value=1,
        ),
    ) -> None:
        """Repays central-bank debt."""
        if interaction.user is None:
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(
            user=user, guild=getattr(interaction, "guild", None)
        )
        result = await repay_central_bank_loans(
            borrower_id=user.id,
            borrower_name=user.name,
            borrower_avatar_url=user_avatar_url,
            amount=amount,
        )
        if result is None:
            await interaction.response.defer(ephemeral=True)
            embed = Embed(
                title="央行還款失敗",
                description="### 沒有有效央行借款，或目前無可扣餘額",
                color=_ERROR_COLOR,
            )
            await _send_private_followup(interaction=interaction, embed=embed)
            return
        await interaction.response.defer()
        embed = Embed(
            title="🏛️ 央行還款完成",
            description=(
                f"### {user.mention}\n"
                + _payment_summary_text(
                    paid_amount=result.paid_amount,
                    interest_paid=result.interest_paid,
                    principal_paid=result.principal_paid,
                    remaining_principal=result.remaining_principal,
                    remaining_interest=result.remaining_interest,
                    borrower_balance=result.borrower_balance,
                )
            ),
            color=_CENTRAL_BANK_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user_avatar_url)
        _set_optional_thumbnail(embed=embed, avatar_url=user_avatar_url)
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @central_bank.subcommand(
        name="call",
        description="Central banker forced collection from a borrower.",
        name_localizations={Locale.zh_TW: "催收", Locale.ja: "回収"},
        description_localizations={
            Locale.zh_TW: "央行成員從借方可用餘額強制回收",
            Locale.ja: "central banker が borrower から強制回収します。",
        },
    )
    async def central_bank_call(
        self,
        interaction: Interaction,
        member: Member = SlashOption(  # noqa: B008 -- nextcord SlashOption is the canonical default
            name="member",
            description="The borrower to collect from.",
            name_localizations={Locale.zh_TW: "借方", Locale.ja: "借り手"},
            description_localizations={
                Locale.zh_TW: "要向誰催收",
                Locale.ja: "回収対象 borrower。",
            },
            required=True,
        ),
        amount: int = SlashOption(
            name="amount",
            description="Maximum amount to collect; 0 means all owed.",
            name_localizations={Locale.zh_TW: "金額", Locale.ja: "金額"},
            description_localizations={
                Locale.zh_TW: "最多回收多少；0 代表嘗試全收",
                Locale.ja: "回収上限。0 は全額。",
            },
            required=False,
            default=0,
            min_value=0,
        ),
    ) -> None:
        """Central-bank forced collection."""
        if interaction.user is None:
            return
        if not await get_central_banker(user_id=interaction.user.id):
            await interaction.response.defer(ephemeral=True)
            embed = Embed(
                title="權限不足",
                description="### 只有央行成員可以執行央行催收",
                color=_ERROR_COLOR,
            )
            await _send_private_followup(interaction=interaction, embed=embed)
            return
        borrower_avatar_url = await guild_avatar_url(
            user=member, guild=getattr(interaction, "guild", None)
        )
        result = await call_central_bank_loans(
            borrower_id=member.id,
            borrower_name=member.name,
            borrower_avatar_url=borrower_avatar_url,
            amount=amount or None,
        )
        if result is None:
            await interaction.response.defer(ephemeral=True)
            embed = Embed(
                title="央行催收失敗",
                description="### 目標沒有有效央行借款，或目前無可扣餘額",
                color=_ERROR_COLOR,
            )
            await _send_private_followup(interaction=interaction, embed=embed)
            return
        await interaction.response.defer()
        embed = Embed(
            title="🏛️ 央行催收完成",
            description=(
                f"### 從 {member.mention} 回收\n"
                + _payment_summary_text(
                    paid_amount=result.paid_amount,
                    interest_paid=result.interest_paid,
                    principal_paid=result.principal_paid,
                    remaining_principal=result.remaining_principal,
                    remaining_interest=result.remaining_interest,
                    borrower_balance=result.borrower_balance,
                )
            ),
            color=_CENTRAL_BANK_COLOR,
        )
        embed.set_author(
            name=interaction.user.display_name, icon_url=interaction.user.display_avatar.url
        )
        _set_optional_thumbnail(embed=embed, avatar_url=borrower_avatar_url)
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @central_bank.subcommand(
        name="status",
        description="Show central bank lending capacity.",
        name_localizations={Locale.zh_TW: "狀態", Locale.ja: "状態"},
        description_localizations={
            Locale.zh_TW: "查看中央銀行可放貸額度",
            Locale.ja: "中央銀行の lending capacity を表示します。",
        },
    )
    async def central_bank_status(self, interaction: Interaction) -> None:
        """Shows central bank lending capacity."""
        await interaction.response.defer()
        exclude_user_ids = (self.bot.user.id,) if self.bot.user else ()
        status = await get_central_bank_status(exclude_user_ids=exclude_user_ids)
        embed = Embed(
            title="🏛️ 中央銀行狀態",
            description=f"## 可放貸 {bold_currency(amount=status.available_credit, compact=True)}",
            color=_CENTRAL_BANK_COLOR,
        )
        embed.add_field(
            name="資金池",
            value=(
                f"全體正餘額 "
                f"{amount_code(amount=status.total_positive_user_balance, compact=True)}\n"
                f"未還本金 {amount_code(amount=status.outstanding_principal, compact=True)}"
            ),
            inline=False,
        )
        await _send_expiring_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="checkin",
        description=f"Daily {CURRENCY_NAME} check-in with a 7-day streak bonus.",
        name_localizations={Locale.zh_TW: "簽到", Locale.ja: "デイリーチェックイン"},
        description_localizations={
            Locale.zh_TW: (
                f"每日簽到領 {currency_text(amount=BASE_CHECKIN_REWARD_AMOUNT, compact=True)}, "
                "連續 7 天加成, VIP 2x"
            ),
            Locale.ja: (
                f"毎日{currency_text(amount=BASE_CHECKIN_REWARD_AMOUNT, compact=True)}, "
                "7日連続でボーナス。"
            ),
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
        user_avatar_url = await guild_avatar_url(
            user=user, guild=getattr(interaction, "guild", None)
        )
        result = await checkin(user_id=user.id, name=user.name, avatar_url=user_avatar_url)
        if result is None:
            embed = Embed(
                title="今天已經簽到過了",
                description="### 0:00 (Asia/Taipei) 後再回來簽吧",
                color=_ERROR_COLOR,
            )
            embed.set_author(name=user.display_name, icon_url=user_avatar_url)
            await _send_private_followup(interaction=interaction, embed=embed)
            return

        vip_badge = " · 👑 VIP 2x" if result.is_vip else ""
        embed = Embed(
            title="📅 每日簽到",
            description=f"## {currency_text(amount=result.amount, signed=True, compact=True)} 入帳",
            color=_CHECKIN_COLOR,
        )
        embed.set_author(name=f"{user.display_name} 的簽到", icon_url=user_avatar_url)
        _set_optional_thumbnail(embed=embed, avatar_url=user_avatar_url)
        embed.add_field(name="連續簽到", value=f"第 {result.streak} / 7 天", inline=True)
        embed.add_field(
            name="目前餘額",
            value=amount_code(amount=result.new_balance, compact=True),
            inline=True,
        )
        if result.is_vip:
            base_reward = checkin_reward(streak=result.streak, is_vip=False)
            embed.add_field(
                name="👑 VIP加成",
                value=(
                    f"本日簽到 {amount_code(amount=base_reward, compact=True)} → "
                    f"{amount_code(amount=result.amount, compact=True)}"
                ),
                inline=False,
            )
        embed.set_footer(text=f"連續 7 天為一個 cycle | 每天 0:00 (Asia/Taipei) 重置{vip_badge}")
        await _send_private_followup(interaction=interaction, embed=embed)

    @nextcord.slash_command(
        name="vip",
        description=(
            f"Buy permanent VIP for {currency_text(amount=VIP_PURCHASE_COST, compact=True)}: "
            "2x check-in and 1.5x Blackjack wins."
        ),
        name_localizations={Locale.zh_TW: "購買vip", Locale.ja: "vip購入"},
        description_localizations={
            Locale.zh_TW: "購買永久 VIP：簽到 2x、Blackjack 贏局 1.5x",
            Locale.ja: "永久 VIP を購入: check-in 2x、Blackjack 勝利 1.5x。",
        },
        nsfw=False,
    )
    async def vip_command(self, interaction: Interaction) -> None:
        """Buys the permanent VIP perk for a one-time fixed cost.

        Args:
            interaction: The interaction that triggered the command.
        """
        await interaction.response.defer(ephemeral=True)
        if interaction.user is None:
            return
        user = interaction.user
        user_avatar_url = await guild_avatar_url(
            user=user, guild=getattr(interaction, "guild", None)
        )
        already_vip = await get_vip(user_id=user.id)
        if already_vip:
            embed = Embed(
                title="已經是 VIP",
                description="### 你已經擁有永久 VIP 了, 不用再買一次",
                color=_VIP_COLOR,
            )
            embed.set_author(name=user.display_name, icon_url=user_avatar_url)
            embed.add_field(name="👑 VIP加成", value=_vip_perk_lines(), inline=False)
            await _send_private_followup(interaction=interaction, embed=embed)
            return

        result = await buy_vip(user_id=user.id, name=user.name, avatar_url=user_avatar_url)
        if result is None:
            balance_now = await get_balance(user_id=user.id)
            embed = Embed(
                title="VIP 購買失敗",
                description=(
                    f"### 餘額不足\n"
                    f"目前 {bold_currency(amount=balance_now, compact=True)}\n"
                    f"需要 {bold_currency(amount=VIP_PURCHASE_COST, compact=True)}"
                ),
                color=_ERROR_COLOR,
            )
            embed.set_author(name=user.display_name, icon_url=user_avatar_url)
            embed.add_field(name="👑 VIP權益", value=_vip_perk_lines(), inline=False)
            await _send_private_followup(interaction=interaction, embed=embed)
            return

        embed = Embed(
            title="👑 升級 VIP 成功",
            description=(
                f"### {currency_text(amount=-result.cost, signed=True, compact=True)} 扣款\n"
                "簽到與 Blackjack 贏局加成已生效"
            ),
            color=_VIP_COLOR,
        )
        embed.set_author(name=user.display_name, icon_url=user_avatar_url)
        _set_optional_thumbnail(embed=embed, avatar_url=user_avatar_url)
        embed.add_field(name="👑 VIP加成", value=_vip_perk_lines(), inline=False)
        embed.add_field(
            name="目前餘額",
            value=amount_code(amount=result.new_balance, compact=True),
            inline=False,
        )
        await _send_private_followup(interaction=interaction, embed=embed)


def setup(bot: commands.Bot) -> None:
    """Adds the EconomyCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(EconomyCogs(bot), override=True)
