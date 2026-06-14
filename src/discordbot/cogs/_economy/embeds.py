"""Embed and text builders for economy command and view responses."""

from nextcord import Embed
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.stock import StockPortfolioView, StockPortfolioHolding
from discordbot.typings.colors import TRANSFER_COLOR
from discordbot.typings.economy import (
    VIP_PURCHASE_COST,
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    CheckinResult,
    PortfolioView,
    TransferResult,
    LeaderboardEntry,
    LoanContractView,
    CentralBankStatus,
    LoanPaymentResult,
    VipPurchaseResult,
    CasinoLedgerSnapshot,
    LossLeaderboardEntry,
    BalanceAdjustmentResult,
    LoanProposalAcceptResult,
)
from discordbot.utils.number_text import share_quantity_text
from discordbot.cogs._stock.market import format_price
from discordbot.cogs._economy.boards import (
    LOSS_LEADERBOARD_BOARD_FILENAME,
    BALANCE_LEADERBOARD_BOARD_FILENAME,
)
from discordbot.cogs._economy.database import (
    checkin_reward,
    apply_vip_blackjack_bonus,
    monthly_rate_bps_to_percent,
)
from discordbot.cogs._economy.presentation import (
    CURRENCY_NAME,
    amount_code,
    bold_currency,
    currency_text,
)

BALANCE_COLOR = 0x57F287
LEADERBOARD_COLOR = 0xFEE75C
LOSS_LEADERBOARD_COLOR = 0xE67E22
ADMIN_COLOR = 0x3498DB
CASINO_COLOR = 0xEB459E
BORROW_COLOR = 0xF1C40F
REPAY_COLOR = 0x2ECC71
CENTRAL_BANK_COLOR = 0x1ABC9C
CHECKIN_COLOR = 0x9B59B6
VIP_COLOR = 0xF1C40F
ERROR_COLOR = 0xED4245
_STOCK_POSITION_LINE_LIMIT = 5
_STOCK_POSITION_NAME_LIMIT = 20


class TransferParticipant(BaseModel):
    """Display identity for one side of a transfer embed."""

    model_config = ConfigDict(frozen=True)

    mention: str = Field(
        ..., description="Discord mention string (<@user_id>) shown in the embed."
    )
    display_name: str = Field(..., description="Display name shown next to the mention.")


class LoanParty(BaseModel):
    """Display identity for one side of a loan request embed."""

    model_config = ConfigDict(frozen=True)

    mention: str = Field(
        ..., description="Discord mention string (<@user_id>) shown in the embed."
    )
    display_name: str = Field(default="", description="Display name shown next to the mention.")
    avatar_url: str = Field(default="", description="Avatar URL for the embed thumbnail.")


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


def rate_text(monthly_rate_bps: int) -> str:
    """Formats a monthly loan rate."""
    return f"每月 {monthly_rate_bps_to_percent(monthly_rate_bps=monthly_rate_bps):g}%"


def _loan_terms_text(amount: int, monthly_rate_bps: int) -> str:
    """Formats the loan terms shown before acceptance."""
    return (
        f"本金 {amount_code(amount=amount, compact=True)}\n"
        f"利率 `{rate_text(monthly_rate_bps=monthly_rate_bps)}`\n"
        "利息採單利，依經過天數按比例計算\n"
        "還款會先抵利息，再抵本金；貸方可催收"
    )


def payment_summary_text(  # noqa: PLR0913 -- summary needs all visible repayment fields
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


def build_error_embed(
    *,
    title: str,
    description: str,
    author_name: str | None = None,
    author_icon_url: str | None = None,
    thumbnail_url: str | None = None,
) -> Embed:
    """Builds a red error embed with optional author and thumbnail.

    The `description` is used verbatim, including any leading Markdown heading.
    """
    embed = Embed(title=title, description=description, color=ERROR_COLOR)
    if author_name is not None:
        embed.set_author(name=author_name, icon_url=author_icon_url)
    if thumbnail_url is not None:
        _set_optional_thumbnail(embed=embed, avatar_url=thumbnail_url)
    return embed


def build_simple_embed(  # noqa: PLR0913 -- generic single-section embed exposes each optional slot
    *,
    title: str,
    description: str,
    color: int,
    author_name: str | None = None,
    author_icon_url: str | None = None,
    thumbnail_url: str | None = None,
    footer_text: str | None = None,
) -> Embed:
    """Builds a single-section embed with optional author, thumbnail, and footer."""
    embed = Embed(title=title, description=description, color=color)
    if author_name is not None:
        embed.set_author(name=author_name, icon_url=author_icon_url)
    if thumbnail_url is not None:
        _set_optional_thumbnail(embed=embed, avatar_url=thumbnail_url)
    if footer_text is not None:
        embed.set_footer(text=footer_text)
    return embed


def build_invalid_amount_embed(*, title: str) -> Embed:
    """Builds the validation embed for malformed point amount text."""
    return Embed(
        title=title,
        description="### 金額格式錯誤\n請輸入正整數，可以加逗號，例如 `1,000`。",
        color=ERROR_COLOR,
    )


def build_admin_adjustment_embed(  # noqa: PLR0913 -- mirrors every visible adjustment field
    *,
    title: str,
    member_mention: str,
    actor_name: str,
    actor_avatar_url: str,
    member_avatar_url: str,
    requested_delta: int,
    result: BalanceAdjustmentResult,
    is_collect_clamped: bool,
) -> Embed:
    """Builds the public result embed for an admin balance adjustment."""
    embed = Embed(
        title=title,
        description=(
            f"### {member_mention}\n"
            f"{currency_text(amount=result.applied_delta, signed=True, compact=True)}"
        ),
        color=ADMIN_COLOR,
    )
    embed.set_author(name=actor_name, icon_url=actor_avatar_url)
    _set_optional_thumbnail(embed=embed, avatar_url=member_avatar_url)
    embed.add_field(
        name="操作結果",
        value=(
            f"申請 {amount_code(amount=requested_delta, signed=True, compact=True)}\n"
            f"實際 {amount_code(amount=result.applied_delta, signed=True, compact=True)}\n"
            f"餘額 {amount_code(amount=result.new_balance, compact=True)}"
        ),
        inline=False,
    )
    if is_collect_clamped:
        embed.set_footer(text="收稅最多扣到餘額 0")
    return embed


def build_balance_embed(  # noqa: PLR0913 -- mirrors every financial-overview field
    *,
    display_name: str,
    avatar_url: str,
    portfolio: PortfolioView,
    stock_portfolio: StockPortfolioView,
    is_vip: bool,
    age_days: int,
) -> Embed:
    """Builds the private financial-overview embed for a member."""
    net_worth = portfolio.net_worth + stock_portfolio.equity_value
    embed = Embed(
        title="💰 財務總覽",
        color=BALANCE_COLOR,
        description=(f"## {display_name}\n淨資產 {bold_currency(amount=net_worth, compact=True)}"),
    )
    embed.set_author(name=f"{display_name} 的財務總覽", icon_url=avatar_url)
    _set_optional_thumbnail(embed=embed, avatar_url=avatar_url)
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
        name="股票部位", value=_stock_position_lines(stock_portfolio=stock_portfolio), inline=False
    )
    embed.add_field(name="會員狀態", value=_vip_status_text(is_vip=is_vip), inline=False)
    vip_badge = " · 👑 VIP" if is_vip else ""
    embed.set_footer(text=f"帳號 {age_days} 天{vip_badge}")
    return embed


def build_leaderboard_embed(*, champion: LeaderboardEntry) -> Embed:
    """Builds the public balance leaderboard embed referencing its board image."""
    embed = Embed(
        title=f"🏆 {CURRENCY_NAME} Top 10",
        description="### 公開排行榜\n依可用餘額排序。",
        color=LEADERBOARD_COLOR,
    )
    embed.set_author(name="目前第一名", icon_url=champion.avatar_url or None)
    _set_optional_thumbnail(embed=embed, avatar_url=champion.avatar_url)
    embed.set_image(url=f"attachment://{BALANCE_LEADERBOARD_BOARD_FILENAME}")
    return embed


def build_loss_leaderboard_embed(*, champion: LossLeaderboardEntry) -> Embed:
    """Builds the public daily loss leaderboard embed referencing its board image."""
    embed = Embed(
        title=f"💸 今日輸局累計 {CURRENCY_NAME}",
        description="### 今日累計輸排序\n以 gross loss 排名，贏回來不抵扣。",
        color=LOSS_LEADERBOARD_COLOR,
    )
    embed.set_author(name="今日累計輸最多", icon_url=champion.avatar_url or None)
    _set_optional_thumbnail(embed=embed, avatar_url=champion.avatar_url)
    embed.set_image(url=f"attachment://{LOSS_LEADERBOARD_BOARD_FILENAME}")
    embed.set_footer(text="今日實際輸掉累計 | 贏回來不抵扣 | 每天 0:00 (Asia/Taipei) 重置")
    return embed


def build_transfer_embed(  # noqa: PLR0913 -- mirrors both transfer sides and balances
    *,
    amount: int,
    sender: TransferParticipant,
    sender_avatar_url: str,
    receiver: TransferParticipant,
    receiver_avatar_url: str,
    result: TransferResult,
) -> Embed:
    """Builds the public transfer-completed embed."""
    description = (
        f"### {currency_text(amount=amount, compact=True)}\n{sender.mention} → {receiver.mention}"
    )
    if result.tax_amount > 0:
        description += (
            f"\n實收 {currency_text(amount=result.received_amount, compact=True)}"
            f"（已扣稅 {currency_text(amount=result.tax_amount, compact=True)}）"
        )
    embed = Embed(title="💸 轉帳完成", description=description, color=TRANSFER_COLOR)
    embed.set_author(name=sender.display_name, icon_url=sender_avatar_url)
    _set_optional_thumbnail(embed=embed, avatar_url=receiver_avatar_url)
    embed.add_field(
        name="轉帳後餘額",
        value=(
            f"**{sender.display_name}** "
            f"{amount_code(amount=result.sender_balance, compact=True)}\n"
            f"**{receiver.display_name}** "
            f"{amount_code(amount=result.receiver_balance, compact=True)}"
        ),
        inline=False,
    )
    return embed


def build_transfer_insufficient_embed(
    *, sender_name: str, sender_avatar_url: str, balance_now: int, amount: int
) -> Embed:
    """Builds the transfer failure embed for an insufficient sender balance."""
    return build_error_embed(
        title="轉帳失敗",
        description=(
            f"### 餘額不足\n"
            f"目前 {bold_currency(amount=balance_now, compact=True)}\n"
            f"想轉 {bold_currency(amount=amount, compact=True)}"
        ),
        author_name=sender_name,
        author_icon_url=sender_avatar_url,
    )


def build_casino_embed(*, snapshot: CasinoLedgerSnapshot) -> Embed:
    """Builds the casino system cumulative P&L embed."""
    balance = snapshot.balance
    if balance > 0:
        verdict = rf"\+ {bold_currency(amount=balance, compact=True)}"
        color = BALANCE_COLOR
    elif balance < 0:
        verdict = rf"\- {bold_currency(amount=abs(balance), compact=True)}"
        color = ERROR_COLOR
    else:
        verdict = "⚖️ 打平"
        color = CASINO_COLOR
    embed = Embed(title="🎰 賭場戰績", description=f"## {verdict}", color=color)
    embed.set_author(name="賭場系統")
    embed.add_field(
        name="流水",
        value=(
            f"贏到 {amount_code(amount=snapshot.total_earned, compact=True)}\n"
            f"賠出 {amount_code(amount=snapshot.total_spent, compact=True)}"
        ),
        inline=False,
    )
    embed.set_footer(text="跨伺服器累積 | 賭場資金無上限")
    return embed


def build_pocat_embed(
    *, name: str, avatar_url: str, balance: int, total_earned: int, total_spent: int
) -> Embed:
    """Builds the bot player's own wallet embed."""
    if balance > 0:
        verdict = rf"{bold_currency(amount=balance, compact=True)}"
        color = BALANCE_COLOR
    elif balance < 0:
        verdict = rf"\- {bold_currency(amount=abs(balance), compact=True)}"
        color = ERROR_COLOR
    else:
        verdict = "餘額 0"
        color = CASINO_COLOR
    embed = Embed(title="🐱 破貓戰績", description=f"## {verdict}", color=color)
    embed.set_author(name=name, icon_url=avatar_url)
    _set_optional_thumbnail(embed=embed, avatar_url=avatar_url)
    embed.add_field(
        name="流水",
        value=(
            f"贏到 {amount_code(amount=total_earned, compact=True)}\n"
            f"賠出 {amount_code(amount=total_spent, compact=True)}"
        ),
        inline=False,
    )
    embed.set_footer(text="bot 玩家錢包")
    return embed


def build_credit_request_embed(
    *, borrower: LoanParty, lender: LoanParty, amount: int, monthly_rate_bps: int
) -> Embed:
    """Builds the public personal credit request embed."""
    embed = Embed(
        title="💴 信貸申請已建立",
        description=(
            f"### {borrower.mention} → {lender.mention}\n"
            f"{currency_text(amount=amount, compact=True)}"
        ),
        color=BORROW_COLOR,
    )
    embed.set_author(name=borrower.display_name, icon_url=borrower.avatar_url)
    _set_optional_thumbnail(embed=embed, avatar_url=lender.avatar_url)
    embed.add_field(
        name="條款",
        value=_loan_terms_text(amount=amount, monthly_rate_bps=monthly_rate_bps),
        inline=False,
    )
    embed.set_footer(text=_credit_request_footer())
    return embed


def build_central_bank_request_embed(
    *, borrower: LoanParty, amount: int, monthly_rate_bps: int
) -> Embed:
    """Builds the public central-bank loan request embed."""
    embed = Embed(
        title="🏛️ 央行借款申請已建立",
        description=f"### {borrower.mention}\n{currency_text(amount=amount, compact=True)}",
        color=CENTRAL_BANK_COLOR,
    )
    embed.set_author(name=borrower.display_name, icon_url=borrower.avatar_url)
    embed.add_field(
        name="條款",
        value=_loan_terms_text(amount=amount, monthly_rate_bps=monthly_rate_bps),
        inline=False,
    )
    embed.set_footer(text=_central_bank_request_footer())
    return embed


def build_credit_repay_embed(
    *, actor_name: str, actor_avatar_url: str, lender_display_name: str, result: LoanPaymentResult
) -> Embed:
    """Builds the personal credit repayment result embed."""
    embed = Embed(
        title="🧾 信貸還款完成",
        description=(
            f"### {currency_text(amount=-result.paid_amount, signed=True, compact=True)} 扣款"
        ),
        color=REPAY_COLOR,
    )
    embed.set_author(name=actor_name, icon_url=actor_avatar_url)
    _set_optional_thumbnail(embed=embed, avatar_url=actor_avatar_url)
    embed.add_field(
        name=f"還給 {lender_display_name}",
        value=payment_summary_text(
            paid_amount=result.paid_amount,
            interest_paid=result.interest_paid,
            principal_paid=result.principal_paid,
            remaining_principal=result.remaining_principal,
            remaining_interest=result.remaining_interest,
            borrower_balance=result.borrower_balance,
        ),
        inline=False,
    )
    return embed


def build_credit_call_embed(
    *, actor_name: str, actor_avatar_url: str, borrower_mention: str, result: LoanPaymentResult
) -> Embed:
    """Builds the personal credit forced-collection result embed."""
    embed = Embed(
        title="📣 信貸催收完成",
        description=(
            f"### 從 {borrower_mention} 回收 "
            f"{currency_text(amount=result.paid_amount, compact=True)}"
        ),
        color=REPAY_COLOR,
    )
    embed.set_author(name=actor_name, icon_url=actor_avatar_url)
    embed.add_field(
        name="回收明細",
        value=payment_summary_text(
            paid_amount=result.paid_amount,
            interest_paid=result.interest_paid,
            principal_paid=result.principal_paid,
            remaining_principal=result.remaining_principal,
            remaining_interest=result.remaining_interest,
            borrower_balance=result.borrower_balance,
        ),
        inline=False,
    )
    return embed


def build_credit_status_embed(*, contracts: list[LoanContractView], viewer_id: int) -> Embed:
    """Builds the caller's active personal credit contracts embed."""
    lines = [
        (
            f"{'欠 ' + contract.lender_name if contract.borrower_id == viewer_id else contract.borrower_name + ' 欠你'} "
            f"本金 {amount_code(amount=contract.principal_remaining, compact=True)} · "
            f"利息 {amount_code(amount=contract.interest_due, compact=True)} · "
            f"{rate_text(monthly_rate_bps=contract.monthly_rate_bps)}"
        )
        for contract in contracts[:10]
    ]
    return Embed(title="信貸狀態", description="\n".join(lines), color=BORROW_COLOR)


def build_central_bank_repay_embed(
    *, actor_name: str, actor_avatar_url: str, user_mention: str, result: LoanPaymentResult
) -> Embed:
    """Builds the central-bank repayment result embed."""
    embed = Embed(
        title="🏛️ 央行還款完成",
        description=(
            f"### {user_mention}\n"
            + payment_summary_text(
                paid_amount=result.paid_amount,
                interest_paid=result.interest_paid,
                principal_paid=result.principal_paid,
                remaining_principal=result.remaining_principal,
                remaining_interest=result.remaining_interest,
                borrower_balance=result.borrower_balance,
            )
        ),
        color=CENTRAL_BANK_COLOR,
    )
    embed.set_author(name=actor_name, icon_url=actor_avatar_url)
    _set_optional_thumbnail(embed=embed, avatar_url=actor_avatar_url)
    return embed


def build_central_bank_call_embed(
    *,
    actor_name: str,
    actor_avatar_url: str,
    borrower_mention: str,
    borrower_avatar_url: str,
    result: LoanPaymentResult,
) -> Embed:
    """Builds the central-bank forced-collection result embed."""
    embed = Embed(
        title="🏛️ 央行催收完成",
        description=(
            f"### 從 {borrower_mention} 回收\n"
            + payment_summary_text(
                paid_amount=result.paid_amount,
                interest_paid=result.interest_paid,
                principal_paid=result.principal_paid,
                remaining_principal=result.remaining_principal,
                remaining_interest=result.remaining_interest,
                borrower_balance=result.borrower_balance,
            )
        ),
        color=CENTRAL_BANK_COLOR,
    )
    embed.set_author(name=actor_name, icon_url=actor_avatar_url)
    _set_optional_thumbnail(embed=embed, avatar_url=borrower_avatar_url)
    return embed


def build_central_bank_status_embed(*, status: CentralBankStatus) -> Embed:
    """Builds the central bank lending-capacity embed."""
    embed = Embed(
        title="🏛️ 中央銀行狀態",
        description=f"## 可放貸 {bold_currency(amount=status.available_credit, compact=True)}",
        color=CENTRAL_BANK_COLOR,
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
    return embed


def build_checkin_embed(*, actor_name: str, avatar_url: str, result: CheckinResult) -> Embed:
    """Builds the daily check-in result embed."""
    vip_badge = " · 👑 VIP 2x" if result.is_vip else ""
    embed = Embed(
        title="📅 每日簽到",
        description=f"## {currency_text(amount=result.amount, signed=True, compact=True)} 入帳",
        color=CHECKIN_COLOR,
    )
    embed.set_author(name=f"{actor_name} 的簽到", icon_url=avatar_url)
    _set_optional_thumbnail(embed=embed, avatar_url=avatar_url)
    embed.add_field(name="連續簽到", value=f"第 {result.streak} / 7 天", inline=True)
    embed.add_field(
        name="目前餘額", value=amount_code(amount=result.new_balance, compact=True), inline=True
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
    return embed


def build_vip_already_embed(*, actor_name: str, avatar_url: str) -> Embed:
    """Builds the embed shown when a member already owns VIP."""
    embed = Embed(
        title="已經是 VIP", description="### 你已經擁有永久 VIP 了, 不用再買一次", color=VIP_COLOR
    )
    embed.set_author(name=actor_name, icon_url=avatar_url)
    embed.add_field(name="👑 VIP加成", value=_vip_perk_lines(), inline=False)
    return embed


def build_vip_insufficient_embed(*, actor_name: str, avatar_url: str, balance_now: int) -> Embed:
    """Builds the VIP purchase failure embed for an insufficient balance."""
    embed = Embed(
        title="VIP 購買失敗",
        description=(
            f"### 餘額不足\n"
            f"目前 {bold_currency(amount=balance_now, compact=True)}\n"
            f"需要 {bold_currency(amount=VIP_PURCHASE_COST, compact=True)}"
        ),
        color=ERROR_COLOR,
    )
    embed.set_author(name=actor_name, icon_url=avatar_url)
    embed.add_field(name="👑 VIP權益", value=_vip_perk_lines(), inline=False)
    return embed


def build_vip_success_embed(
    *, actor_name: str, avatar_url: str, result: VipPurchaseResult
) -> Embed:
    """Builds the VIP purchase success embed."""
    embed = Embed(
        title="👑 升級 VIP 成功",
        description=(
            f"### {currency_text(amount=-result.cost, signed=True, compact=True)} 扣款\n"
            "簽到與 Blackjack 贏局加成已生效"
        ),
        color=VIP_COLOR,
    )
    embed.set_author(name=actor_name, icon_url=avatar_url)
    _set_optional_thumbnail(embed=embed, avatar_url=avatar_url)
    embed.add_field(name="👑 VIP加成", value=_vip_perk_lines(), inline=False)
    embed.add_field(
        name="目前餘額", value=amount_code(amount=result.new_balance, compact=True), inline=False
    )
    return embed


def build_credit_approved_embed(
    *, result: LoanProposalAcceptResult, approver_mention: str, lender_avatar_url: str
) -> Embed:
    """Builds the personal credit approval embed used by the decision view."""
    embed = Embed(
        title="✅ 信貸已批准",
        description=(
            f"### {currency_text(amount=result.contract.principal_remaining, compact=True)} 已入帳"
        ),
        color=BORROW_COLOR,
    )
    embed.add_field(name="批准者", value=approver_mention, inline=True)
    embed.add_field(
        name="利率",
        value=f"`{rate_text(monthly_rate_bps=result.contract.monthly_rate_bps)}`",
        inline=True,
    )
    embed.add_field(
        name="貸方餘額",
        value=amount_code(amount=result.lender_balance or 0, compact=True),
        inline=True,
    )
    _set_optional_thumbnail(embed=embed, avatar_url=lender_avatar_url)
    return embed


def build_central_bank_approved_embed(
    *, result: LoanProposalAcceptResult, approver_mention: str
) -> Embed:
    """Builds the central-bank approval embed used by the decision view."""
    embed = Embed(
        title="🏛️ 央行借款已批准",
        description=(
            f"### {currency_text(amount=result.contract.principal_remaining, compact=True)} 已入帳"
        ),
        color=CENTRAL_BANK_COLOR,
    )
    embed.add_field(name="批准者", value=approver_mention, inline=True)
    embed.add_field(
        name="央行剩餘額度",
        value=amount_code(amount=result.central_bank_available_credit or 0, compact=True),
        inline=True,
    )
    return embed
