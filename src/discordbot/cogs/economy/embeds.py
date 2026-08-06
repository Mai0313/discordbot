"""Every embed the economy surface shows, kept where both of its Discord entry points reach it.

`cog.py` owns the slash commands and the ledger calls, `views.py` owns the loan-decision buttons,
and both end a flow by showing an embed. Several of those embeds are reached from either side: a
personal credit request is created by a command and approved by a button, and the timed-out /
rejected / cancelled closures are all `build_simple_embed` over the same two loan colors. Holding
the whole vocabulary in one module is what keeps one figure from growing two spellings.

Nothing here sends, defers or edits, and nothing here reads the database. A builder takes an
already-settled result model from `typings/economy.py` and returns a `nextcord.Embed`; whether that
embed goes out publicly with scheduled cleanup or ephemerally stays the caller's decision, which is
where the cog's public-vs-private boundary actually lives.

The one thing it does reach back into `services/economy/database.py` for is the ledger's own pure
formulas (`checkin_reward`, `apply_vip_blackjack_bonus`, `monthly_rate_bps_to_percent`), so the VIP
perks and the check-in bonus are priced by the code that will settle them instead of by a second
copy of the multiplier. Amount text is not this module's either: every figure goes out through
`services/economy/presentation.py`'s `currency_text` / `bold_currency` / `amount_code`, which is
what makes a balance here read the same as a Blackjack seat or a fishing panel. The two
leaderboards carry no rows at all — `boards.py` renders those as a PNG and the embed only points at
it through `attachment://`, so the caller has to attach the file under the same name.

The module constants are this cog's palette. `BALANCE_COLOR` / `LEADERBOARD_COLOR` / `ERROR_COLOR`
re-alias the shared Discord hexes under economy words and the rest are per-surface accents that
stay local, both following the habit `typings/colors.py` describes.
"""

from nextcord import Embed
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.stock import StockPortfolioView, StockPortfolioHolding
from discordbot.typings.colors import DISCORD_RED, DISCORD_GREEN, DISCORD_YELLOW, TRANSFER_COLOR
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
from discordbot.cogs.economy.boards import (
    LOSS_LEADERBOARD_BOARD_FILENAME,
    BALANCE_LEADERBOARD_BOARD_FILENAME,
)
from discordbot.services.stock.market import format_price
from discordbot.services.economy.database import (
    checkin_reward,
    apply_vip_blackjack_bonus,
    monthly_rate_bps_to_percent,
)
from discordbot.services.economy.presentation import (
    CURRENCY_NAME,
    amount_code,
    bold_currency,
    currency_text,
)

BALANCE_COLOR = DISCORD_GREEN
LEADERBOARD_COLOR = DISCORD_YELLOW
LOSS_LEADERBOARD_COLOR = 0xE67E22
ADMIN_COLOR = 0x3498DB
CASINO_COLOR = 0xEB459E
BORROW_COLOR = 0xF1C40F
REPAY_COLOR = 0x2ECC71
CENTRAL_BANK_COLOR = 0x1ABC9C
CHECKIN_COLOR = 0x9B59B6
VIP_COLOR = 0xF1C40F
ERROR_COLOR = DISCORD_RED
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
    """Display identity for one side of a loan request embed.

    Only the requesting side is named in words: the two optional fields are empty on the lender a
    personal credit request is aimed at, which is built from a mention alone. Which slot the avatar
    fills depends on the side too, the borrower's leading the author line and the lender's the
    thumbnail.
    """

    model_config = ConfigDict(frozen=True)

    mention: str = Field(
        ..., description="Discord mention string (<@user_id>) shown in the embed."
    )
    display_name: str = Field(default="", description="Display name shown next to the mention.")
    avatar_url: str = Field(
        default="", description="Avatar URL for the embed thumbnail or author icon."
    )


def _vip_perk_lines(checkin_streak: int = 1) -> str:
    """Formats what VIP is worth, as a before/after pair for check-in and for a Blackjack win.

    Both sides of both pairs come from the ledger's own formulas rather than from a multiplier
    written down here, so retuning either one moves what the badge advertises with it. The
    Blackjack line prices a fixed sample win, not anything the reader earned.

    Args:
        checkin_streak (int): Streak day to price the check-in line at. Day 1 is labelled as the
            base reward instead of by its day number.

    Returns:
        Two lines of Markdown for a VIP field value.
    """
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
    """Sets an embed thumbnail in place, doing nothing when no avatar URL is known.

    The empty string is an ordinary outcome rather than an error: the ledger's stored avatar is a
    last-seen cache that is never backfilled, so an account nobody has been seen under has none.

    Args:
        embed (Embed): Embed mutated in place.
        avatar_url (str): Avatar URL, or an empty string when none is known.
    """
    if avatar_url:
        embed.set_thumbnail(url=avatar_url)


def _stock_position_lines(stock_portfolio: StockPortfolioView) -> str:
    """Formats the stock field of the private balance embed, one line per holding.

    Only the first `_STOCK_POSITION_LINE_LIMIT` holdings in the order given are drawn; the rest
    collapse into a count line, so a wide portfolio cannot push the field past its own limit.

    Args:
        stock_portfolio (StockPortfolioView): The member's whole market exposure.

    Returns:
        The field value, or the no-positions placeholder when there are no holdings.
    """
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
    """Formats one holding as a compact balance line, both legs when both are open.

    The short leg is shown as its equity — collateral plus entry value less what covering costs
    right now — rather than as a negative market value, which is how `StockPortfolioView` totals
    it, so the line and the field above it cannot disagree.

    Args:
        holding (StockPortfolioHolding): One non-zero position valued at the current quote.

    Returns:
        One line of Markdown. The `無部位` branch is defensive: a holding that reached the
        portfolio always carries at least one leg.
    """
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
    """Keeps a long company name from filling the stock field.

    Args:
        name (str): Display name of the virtual company.

    Returns:
        The name, or its first `_STOCK_POSITION_NAME_LIMIT` characters with an ellipsis.
    """
    if len(name) <= _STOCK_POSITION_NAME_LIMIT:
        return name
    return f"{name[:_STOCK_POSITION_NAME_LIMIT]}..."


def _debt_summary_text(*, principal: int, interest: int) -> str:
    """Formats the debt field of the private balance embed.

    Args:
        principal (int): Total outstanding loan principal.
        interest (int): Total accrued loan interest due.

    Returns:
        The two-line principal and interest block, or `無未還債務` when neither is owed.
    """
    if principal <= 0 and interest <= 0:
        return "無未還債務"
    return (
        f"本金 {amount_code(amount=principal, compact=True)}\n"
        f"利息 {amount_code(amount=interest, compact=True)}"
    )


def _vip_status_text(is_vip: bool) -> str:
    """Formats the membership field of the private balance embed.

    A VIP account gets the perk lines priced at the first streak day, so the field says what the
    badge is worth rather than only that it is held.

    Args:
        is_vip (bool): VIP status of the account.

    Returns:
        The field value.
    """
    if not is_vip:
        return "一般會員"
    return f"👑 VIP\n{_vip_perk_lines()}"


def rate_text(monthly_rate_bps: int) -> str:
    """Formats a stored monthly loan rate as the percent a borrower reads.

    Args:
        monthly_rate_bps (int): Monthly simple-interest rate in basis points.

    Returns:
        Text such as `每月 1.5%`, a whole percent printed without a decimal part.
    """
    return f"每月 {monthly_rate_bps_to_percent(monthly_rate_bps=monthly_rate_bps):g}%"


def _loan_terms_text(amount: int, monthly_rate_bps: int) -> str:
    """Formats the terms block a request shows before anyone approves it.

    The rules spelled out under the figures are the ledger's real repayment behavior — simple
    interest, accrued pro rata over elapsed days, and a payment clearing interest before
    principal — so they move only when settlement does.

    Args:
        amount (int): Principal being requested.
        monthly_rate_bps (int): Monthly simple-interest rate in basis points.

    Returns:
        The value of the 條款 field.
    """
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
    """Formats the breakdown every repayment and collection embed shows.

    One wording for four commands, so a personal repayment and a central-bank collection account
    for the same money in the same order.

    Args:
        paid_amount (int): Total amount this payment moved.
        interest_paid (int): Portion of it applied to interest.
        principal_paid (int): Portion of it applied to principal.
        remaining_principal (int): Principal still owed afterwards.
        remaining_interest (int): Interest still due afterwards.
        borrower_balance (int): Borrower balance after the payment.

    Returns:
        Six lines of Markdown, used as a field value or spliced into a description.
    """
    return (
        f"本次扣款 {amount_code(amount=paid_amount, compact=True)}\n"
        f"償還利息 {amount_code(amount=interest_paid, compact=True)}\n"
        f"償還本金 {amount_code(amount=principal_paid, compact=True)}\n"
        f"剩餘本金 {amount_code(amount=remaining_principal, compact=True)}\n"
        f"剩餘利息 {amount_code(amount=remaining_interest, compact=True)}\n"
        f"借方餘額 {amount_code(amount=borrower_balance, compact=True)}"
    )


def _credit_request_footer() -> str:
    """Formats the button hint under a personal credit request.

    Returns:
        Footer text naming who may act and the deadline after which the request auto-rejects,
        which is the deciding view's own timeout.
    """
    return (
        f"貸方可用下方按鈕批准或拒絕，發起者可取消，{LOAN_PROPOSAL_TIMEOUT_SECONDS} 秒後自動拒絕"
    )


def _central_bank_request_footer() -> str:
    """Formats the button hint under a central-bank loan request.

    Returns:
        Footer text naming who may act and the deadline after which the request auto-rejects,
        which is the deciding view's own timeout.
    """
    return f"央行成員可用下方按鈕批准或拒絕，發起者可取消，{LOAN_PROPOSAL_TIMEOUT_SECONDS} 秒後自動拒絕"


def build_error_embed(
    *,
    title: str,
    description: str,
    author_name: str | None = None,
    author_icon_url: str | None = None,
    thumbnail_url: str | None = None,
) -> Embed:
    """Builds the red failure embed every economy command and decision button shares.

    The `description` is used verbatim, including any leading Markdown heading, so each caller
    states its own reason for refusing.

    Args:
        title (str): Embed title, naming what failed.
        description (str): Failure text, used verbatim.
        author_name (str | None): Name for the author line, or None to leave it unset.
        author_icon_url (str | None): Icon for the author line; read only when `author_name` is
            given.
        thumbnail_url (str | None): Avatar URL for the thumbnail, or None to leave it unset.

    Returns:
        The error embed, unsent.
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
    """Builds the one-section embed the flows with nothing to itemize use.

    This is what an empty leaderboard, an empty credit list and every terminal loan-request state
    (timed out, rejected, cancelled) are built from, so `color` is what tells them apart.

    Args:
        title (str): Embed title.
        description (str): Body text, used verbatim.
        color (int): Embed color, usually one of this module's constants.
        author_name (str | None): Name for the author line, or None to leave it unset.
        author_icon_url (str | None): Icon for the author line; read only when `author_name` is
            given.
        thumbnail_url (str | None): Avatar URL for the thumbnail, or None to leave it unset.
        footer_text (str | None): Footer text, or None to leave it unset.

    Returns:
        The embed, unsent.
    """
    embed = Embed(title=title, description=description, color=color)
    if author_name is not None:
        embed.set_author(name=author_name, icon_url=author_icon_url)
    if thumbnail_url is not None:
        _set_optional_thumbnail(embed=embed, avatar_url=thumbnail_url)
    if footer_text is not None:
        embed.set_footer(text=footer_text)
    return embed


def build_invalid_amount_embed(*, title: str) -> Embed:
    """Builds the shared refusal for money text the amount parser rejected.

    Every money option is a string so an amount can exceed Discord's integer ceiling, which makes
    this the one wording a malformed figure gets, and it is always shown before anything is moved.

    Args:
        title (str): Embed title naming the command that refused, such as `轉帳失敗`.

    Returns:
        The validation embed.
    """
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
    """Builds the public receipt for an admin balance adjustment.

    What was asked for and what actually applied are shown side by side because they part company
    whenever a collection ran into the target's balance, and `is_collect_clamped` adds the footer
    saying that is what happened rather than leaving the reader to compare two figures.

    Args:
        title (str): Embed title, naming the adjustment.
        member_mention (str): Mention of the adjusted member, shown as the heading.
        actor_name (str): Admin who ran the command, shown as the author.
        actor_avatar_url (str): Admin avatar, shown as the author icon.
        member_avatar_url (str): Adjusted member's avatar, shown as the thumbnail.
        requested_delta (int): Signed delta the admin asked for.
        result (BalanceAdjustmentResult): Settled adjustment, carrying the delta that applied.
        is_collect_clamped (bool): Whether the collection stopped at the member's balance.

    Returns:
        The adjustment embed.
    """
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
    """Builds the private financial overview across the wallet and the market.

    The headline is the only figure this module derives rather than displays: the portfolio's own
    net worth (cash less debt) plus the stock portfolio's equity, since the two ledgers are read
    separately and neither knows about the other.

    Args:
        display_name (str): Member the overview is about, shown in the heading and author line.
        avatar_url (str): Member avatar, used for both the author icon and the thumbnail.
        portfolio (PortfolioView): Wallet and debt totals.
        stock_portfolio (StockPortfolioView): Market exposure valued at the current quotes.
        is_vip (bool): VIP status, which adds the perk lines and the footer badge.
        age_days (int): Account age in days, shown in the footer.

    Returns:
        The overview embed.
    """
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
    """Builds the frame around the rendered balance leaderboard.

    Only the champion reaches the embed; the ranking itself is the PNG `boards.py` draws, so the
    caller has to attach it under `BALANCE_LEADERBOARD_BOARD_FILENAME` or the `attachment://`
    reference resolves to nothing.

    Args:
        champion (LeaderboardEntry): Top-ranked account, whose avatar leads the author line and
            fills the thumbnail.

    Returns:
        The leaderboard embed.
    """
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
    """Builds the frame around the rendered daily loss leaderboard.

    Same split as the balance board: the rows are the PNG `boards.py` draws and the caller has to
    attach it under `LOSS_LEADERBOARD_BOARD_FILENAME`. The footer states the two rules a reader
    would otherwise guess wrong, that the ranking is gross loss with wins not netted off and that
    it resets on the Taipei day boundary.

    Args:
        champion (LossLeaderboardEntry): Biggest current-day loser, whose avatar leads the author
            line and fills the thumbnail.

    Returns:
        The loss leaderboard embed.
    """
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
    """Builds the public receipt for a completed transfer.

    The net-received line appears only when the transfer tax actually burned something, so a
    zero-tax transfer reads as one figure instead of the same figure twice.

    Args:
        amount (int): Gross amount the sender moved, before the tax burn.
        sender (TransferParticipant): Sender's mention and display name.
        sender_avatar_url (str): Sender avatar, shown as the author icon.
        receiver (TransferParticipant): Receiver's mention and display name.
        receiver_avatar_url (str): Receiver avatar, shown as the thumbnail.
        result (TransferResult): Settled transfer, carrying both balances and the burn.

    Returns:
        The transfer embed.
    """
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
    """Builds the transfer refusal for a sender who cannot cover the amount.

    Both figures are shown because the shortfall is the point; nothing moved.

    Args:
        sender_name (str): Sender display name, shown as the author.
        sender_avatar_url (str): Sender avatar, shown as the author icon.
        balance_now (int): Sender balance at the moment of the refusal.
        amount (int): Amount the sender tried to move.

    Returns:
        The refusal embed.
    """
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
    """Builds the casino system's cumulative profit and loss.

    Read from the house's side: a positive ledger is the casino ahead of the table and takes the
    balance color, a negative one the error color. This is the `casino_ledger` row and never the
    bot player's wallet, which `build_pocat_embed` shows instead. The headline sign is
    Markdown-escaped so it prints as written.

    Args:
        snapshot (CasinoLedgerSnapshot): Read-only casino ledger totals.

    Returns:
        The casino embed.
    """
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
    """Builds the bot player's own wallet embed.

    The bot sits at the table as an ordinary player, so these are its `user_wallet` figures and
    say nothing about how the house is doing; `build_casino_embed` is the other question.

    Args:
        name (str): Bot's display name, shown as the author.
        avatar_url (str): Bot avatar, used for both the author icon and the thumbnail.
        balance (int): Current wallet balance, which drives the headline and the color.
        total_earned (int): Lifetime gross amount won.
        total_spent (int): Lifetime gross amount lost.

    Returns:
        The wallet embed.
    """
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
    """Builds the public personal credit request the lender then decides with buttons.

    Nothing has moved yet: a personal loan debits the lender at acceptance, so the terms field
    describes the contract acceptance would create and the footer names the auto-reject deadline.

    Args:
        borrower (LoanParty): Requesting member, shown as the author.
        lender (LoanParty): Requested lender, whose avatar becomes the thumbnail.
        amount (int): Principal being requested.
        monthly_rate_bps (int): Monthly simple-interest rate in basis points.

    Returns:
        The request embed, to be sent with its decision view.
    """
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
    """Builds the public central-bank loan request any banker can then decide.

    The same terms block as a personal request, minus a counterparty: approval mints the principal
    rather than debiting a lender, so there is no second identity to show.

    Args:
        borrower (LoanParty): Requesting member, shown as the author.
        amount (int): Principal being requested.
        monthly_rate_bps (int): Monthly simple-interest rate in basis points.

    Returns:
        The request embed, to be sent with its decision view.
    """
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
    """Builds the receipt for a borrower repaying a personal loan.

    The lender is named in the field heading rather than mentioned, since the borrower is the one
    who acted and the payment has already landed.

    Args:
        actor_name (str): Repaying member, shown as the author.
        actor_avatar_url (str): Their avatar, used for both the author icon and the thumbnail.
        lender_display_name (str): Lender named in the field heading.
        result (LoanPaymentResult): Settled payment, split into interest and principal.

    Returns:
        The repayment embed.
    """
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
    """Builds the receipt for a lender collecting on a personal loan.

    The borrower is mentioned rather than merely named: a collection is taken from them without
    their acting, so the public record says who paid.

    Args:
        actor_name (str): Collecting lender, shown as the author.
        actor_avatar_url (str): Lender avatar, shown as the author icon.
        borrower_mention (str): Mention of the borrower the amount came from.
        result (LoanPaymentResult): Settled collection, split into interest and principal.

    Returns:
        The collection embed.
    """
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
    """Builds the caller's list of active personal credit contracts.

    Every line is written from the viewer's side, so one contract reads `欠 <lender>` to its
    borrower and `<borrower> 欠你` to its lender. Only the first ten contracts in the order given
    are drawn and the rest are dropped without a notice. An empty list would build an embed with an
    empty description, so the caller answers that case itself.

    Args:
        contracts (list[LoanContractView]): Active personal contracts the viewer is a party to.
        viewer_id (int): Discord user ID whose side each line is written from.

    Returns:
        The contract list embed.
    """
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
    """Builds the receipt for a member repaying the central bank.

    The breakdown sits in the description rather than a field, since there is no counterparty to
    head one with: repaid principal is burned instead of credited to a lender.

    Args:
        actor_name (str): Repaying member, shown as the author.
        actor_avatar_url (str): Their avatar, used for both the author icon and the thumbnail.
        user_mention (str): Mention of the member the payment came from.
        result (LoanPaymentResult): Settled payment, split into interest and principal.

    Returns:
        The repayment embed.
    """
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
    """Builds the receipt for a banker collecting a central-bank loan.

    The collected member's avatar is the thumbnail rather than the banker's, so the public record
    shows who was collected from and not only who ran the command.

    Args:
        actor_name (str): Banker who collected, shown as the author.
        actor_avatar_url (str): Banker avatar, shown as the author icon.
        borrower_mention (str): Mention of the borrower the amount came from.
        borrower_avatar_url (str): Borrower avatar, shown as the thumbnail.
        result (LoanPaymentResult): Settled collection, split into interest and principal.

    Returns:
        The collection embed.
    """
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
    """Builds the central bank's lending-capacity embed.

    The headline is the capacity a request is checked against; the pool field shows the two
    aggregates it is computed from, since a borrower refused for capacity would otherwise have
    nothing to look at.

    Args:
        status (CentralBankStatus): Aggregated lending capacity.

    Returns:
        The status embed.
    """
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
    """Builds the daily check-in receipt.

    A VIP account gets one extra field, which re-prices the same streak day without the badge so
    the bonus reads as a before/after rather than as a multiplier the reader has to apply. The
    streak is shown against the seven-day cycle it wraps on.

    Args:
        actor_name (str): Member who checked in, shown as the author.
        avatar_url (str): Their avatar, used for both the author icon and the thumbnail.
        result (CheckinResult): Settled check-in, carrying the payout, streak and VIP status.

    Returns:
        The check-in embed.
    """
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
    """Builds the answer to buying VIP twice.

    VIP is permanent, so this is a refusal with nothing to fix. It still lists the perks, which is
    the only useful thing left to say to someone asking about a badge they already hold.

    Args:
        actor_name (str): Member who tried to buy, shown as the author.
        avatar_url (str): Their avatar, shown as the author icon.

    Returns:
        The refusal embed.
    """
    embed = Embed(
        title="已經是 VIP", description="### 你已經擁有永久 VIP 了, 不用再買一次", color=VIP_COLOR
    )
    embed.set_author(name=actor_name, icon_url=avatar_url)
    embed.add_field(name="👑 VIP加成", value=_vip_perk_lines(), inline=False)
    return embed


def build_vip_insufficient_embed(*, actor_name: str, avatar_url: str, balance_now: int) -> Embed:
    """Builds the VIP purchase refusal for a member who cannot cover the price.

    The price comes from `VIP_PURCHASE_COST` itself, so what is quoted here is what the purchase
    will charge. The perks ride along, since the refusal is also the sales pitch.

    Args:
        actor_name (str): Member who tried to buy, shown as the author.
        avatar_url (str): Their avatar, shown as the author icon.
        balance_now (int): Balance at the moment of the refusal.

    Returns:
        The refusal embed.
    """
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
    """Builds the VIP purchase receipt.

    Args:
        actor_name (str): Member who bought VIP, shown as the author.
        avatar_url (str): Their avatar, used for both the author icon and the thumbnail.
        result (VipPurchaseResult): Settled purchase; a value at all means the debit landed and
            the flag flipped.

    Returns:
        The purchase embed.
    """
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
    """Builds what replaces a personal credit request once its lender approves it.

    The view edits the request message with this, so it stands alone as the record of the loan
    rather than adding to what the request said. `lender_balance` is optional only because the
    same result model also carries central-bank acceptances, which have no lender; a personal
    acceptance always fills it, so the fallback is unreachable in practice.

    Args:
        result (LoanProposalAcceptResult): Accepted proposal and the contract it created.
        approver_mention (str): Mention of the lender who approved.
        lender_avatar_url (str): Lender avatar, shown as the thumbnail.

    Returns:
        The approval embed.
    """
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
    """Builds what replaces a central-bank request once a banker approves it.

    Where the personal twin reports the lender's balance, this reports what the bank has left to
    lend, since approval minted the principal instead of moving it from anyone. The field is
    optional on the shared result model for the same reason its counterpart is, and a central-bank
    acceptance always fills it.

    Args:
        result (LoanProposalAcceptResult): Accepted proposal and the contract it created.
        approver_mention (str): Mention of the banker who approved.

    Returns:
        The approval embed.
    """
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
