"""Shared economy result types, enums, tuning constants, and rate converters."""

from enum import StrEnum
from typing import Final
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict

BASE_MESSAGE_REWARD_AMOUNT: Final[int] = 10
VIP_PURCHASE_COST: Final[int] = 50_000
LOAN_PROPOSAL_TIMEOUT_SECONDS: Final[int] = 180
DEFAULT_LOAN_MONTHLY_RATE_BPS: Final[int] = 300
MIN_LOAN_MONTHLY_RATE_BPS: Final[int] = 0
MAX_LOAN_MONTHLY_RATE_BPS: Final[int] = 10_000
# Minimum interest a borrower owes on a contract regardless of repayment timing.
MIN_INTEREST_DAYS: Final[int] = 30

# Anti-inflation levers; re-measure before changing them.
# Absolute ceiling on any single casino wager. Invisible to ordinary players; it
# bounds all-in doubling to linear growth once a balance gets large.
MAX_SINGLE_BET: Final[int] = 1_000_000
# Per-user cooldown between message rewards, so the flat per-message grant cannot
# be farmed by spamming.
MESSAGE_REWARD_COOLDOWN_SECONDS: Final[float] = 60.0
# Permanent money sink: the burn on every transfer, in basis points.
TRANSFER_TAX_BPS: Final[int] = 500
# VIP perk: 1.2x payout on a winning round.
_VIP_WIN_MULTIPLIER_NUM: Final[int] = 6
_VIP_WIN_MULTIPLIER_DEN: Final[int] = 5

# Central-bank levers; re-measure before changing them.
# How many times their own free equity a borrower may owe the central bank.
CENTRAL_BANK_CREDIT_MULTIPLIER: Final[int] = 2
# The central bank's starting capital. The interest it keeps is added on top, and
# the two together are what a server whose own members hold almost nothing can
# still borrow against. A constant rather than a seeded ledger row, so the row only
# ever holds earnings and a database created before this existed needs no fix-up.
CENTRAL_BANK_BASE_CAPACITY: Final[int] = 5_000_000


def monthly_rate_percent_to_bps(monthly_rate_percent: float) -> int:
    """Converts a user-facing monthly percent into basis points."""
    return max(
        MIN_LOAN_MONTHLY_RATE_BPS,
        min(MAX_LOAN_MONTHLY_RATE_BPS, round(monthly_rate_percent * 100)),
    )


def monthly_rate_bps_to_percent(monthly_rate_bps: int) -> float:
    """Converts stored monthly basis points into a display percent."""
    return monthly_rate_bps / 100


def apply_vip_blackjack_bonus(delta: int, is_vip: bool) -> int:
    """Applies the VIP 1.2x payout multiplier on a winning player delta.

    The bonus only fires on positive deltas (wins). Pushes and losses pass
    through unchanged so VIP never softens a loss.

    Args:
        delta: Pre-bonus player delta for the round.
        is_vip: VIP status of the account at settlement time.

    Returns:
        Post-bonus player delta.
    """
    if not is_vip or delta <= 0:
        return delta
    return delta * _VIP_WIN_MULTIPLIER_NUM // _VIP_WIN_MULTIPLIER_DEN


def central_bank_credit_ceiling(balance: int, total_debt: int) -> int:
    """Returns how much more central-bank credit one borrower may still draw.

    Mirrors the lending pool's own double subtraction. A central-bank loan mints
    into the borrower's balance, so the first term takes that mint back out and
    the second charges the debt as capacity already used. Borrowing `x` therefore
    lowers the result by exactly `x`, which is what keeps total minting tied to
    equity somebody actually earned.

    `total_debt` counts every lender, not just the central bank. A personal loan
    carries no transfer tax and may be written at 0%, so a ceiling blind to it is
    reset by lending a minted balance on to a second account, and the pair can
    then take turns borrowing against each other without limit.

    Args:
        balance: Current wallet balance.
        total_debt: Outstanding principal plus accrued interest the borrower owes
            across every active contract, whoever lent it.

    Returns:
        The remaining ceiling, never negative.
    """
    return max((balance - total_debt) * CENTRAL_BANK_CREDIT_MULTIPLIER - total_debt, 0)


class LoanLenderType(StrEnum):
    """Kinds of lender backing a long-term loan contract."""

    USER = "user"
    CENTRAL_BANK = "central_bank"


class LoanProposalKind(StrEnum):
    """Pending loan proposal flow types."""

    PERSONAL_REQUEST = "personal_request"
    CENTRAL_BANK_REQUEST = "central_bank_request"


class LoanProposalStatus(StrEnum):
    """Lifecycle states for a pending loan proposal."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CANCELED = "canceled"


class LoanContractStatus(StrEnum):
    """Lifecycle states for a long-term loan contract."""

    ACTIVE = "active"
    CLOSED = "closed"


class AccountSnapshot(BaseModel):
    """Read-only account totals for maintenance and house-ledger views."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Last-seen Discord account name.")
    balance: int = Field(..., description="Current point balance.")
    total_earned: int = Field(..., description="Lifetime gross earned amount.")
    total_spent: int = Field(..., description="Lifetime gross spent amount.")


class LeaderboardEntry(BaseModel):
    """One account row in the balance leaderboard."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(..., description="Discord user ID of the leaderboard account.")
    name: str = Field(..., description="Last-seen Discord account name.")
    balance: int = Field(..., description="Current point balance used for ranking.")
    avatar_url: str = Field(
        default="", description="Last-seen Discord avatar URL for the account."
    )


class LossLeaderboardEntry(BaseModel):
    """One account row in the daily casino loss leaderboard."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(..., description="Discord user ID of the leaderboard account.")
    name: str = Field(..., description="Last-seen Discord account name.")
    loss_amount: int = Field(..., description="Gross current-day casino loss used for ranking.")
    avatar_url: str = Field(
        default="", description="Last-seen Discord avatar URL for the account."
    )


class CreditResult(BaseModel):
    """Outcome of an income event."""

    model_config = ConfigDict(frozen=True)

    new_balance: int = Field(..., description="User balance after the credit.")
    credited_amount: int = Field(..., description="Amount that landed in balance.")


class BalanceAdjustmentResult(BaseModel):
    """Outcome of a manual balance adjustment."""

    model_config = ConfigDict(frozen=True)

    new_balance: int = Field(..., description="User balance after the adjustment.")
    applied_delta: int = Field(..., description="Signed balance delta that was actually applied.")


class JackpotSettlementRequest(BaseModel):
    """One player-side settlement against a shared jackpot pool."""

    model_config = ConfigDict(frozen=True)

    player_id: int = Field(..., description="Discord user ID for the player account.")
    player_account_name: str = Field(
        ..., description="Last-seen account name stored on the player row."
    )
    player_delta: int = Field(
        ..., description="Signed change for the player; the pool receives the inverse."
    )
    player_avatar_url: str = Field(
        default="", description="Last-seen Discord avatar URL for the player."
    )
    require_full_debit: bool = Field(
        default=False,
        description="Whether a negative delta must be applied in full, rejecting the whole batch instead of clamping at the player's current balance.",
    )
    expected_jackpot_generation: int | None = Field(
        default=None,
        description="Optional jackpot generation observed by the game view; positive payouts only claim from this generation, so a stale action cannot spend a freshly reseeded pool.",
    )


class JackpotSettlementBatchResult(BaseModel):
    """Outcome of one or more settlements against a shared jackpot pool."""

    model_config = ConfigDict(frozen=True)

    player_balances: dict[int, int] = Field(
        ..., description="Latest post-settlement balance for each touched player."
    )
    applied_player_deltas: dict[int, int] = Field(
        ...,
        description="Signed player deltas that were actually applied; losses may be smaller than requested when the balance clamps at zero.",
    )
    jackpot_balance: int = Field(
        ..., description="Pool balance after the final settlement and any reseed."
    )
    jackpot_generation: int = Field(
        default=0, description="Pool generation after the final settlement and any reseed."
    )
    jackpot_depleted: bool = Field(
        default=False,
        description="True when a seeded pool was drained and automatically replenished during this batch.",
    )
    rejected_player_ids: tuple[int, ...] = Field(
        default=(),
        description="Player IDs whose required full debit could not be applied; no mutation is committed when non-empty.",
    )


class JackpotSnapshot(BaseModel):
    """Read-only snapshot of a shared jackpot pool."""

    model_config = ConfigDict(frozen=True)

    balance: int = Field(..., description="Current jackpot pool balance.")
    generation: int = Field(default=0, description="Current jackpot pool generation counter.")


class JackpotSettlementResult(BaseModel):
    """Outcome of a single player settlement against a shared jackpot pool."""

    model_config = ConfigDict(frozen=True)

    player_balance: int = Field(..., description="Player balance after this settlement.")
    jackpot_balance: int = Field(
        ..., description="Pool balance after this settlement and any reseed."
    )
    jackpot_generation: int = Field(
        default=0, description="Pool generation after this settlement and any reseed."
    )
    applied_player_delta: int = Field(
        ..., description="Signed player delta that was actually applied."
    )
    jackpot_depleted: bool = Field(
        default=False,
        description="True when a seeded pool was drained and replenished during this settlement.",
    )
    rejected: bool = Field(
        default=False,
        description="True when a required full debit could not be applied and no mutation was committed.",
    )


class CasinoLedgerSnapshot(BaseModel):
    """Read-only snapshot of the casino system ledger."""

    model_config = ConfigDict(frozen=True)

    balance: int = Field(..., description="Current casino system ledger balance.")
    total_earned: int = Field(
        ..., description="Lifetime gross amount earned by the casino ledger."
    )
    total_spent: int = Field(
        ..., description="Lifetime gross amount paid out by the casino ledger."
    )
    updated_at: datetime = Field(..., description="Timestamp of the last casino ledger update.")


class CasinoDailyStats(BaseModel):
    """Per-user current-day casino loss/win/net totals.

    All zero when no row exists or the stored counters belong to a previous
    Taipei day.
    """

    model_config = ConfigDict(frozen=True)

    daily_loss: int = Field(..., description="Gross current-day casino loss total.")
    daily_win: int = Field(..., description="Gross current-day casino win total.")
    daily_net: int = Field(..., description="Net current-day casino result (win minus loss).")


class RoundSettlementResult(BaseModel):
    """Outcome of an atomic player + casino ledger settlement."""

    model_config = ConfigDict(frozen=True)

    player_balance: int = Field(..., description="Player balance after the round settlement.")
    casino_balance: int = Field(
        ..., description="Casino system ledger balance after the round settlement."
    )


class TransferResult(BaseModel):
    """A successful point transfer."""

    model_config = ConfigDict(frozen=True)

    sender_balance: int = Field(..., description="Sender balance after the debit.")
    receiver_balance: int = Field(..., description="Receiver balance after the credit.")
    received_amount: int = Field(
        ..., description="Net amount credited to the receiver after the tax burn."
    )
    tax_amount: int = Field(
        ..., description="Amount burned by the transfer tax (removed from circulation)."
    )


class VipPurchaseResult(BaseModel):
    """Outcome of a successful VIP purchase."""

    model_config = ConfigDict(frozen=True)

    new_balance: int = Field(..., description="User balance after the VIP_PURCHASE_COST debit.")
    cost: int = Field(..., description="Points deducted for the purchase.")


class LoanProposalView(BaseModel):
    """Read-only loan proposal projected for command responses."""

    model_config = ConfigDict(frozen=True)

    proposal_id: int = Field(..., description="Row ID of the loan proposal.")
    kind: LoanProposalKind = Field(..., description="Pending loan proposal flow type.")
    status: LoanProposalStatus = Field(..., description="Current lifecycle state of the proposal.")
    lender_type: LoanLenderType = Field(
        ..., description="Kind of lender backing the proposed loan."
    )
    borrower_id: int = Field(..., description="Discord user ID of the borrower.")
    borrower_name: str = Field(..., description="Last-seen account name of the borrower.")
    lender_id: int | None = Field(
        ..., description="Discord user ID of the lender, or None for central-bank loans."
    )
    lender_name: str = Field(..., description="Display name of the lender.")
    amount: int = Field(..., description="Proposed loan principal amount.")
    monthly_rate_bps: int = Field(..., description="Monthly simple-interest rate in basis points.")
    escrow_amount: int = Field(
        ..., description="Amount held in escrow while the proposal is pending."
    )
    created_at: datetime = Field(..., description="Timestamp the proposal was created.")


class LoanContractView(BaseModel):
    """Read-only long-term loan contract snapshot."""

    model_config = ConfigDict(frozen=True)

    contract_id: int = Field(..., description="Row ID of the loan contract.")
    lender_type: LoanLenderType = Field(..., description="Kind of lender backing the contract.")
    lender_id: int | None = Field(
        ..., description="Discord user ID of the lender, or None for central-bank loans."
    )
    lender_name: str = Field(..., description="Display name of the lender.")
    borrower_id: int = Field(..., description="Discord user ID of the borrower.")
    borrower_name: str = Field(..., description="Last-seen account name of the borrower.")
    principal_remaining: int = Field(..., description="Outstanding loan principal still owed.")
    interest_due: int = Field(..., description="Accrued interest currently due on the contract.")
    monthly_rate_bps: int = Field(..., description="Monthly simple-interest rate in basis points.")
    opened_at: datetime = Field(..., description="Timestamp the contract was opened.")
    last_interest_accrued_at: datetime = Field(
        ..., description="Timestamp through which interest has been accrued."
    )
    status: LoanContractStatus = Field(..., description="Current lifecycle state of the contract.")


class LoanProposalAcceptResult(BaseModel):
    """Outcome of accepting a loan proposal."""

    model_config = ConfigDict(frozen=True)

    contract: LoanContractView = Field(
        ..., description="The loan contract created from the accepted proposal."
    )
    borrower_balance: int = Field(..., description="Borrower balance after acceptance.")
    lender_balance: int | None = Field(
        default=None,
        description="Lender balance after acceptance, or None for central-bank loans.",
    )
    central_bank_available_credit: int | None = Field(
        default=None,
        description="Remaining central bank available credit after acceptance, or None for personal loans.",
    )


class LoanPaymentResult(BaseModel):
    """Outcome of one loan repayment or forced collection command."""

    model_config = ConfigDict(frozen=True)

    paid_amount: int = Field(..., description="Total amount paid in this repayment or collection.")
    interest_paid: int = Field(..., description="Portion of the payment applied to interest.")
    principal_paid: int = Field(..., description="Portion of the payment applied to principal.")
    borrower_balance: int = Field(..., description="Borrower balance after the payment.")
    lender_balance: int | None = Field(
        default=None,
        description="Lender balance after the payment, or None for central-bank loans.",
    )
    remaining_principal: int = Field(
        ..., description="Outstanding principal still owed after the payment."
    )
    remaining_interest: int = Field(
        ..., description="Accrued interest still due after the payment."
    )
    closed_contract_ids: tuple[int, ...] = Field(
        default=(), description="Contract IDs closed as a result of this payment."
    )


class CentralBankStatus(BaseModel):
    """One guild's central bank lending capacity."""

    model_config = ConfigDict(frozen=True)

    participant_count: int = Field(
        ..., description="How many users are recorded as taking part in this guild's economy."
    )
    total_positive_user_balance: int = Field(
        ...,
        description="Sum of this guild's participants' positive balances backing its central bank credit.",
    )
    outstanding_principal: int = Field(
        ...,
        description="Central bank loan principal outstanding across the whole bank, which every guild's capacity is charged for.",
    )
    available_credit: int = Field(
        ...,
        description="Remaining lending capacity for this guild, including the flat base capacity every guild carries.",
    )
    ledger_balance: int = Field(
        ...,
        description="Interest the central bank has kept; lent out again on top of its starting capital.",
    )


class PortfolioView(BaseModel):
    """Aggregated wallet and debt view."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(..., description="Discord user ID of the account.")
    name: str = Field(..., description="Last-seen Discord account name.")
    balance: int = Field(..., description="Current spendable wallet balance.")
    debt_principal: int = Field(..., description="Total outstanding loan principal.")
    debt_interest: int = Field(..., description="Total accrued loan interest due.")
    net_worth: int = Field(..., description="Balance minus total debt principal and interest.")


__all__ = [
    "BASE_MESSAGE_REWARD_AMOUNT",
    "CENTRAL_BANK_BASE_CAPACITY",
    "CENTRAL_BANK_CREDIT_MULTIPLIER",
    "DEFAULT_LOAN_MONTHLY_RATE_BPS",
    "LOAN_PROPOSAL_TIMEOUT_SECONDS",
    "MAX_LOAN_MONTHLY_RATE_BPS",
    "MAX_SINGLE_BET",
    "MESSAGE_REWARD_COOLDOWN_SECONDS",
    "MIN_INTEREST_DAYS",
    "MIN_LOAN_MONTHLY_RATE_BPS",
    "TRANSFER_TAX_BPS",
    "VIP_PURCHASE_COST",
    "AccountSnapshot",
    "BalanceAdjustmentResult",
    "CasinoDailyStats",
    "CasinoLedgerSnapshot",
    "CentralBankStatus",
    "CreditResult",
    "JackpotSettlementBatchResult",
    "JackpotSettlementRequest",
    "JackpotSettlementResult",
    "JackpotSnapshot",
    "LeaderboardEntry",
    "LoanContractStatus",
    "LoanContractView",
    "LoanLenderType",
    "LoanPaymentResult",
    "LoanProposalAcceptResult",
    "LoanProposalKind",
    "LoanProposalStatus",
    "LoanProposalView",
    "LossLeaderboardEntry",
    "PortfolioView",
    "RoundSettlementResult",
    "TransferResult",
    "VipPurchaseResult",
    "apply_vip_blackjack_bonus",
    "central_bank_credit_ceiling",
    "monthly_rate_bps_to_percent",
    "monthly_rate_percent_to_bps",
]
