from enum import StrEnum
from typing import Final
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict

BASE_MESSAGE_REWARD_AMOUNT: Final[int] = 10
BASE_CHECKIN_REWARD_AMOUNT: Final[int] = 500
VIP_PURCHASE_COST: Final[int] = 50_000
LOAN_PROPOSAL_TIMEOUT_SECONDS: Final[int] = 180
DEFAULT_LOAN_MONTHLY_RATE_BPS: Final[int] = 300
MIN_LOAN_MONTHLY_RATE_BPS: Final[int] = 0
MAX_LOAN_MONTHLY_RATE_BPS: Final[int] = 10_000
# Minimum interest a borrower owes on a contract regardless of repayment timing.
# Prepaid at acceptance so borrow-then-immediately-repay still costs MIN_INTEREST_DAYS worth.
MIN_INTEREST_DAYS: Final[int] = 30
# Daily check-in streak cycles through 1..7 then loops back to 1.
CHECKIN_STREAK_CYCLE: Final[int] = 7

# Anti-inflation guardrails. Faucets are deflated and a few structural caps keep
# balances from compounding back to pre-reset astronomical levels.
# Absolute ceiling on any single casino wager (Blackjack table bet, Dragon Gate
# bet). Invisible to ordinary players; it turns runaway exponential growth from
# all-in doubling into bounded linear growth once a balance gets large.
MAX_SINGLE_BET: Final[int] = 1_000_000
# Per-user cooldown between message rewards, so the flat per-message grant cannot
# be farmed by spamming. Tracked process-locally; resets on restart by design.
MESSAGE_REWARD_COOLDOWN_SECONDS: Final[float] = 60.0
# Chat reward is token-based; divide and cap so a single long (e.g. web-search)
# reply cannot mint tens of thousands of points at once.
CHAT_REWARD_TOKEN_DIVISOR: Final[int] = 100
CHAT_REWARD_MAX_PER_REPLY: Final[int] = 50
# Permanent money sink: a burn on every /give transfer, in basis points.
TRANSFER_TAX_BPS: Final[int] = 500


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


class WalletResetMode(StrEnum):
    """How an offline economy reset rewrites every wallet balance.

    LOG_COMPRESS: monotonic log10 compression that preserves rank ordering
        while collapsing absolute magnitude.
    FIXED: set every account to the same starting balance.
    WIPE: set every account to zero.
    """

    LOG_COMPRESS = "log_compress"
    FIXED = "fixed"
    WIPE = "wipe"


class AccountSnapshot(BaseModel):
    """Read-only account totals for maintenance and house-ledger views.

    Attributes:
        name: Last-seen Discord account name.
        balance: Current point balance.
        total_earned: Lifetime gross earned amount.
        total_spent: Lifetime gross spent amount.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Last-seen Discord account name.")
    balance: int = Field(description="Current point balance.")
    total_earned: int = Field(description="Lifetime gross earned amount.")
    total_spent: int = Field(description="Lifetime gross spent amount.")


class AdminAccount(BaseModel):
    """Read-only economy admin account row."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(description="Discord user ID of the economy admin account.")
    name: str = Field(description="Last-seen Discord account name.")


class CentralBankerAccount(BaseModel):
    """Read-only central banker account row."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(description="Discord user ID of the central banker account.")
    name: str = Field(description="Last-seen Discord account name.")


class LeaderboardEntry(BaseModel):
    """One account row in the balance leaderboard."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(description="Discord user ID of the leaderboard account.")
    name: str = Field(description="Last-seen Discord account name.")
    balance: int = Field(description="Current point balance used for ranking.")
    avatar_url: str = Field(
        default="", description="Last-seen Discord avatar URL for the account."
    )


class LossLeaderboardEntry(BaseModel):
    """One account row in the daily casino loss leaderboard."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(description="Discord user ID of the leaderboard account.")
    name: str = Field(description="Last-seen Discord account name.")
    loss_amount: int = Field(description="Gross current-day casino loss used for ranking.")
    avatar_url: str = Field(
        default="", description="Last-seen Discord avatar URL for the account."
    )


class CreditResult(BaseModel):
    """Outcome of an income event.

    Attributes:
        new_balance: User balance after the credit.
        credited_amount: Amount that landed in balance.
        principal_repaid: Always zero; long-term loans are repaid explicitly.
        remaining_debt: Always zero; use portfolio / loan views for active debt.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int = Field(description="User balance after the credit.")
    credited_amount: int = Field(description="Amount that landed in balance.")
    principal_repaid: int = Field(
        description="Always zero; long-term loans are repaid explicitly."
    )
    remaining_debt: int = Field(
        description="Always zero; use portfolio / loan views for active debt."
    )


class BalanceAdjustmentResult(BaseModel):
    """Outcome of a manual balance adjustment.

    Attributes:
        new_balance: User balance after the adjustment.
        applied_delta: Signed balance delta that was actually applied.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int = Field(description="User balance after the adjustment.")
    applied_delta: int = Field(description="Signed balance delta that was actually applied.")


class WalletResetSummary(BaseModel):
    """Outcome of a bulk offline wallet reset.

    Attributes:
        mode: Reset transform that was applied.
        accounts: Number of wallet rows rewritten.
        total_before: Sum of every balance before the reset.
        total_after: Sum of every balance after the reset.
        max_before: Largest single balance before the reset.
        max_after: Largest single balance after the reset.
        dry_run: Whether the reset only computed the summary without writing.
    """

    model_config = ConfigDict(frozen=True)

    mode: WalletResetMode = Field(description="Reset transform that was applied.")
    accounts: int = Field(description="Number of wallet rows rewritten.")
    total_before: int = Field(description="Sum of every balance before the reset.")
    total_after: int = Field(description="Sum of every balance after the reset.")
    max_before: int = Field(description="Largest single balance before the reset.")
    max_after: int = Field(description="Largest single balance after the reset.")
    dry_run: bool = Field(
        default=False, description="Whether the reset only computed the summary without writing."
    )


class WalletDeltaLeg(BaseModel):
    """One ordered wallet delta requested by another domain service."""

    model_config = ConfigDict(frozen=True)

    delta: int = Field(description="Signed wallet delta to apply for this leg.")
    reason: str = Field(default="", description="Optional reason describing this wallet delta.")


class OrderedWalletDeltaResult(BaseModel):
    """Outcome of applying ordered wallet deltas without netting them."""

    model_config = ConfigDict(frozen=True)

    new_balance: int = Field(description="User balance after all ordered deltas are applied.")
    applied_deltas: tuple[int, ...] = Field(
        description="Signed deltas that were actually applied, in order."
    )


class JackpotSettlementRequest(BaseModel):
    """One player-side settlement against a shared jackpot pool.

    Attributes:
        player_id: Discord user ID for the player account.
        player_account_name: Last-seen account name stored on the player row.
        player_delta: Signed change for the player; the pool receives the inverse.
        player_avatar_url: Last-seen Discord avatar URL for the player.
        require_full_debit: Whether a negative delta must be applied in full,
            rejecting the whole batch instead of clamping at the player's
            current balance. Used by pre-game antes.
        expected_jackpot_generation: Optional jackpot generation observed by
            the game view. Positive payouts only claim from this generation,
            so a stale action cannot spend a freshly reseeded pool.
    """

    model_config = ConfigDict(frozen=True)

    player_id: int = Field(description="Discord user ID for the player account.")
    player_account_name: str = Field(
        description="Last-seen account name stored on the player row."
    )
    player_delta: int = Field(
        description="Signed change for the player; the pool receives the inverse."
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
        description="Optional jackpot generation observed by the game view; positive payouts only claim from this generation.",
    )


class JackpotSettlementBatchResult(BaseModel):
    """Outcome of one or more settlements against a shared jackpot pool.

    Attributes:
        player_balances: Latest post-settlement balance for each touched player.
        applied_player_deltas: Signed player deltas that were actually applied.
            Losses may be smaller than requested when the balance clamps at zero.
        jackpot_balance: Pool balance after the final settlement and any reseed.
        jackpot_generation: Pool generation after the final settlement and any
            reseed.
        jackpot_depleted: True when a seeded pool was drained and automatically
            replenished during this batch.
        rejected_player_ids: Player IDs whose required full debit could not be
            applied; no mutation is committed when this is non-empty.
    """

    model_config = ConfigDict(frozen=True)

    player_balances: dict[int, int] = Field(
        description="Latest post-settlement balance for each touched player."
    )
    applied_player_deltas: dict[int, int] = Field(
        description="Signed player deltas that were actually applied; losses may be smaller than requested when the balance clamps at zero."
    )
    jackpot_balance: int = Field(
        description="Pool balance after the final settlement and any reseed."
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

    balance: int = Field(description="Current jackpot pool balance.")
    generation: int = Field(default=0, description="Current jackpot pool generation counter.")


class JackpotSettlementResult(BaseModel):
    """Outcome of a single player settlement against a shared jackpot pool."""

    model_config = ConfigDict(frozen=True)

    player_balance: int = Field(description="Player balance after this settlement.")
    jackpot_balance: int = Field(description="Pool balance after this settlement and any reseed.")
    jackpot_generation: int = Field(
        default=0, description="Pool generation after this settlement and any reseed."
    )
    applied_player_delta: int = Field(description="Signed player delta that was actually applied.")
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

    balance: int = Field(description="Current casino system ledger balance.")
    total_earned: int = Field(description="Lifetime gross amount earned by the casino ledger.")
    total_spent: int = Field(description="Lifetime gross amount paid out by the casino ledger.")
    updated_at: datetime = Field(description="Timestamp of the last casino ledger update.")


class CasinoDailyStats(BaseModel):
    """Per-user current-day casino loss/win/net totals.

    Returned by `get_casino_daily_stats`; all zero when no row exists or the
    stored counters belong to a previous Taipei day.
    """

    model_config = ConfigDict(frozen=True)

    daily_loss: int = Field(description="Gross current-day casino loss total.")
    daily_win: int = Field(description="Gross current-day casino win total.")
    daily_net: int = Field(description="Net current-day casino result (win minus loss).")


class RoundSettlementResult(BaseModel):
    """Outcome of an atomic player + casino ledger settlement."""

    model_config = ConfigDict(frozen=True)

    player_balance: int = Field(description="Player balance after the round settlement.")
    casino_balance: int = Field(
        description="Casino system ledger balance after the round settlement."
    )


class TransferResult(BaseModel):
    """A successful point transfer.

    Attributes:
        sender_balance: Sender balance after the debit.
        receiver_balance: Receiver balance after the credit.
        received_amount: Net amount credited to the receiver after the tax burn.
        tax_amount: Amount burned by the transfer tax (removed from circulation).
    """

    model_config = ConfigDict(frozen=True)

    sender_balance: int = Field(description="Sender balance after the debit.")
    receiver_balance: int = Field(description="Receiver balance after the credit.")
    received_amount: int = Field(
        description="Net amount credited to the receiver after the tax burn."
    )
    tax_amount: int = Field(
        description="Amount burned by the transfer tax (removed from circulation)."
    )


class CheckinResult(BaseModel):
    """Outcome of a successful daily check-in.

    Attributes:
        new_balance: User balance after the payout.
        amount: Total amount credited for this check-in (base * streak bonus * VIP multiplier).
        streak: Streak counter persisted on the account after this check-in
            (1..`CHECKIN_STREAK_CYCLE`).
        is_vip: VIP status of the account at check-in time, surfaced so the
            embed can label the bonus correctly.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int = Field(description="User balance after the payout.")
    amount: int = Field(
        description="Total amount credited for this check-in (base * streak bonus * VIP multiplier)."
    )
    streak: int = Field(
        description="Streak counter persisted on the account after this check-in (1..CHECKIN_STREAK_CYCLE)."
    )
    is_vip: bool = Field(
        description="VIP status of the account at check-in time, surfaced so the embed can label the bonus correctly."
    )


class VipPurchaseResult(BaseModel):
    """Outcome of a successful VIP purchase.

    Attributes:
        new_balance: User balance after the 10M debit.
        cost: Points deducted for the purchase.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int = Field(description="User balance after the 10M debit.")
    cost: int = Field(description="Points deducted for the purchase.")


class LoanProposalView(BaseModel):
    """Read-only loan proposal projected for command responses."""

    model_config = ConfigDict(frozen=True)

    proposal_id: int = Field(description="Row ID of the loan proposal.")
    kind: LoanProposalKind = Field(description="Pending loan proposal flow type.")
    status: LoanProposalStatus = Field(description="Current lifecycle state of the proposal.")
    lender_type: LoanLenderType = Field(description="Kind of lender backing the proposed loan.")
    borrower_id: int = Field(description="Discord user ID of the borrower.")
    borrower_name: str = Field(description="Last-seen account name of the borrower.")
    lender_id: int | None = Field(
        description="Discord user ID of the lender, or None for central-bank loans."
    )
    lender_name: str = Field(description="Display name of the lender.")
    amount: int = Field(description="Proposed loan principal amount.")
    monthly_rate_bps: int = Field(description="Monthly simple-interest rate in basis points.")
    escrow_amount: int = Field(description="Amount held in escrow while the proposal is pending.")
    created_at: datetime = Field(description="Timestamp the proposal was created.")


class LoanContractView(BaseModel):
    """Read-only long-term loan contract snapshot."""

    model_config = ConfigDict(frozen=True)

    contract_id: int = Field(description="Row ID of the loan contract.")
    lender_type: LoanLenderType = Field(description="Kind of lender backing the contract.")
    lender_id: int | None = Field(
        description="Discord user ID of the lender, or None for central-bank loans."
    )
    lender_name: str = Field(description="Display name of the lender.")
    borrower_id: int = Field(description="Discord user ID of the borrower.")
    borrower_name: str = Field(description="Last-seen account name of the borrower.")
    principal_remaining: int = Field(description="Outstanding loan principal still owed.")
    interest_due: int = Field(description="Accrued interest currently due on the contract.")
    monthly_rate_bps: int = Field(description="Monthly simple-interest rate in basis points.")
    opened_at: datetime = Field(description="Timestamp the contract was opened.")
    last_interest_accrued_at: datetime = Field(
        description="Timestamp through which interest has been accrued."
    )
    status: LoanContractStatus = Field(description="Current lifecycle state of the contract.")


class LoanProposalAcceptResult(BaseModel):
    """Outcome of accepting a loan proposal."""

    model_config = ConfigDict(frozen=True)

    contract: LoanContractView = Field(
        description="The loan contract created from the accepted proposal."
    )
    borrower_balance: int = Field(description="Borrower balance after acceptance.")
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

    paid_amount: int = Field(description="Total amount paid in this repayment or collection.")
    interest_paid: int = Field(description="Portion of the payment applied to interest.")
    principal_paid: int = Field(description="Portion of the payment applied to principal.")
    borrower_balance: int = Field(description="Borrower balance after the payment.")
    lender_balance: int | None = Field(
        default=None,
        description="Lender balance after the payment, or None for central-bank loans.",
    )
    remaining_principal: int = Field(
        description="Outstanding principal still owed after the payment."
    )
    remaining_interest: int = Field(description="Accrued interest still due after the payment.")
    closed_contract_ids: tuple[int, ...] = Field(
        default=(), description="Contract IDs closed as a result of this payment."
    )


class CentralBankStatus(BaseModel):
    """Aggregated central bank lending capacity."""

    model_config = ConfigDict(frozen=True)

    total_positive_user_balance: int = Field(
        description="Sum of all positive user balances backing central bank credit."
    )
    outstanding_principal: int = Field(
        description="Total central bank loan principal currently outstanding."
    )
    available_credit: int = Field(description="Remaining central bank lending capacity.")


class PortfolioView(BaseModel):
    """Aggregated wallet and debt view."""

    model_config = ConfigDict(frozen=True)

    user_id: int = Field(description="Discord user ID of the account.")
    name: str = Field(description="Last-seen Discord account name.")
    balance: int = Field(description="Current spendable wallet balance.")
    debt_principal: int = Field(description="Total outstanding loan principal.")
    debt_interest: int = Field(description="Total accrued loan interest due.")
    net_worth: int = Field(description="Balance minus total debt principal and interest.")


__all__ = [
    "BASE_CHECKIN_REWARD_AMOUNT",
    "BASE_MESSAGE_REWARD_AMOUNT",
    "CHAT_REWARD_MAX_PER_REPLY",
    "CHAT_REWARD_TOKEN_DIVISOR",
    "CHECKIN_STREAK_CYCLE",
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
    "AdminAccount",
    "BalanceAdjustmentResult",
    "CasinoDailyStats",
    "CasinoLedgerSnapshot",
    "CentralBankStatus",
    "CentralBankerAccount",
    "CheckinResult",
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
    "OrderedWalletDeltaResult",
    "PortfolioView",
    "RoundSettlementResult",
    "TransferResult",
    "VipPurchaseResult",
    "WalletDeltaLeg",
    "WalletResetMode",
    "WalletResetSummary",
]
