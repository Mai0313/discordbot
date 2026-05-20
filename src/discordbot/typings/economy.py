"""Pydantic models and enums for the economy domain.

Pure type definitions live here so ``cogs/_economy/database.py`` can import
them without pulling in ``cogs/`` modules. Lifecycle results are frozen so
they cannot be mutated after they leave the database layer.
"""

from enum import StrEnum
from typing import Final
from datetime import datetime

from pydantic import BaseModel, ConfigDict

BASE_MESSAGE_REWARD_AMOUNT: Final[int] = 5_000
BASE_CHECKIN_REWARD_AMOUNT: Final[int] = 100_000
VIP_PURCHASE_COST: Final[int] = 10_000_000
STOCK_ISSUE_MIN_NET_WORTH: Final[int] = 10_000_000
DEFAULT_LOAN_MONTHLY_RATE_BPS: Final[int] = 300
MIN_LOAN_MONTHLY_RATE_BPS: Final[int] = 0
MAX_LOAN_MONTHLY_RATE_BPS: Final[int] = 10_000
# Daily check-in streak cycles through 1..7 then loops back to 1.
CHECKIN_STREAK_CYCLE: Final[int] = 7


class TransactionKind(StrEnum):
    """Labels the source of a balance credit or debit.

    The economy DB no longer persists a per-mutation transaction log, but these
    labels keep reward call sites explicit and leave room for future event
    routing without changing the public database facade.

    Attributes:
        MESSAGE_REWARD: Base reward for every non-bot user message.
        CHAT_REWARD: Streaming AI reply token reward.
        CHECKIN_REWARD: Daily check-in payout, including streak bonus.
        CASINO_BET: Wager debit, including deferred settlement losses.
        CASINO_PAYOUT: Player-side payout from a finished casino round.
        HOUSE_SETTLE: Dealer-side mirror of a player settlement.
        BORROW: Loan disbursement from ``borrow``.
        REPAY: Manual repayment via ``repay``.
        TRANSFER_OUT: Sender side of ``/give``.
        TRANSFER_IN: Receiver side of ``/give``.
        VIP_PURCHASE: Debit for buying the permanent VIP perk.
        MANUAL_ADJUSTMENT: Admin-side balance adjustment from maintenance tooling.
    """

    MESSAGE_REWARD = "message_reward"
    CHAT_REWARD = "chat_reward"
    CHECKIN_REWARD = "checkin_reward"
    CASINO_BET = "casino_bet"
    CASINO_PAYOUT = "casino_payout"
    HOUSE_SETTLE = "house_settle"
    BORROW = "borrow"
    REPAY = "repay"
    TRANSFER_OUT = "transfer_out"
    TRANSFER_IN = "transfer_in"
    VIP_PURCHASE = "vip_purchase"
    MANUAL_ADJUSTMENT = "manual_adjustment"


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


class StockEventKind(StrEnum):
    """Stock ledger event kinds."""

    ISSUE = "issue"
    BUY = "buy"
    DIVIDEND = "dividend"


class AccountSnapshot(BaseModel):
    """Read-only account totals for maintenance and house-ledger views.

    Attributes:
        name: Last-seen Discord account name.
        balance: Current point balance.
        total_earned: Lifetime gross earned amount.
        total_spent: Lifetime gross spent amount.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    balance: int
    total_earned: int
    total_spent: int


class AdminAccount(BaseModel):
    """Read-only economy admin account row."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    name: str


class CentralBankerAccount(BaseModel):
    """Read-only central banker account row."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    name: str


class LeaderboardEntry(BaseModel):
    """One account row in the balance leaderboard."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    name: str
    balance: int
    avatar_url: str = ""


class LossLeaderboardEntry(BaseModel):
    """One account row in the daily casino loss leaderboard."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    name: str
    loss_amount: int
    avatar_url: str = ""


class CreditResult(BaseModel):
    """Outcome of an income event.

    Attributes:
        new_balance: User balance after the credit.
        credited_amount: Amount that landed in balance.
        principal_repaid: Always zero; long-term loans are repaid explicitly.
        remaining_debt: Always zero; use portfolio / loan views for active debt.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int
    credited_amount: int
    principal_repaid: int
    remaining_debt: int


class BalanceAdjustmentResult(BaseModel):
    """Outcome of a manual balance adjustment.

    Attributes:
        new_balance: User balance after the adjustment.
        applied_delta: Signed balance delta that was actually applied.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int
    applied_delta: int


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

    player_id: int
    player_account_name: str
    player_delta: int
    player_avatar_url: str = ""
    require_full_debit: bool = False
    expected_jackpot_generation: int | None = None


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

    player_balances: dict[int, int]
    applied_player_deltas: dict[int, int]
    jackpot_balance: int
    jackpot_generation: int = 0
    jackpot_depleted: bool = False
    rejected_player_ids: tuple[int, ...] = ()


class JackpotSnapshot(BaseModel):
    """Read-only snapshot of a shared jackpot pool."""

    model_config = ConfigDict(frozen=True)

    balance: int
    generation: int = 0


class JackpotSettlementResult(BaseModel):
    """Outcome of a single player settlement against a shared jackpot pool."""

    model_config = ConfigDict(frozen=True)

    player_balance: int
    jackpot_balance: int
    jackpot_generation: int = 0
    applied_player_delta: int
    jackpot_depleted: bool = False
    rejected: bool = False


class TransferResult(BaseModel):
    """A successful point transfer.

    Attributes:
        sender_balance: Sender balance after the debit.
        receiver_balance: Receiver balance after the credit.
    """

    model_config = ConfigDict(frozen=True)

    sender_balance: int
    receiver_balance: int


class CheckinResult(BaseModel):
    """Outcome of a successful daily check-in.

    Attributes:
        new_balance: User balance after the payout.
        amount: Total amount credited for this check-in (base * streak bonus * VIP multiplier).
        streak: Streak counter persisted on the account after this check-in
            (1..``CHECKIN_STREAK_CYCLE``).
        is_vip: VIP status of the account at check-in time, surfaced so the
            embed can label the bonus correctly.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int
    amount: int
    streak: int
    is_vip: bool


class VipPurchaseResult(BaseModel):
    """Outcome of a successful VIP purchase.

    Attributes:
        new_balance: User balance after the 10M debit.
        cost: Points deducted for the purchase.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int
    cost: int


class LoanProposalView(BaseModel):
    """Read-only loan proposal projected for command responses."""

    model_config = ConfigDict(frozen=True)

    proposal_id: int
    kind: LoanProposalKind
    status: LoanProposalStatus
    lender_type: LoanLenderType
    borrower_id: int
    borrower_name: str
    lender_id: int | None
    lender_name: str
    amount: int
    monthly_rate_bps: int
    escrow_amount: int
    created_at: datetime


class LoanContractView(BaseModel):
    """Read-only long-term loan contract snapshot."""

    model_config = ConfigDict(frozen=True)

    contract_id: int
    lender_type: LoanLenderType
    lender_id: int | None
    lender_name: str
    borrower_id: int
    borrower_name: str
    principal_remaining: int
    interest_due: int
    monthly_rate_bps: int
    opened_at: datetime
    last_interest_accrued_at: datetime
    status: LoanContractStatus


class LoanProposalAcceptResult(BaseModel):
    """Outcome of accepting a loan proposal."""

    model_config = ConfigDict(frozen=True)

    contract: LoanContractView
    borrower_balance: int
    lender_balance: int | None = None
    central_bank_available_credit: int | None = None


class LoanPaymentResult(BaseModel):
    """Outcome of one loan repayment or forced collection command."""

    model_config = ConfigDict(frozen=True)

    paid_amount: int
    interest_paid: int
    principal_paid: int
    borrower_balance: int
    lender_balance: int | None = None
    remaining_principal: int
    remaining_interest: int
    closed_contract_ids: tuple[int, ...] = ()


class CentralBankStatus(BaseModel):
    """Aggregated central bank lending capacity."""

    model_config = ConfigDict(frozen=True)

    total_positive_user_balance: int
    outstanding_principal: int
    available_credit: int


class StockProfileView(BaseModel):
    """Read-only stock profile for one issuer."""

    model_config = ConfigDict(frozen=True)

    issuer_id: int
    issuer_name: str
    total_shares: int
    treasury_shares: int
    issue_price: int
    sold_shares: int


class StockHoldingView(BaseModel):
    """Read-only stock holding row."""

    model_config = ConfigDict(frozen=True)

    issuer_id: int
    issuer_name: str
    holder_id: int
    holder_name: str
    shares: int
    issue_price: int
    estimated_value: int


class StockPurchaseResult(BaseModel):
    """Outcome of buying issuer treasury shares."""

    model_config = ConfigDict(frozen=True)

    buyer_balance: int
    issuer_balance: int
    shares_bought: int
    total_cost: int
    treasury_shares: int


class DividendResult(BaseModel):
    """Outcome of a manual stock dividend."""

    model_config = ConfigDict(frozen=True)

    distributed_amount: int
    issuer_balance: int
    recipient_count: int


class PortfolioView(BaseModel):
    """Aggregated wallet, debt, and stock view."""

    model_config = ConfigDict(frozen=True)

    user_id: int
    name: str
    balance: int
    stock_value: int
    debt_principal: int
    debt_interest: int
    net_worth: int
    holdings: tuple[StockHoldingView, ...] = ()


__all__ = [
    "BASE_CHECKIN_REWARD_AMOUNT",
    "BASE_MESSAGE_REWARD_AMOUNT",
    "CHECKIN_STREAK_CYCLE",
    "DEFAULT_LOAN_MONTHLY_RATE_BPS",
    "MAX_LOAN_MONTHLY_RATE_BPS",
    "MIN_LOAN_MONTHLY_RATE_BPS",
    "STOCK_ISSUE_MIN_NET_WORTH",
    "VIP_PURCHASE_COST",
    "AccountSnapshot",
    "AdminAccount",
    "BalanceAdjustmentResult",
    "CentralBankStatus",
    "CentralBankerAccount",
    "CheckinResult",
    "CreditResult",
    "DividendResult",
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
    "StockEventKind",
    "StockHoldingView",
    "StockProfileView",
    "StockPurchaseResult",
    "TransactionKind",
    "TransferResult",
    "VipPurchaseResult",
]
