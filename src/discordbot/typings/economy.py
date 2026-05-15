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
# Daily check-in streak cycles through 1..7 then loops back to 1.
CHECKIN_STREAK_CYCLE: Final[int] = 7


class TransactionKind(StrEnum):
    """Categorises a row in the ``point_transaction`` audit log.

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


class LoanView(BaseModel):
    """Read-only snapshot of a user's loan state.

    Loan principal is wiped at the Taipei daily midnight reset; ``opened_at``
    is therefore always today's borrow timestamp (or ``None`` after the
    nightly reset has cleared it).

    Attributes:
        principal: Outstanding loan principal that has not yet expired.
        opened_at: Timestamp the user first borrowed today; ``None`` once
            the daily reset has cleared the loan.
        total_borrowed: Lifetime gross borrowed amount.
        total_repaid: Lifetime gross repaid amount.
    """

    model_config = ConfigDict(frozen=True)

    principal: int
    opened_at: datetime | None
    total_borrowed: int
    total_repaid: int


class CreditResult(BaseModel):
    """Outcome of an income event that may auto-repay outstanding debt.

    Attributes:
        new_balance: User balance after the credit.
        credited_amount: Amount that landed in balance; ``amount - to_repay``.
        principal_repaid: Amount that paid down ``loan_principal``.
        remaining_debt: ``loan_principal`` after the operation.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int
    credited_amount: int
    principal_repaid: int
    remaining_debt: int


class BorrowResult(BaseModel):
    """Outcome of a successful borrow.

    Attributes:
        new_balance: User balance after the disbursement.
        principal: Outstanding principal after this borrow.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int
    principal: int


class RepayResult(BaseModel):
    """Outcome of a successful repay.

    Attributes:
        new_balance: User balance after the deduction.
        principal_repaid: Amount that paid down ``loan_principal``.
        remaining_debt: ``loan_principal`` after the operation.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int
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


__all__ = [
    "BASE_CHECKIN_REWARD_AMOUNT",
    "BASE_MESSAGE_REWARD_AMOUNT",
    "CHECKIN_STREAK_CYCLE",
    "VIP_PURCHASE_COST",
    "BalanceAdjustmentResult",
    "BorrowResult",
    "CheckinResult",
    "CreditResult",
    "JackpotSettlementBatchResult",
    "JackpotSettlementRequest",
    "JackpotSettlementResult",
    "JackpotSnapshot",
    "LoanView",
    "RepayResult",
    "TransactionKind",
    "TransferResult",
    "VipPurchaseResult",
]
