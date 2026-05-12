"""Pydantic models and enums for the economy domain.

Pure type definitions live here so ``cogs/_economy/database.py`` can import
them without pulling in ``cogs/`` modules. Lifecycle results are frozen so
they cannot be mutated after they leave the database layer.
"""

from enum import StrEnum
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TransactionKind(StrEnum):
    """Categorises a row in the ``point_transaction`` audit log.

    Attributes:
        CHAT_REWARD: Streaming AI reply token reward.
        CASINO_BET: Wager debit, including deferred settlement losses.
        CASINO_PAYOUT: Player-side payout from a finished casino round.
        HOUSE_SETTLE: Dealer-side mirror of a player settlement.
        BORROW: Loan disbursement from ``borrow``.
        REPAY: Manual repayment via ``repay``.
        TRANSFER_OUT: Sender side of ``/give``.
        TRANSFER_IN: Receiver side of ``/give``.
    """

    CHAT_REWARD = "chat_reward"
    CASINO_BET = "casino_bet"
    CASINO_PAYOUT = "casino_payout"
    HOUSE_SETTLE = "house_settle"
    BORROW = "borrow"
    REPAY = "repay"
    TRANSFER_OUT = "transfer_out"
    TRANSFER_IN = "transfer_in"


class LoanView(BaseModel):
    """Read-only snapshot of a user's loan state.

    Callers add the pending accrual (see ``accrual_delta`` in
    ``cogs/_economy/database.py``) to ``interest_stored`` to obtain the
    effective interest as of the read time. ``last_accrual_at`` is the
    point in time the stored interest reflects.

    Attributes:
        principal: Outstanding loan principal.
        interest_stored: Interest already accrued and persisted.
        last_accrual_at: Timestamp the stored interest was last brought up to date.
        opened_at: Timestamp the user first borrowed; ``None`` if never borrowed.
        total_borrowed: Lifetime gross borrowed amount.
        total_repaid: Lifetime gross repaid amount.
    """

    model_config = ConfigDict(frozen=True)

    principal: int
    interest_stored: int
    last_accrual_at: datetime | None
    opened_at: datetime | None
    total_borrowed: int
    total_repaid: int


class CreditResult(BaseModel):
    """Outcome of an income event that may auto-repay outstanding debt.

    Attributes:
        new_balance: User balance after the credit.
        credited_amount: Amount that landed in balance; ``amount - to_repay``.
        interest_repaid: Amount that paid down ``loan_interest``.
        principal_repaid: Amount that paid down ``loan_principal``.
        remaining_debt: ``loan_principal + loan_interest`` after the operation.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int
    credited_amount: int
    interest_repaid: int
    principal_repaid: int
    remaining_debt: int


class BorrowResult(BaseModel):
    """Outcome of a successful borrow.

    Attributes:
        new_balance: User balance after the disbursement.
        principal: Outstanding principal after this borrow.
        interest: Outstanding interest after the implicit accrual.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int
    principal: int
    interest: int


class RepayResult(BaseModel):
    """Outcome of a successful repay.

    Attributes:
        new_balance: User balance after the deduction.
        interest_repaid: Amount that paid down ``loan_interest``.
        principal_repaid: Amount that paid down ``loan_principal``.
        remaining_debt: ``loan_principal + loan_interest`` after the operation.
    """

    model_config = ConfigDict(frozen=True)

    new_balance: int
    interest_repaid: int
    principal_repaid: int
    remaining_debt: int


class PlacedBet(BaseModel):
    """A successfully withdrawn wager.

    Attributes:
        amount: Actual amount withdrawn. This may be lower than the requested amount for all-in.
        balance_after: Account balance after the bet was withdrawn.
        is_allin: True when the requested bet was clamped to the available balance.
    """

    model_config = ConfigDict(frozen=True)

    amount: int
    balance_after: int
    is_allin: bool


class PreparedBet(BaseModel):
    """A wager accepted for a round but not yet deducted.

    Attributes:
        amount: Effective wager amount. This may be lower than the requested amount for all-in.
        balance_at_start: Account balance observed when the round started.
        is_allin: True when the requested bet was clamped to the available balance.
    """

    model_config = ConfigDict(frozen=True)

    amount: int
    balance_at_start: int
    is_allin: bool


class TransferResult(BaseModel):
    """A successful point transfer.

    Attributes:
        sender_balance: Sender balance after the debit.
        receiver_balance: Receiver balance after the credit.
    """

    model_config = ConfigDict(frozen=True)

    sender_balance: int
    receiver_balance: int


__all__ = [
    "BorrowResult",
    "CreditResult",
    "LoanView",
    "PlacedBet",
    "PreparedBet",
    "RepayResult",
    "TransactionKind",
    "TransferResult",
]
