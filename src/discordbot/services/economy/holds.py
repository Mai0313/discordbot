"""Money a player has on a casino table, which no other debit may take while the round plays.

A Blackjack stake is settled only after play, and its loss is collected only up to what the wallet
holds then, so money a seated player moved out mid-round (a transfer, a VIP purchase, a loan
repayment or one they fund) would leave that loss uncollectable. While a round seats a player,
the most it can take from them stays on the table: every voluntary debit sees the wallet minus
that hold, and runs under the player's spending lock, so a table that claims the player can wait
out a debit already in flight before it reads the balance. Money earned during the round stays
spendable.

Held in process only: a round dies with the process without touching a balance, and its hold
goes with it.
"""

from typing import Protocol
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from discordbot.utils.asyncio_locks import KeyedLockManager, LoopLocalRegistry


class StakeHolder(Protocol):
    """A lobby or table that can hold a player's stake."""

    def holds_seats(self) -> bool:
        """Returns whether this holder's seats, and so its holds, still count."""


# user id -> id(holder) -> (holder, held amount), where None holds the whole wallet. Keyed by
# identity since a holder need not be hashable; the entry keeps the holder alive, so its id is
# never reused while it is held.
_holds: LoopLocalRegistry[int, dict[int, tuple[StakeHolder, int | None]]] = LoopLocalRegistry()
_spending: KeyedLockManager[int] = KeyedLockManager()


def hold_stake(user_id: int, holder: StakeHolder, amount: int | None) -> None:
    """Holds `amount` of the user's wallet for `holder`; None holds all of it.

    Args:
        user_id (int): Discord user ID whose wallet is held.
        holder (StakeHolder): Lobby or table placing the hold.
        amount (int | None): Points to hold, or None to hold the entire wallet.
    """
    _holds.setdefault(key=user_id, default={})[id(holder)] = (holder, amount)


def release_stake(user_id: int, holder: StakeHolder) -> None:
    """Drops `holder`'s hold on the user's wallet, leaving any other hold alone.

    Args:
        user_id (int): Discord user ID whose hold is released.
        holder (StakeHolder): Lobby or table releasing its hold.
    """
    held = _holds.get(key=user_id)
    if held is None:
        return
    held.pop(id(holder), None)
    if not held:
        _holds.pop(key=user_id)


def held_amount(user_id: int) -> int | None:
    """Returns how much of the user's wallet is held: 0 for none, None for all of it.

    A holder that no longer holds seats holds nothing, so a hold its table failed to release
    cannot freeze the wallet.

    Args:
        user_id (int): Discord user ID to inspect.
    """
    live = [
        amount
        for holder, amount in (_holds.get(key=user_id) or {}).values()
        if holder.holds_seats()
    ]
    if any(amount is None for amount in live):
        return None
    return sum(amount for amount in live if amount is not None)


def spendable(user_id: int, balance: int) -> int:
    """Returns the part of `balance` no table holds.

    Args:
        user_id (int): Discord user ID whose holds apply.
        balance (int): Observed wallet balance before holds.
    """
    held = held_amount(user_id=user_id)
    if held is None:
        return 0
    return max(balance - held, 0)


@asynccontextmanager
async def spending(user_id: int) -> AsyncGenerator[None]:
    """Serializes the user's voluntary debits, check through commit, against a table's claim.

    Args:
        user_id (int): Discord user ID whose voluntary debits are serialized.
    """
    async with _spending.hold(key=user_id):
        yield


async def wait_for_spending(user_ids: list[int]) -> None:
    """Waits until no voluntary debit of `user_ids` that started before the call is in flight.

    Args:
        user_ids (list[int]): Discord user IDs to wait on in order.
    """
    for user_id in user_ids:
        async with _spending.hold(key=user_id):
            pass
