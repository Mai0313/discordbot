"""In-process seat registry limiting each human player to one active casino round at a time.

A seat runs from lobby admission through round settlement and registers a wallet hold in
`services/economy/holds.py`: the full wallet while the lobby reads balances, then the table's
maximum possible loss for Blackjack or net positive table winnings (starting at `0` and updated
after each turn) for 射龍門 so mid-round winnings stay protected for 逆贏不拿. The bot player
is never seated here. State is in-process only and clears on restart.
"""

from typing import Protocol

from discordbot.utils.asyncio_locks import LoopLocalRegistry
from discordbot.services.economy.holds import hold_stake, release_stake


class SeatHolder(Protocol):
    """A lobby or table that can hold players' seats."""

    def holds_seats(self) -> bool:
        """Returns whether the seats this holder claimed still count."""


_seats: LoopLocalRegistry[int, SeatHolder] = LoopLocalRegistry()


def seated_elsewhere(user_id: int, holder: SeatHolder | None = None) -> bool:
    """Returns whether `user_id` holds a live seat anywhere but `holder`.

    Args:
        user_id (int): Discord user ID to check.
        holder (SeatHolder | None): Holder to exclude from the check, if any.
    """
    current = _seats.get(key=user_id)
    return current is not None and current is not holder and current.holds_seats()


def claim_seat(user_id: int, holder: SeatHolder) -> None:
    """Seats `user_id` at `holder`, holding their whole wallet until the stake is known.

    The caller has just checked `seated_elsewhere`.

    Args:
        user_id (int): Discord user ID to seat.
        holder (SeatHolder): Lobby claiming the seat.
    """
    _seats.set(key=user_id, value=holder)
    hold_stake(user_id=user_id, holder=holder, amount=None)


def hand_over_seats(
    stakes: dict[int, int], from_holder: SeatHolder, to_holder: SeatHolder
) -> None:
    """Moves the seats `from_holder` holds to `to_holder`, each holding its player's stake.

    Args:
        stakes (dict[int, int]): What the table holds of each player's wallet: the most the round
            can take from them (`blackjack.py::max_round_loss`), or 0 for a game that settles as
            it goes.
        from_holder (SeatHolder): The lobby that claimed the seats.
        to_holder (SeatHolder): The table that now plays them.
    """
    for user_id, stake in stakes.items():
        if _seats.get(key=user_id) is from_holder:
            _seats.set(key=user_id, value=to_holder)
            release_stake(user_id=user_id, holder=from_holder)
            hold_stake(user_id=user_id, holder=to_holder, amount=stake)


def update_seat_stake(user_id: int, holder: SeatHolder, amount: int) -> None:
    """Updates `holder`'s wallet hold for `user_id` if `holder` still seats them.

    Args:
        user_id (int): Discord user ID whose hold to update.
        holder (SeatHolder): Active table holding the seat.
        amount (int): Points to hold (clamped at 0).
    """
    if _seats.get(key=user_id) is holder:
        hold_stake(user_id=user_id, holder=holder, amount=max(amount, 0))


def release_seats(user_ids: list[int], holder: SeatHolder) -> None:
    """Frees the seats `holder` holds for `user_ids`, leaving anyone seated elsewhere alone.

    Args:
        user_ids (list[int]): Discord user IDs whose seats to release.
        holder (SeatHolder): Lobby or table releasing its seats.
    """
    for user_id in user_ids:
        if _seats.get(key=user_id) is holder:
            _seats.pop(key=user_id)
            release_stake(user_id=user_id, holder=holder)
