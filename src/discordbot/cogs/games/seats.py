"""Which started Blackjack round each human player is seated at.

A Blackjack stake is never escrowed, and its settlement clamps a loss at what the wallet holds
while paying a win in full, so one balance seated at two unsettled rounds is paid every win but
loses at most once. A player therefore holds one seat at a time, from the moment a lobby admits
them to its round until that round's settlements are written. The bot player is never seated
here: it joins every table on purpose.

Held in process only: a round dies with the process without touching a balance, and its seat
goes with it.
"""

from typing import Protocol

from discordbot.utils.asyncio_locks import LoopLocalRegistry


class SeatHolder(Protocol):
    """A lobby or table that can hold players' seats."""

    def holds_seats(self) -> bool:
        """Whether the seats this holder claimed still count; a stale holder blocks nobody."""


_seats: LoopLocalRegistry[int, SeatHolder] = LoopLocalRegistry()


def seated_elsewhere(user_id: int, holder: SeatHolder | None = None) -> bool:
    """Returns whether the user holds a live seat anywhere but `holder`."""
    current = _seats.get(key=user_id)
    return current is not None and current is not holder and current.holds_seats()


def claim_seat(user_id: int, holder: SeatHolder) -> None:
    """Seats the user at `holder`; the caller has just checked `seated_elsewhere`."""
    _seats.set(key=user_id, value=holder)


def hand_over_seats(user_ids: list[int], from_holder: SeatHolder, to_holder: SeatHolder) -> None:
    """Moves the seats `from_holder` holds for these users to `to_holder`."""
    for user_id in user_ids:
        if _seats.get(key=user_id) is from_holder:
            _seats.set(key=user_id, value=to_holder)


def release_seats(user_ids: list[int], holder: SeatHolder) -> None:
    """Frees the seats `holder` holds for these users, leaving anyone seated elsewhere alone."""
    for user_id in user_ids:
        if _seats.get(key=user_id) is holder:
            _seats.pop(key=user_id)
