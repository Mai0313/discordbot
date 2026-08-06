"""Turns a wager and an observed balance into the seat a wagered table plays with.

This is the only place a `GameParticipant` is built, which makes it the one chokepoint for the
two rules every stake has to pass before a round can use it: nobody wagers more than the balance
that was just read, and a table bet never exceeds `MAX_SINGLE_BET`. Every seat at both wagered
tables comes through here — the Blackjack owner's stake, each Join, the bot player's own Kelly
bet, the re-check of every queued seat just before the deal, and 射龍門's fixed ante — so the cap
cannot be forgotten at one of them.

`WagerMode` is the difference between the two games' money shapes. A table stake is `clamp`: a
player short of the posted bet still gets a seat, wagering everything they hold. An ante is
`exact`: it is covered in full or the seat is refused, since 射龍門 charges it into the shared
jackpot as one transaction and a partial one would under-fund the pool the table pays out of.

Nothing here reads or writes a wallet. The balance arrives already read, and the seat is a frozen
snapshot of it, so a balance that moves afterwards cannot change the stake the table is playing
for.

It is a separate module rather than part of `cog.py` because `dragon_gate_views.py` parses bet
text too and `cog.py` already imports the views; it stays under `cogs/games/` rather than moving
down to `services/` because no second cog wagers.
"""

from typing import Literal

from discordbot.typings.games import GameParticipant, GameParticipantIdentity
from discordbot.typings.economy import MAX_SINGLE_BET
from discordbot.utils.amount_parsing import parse_decimal_amount

# How a wager is reconciled with a balance that cannot cover it: `clamp` seats the player at
# their whole balance, `exact` refuses the seat.
WagerMode = Literal["clamp", "exact"]


def parse_wager_amount(raw_amount: str | None) -> int | None:
    """Parses user-entered wager text into a non-negative amount.

    Zero survives parsing rather than being rejected here, because each table decides what it
    means: `/games blackjack` reads it as all in, while 射龍門's own legal-range check refuses
    it. Nothing else is range-checked either — `MAX_SINGLE_BET` is applied when the seat is
    built, and the caller answers a None with its own refusal before any money moves.

    Args:
        raw_amount (str | None): The raw bet text a slash option or a modal collected.

    Returns:
        The parsed amount, or None when the text was not plain decimal digits.
    """
    return parse_decimal_amount(raw=raw_amount)


def build_wager_participant(
    identity: GameParticipantIdentity, balance: int, wager: int, mode: WagerMode
) -> GameParticipant | None:
    """Fixes a stake against an observed balance and returns the seat it buys.

    A seat that cannot be built is None rather than an exception: callers turn it into the
    game's own refusal embed, or into a name on the list of players dropped at the deal. A
    non-positive wager or balance never seats anyone, in either mode.

    `is_allin` is measured against the balance, so a bet the cap reduced below what the player
    holds is not all in. The result is frozen, which is what lets a round settle at the stake it
    started with even though the wallet keeps moving underneath it.

    Args:
        identity (GameParticipantIdentity): The money-free half of the seat, already resolved.
        balance (int): Balance just read for this player, snapshotted onto the seat.
        wager (int): Stake being asked for, in economy points.
        mode (WagerMode): `clamp` for a table stake, `exact` for an ante.

    Returns:
        The seat, or None when the balance cannot support the wager under this mode.
    """
    if wager <= 0 or balance <= 0:
        return None
    if mode == "exact" and balance < wager:
        return None

    bet = min(wager, balance)
    if mode == "clamp":
        # MAX_SINGLE_BET caps player-chosen table bets so balances cannot compound
        # exponentially through repeated all-in doubling. Exact-mode antes must be
        # paid in full, so they are never reduced by the cap.
        bet = min(bet, MAX_SINGLE_BET)
    return GameParticipant(
        user_id=identity.user_id,
        account_name=identity.account_name,
        display_name=identity.display_name,
        avatar_url=identity.avatar_url,
        bet=bet,
        balance_at_start=balance,
        is_allin=bet == balance,
    )
