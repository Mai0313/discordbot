"""Shared helpers for turning balances and table stakes into game participants."""

from typing import Literal

from discordbot.typings.games import GameParticipant, GameParticipantIdentity
from discordbot.typings.economy import MAX_SINGLE_BET

WagerMode = Literal["clamp", "exact"]


def parse_wager_amount(raw_amount: str | None) -> int | None:
    """Parses user-entered wager text with optional comma separators."""
    normalized = (raw_amount or "").replace(",", "").strip()
    if not normalized.isdecimal():
        return None
    try:
        return int(normalized)
    except ValueError:
        return None


def build_wager_participant(
    identity: GameParticipantIdentity, balance: int, wager: int, mode: WagerMode
) -> GameParticipant | None:
    """Builds a participant for a table stake under the requested wager mode.

    `clamp` allows a lower-balance player to join by wagering their full balance.
    `exact` requires the player to cover the full wager, which is used for antes.
    """
    if wager <= 0 or balance <= 0:
        return None
    if mode == "exact" and balance < wager:
        return None

    # MAX_SINGLE_BET caps any single wager so balances cannot compound
    # exponentially through repeated all-in doubling.
    bet = min(wager, balance, MAX_SINGLE_BET)
    return GameParticipant(
        user_id=identity.user_id,
        account_name=identity.account_name,
        display_name=identity.display_name,
        avatar_url=identity.avatar_url,
        bet=bet,
        balance_at_start=balance,
        is_allin=bet == balance,
    )
