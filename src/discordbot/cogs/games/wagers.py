"""Shared helpers for turning balances and table stakes into game participants."""

from typing import Literal

from discordbot.typings.games import GameParticipant, GameParticipantIdentity
from discordbot.typings.economy import MAX_SINGLE_BET
from discordbot.utils.amount_parsing import parse_decimal_amount

WagerMode = Literal["clamp", "exact"]


def parse_wager_amount(raw_amount: str | None) -> int | None:
    """Parses user-entered wager text; zero parses rather than being rejected.

    What zero means is the caller's rule, never this parser's.
    """
    return parse_decimal_amount(raw=raw_amount)


def build_wager_participant(
    identity: GameParticipantIdentity, balance: int, wager: int, mode: WagerMode
) -> GameParticipant | None:
    """Builds a participant for a table stake under the requested wager mode.

    `clamp` allows a lower-balance player to join by wagering their full balance.
    `exact` requires the player to cover the full wager, which is used for antes.

    Returns:
        The seated participant, or None when the wager or the balance is
        non-positive, or when `exact` mode meets a balance below the wager.
    """
    if wager <= 0 or balance <= 0:
        return None
    if mode == "exact" and balance < wager:
        return None

    bet = min(wager, balance)
    if mode == "clamp":
        # Exact-mode antes must be paid in full, so only a clamp-mode stake is
        # ever reduced to the cap.
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
