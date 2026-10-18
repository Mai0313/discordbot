"""Shared helpers for turning balances and table stakes into game participants."""

from typing import Literal

from discordbot.typings.games import GameParticipant, GameParticipantIdentity
from discordbot.typings.economy import MAX_SINGLE_BET

WagerMode = Literal["clamp", "exact"]


def build_wager_participant(
    identity: GameParticipantIdentity, balance: int, wager: int, mode: WagerMode
) -> GameParticipant | None:
    """Returns a seated participant for `wager` under `mode`, or None when ineligible.

    `clamp` lets a lower-balance player join by wagering their full balance, capped at
    `MAX_SINGLE_BET`; `exact` requires `balance >= wager`. Returns None when `wager` or
    `balance` is non-positive, or when `exact` mode meets a balance below `wager`.

    Args:
        identity (GameParticipantIdentity): Discord user identity and display metadata.
        balance (int): Current wallet balance of the user.
        wager (int): Requested stake or ante amount.
        mode (WagerMode): `"clamp"` to cap at balance and `MAX_SINGLE_BET`, or `"exact"`
            to require the full wager.
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
