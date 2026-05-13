"""Shared helpers for turning balances and table stakes into game participants."""

from typing import Literal

from discordbot.typings.games import GameParticipant, GameParticipantIdentity

WagerMode = Literal["clamp", "exact"]


def build_wager_participant(
    *, identity: GameParticipantIdentity, balance: int, wager: int, mode: WagerMode
) -> GameParticipant | None:
    """Builds a participant for a table stake under the requested wager mode.

    ``clamp`` allows a lower-balance player to join by wagering their full balance.
    ``exact`` requires the player to cover the full wager, which is used for antes.
    """
    if wager <= 0 or balance <= 0:
        return None
    if mode == "exact" and balance < wager:
        return None

    bet = min(wager, balance)
    return GameParticipant(
        user_id=identity.user_id,
        account_name=identity.account_name,
        display_name=identity.display_name,
        avatar_url=identity.avatar_url,
        bet=bet,
        balance_at_start=balance,
        is_allin=bet == balance,
    )
