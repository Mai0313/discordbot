"""Pydantic models and literals for the casino games domain.

Pure data types live here so the rules engines in ``cogs/_games/`` and the
slash-command surface in ``cogs/games.py`` can share them without circular
imports. ``BlackjackHand`` is the only game-state type that intentionally
stays in ``cogs/_games/blackjack.py`` because it owns mutating rules methods
(``hit`` / ``stand`` / ``deal_initial``).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

SettleOutcome = Literal["win", "lose", "push", "blackjack", "player_bust", "dealer_bust"]
GameKind = Literal["blackjack"]


class Card(BaseModel):
    """A single playing card.

    Attributes:
        rank: One of A, 2-10, J, Q, K.
        suit: One of the four unicode suit glyphs.
    """

    model_config = ConfigDict(frozen=True)

    rank: str
    suit: str

    def __str__(self) -> str:
        """Human-readable label like ``A♠``."""
        return f"{self.rank}{self.suit}"


class GameParticipant(BaseModel):
    """A Discord user registered for a casino game session.

    Attributes:
        user_id: Discord user ID for the account row and interaction checks.
        account_name: Stable Discord username stored in the economy account row.
        display_name: Guild-aware display name shown in game embeds.
        avatar_url: Last-seen Discord avatar URL for the economy account row.
        bet: Effective wager for this player after all-in clamping.
        balance_at_start: Balance observed when the game session starts.
        is_allin: True when the requested bet was clamped to the player's balance.
    """

    model_config = ConfigDict(frozen=True)

    user_id: int
    account_name: str
    display_name: str
    avatar_url: str = ""
    bet: int
    balance_at_start: int
    is_allin: bool


class WagerSettlement(BaseModel):
    """Database-backed settlement result for a finished wager.

    Attributes:
        delta: Net point change for the round.
        payout: Positive player credit from the round, excluding losses and pushes.
        new_balance: Player balance after applying the signed round delta.
        house_balance: Dealer ledger balance after mirroring the player's net change.
    """

    model_config = ConfigDict(frozen=True)

    delta: int
    payout: int
    new_balance: int
    house_balance: int


class BlackjackSettlement(WagerSettlement):
    """Database-backed settlement result for a finished Blackjack round.

    Attributes:
        outcome: Player-facing outcome label from the Blackjack rules engine.
        detail: Short game-state summary for the dealer AI prompt.
    """

    outcome: SettleOutcome
    detail: str


class BlackjackPlayerResult(BaseModel):
    """Settlement result for one player at a Blackjack table.

    Attributes:
        participant: Player identity and wager metadata.
        settlement: Database-backed result for that player's hand.
    """

    model_config = ConfigDict(frozen=True)

    participant: GameParticipant
    settlement: BlackjackSettlement


__all__ = [
    "BlackjackPlayerResult",
    "BlackjackSettlement",
    "Card",
    "GameKind",
    "GameParticipant",
    "SettleOutcome",
    "WagerSettlement",
]
