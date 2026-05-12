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


__all__ = ["BlackjackSettlement", "Card", "GameKind", "SettleOutcome", "WagerSettlement"]
