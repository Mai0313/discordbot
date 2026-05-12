"""Pydantic models and literals for the casino games domain.

Pure data types live here so the rules engines in ``cogs/_games/`` and the
slash-command surface in ``cogs/games.py`` can share them without circular
imports. ``BlackjackHand`` is the only game-state type that intentionally
stays in ``cogs/_games/blackjack.py`` because it owns mutating rules methods
(``hit`` / ``stand`` / ``deal_initial``).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

DiceOutcome = Literal["win", "lose", "push"]
DragonGateOutcome = Literal["win", "lose", "push"]
SettleOutcome = Literal["win", "lose", "push", "blackjack", "player_bust", "dealer_bust"]
GameKind = Literal["dice", "blackjack", "dragon_gate"]


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


class DiceResult(BaseModel):
    """Result of one dice round.

    Attributes:
        player_rolls: Player's three rolls in order.
        dealer_rolls: Dealer's three rolls in order.
        player_total: Sum of the player rolls.
        dealer_total: Sum of the dealer rolls.
        outcome: ``win`` / ``lose`` / ``push`` from the player's perspective.
    """

    model_config = ConfigDict(frozen=True)

    player_rolls: tuple[int, ...]
    dealer_rolls: tuple[int, ...]
    player_total: int
    dealer_total: int
    outcome: DiceOutcome


class DragonGateResult(BaseModel):
    """Result of one Dragon Gate round.

    Attributes:
        first_gate: First gate card as drawn.
        second_gate: Second gate card as drawn.
        lower_gate: Lower-valued gate card.
        upper_gate: Higher-valued gate card.
        shot: The player's shot card.
        outcome: Player-facing settlement label.
    """

    model_config = ConfigDict(frozen=True)

    first_gate: Card
    second_gate: Card
    lower_gate: Card
    upper_gate: Card
    shot: Card
    outcome: DragonGateOutcome


class WagerSettlement(BaseModel):
    """Database-backed settlement result for a finished wager.

    Attributes:
        delta: Net point change relative to the withdrawn bet.
        payout: Gross amount credited back to the player after the upfront bet withdrawal.
        new_balance: Player balance after crediting the payout.
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


__all__ = [
    "BlackjackSettlement",
    "Card",
    "DiceOutcome",
    "DiceResult",
    "DragonGateOutcome",
    "DragonGateResult",
    "GameKind",
    "SettleOutcome",
    "WagerSettlement",
]
