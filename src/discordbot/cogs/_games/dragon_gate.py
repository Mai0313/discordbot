"""Pure rules for 射龍門 (In-Between / Acey Deucey).

The round keeps the central pot in memory and records each player's cumulative
net delta. Database writes happen only when the Discord table finalizes, so a
bot restart discards an unfinished table instead of half-settling it.
"""

from random import Random
from typing import Literal

from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.games import Card, GameParticipant

DragonGateDirection = Literal["higher", "lower"]
DragonGateOutcome = Literal[
    "gate_win", "outside_lose", "pillar_hit", "pair_win", "pair_lose", "pair_pillar_hit"
]

RANKS: tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS: tuple[str, ...] = ("♠", "♥", "♦", "♣")


class DragonGateError(ValueError):
    """Base error for invalid 射龍門 rule operations."""


class DragonGateTableFinishedError(DragonGateError):
    """Raised when a caller tries to act after the table has finished."""


class DragonGateTurnError(DragonGateError):
    """Raised when a caller tries to act outside their active turn."""


class DragonGatePairChoiceRequiredError(DragonGateError):
    """Raised when a same-point gate needs high / low before betting."""


class DragonGatePairChoiceUnavailableError(DragonGateError):
    """Raised when high / low is selected for a non-pair gate."""


class DragonGateBetRangeError(DragonGateError):
    """Raised when a bet is outside the current legal range."""


def draw_card(*, rng: Random) -> Card:
    """Draws one card from a notional infinite shoe."""
    return Card(rank=rng.choice(seq=RANKS), suit=rng.choice(seq=SUITS))


def card_value(*, card: Card) -> int:
    """Returns the 射龍門 point value for a card, with Ace low."""
    if card.rank == "A":
        return 1
    if card.rank == "J":
        return 11
    if card.rank == "Q":
        return 12
    if card.rank == "K":
        return 13
    return int(card.rank)


def render_cards(*, cards: list[Card]) -> str:
    """Formats cards for display."""
    return " ".join(str(card) for card in cards)


class DragonGateTurn(BaseModel):
    """Mutable state for one active player's 射龍門 attempt."""

    model_config = ConfigDict(frozen=True)

    turn_number: int
    participant: GameParticipant
    pillars: list[Card]
    direction: DragonGateDirection | None = None

    @property
    def is_pair(self) -> bool:
        """Returns whether the two pillar cards have the same point value."""
        return card_value(card=self.pillars[0]) == card_value(card=self.pillars[1])

    @property
    def lower_value(self) -> int:
        """Returns the lower pillar point value."""
        return min(card_value(card=self.pillars[0]), card_value(card=self.pillars[1]))

    @property
    def upper_value(self) -> int:
        """Returns the higher pillar point value."""
        return max(card_value(card=self.pillars[0]), card_value(card=self.pillars[1]))


class DragonGateTurnResult(BaseModel):
    """Resolved result for one 射龍門 attempt."""

    model_config = ConfigDict(frozen=True)

    turn_number: int
    participant: GameParticipant
    pillars: list[Card]
    third_card: Card
    bet: int
    outcome: DragonGateOutcome
    delta: int
    pot_after: int
    direction: DragonGateDirection | None = None


class DragonGateRound(BaseModel):
    """Mutable 射龍門 table state with a central pot and rotating turns."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rng: Random
    participants: list[GameParticipant]
    ante: int
    pot: int = 0
    current_player_index: int = 0
    turn_number: int = 0
    active_turn: DragonGateTurn | None = None
    last_result: DragonGateTurnResult | None = None
    player_deltas: dict[int, int] = Field(default_factory=dict)
    finished: bool = False

    @classmethod
    def from_participants(
        cls, *, rng: Random, participants: list[GameParticipant], ante: int
    ) -> "DragonGateRound":
        """Builds and starts a 射龍門 round from lobby participants."""
        if ante <= 0:
            raise ValueError("Ante must be positive")
        if not participants:
            raise ValueError("At least one participant is required")
        round_state = cls(rng=rng, participants=participants, ante=ante)
        round_state.pot = ante * len(participants)
        round_state.player_deltas = {participant.user_id: -ante for participant in participants}
        round_state._deal_next_turn()
        return round_state

    def current_min_bet(self) -> int:
        """Returns the minimum legal bet for the active turn."""
        if self.pot <= 0:
            return 0
        return min(self.ante, self.pot)

    def current_max_bet(self) -> int:
        """Returns the maximum legal bet for the active turn."""
        return max(self.pot, 0)

    def choose_pair_direction(self, *, user_id: int, direction: DragonGateDirection) -> None:
        """Stores the active player's high/low choice for a same-point gate."""
        active_turn = self._require_active_turn(user_id=user_id)
        if not active_turn.is_pair:
            raise DragonGatePairChoiceUnavailableError("This turn is not a pair")
        self.active_turn = active_turn.model_copy(update={"direction": direction})

    def needs_pair_choice(self) -> bool:
        """Returns whether the active player must choose higher or lower first."""
        return (
            self.active_turn is not None
            and self.active_turn.is_pair
            and self.active_turn.direction is None
        )

    def place_bet(self, *, user_id: int, amount: int) -> DragonGateTurnResult:
        """Resolves the active player's bet by drawing the third card.

        Args:
            user_id: Discord user ID that must match the active player.
            amount: Bet amount, constrained to current min bet..pot.

        Returns:
            The resolved turn result.

        Raises:
            DragonGateError: The table is finished, it is not this user's turn,
                the pair direction is missing, or the bet is outside the legal range.
        """
        active_turn = self._require_active_turn(user_id=user_id)
        if self.needs_pair_choice():
            raise DragonGatePairChoiceRequiredError("Pair direction is required")

        minimum = self.current_min_bet()
        maximum = self.current_max_bet()
        if amount < minimum or amount > maximum:
            raise DragonGateBetRangeError("Bet outside legal range")

        third_card = draw_card(rng=self.rng)
        outcome, delta = self._resolve_turn(turn=active_turn, third_card=third_card, amount=amount)
        self.pot -= delta
        self.player_deltas[active_turn.participant.user_id] += delta
        result = DragonGateTurnResult(
            turn_number=active_turn.turn_number,
            participant=active_turn.participant,
            pillars=list(active_turn.pillars),
            third_card=third_card,
            bet=amount,
            outcome=outcome,
            delta=delta,
            pot_after=self.pot,
            direction=active_turn.direction,
        )
        self.last_result = result
        if self.pot <= 0:
            self.pot = 0
            self.finished = True
            self.active_turn = None
        else:
            self.current_player_index = (self.current_player_index + 1) % len(self.participants)
            self._deal_next_turn()
        return result

    def player_delta(self, *, user_id: int) -> int:
        """Returns a player's cumulative net delta for the table."""
        return self.player_deltas.get(user_id, 0)

    def _deal_next_turn(self) -> None:
        self.turn_number += 1
        participant = self.participants[self.current_player_index]
        self.active_turn = DragonGateTurn(
            turn_number=self.turn_number,
            participant=participant,
            pillars=[draw_card(rng=self.rng), draw_card(rng=self.rng)],
        )

    def _require_active_turn(self, *, user_id: int) -> DragonGateTurn:
        if self.finished or self.active_turn is None:
            raise DragonGateTableFinishedError("Table is finished")
        if self.active_turn.participant.user_id != user_id:
            raise DragonGateTurnError("Not this player's turn")
        return self.active_turn

    def _resolve_turn(
        self, *, turn: DragonGateTurn, third_card: Card, amount: int
    ) -> tuple[DragonGateOutcome, int]:
        third_value = card_value(card=third_card)
        if turn.is_pair:
            return self._resolve_pair_turn(turn=turn, third_value=third_value, amount=amount)
        if third_value in (turn.lower_value, turn.upper_value):
            return "pillar_hit", -amount * 2
        if turn.lower_value < third_value < turn.upper_value:
            return "gate_win", amount
        return "outside_lose", -amount

    def _resolve_pair_turn(
        self, *, turn: DragonGateTurn, third_value: int, amount: int
    ) -> tuple[DragonGateOutcome, int]:
        pillar_value = turn.lower_value
        if third_value == pillar_value:
            return "pair_pillar_hit", -amount * 3
        if turn.direction == "higher" and third_value > pillar_value:
            return "pair_win", amount
        if turn.direction == "lower" and third_value < pillar_value:
            return "pair_win", amount
        return "pair_lose", -amount


__all__ = [
    "DragonGateBetRangeError",
    "DragonGateDirection",
    "DragonGateError",
    "DragonGateOutcome",
    "DragonGatePairChoiceRequiredError",
    "DragonGatePairChoiceUnavailableError",
    "DragonGateRound",
    "DragonGateTableFinishedError",
    "DragonGateTurn",
    "DragonGateTurnError",
    "DragonGateTurnResult",
    "card_value",
    "draw_card",
    "render_cards",
]
