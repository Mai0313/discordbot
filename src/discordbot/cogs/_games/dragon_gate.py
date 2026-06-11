"""Pure rules for 射龍門 (In-Between / Acey Deucey).

The pot is no longer round-local: a single jackpot row in
`data/database/economy.db` is shared across every table of this game, so this
module limits itself to rotation / pillar / direction state and emits a
signed `delta` per turn. The view layer applies the delta atomically
against the player row and the jackpot pool, then passes the updated
pool back so `current_min_bet` / `current_max_bet` reflect the
post-settlement value.
"""

from random import Random
from typing import Final, Literal

from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.games import Card, GameParticipant
from discordbot.typings.economy import MAX_SINGLE_BET

DragonGateDirection = Literal["higher", "lower"]
DragonGateOutcome = Literal[
    "gate_win", "outside_lose", "pillar_hit", "pair_win", "pair_lose", "pair_pillar_hit"
]

RANKS: tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS: tuple[str, ...] = ("♠", "♥", "♦", "♣")

GAME_ID: Final[str] = "dragon_gate"
ANTE: Final[int] = 10
MIN_BET: Final[int] = 20


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


class DragonGateParticipantUnknownError(DragonGateError):
    """Raised when a withdraw or lookup targets a user not at the table."""


def draw_card(rng: Random) -> Card:
    """Draws one card from a notional infinite shoe."""
    return Card(rank=rng.choice(seq=RANKS), suit=rng.choice(seq=SUITS))


def card_value(card: Card) -> int:
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


def render_cards(cards: list[Card]) -> str:
    """Formats cards for display."""
    return " ".join(str(card) for card in cards)


def has_open_gate(pillars: list[Card]) -> bool:
    """Returns whether the pillars produce a playable gate."""
    values = sorted(card_value(card=card) for card in pillars)
    return values[0] == values[1] or values[1] - values[0] > 1


class DragonGateTurn(BaseModel):
    """Mutable state for one active player's 射龍門 attempt."""

    model_config = ConfigDict(frozen=True)

    turn_number: int = Field(description="Sequence number of this turn within the table.")
    participant: GameParticipant = Field(description="Player taking this turn.")
    pillars: list[Card] = Field(description="The two gate pillar cards.")
    direction: DragonGateDirection | None = Field(
        default=None, description="High/low choice for a same-point gate, None until chosen."
    )

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

    turn_number: int = Field(description="Sequence number of the resolved turn.")
    participant: GameParticipant = Field(description="Player whose attempt was resolved.")
    pillars: list[Card] = Field(description="The two gate pillar cards.")
    third_card: Card = Field(description="The third card drawn to resolve the bet.")
    bet: int = Field(description="Bet amount placed on this turn.")
    outcome: DragonGateOutcome = Field(description="Resolved outcome label for the turn.")
    delta: int = Field(description="Signed point change applied to the player's balance.")
    direction: DragonGateDirection | None = Field(
        default=None, description="High/low choice used for a same-point gate, if any."
    )


class DragonGateRound(BaseModel):
    """Mutable 射龍門 table state with rotating turns over a shared jackpot.

    `player_deltas` is the **in-memory** running total of each player's
    wins minus losses since they joined the table (ante excluded; ante is
    already settled into the jackpot when the round starts). The view
    layer reads this on withdraw / timeout to decide whether to apply the
    "逆贏不拿" refund (clawing winnings back into the jackpot when a
    player leaves while ahead).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rng: Random = Field(description="Random source used for card draws.")
    participants: list[GameParticipant] = Field(description="Seated players in rotation order.")
    current_player_index: int = Field(
        default=0, description="Index of the participant whose turn is active."
    )
    turn_number: int = Field(default=0, description="Number of turns dealt so far.")
    active_turn: DragonGateTurn | None = Field(
        default=None, description="Turn awaiting a bet, None when the table is finished."
    )
    last_result: DragonGateTurnResult | None = Field(
        default=None, description="Most recently resolved turn result."
    )
    player_deltas: dict[int, int] = Field(
        default_factory=dict,
        description="In-memory running net delta per player since joining, ante excluded.",
    )
    withdrawn_user_ids: set[int] = Field(
        default_factory=set, description="User IDs of players who have left the table."
    )
    finished: bool = Field(default=False, description="True once every seat has withdrawn.")

    @classmethod
    def from_participants(
        cls, rng: Random, participants: list[GameParticipant]
    ) -> "DragonGateRound":
        """Builds and starts a 射龍門 round from lobby participants."""
        if not participants:
            raise ValueError("At least one participant is required")
        round_state = cls(rng=rng, participants=participants)
        round_state.player_deltas = {participant.user_id: 0 for participant in participants}
        round_state._deal_next_turn()
        return round_state

    def current_min_bet(self, jackpot: int) -> int:
        """Returns the minimum legal bet given the live jackpot snapshot."""
        if jackpot <= 0:
            return 0
        return min(MIN_BET, jackpot)

    def current_max_bet(self, jackpot: int) -> int:
        """Returns the maximum legal bet given the live jackpot snapshot.

        Capped by `MAX_SINGLE_BET` so a large pool cannot fund an unbounded
        single wager; the view layer further clamps to the player's balance.
        """
        return min(max(jackpot, 0), MAX_SINGLE_BET)

    def choose_pair_direction(self, user_id: int, direction: DragonGateDirection) -> None:
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

    def place_bet(self, user_id: int, amount: int, jackpot: int) -> DragonGateTurnResult:
        """Resolves the active player's bet by drawing the third card.

        The jackpot itself lives in the database and is not mutated here;
        the caller passes the current snapshot so the legal bet range can
        be enforced. The returned `delta` is the signed change applied
        to the player's balance (and inverted into the jackpot) by the
        caller; this method only updates rotation state.

        Args:
            user_id: Discord user ID that must match the active player.
            amount: Bet amount, constrained to current min bet..jackpot.
            jackpot: Live jackpot balance used to bound the legal bet
                range; tracked outside this module.

        Returns:
            The resolved turn result.

        Raises:
            DragonGateError: The table is finished, it is not this user's
                turn, the pair direction is missing, or the bet is outside
                the legal range.
        """
        active_turn = self._require_active_turn(user_id=user_id)
        if self.needs_pair_choice():
            raise DragonGatePairChoiceRequiredError("Pair direction is required")

        minimum = self.current_min_bet(jackpot=jackpot)
        maximum = self.current_max_bet(jackpot=jackpot)
        if amount < minimum or amount > maximum:
            raise DragonGateBetRangeError("Bet outside legal range")

        third_card = draw_card(rng=self.rng)
        outcome, delta = self._resolve_turn(turn=active_turn, third_card=third_card, amount=amount)
        self.player_deltas[active_turn.participant.user_id] += delta
        result = DragonGateTurnResult(
            turn_number=active_turn.turn_number,
            participant=active_turn.participant,
            pillars=list(active_turn.pillars),
            third_card=third_card,
            bet=amount,
            outcome=outcome,
            delta=delta,
            direction=active_turn.direction,
        )
        self.last_result = result
        self._advance_to_next_active_turn()
        return result

    def player_delta(self, user_id: int) -> int:
        """Returns a player's cumulative net delta for the table."""
        return self.player_deltas.get(user_id, 0)

    def replace_last_result_delta(self, user_id: int, delta: int) -> DragonGateTurnResult:
        """Replaces the latest result delta after database-side clamping."""
        result = self.last_result
        if result is None or result.participant.user_id != user_id:
            raise DragonGateParticipantUnknownError("No latest result for user")
        previous_delta = result.delta
        self.player_deltas[user_id] += delta - previous_delta
        adjusted = result.model_copy(update={"delta": delta})
        self.last_result = adjusted
        return adjusted

    def is_active(self, user_id: int) -> bool:
        """Returns whether the given user is still seated and not withdrawn."""
        return (
            any(participant.user_id == user_id for participant in self.participants)
            and user_id not in self.withdrawn_user_ids
        )

    def active_participants(self) -> list[GameParticipant]:
        """Returns participants who have not withdrawn from the table yet."""
        return [
            participant
            for participant in self.participants
            if participant.user_id not in self.withdrawn_user_ids
        ]

    def withdraw(self, user_id: int) -> int:
        """Removes a player from the rotation and returns their running delta.

        The caller is responsible for the financial side of "逆贏不拿":
        when the returned delta is positive, the view layer pushes that
        many points back into the jackpot. The rotation skips to the
        next non-withdrawn player; if every seat is withdrawn the round
        is marked finished.

        Args:
            user_id: Discord user ID of the leaver.

        Returns:
            The leaver's running delta at the moment of withdrawal.

        Raises:
            DragonGateParticipantUnknownError: `user_id` is not seated
                at this table or has already withdrawn.
        """
        if not self.is_active(user_id=user_id):
            raise DragonGateParticipantUnknownError("User is not active at this table")
        self.withdrawn_user_ids.add(user_id)
        delta = self.player_deltas.get(user_id, 0)
        if not self.active_participants():
            self.finished = True
            self.active_turn = None
            return delta
        if self.active_turn is not None and self.active_turn.participant.user_id == user_id:
            self._advance_to_next_active_turn()
        return delta

    def _advance_to_next_active_turn(self) -> None:
        """Advances the cursor to the next non-withdrawn participant."""
        if not self.active_participants():
            self.finished = True
            self.active_turn = None
            return
        seats = len(self.participants)
        for _ in range(seats):
            self.current_player_index = (self.current_player_index + 1) % seats
            if self.participants[self.current_player_index].user_id not in self.withdrawn_user_ids:
                self._deal_next_turn()
                return
        self.finished = True
        self.active_turn = None

    def _deal_next_turn(self) -> None:
        """Deals a new playable gate for the current participant."""
        participant = self.participants[self.current_player_index]
        pillars = self._draw_open_gate_pillars()
        self.turn_number += 1
        self.active_turn = DragonGateTurn(
            turn_number=self.turn_number, participant=participant, pillars=pillars
        )

    def _draw_open_gate_pillars(self) -> list[Card]:
        """Draws pillar cards until the pair or gap creates a legal gate."""
        while True:
            pillars = [draw_card(rng=self.rng), draw_card(rng=self.rng)]
            if has_open_gate(pillars=pillars):
                return pillars

    def _require_active_turn(self, user_id: int) -> DragonGateTurn:
        """Returns the active turn or raises the matching rule error."""
        if self.finished or self.active_turn is None:
            raise DragonGateTableFinishedError("Table is finished")
        if self.active_turn.participant.user_id != user_id:
            raise DragonGateTurnError("Not this player's turn")
        return self.active_turn

    def _resolve_turn(
        self, turn: DragonGateTurn, third_card: Card, amount: int
    ) -> tuple[DragonGateOutcome, int]:
        """Resolves a non-pair or pair gate into outcome and player delta."""
        third_value = card_value(card=third_card)
        if turn.is_pair:
            return self._resolve_pair_turn(turn=turn, third_value=third_value, amount=amount)
        if third_value in (turn.lower_value, turn.upper_value):
            return "pillar_hit", -amount * 2
        if turn.lower_value < third_value < turn.upper_value:
            return "gate_win", amount
        return "outside_lose", -amount

    def _resolve_pair_turn(
        self, turn: DragonGateTurn, third_value: int, amount: int
    ) -> tuple[DragonGateOutcome, int]:
        """Resolves a same-point gate using the selected high or low direction."""
        pillar_value = turn.lower_value
        if third_value == pillar_value:
            return "pair_pillar_hit", -amount * 3
        if turn.direction == "higher" and third_value > pillar_value:
            return "pair_win", amount
        if turn.direction == "lower" and third_value < pillar_value:
            return "pair_win", amount
        return "pair_lose", -amount


__all__ = [
    "ANTE",
    "GAME_ID",
    "MIN_BET",
    "DragonGateBetRangeError",
    "DragonGateDirection",
    "DragonGateError",
    "DragonGateOutcome",
    "DragonGatePairChoiceRequiredError",
    "DragonGatePairChoiceUnavailableError",
    "DragonGateParticipantUnknownError",
    "DragonGateRound",
    "DragonGateTableFinishedError",
    "DragonGateTurn",
    "DragonGateTurnError",
    "DragonGateTurnResult",
    "card_value",
    "draw_card",
    "has_open_gate",
    "render_cards",
]
