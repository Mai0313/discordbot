"""Pure rules for 射龍門 (In-Between / Acey Deucey).

The pot is a single jackpot row shared across every table of this game rather
than round-local state, so this module limits itself to rotation / pillar /
direction state and emits a signed `delta` per turn. It mutates no money: the
caller applies each `delta` to the player and its inverse to the pool, then
supplies the resulting jackpot back as the snapshot the legal bet range is
bounded against.
"""

from random import Random
from typing import Final, Literal

from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.games import Card, GameParticipant
from discordbot.typings.economy import MAX_SINGLE_BET
from discordbot.cogs.games.blackjack import draw_card

DragonGateDirection = Literal["higher", "lower"]
DragonGateOutcome = Literal[
    "gate_win", "outside_lose", "pillar_hit", "pair_win", "pair_lose", "pair_pillar_hit"
]

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
    """Raised when a high / low guess the gate does not offer is selected."""


class DragonGateBetRangeError(DragonGateError):
    """Raised when a bet is outside the current legal range."""


class DragonGateParticipantUnknownError(DragonGateError):
    """Raised when a withdraw or lookup targets a user not at the table."""


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


def has_open_gate(pillars: list[Card]) -> bool:
    """Returns whether the pillars produce a playable gate."""
    values = sorted(card_value(card=card) for card in pillars)
    return values[0] == values[1] or values[1] - values[0] > 1


class DragonGateTurn(BaseModel):
    """Frozen state for one active player's 射龍門 attempt.

    A pair-gate direction choice replaces the whole turn rather than mutating
    it; see `DragonGateRound.choose_pair_direction`.
    """

    model_config = ConfigDict(frozen=True)

    turn_number: int = Field(..., description="Sequence number of this turn within the table.")
    participant: GameParticipant = Field(..., description="Player taking this turn.")
    pillars: list[Card] = Field(..., description="The two gate pillar cards.")
    direction: DragonGateDirection | None = Field(
        default=None,
        description=(
            "High/low choice for a same-point gate, None until chosen; set at deal when the"
            " gate offers only one guess."
        ),
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

    @property
    def pair_directions(self) -> tuple[DragonGateDirection, ...]:
        """Returns the high/low guesses this gate offers: none off a pair, else each that can win.

        Nothing ranks below an ace or above a king, so an ace pair offers only higher and a king
        pair only lower.
        """
        if not self.is_pair:
            return ()
        if self.pillars[0].rank == "A":
            return ("higher",)
        if self.pillars[0].rank == "K":
            return ("lower",)
        return ("higher", "lower")


class DragonGateTurnResult(BaseModel):
    """Resolved result for one 射龍門 attempt."""

    model_config = ConfigDict(frozen=True)

    turn_number: int = Field(..., description="Sequence number of the resolved turn.")
    participant: GameParticipant = Field(..., description="Player whose attempt was resolved.")
    pillars: list[Card] = Field(..., description="The two gate pillar cards.")
    third_card: Card = Field(..., description="The third card drawn to resolve the bet.")
    bet: int = Field(..., description="Bet amount placed on this turn.")
    outcome: DragonGateOutcome = Field(..., description="Resolved outcome label for the turn.")
    delta: int = Field(..., description="Signed point change applied to the player's balance.")
    direction: DragonGateDirection | None = Field(
        default=None, description="High/low choice used for a same-point gate, if any."
    )


class DragonGatePlayerResult(BaseModel):
    """Final outcome for one player after a 射龍門 table closes.

    Each bet settles the moment it's placed, so the table close-out has no
    per-player wager settlement to apply; this model just captures the running
    totals and whether "逆贏不拿" was triggered for the leaver.
    """

    model_config = ConfigDict(frozen=True)

    participant: GameParticipant = Field(..., description="Player identity and ante metadata.")
    delta: int = Field(
        ...,
        description=(
            "Running win/loss for the table (ante excluded; ante was already pushed into "
            "the jackpot when the round started)."
        ),
    )
    final_balance: int = Field(
        ..., description="Player balance after the last settlement event touching this account."
    )
    withdrawn: bool = Field(
        ...,
        description=(
            "True when the player left before the table closed, by pressing leave or because a"
            " loss emptied their wallet."
        ),
    )
    refunded_to_pool: int = Field(
        default=0,
        description='Amount refunded into the jackpot under "逆贏不拿" when the player left while ahead.',
    )


class DragonGateRound(BaseModel):
    """Mutable 射龍門 table state with rotating turns over a shared jackpot."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rng: Random = Field(..., description="Random source used for card draws.")
    participants: list[GameParticipant] = Field(
        ..., description="Seated players in rotation order."
    )
    current_player_index: int = Field(
        default=0, description="Index of the participant whose turn is active."
    )
    turn_number: int = Field(default=0, description="Number of turns dealt so far.")
    active_turn: DragonGateTurn | None = Field(
        default=None, description="Turn awaiting a bet, None when the table is finished."
    )
    last_result: DragonGateTurnResult | None = Field(
        default=None, description="Most recently recorded turn result."
    )
    player_deltas: dict[int, int] = Field(
        default_factory=dict,
        description=(
            "In-memory running net delta per player since joining, ante excluded because it is"
            " already settled into the jackpot when the round starts."
        ),
    )
    withdrawn_user_ids: set[int] = Field(
        default_factory=set, description="User IDs of players who have left the table."
    )
    finished: bool = Field(default=False, description="True once every seat has withdrawn.")

    @classmethod
    def from_participants(
        cls, rng: Random, participants: list[GameParticipant]
    ) -> "DragonGateRound":
        """Builds a 射龍門 round from lobby participants and deals the first gate.

        Raises:
            ValueError: `participants` is empty.
        """
        if not participants:
            raise ValueError("At least one participant is required")
        round_state = cls(
            rng=rng,
            participants=participants,
            player_deltas={participant.user_id: 0 for participant in participants},
        )
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
        single wager.
        """
        return min(max(jackpot, 0), MAX_SINGLE_BET)

    def choose_pair_direction(self, user_id: int, direction: DragonGateDirection) -> None:
        """Stores the active player's high/low choice for a same-point gate."""
        active_turn = self._require_active_turn(user_id=user_id)
        if direction not in active_turn.pair_directions:
            raise DragonGatePairChoiceUnavailableError("This gate does not offer that direction")
        self.active_turn = active_turn.model_copy(update={"direction": direction})

    def needs_pair_choice(self) -> bool:
        """Returns whether the active player must choose higher or lower first."""
        return (
            self.active_turn is not None
            and self.active_turn.is_pair
            and self.active_turn.direction is None
        )

    def resolve_bet(self, user_id: int, amount: int, jackpot: int) -> DragonGateTurnResult:
        """Resolves the active player's bet by drawing the third card, booking nothing.

        The table moves only when the caller hands the result to `record_result`, so a bet whose
        settlement never lands leaves the round as it was.

        Args:
            user_id: Discord user ID that must match the active player.
            amount: Bet amount, constrained to the `current_min_bet` /
                `current_max_bet` range for the supplied jackpot.
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
        return DragonGateTurnResult(
            turn_number=active_turn.turn_number,
            participant=active_turn.participant,
            pillars=list(active_turn.pillars),
            third_card=third_card,
            bet=amount,
            outcome=outcome,
            delta=delta,
            direction=active_turn.direction,
        )

    def record_result(self, result: DragonGateTurnResult) -> None:
        """Books a settled bet into its player's running delta and deals the next turn.

        Call it only once the result's settlement has landed, carrying the delta that settlement
        applied.
        """
        self.player_deltas[result.participant.user_id] += result.delta
        self.last_result = result
        self._advance_to_next_active_turn()

    def player_delta(self, user_id: int) -> int:
        """Returns a player's cumulative net delta for the table."""
        return self.player_deltas.get(user_id, 0)

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

        A caller that claws back "逆贏不拿" writes it before calling this, so a
        clawback that never lands leaves the player seated.

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

    def _deal_next_turn(self) -> None:
        """Deals a new playable gate for the current participant.

        A pair that offers only one guess is dealt with that guess already made.
        """
        participant = self.participants[self.current_player_index]
        pillars = self._draw_open_gate_pillars()
        self.turn_number += 1
        turn = DragonGateTurn(
            turn_number=self.turn_number, participant=participant, pillars=pillars
        )
        if len(turn.pair_directions) == 1:
            turn = turn.model_copy(update={"direction": turn.pair_directions[0]})
        self.active_turn = turn

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
