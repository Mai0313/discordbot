"""Pure rules for 射龍門 (In-Between / Acey Deucey), the table behind `/games dragon_gate`.

Side-effect free: no Discord object, no database, no clock and no LLM, so a whole table replays
from the `random.Random` it is handed (production passes `random.SystemRandom()`, the tests a
scripted one). The Discord surface is `dragon_gate_views.py`, the lobby it starts from is
`lobby.py::BaseJackpotLobbyView`, and the money is the shared `jackpot_pool` row keyed on
`GAME_ID` in `data/database/economy.db`, moved by
`services/economy/database.py::apply_jackpot_settlement`.

The pot is not round-local: that one jackpot row is shared across every table of this game, so
this module limits itself to rotation / pillar / direction state and emits a signed `delta` per
turn. The view applies the delta atomically against the player row and the pool, then passes the
post-settlement pool back so `current_min_bet` / `current_max_bet` bound the next bet by what the
pool can actually pay. The database has the last word on the figure (a loss clamps at the
player's balance, and a win against a stale pool generation applies as zero), so what it really
applied comes back through `replace_last_result_delta`, and the running totals kept here are
corrected to it rather than being authoritative.

The house rules this file encodes:

- Pillars are redealt until they make a gate. Two adjacent ranks leave nothing to shoot at; a
  same-point pair is a gate of its own kind. A redeal costs no turn number, so `turn_number`
  counts gates offered rather than pairs dealt.
- On an ordinary gate the third card pays +1x the bet inside the pillars, -1x outside them, and
  -2x on either pillar itself.
- A same-point gate is a higher/lower call made before the bet: +1x right, -1x wrong, and -3x
  when the third card is that same point again.

Two pieces of money a reader will not find here. The ante (`ANTE`) is charged into the pool by
the lobby before the first gate is dealt, so `player_deltas` never carries it. And the "逆贏不拿"
clawback, which pushes a leaver's winnings back into the pool, is only prepared here: `withdraw`
hands back the running delta and the view decides what to do with it.
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

# Keys the shared `jackpot_pool` row every table of this game settles against.
GAME_ID: Final[str] = "dragon_gate"
# Charged into that pool per seat by the lobby before the first gate is dealt.
ANTE: Final[int] = 10
# Table floor, which `current_min_bet` follows down when the pool holds less than this.
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
    """Draws one card from a notional infinite shoe.

    Rank and suit are chosen independently and nothing is ever removed, so every gate faces the
    same fixed distribution. The Blackjack table next door keeps a finite shoe across rounds
    precisely so card counting has signal; this game has no such shoe to count.

    Args:
        rng (Random): Random source used to choose rank and suit.

    Returns:
        The drawn card.
    """
    return Card(rank=rng.choice(seq=RANKS), suit=rng.choice(seq=SUITS))


def card_value(card: Card) -> int:
    """Returns the 射龍門 point value for a card, with Ace low.

    Ace is 1 here rather than Blackjack's 11-or-1, so every rank sits at one fixed spot on a
    1-13 ladder and a gate's width is plain subtraction.

    Args:
        card (Card): Card to score.

    Returns:
        The card's point value, 1 for Ace through 13 for King.
    """
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
    """Joins card labels into one display string.

    Exported but unused: `dragon_gate_views.py` builds the same string inline everywhere it
    shows pillars.

    Args:
        cards (list[Card]): Cards to render, in the order they should appear.

    Returns:
        The card labels separated by single spaces.
    """
    return " ".join(str(card) for card in cards)


def has_open_gate(pillars: list[Card]) -> bool:
    """Returns whether the pillars produce a playable gate.

    Two adjacent ranks have no gate between them and are redealt by `_draw_open_gate_pillars`
    instead of being offered. A same-point pair passes as its own kind of gate: it is played as
    a higher/lower call rather than as a shot between the pillars.

    Args:
        pillars (list[Card]): The two pillar cards, in either order.

    Returns:
        True when the pillars are a same-point pair or leave at least one rank between them.
    """
    values = sorted(card_value(card=card) for card in pillars)
    return values[0] == values[1] or values[1] - values[0] > 1


class DragonGateTurn(BaseModel):
    """The gate one player is currently facing, before any bet resolves it.

    Frozen, so recording the higher/lower call replaces `DragonGateRound.active_turn` with a
    copy rather than mutating the turn a caller may already be holding.
    """

    model_config = ConfigDict(frozen=True)

    turn_number: int = Field(..., description="Sequence number of this turn within the table.")
    participant: GameParticipant = Field(..., description="Player taking this turn.")
    pillars: list[Card] = Field(..., description="The two gate pillar cards.")
    direction: DragonGateDirection | None = Field(
        default=None, description="High/low choice for a same-point gate, None until chosen."
    )

    @property
    def is_pair(self) -> bool:
        """Returns whether the two pillars share a point value, the one shape needing a call.

        Ranks are compared by point value, so 10 and any face card are still distinct pillars.
        """
        return card_value(card=self.pillars[0]) == card_value(card=self.pillars[1])

    @property
    def lower_value(self) -> int:
        """Returns the lower pillar point value, which on a pair is both of them."""
        return min(card_value(card=self.pillars[0]), card_value(card=self.pillars[1]))

    @property
    def upper_value(self) -> int:
        """Returns the higher pillar point value, which on a pair is both of them."""
        return max(card_value(card=self.pillars[0]), card_value(card=self.pillars[1]))


class DragonGateTurnResult(BaseModel):
    """One resolved 射龍門 attempt: the gate, the card that answered it, and the signed delta.

    Frozen, so the correction the database forces takes a `model_copy` through
    `DragonGateRound.replace_last_result_delta`; the view keeps these in a history list and
    renders them long after the turn ended.
    """

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


class DragonGateRound(BaseModel):
    """Mutable 射龍門 table state: the seats, the rotation cursor, and the gate in play.

    The one unfrozen model here, and the only one holding a `random.Random`, which is what
    `arbitrary_types_allowed` is for.

    `player_deltas` is the **in-memory** running total of each player's wins minus losses since
    they joined the table (ante excluded; the ante is already in the jackpot before the first
    gate is dealt). The view layer reads it on withdraw / timeout to decide whether to apply the
    "逆贏不拿" refund, clawing winnings back into the jackpot when a player leaves while ahead,
    so it is the only trace of an obligation nothing has settled yet.

    A seat leaves the rotation only through `withdraw`. Busting out is the view's call, made
    after the wallet write reports a zero balance, rather than a rule decided here.
    """

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
        """Builds a round with every seat at zero and the first gate already dealt.

        Dealing here rather than on the first interaction is what lets the lobby edit its
        message straight into a table that already shows a gate.

        Args:
            rng (Random): Random source the whole table draws from.
            participants (list[GameParticipant]): Seated players, in rotation order.

        Returns:
            The started round, with `active_turn` on the first participant.

        Raises:
            ValueError: `participants` is empty, so there is nobody to deal to.
        """
        if not participants:
            raise ValueError("At least one participant is required")
        round_state = cls(rng=rng, participants=participants)
        round_state.player_deltas = {participant.user_id: 0 for participant in participants}
        round_state._deal_next_turn()
        return round_state

    def current_min_bet(self, jackpot: int) -> int:
        """Returns the minimum legal bet given the live jackpot snapshot.

        The floor follows a thin pool down instead of holding at `MIN_BET`: a win is paid out of
        the pool, so a fixed floor above what the pool holds would sit above `current_max_bet`
        and leave the turn with no legal bet at all.

        Args:
            jackpot (int): Live jackpot balance, snapshotted by the caller from the database.

        Returns:
            `MIN_BET`, or the whole pool when it holds less, or 0 when it is empty.
        """
        if jackpot <= 0:
            return 0
        return min(MIN_BET, jackpot)

    def current_max_bet(self, jackpot: int) -> int:
        """Returns the maximum legal bet given the live jackpot snapshot.

        Capped by `MAX_SINGLE_BET` so a large pool cannot fund an unbounded single wager. The
        view layer clamps this again to the player's own balance, since a loss clamps there and
        an unclamped ceiling would let a near-broke seat risk its wallet to win the whole pool.

        Args:
            jackpot (int): Live jackpot balance, snapshotted by the caller from the database.

        Returns:
            The pool balance, floored at 0 and capped at `MAX_SINGLE_BET`.
        """
        return min(max(jackpot, 0), MAX_SINGLE_BET)

    def choose_pair_direction(self, user_id: int, direction: DragonGateDirection) -> None:
        """Records the active player's higher/lower call for a same-point gate.

        Replaces `active_turn` with a copy, since the turn model is frozen. The seat and table
        checks are `_require_active_turn`'s, so a finished table or a caller acting out of turn
        raises from there.

        Args:
            user_id (int): Discord user ID, which must be the active player's.
            direction (DragonGateDirection): The call to record for this gate.

        Raises:
            DragonGatePairChoiceUnavailableError: The active gate is not a same-point pair, so
                there is nothing to call.
        """
        active_turn = self._require_active_turn(user_id=user_id)
        if not active_turn.is_pair:
            raise DragonGatePairChoiceUnavailableError("This turn is not a pair")
        self.active_turn = active_turn.model_copy(update={"direction": direction})

    def needs_pair_choice(self) -> bool:
        """Returns whether the active player still owes a higher/lower call before betting.

        False on a finished table too, so a caller can read it without first checking that a
        turn is live.
        """
        return (
            self.active_turn is not None
            and self.active_turn.is_pair
            and self.active_turn.direction is None
        )

    def place_bet(self, user_id: int, amount: int, jackpot: int) -> DragonGateTurnResult:
        """Resolves the active player's bet by drawing the third card, then rotates the table.

        The jackpot lives in the database and is not touched here: the caller passes a snapshot
        so the legal range can be enforced, and applies the returned `delta` to the player row
        and its inverse to the pool afterwards. When that write comes back with a different
        figure, hand it to `replace_last_result_delta` rather than re-reading the result.

        The rotation moves on whatever the outcome was, so the returned result is the only
        handle on the turn that just ended. The seat and table checks are
        `_require_active_turn`'s, so a finished table or a caller acting out of turn raises from
        there.

        Args:
            user_id (int): Discord user ID, which must be the active player's.
            amount (int): Bet amount, which must fall between `current_min_bet` and
                `current_max_bet` for this snapshot.
            jackpot (int): Live jackpot balance, snapshotted by the caller from the database.

        Returns:
            The resolved turn, already recorded as `last_result` and added into `player_deltas`.

        Raises:
            DragonGatePairChoiceRequiredError: The gate is a same-point pair with no call made.
            DragonGateBetRangeError: The amount is outside the range this snapshot allows.
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
        """Returns a player's running net delta for the table, 0 for anyone unseated.

        Args:
            user_id (int): Discord user ID to look up.

        Returns:
            Wins minus losses since the player joined, ante excluded.
        """
        return self.player_deltas.get(user_id, 0)

    def replace_last_result_delta(self, user_id: int, delta: int) -> DragonGateTurnResult:
        """Rewrites the latest result with the delta the database actually applied.

        The wallet write has the last word: a loss clamps at the player's balance, and a win
        against a stale pool generation applies as zero. Both leave the figure this file
        computed too large, so the running total is corrected by the difference rather than
        overwritten; the 逆贏不拿 refund and the scoreboard would otherwise hand back money that
        never moved.

        Args:
            user_id (int): Discord user ID the latest result must belong to.
            delta (int): Signed change the database reported as applied.

        Returns:
            The corrected result, which also replaces `last_result`.

        Raises:
            DragonGateParticipantUnknownError: There is no latest result, or it belongs to
                someone else, so there is nothing this correction can safely apply to.
        """
        result = self.last_result
        if result is None or result.participant.user_id != user_id:
            raise DragonGateParticipantUnknownError("No latest result for user")
        previous_delta = result.delta
        self.player_deltas[user_id] += delta - previous_delta
        adjusted = result.model_copy(update={"delta": delta})
        self.last_result = adjusted
        return adjusted

    def is_active(self, user_id: int) -> bool:
        """Returns whether the given user is still seated and has not withdrawn.

        Args:
            user_id (int): Discord user ID to check.

        Returns:
            True only for a seat of this table that is still in the rotation.
        """
        return (
            any(participant.user_id == user_id for participant in self.participants)
            and user_id not in self.withdrawn_user_ids
        )

    def active_participants(self) -> list[GameParticipant]:
        """Returns the seats still in the rotation, in seating order.

        Withdrawn players stay in `participants` so the final embed can still settle and name
        them; this is the filtered view the rotation and the finish check run on.
        """
        return [
            participant
            for participant in self.participants
            if participant.user_id not in self.withdrawn_user_ids
        ]

    def withdraw(self, user_id: int) -> int:
        """Removes a player from the rotation and returns their running delta.

        The money side of "逆贏不拿" is the caller's: a positive returned delta is what the view
        pushes back into the jackpot. Nothing here re-reads it afterwards, so a caller that
        drops the value has silently forgiven the clawback.

        Leaving mid-turn is fine: the rotation only skips ahead when the leaver held the active
        turn, and that deals a fresh gate to the next seat. The last seat leaving finishes the
        table and clears `active_turn` instead.

        Args:
            user_id (int): Discord user ID of the leaver.

        Returns:
            The leaver's running delta at the moment of withdrawal.

        Raises:
            DragonGateParticipantUnknownError: `user_id` is not seated at this table, or has
                already withdrawn.
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
        """Moves the cursor to the next seat still in the rotation and deals it a gate.

        The scan is bounded by the seat count, so a table where every seat has withdrawn
        finishes rather than spinning; the leading check makes that the normal exit and the one
        after the loop the backstop.
        """
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
        """Deals a new playable gate to the seat the cursor is on.

        `turn_number` is bumped only once a playable gate exists, so the redeals behind
        `_draw_open_gate_pillars` never show up as turns nobody played.
        """
        participant = self.participants[self.current_player_index]
        pillars = self._draw_open_gate_pillars()
        self.turn_number += 1
        self.active_turn = DragonGateTurn(
            turn_number=self.turn_number, participant=participant, pillars=pillars
        )

    def _draw_open_gate_pillars(self) -> list[Card]:
        """Draws pairs of pillars until one of them opens a gate.

        Unbounded on purpose: `draw_card` never exhausts, and only adjacent non-pair ranks are
        redealt, which is 24 of the 169 rank pairings, so a retry budget would bound nothing a
        run of unlucky draws could realistically reach.

        Returns:
            The two pillar cards, in the order drawn.
        """
        while True:
            pillars = [draw_card(rng=self.rng), draw_card(rng=self.rng)]
            if has_open_gate(pillars=pillars):
                return pillars

    def _require_active_turn(self, user_id: int) -> DragonGateTurn:
        """Returns the active turn, or raises the rule error that says why there is none.

        The single gate every public mutation goes through, so the two errors below reach
        callers of `choose_pair_direction` and `place_bet` as well; the view maps each to its
        own ephemeral notice, which is why they are distinct types rather than one.

        Args:
            user_id (int): Discord user ID that must own the active turn.

        Returns:
            The live turn belonging to that user.

        Raises:
            DragonGateTableFinishedError: The table has finished, or holds no active turn.
            DragonGateTurnError: Someone other than the active player is acting.
        """
        if self.finished or self.active_turn is None:
            raise DragonGateTableFinishedError("Table is finished")
        if self.active_turn.participant.user_id != user_id:
            raise DragonGateTurnError("Not this player's turn")
        return self.active_turn

    def _resolve_turn(
        self, turn: DragonGateTurn, third_card: Card, amount: int
    ) -> tuple[DragonGateOutcome, int]:
        """Resolves an ordinary gate into an outcome label and the player's signed delta.

        A same-point gate is handed to `_resolve_pair_turn` instead. The pillar check has to run
        before the fall-through: a pillar card is not strictly between the pillars, so without
        it the double loss would be priced as an ordinary miss.

        Args:
            turn (DragonGateTurn): The gate being shot at.
            third_card (Card): The card just drawn against it.
            amount (int): Bet the outcome is priced against.

        Returns:
            `(outcome, delta)`: +1x the bet inside the gate, -1x outside it, -2x on a pillar.
        """
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
        """Resolves a same-point gate against the higher/lower call that was made.

        The third card matching the pair is the worst outcome on the table at -3x, and it is
        tested first because it would otherwise fall through as a wrong call at -1x. A missing
        `direction` cannot reach here, since `place_bet` refuses the bet first; it would read as
        a wrong call if it did.

        Args:
            turn (DragonGateTurn): The same-point gate, carrying the recorded call.
            third_value (int): Point value of the card just drawn.
            amount (int): Bet the outcome is priced against.

        Returns:
            `(outcome, delta)`: +1x the bet on a right call, -1x on a wrong one, -3x when the
            third card is the pair's own point again.
        """
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
