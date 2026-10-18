"""Pure Blackjack rules and shoe helpers.

All deals draw from an injected `rng`: a fixed one in deterministic tests,
`random.SystemRandom()` in production.
"""

from random import Random
from typing import Final, Literal

from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.games import (
    Card,
    BotAction,
    SettleOutcome,
    GameParticipant,
    BlackjackDealerStep,
)
from discordbot.typings.economy import MAX_SINGLE_BET

RoundPhase = Literal["insurance", "player_actions", "settled"]


class InsuranceRefusedError(ValueError):
    """The round refused an insurance side bet.

    Subclasses distinguish closed insurance, insufficient balance, and a sub-unit
    half-bet so the view can pick a player-facing notice from the exception type
    while rules callers can still catch `ValueError`.
    """


class InsuranceClosedError(InsuranceRefusedError):
    """Insurance is no longer on offer here: the phase moved on, or this seat already decided."""


class InsuranceBeyondBalanceError(InsuranceRefusedError):
    """What the seat has left will not cover the half-bet side wager."""


class InsuranceBetTooSmallError(InsuranceRefusedError):
    """Half the seat's bet rounds to zero, so there is no side wager to place.

    Permanent for the round: no refresh and no other seat's move changes it.
    """


SHOE_DECK_COUNT = 4
CARD_RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
CARD_SUITS = ("♠", "♥", "♦", "♣")
TEN_VALUE_RANKS: Final[frozenset[str]] = frozenset({"10", "J", "Q", "K"})
FIVE_CARD_WIN_CARDS: Final[int] = 5


def draw_card(rng: Random) -> Card:
    """Returns one card drawn with replacement from a notional infinite shoe.

    Args:
        rng (Random): Random source used to choose rank and suit.
    """
    return Card(rank=rng.choice(seq=CARD_RANKS), suit=rng.choice(seq=CARD_SUITS))


def build_shoe(rng: Random) -> list[Card]:
    """Returns a shuffled `SHOE_DECK_COUNT`-deck shoe (4 decks = 208 cards).

    Cards are popped from index 0 (FIFO). The deck count is sized so a full
    table taking splits and double-downs never runs the shoe out mid-round.

    Args:
        rng (Random): Random source used to shuffle the shoe.
    """
    shoe: list[Card] = [
        Card(rank=rank, suit=suit)
        for _ in range(SHOE_DECK_COUNT)
        for suit in CARD_SUITS
        for rank in CARD_RANKS
    ]
    rng.shuffle(shoe)
    return shoe


def hand_value(cards: list[Card]) -> int:
    """Returns the best Blackjack value for a hand.

    Aces start at 11 each and are demoted to 1 one at a time while the total
    is over 21. Returns the over-21 total when no aces remain to demote.

    Args:
        cards (list[Card]): Cards to evaluate.
    """
    return is_soft_total(cards=cards)[1]


def card_blackjack_value(card: Card) -> int:
    """Returns a card's Blackjack value, counting an Ace at its high 11.

    Args:
        card (Card): Card to evaluate.
    """
    if card.rank == "A":
        return 11
    if card.rank in TEN_VALUE_RANKS:
        return 10
    return int(card.rank)


def is_blackjack(cards: list[Card]) -> bool:
    """Returns whether `cards` is a two-card natural 21.

    Args:
        cards (list[Card]): Cards to evaluate.
    """
    return len(cards) == 2 and hand_value(cards=cards) == 21


def is_five_card_twenty_one(cards: list[Card]) -> bool:
    """Returns whether `cards` has at least five cards totaling 21.

    Args:
        cards (list[Card]): Cards to evaluate.
    """
    return len(cards) >= FIVE_CARD_WIN_CARDS and hand_value(cards=cards) == 21


def is_five_card_win(cards: list[Card]) -> bool:
    """Returns whether `cards` qualifies for a non-bust five-card win.

    Args:
        cards (list[Card]): Cards to evaluate.
    """
    return len(cards) >= FIVE_CARD_WIN_CARDS and hand_value(cards=cards) <= 21


def is_bust(cards: list[Card]) -> bool:
    """Returns whether `cards` totals more than 21.

    Args:
        cards (list[Card]): Cards to evaluate.
    """
    return hand_value(cards=cards) > 21


def is_pair(cards: list[Card]) -> bool:
    """Returns whether `cards` is a two-card same-value pair.

    Face cards and 10 all count as 10-value cards and can be split with each
    other; an Ace counts as 11 here and never pairs with a 10-value card.

    Args:
        cards (list[Card]): Cards to evaluate.
    """
    return len(cards) == 2 and card_blackjack_value(card=cards[0]) == card_blackjack_value(
        card=cards[1]
    )


def is_soft_total(cards: list[Card]) -> tuple[bool, int]:
    """Returns `(is_soft, total)` where `is_soft` is True while an Ace still counts as 11.

    Args:
        cards (list[Card]): Cards to evaluate.
    """
    total = sum(card_blackjack_value(card=card) for card in cards)
    aces_high = sum(1 for card in cards if card.rank == "A")
    while total > 21 and aces_high > 0:
        total -= 10
        aces_high -= 1
    return aces_high > 0, total


def is_soft_17(cards: list[Card]) -> bool:
    """Returns whether `cards` totals 17 with at least one Ace counted as 11.

    Args:
        cards (list[Card]): Cards to evaluate.
    """
    soft, total = is_soft_total(cards=cards)
    return soft and total == 17


def dealer_must_hit(cards: list[Card]) -> bool:
    """Returns whether the dealer draws on `cards` under H17 rules.

    The dealer hits anything below 17 and a soft 17, and stands on a hard 17
    or better.

    Args:
        cards (list[Card]): Dealer cards to evaluate.
    """
    return hand_value(cards=cards) < 17 or is_soft_17(cards=cards)


def dealer_up_card(dealer: list[Card]) -> Card | None:
    """Returns the dealer's visible up-card, or None if no cards have been dealt.

    The first dealt card is the hole card (hidden) and the second is the
    up-card; when the dealer has only one card, that card is treated as visible.

    Args:
        dealer (list[Card]): Dealer's cards in draw order.
    """
    if not dealer:
        return None
    return dealer[1] if len(dealer) > 1 else dealer[0]


class BlackjackHandState(BaseModel):
    """One sub-hand owned by a multiplayer Blackjack participant.

    Split turns a single hand into two sibling hand states sharing one
    participant; otherwise a participant has exactly one entry in
    `BlackjackPlayerHand.hands`.
    """

    cards: list[Card] = Field(
        default_factory=list, description="Cards currently held in this hand."
    )
    bet: int = Field(..., description="Active wager for this hand (doubled after Double Down).")
    base_bet: int = Field(..., description="Original wager kept for Surrender refund math.")
    finished: bool = Field(
        default=False, description="True once this hand no longer needs Hit / Stand actions."
    )
    doubled: bool = Field(default=False, description="True after a Double Down on this hand.")
    surrendered: bool = Field(default=False, description="True after a Surrender on this hand.")
    is_split_hand: bool = Field(
        default=False, description="True when this hand came out of a Split."
    )
    is_split_aces: bool = Field(
        default=False, description="True when both split halves came from an Ace pair."
    )
    actions_taken: int = Field(
        default=0, description="Hit / Double / Surrender counter used by action guards."
    )

    def total(self) -> int:
        """Returns the current best total for this sub-hand."""
        return hand_value(cards=self.cards)

    def is_blackjack(self) -> bool:
        """Returns whether this sub-hand is a natural Blackjack."""
        return not self.is_split_hand and is_blackjack(cards=self.cards)

    def is_bust(self) -> bool:
        """Returns whether this sub-hand has busted."""
        return is_bust(cards=self.cards)


class BlackjackPlayerHand(BaseModel):
    """Container for one participant's hands at a multiplayer table.

    Holds the original `GameParticipant` plus one or more
    `BlackjackHandState` rows. Split adds a second entry; everything else
    keeps a single hand entry.
    """

    participant: GameParticipant = Field(..., description="Discord player and wager metadata.")
    hands: list[BlackjackHandState] = Field(
        default_factory=list, description="All active sub-hands in display order."
    )
    insurance_bet: int = Field(
        default=0, description="Insurance side bet amount, 0 when none was taken."
    )
    insurance_resolved: bool = Field(
        default=False,
        description=(
            "True once the player has taken or declined insurance; the phase closes"
            " only once every player is resolved."
        ),
    )

    @property
    def finished(self) -> bool:
        """Returns True once every owned hand has finished."""
        return bool(self.hands) and all(hand.finished for hand in self.hands)

    @property
    def balance_remaining(self) -> int:
        """Returns the starting balance left once every hand bet and the insurance are covered.

        This is what a Double, a Split or an insurance side bet has to fit in.
        """
        return (
            self.participant.balance_at_start
            - sum(hand.bet for hand in self.hands)
            - self.insurance_bet
        )


def can_double(hand: BlackjackHandState, balance_remaining: int) -> bool:
    """Returns whether Double Down is allowed on `hand` right now.

    Allowed only on an unsplit two-card hand with no actions taken, when the
    doubled stake fits `MAX_SINGLE_BET` and `balance_remaining` covers the
    extra wager.

    Args:
        hand (BlackjackHandState): Hand to inspect.
        balance_remaining (int): Points still available after current commitments.
    """
    if hand.finished or hand.surrendered or hand.doubled:
        return False
    if len(hand.cards) != 2 or hand.actions_taken != 0:
        return False
    if hand.is_split_hand:
        return False
    # Doubling doubles the hand's stake; keep it within the single-bet cap so it
    # cannot bypass the anti-inflation guardrail that bounds every wager.
    if hand.bet * 2 > MAX_SINGLE_BET:
        return False
    return balance_remaining >= hand.bet


def can_split(hand: BlackjackHandState, balance_remaining: int) -> bool:
    """Returns whether Split is allowed on `hand` right now.

    Allowed only on an unsplit two-card same-value pair with no actions taken,
    when `balance_remaining` covers a matching stake on the second sub-hand.

    Args:
        hand (BlackjackHandState): Hand to inspect.
        balance_remaining (int): Points still available after current commitments.
    """
    if hand.finished or hand.surrendered or hand.doubled or hand.is_split_hand:
        return False
    if hand.actions_taken != 0:
        return False
    if not is_pair(cards=hand.cards):
        return False
    return balance_remaining >= hand.bet


def can_surrender(hand: BlackjackHandState, peeked_blackjack: bool) -> bool:
    """Returns whether Late Surrender is allowed on `hand` right now.

    Allowed only on an unsplit two-card hand with no actions taken before a
    dealer hole-card peek reveals Blackjack.

    Args:
        hand (BlackjackHandState): Hand to inspect.
        peeked_blackjack (bool): Whether the dealer already peeked a Blackjack.
    """
    if peeked_blackjack:
        return False
    if hand.finished or hand.surrendered or hand.doubled or hand.is_split_hand:
        return False
    return len(hand.cards) == 2 and hand.actions_taken == 0


def max_round_loss(bet: int, balance_at_start: int) -> int:
    """Returns the most one seat can lose in a round, which is what its table holds of the wallet.

    A Double or a Split each doubles the stake and neither is open on a split hand, insurance adds
    half the original bet, and all of it has to fit the balance the seat started with.

    Args:
        bet (int): Base hand wager for the seat.
        balance_at_start (int): Player's wallet balance when seated.
    """
    return min(balance_at_start, bet * 2 + bet // 2)


def surrender_loss(bet: int) -> int:
    """Returns what a Late Surrender forfeits: half `bet`, rounded up so a 1-point bet is not free.

    Args:
        bet (int): Original hand wager.
    """
    return (bet + 1) // 2


def _settle_regular_hand(
    hand: BlackjackHandState, dealer: list[Card]
) -> tuple[SettleOutcome, int]:
    """Resolves a finished non-surrender, non-special Blackjack sub-hand."""
    bet = hand.bet
    player_total = hand.total()
    dealer_total = hand_value(cards=dealer)
    player_bj = hand.is_blackjack()
    dealer_bj = is_blackjack(cards=dealer)

    if player_bj and dealer_bj:
        outcome: SettleOutcome = "push"
        delta = 0
    elif player_bj:
        # A natural pays 3:2.
        outcome, delta = "blackjack", bet * 3 // 2
    elif dealer_bj:
        outcome, delta = "lose", -bet
    elif hand.is_bust():
        outcome, delta = "player_bust", -bet
    elif is_bust(cards=dealer):
        outcome, delta = "dealer_bust", bet
    elif player_total > dealer_total:
        outcome, delta = "win", bet
    elif player_total < dealer_total:
        outcome, delta = "lose", -bet
    else:
        outcome, delta = "push", 0
    return outcome, delta


def settle_hand(hand: BlackjackHandState, dealer: list[Card]) -> tuple[SettleOutcome, int]:
    """Returns `(outcome, delta)` for one finished sub-hand against `dealer`.

    Surrender short-circuits to a half-bet refund. `is_five_card_win` also
    matches a five-card 21, so the 21 is tested first to preserve its bonus
    label; a five-card 21 pushes its main leg against a dealer 21 while the
    settlement layer adds its system-funded bonus. A split hand's two-card 21
    is not a natural and settles as an ordinary 21.

    Args:
        hand (BlackjackHandState): Finished sub-hand to settle.
        dealer (list[Card]): Final dealer cards.

    Raises:
        ValueError: The hand has not finished yet.
    """
    if not hand.finished:
        raise ValueError("Cannot settle an unfinished Blackjack hand")
    if hand.surrendered:
        return "surrender", -surrender_loss(bet=hand.base_bet)
    if is_five_card_twenty_one(cards=hand.cards):
        dealer_total = hand_value(cards=dealer)
        delta = 0 if dealer_total == 21 else hand.bet
        return "five_card_twenty_one", delta
    if is_five_card_win(cards=hand.cards):
        return "five_card_win", hand.bet
    return _settle_regular_hand(hand=hand, dealer=dealer)


class BlackjackRound(BaseModel):
    """Mutable state for a multiplayer Blackjack table.

    One dealer hand is shared by every player. Players act in lobby join
    order, advancing through each owned sub-hand before moving on to the
    next player; natural Blackjacks, surrendered, doubled, and busted hands
    are skipped automatically.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rng: Random = Field(..., description="Random source used for card draws.")
    players: list[BlackjackPlayerHand] = Field(
        ..., description="Per-player containers, each holding one or more sub-hands."
    )
    dealer: list[Card] = Field(
        default_factory=list, description="Dealer cards shared by the table."
    )
    shoe: list[Card] = Field(
        default_factory=list, description="Remaining cards in the FIFO multi-deck shoe."
    )
    current_player_index: int = Field(
        default=0, description="Index of the player whose turn is active."
    )
    current_hand_index: int = Field(
        default=0, description="Index of the active sub-hand within the active player."
    )
    dealer_played: bool = Field(
        default=False, description="True once the dealer has drawn for all standing players."
    )
    phase: RoundPhase = Field(
        default="player_actions", description="Lifecycle phase of the round."
    )
    peeked_blackjack: bool = Field(
        default=False,
        description="True once the dealer's hole-card peek revealed a natural Blackjack.",
    )

    @property
    def finished(self) -> bool:
        """Returns True once the round is settled and no more player actions remain."""
        return self.phase == "settled"

    @classmethod
    def from_participants(
        cls, rng: Random, participants: list[GameParticipant], shoe: list[Card] | None = None
    ) -> "BlackjackRound":
        """Returns a round initialized from `participants`.

        When `shoe` is provided, the round validates a copy into `round_state.shoe`
        so dealing never mutates the caller's list; otherwise a fresh shuffled
        shoe is built.

        Args:
            rng (Random): Random source used for card draws and fresh shoes.
            participants (list[GameParticipant]): Seated players in join order.
            shoe (list[Card] | None): Optional persistent shoe to deal from.
        """
        players = [
            BlackjackPlayerHand(
                participant=participant,
                hands=[BlackjackHandState(bet=participant.bet, base_bet=participant.bet)],
            )
            for participant in participants
        ]
        return cls(
            rng=rng, players=players, shoe=shoe if shoe is not None else build_shoe(rng=rng)
        )

    def _draw_one_card(self) -> Card:
        """Pops the next card from the FIFO shoe, falling back to `draw_card` when empty."""
        if self.shoe:
            return self.shoe.pop(0)
        return draw_card(rng=self.rng)

    def deal_initial(self) -> None:
        """Deals two cards to every player and two cards to the dealer.

        Dealer up-card drives the post-deal lifecycle:
        - Up-card is Ace: enter `insurance` phase and let players decide
          before peeking the hole card. The peek runs at the close of the
          insurance phase.
        - Up-card is a 10-value card: peek silently (no insurance offered);
          if the peek reveals a Blackjack the round settles immediately.
        - Anything else: jump straight to `player_actions`.
        """
        for player in self.players:
            for hand in player.hands:
                hand.cards = [self._draw_one_card(), self._draw_one_card()]
        self.dealer = [self._draw_one_card(), self._draw_one_card()]
        up = dealer_up_card(dealer=self.dealer)
        if up is not None and up.rank == "A":
            self.phase = "insurance"
            return

        ten_value_up = up is not None and up.rank in TEN_VALUE_RANKS
        self._resolve_peek(dealer_has_blackjack=ten_value_up and is_blackjack(cards=self.dealer))

    def take_insurance(self, user_id: int) -> None:
        """Records the player's insurance side bet, half their original bet rounded down.

        Args:
            user_id (int): Discord user ID placing the insurance.

        Raises:
            InsuranceClosedError: The round is not in the insurance phase, or this
                seat already decided.
            InsuranceBetTooSmallError: Half the original bet rounds to zero.
            InsuranceBeyondBalanceError: The remaining balance cannot cover it.
            ValueError: The user is not seated at this table.
        """
        if self.phase != "insurance":
            raise InsuranceClosedError("Insurance is not currently offered")
        player = self._find_player(user_id=user_id)
        if player.insurance_resolved:
            raise InsuranceClosedError("Insurance already decided")
        cost = player.participant.bet // 2
        if cost <= 0:
            raise InsuranceBetTooSmallError("Half of the original bet rounds to zero")
        if player.balance_remaining < cost:
            raise InsuranceBeyondBalanceError("Not enough balance for insurance")
        player.insurance_bet = cost
        player.insurance_resolved = True
        self._maybe_close_insurance_phase()

    def decline_insurance(self, user_id: int) -> None:
        """Records that the player has declined to take insurance.

        Args:
            user_id (int): Discord user ID declining the insurance offer.

        Raises:
            InsuranceClosedError: The round is not in the insurance phase, or this
                seat already decided.
            ValueError: The user is not seated at this table.
        """
        if self.phase != "insurance":
            raise InsuranceClosedError("Insurance is not currently offered")
        player = self._find_player(user_id=user_id)
        if player.insurance_resolved:
            raise InsuranceClosedError("Insurance already decided")
        player.insurance_resolved = True
        self._maybe_close_insurance_phase()

    def decline_insurance_for_all_unresolved(self) -> None:
        """Marks every undecided player as declining insurance and closes the phase."""
        if self.phase != "insurance":
            return
        for player in self.players:
            if not player.insurance_resolved:
                player.insurance_resolved = True
        self._maybe_close_insurance_phase()

    def active_player(self) -> BlackjackPlayerHand | None:
        """Returns the player whose turn is active, if any.

        Advances the table past players who have finished, so this is not a
        pure read: it can move the turn cursor and settle the round.
        """
        if self.finished or self.phase != "player_actions":
            return None
        if self.current_player_index >= len(self.players):
            return None
        player = self.players[self.current_player_index]
        if player.finished:
            self._advance_or_finish()
            return self.active_player()
        return player

    def active_hand(self) -> BlackjackHandState | None:
        """Returns the active sub-hand of the active player, if any.

        Advances past finished sub-hands, with the same non-pure-read caveat as
        `active_player`.
        """
        player = self.active_player()
        if player is None:
            return None
        if self.current_hand_index >= len(player.hands):
            self._advance_or_finish()
            return self.active_hand()
        hand = player.hands[self.current_hand_index]
        if hand.finished:
            self._advance_or_finish()
            return self.active_hand()
        return hand

    def allowed_actions(self) -> tuple[BotAction, ...]:
        """Returns what the active sub-hand may do now, in hit, stand, double, split, surrender order.

        Empty when no sub-hand is waiting on an action. Reads the turn through
        `active_hand`, with the same non-pure-read caveat as `active_player`.
        """
        player = self.active_player()
        hand = self.active_hand()
        if player is None or hand is None:
            return ()
        actions: list[BotAction] = []
        if not hand.is_split_aces:
            actions.extend(("hit", "stand"))
        if can_double(hand=hand, balance_remaining=player.balance_remaining):
            actions.append("double")
        if can_split(hand=hand, balance_remaining=player.balance_remaining):
            actions.append("split")
        if can_surrender(hand=hand, peeked_blackjack=self.peeked_blackjack):
            actions.append("surrender")
        return tuple(actions)

    def hit(self, user_id: int) -> Card:
        """Returns one card drawn for the active sub-hand, finishing it on bust or five-card win.

        Args:
            user_id (int): Discord user ID that must match the active player.

        Raises:
            ValueError: The round is not in the player-action phase, the user
                is not the active player, or the hand came out of split Aces.
        """
        _, hand = self._require_active(user_id=user_id)
        if hand.is_split_aces:
            raise ValueError("Cannot hit after splitting Aces")
        card = self._draw_one_card()
        hand.cards.append(card)
        hand.actions_taken += 1
        if hand.is_bust() or is_five_card_win(cards=hand.cards):
            hand.finished = True
            self._advance_or_finish()
        return card

    def stand(self, user_id: int) -> None:
        """Marks the active sub-hand as standing and advances the table.

        Args:
            user_id (int): Discord user ID that must match the active player.

        Raises:
            ValueError: The round is not in the player-action phase or the
                user is not the active player.
        """
        _, hand = self._require_active(user_id=user_id)
        hand.finished = True
        self._advance_or_finish()

    def double_down(self, user_id: int) -> Card:
        """Returns the single card drawn after doubling the active hand's wager and finishing it.

        Args:
            user_id (int): Discord user ID that must match the active player.

        Raises:
            ValueError: The round is not in the player-action phase, the user
                is not the active player, or `can_double` rejects the hand.
        """
        player, hand = self._require_active(user_id=user_id)
        if not can_double(hand=hand, balance_remaining=player.balance_remaining):
            raise ValueError("Cannot double this hand")
        hand.bet *= 2
        hand.doubled = True
        card = self._draw_one_card()
        hand.cards.append(card)
        hand.actions_taken += 1
        hand.finished = True
        self._advance_or_finish()
        return card

    def split(self, user_id: int) -> None:
        """Splits the active hand into two sibling sub-hands.

        Each sibling gets one original card plus one fresh draw; splitting Aces
        marks both siblings `is_split_aces` and finishes them immediately.

        Args:
            user_id (int): Discord user ID that must match the active player.

        Raises:
            ValueError: The round is not in the player-action phase, the user
                is not the active player, or `can_split` rejects the hand.
        """
        player, hand = self._require_active(user_id=user_id)
        if not can_split(hand=hand, balance_remaining=player.balance_remaining):
            raise ValueError("Cannot split this hand")
        split_aces = hand.cards[0].rank == "A"
        first_card, second_card = hand.cards[0], hand.cards[1]
        new_hand = BlackjackHandState(
            cards=[second_card, self._draw_one_card()],
            bet=hand.base_bet,
            base_bet=hand.base_bet,
            is_split_hand=True,
            is_split_aces=split_aces,
            finished=split_aces,
        )
        hand.cards = [first_card, self._draw_one_card()]
        hand.is_split_hand = True
        hand.is_split_aces = split_aces
        hand.finished = split_aces
        hand.actions_taken = 0
        active_index = self.current_hand_index
        player.hands.insert(active_index + 1, new_hand)
        self._advance_or_finish()

    def surrender(self, user_id: int) -> None:
        """Surrenders the active hand for a half-bet refund.

        Args:
            user_id (int): Discord user ID that must match the active player.

        Raises:
            ValueError: The round is not in the player-action phase, the user
                is not the active player, or `can_surrender` rejects the hand.
        """
        _, hand = self._require_active(user_id=user_id)
        if not can_surrender(hand=hand, peeked_blackjack=self.peeked_blackjack):
            raise ValueError("Cannot surrender this hand")
        hand.surrendered = True
        hand.finished = True
        hand.actions_taken += 1
        self._advance_or_finish()

    def stand_all_remaining(self) -> None:
        """Declines undecided insurance, stands every unresolved hand, and settles the round."""
        if self.phase == "insurance":
            self.decline_insurance_for_all_unresolved()
        for player in self.players:
            for hand in player.hands:
                hand.finished = True
        self._finish_after_players_done()

    def dealer_total(self) -> int:
        """Returns the current best total for the dealer hand."""
        return hand_value(cards=self.dealer)

    def needs_dealer_play(self) -> bool:
        """Returns whether the dealer must draw before settlement."""
        if self.peeked_blackjack:
            return False
        for player in self.players:
            for hand in player.hands:
                if hand.surrendered:
                    continue
                if hand.is_blackjack():
                    continue
                if hand.is_bust():
                    continue
                if is_five_card_win(cards=hand.cards) and not is_five_card_twenty_one(
                    cards=hand.cards
                ):
                    continue
                return True
        return False

    def play_dealer(self) -> list[BlackjackDealerStep]:
        """Draws for the dealer under H17 rules, then closes the dealer phase.

        Returns:
            Each hit in draw order, then a stand unless the dealer bust. Empty, with the round
            left as it was, when the dealer already played or no hand needs it to.
        """
        if self.dealer_played or not self.needs_dealer_play():
            return []
        steps: list[BlackjackDealerStep] = []
        while dealer_must_hit(cards=self.dealer):
            total_before = self.dealer_total()
            drawn_card = self._draw_one_card()
            self.dealer.append(drawn_card)
            steps.append(
                BlackjackDealerStep(
                    total_before=total_before,
                    action="hit",
                    drawn_card=drawn_card,
                    total_after=self.dealer_total(),
                )
            )
        final_total = self.dealer_total()
        if final_total <= 21:
            steps.append(BlackjackDealerStep(total_before=final_total, action="stand"))
        self.dealer_played = True
        self.phase = "settled"
        return steps

    def find_player(self, user_id: int) -> BlackjackPlayerHand | None:
        """Returns the seat `user_id` holds at this table, or None when they hold none.

        Args:
            user_id (int): Discord user ID to look up.
        """
        for player in self.players:
            if player.participant.user_id == user_id:
                return player
        return None

    def _find_player(self, user_id: int) -> BlackjackPlayerHand:
        """Returns the player by `user_id`, or raises `ValueError` when unknown."""
        player = self.find_player(user_id=user_id)
        if player is None:
            raise ValueError("Unknown user for this round")
        return player

    def _require_active(self, user_id: int) -> tuple[BlackjackPlayerHand, BlackjackHandState]:
        """Returns the active `(player, hand)` tuple, or raises `ValueError` when out of turn."""
        if self.phase != "player_actions":
            raise ValueError("Not in player action phase")
        player = self.active_player()
        hand = self.active_hand()
        if player is None or hand is None or player.participant.user_id != user_id:
            raise ValueError("Not this player's turn")
        return player, hand

    def _maybe_close_insurance_phase(self) -> None:
        """Closes the insurance phase and peeks the hole card once every player has decided."""
        if self.phase != "insurance":
            return
        if not all(player.insurance_resolved for player in self.players):
            return
        self._resolve_peek(dealer_has_blackjack=is_blackjack(cards=self.dealer))

    def _resolve_peek(self, dealer_has_blackjack: bool) -> None:
        """Applies the hole-card peek result, settling on a dealer natural or entering `player_actions`."""
        if dealer_has_blackjack:
            self.peeked_blackjack = True
            for player in self.players:
                for hand in player.hands:
                    hand.finished = True
            self.phase = "settled"
            self.dealer_played = True
            return
        self.phase = "player_actions"
        for player in self.players:
            for hand in player.hands:
                if hand.is_blackjack():
                    hand.finished = True
        self._advance_or_finish()

    def _advance_or_finish(self) -> None:
        """Skips completed sub-hands and settles the table when none remain."""
        while self.current_player_index < len(self.players):
            player = self.players[self.current_player_index]
            while self.current_hand_index < len(player.hands):
                if not player.hands[self.current_hand_index].finished:
                    return
                self.current_hand_index += 1
            self.current_player_index += 1
            self.current_hand_index = 0
        self._finish_after_players_done()

    def _finish_after_players_done(self) -> None:
        """Marks the round settled after all player actions have resolved."""
        self.phase = "settled"
