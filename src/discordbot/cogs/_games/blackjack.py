"""Pure Blackjack rules and shoe helpers.

Kept side-effect free so the unit tests can drive deterministic deals via a
fixed `rng`. The cog wires this up with `random.SystemRandom()` for production.
"""

from random import Random
from typing import Literal

from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.games import Card, SettleOutcome, GameParticipant

RoundPhase = Literal["insurance", "player_actions", "dealer", "settled"]


SHOE_DECK_COUNT = 4
_CARD_RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
_CARD_SUITS = ("♠", "♥", "♦", "♣")


def draw_card(rng: Random) -> Card:
    """Draws one card from a notional infinite shoe (independent rank + suit).

    Production paths build a shuffled 4-deck shoe inside `BlackjackRound` and
    draw from there, so a single hand never sees the same card twice. This
    helper stays as the round-bootstrap fallback (and the test-monkeypatch
    seam) for when the shoe is empty.

    Args:
        rng: Random source used to choose rank and suit.

    Returns:
        The drawn card.
    """
    return Card(rank=rng.choice(seq=_CARD_RANKS), suit=rng.choice(seq=_CARD_SUITS))


def build_shoe(rng: Random, deck_count: int = SHOE_DECK_COUNT) -> list[Card]:
    """Returns a shuffled multi-deck shoe (default 4 decks = 208 cards).

    Cards are popped from index 0 (FIFO); the head of the list is the next
    card. The shoe is sized to comfortably cover the worst-case 6-player table
    with splits and double-downs without ever needing to reshuffle mid-round.
    """
    shoe: list[Card] = [
        Card(rank=rank, suit=suit)
        for _ in range(deck_count)
        for suit in _CARD_SUITS
        for rank in _CARD_RANKS
    ]
    rng.shuffle(shoe)
    return shoe


def hand_value(cards: list[Card]) -> int:
    """Returns the best Blackjack value for a hand.

    Aces start at 11 each and are demoted to 1 one at a time while the total
    is over 21. Returns the over-21 total when no aces remain to demote, so
    callers can detect a bust by checking `> 21`.

    Args:
        cards: Cards to evaluate.

    Returns:
        Best total for the hand under Blackjack ace rules.
    """
    total = 0
    aces = 0
    for card in cards:
        if card.rank == "A":
            aces += 1
            total += 11
        elif card.rank in ("J", "Q", "K"):
            total += 10
        else:
            total += int(card.rank)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def _card_blackjack_value(card: Card) -> int:
    """Returns the Blackjack value used for pair and up-card checks."""
    if card.rank == "A":
        return 11
    if card.rank in ("J", "Q", "K"):
        return 10
    return int(card.rank)


def is_blackjack(cards: list[Card]) -> bool:
    """Returns whether a hand is a natural Blackjack.

    Args:
        cards: Cards to evaluate.

    Returns:
        True only when the hand has exactly two cards summing to 21.
    """
    return len(cards) == 2 and hand_value(cards=cards) == 21


def is_five_card_twenty_one(cards: list[Card]) -> bool:
    """Returns whether a hand has five or more cards totaling 21.

    Args:
        cards: Cards to evaluate.

    Returns:
        True only when the hand has at least five cards and totals 21.
    """
    return len(cards) >= 5 and hand_value(cards=cards) == 21


def is_five_card_win(cards: list[Card]) -> bool:
    """Returns whether a hand qualifies for the non-bust five-card win."""
    return len(cards) >= 5 and hand_value(cards=cards) <= 21


def is_bust(cards: list[Card]) -> bool:
    """Returns whether a hand is over 21.

    Args:
        cards: Cards to evaluate.

    Returns:
        True when the hand total is greater than 21.
    """
    return hand_value(cards=cards) > 21


def is_pair(cards: list[Card]) -> bool:
    """Returns whether two cards form a same-value pair.

    Face cards and 10 are all 10-value cards, so 10/J/Q/K can be split with
    each other. A + 10 remains non-splittable because Ace uses its Blackjack
    value of 11 here.

    Args:
        cards: Cards to evaluate; only meaningful on exactly two cards.

    Returns:
        True only when exactly two cards share the same Blackjack value.
    """
    return len(cards) == 2 and _card_blackjack_value(card=cards[0]) == _card_blackjack_value(
        card=cards[1]
    )


def is_soft_total(cards: list[Card]) -> tuple[bool, int]:
    """Returns whether the hand is currently soft and its best total.

    A hand is "soft" while at least one Ace is still being counted as 11.

    Args:
        cards: Cards to evaluate.

    Returns:
        `(is_soft, total)` where total is the best Blackjack total.
    """
    raw_total = 0
    aces = 0
    for card in cards:
        if card.rank == "A":
            aces += 1
            raw_total += 11
        elif card.rank in ("J", "Q", "K"):
            raw_total += 10
        else:
            raw_total += int(card.rank)
    aces_high = aces
    total = raw_total
    while total > 21 and aces_high > 0:
        total -= 10
        aces_high -= 1
    return aces_high > 0, total


def is_soft_17(cards: list[Card]) -> bool:
    """Returns whether the hand is exactly a soft 17.

    Args:
        cards: Cards to evaluate.

    Returns:
        True only when the hand totals 17 with at least one Ace still
        counted as 11.
    """
    soft, total = is_soft_total(cards=cards)
    return soft and total == 17


def dealer_up_card(dealer: list[Card]) -> Card | None:
    """Returns the dealer's visible up-card.

    The first dealt card is the hole card (hidden); the second is the up-card.
    When the dealer only has one card so far that card is treated as visible.

    Args:
        dealer: Dealer's cards in draw order.

    Returns:
        The visible card, or `None` if the dealer has not been dealt yet.
    """
    if not dealer:
        return None
    return dealer[1] if len(dealer) > 1 else dealer[0]


class BlackjackHandState(BaseModel):
    """One sub-hand owned by a multiplayer Blackjack participant.

    Split turns a single hand into two sibling hand states sharing one
    participant; otherwise a participant has exactly one entry in
    `BlackjackPlayerHand.hands`.

    Attributes:
        cards: Cards currently held in this hand.
        bet: Active wager for this hand (doubled after Double Down).
        base_bet: Original wager kept for Surrender refund math.
        finished: True once this hand no longer needs Hit / Stand actions.
        doubled: True after a Double Down on this hand.
        surrendered: True after a Surrender on this hand.
        is_split_hand: True when this hand came out of a Split.
        is_split_aces: True when both split halves came from an Ace pair.
        actions_taken: Hit / Double / Surrender counter used by action guards.
    """

    cards: list[Card] = Field(default_factory=list)
    bet: int
    base_bet: int
    finished: bool = False
    doubled: bool = False
    surrendered: bool = False
    is_split_hand: bool = False
    is_split_aces: bool = False
    actions_taken: int = 0

    def total(self) -> int:
        """Returns the current best total for this sub-hand."""
        return hand_value(cards=self.cards)

    def is_blackjack(self) -> bool:
        """Returns whether this sub-hand is a natural Blackjack."""
        return not self.is_split_hand and is_blackjack(cards=self.cards)

    def is_bust(self) -> bool:
        """Returns whether this sub-hand has busted."""
        return is_bust(cards=self.cards)

    def soft_total(self) -> tuple[bool, int]:
        """Returns `(is_soft, total)` for this sub-hand."""
        return is_soft_total(cards=self.cards)


class BlackjackPlayerHand(BaseModel):
    """Container for one participant's hands at a multiplayer table.

    Holds the original `GameParticipant` plus one or more
    `BlackjackHandState` rows. Split adds a second entry; everything else
    keeps a single hand entry.

    Attributes:
        participant: Discord player and wager metadata.
        hands: All active sub-hands in display order.
        insurance_bet: Insurance side bet amount, `0` when none was taken.
        insurance_resolved: True once the player has made an insurance choice
            (yes or no) and the surrounding round phase has progressed past
            insurance.
    """

    participant: GameParticipant
    hands: list[BlackjackHandState] = Field(default_factory=list)
    insurance_bet: int = 0
    insurance_resolved: bool = False

    @property
    def finished(self) -> bool:
        """Returns True once every owned hand has finished."""
        return bool(self.hands) and all(hand.finished for hand in self.hands)


def committed_wagers(player: BlackjackPlayerHand) -> int:
    """Returns the total points already committed for one participant.

    Sums every active hand bet plus any insurance side bet, so callers can
    measure how much of the player's starting balance is still spoken for
    when validating Double / Split / Insurance affordability.

    Args:
        player: The player whose committed wagers should be summed.

    Returns:
        Total committed points across hands and insurance for the player.
    """
    return sum(hand.bet for hand in player.hands) + player.insurance_bet


def can_double(
    hand: BlackjackHandState, balance_remaining: int, allow_after_split: bool = False
) -> bool:
    """Returns whether Double Down is allowed on this hand right now.

    Args:
        hand: Hand to inspect.
        balance_remaining: Points still available after current commitments.
        allow_after_split: Whether the house rule permits Double after Split.

    Returns:
        True only when no actions have been taken yet, the hand has exactly
        two cards, the DAS rule allows it, and the player can still afford
        the extra wager.
    """
    if hand.finished or hand.surrendered or hand.doubled:
        return False
    if len(hand.cards) != 2 or hand.actions_taken != 0:
        return False
    if hand.is_split_hand and not allow_after_split:
        return False
    return balance_remaining >= hand.bet


def can_split(hand: BlackjackHandState, balance_remaining: int) -> bool:
    """Returns whether Split is allowed on this hand right now.

    Args:
        hand: Hand to inspect.
        balance_remaining: Points still available after current commitments.

    Returns:
        True only when the hand has exactly two cards of the same value, has
        not been split or otherwise acted on yet, and the player can still
        afford to mirror the original wager on the second sub-hand.
    """
    if hand.finished or hand.surrendered or hand.doubled or hand.is_split_hand:
        return False
    if hand.actions_taken != 0:
        return False
    if not is_pair(cards=hand.cards):
        return False
    return balance_remaining >= hand.bet


def can_insure(player: "BlackjackPlayerHand", balance_remaining: int) -> bool:
    """Returns whether the player can still place an insurance side bet.

    Args:
        player: Player container to inspect.
        balance_remaining: Points still available after current commitments.

    Returns:
        True only when insurance was offered for this player and they have
        not yet decided, and the half-bet side wager fits the remaining
        balance.
    """
    if player.insurance_resolved or player.insurance_bet != 0:
        return False
    insurance_amount = player.participant.bet // 2
    if insurance_amount <= 0:
        return False
    return balance_remaining >= insurance_amount


def can_surrender(hand: BlackjackHandState, peeked_blackjack: bool) -> bool:
    """Returns whether Late Surrender is allowed on this hand right now.

    Args:
        hand: Hand to inspect.
        peeked_blackjack: Whether the dealer already peeked a Blackjack;
            Surrender is closed once that happened.

    Returns:
        True only when the hand has exactly two cards, has not been acted
        on yet, did not come out of a Split, and the dealer has not already
        revealed a Blackjack via peek.
    """
    if peeked_blackjack:
        return False
    if hand.finished or hand.surrendered or hand.doubled or hand.is_split_hand:
        return False
    return len(hand.cards) == 2 and hand.actions_taken == 0


def _settle_split_twenty_one(
    hand: BlackjackHandState, dealer: list[Card]
) -> tuple[SettleOutcome, int]:
    """Resolves a split-derived two-card 21 without treating it as natural Blackjack."""
    dealer_total = hand_value(cards=dealer)
    if is_blackjack(cards=dealer):
        outcome: SettleOutcome = "lose"
        delta = -hand.bet
    elif dealer_total == 21:
        outcome, delta = "push", 0
    else:
        outcome, delta = "win", hand.bet
    return outcome, delta


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
        outcome, delta = "blackjack", int(bet * 3 // 2)
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
    """Resolves one finished sub-hand into an outcome label and net delta.

    Surrender short-circuits to a half-bet refund. Five-card 21 is flagged
    before the generic five-card win so split hands can still earn the
    five-card bonus. Split-derived two-card 21 is handled before natural
    Blackjack so it never receives the natural Blackjack payout.

    Args:
        hand: Finished sub-hand to settle.
        dealer: Final dealer cards.

    Returns:
        `(outcome, delta)` where delta is the signed point change for
        this single hand.
    """
    if not hand.finished:
        raise ValueError("Cannot settle an unfinished Blackjack hand")
    if hand.surrendered:
        return "surrender", -((hand.base_bet + 1) // 2)
    if not hand.doubled and is_five_card_twenty_one(cards=hand.cards):
        dealer_total = hand_value(cards=dealer)
        delta = 0 if dealer_total == 21 else hand.bet
        return "five_card_twenty_one", delta
    if not hand.doubled and is_five_card_win(cards=hand.cards):
        return "five_card_win", hand.bet
    if hand.is_split_hand and is_blackjack(cards=hand.cards):
        return _settle_split_twenty_one(hand=hand, dealer=dealer)
    return _settle_regular_hand(hand=hand, dealer=dealer)


class BlackjackRound(BaseModel):
    """Mutable state for a multiplayer Blackjack table.

    One dealer hand is shared by every player. Players act in lobby join
    order, advancing through each owned sub-hand before moving on to the
    next player; natural Blackjacks, surrendered, doubled, and busted hands
    are skipped automatically.

    Attributes:
        rng: Random source used for card draws.
        players: Per-player containers (each holds one or more sub-hands).
        dealer: Dealer cards shared by the table.
        current_player_index: Index of the player whose turn is active.
        current_hand_index: Index of the active sub-hand within that player.
        dealer_played: True once the dealer has drawn for all standing
            players.
        finished: True once no more player actions remain.
        auto_play_dealer: True when the pure rules should draw dealer cards
            synchronously after player actions finish.
        phase: Lifecycle phase of the round (insurance / player_actions /
            dealer / settled).
        insurance_offered: True only when the dealer up-card is an Ace.
        peeked_blackjack: True once the dealer's hole-card peek revealed a
            natural Blackjack.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rng: Random
    players: list[BlackjackPlayerHand]
    dealer: list[Card] = Field(default_factory=list)
    shoe: list[Card] = Field(default_factory=list)
    current_player_index: int = 0
    current_hand_index: int = 0
    dealer_played: bool = False
    finished: bool = False
    auto_play_dealer: bool = True
    phase: RoundPhase = "player_actions"
    insurance_offered: bool = False
    peeked_blackjack: bool = False

    @classmethod
    def from_participants(
        cls, rng: Random, participants: list[GameParticipant], auto_play_dealer: bool = True
    ) -> "BlackjackRound":
        """Builds a round from registered lobby participants."""
        players = [
            BlackjackPlayerHand(
                participant=participant,
                hands=[BlackjackHandState(bet=participant.bet, base_bet=participant.bet)],
            )
            for participant in participants
        ]
        return cls(
            rng=rng, players=players, auto_play_dealer=auto_play_dealer, shoe=build_shoe(rng=rng)
        )

    def _draw_one_card(self) -> Card:
        """Pops the next card from the round's shoe, falling back when empty.

        Cards come from the FIFO shoe so no two seats in the same round can
        receive duplicate cards. The 4-deck shoe holds 208 cards which is more
        than enough for a 6-seat table; tests that want deterministic draws
        clear `self.shoe` to force the `draw_card` fallback they monkeypatch.
        """
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
            self.insurance_offered = True
            return

        if up is not None and up.rank in ("J", "Q", "K", "10") and is_blackjack(cards=self.dealer):
            self.peeked_blackjack = True
            for player in self.players:
                for hand in player.hands:
                    hand.finished = True
            self.phase = "settled"
            self.finished = True
            self.dealer_played = True
            return

        self.phase = "player_actions"
        for player in self.players:
            for hand in player.hands:
                if hand.is_blackjack():
                    hand.finished = True
        self._advance_or_finish()

    def take_insurance(self, user_id: int, amount: int) -> None:
        """Records an insurance side bet for the player.

        Args:
            user_id: Discord user ID placing the insurance.
            amount: Side-bet amount; must equal `participant.bet // 2`.
        """
        if self.phase != "insurance":
            raise ValueError("Insurance is not currently offered")
        player = self._find_player(user_id=user_id)
        if player.insurance_resolved:
            raise ValueError("Insurance already decided")
        expected = player.participant.bet // 2
        if expected <= 0 or amount <= 0:
            raise ValueError("Insurance side bet must be positive")
        if amount != expected:
            raise ValueError("Insurance amount must equal half of the original bet")
        balance_remaining = player.participant.balance_at_start - committed_wagers(player=player)
        if not can_insure(player=player, balance_remaining=balance_remaining):
            raise ValueError("Not enough balance for insurance")
        player.insurance_bet = amount
        player.insurance_resolved = True
        self._maybe_close_insurance_phase()

    def decline_insurance(self, user_id: int) -> None:
        """Records that the player has declined to take insurance.

        Args:
            user_id: Discord user ID declining the insurance offer.
        """
        if self.phase != "insurance":
            raise ValueError("Insurance is not currently offered")
        player = self._find_player(user_id=user_id)
        if player.insurance_resolved:
            raise ValueError("Insurance already decided")
        player.insurance_resolved = True
        self._maybe_close_insurance_phase()

    def decline_insurance_for_all_unresolved(self) -> None:
        """Marks every undecided player as declining insurance.

        Used by view timeouts and forced-finish paths so the round can leave
        the insurance phase even when one of the players never clicked.
        """
        if self.phase != "insurance":
            return
        for player in self.players:
            if not player.insurance_resolved:
                player.insurance_resolved = True
        self._maybe_close_insurance_phase()

    def active_player(self) -> BlackjackPlayerHand | None:
        """Returns the player whose turn is active, if any."""
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
        """Returns the active sub-hand of the active player, if any."""
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

    def hit(self, user_id: int) -> Card:
        """Draws one card for the active sub-hand.

        Args:
            user_id: Discord user ID that must match the active player.

        Returns:
            The drawn card.

        Raises:
            ValueError: The user is not the active player.
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
        """Marks the active sub-hand as standing and advances the table."""
        _, hand = self._require_active(user_id=user_id)
        hand.finished = True
        self._advance_or_finish()

    def double_down(self, user_id: int) -> Card:
        """Doubles the active hand's wager, draws one card, then finishes it.

        Args:
            user_id: Discord user ID that must match the active player.

        Returns:
            The single card drawn after the bet was doubled.
        """
        player, hand = self._require_active(user_id=user_id)
        balance_remaining = player.participant.balance_at_start - committed_wagers(player=player)
        if not can_double(hand=hand, balance_remaining=balance_remaining):
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

        Each sibling gets the matching original card plus one fresh draw.
        Splitting Aces marks both siblings as `is_split_aces` and finishes
        them after a single draw, matching standard house rules.

        Args:
            user_id: Discord user ID that must match the active player.
        """
        player, hand = self._require_active(user_id=user_id)
        balance_remaining = player.participant.balance_at_start - committed_wagers(player=player)
        if not can_split(hand=hand, balance_remaining=balance_remaining):
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
        """Surrenders the active hand for a half-bet refund."""
        _, hand = self._require_active(user_id=user_id)
        if not can_surrender(hand=hand, peeked_blackjack=self.peeked_blackjack):
            raise ValueError("Cannot surrender this hand")
        hand.surrendered = True
        hand.finished = True
        hand.actions_taken += 1
        self._advance_or_finish()

    def stand_all_remaining(self) -> None:
        """Marks every unresolved hand as standing, then finishes the table."""
        if self.phase == "insurance":
            self.decline_insurance_for_all_unresolved()
        for player in self.players:
            for hand in player.hands:
                hand.finished = True
        self._finish_after_players_done()

    def dealer_total(self) -> int:
        """Returns the current best total for the dealer hand."""
        return hand_value(cards=self.dealer)

    def dealer_visible_value(self) -> int:
        """Returns the visible dealer card value for hint prompts."""
        return dealer_visible_value(dealer=self.dealer)

    def dealer_is_soft_total(self) -> tuple[bool, int]:
        """Returns `(is_soft, total)` for the dealer hand."""
        return is_soft_total(cards=self.dealer)

    def dealer_is_soft_17(self) -> bool:
        """Returns whether the dealer hand is currently a soft 17."""
        return is_soft_17(cards=self.dealer)

    def needs_dealer_play(self) -> bool:
        """Returns whether the dealer still needs a draw/stand phase."""
        return self._needs_dealer_play()

    def draw_dealer_card(self) -> Card:
        """Draws one card into the dealer hand and returns it."""
        card = self._draw_one_card()
        self.dealer.append(card)
        return card

    def mark_dealer_played(self) -> None:
        """Marks the dealer phase complete."""
        self.dealer_played = True
        self.finished = True
        self.phase = "settled"

    def _find_player(self, user_id: int) -> BlackjackPlayerHand:
        """Returns the player by user_id or raises when unknown."""
        for player in self.players:
            if player.participant.user_id == user_id:
                return player
        raise ValueError("Unknown user for this round")

    def _require_active(self, user_id: int) -> tuple[BlackjackPlayerHand, BlackjackHandState]:
        """Returns the active (player, hand) tuple or raises when not turn."""
        if self.phase != "player_actions":
            raise ValueError("Not in player action phase")
        player = self.active_player()
        hand = self.active_hand()
        if player is None or hand is None or player.participant.user_id != user_id:
            raise ValueError("Not this player's turn")
        return player, hand

    def _maybe_close_insurance_phase(self) -> None:
        """Closes the insurance phase, peeks the hole card, and advances.

        Runs only once every player has either taken or declined insurance.
        A natural Blackjack peek short-circuits the round to `settled`; a
        non-peek pushes the round into `player_actions` and auto-finishes
        any player who was already dealt a natural Blackjack.
        """
        if self.phase != "insurance":
            return
        if not all(player.insurance_resolved for player in self.players):
            return
        if is_blackjack(cards=self.dealer):
            self.peeked_blackjack = True
            for player in self.players:
                for hand in player.hands:
                    hand.finished = True
            self.phase = "settled"
            self.finished = True
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
        """Finishes the round after all player actions have resolved."""
        if self.finished:
            return
        if self._needs_dealer_play() and self.auto_play_dealer:
            self.phase = "dealer"
            self._play_dealer()
        self.finished = True
        self.phase = "settled"

    def _needs_dealer_play(self) -> bool:
        """Returns whether the dealer must draw before settlement."""
        if self.peeked_blackjack:
            return False
        if is_blackjack(cards=self.dealer):
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

    def _play_dealer(self) -> None:
        """Draws dealer cards under H17 rules (hits soft 17, stands hard 17+)."""
        while True:
            total = hand_value(cards=self.dealer)
            if total < 17:
                self.draw_dealer_card()
                continue
            if total == 17 and is_soft_17(cards=self.dealer):
                self.draw_dealer_card()
                continue
            break
        self.mark_dealer_played()


def render_hand(cards: list[Card], hide_first: bool = False) -> str:
    """Formats a hand for display.

    Args:
        cards: Cards to render.
        hide_first: Whether to replace the first card with a hidden-card marker.

    Returns:
        A space-separated display string for the hand.
    """
    if hide_first and cards:
        rest = " ".join(str(card) for card in cards[1:])
        return f"🂠 {rest}".strip()
    return " ".join(str(card) for card in cards)


def dealer_visible_value(dealer: list[Card]) -> int:
    """Returns the numeric value of the dealer's visible card.

    The second dealer card is visible while the first card is hidden. If only
    one card exists, that card is treated as visible.

    Args:
        dealer: Dealer cards in draw order.

    Returns:
        The visible card's Blackjack value, or 0 when the dealer has no cards.
    """
    up = dealer_up_card(dealer=dealer)
    if up is None:
        return 0
    if up.rank == "A":
        return 11
    if up.rank in ("J", "Q", "K"):
        return 10
    return int(up.rank)
