"""Pure rules engine for the multiplayer Blackjack table behind `/games blackjack`.

Everything here is side-effect free: no Discord object, no database, no clock and no LLM. A round
is driven entirely by the `random.Random` it is handed (production passes `random.SystemRandom()`,
the tests a seeded `Random`), so a whole table replays deterministically. The Discord surface is
`blackjack_views.py`, the money is `settlement.py`, the bot player's decisions are `bot_player.py`
/ `blackjack_ev.py`, and the cross-round shoe is `shoe.py`; all of them read this file and none of
it reads them.

Three layers live here:

- Hand vocabulary. `hand_value` and the `is_*` predicates each answer one question about a list of
  `Card`s, and are the only place ace demotion and 10/J/Q/K equivalence are decided. The EV engine,
  the bot player and the history store all evaluate hands through them.
- `BlackjackRound`, the mutable table. `BlackjackPlayerHand` is one seat, `BlackjackHandState` one
  of its hands (two only after a Split), and the round walks `phase` through insurance ->
  player_actions -> dealer -> settled while the `can_*` predicates say which controls a view may
  still offer.
- `settle_hand`, which turns one finished hand plus the final dealer cards into an outcome label
  and the dealer-paid delta. That delta is the whole of what this file decides: the system-funded
  過五關 21 bonus, the VIP multiplier and the wallet write belong to `settlement.py`.

The house rules encoded here are not the same everywhere, so they are worth naming: the dealer
hits soft 17, a natural pays 3:2, Late Surrender ends a hand at half the original bet rounded up,
Double after Split is a caller-supplied flag the views never set, no hand can be re-split (so a
seat holds at most two), split Aces take exactly one card each and cannot be hit again, a
split-derived two-card 21 is not a natural, and 過五關 pays any five-or-more-card non-bust hand
whatever the dealer holds — with the five-card 21 the one exception that still needs the dealer to
play, since its main leg pushes against a dealer 21.

Two seams exist for the callers rather than for this file's own use. `auto_play_dealer` decides
whether the round draws the dealer's cards the moment the last player finishes; the views set it
False and animate the draws one message edit at a time through `needs_dealer_play` /
`draw_dealer_card` / `mark_dealer_played`. And a round deals from a finite FIFO `shoe` so that card
counting has signal, with `shoe.py` carrying it between rounds in the same channel; `draw_card`'s
notional infinite shoe is left as the fallback for an empty one and as the seam the tests
monkeypatch.
"""

from random import Random
from typing import Final, Literal

from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.games import Card, SettleOutcome, GameParticipant
from discordbot.typings.economy import MAX_SINGLE_BET

RoundPhase = Literal["insurance", "player_actions", "dealer", "settled"]


SHOE_DECK_COUNT = 4
# Natural Blackjack pays 3:2.
_BLACKJACK_PAYOUT_NUM: Final[int] = 3
_BLACKJACK_PAYOUT_DEN: Final[int] = 2
_CARD_RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
_CARD_SUITS = ("♠", "♥", "♦", "♣")


def draw_card(rng: Random) -> Card:
    """Draws one card from a notional infinite shoe (independent rank + suit).

    Production paths build a shuffled 4-deck shoe inside `BlackjackRound` and
    draw from there, so rank/suit frequency is finite even though duplicate
    rank/suit labels can appear from different decks. This helper stays as the
    round-bootstrap fallback (and the test-monkeypatch seam) for when the shoe
    is empty.

    Args:
        rng (Random): Random source used to choose rank and suit.

    Returns:
        The drawn card.
    """
    return Card(rank=rng.choice(seq=_CARD_RANKS), suit=rng.choice(seq=_CARD_SUITS))


def build_shoe(rng: Random, deck_count: int = SHOE_DECK_COUNT) -> list[Card]:
    """Returns a shuffled multi-deck shoe (default 4 decks = 208 cards).

    Cards are popped from index 0, so the head of the list is the next card out. 208 cards is far
    more than the worst-case single round needs, and `shoe.py` cuts a fresh shoe before a round
    starts once the remainder drops under its threshold, which together are what stop a shoe
    emptying mid-round into the `draw_card` fallback and corrupting the running count.

    Args:
        rng (Random): Random source used to shuffle the built shoe.
        deck_count (int): Number of 52-card decks to stack into the shoe.

    Returns:
        The shuffled shoe, next card first.
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
        cards (list[Card]): Cards to evaluate.

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
    """Returns the Blackjack value used for pair and up-card checks.

    An Ace is always 11 here, never demoted, which is what keeps A + 10 out of `is_pair`. Read by
    `bot_player.py` too, so a change here moves the bot's decisions as well as the Split guard.

    Args:
        card (Card): Card to value.

    Returns:
        11 for an Ace, 10 for a face card, otherwise the rank's own number.
    """
    if card.rank == "A":
        return 11
    if card.rank in ("J", "Q", "K"):
        return 10
    return int(card.rank)


def is_blackjack(cards: list[Card]) -> bool:
    """Returns whether a hand is a natural Blackjack.

    Card-count only: a hand that reached 21 in three cards, and a split hand's two-card 21, are
    both excluded (the split exclusion is `BlackjackHandState.is_blackjack`'s job, since only the
    hand state knows where it came from).

    Args:
        cards (list[Card]): Cards to evaluate.

    Returns:
        True only when the hand has exactly two cards summing to 21.
    """
    return len(cards) == 2 and hand_value(cards=cards) == 21


def is_five_card_twenty_one(cards: list[Card]) -> bool:
    """Returns whether a hand has five or more cards totaling 21.

    The 過五關 21, the one five-card hand whose main leg is not an automatic win: it pushes
    against a dealer 21, and pays a separate system-funded bonus that `settlement.py` mints.

    Args:
        cards (list[Card]): Cards to evaluate.

    Returns:
        True only when the hand has at least five cards and totals 21.
    """
    return len(cards) >= 5 and hand_value(cards=cards) == 21


def is_five_card_win(cards: list[Card]) -> bool:
    """Returns whether a hand qualifies for the non-bust five-card win.

    The plain 過五關: five or more cards without busting beats whatever the dealer ends on. `hit`
    reads it to auto-stand the hand the moment it lands.

    Args:
        cards (list[Card]): Cards to evaluate.

    Returns:
        True when the hand holds at least five cards and has not busted.
    """
    return len(cards) >= 5 and hand_value(cards=cards) <= 21


def is_bust(cards: list[Card]) -> bool:
    """Returns whether a hand is over 21.

    Args:
        cards (list[Card]): Cards to evaluate.

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
        cards (list[Card]): Cards to evaluate; only meaningful on exactly two cards.

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
        cards (list[Card]): Cards to evaluate.

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

    The dealer's one discretionary rule keys off this: H17 means the dealer draws on a soft 17 and
    stands on a hard one.

    Args:
        cards (list[Card]): Cards to evaluate.

    Returns:
        True only when the hand totals 17 with at least one Ace still
        counted as 11.
    """
    soft, total = is_soft_total(cards=cards)
    return soft and total == 17


def dealer_up_card(dealer: list[Card]) -> Card | None:
    """Returns the dealer's visible up-card.

    The first dealt card is the hole card (hidden); the second is the up-card. That ordering is
    this file's convention and `render_hand(hide_first=True)` is its other half, so a caller that
    reorders the dealer list shows the wrong card rather than getting an error.

    Args:
        dealer (list[Card]): Dealer's cards in draw order.

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
    `BlackjackPlayerHand.hands`. Re-splitting is refused, so two is the ceiling.

    `bet` is what the round settles and `base_bet` what it was dealt at: they differ only after a
    Double Down, and both the Surrender refund and a Split sibling's stake read `base_bet`, so
    neither follows a doubled wager.

    Attributes:
        cards: Cards currently held in this hand.
        bet: Active wager for this hand (doubled after Double Down).
        base_bet: Original wager, read by the Surrender refund and by the Split sibling's stake.
        finished: True once this hand no longer needs Hit / Stand actions.
        doubled: True after a Double Down on this hand.
        surrendered: True after a Surrender on this hand.
        is_split_hand: True when this hand came out of a Split.
        is_split_aces: True when both split halves came from an Ace pair.
        actions_taken: Hit / Double / Surrender counter used by the action guards. Only an
            untouched hand may Double, Split or Surrender, so this reaching 1 closes all three.
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
        """Returns whether this sub-hand is a natural Blackjack.

        A split half that draws to 21 in two cards is deliberately not one, so it never collects
        the 3:2 payout; `settle_hand` routes it to the even-money path instead.
        """
        return not self.is_split_hand and is_blackjack(cards=self.cards)

    def is_bust(self) -> bool:
        """Returns whether this sub-hand has busted."""
        return is_bust(cards=self.cards)


class BlackjackPlayerHand(BaseModel):
    """Container for one participant's hands at a multiplayer table.

    Holds the original `GameParticipant` plus one or more
    `BlackjackHandState` rows. Split adds a second entry; everything else
    keeps a single hand entry.

    The seat is what the balance is measured against, not the hand: `participant.balance_at_start`
    is a snapshot taken when the round began, so every affordability check subtracts
    `committed_wagers` from it rather than re-reading a wallet that may have moved since.

    Attributes:
        participant: Discord player and wager metadata.
        hands: All active sub-hands in display order.
        insurance_bet: Insurance side bet amount, `0` when none was taken.
        insurance_resolved: True once this player has taken or declined insurance, including the
            forced decline a timeout applies. Every seat carrying it is what closes the phase, so
            it is set before the round leaves `insurance`, not after.
    """

    participant: GameParticipant = Field(..., description="Discord player and wager metadata.")
    hands: list[BlackjackHandState] = Field(
        default_factory=list, description="All active sub-hands in display order."
    )
    insurance_bet: int = Field(
        default=0, description="Insurance side bet amount, 0 when none was taken."
    )
    insurance_resolved: bool = Field(
        default=False, description="True once the player has made an insurance choice."
    )

    @property
    def finished(self) -> bool:
        """Returns True once every owned hand has finished.

        A seat holding no hands at all reads as unfinished, so a half-built player cannot let the
        turn cursor skip straight past it on an empty `all()`.
        """
        return bool(self.hands) and all(hand.finished for hand in self.hands)


def committed_wagers(player: BlackjackPlayerHand) -> int:
    """Returns the total points already committed for one participant.

    Sums every active hand bet plus any insurance side bet, so callers can
    measure how much of the player's starting balance is still spoken for
    when validating Double / Split / Insurance affordability.

    Args:
        player (BlackjackPlayerHand): The player whose committed wagers should be summed.

    Returns:
        Total committed points across hands and insurance for the player.
    """
    return sum(hand.bet for hand in player.hands) + player.insurance_bet


def can_double(
    hand: BlackjackHandState, balance_remaining: int, allow_after_split: bool = False
) -> bool:
    """Returns whether Double Down is allowed on this hand right now.

    `allow_after_split` defaults off and the views never pass it, so Double after Split is closed
    in production; the parameter exists so the rule can be flipped in one place.

    Args:
        hand (BlackjackHandState): Hand to inspect.
        balance_remaining (int): Points still available after current commitments.
        allow_after_split (bool): Whether the house rule permits Double after Split.

    Returns:
        True only when no action has been taken on a two-card hand, the DAS rule allows it, the
        doubled stake still fits `MAX_SINGLE_BET`, and the player can afford the extra wager.
    """
    if hand.finished or hand.surrendered or hand.doubled:
        return False
    if len(hand.cards) != 2 or hand.actions_taken != 0:
        return False
    if hand.is_split_hand and not allow_after_split:
        return False
    # Doubling doubles the hand's stake; keep it within the single-bet cap so it
    # cannot bypass the anti-inflation guardrail that bounds every wager.
    if hand.bet * 2 > MAX_SINGLE_BET:
        return False
    return balance_remaining >= hand.bet


def can_split(hand: BlackjackHandState, balance_remaining: int) -> bool:
    """Returns whether Split is allowed on this hand right now.

    A hand that already came out of a Split is refused, so re-splitting is impossible and a seat
    never holds more than two hands.

    Args:
        hand (BlackjackHandState): Hand to inspect.
        balance_remaining (int): Points still available after current commitments.

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

    Reads the seat alone: whether the table is even offering insurance is the round's `phase`,
    which `take_insurance` checks before calling here. The side bet is always half the original
    participant bet, so a 1-point seat rounds down to 0 and is refused rather than insured free.

    Args:
        player (BlackjackPlayerHand): Player container to inspect.
        balance_remaining (int): Points still available after current commitments.

    Returns:
        True only when this player has not yet decided, has no insurance bet on the table, and the
        half-bet side wager fits the remaining balance.
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
        hand (BlackjackHandState): Hand to inspect.
        peeked_blackjack (bool): Whether the dealer already peeked a Blackjack;
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
    """Resolves a split-derived two-card 21 without treating it as natural Blackjack.

    It wins even money instead of 3:2, pushes against any other dealer 21, and still loses
    outright to a dealer natural.

    Args:
        hand (BlackjackHandState): The split half holding a two-card 21.
        dealer (list[Card]): Final dealer cards.

    Returns:
        `(outcome, delta)` for this hand alone.
    """
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
    """Resolves a finished non-surrender, non-special Blackjack sub-hand.

    Ordinary Blackjack comparison: naturals first (a mutual one pushes), then either side's bust,
    then the higher total. The natural pays `_BLACKJACK_PAYOUT_NUM / _BLACKJACK_PAYOUT_DEN`
    floored, so an odd bet keeps its 3:2 rounded in the casino's favour.

    Args:
        hand (BlackjackHandState): Finished sub-hand to resolve.
        dealer (list[Card]): Final dealer cards.

    Returns:
        `(outcome, delta)` for this hand alone.
    """
    bet = hand.bet
    player_total = hand.total()
    dealer_total = hand_value(cards=dealer)
    player_bj = hand.is_blackjack()
    dealer_bj = is_blackjack(cards=dealer)

    if player_bj and dealer_bj:
        outcome: SettleOutcome = "push"
        delta = 0
    elif player_bj:
        outcome, delta = "blackjack", int(bet * _BLACKJACK_PAYOUT_NUM // _BLACKJACK_PAYOUT_DEN)
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

    The delta is only what the casino pays: the system-funded 過五關 21 bonus is minted by
    `settlement.py` off the returned `five_card_twenty_one` label, and the VIP multiplier and the
    wallet write happen there too.

    The order of the checks is the rule set. Surrender short-circuits to a loss of half the hand's
    `base_bet` rounded up, never its `bet`. Both 過五關 checks run before
    either Blackjack check, so a split half that reaches five cards still earns them, and the
    five-card 21 is tested first because it is also a five-card win but pushes against a dealer 21
    instead of winning outright. A split-derived two-card 21 is settled before the natural path so
    it never collects 3:2. The `doubled` guard on the 過五關 checks is belt and braces: a Double
    draws exactly one card and finishes the hand, so a doubled hand holds three cards at most.

    Args:
        hand (BlackjackHandState): Finished sub-hand to settle.
        dealer (list[Card]): Final dealer cards.

    Returns:
        `(outcome, delta)` where delta is the signed point change for
        this single hand.

    Raises:
        ValueError: The hand has not finished, so its outcome is not yet decided.
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

    `phase` is the lifecycle and each transition is one-way: a round opens in `insurance` only
    when the up-card is an Ace, otherwise straight into `player_actions`, and reaches `settled`
    either through the dealer or through a peeked Blackjack that ends it before anyone acts. Every
    player action refuses to run outside the phase that owns it, so a view may gate its buttons on
    `phase` without becoming the thing that enforces the rule.

    The turn cursor is the `(current_player_index, current_hand_index)` pair and only
    `_advance_or_finish` moves it. Because `active_player` / `active_hand` call it to skip past
    what is already finished, reading the active seat is itself a mutation.

    Attributes:
        rng: Random source used for card draws.
        players: Per-player containers (each holds one or more sub-hands).
        dealer: Dealer cards shared by the table, hole card first.
        shoe: Cards left in the round's FIFO shoe; the caller persists this list, not the one it
            passed to `from_participants`.
        current_player_index: Index of the player whose turn is active.
        current_hand_index: Index of the active sub-hand within that player.
        dealer_played: True once the dealer has drawn for all standing
            players.
        finished: True once no more player actions remain.
        auto_play_dealer: True when the pure rules should draw dealer cards
            synchronously after player actions finish. The views set it False and animate the
            dealer themselves.
        phase: Lifecycle phase of the round (insurance / player_actions /
            dealer / settled).
        insurance_offered: True only when the dealer up-card is an Ace.
        peeked_blackjack: True once the dealer's hole-card peek revealed a
            natural Blackjack.
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
    finished: bool = Field(default=False, description="True once no more player actions remain.")
    auto_play_dealer: bool = Field(
        default=True,
        description="True when dealer cards are drawn synchronously after player actions finish.",
    )
    phase: RoundPhase = Field(
        default="player_actions", description="Lifecycle phase of the round."
    )
    insurance_offered: bool = Field(
        default=False, description="True only when the dealer up-card is an Ace."
    )
    peeked_blackjack: bool = Field(
        default=False,
        description="True once the dealer's hole-card peek revealed a natural Blackjack.",
    )

    @classmethod
    def from_participants(
        cls,
        rng: Random,
        participants: list[GameParticipant],
        auto_play_dealer: bool = True,
        shoe: list[Card] | None = None,
    ) -> "BlackjackRound":
        """Builds a round from registered lobby participants.

        Deals nobody in: the table is seated with one empty hand each and `deal_initial` is a
        separate call, so a caller can inject cards before the deal.

        When `shoe` is provided the round deals from it (a persistent per-channel
        shoe carried across rounds for card counting); otherwise a fresh shuffled
        multi-deck shoe is built. The shoe is validated into `round_state.shoe`,
        which may be a copy of the passed list, so callers persist card depletion
        by saving `round_state.shoe` after the round, not the list passed in.

        Args:
            rng (Random): Random source for the shoe shuffle and every later draw.
            participants (list[GameParticipant]): Seats in the order they will act.
            auto_play_dealer (bool): Whether the round draws the dealer's cards itself once the
                last player finishes.
            shoe (list[Card] | None): Shoe to deal from, or None to build a fresh shuffled one.

        Returns:
            The seated round, before any card has been dealt.
        """
        players = [
            BlackjackPlayerHand(
                participant=participant,
                hands=[BlackjackHandState(bet=participant.bet, base_bet=participant.bet)],
            )
            for participant in participants
        ]
        return cls(
            rng=rng,
            players=players,
            auto_play_dealer=auto_play_dealer,
            shoe=shoe if shoe is not None else build_shoe(rng=rng),
        )

    def _draw_one_card(self) -> Card:
        """Pops the next card from the round's shoe, falling back when empty.

        Cards come from the FIFO shoe, so draws are capped by the finite
        multi-deck shoe instead of independent replacement. The 4-deck shoe
        holds 208 cards which is more than enough for a 6-seat table; tests
        that want deterministic draws clear `self.shoe` to force the
        `draw_card` fallback they monkeypatch.

        Returns:
            The next card out of the shoe, or a freshly rolled one once it is empty.
        """
        if self.shoe:
            return self.shoe.pop(0)
        return draw_card(rng=self.rng)

    def deal_initial(self) -> None:
        """Deals two cards to every player and two cards to the dealer.

        The dealer's first card is the hole and its second the up-card, which then drives the
        post-deal lifecycle:
        - Up-card is Ace: enter `insurance` phase and let players decide
          before peeking the hole card. The peek runs at the close of the
          insurance phase.
        - Up-card is a 10-value card: peek silently (no insurance offered);
          if the peek reveals a Blackjack the round settles immediately.
        - Anything else: jump straight to `player_actions`.

        Whichever way the round reaches `player_actions` (here, or at the close of insurance) a
        player dealt a natural is finished on the spot, so the cursor never stops on a hand that
        has nothing left to decide.
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

        Closes the insurance phase, and with it peeks the hole card, once this was the last
        undecided seat.

        Args:
            user_id (int): Discord user ID placing the insurance.
            amount (int): Side-bet amount; must equal `participant.bet // 2`.

        Raises:
            ValueError: The round is not in the insurance phase, the user is not seated, this
                player already decided, the amount is not exactly half the original bet, or the
                balance cannot cover it.
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

        Closes the insurance phase, and with it peeks the hole card, once this was the last
        undecided seat.

        Args:
            user_id (int): Discord user ID declining the insurance offer.

        Raises:
            ValueError: The round is not in the insurance phase, the user is not seated, or this
                player already decided.
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
        the insurance phase even when one of the players never clicked. A no-op outside that
        phase, so a timeout firing late cannot disturb a round already in play.
        """
        if self.phase != "insurance":
            return
        for player in self.players:
            if not player.insurance_resolved:
                player.insurance_resolved = True
        self._maybe_close_insurance_phase()

    def active_player(self) -> BlackjackPlayerHand | None:
        """Returns the player whose turn is active, if any.

        Not a pure read: a seat that has already finished advances the turn cursor and, once the
        last one does, settles the round.
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

        Advances the turn cursor past finished hands the same way `active_player` does, so this is
        a mutation too.
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

    def hit(self, user_id: int) -> Card:
        """Draws one card for the active sub-hand.

        Finishes the hand and advances the table on a bust or on the fifth non-bust card, since
        過五關 auto-stands rather than letting the player keep drawing.

        Args:
            user_id (int): Discord user ID that must match the active player.

        Returns:
            The drawn card.

        Raises:
            ValueError: The round is not in the player-action phase, the user is not the active
                player, or the hand came out of a Split of Aces.
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

        Propagates `_require_active`'s `ValueError` when the round is not in the player-action
        phase or the caller is not the active player.

        Args:
            user_id (int): Discord user ID that must match the active player.
        """
        _, hand = self._require_active(user_id=user_id)
        hand.finished = True
        self._advance_or_finish()

    def double_down(self, user_id: int) -> Card:
        """Doubles the active hand's wager, draws one card, then finishes it.

        Only `bet` doubles; `base_bet` keeps the original wager. The hand is finished after the
        single card, so a doubled hand never acts again.

        Args:
            user_id (int): Discord user ID that must match the active player.

        Returns:
            The single card drawn after the bet was doubled.

        Raises:
            ValueError: The round is not in the player-action phase, the user is not the active
                player, or `can_double` refuses the hand.
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

        Each sibling gets the matching original card plus one fresh draw, both staked at
        `base_bet`. Splitting Aces marks both siblings as `is_split_aces` and finishes
        them after a single draw, matching standard house rules.

        The new hand is inserted directly after the active one, so the sibling is played next
        rather than behind the rest of the seat's hands.

        Args:
            user_id (int): Discord user ID that must match the active player.

        Raises:
            ValueError: The round is not in the player-action phase, the user is not the active
                player, or `can_split` refuses the hand.
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
        """Surrenders the active hand, conceding half its original wager.

        Args:
            user_id (int): Discord user ID that must match the active player.

        Raises:
            ValueError: The round is not in the player-action phase, the user is not the active
                player, or `can_surrender` refuses the hand.
        """
        _, hand = self._require_active(user_id=user_id)
        if not can_surrender(hand=hand, peeked_blackjack=self.peeked_blackjack):
            raise ValueError("Cannot surrender this hand")
        hand.surrendered = True
        hand.finished = True
        hand.actions_taken += 1
        self._advance_or_finish()

    def stand_all_remaining(self) -> None:
        """Marks every unresolved hand as standing, then finishes the table.

        The view's timeout path. Insurance is declined for anyone still undecided first, since a
        round abandoned during that phase has to leave it before it can settle.
        """
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
        """Returns the value of the dealer's up-card, hole card excluded."""
        return dealer_visible_value(dealer=self.dealer)

    def dealer_is_soft_17(self) -> bool:
        """Returns whether the dealer hand is currently a soft 17."""
        return is_soft_17(cards=self.dealer)

    def needs_dealer_play(self) -> bool:
        """Returns whether the dealer still needs a draw/stand phase.

        The animated dealer's entry point: the view asks this before it starts editing, so a round
        already decided never shows a dealer turn that changes nothing.
        """
        return self._needs_dealer_play()

    def draw_dealer_card(self) -> Card:
        """Draws one card into the dealer hand.

        One step of the animated dealer, so the caller owns the H17 decision and the stopping
        condition; `_play_dealer` is the same loop run to completion in one go.

        Returns:
            The card added to the dealer's hand.
        """
        card = self._draw_one_card()
        self.dealer.append(card)
        return card

    def mark_dealer_played(self) -> None:
        """Closes the dealer phase and settles the round.

        The animated dealer's exit: it lands the round in exactly the state `_play_dealer` would
        have left it in, so both paths reach settlement identically.
        """
        self.dealer_played = True
        self.finished = True
        self.phase = "settled"

    def _find_player(self, user_id: int) -> BlackjackPlayerHand:
        """Returns the seat belonging to a user.

        Args:
            user_id (int): Discord user ID to look up.

        Returns:
            The player container for that user.

        Raises:
            ValueError: No seat at this table belongs to that user.
        """
        for player in self.players:
            if player.participant.user_id == user_id:
                return player
        raise ValueError("Unknown user for this round")

    def _require_active(self, user_id: int) -> tuple[BlackjackPlayerHand, BlackjackHandState]:
        """Returns the active seat and hand, refusing anyone whose turn it is not.

        The single gate every player action goes through, which is why none of them re-checks the
        phase or the identity itself.

        Args:
            user_id (int): Discord user ID the action arrived from.

        Returns:
            The active `(player, hand)` pair.

        Raises:
            ValueError: The round is not in the player-action phase, no hand is active, or the
                active hand belongs to someone else.
        """
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
        """Skips completed sub-hands and settles the table when none remain.

        The only writer of the turn cursor. It walks hands within a seat before moving to the next
        seat, so a Split is played out in place rather than queued behind the other players.
        """
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
        """Finishes the round after all player actions have resolved.

        Draws the dealer here only under `auto_play_dealer`. With it off the round still lands in
        `settled` but `dealer_played` stays False, which is the signal the caller reads before
        animating the dealer itself through `needs_dealer_play` / `draw_dealer_card` /
        `mark_dealer_played`.
        """
        if self.finished:
            return
        if self._needs_dealer_play() and self.auto_play_dealer:
            self.phase = "dealer"
            self._play_dealer()
        self.finished = True
        self.phase = "settled"

    def _needs_dealer_play(self) -> bool:
        """Returns whether the dealer must draw before settlement.

        False once nothing is left that a dealer total could still change: a peeked or dealt
        dealer natural ends the round, and surrendered, natural, busted and plain 過五關 hands are
        each already decided. A five-card 21 is the exception that keeps the dealer playing, since
        its main leg pushes against a dealer 21 and wins against anything else.
        """
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
        """Draws dealer cards under H17 rules (hits soft 17, stands hard 17+).

        The `auto_play_dealer` path only. The animated path in `blackjack_views.py` runs the same
        rule step by step and records each decision, so a change to H17 here has to be made there
        too.
        """
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

    Lives here rather than in the views because `hide_first` encodes the same hole-card-first
    convention `dealer_up_card` reads, and the two have to agree on which card stays covered.

    Args:
        cards (list[Card]): Cards to render.
        hide_first (bool): Whether to replace the first card with a hidden-card marker.

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
    one card exists, that card is treated as visible. An Ace is 11 here, the same fixed value
    `_card_blackjack_value` uses, since a single card has nothing to demote against.

    Args:
        dealer (list[Card]): Dealer cards in draw order.

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
