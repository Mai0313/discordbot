"""Pure Blackjack rules and shoe helpers.

Kept side-effect free so the unit tests can drive deterministic deals via a
fixed `rng`. The cog wires this up with `random.SystemRandom()` for production.
"""

from random import Random
from typing import Literal
from dataclasses import field, dataclass

_RANKS: tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
_SUITS: tuple[str, ...] = ("♠", "♥", "♦", "♣")
_BLACKJACK_TARGET = 21
_DEALER_STAND_AT = 17
_BLACKJACK_HAND_SIZE = 2

OutcomeLabel = Literal["win", "lose", "push", "blackjack", "player_bust", "dealer_bust"]


@dataclass(frozen=True)
class Card:
    """A single playing card.

    Attributes:
        rank: One of A, 2-10, J, Q, K.
        suit: One of the four unicode suit glyphs.
    """

    rank: str
    suit: str

    def __str__(self) -> str:
        """Human-readable label like ``A♠``."""
        return f"{self.rank}{self.suit}"


def draw_card(rng: Random) -> Card:
    """Draws a single card from a notional infinite shoe.

    Using an infinite shoe (independent draws) intentionally, we avoid having
    to track a deck across hands, and the mathematical edge difference is
    negligible at this stake level.

    Args:
        rng: Random source used to choose rank and suit.

    Returns:
        The drawn card.
    """
    rank = rng.choice(seq=_RANKS)
    suit = rng.choice(seq=_SUITS)
    return Card(rank=rank, suit=suit)


def hand_value(cards: list[Card]) -> int:
    """Returns the best Blackjack value for a hand.

    Aces start at 11 each and are demoted to 1 one at a time while the total
    is over 21. Returns the over-21 total when no aces remain to demote, so
    callers can detect a bust by checking ``> 21``.

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
    while total > _BLACKJACK_TARGET and aces > 0:
        total -= 10
        aces -= 1
    return total


def is_blackjack(cards: list[Card]) -> bool:
    """Returns whether a hand is a natural Blackjack.

    Args:
        cards: Cards to evaluate.

    Returns:
        True only when the hand has exactly two cards summing to 21.
    """
    return len(cards) == _BLACKJACK_HAND_SIZE and hand_value(cards=cards) == _BLACKJACK_TARGET


def is_bust(cards: list[Card]) -> bool:
    """Returns whether a hand is over 21.

    Args:
        cards: Cards to evaluate.

    Returns:
        True when the hand total is greater than 21.
    """
    return hand_value(cards=cards) > _BLACKJACK_TARGET


@dataclass
class BlackjackHand:
    """Mutable state for one Blackjack round.

    Attributes:
        rng: Random source; injectable for tests.
        bet: Original bet amount (in points).
        player: Player's cards.
        dealer: Dealer's cards.
        finished: True once both sides have stopped drawing.
    """

    rng: Random
    bet: int
    player: list[Card] = field(default_factory=list)
    dealer: list[Card] = field(default_factory=list)
    finished: bool = False

    def deal_initial(self) -> None:
        """Deals two cards to the player and two to the dealer."""
        self.player = [draw_card(rng=self.rng), draw_card(rng=self.rng)]
        self.dealer = [draw_card(rng=self.rng), draw_card(rng=self.rng)]
        # Natural Blackjack short-circuits the round; player can't act.
        if is_blackjack(cards=self.player) or is_blackjack(cards=self.dealer):
            self.finished = True

    def hit(self) -> Card:
        """Draws one card for the player.

        Ends the round if the player's hand busts.

        Returns:
            The card drawn for the player.
        """
        card = draw_card(rng=self.rng)
        self.player.append(card)
        if is_bust(cards=self.player):
            self.finished = True
        return card

    def stand(self) -> None:
        """Player stops; dealer draws to ``_DEALER_STAND_AT`` then resolves."""
        while hand_value(cards=self.dealer) < _DEALER_STAND_AT:
            self.dealer.append(draw_card(rng=self.rng))
        self.finished = True

    def player_total(self) -> int:
        """Returns the current best total for the player's hand.

        Returns:
            Best total for the player's hand.
        """
        return hand_value(cards=self.player)

    def dealer_total(self) -> int:
        """Returns the current best total for the dealer's hand.

        Returns:
            Best total for the dealer's hand.
        """
        return hand_value(cards=self.dealer)


def settle(hand: BlackjackHand) -> tuple[OutcomeLabel, int]:
    """Resolves a finished hand into an outcome label and the player's net delta.

    Delta is computed against the bet, not against the bankroll:
    - natural Blackjack pays 1.5x (rounded to int) on top of the bet's preservation;
    - regular win pays 1x;
    - push returns 0;
    - loss returns ``-bet``.

    Args:
        hand: Finished Blackjack hand to settle.

    Returns:
        A tuple of `(outcome, delta)`, where `delta` is the player's net point
        change relative to the withdrawn bet.

    Raises:
        ValueError: The hand is not finished yet.
    """
    if not hand.finished:
        raise ValueError("Cannot settle an unfinished Blackjack hand")

    bet = hand.bet
    player_total = hand.player_total()
    dealer_total = hand.dealer_total()
    player_bj = is_blackjack(cards=hand.player)
    dealer_bj = is_blackjack(cards=hand.dealer)

    if player_bj and dealer_bj:
        outcome: OutcomeLabel = "push"
        delta = 0
    elif player_bj:
        outcome, delta = "blackjack", int(bet * 3 // 2)
    elif dealer_bj:
        outcome, delta = "lose", -bet
    elif is_bust(cards=hand.player):
        outcome, delta = "player_bust", -bet
    elif is_bust(cards=hand.dealer):
        outcome, delta = "dealer_bust", bet
    elif player_total > dealer_total:
        outcome, delta = "win", bet
    elif player_total < dealer_total:
        outcome, delta = "lose", -bet
    else:
        outcome, delta = "push", 0
    return outcome, delta


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


def dealer_visible_value(hand: BlackjackHand) -> int:
    """Returns the numeric value of the dealer's visible card.

    The second dealer card is visible while the first card is hidden. If only
    one card exists, that card is treated as visible.

    Args:
        hand: Blackjack hand containing the dealer cards.

    Returns:
        The visible card's Blackjack value, or 0 when the dealer has no cards.
    """
    if not hand.dealer:
        return 0
    up = hand.dealer[1] if len(hand.dealer) > 1 else hand.dealer[0]
    if up.rank == "A":
        return 11
    if up.rank in ("J", "Q", "K"):
        return 10
    return int(up.rank)
