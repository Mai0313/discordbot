"""Pure Blackjack rules and shoe helpers.

Kept side-effect free so the unit tests can drive deterministic deals via a
fixed `rng`. The cog wires this up with `random.SystemRandom()` for production.
"""

from random import Random

from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.games import Card, SettleOutcome, GameParticipant


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
    rank = rng.choice(seq=("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"))
    suit = rng.choice(seq=("♠", "♥", "♦", "♣"))
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
    while total > 21 and aces > 0:
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
    return len(cards) == 2 and hand_value(cards=cards) == 21


def is_bust(cards: list[Card]) -> bool:
    """Returns whether a hand is over 21.

    Args:
        cards: Cards to evaluate.

    Returns:
        True when the hand total is greater than 21.
    """
    return hand_value(cards=cards) > 21


class BlackjackHand(BaseModel):
    """Mutable state for one Blackjack round.

    Attributes:
        rng: Random source; injectable for tests.
        bet: Original bet amount (in points).
        player: Player's cards.
        dealer: Dealer's cards.
        finished: True once both sides have stopped drawing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rng: Random
    bet: int
    player: list[Card] = Field(default_factory=list)
    dealer: list[Card] = Field(default_factory=list)
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
        """Player stops; dealer hits until reaching 17, then resolves."""
        while hand_value(cards=self.dealer) < 17:
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


class BlackjackPlayerHand(BaseModel):
    """Mutable Blackjack hand state for one participant at a shared table.

    Attributes:
        participant: Discord player and wager metadata.
        cards: Cards currently held by the player.
        finished: True once this player no longer needs Hit / Stand actions.
    """

    participant: GameParticipant
    cards: list[Card] = Field(default_factory=list)
    finished: bool = False

    def total(self) -> int:
        """Returns the current best total for this player's hand."""
        return hand_value(cards=self.cards)

    def is_blackjack(self) -> bool:
        """Returns whether this player's first two cards are a natural Blackjack."""
        return is_blackjack(cards=self.cards)

    def is_bust(self) -> bool:
        """Returns whether this player has busted."""
        return is_bust(cards=self.cards)


class BlackjackRound(BaseModel):
    """Mutable state for a multiplayer Blackjack table.

    One dealer hand is shared by every player. Players act in lobby join order,
    while natural Blackjacks and busted hands are skipped automatically.

    Attributes:
        rng: Random source used for card draws.
        players: Per-player hand states.
        dealer: Dealer cards shared by the table.
        current_player_index: Index of the player whose turn is active.
        dealer_played: True once the dealer has drawn for all standing players.
        finished: True once no more player actions remain.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rng: Random
    players: list[BlackjackPlayerHand]
    dealer: list[Card] = Field(default_factory=list)
    current_player_index: int = 0
    dealer_played: bool = False
    finished: bool = False

    @classmethod
    def from_participants(
        cls, *, rng: Random, participants: list[GameParticipant]
    ) -> "BlackjackRound":
        """Builds a round from registered lobby participants."""
        players = [BlackjackPlayerHand(participant=participant) for participant in participants]
        return cls(rng=rng, players=players)

    def deal_initial(self) -> None:
        """Deals two cards to every player and two cards to the dealer."""
        for player in self.players:
            player.cards = [draw_card(rng=self.rng), draw_card(rng=self.rng)]
        self.dealer = [draw_card(rng=self.rng), draw_card(rng=self.rng)]
        dealer_blackjack = is_blackjack(cards=self.dealer)
        for player in self.players:
            if dealer_blackjack or player.is_blackjack():
                player.finished = True
        self._advance_or_finish()

    def active_player(self) -> BlackjackPlayerHand | None:
        """Returns the player whose turn is active, if any."""
        if self.finished:
            return None
        if self.current_player_index >= len(self.players):
            return None
        player = self.players[self.current_player_index]
        if player.finished:
            self._advance_or_finish()
            return self.active_player()
        return player

    def hit(self, *, user_id: int) -> Card:
        """Draws one card for the active player.

        Args:
            user_id: Discord user ID that must match the active player.

        Returns:
            The drawn card.

        Raises:
            ValueError: The user is not the active player.
        """
        player = self._require_active_player(user_id=user_id)
        card = draw_card(rng=self.rng)
        player.cards.append(card)
        if player.is_bust():
            player.finished = True
            self._advance_or_finish()
        return card

    def stand(self, *, user_id: int) -> None:
        """Marks the active player as standing and advances the table."""
        player = self._require_active_player(user_id=user_id)
        player.finished = True
        self._advance_or_finish()

    def stand_all_remaining(self) -> None:
        """Marks every unresolved player as standing, then finishes the table."""
        for player in self.players:
            player.finished = True
        self._finish_after_players_done()

    def dealer_total(self) -> int:
        """Returns the current best total for the dealer hand."""
        return hand_value(cards=self.dealer)

    def dealer_visible_value(self) -> int:
        """Returns the visible dealer card value for hint prompts."""
        hand = BlackjackHand(rng=self.rng, bet=0, dealer=list(self.dealer))
        return dealer_visible_value(hand=hand)

    def settlement_hand(self, *, player: BlackjackPlayerHand) -> BlackjackHand:
        """Builds the single-player hand shape used by settlement helpers."""
        return BlackjackHand(
            rng=self.rng,
            bet=player.participant.bet,
            player=list(player.cards),
            dealer=list(self.dealer),
            finished=True,
        )

    def _require_active_player(self, *, user_id: int) -> BlackjackPlayerHand:
        player = self.active_player()
        if player is None or player.participant.user_id != user_id:
            raise ValueError("Not this player's turn")
        return player

    def _advance_or_finish(self) -> None:
        while self.current_player_index < len(self.players):
            if not self.players[self.current_player_index].finished:
                return
            self.current_player_index += 1
        self._finish_after_players_done()

    def _finish_after_players_done(self) -> None:
        if self.finished:
            return
        if self._needs_dealer_play():
            self._play_dealer()
        self.finished = True

    def _needs_dealer_play(self) -> bool:
        if is_blackjack(cards=self.dealer):
            return False
        return any(not player.is_blackjack() and not player.is_bust() for player in self.players)

    def _play_dealer(self) -> None:
        while hand_value(cards=self.dealer) < 17:
            self.dealer.append(draw_card(rng=self.rng))
        self.dealer_played = True


def settle(hand: BlackjackHand) -> tuple[SettleOutcome, int]:
    """Resolves a finished hand into an outcome label and the player's net delta.

    Delta is computed as the direct bankroll change:
    - natural Blackjack pays 1.5x (rounded to int);
    - regular win pays 1x;
    - push returns 0;
    - loss returns ``-bet``.

    Args:
        hand: Finished Blackjack hand to settle.

    Returns:
        A tuple of `(outcome, delta)`, where `delta` is the player's net point
        change for the round.

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
        outcome: SettleOutcome = "push"
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
