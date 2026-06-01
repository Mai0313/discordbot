"""Tests for the Blackjack pure-rules module."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random
from unittest.mock import patch

import pytest

from discordbot.typings.games import (
    GameParticipant,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
)
from discordbot.typings.economy import MAX_SINGLE_BET
from discordbot.cogs._games.blackjack import (
    Card,
    BlackjackRound,
    BlackjackHandState,
    is_bust,
    is_pair,
    can_split,
    can_double,
    can_insure,
    hand_value,
    is_soft_17,
    render_hand,
    settle_hand,
    is_blackjack,
    can_surrender,
    is_soft_total,
    is_five_card_win,
    dealer_visible_value,
    is_five_card_twenty_one,
)
from discordbot.cogs._games.settlement import blackjack_player_early_finish_note
from discordbot.cogs._games.presentation import settlement_metadata
from discordbot.cogs._games.blackjack_views import (
    build_player_seat_embed,
    build_in_progress_embeds,
)


def test_hand_value_no_aces() -> None:
    """Plain numeric cards just sum their face values."""
    assert hand_value(cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")]) == 19


def test_hand_value_face_cards_count_as_ten() -> None:
    """Each of J/Q/K is worth 10 points."""
    assert hand_value(cards=[Card(rank="K", suit="♠"), Card(rank="Q", suit="♥")]) == 20


def test_hand_value_ace_high_when_safe() -> None:
    """An ace counts as 11 when it doesn't push the hand over 21."""
    assert hand_value(cards=[Card(rank="A", suit="♠"), Card(rank="10", suit="♥")]) == 21


def test_hand_value_ace_demoted_when_needed() -> None:
    """Aces drop to 1 to avoid a bust."""
    cards = [Card(rank="A", suit="♠"), Card(rank="10", suit="♥"), Card(rank="3", suit="♣")]
    assert hand_value(cards=cards) == 14


def test_hand_value_double_ace_demotes_one() -> None:
    """A+A starts at 22 and demotes one ace to 1, giving 12."""
    assert hand_value(cards=[Card(rank="A", suit="♠"), Card(rank="A", suit="♥")]) == 12


def test_is_blackjack_only_for_two_card_21() -> None:
    """Three sevens is 21 but not a natural Blackjack."""
    assert is_blackjack(cards=[Card(rank="A", suit="♠"), Card(rank="K", suit="♥")]) is True
    triple_seven = [Card(rank="7", suit="♠"), Card(rank="7", suit="♥"), Card(rank="7", suit="♣")]
    assert is_blackjack(cards=triple_seven) is False


def test_is_five_card_twenty_one_accepts_five_or_more_cards_at_21() -> None:
    """過五關 21 bonus applies to five or more cards totaling 21."""
    five_card_21 = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
        Card(rank="7", suit="♠"),
    ]
    six_card_21 = [
        Card(rank="A", suit="♠"),
        Card(rank="2", suit="♥"),
        Card(rank="3", suit="♣"),
        Card(rank="4", suit="♦"),
        Card(rank="5", suit="♠"),
        Card(rank="6", suit="♥"),
    ]
    four_card_21 = [
        Card(rank="2", suit="♠"),
        Card(rank="4", suit="♥"),
        Card(rank="5", suit="♣"),
        Card(rank="10", suit="♦"),
    ]
    five_card_20 = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
        Card(rank="6", suit="♠"),
    ]

    assert is_five_card_twenty_one(cards=five_card_21) is True
    assert is_five_card_twenty_one(cards=six_card_21) is True
    assert is_five_card_twenty_one(cards=four_card_21) is False
    assert is_five_card_twenty_one(cards=five_card_20) is False


def test_is_five_card_win_accepts_any_five_card_non_bust() -> None:
    """過五關 win applies to five or more cards that have not busted."""
    five_card_20 = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
        Card(rank="6", suit="♠"),
    ]
    five_card_bust = [
        Card(rank="7", suit="♠"),
        Card(rank="8", suit="♥"),
        Card(rank="9", suit="♣"),
        Card(rank="2", suit="♦"),
        Card(rank="K", suit="♠"),
    ]
    four_card_20 = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="5", suit="♣"),
        Card(rank="10", suit="♦"),
    ]

    assert is_five_card_win(cards=five_card_20) is True
    assert is_five_card_win(cards=five_card_bust) is False
    assert is_five_card_win(cards=four_card_20) is False


def test_is_bust_above_21() -> None:
    """is_bust returns True only when the value exceeds 21."""
    bust = [Card(rank="K", suit="♠"), Card(rank="Q", suit="♥"), Card(rank="2", suit="♣")]
    safe = [Card(rank="K", suit="♠"), Card(rank="Q", suit="♥")]
    assert is_bust(cards=bust) is True
    assert is_bust(cards=safe) is False


def _settled_hand(cards: list[Card], bet: int = 100) -> BlackjackHandState:
    """Builds a finished production hand state for settlement assertions."""
    return BlackjackHandState(cards=cards, bet=bet, base_bet=bet, finished=True)


def _settle_cards(player: list[Card], dealer: list[Card], bet: int = 100) -> tuple[str, int]:
    """Settles a finished hand state against dealer cards."""
    return settle_hand(hand=_settled_hand(cards=player, bet=bet), dealer=dealer)


def _participant(
    user_id: int, display_name: str, bet: int = 100, balance_at_start: int = 1_000
) -> GameParticipant:
    """Builds a prepared Blackjack participant for round tests."""
    return GameParticipant(
        user_id=user_id,
        account_name=display_name.lower(),
        display_name=display_name,
        bet=bet,
        balance_at_start=balance_at_start,
        is_allin=False,
    )


def test_settle_player_blackjack_pays_three_to_two() -> None:
    """A natural Blackjack pays 1.5x the bet (rounded down)."""
    outcome, delta = _settle_cards(
        player=[Card(rank="A", suit="♠"), Card(rank="K", suit="♥")],
        dealer=[Card(rank="9", suit="♠"), Card(rank="7", suit="♥")],
    )
    assert outcome == "blackjack"
    assert delta == 150


def test_settlement_metadata_shows_vip_bonus_numbers() -> None:
    """VIP winning settlements surface the base and boosted deltas."""
    metadata = settlement_metadata(
        delta=150, new_balance=1_150, is_allin=False, base_delta=100, vip_bonus=50
    )

    assert metadata == "-# 本局 `+150` · VIP加成 `+50` · 餘額 `1,150`"


def test_settled_bot_reason_appears_before_settlement_metadata() -> None:
    """Final bot reasoning appears above the round delta metadata."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=999, display_name="Dealer Bot")]
    )
    player = round_state.players[0]
    player.hands[0].cards = [Card(rank="6", suit="♠"), Card(rank="5", suit="♥")]
    round_state.dealer = [Card(rank="6", suit="♦"), Card(rank="10", suit="♣")]
    settlement = BlackjackPlayerSettlement(
        delta=80_000_000,
        payout=80_000_000,
        new_balance=815_000_000,
        casino_balance=0,
        base_delta=60_000_000,
        vip_bonus=20_000_000,
        is_vip=True,
        outcome="win",
        detail="win",
        hands=[
            BlackjackHandSettlement(
                cards=player.hands[0].cards,
                bet=30_000_000,
                outcome="win",
                delta=60_000_000,
                doubled=True,
            )
        ],
    )

    embed = build_player_seat_embed(
        player=player,
        round_state=round_state,
        active_hand_index=None,
        insurance_status=None,
        settlement=settlement,
        dealer_total=20,
        bot_reason="double: 手牌十一點對上莊家六點，期望值極高，加倍投注拼一張大牌。",
    )

    assert embed.description is not None
    assert embed.description.index("💭 double:") < embed.description.index("-# 本局")


def test_settle_double_blackjack_is_push() -> None:
    """Two Blackjacks at the table cancel out."""
    outcome, delta = _settle_cards(
        player=[Card(rank="A", suit="♠"), Card(rank="K", suit="♥")],
        dealer=[Card(rank="A", suit="♣"), Card(rank="Q", suit="♦")],
    )
    assert outcome == "push"
    assert delta == 0


def test_blackjack_early_finish_note_explains_dealer_natural() -> None:
    """A dealer natural Blackjack can end the round before the player acts."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=1, display_name="Bob")]
    )
    player = round_state.players[0]
    player.hands[0].cards = [Card(rank="9", suit="♠"), Card(rank="7", suit="♥")]
    assert (
        blackjack_player_early_finish_note(
            player=player,
            dealer=[Card(rank="A", suit="♣"), Card(rank="Q", suit="♦")],
            peeked_blackjack=False,
        )
        == "莊家起手 Blackjack, 依規則本局直接結算"
    )


def test_blackjack_early_finish_note_ignores_regular_twenty_one() -> None:
    """A non-natural 21 should not be described as an early Blackjack finish."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=1, display_name="Bob")]
    )
    player = round_state.players[0]
    player.hands[0].cards = [Card(rank="9", suit="♠"), Card(rank="7", suit="♥")]
    assert (
        blackjack_player_early_finish_note(
            player=player,
            dealer=[Card(rank="7", suit="♣"), Card(rank="7", suit="♦"), Card(rank="7", suit="♠")],
            peeked_blackjack=False,
        )
        is None
    )


def test_blackjack_player_early_finish_note_names_peeked_up_card() -> None:
    """Peek notes tell players the dealer used the visible up-card plus hole card."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=1, display_name="Bob")]
    )
    player = round_state.players[0]
    player.hands[0].cards = [Card(rank="9", suit="♠"), Card(rank="8", suit="♥")]

    note = blackjack_player_early_finish_note(
        player=player,
        dealer=[Card(rank="A", suit="♣"), Card(rank="K", suit="♦")],
        peeked_blackjack=True,
    )

    assert note == "莊家明牌 K♦, peek 暗牌確認 Blackjack, 本局直接結算"


def test_settle_player_bust_loses_bet() -> None:
    """When the player busts the dealer wins regardless of dealer total."""
    outcome, delta = _settle_cards(
        player=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥"), Card(rank="5", suit="♣")],
        dealer=[Card(rank="10", suit="♣"), Card(rank="6", suit="♦")],
        bet=50,
    )
    assert outcome == "player_bust"
    assert delta == -50


def test_settle_dealer_bust_pays_even_money() -> None:
    """When the dealer busts the player wins the bet."""
    outcome, delta = _settle_cards(
        player=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer=[Card(rank="10", suit="♣"), Card(rank="6", suit="♦"), Card(rank="K", suit="♠")],
        bet=50,
    )
    assert outcome == "dealer_bust"
    assert delta == 50


def test_settle_higher_total_wins() -> None:
    """The higher (non-bust) total wins one bet at even money."""
    outcome, delta = _settle_cards(
        player=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
        dealer=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
        bet=50,
    )
    assert outcome == "win"
    assert delta == 50


def test_settle_lower_total_loses() -> None:
    """A lower (non-bust) total loses the bet."""
    outcome, delta = _settle_cards(
        player=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")],
        dealer=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
        bet=50,
    )
    assert outcome == "lose"
    assert delta == -50


def test_settle_equal_total_is_push() -> None:
    """Equal totals push regardless of card composition."""
    outcome, delta = _settle_cards(
        player=[Card(rank="10", suit="♠"), Card(rank="8", suit="♥")],
        dealer=[Card(rank="9", suit="♣"), Card(rank="9", suit="♦")],
        bet=50,
    )
    assert outcome == "push"
    assert delta == 0


def test_settle_five_card_twenty_one_keeps_main_hand_push_against_dealer_21() -> None:
    """Five-card 21 uses its own outcome while the main hand can still push."""
    outcome, delta = _settle_cards(
        player=[
            Card(rank="2", suit="♠"),
            Card(rank="3", suit="♥"),
            Card(rank="4", suit="♣"),
            Card(rank="5", suit="♦"),
            Card(rank="7", suit="♠"),
        ],
        dealer=[Card(rank="7", suit="♣"), Card(rank="7", suit="♦"), Card(rank="7", suit="♥")],
        bet=50,
    )

    assert outcome == "five_card_twenty_one"
    assert delta == 0


def test_settle_five_card_non_21_wins_against_dealer_21() -> None:
    """Five-card non-bust hands win normally even when the dealer reaches 21."""
    outcome, delta = _settle_cards(
        player=[
            Card(rank="2", suit="♠"),
            Card(rank="3", suit="♥"),
            Card(rank="4", suit="♣"),
            Card(rank="5", suit="♦"),
            Card(rank="6", suit="♠"),
        ],
        dealer=[Card(rank="7", suit="♣"), Card(rank="7", suit="♦"), Card(rank="7", suit="♥")],
        bet=50,
    )

    assert outcome == "five_card_win"
    assert delta == 50


def test_settle_unfinished_hand_raises() -> None:
    """Trying to settle a still-live hand is a programmer error."""
    hand = BlackjackHandState(cards=[Card(rank="10", suit="♠")], bet=50, base_bet=50)
    with pytest.raises(expected_exception=ValueError, match="unfinished"):
        settle_hand(hand=hand, dealer=[Card(rank="9", suit="♣"), Card(rank="8", suit="♦")])


def test_dealer_keeps_drawing_below_17() -> None:
    """Dealer must hit until the hand value is ≥ 17 (or it busts)."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=12345), participants=[_participant(user_id=1, display_name="Alice")]
    )
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="9", suit="♥")]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    round_state.stand(user_id=1)
    final = round_state.dealer_total()
    assert final >= 17 or is_bust(cards=round_state.dealer)


def test_round_dealer_stops_on_hard_17_and_hits_soft_17() -> None:
    """Under H17 the dealer stops on hard 17 but keeps drawing on soft 17."""
    hard = BlackjackRound.from_participants(
        rng=Random(x=12345), participants=[_participant(user_id=1, display_name="Alice")]
    )
    hard.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="9", suit="♥")]
    hard.dealer = [Card(rank="10", suit="♣"), Card(rank="7", suit="♦")]
    hard.stand(user_id=1)

    soft = BlackjackRound.from_participants(
        rng=Random(x=12345), participants=[_participant(user_id=1, display_name="Alice")]
    )
    soft.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="9", suit="♥")]
    soft.dealer = [Card(rank="A", suit="♣"), Card(rank="6", suit="♦")]
    soft.stand(user_id=1)

    assert [str(card) for card in hard.dealer] == ["10♣", "7♦"]
    assert len(soft.dealer) >= 3, "H17 requires the dealer to draw on soft 17"
    assert soft.dealer[:2] == [Card(rank="A", suit="♣"), Card(rank="6", suit="♦")]


def test_blackjack_round_advances_players_and_dealer_after_all_stand() -> None:
    """The multiplayer round advances in join order and resolves dealer play once."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=12345),
        participants=[
            _participant(user_id=1, display_name="Alice"),
            _participant(user_id=2, display_name="Bob"),
        ],
    )
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="8", suit="♥")]
    round_state.players[1].hands[0].cards = [Card(rank="9", suit="♣"), Card(rank="8", suit="♦")]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]

    assert round_state.active_player() == round_state.players[0]
    round_state.stand(user_id=1)
    assert round_state.active_player() == round_state.players[1]
    round_state.stand(user_id=2)

    assert round_state.finished is True
    assert round_state.dealer_played is True
    assert round_state.dealer_total() >= 17 or is_bust(cards=round_state.dealer)


def test_blackjack_round_can_wait_for_async_dealer_play() -> None:
    """Async dealer mode leaves the dealer hand unchanged after players stand."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=12345),
        participants=[
            _participant(user_id=1, display_name="Alice"),
            _participant(user_id=2, display_name="Bob"),
        ],
        auto_play_dealer=False,
    )
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="8", suit="♥")]
    round_state.players[1].hands[0].cards = [Card(rank="9", suit="♣"), Card(rank="8", suit="♦")]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]

    round_state.stand(user_id=1)
    round_state.stand(user_id=2)

    assert round_state.finished is True
    assert round_state.dealer_played is False
    assert round_state.needs_dealer_play() is True
    assert [str(card) for card in round_state.dealer] == ["5♣", "6♦"]

    round_state.draw_dealer_card()
    round_state.mark_dealer_played()

    assert round_state.dealer_played is True
    assert len(round_state.dealer) == 3


def test_blackjack_round_rejects_action_from_non_active_player() -> None:
    """Only the current player can mutate the shared round."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[
            _participant(user_id=1, display_name="Alice"),
            _participant(user_id=2, display_name="Bob"),
        ],
    )
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="8", suit="♥")]
    round_state.players[1].hands[0].cards = [Card(rank="9", suit="♣"), Card(rank="8", suit="♦")]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]

    with pytest.raises(expected_exception=ValueError, match="turn"):
        round_state.hit(user_id=2)

    assert len(round_state.players[0].hands[0].cards) == 2


def test_render_hand_hides_first_card() -> None:
    """When the hole card is hidden, only the up-card and a back glyph appear."""
    cards = [Card(rank="A", suit="♠"), Card(rank="K", suit="♥")]
    rendered = render_hand(cards=cards, hide_first=True)
    assert "🂠" in rendered
    assert "A" not in rendered
    assert "K" in rendered


def test_dealer_visible_value_uses_up_card() -> None:
    """The visible value matches the card not hidden from the player."""
    assert dealer_visible_value(dealer=[Card(rank="A", suit="♠"), Card(rank="K", suit="♥")]) == 10
    assert dealer_visible_value(dealer=[Card(rank="K", suit="♠"), Card(rank="A", suit="♥")]) == 11
    assert dealer_visible_value(dealer=[Card(rank="7", suit="♠")]) == 7


def test_blackjack_in_progress_dealer_seat_hides_hole_card() -> None:
    """The dealer seat embed shows one hidden card marker plus the visible up-card."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=1, display_name="Bob")]
    )
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    round_state.dealer = [Card(rank="8", suit="♣"), Card(rank="K", suit="♦")]

    embeds = build_in_progress_embeds(
        round_state=round_state, system_name="賭場系統", system_avatar_url=""
    )
    dealer_embed = embeds[0]

    assert isinstance(dealer_embed.description, str)
    assert "🂠" in dealer_embed.description
    assert "K♦" in dealer_embed.description
    assert "8♣" not in dealer_embed.description


def test_blackjack_in_progress_dealer_seat_single_card_is_visible() -> None:
    """A one-card dealer fallback should not render as a hidden hole card."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=1, display_name="Bob")]
    )
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    round_state.dealer = [Card(rank="8", suit="♣")]

    embeds = build_in_progress_embeds(
        round_state=round_state, system_name="賭場系統", system_avatar_url=""
    )
    dealer_embed = embeds[0]

    assert isinstance(dealer_embed.description, str)
    assert "8♣" in dealer_embed.description
    assert "🂠" not in dealer_embed.description


# Helper predicates ---------------------------------------------------------


def test_is_pair_same_blackjack_value() -> None:
    """Pair detection treats 10/J/Q/K as splittable 10-value cards."""
    assert is_pair(cards=[Card(rank="8", suit="♠"), Card(rank="8", suit="♥")]) is True
    assert is_pair(cards=[Card(rank="A", suit="♠"), Card(rank="A", suit="♥")]) is True
    assert is_pair(cards=[Card(rank="10", suit="♠"), Card(rank="K", suit="♥")]) is True
    assert is_pair(cards=[Card(rank="Q", suit="♠"), Card(rank="J", suit="♥")]) is True
    assert is_pair(cards=[Card(rank="A", suit="♠"), Card(rank="10", suit="♥")]) is False
    assert is_pair(cards=[Card(rank="8", suit="♠")]) is False


def test_is_soft_total_when_ace_is_high() -> None:
    """`is_soft_total` returns True only while at least one Ace is 11."""
    soft, total = is_soft_total(cards=[Card(rank="A", suit="♠"), Card(rank="6", suit="♥")])
    assert (soft, total) == (True, 17)


def test_is_soft_total_when_ace_demoted_is_no_longer_soft() -> None:
    """A demoted Ace counts as 1 and the hand is hard."""
    cards = [Card(rank="A", suit="♠"), Card(rank="10", suit="♥"), Card(rank="5", suit="♣")]
    soft, total = is_soft_total(cards=cards)
    assert (soft, total) == (False, 16)


def test_is_soft_17_only_when_soft_and_seventeen() -> None:
    """Soft 17 must hold both conditions."""
    assert is_soft_17(cards=[Card(rank="A", suit="♠"), Card(rank="6", suit="♥")]) is True
    assert is_soft_17(cards=[Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]) is False


def _make_hand(cards: list[Card], bet: int = 100) -> BlackjackHandState:
    """Helper for hand-state predicates."""
    return BlackjackHandState(cards=cards, bet=bet, base_bet=bet)


def test_can_double_only_on_two_cards() -> None:
    """Double is offered only on the initial deal before any action."""
    fresh = _make_hand(cards=[Card(rank="5", suit="♠"), Card(rank="6", suit="♥")])
    assert can_double(hand=fresh, balance_remaining=200) is True
    fresh.actions_taken = 1
    assert can_double(hand=fresh, balance_remaining=200) is False


def test_can_double_rejected_when_balance_low() -> None:
    """Double needs an extra wager equal to the original bet."""
    fresh = _make_hand(cards=[Card(rank="5", suit="♠"), Card(rank="6", suit="♥")])
    assert can_double(hand=fresh, balance_remaining=99) is False


def test_can_double_after_split_disabled_by_default() -> None:
    """Double-after-Split is disabled unless the caller explicitly allows it."""
    split_hand = _make_hand(cards=[Card(rank="5", suit="♠"), Card(rank="6", suit="♥")])
    split_hand.is_split_hand = True
    assert can_double(hand=split_hand, balance_remaining=200) is False
    assert can_double(hand=split_hand, balance_remaining=200, allow_after_split=True) is True


def test_can_double_rejected_when_doubling_exceeds_single_bet_cap() -> None:
    """Doubling cannot push the hand stake past MAX_SINGLE_BET."""
    over_cap = _make_hand(
        cards=[Card(rank="5", suit="♠"), Card(rank="6", suit="♥")], bet=MAX_SINGLE_BET // 2 + 1
    )
    assert can_double(hand=over_cap, balance_remaining=MAX_SINGLE_BET) is False
    at_cap = _make_hand(
        cards=[Card(rank="5", suit="♠"), Card(rank="6", suit="♥")], bet=MAX_SINGLE_BET // 2
    )
    assert can_double(hand=at_cap, balance_remaining=MAX_SINGLE_BET) is True


def test_can_split_only_on_same_value_pairs() -> None:
    """Split is offered on same-value pairs with enough balance."""
    pair = _make_hand(cards=[Card(rank="8", suit="♠"), Card(rank="8", suit="♥")])
    assert can_split(hand=pair, balance_remaining=200) is True
    face_pair = _make_hand(cards=[Card(rank="10", suit="♠"), Card(rank="K", suit="♥")])
    assert can_split(hand=face_pair, balance_remaining=200) is True
    non_pair = _make_hand(cards=[Card(rank="A", suit="♠"), Card(rank="10", suit="♥")])
    assert can_split(hand=non_pair, balance_remaining=200) is False
    assert can_split(hand=pair, balance_remaining=50) is False


def test_can_surrender_only_before_any_action() -> None:
    """Surrender is offered only on the very first action of the original hand."""
    fresh = _make_hand(cards=[Card(rank="10", suit="♠"), Card(rank="6", suit="♥")])
    assert can_surrender(hand=fresh, peeked_blackjack=False) is True
    fresh.actions_taken = 1
    assert can_surrender(hand=fresh, peeked_blackjack=False) is False
    assert (
        can_surrender(
            hand=_make_hand(cards=[Card(rank="10", suit="♠"), Card(rank="6", suit="♥")]),
            peeked_blackjack=True,
        )
        is False
    )


# Round actions -------------------------------------------------------------


def _two_player_round(
    cards_a: list[Card], cards_b: list[Card], dealer: list[Card]
) -> BlackjackRound:
    """Builds a deterministic two-player round skipping `deal_initial`."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[
            _participant(user_id=1, display_name="Alice"),
            _participant(user_id=2, display_name="Bob"),
        ],
    )
    round_state.players[0].hands[0].cards = cards_a
    round_state.players[1].hands[0].cards = cards_b
    round_state.dealer = dealer
    return round_state


def test_single_player_round_hit_finishes_on_fifth_card_twenty_one() -> None:
    """A production round hand auto-finishes when the fifth card makes 21."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=1, display_name="Alice")]
    )
    round_state.players[0].hands[0].cards = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
    ]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    round_state.shoe = []

    with patch(
        "discordbot.cogs._games.blackjack.draw_card", return_value=Card(rank="7", suit="♠")
    ):
        round_state.hit(user_id=1)

    assert round_state.players[0].hands[0].finished is True


def test_hit_auto_stands_on_fifth_card_non_bust() -> None:
    """Five cards that do not bust auto-stand even when below 21."""
    round_state = _two_player_round(
        cards_a=[
            Card(rank="2", suit="♠"),
            Card(rank="3", suit="♥"),
            Card(rank="4", suit="♣"),
            Card(rank="5", suit="♦"),
        ],
        cards_b=[Card(rank="9", suit="♣"), Card(rank="9", suit="♦")],
        dealer=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.shoe = []

    with patch(
        "discordbot.cogs._games.blackjack.draw_card", return_value=Card(rank="6", suit="♠")
    ):
        round_state.hit(user_id=1)

    alice = round_state.players[0].hands[0]
    assert alice.total() == 20
    assert alice.finished is True
    assert round_state.active_player() == round_state.players[1]


def test_hit_auto_stands_on_fifth_card_twenty_one() -> None:
    """A multiplayer hand advances after the fifth card makes 21."""
    round_state = _two_player_round(
        cards_a=[
            Card(rank="2", suit="♠"),
            Card(rank="3", suit="♥"),
            Card(rank="4", suit="♣"),
            Card(rank="5", suit="♦"),
        ],
        cards_b=[Card(rank="9", suit="♣"), Card(rank="9", suit="♦")],
        dealer=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )
    round_state.shoe = []

    with patch(
        "discordbot.cogs._games.blackjack.draw_card", return_value=Card(rank="7", suit="♠")
    ):
        round_state.hit(user_id=1)

    alice = round_state.players[0].hands[0]
    assert alice.total() == 21
    assert alice.finished is True
    assert round_state.active_player() == round_state.players[1]


def test_split_hand_can_auto_stand_on_fifth_card_twenty_one() -> None:
    """Non-Ace split hands are evaluated independently for five-card 21."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[_participant(user_id=1, display_name="Alice")],
        auto_play_dealer=False,
    )
    player = round_state.players[0]
    player.hands = [
        BlackjackHandState(
            cards=[Card(rank="8", suit="♠"), Card(rank="10", suit="♥")],
            bet=100,
            base_bet=100,
            is_split_hand=True,
            finished=True,
        ),
        BlackjackHandState(
            cards=[
                Card(rank="2", suit="♠"),
                Card(rank="3", suit="♥"),
                Card(rank="4", suit="♣"),
                Card(rank="5", suit="♦"),
            ],
            bet=100,
            base_bet=100,
            is_split_hand=True,
        ),
    ]
    round_state.current_hand_index = 1
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    round_state.shoe = []

    with patch(
        "discordbot.cogs._games.blackjack.draw_card", return_value=Card(rank="7", suit="♠")
    ):
        round_state.hit(user_id=1)

    assert is_five_card_twenty_one(cards=player.hands[1].cards) is True
    assert player.hands[1].total() == 21
    assert player.hands[1].finished is True
    assert round_state.finished is True


def test_double_down_doubles_bet_and_finishes_hand() -> None:
    """Double Down doubles the wager, draws one card, and stops the hand."""
    round_state = _two_player_round(
        cards_a=[Card(rank="5", suit="♠"), Card(rank="6", suit="♥")],
        cards_b=[Card(rank="9", suit="♣"), Card(rank="9", suit="♦")],
        dealer=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )

    round_state.double_down(user_id=1)

    alice = round_state.players[0].hands[0]
    assert alice.doubled is True
    assert alice.finished is True
    assert alice.bet == 200
    assert len(alice.cards) == 3
    assert round_state.active_player() == round_state.players[1]


def test_split_creates_two_hands_with_fresh_draws() -> None:
    """Split turns one pair into two sibling sub-hands, each drawing once."""
    round_state = _two_player_round(
        cards_a=[Card(rank="8", suit="♠"), Card(rank="8", suit="♥")],
        cards_b=[Card(rank="9", suit="♣"), Card(rank="9", suit="♦")],
        dealer=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )

    round_state.split(user_id=1)

    alice = round_state.players[0]
    assert len(alice.hands) == 2
    assert alice.hands[0].is_split_hand is True
    assert alice.hands[1].is_split_hand is True
    assert alice.hands[0].is_split_aces is False
    assert len(alice.hands[0].cards) == 2
    assert len(alice.hands[1].cards) == 2
    assert alice.hands[0].bet == 100
    assert alice.hands[1].bet == 100


def test_split_accepts_ten_value_pairs() -> None:
    """Split accepts any two 10-value cards, not just identical ranks."""
    round_state = _two_player_round(
        cards_a=[Card(rank="10", suit="♠"), Card(rank="K", suit="♥")],
        cards_b=[Card(rank="9", suit="♣"), Card(rank="9", suit="♦")],
        dealer=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )

    round_state.split(user_id=1)

    alice = round_state.players[0]
    assert len(alice.hands) == 2
    assert alice.hands[0].cards[0] == Card(rank="10", suit="♠")
    assert alice.hands[1].cards[0] == Card(rank="K", suit="♥")


def test_split_aces_locks_each_hand_after_one_draw() -> None:
    """Splitting Aces marks both halves finished after a single draw each."""
    round_state = _two_player_round(
        cards_a=[Card(rank="A", suit="♠"), Card(rank="A", suit="♥")],
        cards_b=[Card(rank="9", suit="♣"), Card(rank="9", suit="♦")],
        dealer=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )

    round_state.split(user_id=1)

    alice = round_state.players[0]
    assert len(alice.hands) == 2
    assert alice.hands[0].is_split_aces is True
    assert alice.hands[1].is_split_aces is True
    assert alice.hands[0].finished is True
    assert alice.hands[1].finished is True
    assert round_state.active_player() == round_state.players[1]


def test_split_aces_twenty_one_settles_as_regular_win_not_blackjack() -> None:
    """Hitting 21 on a split hand counts as 1:1 win, not 3:2 Blackjack."""
    hand = BlackjackHandState(
        cards=[Card(rank="A", suit="♠"), Card(rank="10", suit="♥")],
        bet=100,
        base_bet=100,
        is_split_hand=True,
        is_split_aces=True,
        finished=True,
    )
    outcome, delta = settle_hand(
        hand=hand, dealer=[Card(rank="9", suit="♠"), Card(rank="9", suit="♣")]
    )
    assert outcome == "win"
    assert delta == 100


def test_split_twenty_one_loses_to_dealer_natural_blackjack() -> None:
    """A split-derived 21 is not natural and loses to dealer Blackjack."""
    hand = BlackjackHandState(
        cards=[Card(rank="A", suit="♠"), Card(rank="10", suit="♥")],
        bet=100,
        base_bet=100,
        is_split_hand=True,
        is_split_aces=True,
        finished=True,
    )
    outcome, delta = settle_hand(
        hand=hand, dealer=[Card(rank="A", suit="♣"), Card(rank="K", suit="♦")]
    )
    assert outcome == "lose"
    assert delta == -100


def test_split_twenty_one_pushes_dealer_non_natural_twenty_one() -> None:
    """A split-derived 21 pushes a dealer 21 made with more than two cards."""
    hand = BlackjackHandState(
        cards=[Card(rank="A", suit="♠"), Card(rank="10", suit="♥")],
        bet=100,
        base_bet=100,
        is_split_hand=True,
        is_split_aces=True,
        finished=True,
    )
    outcome, delta = settle_hand(
        hand=hand,
        dealer=[Card(rank="7", suit="♣"), Card(rank="7", suit="♦"), Card(rank="7", suit="♠")],
    )
    assert outcome == "push"
    assert delta == 0


def test_surrender_marks_hand_with_half_bet_refund() -> None:
    """Surrender stops the hand and books a half-bet loss at settlement."""
    round_state = _two_player_round(
        cards_a=[Card(rank="10", suit="♠"), Card(rank="6", suit="♥")],
        cards_b=[Card(rank="9", suit="♣"), Card(rank="9", suit="♦")],
        dealer=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")],
    )

    round_state.surrender(user_id=1)

    alice = round_state.players[0].hands[0]
    assert alice.surrendered is True
    assert alice.finished is True
    outcome, delta = settle_hand(hand=alice, dealer=round_state.dealer)
    assert outcome == "surrender"
    assert delta == -50


@pytest.mark.parametrize(argnames=("bet", "expected_delta"), argvalues=[(1, -1), (101, -51)])
def test_surrender_uses_ceil_half_loss_for_integer_chips(bet: int, expected_delta: int) -> None:
    """Surrender loses ceil(half bet), so a 1-point bet is not free."""
    hand = BlackjackHandState(
        cards=[Card(rank="10", suit="♠"), Card(rank="6", suit="♥")],
        bet=bet,
        base_bet=bet,
        surrendered=True,
        finished=True,
    )

    outcome, delta = settle_hand(
        hand=hand, dealer=[Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    )

    assert outcome == "surrender"
    assert delta == expected_delta


def test_take_insurance_requires_ace_phase() -> None:
    """Insurance can only be placed during the dedicated insurance phase."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=1, display_name="Alice")]
    )
    with pytest.raises(expected_exception=ValueError, match="Insurance"):
        round_state.take_insurance(user_id=1, amount=50)


def test_take_insurance_requires_uncommitted_balance() -> None:
    """All-in players cannot add an insurance side bet on top of their wager."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[
            _participant(user_id=1, display_name="Alice", bet=100, balance_at_start=100)
        ],
    )
    round_state.phase = "insurance"
    round_state.insurance_offered = True

    with pytest.raises(expected_exception=ValueError, match="balance"):
        round_state.take_insurance(user_id=1, amount=50)

    player = round_state.players[0]
    assert player.insurance_bet == 0
    assert player.insurance_resolved is False


def test_take_insurance_rejects_zero_chip_half_bet() -> None:
    """A 1-point original bet cannot buy 0-cost insurance."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[_participant(user_id=1, display_name="Alice", bet=1, balance_at_start=10)],
    )
    round_state.phase = "insurance"
    round_state.insurance_offered = True
    player = round_state.players[0]

    assert can_insure(player=player, balance_remaining=9) is False
    with pytest.raises(expected_exception=ValueError, match="positive"):
        round_state.take_insurance(user_id=1, amount=0)

    assert player.insurance_bet == 0
    assert player.insurance_resolved is False


def test_deal_initial_offers_insurance_when_dealer_shows_ace() -> None:
    """Dealer up-card A puts the round into the insurance phase."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=1, display_name="Alice")]
    )
    # Force a deterministic deal by pre-loading the shoe in FIFO order.
    round_state.shoe = [
        Card(rank="10", suit="♠"),
        Card(rank="10", suit="♥"),  # player
        Card(rank="5", suit="♣"),  # dealer hole
        Card(rank="A", suit="♦"),  # dealer up
    ]

    round_state.deal_initial()

    assert round_state.phase == "insurance"
    assert round_state.insurance_offered is True
    assert round_state.peeked_blackjack is False


def test_dealer_peek_blackjack_settles_round_immediately() -> None:
    """A 10-up dealer Blackjack short-circuits to the settled phase."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=1, display_name="Alice")]
    )
    round_state.shoe = [
        Card(rank="9", suit="♠"),
        Card(rank="8", suit="♥"),  # player
        Card(rank="A", suit="♣"),  # dealer hole
        Card(rank="K", suit="♦"),  # dealer up — peek triggers
    ]

    round_state.deal_initial()

    assert round_state.peeked_blackjack is True
    assert round_state.phase == "settled"
    assert round_state.finished is True


def test_insurance_phase_closes_after_all_decisions_and_peeks() -> None:
    """After each player decides, the round peeks and advances accordingly."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[_participant(user_id=1, display_name="Alice")]
    )
    round_state.shoe = [
        Card(rank="9", suit="♠"),
        Card(rank="8", suit="♥"),  # player
        Card(rank="K", suit="♣"),  # dealer hole
        Card(rank="A", suit="♦"),  # dealer up — BJ!
    ]

    round_state.deal_initial()
    assert round_state.phase == "insurance"
    round_state.take_insurance(user_id=1, amount=50)

    assert round_state.peeked_blackjack is True
    assert round_state.phase == "settled"
    assert round_state.players[0].insurance_bet == 50


def test_from_participants_deals_from_an_injected_shoe() -> None:
    """An injected persistent shoe is the round's deck, dealt front to back.

    The round's own `shoe` depletes as cards are dealt; the caller persists card
    counting by saving `round_state.shoe` after the round, not by relying on the
    passed list being mutated in place.
    """
    injected = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="9", suit="♣"),
        Card(rank="7", suit="♦"),
        Card(rank="5", suit="♠"),
        Card(rank="6", suit="♥"),
    ]
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[_participant(user_id=1, display_name="Alice")],
        shoe=injected,
    )
    assert round_state.shoe == injected

    round_state.deal_initial()

    # Two player cards plus two dealer cards were dealt from the front.
    assert round_state.players[0].hands[0].cards == [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
    ]
    assert round_state.shoe == [Card(rank="5", suit="♠"), Card(rank="6", suit="♥")]
