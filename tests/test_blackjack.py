"""Tests for the Blackjack rules and for settling a finished hand against the ledger."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random

import pytest

from discordbot.typings.economy import MAX_SINGLE_BET, VIP_PURCHASE_COST
from discordbot.cogs.games.blackjack import (
    Card,
    BlackjackRound,
    BlackjackHandState,
    InsuranceClosedError,
    InsuranceBetTooSmallError,
    InsuranceBeyondBalanceError,
    is_bust,
    is_pair,
    can_split,
    can_double,
    hand_value,
    is_soft_17,
    render_hand,
    settle_hand,
    is_blackjack,
    can_surrender,
    is_soft_total,
    dealer_must_hit,
    is_five_card_win,
    is_five_card_twenty_one,
)
from discordbot.cogs.games.settlement import blackjack_player_early_finish_note
from discordbot.services.economy.database import buy_vip, get_casino_ledger

from tests.helpers.games import card, seat, settle_only_seat
from tests.helpers.economy import seed_balance
from tests.helpers.economy_invariants import (
    assert_wallet_consistent,
    assert_daily_casino_stats,
    assert_casino_ledger_consistent,
)


def test_hand_value_no_aces() -> None:
    """Plain numeric cards just sum their face values."""
    assert hand_value(cards=[card(rank="10"), card(rank="9", suit="♥")]) == 19


def test_hand_value_face_cards_count_as_ten() -> None:
    """Each of J/Q/K is worth 10 points."""
    assert hand_value(cards=[card(rank="K"), card(rank="Q", suit="♥")]) == 20


def test_hand_value_ace_high_when_safe() -> None:
    """An ace counts as 11 when it doesn't push the hand over 21."""
    assert hand_value(cards=[card(rank="A"), card(rank="10", suit="♥")]) == 21


def test_hand_value_ace_demoted_when_needed() -> None:
    """Aces drop to 1 to avoid a bust."""
    cards = [card(rank="A"), card(rank="10", suit="♥"), card(rank="3", suit="♣")]
    assert hand_value(cards=cards) == 14


def test_hand_value_double_ace_demotes_one() -> None:
    """A+A starts at 22 and demotes one ace to 1, giving 12."""
    assert hand_value(cards=[card(rank="A"), card(rank="A", suit="♥")]) == 12


def test_is_blackjack_only_for_two_card_21() -> None:
    """Three sevens is 21 but not a natural Blackjack."""
    assert is_blackjack(cards=[card(rank="A"), card(rank="K", suit="♥")]) is True
    triple_seven = [card(rank="7"), card(rank="7", suit="♥"), card(rank="7", suit="♣")]
    assert is_blackjack(cards=triple_seven) is False


def test_is_five_card_twenty_one_accepts_five_or_more_cards_at_21() -> None:
    """過五關 21 bonus applies to five or more cards totaling 21."""
    five_card_21 = [
        card(rank="2"),
        card(rank="3", suit="♥"),
        card(rank="4", suit="♣"),
        card(rank="5", suit="♦"),
        card(rank="7"),
    ]
    six_card_21 = [
        card(rank="A"),
        card(rank="2", suit="♥"),
        card(rank="3", suit="♣"),
        card(rank="4", suit="♦"),
        card(rank="5"),
        card(rank="6", suit="♥"),
    ]
    four_card_21 = [
        card(rank="2"),
        card(rank="4", suit="♥"),
        card(rank="5", suit="♣"),
        card(rank="10", suit="♦"),
    ]
    five_card_20 = [
        card(rank="2"),
        card(rank="3", suit="♥"),
        card(rank="4", suit="♣"),
        card(rank="5", suit="♦"),
        card(rank="6"),
    ]

    assert is_five_card_twenty_one(cards=five_card_21) is True
    assert is_five_card_twenty_one(cards=six_card_21) is True
    assert is_five_card_twenty_one(cards=four_card_21) is False
    assert is_five_card_twenty_one(cards=five_card_20) is False


def test_is_five_card_win_accepts_any_five_card_non_bust() -> None:
    """過五關 win applies to five or more cards that have not busted."""
    five_card_20 = [
        card(rank="2"),
        card(rank="3", suit="♥"),
        card(rank="4", suit="♣"),
        card(rank="5", suit="♦"),
        card(rank="6"),
    ]
    five_card_bust = [
        card(rank="7"),
        card(rank="8", suit="♥"),
        card(rank="9", suit="♣"),
        card(rank="2", suit="♦"),
        card(rank="K"),
    ]
    four_card_20 = [
        card(rank="2"),
        card(rank="3", suit="♥"),
        card(rank="5", suit="♣"),
        card(rank="10", suit="♦"),
    ]

    assert is_five_card_win(cards=five_card_20) is True
    assert is_five_card_win(cards=five_card_bust) is False
    assert is_five_card_win(cards=four_card_20) is False


def test_is_bust_above_21() -> None:
    """is_bust returns True only when the value exceeds 21."""
    bust = [card(rank="K"), card(rank="Q", suit="♥"), card(rank="2", suit="♣")]
    safe = [card(rank="K"), card(rank="Q", suit="♥")]
    assert is_bust(cards=bust) is True
    assert is_bust(cards=safe) is False


def _settled_hand(cards: list[Card], bet: int = 100) -> BlackjackHandState:
    """Builds a finished production hand state for settlement assertions."""
    return BlackjackHandState(cards=cards, bet=bet, base_bet=bet, finished=True)


def _settle_cards(player: list[Card], dealer: list[Card], bet: int = 100) -> tuple[str, int]:
    """Settles a finished hand state against dealer cards."""
    return settle_hand(hand=_settled_hand(cards=player, bet=bet), dealer=dealer)


def test_settle_player_blackjack_pays_three_to_two() -> None:
    """A natural Blackjack pays 1.5x the bet (rounded down)."""
    outcome, delta = _settle_cards(
        player=[card(rank="A"), card(rank="K", suit="♥")],
        dealer=[card(rank="9"), card(rank="7", suit="♥")],
    )
    assert outcome == "blackjack"
    assert delta == 150


def test_settle_double_blackjack_is_push() -> None:
    """Two Blackjacks at the table cancel out."""
    outcome, delta = _settle_cards(
        player=[card(rank="A"), card(rank="K", suit="♥")],
        dealer=[card(rank="A", suit="♣"), card(rank="Q", suit="♦")],
    )
    assert outcome == "push"
    assert delta == 0


def test_blackjack_early_finish_note_ignores_regular_twenty_one() -> None:
    """A non-natural 21 should not be described as an early Blackjack finish."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(user_id=1, display_name="Bob")]
    )
    player = round_state.players[0]
    player.hands[0].cards = [card(rank="7", suit="♣"), card(rank="7", suit="♦"), card(rank="7")]
    assert (
        blackjack_player_early_finish_note(
            player=player,
            dealer=[card(rank="9"), card(rank="7", suit="♥")],
            peeked_blackjack=False,
        )
        is None
    )


def test_blackjack_player_early_finish_note_names_peeked_up_card() -> None:
    """Peek notes tell players the dealer used the visible up-card plus hole card."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(user_id=1, display_name="Bob")]
    )
    player = round_state.players[0]
    player.hands[0].cards = [card(rank="9"), card(rank="8", suit="♥")]

    note = blackjack_player_early_finish_note(
        player=player,
        dealer=[card(rank="A", suit="♣"), card(rank="K", suit="♦")],
        peeked_blackjack=True,
    )

    assert note == "莊家明牌 K♦, peek 暗牌確認 Blackjack, 本局直接結算"


def test_settle_player_bust_loses_bet() -> None:
    """When the player busts the dealer wins regardless of dealer total."""
    outcome, delta = _settle_cards(
        player=[card(rank="10"), card(rank="9", suit="♥"), card(rank="5", suit="♣")],
        dealer=[card(rank="10", suit="♣"), card(rank="6", suit="♦")],
        bet=50,
    )
    assert outcome == "player_bust"
    assert delta == -50


def test_settle_dealer_bust_pays_even_money() -> None:
    """When the dealer busts the player wins the bet."""
    outcome, delta = _settle_cards(
        player=[card(rank="10"), card(rank="9", suit="♥")],
        dealer=[card(rank="10", suit="♣"), card(rank="6", suit="♦"), card(rank="K")],
        bet=50,
    )
    assert outcome == "dealer_bust"
    assert delta == 50


def test_settle_higher_total_wins() -> None:
    """The higher (non-bust) total wins one bet at even money."""
    outcome, delta = _settle_cards(
        player=[card(rank="10"), card(rank="9", suit="♥")],
        dealer=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
        bet=50,
    )
    assert outcome == "win"
    assert delta == 50


def test_settle_lower_total_loses() -> None:
    """A lower (non-bust) total loses the bet."""
    outcome, delta = _settle_cards(
        player=[card(rank="10"), card(rank="7", suit="♥")],
        dealer=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
        bet=50,
    )
    assert outcome == "lose"
    assert delta == -50


def test_settle_equal_total_is_push() -> None:
    """Equal totals push regardless of card composition."""
    outcome, delta = _settle_cards(
        player=[card(rank="10"), card(rank="8", suit="♥")],
        dealer=[card(rank="9", suit="♣"), card(rank="9", suit="♦")],
        bet=50,
    )
    assert outcome == "push"
    assert delta == 0


def test_settle_unfinished_hand_raises() -> None:
    """Trying to settle a still-live hand is a programmer error."""
    hand = BlackjackHandState(cards=[card(rank="10")], bet=50, base_bet=50)
    with pytest.raises(expected_exception=ValueError, match="unfinished"):
        settle_hand(hand=hand, dealer=[card(rank="9", suit="♣"), card(rank="8", suit="♦")])


@pytest.mark.parametrize(
    argnames=("ranks", "must_hit"),
    argvalues=[
        (("10", "6"), True),
        (("A", "6"), True),
        (("10", "7"), False),
        (("A", "6", "10"), False),
        (("10", "8"), False),
        (("10", "6", "K"), False),
    ],
    ids=["hard-16", "soft-17", "hard-17", "hard-17-from-a-demoted-ace", "hard-18", "bust"],
)
def test_dealer_must_hit_under_h17(ranks: tuple[str, ...], must_hit: bool) -> None:
    """The dealer hits below 17 and on a soft 17, and stands on a hard 17 or better."""
    assert dealer_must_hit(cards=[card(rank=rank) for rank in ranks]) is must_hit


def test_blackjack_round_advances_players_then_leaves_the_dealer_to_draw() -> None:
    """The round advances in join order, and settling the players draws no dealer card."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=12345),
        participants=[seat(user_id=1, display_name="Alice"), seat(user_id=2, display_name="Bob")],
    )
    round_state.players[0].hands[0].cards = [card(rank="10"), card(rank="8", suit="♥")]
    round_state.players[1].hands[0].cards = [card(rank="9", suit="♣"), card(rank="8", suit="♦")]
    round_state.dealer = [card(rank="5", suit="♣"), card(rank="6", suit="♦")]

    assert round_state.active_player() == round_state.players[0]
    round_state.stand(user_id=1)
    assert round_state.active_player() == round_state.players[1]
    round_state.stand(user_id=2)

    assert round_state.finished is True
    assert round_state.dealer_played is False
    assert round_state.needs_dealer_play() is True
    assert [str(card) for card in round_state.dealer] == ["5♣", "6♦"]


def test_blackjack_round_rejects_action_from_non_active_player() -> None:
    """Only the current player can mutate the shared round."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[seat(user_id=1, display_name="Alice"), seat(user_id=2, display_name="Bob")],
    )
    round_state.players[0].hands[0].cards = [card(rank="10"), card(rank="8", suit="♥")]
    round_state.players[1].hands[0].cards = [card(rank="9", suit="♣"), card(rank="8", suit="♦")]
    round_state.dealer = [card(rank="5", suit="♣"), card(rank="6", suit="♦")]

    with pytest.raises(expected_exception=ValueError, match="turn"):
        round_state.hit(user_id=2)

    assert len(round_state.players[0].hands[0].cards) == 2


def test_render_hand_hides_first_card() -> None:
    """When the hole card is hidden, only the up-card and a back glyph appear."""
    cards = [card(rank="A"), card(rank="K", suit="♥")]
    rendered = render_hand(cards=cards, hide_first=True)
    assert "🂠" in rendered
    assert "A" not in rendered
    assert "K" in rendered


# Helper predicates ---------------------------------------------------------


def test_is_pair_same_blackjack_value() -> None:
    """Pair detection treats 10/J/Q/K as splittable 10-value cards."""
    assert is_pair(cards=[card(rank="8"), card(rank="8", suit="♥")]) is True
    assert is_pair(cards=[card(rank="A"), card(rank="A", suit="♥")]) is True
    assert is_pair(cards=[card(rank="10"), card(rank="K", suit="♥")]) is True
    assert is_pair(cards=[card(rank="Q"), card(rank="J", suit="♥")]) is True
    assert is_pair(cards=[card(rank="A"), card(rank="10", suit="♥")]) is False
    assert is_pair(cards=[card(rank="8")]) is False


def test_is_soft_total_when_ace_is_high() -> None:
    """`is_soft_total` returns True only while at least one Ace is 11."""
    soft, total = is_soft_total(cards=[card(rank="A"), card(rank="6", suit="♥")])
    assert (soft, total) == (True, 17)


def test_is_soft_total_when_ace_demoted_is_no_longer_soft() -> None:
    """A demoted Ace counts as 1 and the hand is hard."""
    cards = [card(rank="A"), card(rank="10", suit="♥"), card(rank="5", suit="♣")]
    soft, total = is_soft_total(cards=cards)
    assert (soft, total) == (False, 16)


def test_is_soft_17_only_when_soft_and_seventeen() -> None:
    """Soft 17 must hold both conditions."""
    assert is_soft_17(cards=[card(rank="A"), card(rank="6", suit="♥")]) is True
    assert is_soft_17(cards=[card(rank="10"), card(rank="7", suit="♥")]) is False


def _make_hand(cards: list[Card], bet: int = 100) -> BlackjackHandState:
    """Helper for hand-state predicates."""
    return BlackjackHandState(cards=cards, bet=bet, base_bet=bet)


def test_can_double_only_on_two_cards() -> None:
    """Double is offered only on the initial deal before any action."""
    fresh = _make_hand(cards=[card(rank="5"), card(rank="6", suit="♥")])
    assert can_double(hand=fresh, balance_remaining=200) is True
    fresh.actions_taken = 1
    assert can_double(hand=fresh, balance_remaining=200) is False


def test_can_double_rejected_when_balance_low() -> None:
    """Double needs an extra wager equal to the original bet."""
    fresh = _make_hand(cards=[card(rank="5"), card(rank="6", suit="♥")])
    assert can_double(hand=fresh, balance_remaining=99) is False


def test_can_double_is_closed_after_split() -> None:
    """A hand that came out of a Split cannot Double (no Double after Split)."""
    split_hand = _make_hand(cards=[card(rank="5"), card(rank="6", suit="♥")])
    split_hand.is_split_hand = True
    assert can_double(hand=split_hand, balance_remaining=200) is False


def test_can_double_rejected_when_doubling_exceeds_single_bet_cap() -> None:
    """Doubling cannot push the hand stake past MAX_SINGLE_BET."""
    over_cap = _make_hand(
        cards=[card(rank="5"), card(rank="6", suit="♥")], bet=MAX_SINGLE_BET // 2 + 1
    )
    assert can_double(hand=over_cap, balance_remaining=MAX_SINGLE_BET) is False
    at_cap = _make_hand(cards=[card(rank="5"), card(rank="6", suit="♥")], bet=MAX_SINGLE_BET // 2)
    assert can_double(hand=at_cap, balance_remaining=MAX_SINGLE_BET) is True


def test_can_split_only_on_same_value_pairs() -> None:
    """Split is offered on same-value pairs with enough balance."""
    pair = _make_hand(cards=[card(rank="8"), card(rank="8", suit="♥")])
    assert can_split(hand=pair, balance_remaining=200) is True
    face_pair = _make_hand(cards=[card(rank="10"), card(rank="K", suit="♥")])
    assert can_split(hand=face_pair, balance_remaining=200) is True
    non_pair = _make_hand(cards=[card(rank="A"), card(rank="10", suit="♥")])
    assert can_split(hand=non_pair, balance_remaining=200) is False
    assert can_split(hand=pair, balance_remaining=50) is False


def test_can_surrender_only_before_any_action() -> None:
    """Surrender is offered only on the very first action of the original hand."""
    fresh = _make_hand(cards=[card(rank="10"), card(rank="6", suit="♥")])
    assert can_surrender(hand=fresh, peeked_blackjack=False) is True
    fresh.actions_taken = 1
    assert can_surrender(hand=fresh, peeked_blackjack=False) is False
    assert (
        can_surrender(
            hand=_make_hand(cards=[card(rank="10"), card(rank="6", suit="♥")]),
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
        participants=[seat(user_id=1, display_name="Alice"), seat(user_id=2, display_name="Bob")],
    )
    round_state.players[0].hands[0].cards = cards_a
    round_state.players[1].hands[0].cards = cards_b
    round_state.dealer = dealer
    return round_state


def test_allowed_actions_list_the_active_hand_in_button_order() -> None:
    """A fresh pair that can cover a second bet may do everything, hit first; a settled round nothing."""
    round_state = _two_player_round(
        cards_a=[card(rank="8"), card(rank="8", suit="♥")],
        cards_b=[card(rank="9", suit="♣"), card(rank="9", suit="♦")],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )

    assert round_state.allowed_actions() == ("hit", "stand", "double", "split", "surrender")

    round_state.stand_all_remaining()

    assert round_state.allowed_actions() == ()


@pytest.mark.parametrize(
    argnames=("ranks", "hand_flags", "balance_at_start", "peeked_blackjack", "expected"),
    argvalues=[
        (("10", "K"), {}, 1_000, False, ("hit", "stand", "double", "split", "surrender")),
        (("A", "10"), {}, 1_000, False, ("hit", "stand", "double", "surrender")),
        (("5", "6", "4"), {"actions_taken": 1}, 1_000, False, ("hit", "stand")),
        (("8", "3"), {"is_split_hand": True}, 1_000, False, ("hit", "stand")),
        (("A", "5"), {"is_split_hand": True, "is_split_aces": True}, 1_000, False, ()),
        (("8", "8"), {}, 150, False, ("hit", "stand", "surrender")),
        (("9", "9"), {}, 1_000, True, ("hit", "stand", "double", "split")),
    ],
    ids=[
        "ten-value-pair",
        "ace-ten",
        "after-a-hit",
        "split-hand",
        "split-aces",
        "short-of-a-second-bet",
        "after-a-peeked-blackjack",
    ],
)
def test_allowed_actions_follow_the_active_hand(
    ranks: tuple[str, ...],
    hand_flags: dict[str, object],
    balance_at_start: int,
    peeked_blackjack: bool,
    expected: tuple[str, ...],
) -> None:
    """Each action is offered only while its rule allows it on the hand waiting to act."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(balance_at_start=balance_at_start)]
    )
    round_state.players[0].hands[0] = BlackjackHandState(
        cards=[card(rank=rank) for rank in ranks], bet=100, base_bet=100
    ).model_copy(update=hand_flags)
    round_state.dealer = [card(rank="5", suit="♣"), card(rank="6", suit="♦")]
    round_state.peeked_blackjack = peeked_blackjack

    assert round_state.allowed_actions() == expected


def test_an_empty_shoe_falls_back_to_drawing_from_an_infinite_deck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A round whose shoe has run out still deals, from `draw_card` rather than raising."""
    fallback_card = card(rank="9", suit="♦")
    monkeypatch.setattr("discordbot.cogs.games.blackjack.draw_card", lambda rng: fallback_card)
    round_state = BlackjackRound.from_participants(rng=Random(x=0), participants=[seat()])
    round_state.players[0].hands[0].cards = [card(rank="2"), card(rank="3", suit="♥")]
    round_state.dealer = [card(rank="5", suit="♣"), card(rank="6", suit="♦")]
    round_state.shoe = []

    assert round_state.hit(user_id=1) is fallback_card
    assert round_state.players[0].hands[0].cards[-1] is fallback_card


@pytest.mark.parametrize(
    argnames=("fifth", "total"), argvalues=[("6", 20), ("7", 21)], ids=["under-21", "21"]
)
def test_hit_auto_stands_on_a_fifth_card_that_does_not_bust(fifth: str, total: int) -> None:
    """Five cards that do not bust stand on their own, at 21 or below it, and the turn moves on."""
    round_state = _two_player_round(
        cards_a=[
            card(rank="2"),
            card(rank="3", suit="♥"),
            card(rank="4", suit="♣"),
            card(rank="5", suit="♦"),
        ],
        cards_b=[card(rank="9", suit="♣"), card(rank="9", suit="♦")],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )
    round_state.shoe = [card(rank=fifth)]

    round_state.hit(user_id=1)

    alice = round_state.players[0].hands[0]
    assert alice.total() == total
    assert alice.finished is True
    assert round_state.active_player() == round_state.players[1]


def test_split_hand_can_auto_stand_on_fifth_card_twenty_one() -> None:
    """Non-Ace split hands are evaluated independently for five-card 21."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(user_id=1, display_name="Alice")]
    )
    player = round_state.players[0]
    player.hands = [
        BlackjackHandState(
            cards=[card(rank="8"), card(rank="10", suit="♥")],
            bet=100,
            base_bet=100,
            is_split_hand=True,
            finished=True,
        ),
        BlackjackHandState(
            cards=[
                card(rank="2"),
                card(rank="3", suit="♥"),
                card(rank="4", suit="♣"),
                card(rank="5", suit="♦"),
            ],
            bet=100,
            base_bet=100,
            is_split_hand=True,
        ),
    ]
    round_state.current_hand_index = 1
    round_state.dealer = [card(rank="5", suit="♣"), card(rank="6", suit="♦")]
    round_state.shoe = [card(rank="7")]

    round_state.hit(user_id=1)

    assert is_five_card_twenty_one(cards=player.hands[1].cards) is True
    assert player.hands[1].total() == 21
    assert player.hands[1].finished is True
    assert round_state.finished is True


def test_double_down_doubles_bet_and_finishes_hand() -> None:
    """Double Down doubles the wager, draws one card, and stops the hand."""
    round_state = _two_player_round(
        cards_a=[card(rank="5"), card(rank="6", suit="♥")],
        cards_b=[card(rank="9", suit="♣"), card(rank="9", suit="♦")],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
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
        cards_a=[card(rank="8"), card(rank="8", suit="♥")],
        cards_b=[card(rank="9", suit="♣"), card(rank="9", suit="♦")],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
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
        cards_a=[card(rank="10"), card(rank="K", suit="♥")],
        cards_b=[card(rank="9", suit="♣"), card(rank="9", suit="♦")],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
    )

    round_state.split(user_id=1)

    alice = round_state.players[0]
    assert len(alice.hands) == 2
    assert alice.hands[0].cards[0] == card(rank="10")
    assert alice.hands[1].cards[0] == card(rank="K", suit="♥")


def test_split_aces_locks_each_hand_after_one_draw() -> None:
    """Splitting Aces marks both halves finished after a single draw each."""
    round_state = _two_player_round(
        cards_a=[card(rank="A"), card(rank="A", suit="♥")],
        cards_b=[card(rank="9", suit="♣"), card(rank="9", suit="♦")],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
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
        cards=[card(rank="A"), card(rank="10", suit="♥")],
        bet=100,
        base_bet=100,
        is_split_hand=True,
        is_split_aces=True,
        finished=True,
    )
    outcome, delta = settle_hand(hand=hand, dealer=[card(rank="9"), card(rank="9", suit="♣")])
    assert outcome == "win"
    assert delta == 100


def test_split_twenty_one_loses_to_dealer_natural_blackjack() -> None:
    """A split-derived 21 is not natural and loses to dealer Blackjack."""
    hand = BlackjackHandState(
        cards=[card(rank="A"), card(rank="10", suit="♥")],
        bet=100,
        base_bet=100,
        is_split_hand=True,
        is_split_aces=True,
        finished=True,
    )
    outcome, delta = settle_hand(
        hand=hand, dealer=[card(rank="A", suit="♣"), card(rank="K", suit="♦")]
    )
    assert outcome == "lose"
    assert delta == -100


def test_split_twenty_one_pushes_dealer_non_natural_twenty_one() -> None:
    """A split-derived 21 pushes a dealer 21 made with more than two cards."""
    hand = BlackjackHandState(
        cards=[card(rank="A"), card(rank="10", suit="♥")],
        bet=100,
        base_bet=100,
        is_split_hand=True,
        is_split_aces=True,
        finished=True,
    )
    outcome, delta = settle_hand(
        hand=hand, dealer=[card(rank="7", suit="♣"), card(rank="7", suit="♦"), card(rank="7")]
    )
    assert outcome == "push"
    assert delta == 0


def test_split_twenty_one_against_a_dealer_bust_reads_as_dealer_bust() -> None:
    """A split-derived 21 beats a busted dealer the way any other hand does, label included."""
    hand = BlackjackHandState(
        cards=[card(rank="A"), card(rank="K")],
        bet=100,
        base_bet=100,
        is_split_hand=True,
        finished=True,
    )
    outcome, delta = settle_hand(
        hand=hand,
        dealer=[card(rank="K", suit="♣"), card(rank="6", suit="♦"), card(rank="K", suit="♥")],
    )
    assert outcome == "dealer_bust"
    assert delta == 100


def test_surrender_marks_hand_with_half_bet_refund() -> None:
    """Surrender stops the hand and books a half-bet loss at settlement."""
    round_state = _two_player_round(
        cards_a=[card(rank="10"), card(rank="6", suit="♥")],
        cards_b=[card(rank="9", suit="♣"), card(rank="9", suit="♦")],
        dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")],
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
        cards=[card(rank="10"), card(rank="6", suit="♥")],
        bet=bet,
        base_bet=bet,
        surrendered=True,
        finished=True,
    )

    outcome, delta = settle_hand(
        hand=hand, dealer=[card(rank="5", suit="♣"), card(rank="6", suit="♦")]
    )

    assert outcome == "surrender"
    assert delta == expected_delta


def test_take_insurance_requires_ace_phase() -> None:
    """Insurance can only be placed during the dedicated insurance phase."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(user_id=1, display_name="Alice")]
    )
    with pytest.raises(expected_exception=InsuranceClosedError):
        round_state.take_insurance(user_id=1)


def test_take_insurance_requires_uncommitted_balance() -> None:
    """All-in players cannot add an insurance side bet on top of their wager."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[seat(user_id=1, display_name="Alice", bet=100, balance_at_start=100)],
    )
    round_state.phase = "insurance"
    round_state.insurance_offered = True

    with pytest.raises(expected_exception=InsuranceBeyondBalanceError):
        round_state.take_insurance(user_id=1)

    player = round_state.players[0]
    assert player.insurance_bet == 0
    assert player.insurance_resolved is False


def test_take_insurance_rejects_zero_chip_half_bet() -> None:
    """A 1-point original bet cannot buy 0-cost insurance."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[seat(user_id=1, display_name="Alice", bet=1, balance_at_start=10)],
    )
    round_state.phase = "insurance"
    round_state.insurance_offered = True
    player = round_state.players[0]

    with pytest.raises(expected_exception=InsuranceBetTooSmallError):
        round_state.take_insurance(user_id=1)

    assert player.insurance_bet == 0
    assert player.insurance_resolved is False


def test_each_insurance_refusal_has_its_own_class() -> None:
    """The three refusals a seat can hit are told apart by class, not by their wording.

    The view picks each one's notice off the class. Read off the wording instead, a 1-point seat
    whose half bet rounds to zero would be sent to refresh a table no refresh can change.
    """
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0),
        participants=[
            seat(user_id=1, display_name="Alice", bet=1, balance_at_start=10),
            seat(user_id=2, display_name="Bob", bet=100, balance_at_start=120),
        ],
    )
    round_state.insurance_offered = True

    round_state.phase = "player_actions"
    with pytest.raises(expected_exception=InsuranceClosedError):
        round_state.take_insurance(user_id=2)

    round_state.phase = "insurance"
    with pytest.raises(expected_exception=InsuranceBetTooSmallError):
        round_state.take_insurance(user_id=1)

    # 120 at the table less the 100 already wagered leaves 20, under the 50 insurance costs.
    with pytest.raises(expected_exception=InsuranceBeyondBalanceError):
        round_state.take_insurance(user_id=2)

    round_state.players[1].insurance_resolved = True
    with pytest.raises(expected_exception=InsuranceClosedError):
        round_state.take_insurance(user_id=2)
    with pytest.raises(expected_exception=InsuranceClosedError):
        round_state.decline_insurance(user_id=2)


def test_an_insurance_bet_counts_against_a_double_or_split() -> None:
    """The half-bet insurance comes out of what a Double or a Split can still draw on.

    220 at the table covers a second 100 bet only while the 50 insurance is not also owed.
    """

    def decided(*, take: bool) -> BlackjackRound:
        round_state = BlackjackRound.from_participants(
            rng=Random(x=0),
            participants=[seat(bet=100, balance_at_start=220)],
            shoe=[
                card(rank="8"),
                card(rank="8", suit="♥"),  # player
                card(rank="5", suit="♣"),  # dealer hole
                card(rank="A", suit="♦"),  # dealer up: insurance, then no Blackjack on the peek
            ],
        )
        round_state.deal_initial()
        if take:
            round_state.take_insurance(user_id=1)
        else:
            round_state.decline_insurance(user_id=1)
        return round_state

    declined = decided(take=False)
    insured = decided(take=True)

    assert {"double", "split"} <= set(declined.allowed_actions())
    assert not {"double", "split"} & set(insured.allowed_actions())
    with pytest.raises(expected_exception=ValueError, match="double"):
        insured.double_down(user_id=1)
    with pytest.raises(expected_exception=ValueError, match="split"):
        insured.split(user_id=1)


def test_deal_initial_offers_insurance_when_dealer_shows_ace() -> None:
    """Dealer up-card A puts the round into the insurance phase."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(user_id=1, display_name="Alice")]
    )
    # Force a deterministic deal by pre-loading the shoe in FIFO order.
    round_state.shoe = [
        card(rank="10"),
        card(rank="10", suit="♥"),  # player
        card(rank="5", suit="♣"),  # dealer hole
        card(rank="A", suit="♦"),  # dealer up
    ]

    round_state.deal_initial()

    assert round_state.phase == "insurance"
    assert round_state.insurance_offered is True
    assert round_state.peeked_blackjack is False


def test_dealer_peek_blackjack_settles_round_immediately() -> None:
    """A 10-up dealer Blackjack short-circuits to the settled phase."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(user_id=1, display_name="Alice")]
    )
    round_state.shoe = [
        card(rank="9"),
        card(rank="8", suit="♥"),  # player
        card(rank="A", suit="♣"),  # dealer hole
        card(rank="K", suit="♦"),  # dealer up — peek triggers
    ]

    round_state.deal_initial()

    assert round_state.peeked_blackjack is True
    assert round_state.phase == "settled"
    assert round_state.finished is True


def test_insurance_phase_closes_after_all_decisions_and_peeks() -> None:
    """After each player decides, the round peeks and advances accordingly."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(user_id=1, display_name="Alice")]
    )
    round_state.shoe = [
        card(rank="9"),
        card(rank="8", suit="♥"),  # player
        card(rank="K", suit="♣"),  # dealer hole
        card(rank="A", suit="♦"),  # dealer up — BJ!
    ]

    round_state.deal_initial()
    assert round_state.phase == "insurance"
    round_state.take_insurance(user_id=1)

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
        card(rank="2"),
        card(rank="3", suit="♥"),
        card(rank="9", suit="♣"),
        card(rank="7", suit="♦"),
        card(rank="5"),
        card(rank="6", suit="♥"),
    ]
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(user_id=1, display_name="Alice")], shoe=injected
    )
    assert round_state.shoe == injected

    round_state.deal_initial()

    # Two player cards plus two dealer cards were dealt from the front.
    assert round_state.players[0].hands[0].cards == [card(rank="2"), card(rank="3", suit="♥")]
    assert round_state.shoe == [card(rank="5"), card(rank="6", suit="♥")]


# Settlement against the ledger --------------------------------------------


def _finished_round(bet: int, balance_at_start: int = 100) -> BlackjackRound:
    """Builds a one-seat round already marked settled; each test deals its own cards."""
    round_state = BlackjackRound.from_participants(
        rng=Random(x=0), participants=[seat(bet=bet, balance_at_start=balance_at_start)]
    )
    round_state.players[0].hands[0].finished = True
    round_state.phase = "settled"
    return round_state


async def test_settle_blackjack_player_updates_player_and_casino() -> None:
    """Shared Blackjack settlement applies net delta and mirrors casino P&L."""
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = _finished_round(bet=50)
    round_state.players[0].hands[0].cards = [card(rank="10"), card(rank="Q", suit="♥")]
    round_state.dealer = [card(rank="10", suit="♣"), card(rank="8", suit="♦")]

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.delta == 50
    assert settlement.payout == 50
    assert settlement.new_balance == 150
    assert settlement.casino_balance == -50
    ledger = await get_casino_ledger()
    assert ledger.balance == -50


async def test_settle_blackjack_player_surrender_returns_half_bet() -> None:
    """Surrender books half the original bet as a loss and mirrors it into the casino ledger."""
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = _finished_round(bet=50)
    hand = round_state.players[0].hands[0]
    hand.cards = [card(rank="10"), card(rank="6", suit="♥")]
    hand.surrendered = True
    round_state.dealer = [card(rank="10", suit="♣"), card(rank="8", suit="♦")]

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.outcome == "surrender"
    assert settlement.delta == -25
    assert settlement.new_balance == 75
    assert settlement.casino_balance == 25


async def test_settle_blackjack_player_double_doubles_loss_when_dealer_higher() -> None:
    """Doubled hands lose 2x the original bet on settlement."""
    await seed_balance(user_id=1, name="alice", amount=200)
    round_state = _finished_round(bet=50)
    hand = round_state.players[0].hands[0]
    hand.cards = [card(rank="5"), card(rank="6", suit="♥"), card(rank="2", suit="♣")]
    hand.bet = 100
    hand.doubled = True
    round_state.dealer = [card(rank="10", suit="♣"), card(rank="9", suit="♦")]

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.delta == -100
    assert settlement.new_balance == 100


def _split_hands(second: Card) -> list[BlackjackHandState]:
    """Builds a finished split of eights: an 8-K hand and an 8 with `second`."""
    return [
        BlackjackHandState(
            cards=[card(rank="8"), card(rank="K", suit="♥")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
            finished=True,
        ),
        BlackjackHandState(
            cards=[card(rank="8", suit="♣"), second],
            bet=50,
            base_bet=50,
            is_split_hand=True,
            finished=True,
        ),
    ]


async def test_settle_blackjack_player_split_both_wins_aggregates_delta() -> None:
    """Both split hands' wins add up into the seat's one settlement."""
    await seed_balance(user_id=1, name="alice", amount=200)
    round_state = _finished_round(bet=50)
    round_state.players[0].hands = _split_hands(second=card(rank="9", suit="♦"))
    round_state.dealer = [card(rank="10", suit="♣"), card(rank="6", suit="♦")]

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.delta == 100
    assert len(settlement.hands) == 2
    assert settlement.hands[0].outcome == "win"
    assert settlement.hands[1].outcome == "win"
    assert settlement.new_balance == 300


async def test_settle_blackjack_player_split_offset_skips_vip_bonus() -> None:
    """A split that nets to zero settles as one zero write: no VIP bonus, no win or loss booked.

    A write per hand would book the losing hand as the day's casino loss and the winning one
    as its win, though the seat neither won nor lost.
    """
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST + 200)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None
    round_state = _finished_round(bet=50)
    round_state.players[0].hands = _split_hands(second=card(rank="2", suit="♦"))
    round_state.dealer = [card(rank="10", suit="♣"), card(rank="7", suit="♦")]

    settlement = await settle_only_seat(round_state=round_state)

    # hand1 win 50, hand2 lose 50 → net 0; VIP perk is suppressed on non-positive.
    assert settlement.base_delta == 0
    assert settlement.delta == 0
    assert settlement.vip_bonus == 0
    await assert_daily_casino_stats(user_id=1, loss=0, win=0, net=0)


@pytest.mark.parametrize(
    argnames=("last_card", "dealer", "dealer_21", "is_vip", "expect_outcome"),
    argvalues=[
        ("6", [("7", "♣"), ("7", "♦"), ("7", "♥")], True, False, "five_card_win"),
        ("7", [("10", "♣"), ("9", "♦")], False, False, "five_card_twenty_one"),
        ("7", [("10", "♣"), ("9", "♦")], False, True, "five_card_twenty_one"),
        ("7", [("7", "♣"), ("7", "♦"), ("7", "♥")], True, False, "five_card_twenty_one"),
        ("7", [("7", "♣"), ("7", "♦"), ("7", "♥")], True, True, "five_card_twenty_one"),
    ],
    ids=[
        "non-21-wins-regardless-of-dealer-21",
        "21-wins-vs-dealer-19",
        "vip-21-wins-vs-dealer-19",
        "21-push-vs-dealer-21",
        "vip-21-push-vs-dealer-21",
    ],
)
async def test_settle_blackjack_player_five_card(
    last_card: str,
    dealer: list[tuple[str, str]],
    dealer_21: bool,
    is_vip: bool,
    expect_outcome: str,
) -> None:
    """Five-card settlement: a non-21 hand wins regardless of dealer, a 21 follows the comparison.

    Every expected delta is derived from the rules -- the bet, the five-card bonus, and the VIP
    perk -- rather than hardcoded. The house ledger and daily counters are checked through the
    invariant helpers, proving the system-funded five-card and VIP-from-bonus payouts never move
    /casino while the dealer-funded portion does.
    """
    bet = 10_000
    starting = 100_000
    seed = (VIP_PURCHASE_COST + starting) if is_vip else starting
    await seed_balance(user_id=1, name="alice", amount=seed)
    if is_vip:
        assert await buy_vip(user_id=1, name="alice") is not None

    round_state = _finished_round(bet=bet, balance_at_start=starting)
    round_state.players[0].hands[0].cards = [
        card(rank="2"),
        card(rank="3", suit="♥"),
        card(rank="4", suit="♣"),
        card(rank="5", suit="♦"),
        card(rank=last_card),
    ]
    round_state.dealer = [card(rank=rank, suit=suit) for rank, suit in dealer]

    settlement = await settle_only_seat(round_state=round_state)

    is_21 = last_card == "7"
    five_card_bonus = bet if is_21 else 0
    # A five-card non-21 hand always wins; a five-card 21 pushes only against a dealer 21.
    base_delta = 0 if (is_21 and dealer_21) else bet
    # VIP perk is max(0.2x dealer-paid win, 0.2x five-card bonus); only the dealer-win share is
    # charged to the house, the rest is system funded along with the five-card bonus itself.
    house_vip = (base_delta * 20 // 100) if is_vip else 0
    vip_bonus = max(base_delta * 20 // 100, five_card_bonus * 20 // 100) if is_vip else 0
    delta = base_delta + vip_bonus + five_card_bonus
    casino_balance = -(base_delta + house_vip)

    assert settlement.outcome == expect_outcome
    assert settlement.hands[0].five_card_twenty_one is is_21
    assert settlement.hands[0].five_card_bonus == five_card_bonus
    assert settlement.base_delta == base_delta
    assert settlement.five_card_bonus == five_card_bonus
    assert settlement.vip_bonus == vip_bonus
    assert settlement.delta == delta
    assert settlement.new_balance == starting + delta
    assert settlement.casino_balance == casino_balance
    await assert_casino_ledger_consistent(expected_balance=casino_balance)
    await assert_daily_casino_stats(user_id=1, loss=0, win=delta, net=delta)
    await assert_wallet_consistent(user_id=1, expected_balance=starting + delta)


async def test_settle_blackjack_player_insurance_won_with_dealer_blackjack() -> None:
    """Insurance pays 2:1 when peek confirms dealer Blackjack."""
    await seed_balance(user_id=1, name="alice", amount=300)
    round_state = _finished_round(bet=100)
    player = round_state.players[0]
    player.hands[0].cards = [card(rank="9"), card(rank="8", suit="♥")]
    player.insurance_bet = 50
    player.insurance_resolved = True
    round_state.dealer = [card(rank="K", suit="♣"), card(rank="A", suit="♦")]
    round_state.peeked_blackjack = True

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.insurance is not None
    assert settlement.insurance.won is True
    assert settlement.insurance.delta == 100
    assert settlement.base_delta == 0  # -100 main bet + +100 insurance
    assert settlement.delta == 0
    assert settlement.outcome == "push"


async def test_settle_blackjack_player_insurance_lost_when_no_dealer_blackjack() -> None:
    """Insurance loses when the peek shows no Blackjack."""
    await seed_balance(user_id=1, name="alice", amount=300)
    round_state = _finished_round(bet=100)
    player = round_state.players[0]
    player.hands[0].cards = [card(rank="K"), card(rank="Q", suit="♥")]
    player.insurance_bet = 50
    player.insurance_resolved = True
    round_state.dealer = [
        card(rank="9", suit="♣"),
        card(rank="A", suit="♦"),
        card(rank="9", suit="♥"),
    ]
    round_state.dealer_played = True

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.insurance is not None
    assert settlement.insurance.won is False
    assert settlement.insurance.delta == -50
    # main win 100 - insurance 50 = +50
    assert settlement.base_delta == 50
    assert settlement.outcome == "win"
