"""Tests for settling a finished Blackjack seat against the ledger."""

import pytest

from discordbot.typings.economy import VIP_PURCHASE_COST
from discordbot.cogs.games.blackjack import Card, BlackjackHandState
from discordbot.services.economy.database import buy_vip, get_casino_ledger

from tests.helpers.games import card, seat, blackjack_hand, blackjack_round, settle_only_seat
from tests.helpers.economy import seed_balance
from tests.helpers.economy_invariants import (
    assert_wallet_consistent,
    assert_daily_casino_stats,
    assert_casino_ledger_consistent,
)


async def test_settle_blackjack_player_updates_player_and_casino() -> None:
    """Shared Blackjack settlement applies net delta and mirrors casino P&L."""
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="Q", suit="♥")]],
        dealer=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
        seats=[seat(bet=50)],
        finished=True,
    )

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.delta == 50
    assert settlement.new_balance == 150
    assert settlement.casino_balance == -50
    ledger = await get_casino_ledger()
    assert ledger.balance == -50


async def test_settle_blackjack_player_surrender_returns_half_bet() -> None:
    """Surrender books half the original bet as a loss and mirrors it into the casino ledger."""
    await seed_balance(user_id=1, name="alice", amount=100)
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="6", suit="♥")]],
        dealer=[card(rank="10", suit="♣"), card(rank="8", suit="♦")],
        seats=[seat(bet=50)],
        finished=True,
    )
    round_state.players[0].hands[0].surrendered = True

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.outcome == "surrender"
    assert settlement.delta == -25
    assert settlement.new_balance == 75
    assert settlement.casino_balance == 25


async def test_settle_blackjack_player_double_doubles_loss_when_dealer_higher() -> None:
    """Doubled hands lose 2x the original bet on settlement."""
    await seed_balance(user_id=1, name="alice", amount=200)
    round_state = blackjack_round(
        hands=[[card(rank="5"), card(rank="6", suit="♥"), card(rank="2", suit="♣")]],
        dealer=[card(rank="10", suit="♣"), card(rank="9", suit="♦")],
        seats=[seat(bet=50)],
        finished=True,
    )
    hand = round_state.players[0].hands[0]
    hand.bet = 100
    hand.doubled = True

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.delta == -100
    assert settlement.new_balance == 100


async def test_settle_blackjack_player_reports_only_what_a_short_wallet_paid() -> None:
    """A loss the wallet can no longer cover settles as what it collected, not as the bet.

    Bets are not taken when a round starts, so the wallet can be short by settlement (the
    same balance at two tables, or a transfer mid-round); the seat and its history row show
    this delta.
    """
    await seed_balance(user_id=1, name="alice", amount=30)
    round_state = blackjack_round(
        hands=[[card(rank="10"), card(rank="6", suit="♥")]],
        dealer=[card(rank="10", suit="♣"), card(rank="9", suit="♦")],
        seats=[seat(bet=100)],
        finished=True,
    )

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.outcome == "lose"
    assert settlement.delta == -30
    assert settlement.new_balance == 0
    assert settlement.casino_balance == 30
    await assert_wallet_consistent(user_id=1, expected_balance=0)
    await assert_casino_ledger_consistent(expected_balance=30)


def _split_hands(second: Card) -> list[BlackjackHandState]:
    """Builds a finished split of eights: an 8-K hand and an 8 with `second`."""
    return [
        blackjack_hand(
            cards=[card(rank="8"), card(rank="K", suit="♥")],
            bet=50,
            is_split_hand=True,
            finished=True,
        ),
        blackjack_hand(
            cards=[card(rank="8", suit="♣"), second], bet=50, is_split_hand=True, finished=True
        ),
    ]


async def test_settle_blackjack_player_split_both_wins_aggregates_delta() -> None:
    """Both split hands' wins add up into the seat's one settlement."""
    await seed_balance(user_id=1, name="alice", amount=200)
    round_state = blackjack_round(
        hands=[[]],
        dealer=[card(rank="10", suit="♣"), card(rank="6", suit="♦")],
        seats=[seat(bet=50)],
        finished=True,
    )
    round_state.players[0].hands = _split_hands(second=card(rank="9", suit="♦"))

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
    round_state = blackjack_round(
        hands=[[]],
        dealer=[card(rank="10", suit="♣"), card(rank="7", suit="♦")],
        seats=[seat(bet=50)],
        finished=True,
    )
    round_state.players[0].hands = _split_hands(second=card(rank="2", suit="♦"))

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

    round_state = blackjack_round(
        hands=[
            [
                card(rank="2"),
                card(rank="3", suit="♥"),
                card(rank="4", suit="♣"),
                card(rank="5", suit="♦"),
                card(rank=last_card),
            ]
        ],
        dealer=[card(rank=rank, suit=suit) for rank, suit in dealer],
        seats=[seat(bet=bet, balance_at_start=starting)],
        finished=True,
    )

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
    round_state = blackjack_round(
        hands=[[card(rank="9"), card(rank="8", suit="♥")]],
        dealer=[card(rank="K", suit="♣"), card(rank="A", suit="♦")],
        finished=True,
    )
    player = round_state.players[0]
    player.insurance_bet = 50
    player.insurance_resolved = True
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
    round_state = blackjack_round(
        hands=[[card(rank="K"), card(rank="Q", suit="♥")]],
        dealer=[card(rank="9", suit="♣"), card(rank="A", suit="♦"), card(rank="9", suit="♥")],
        finished=True,
    )
    player = round_state.players[0]
    player.insurance_bet = 50
    player.insurance_resolved = True

    settlement = await settle_only_seat(round_state=round_state)

    assert settlement.insurance is not None
    assert settlement.insurance.won is False
    assert settlement.insurance.delta == -50
    # main win 100 - insurance 50 = +50
    assert settlement.base_delta == 50
    assert settlement.outcome == "win"
