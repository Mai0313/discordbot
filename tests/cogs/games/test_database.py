"""Tests for Blackjack round-history persistence."""

from pathlib import Path

from discordbot.cogs.games import database as games_database
from discordbot.typings.games import (
    SettleOutcome,
    GameParticipant,
    BlackjackPlayerResult,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
    BlackjackInsuranceSettlement,
)
from discordbot.cogs.games.database import record_blackjack_history, fetch_recent_blackjack_rounds
from discordbot.cogs.games.blackjack import hand_value

from tests.helpers.games import card, seat

_DEALER_CARDS = [card(rank="9", suit="♦"), card(rank="7", suit="♣")]
_DEALER_TOTAL = 16


def _result(  # noqa: PLR0913 -- settlement result needs every per-round field
    participant: GameParticipant,
    outcome: SettleOutcome,
    delta: int,
    hands: list[BlackjackHandSettlement],
    insurance: BlackjackInsuranceSettlement | None = None,
    is_vip: bool = False,
) -> BlackjackPlayerResult:
    """Wraps settlement fields into a `BlackjackPlayerResult`."""
    settlement = BlackjackPlayerSettlement(
        delta=delta,
        new_balance=participant.balance_at_start + delta,
        casino_balance=0,
        base_delta=delta,
        vip_bonus=0,
        is_vip=is_vip,
        outcome=outcome,
        hands=hands,
        insurance=insurance,
    )
    return BlackjackPlayerResult(participant=participant, settlement=settlement)


def test_every_test_gets_its_own_round_history(tmp_path: Path) -> None:
    """A test that asks for no isolation still cannot reach the deployed `games.db`.

    It requests nothing but `tmp_path`, so dropping the autouse from
    `games_history_isolated_db` fails here instead of letting the next settled round write the
    live file. The engine is read off the module because the fixture swaps it per test.
    """
    assert games_database._engine.url.database == str(tmp_path / "games_history.db")


async def test_record_and_fetch_roundtrip() -> None:
    """A settled round persists split hands, insurance, and the dealer hand per player."""
    human = seat(user_id=1, display_name="alice", bet=1_000, balance_at_start=10_000)
    split_hands = [
        BlackjackHandSettlement(
            cards=[card(rank="8", suit="♣"), card(rank="K", suit="♦")],
            bet=1_000,
            outcome="win",
            delta=1_000,
            is_split_hand=True,
        ),
        BlackjackHandSettlement(
            cards=[card(rank="8", suit="♦"), card(rank="9", suit="♥")],
            bet=1_000,
            outcome="lose",
            delta=-1_000,
            is_split_hand=True,
        ),
    ]
    insurance = BlackjackInsuranceSettlement(bet=500, won=False, delta=-500)
    human_result = _result(
        participant=human,
        outcome="win",
        delta=-500,
        hands=split_hands,
        insurance=insurance,
        is_vip=True,
    )
    bot = seat(user_id=999, display_name="po-cat", bet=2_000, balance_at_start=10_000)
    bot_result = _result(
        participant=bot,
        outcome="lose",
        delta=-2_000,
        hands=[
            BlackjackHandSettlement(
                cards=[card(rank="J"), card(rank="Q", suit="♥"), card(rank="5", suit="♣")],
                bet=2_000,
                outcome="player_bust",
                delta=-2_000,
            )
        ],
    )

    await record_blackjack_history(
        round_id="round-1",
        channel_id=10,
        guild_id=20,
        message_id=30,
        bot_user_id=999,
        results=[human_result, bot_result],
        dealer_cards=_DEALER_CARDS,
        dealer_total=_DEALER_TOTAL,
    )

    human_rows = await fetch_recent_blackjack_rounds(user_id=1, limit=50)
    assert len(human_rows) == 1
    record = human_rows[0]
    assert record.round_id == "round-1"
    assert record.user_name == "alice"
    assert record.is_bot is False
    assert record.is_vip is True
    assert record.outcome == "win"
    assert record.delta == -500
    assert record.bet == 1_000
    assert record.payload.dealer_cards == _DEALER_CARDS
    assert record.payload.dealer_total == 16
    assert len(record.payload.hands) == 2
    assert record.payload.hands[0].total == hand_value(cards=split_hands[0].cards)
    assert record.payload.hands[0].is_split_hand is True
    assert record.payload.insurance is not None
    assert record.payload.insurance.won is False
    assert record.payload.insurance.delta == -500

    bot_rows = await fetch_recent_blackjack_rounds(user_id=999, limit=50)
    assert len(bot_rows) == 1
    assert bot_rows[0].is_bot is True
    assert bot_rows[0].payload.hands[0].total == 25


async def test_recent_ordering_and_limit() -> None:
    """Fetch returns the newest rounds first and honors the limit."""
    for bet in (100, 200, 300):
        await record_blackjack_history(
            round_id=f"round-{bet}",
            channel_id=1,
            guild_id=2,
            message_id=bet,
            bot_user_id=None,
            results=[
                _result(
                    participant=seat(
                        user_id=7, display_name="alice", bet=bet, balance_at_start=10_000
                    ),
                    outcome="win",
                    delta=bet,
                    hands=[
                        BlackjackHandSettlement(
                            cards=[card(rank="10"), card(rank="9", suit="♥")],
                            bet=bet,
                            outcome="win",
                            delta=bet,
                        )
                    ],
                )
            ],
            dealer_cards=_DEALER_CARDS,
            dealer_total=_DEALER_TOTAL,
        )

    rows = await fetch_recent_blackjack_rounds(user_id=7, limit=2)
    assert [row.bet for row in rows] == [300, 200]


async def test_fetch_recent_empty() -> None:
    """A player with no recorded rounds returns no records."""
    rows = await fetch_recent_blackjack_rounds(user_id=4242, limit=10)
    assert rows == ()
