"""Tests for Blackjack round-history persistence and its history text renderer."""

from pathlib import Path
from datetime import datetime
from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.typings.games import (
    Card,
    SettleOutcome,
    GameParticipant,
    BlackjackHistoryHand,
    BlackjackPlayerResult,
    BlackjackHistoryRecord,
    BlackjackHandSettlement,
    BlackjackHistoryPayload,
    BlackjackPlayerSettlement,
    BlackjackInsuranceSettlement,
)
from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.cogs._games.database import (
    Base,
    record_blackjack_history,
    fetch_recent_blackjack_rounds,
)
from discordbot.cogs._games.blackjack import hand_value
from discordbot.cogs._games.history_text import _summarize, build_blackjack_history_embed

_DEALER_CARDS = [Card(rank="9", suit="♦"), Card(rank="7", suit="♣")]
_DEALER_TOTAL = 16


@pytest.fixture
async def games_isolated_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Per-test SQLite file with the full games-history schema."""
    db_path = tmp_path / "games.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr("discordbot.cogs._games.database._engine", engine)
    monkeypatch.setattr("discordbot.cogs._games.database._schema_ready_for", None)
    yield
    await engine.dispose()


def _participant(*, user_id: int, name: str, bet: int) -> GameParticipant:
    """Builds a minimal seated participant for settlement input."""
    return GameParticipant(
        user_id=user_id,
        account_name=name,
        display_name=name,
        avatar_url="",
        bet=bet,
        balance_at_start=10_000,
        is_allin=False,
    )


def _result(  # noqa: PLR0913 -- settlement result needs every per-round field
    *,
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
        payout=max(delta, 0),
        new_balance=participant.balance_at_start + delta,
        casino_balance=0,
        base_delta=delta,
        vip_bonus=0,
        is_vip=is_vip,
        outcome=outcome,
        detail="",
        hands=hands,
        insurance=insurance,
    )
    return BlackjackPlayerResult(participant=participant, settlement=settlement)


def _record_view(*, delta: int, outcome: SettleOutcome) -> BlackjackHistoryRecord:
    """Builds a read-model record without touching the database."""
    return BlackjackHistoryRecord(
        round_id="r",
        channel_id=1,
        guild_id=2,
        message_id=3,
        user_id=1,
        user_name="alice",
        is_bot=False,
        is_vip=False,
        bet=1_000,
        outcome=outcome,
        delta=delta,
        payload=BlackjackHistoryPayload(
            hands=[
                BlackjackHistoryHand(
                    cards=[Card(rank="A", suit="♠"), Card(rank="K", suit="♥")],
                    total=21,
                    bet=1_000,
                    outcome=outcome,
                    delta=delta,
                )
            ],
            dealer_cards=_DEALER_CARDS,
            dealer_total=_DEALER_TOTAL,
        ),
        created_at=datetime(2026, 5, 31, 22, 50, tzinfo=TAIWAN_TIMEZONE),
    )


def _wide_record_view() -> BlackjackHistoryRecord:
    """Builds a worst-case record that maxes out every cell width."""
    big = 999_999_999_999
    four_cards = [Card(rank="10", suit=suit) for suit in "♠♥♦♣"]
    hand = BlackjackHistoryHand(
        cards=four_cards, total=40, bet=big, outcome="lose", delta=-big, is_split_hand=True
    )
    return BlackjackHistoryRecord(
        round_id="r",
        channel_id=1,
        guild_id=2,
        message_id=3,
        user_id=1,
        user_name="alice",
        is_bot=False,
        is_vip=False,
        bet=big,
        outcome="lose",
        delta=-big,
        payload=BlackjackHistoryPayload(
            hands=[hand, hand], dealer_cards=four_cards, dealer_total=40
        ),
        created_at=datetime(2026, 5, 31, 22, 50, tzinfo=TAIWAN_TIMEZONE),
    )


async def test_record_and_fetch_roundtrip(games_isolated_db: None) -> None:
    """A settled round persists split hands, insurance, and the dealer hand per player."""
    human = _participant(user_id=1, name="alice", bet=1_000)
    split_hands = [
        BlackjackHandSettlement(
            cards=[Card(rank="8", suit="♣"), Card(rank="K", suit="♦")],
            bet=1_000,
            outcome="win",
            delta=1_000,
            is_split_hand=True,
        ),
        BlackjackHandSettlement(
            cards=[Card(rank="8", suit="♦"), Card(rank="9", suit="♥")],
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
    bot = _participant(user_id=999, name="po-cat", bet=2_000)
    bot_result = _result(
        participant=bot,
        outcome="lose",
        delta=-2_000,
        hands=[
            BlackjackHandSettlement(
                cards=[
                    Card(rank="J", suit="♠"),
                    Card(rank="Q", suit="♥"),
                    Card(rank="5", suit="♣"),
                ],
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
    assert [str(card) for card in record.payload.dealer_cards] == ["9♦", "7♣"]
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


async def test_recent_ordering_and_limit(games_isolated_db: None) -> None:
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
                    participant=_participant(user_id=7, name="alice", bet=bet),
                    outcome="win",
                    delta=bet,
                    hands=[
                        BlackjackHandSettlement(
                            cards=[Card(rank="10", suit="♠"), Card(rank="9", suit="♥")],
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


async def test_fetch_recent_empty(games_isolated_db: None) -> None:
    """A player with no recorded rounds returns no records."""
    rows = await fetch_recent_blackjack_rounds(user_id=4242, limit=10)
    assert rows == ()


def test_summary_counts_by_delta_sign() -> None:
    """Summary counts wins, losses, and pushes by net round delta."""
    records = (
        _record_view(delta=500, outcome="win"),
        _record_view(delta=-300, outcome="lose"),
        _record_view(delta=0, outcome="push"),
    )
    summary = _summarize(records=records)
    assert (summary.wins, summary.losses, summary.pushes) == (1, 1, 1)
    assert summary.net_delta == 200
    assert summary.rounds == 3


def test_history_embed_renders_code_block_table() -> None:
    """The history embed packs rounds into a fenced monospace table."""
    records = tuple(
        _record_view(delta=delta, outcome=outcome)
        for delta, outcome in ((1_500, "blackjack"), (-2_000, "player_bust"), (0, "push"))
    )
    embed = build_blackjack_history_embed(player_name="長名字測試玩家", records=records)
    description = embed.description or ""
    assert "```" in description
    assert "近 3 場 · 1 勝 1 敗 1 和 · 淨損益 -500" in description
    assert "A♠K♥(21)" in description
    assert "BJ" in description
    assert "BUST" in description
    assert "PUSH" in description
    assert "+1,500" in description
    assert "-2,000" in description


def test_history_embed_empty() -> None:
    """An empty history yields a plain notice embed with no code block."""
    embed = build_blackjack_history_embed(player_name="someone", records=())
    assert "```" not in (embed.description or "")
    assert "還沒有任何" in (embed.description or "")


def test_history_embed_fits_description_limit() -> None:
    """A full 50-round history stays within Discord's description limit."""
    records = tuple(_record_view(delta=1_000, outcome="win") for _ in range(50))
    embed = build_blackjack_history_embed(player_name="alice", records=records)
    assert len(embed.description or "") <= 4096
    assert "```" in (embed.description or "")


def test_history_embed_truncates_oversized_history() -> None:
    """A worst-case history trims older rounds and notes the omission within the limit."""
    records = tuple(_wide_record_view() for _ in range(50))
    embed = build_blackjack_history_embed(player_name="alice", records=records)
    description = embed.description or ""
    assert len(description) <= 4096
    assert "未顯示" in description
