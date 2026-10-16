"""Tests for the Blackjack history text renderer."""

from datetime import datetime

from discordbot.typings.games import (
    SettleOutcome,
    BlackjackHistoryHand,
    BlackjackHistoryRecord,
    BlackjackHistoryPayload,
)
from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.utils.number_text import grouped_amount
from discordbot.cogs.games.history_text import _summarize, build_blackjack_history_embed

from tests.helpers.games import card

_DEALER_CARDS = [card(rank="9", suit="♦"), card(rank="7", suit="♣")]
_DEALER_TOTAL = 16


def _record_view(delta: int, outcome: SettleOutcome) -> BlackjackHistoryRecord:
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
                    cards=[card(rank="A"), card(rank="K", suit="♥")],
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
    four_cards = [card(rank="10", suit=suit) for suit in "♠♥♦♣"]
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


def test_summary_counts_by_delta_sign() -> None:
    """Summary counts wins, losses, and pushes by net round delta, not by the hand's outcome.

    A won hand beside a lost insurance bet can net a loss, so the second round disagrees.
    """
    records = (
        _record_view(delta=500, outcome="win"),
        _record_view(delta=-300, outcome="win"),
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
    summary, table, _ = (embed.description or "").split("```")
    rows = table.strip("\n").splitlines()
    assert grouped_amount(amount=-500, signed=True) in summary, "the net over every round"
    assert len(rows) == 3
    for row, record in zip(rows, records, strict=True):
        assert "".join(str(card) for card in record.payload.hands[0].cards) in row
        assert grouped_amount(amount=record.delta, signed=True) in row
    assert len({row.split()[-1] for row in rows}) == 3, "each outcome tagged differently"


def test_history_embed_empty() -> None:
    """An empty history yields a plain notice embed with no code block."""
    embed = build_blackjack_history_embed(player_name="someone", records=())
    assert embed.description
    assert "```" not in embed.description


def test_history_embed_shows_an_ordinary_full_history_whole() -> None:
    """Fifty ordinary rounds, the most the command fetches, all fit, so none is dropped."""
    records = tuple(_record_view(delta=1_000, outcome="win") for _ in range(50))
    embed = build_blackjack_history_embed(player_name="alice", records=records)
    _, table, after = (embed.description or "").split("```")
    assert len(table.strip("\n").splitlines()) == 50
    assert not after.strip(), "no omission note"


def test_history_embed_truncates_oversized_history() -> None:
    """A worst-case history trims older rounds and notes the omission within the limit."""
    records = tuple(_wide_record_view() for _ in range(50))
    embed = build_blackjack_history_embed(player_name="alice", records=records)
    description = embed.description or ""
    _, table, after = description.split("```")
    shown = len(table.strip("\n").splitlines())
    assert len(description) <= 4096
    assert shown < 50
    assert str(50 - shown) in after, "the note counts the rounds it left out"
