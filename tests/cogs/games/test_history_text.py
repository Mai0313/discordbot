"""Tests for the Blackjack history text renderer."""

from datetime import datetime

from discordbot.typings.games import (
    SettleOutcome,
    BlackjackHistoryHand,
    BlackjackHistoryRecord,
    BlackjackHistoryPayload,
)
from discordbot.utils.timezone import TAIWAN_TIMEZONE
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
