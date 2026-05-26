"""Tests for economy ranking board images."""

from io import BytesIO

from PIL import Image
import pytest

from discordbot.typings.economy import LeaderboardEntry, LossLeaderboardEntry
from discordbot.cogs._economy.boards import (
    _RankingBoardSpec,
    _ranking_amount_text,
    _render_ranking_board_image,
    invalidate_economy_board_cache,
    build_loss_leaderboard_board_image,
    build_balance_leaderboard_board_image,
)


def test_balance_leaderboard_board_handles_large_balances_and_long_names() -> None:
    """Balance leaderboard rendering stays image-backed for long table values."""
    image = build_balance_leaderboard_board_image(
        rows=(
            LeaderboardEntry(
                user_id=1, name="超級無敵長名字測試玩家股份有限公司", balance=123_456_789_000_000
            ),
        )
    )

    assert image.startswith(b"\x89PNG")
    with Image.open(BytesIO(image)) as opened:
        assert opened.size[0] == 960
        assert opened.size[1] > 170


def test_loss_leaderboard_board_handles_large_losses() -> None:
    """Loss leaderboard rendering stays image-backed for large daily loss values."""
    image = build_loss_leaderboard_board_image(
        rows=(LossLeaderboardEntry(user_id=1, name="alice", loss_amount=987_654_321_000),)
    )

    assert image.startswith(b"\x89PNG")
    with Image.open(BytesIO(image)) as opened:
        assert opened.size[0] == 960
        assert opened.size[1] > 170


def test_loss_leaderboard_amount_text_has_no_prefix() -> None:
    """Loss leaderboard rows show only the compact amount."""
    assert (
        _ranking_amount_text(
            spec=_RankingBoardSpec(
                title="今日輸錢榜",
                subtitle="",
                amount_header="累計輸",
                amount_label="",
                accent=(0, 0, 0),
                rows=(),
            ),
            amount=9_876_543_210_000,
        )
        == "9.88兆"
    )


def test_balance_leaderboard_amount_text_has_no_prefix() -> None:
    """Balance leaderboard rows show only the compact amount."""
    assert (
        _ranking_amount_text(
            spec=_RankingBoardSpec(
                title="虛擬歡樂豆 排行榜",
                subtitle="",
                amount_header="餘額",
                amount_label="",
                accent=(0, 0, 0),
                rows=(),
            ),
            amount=27_0000_0000_0000,
        )
        == "27兆"
    )


def test_balance_leaderboard_board_image_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated identical board renders reuse the process-local PNG bytes."""
    rows = (LeaderboardEntry(user_id=1, name="alice", balance=100),)
    invalidate_economy_board_cache()
    first = build_balance_leaderboard_board_image(rows=rows)

    def fail_render(spec: _RankingBoardSpec) -> bytes:
        del spec
        raise AssertionError("render should be cached")

    monkeypatch.setattr("discordbot.cogs._economy.boards._render_ranking_board_image", fail_render)
    assert build_balance_leaderboard_board_image(rows=rows) == first


def test_economy_board_cache_invalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit invalidation forces the next matching board to render again."""
    rows = (LeaderboardEntry(user_id=1, name="alice", balance=100),)
    invalidate_economy_board_cache()
    build_balance_leaderboard_board_image(rows=rows)
    invalidate_economy_board_cache()

    calls = 0

    def count_render(spec: _RankingBoardSpec) -> bytes:
        nonlocal calls
        calls += 1
        return _render_ranking_board_image(spec=spec)

    monkeypatch.setattr(
        "discordbot.cogs._economy.boards._render_ranking_board_image", count_render
    )
    build_balance_leaderboard_board_image(rows=rows)
    assert calls == 1
