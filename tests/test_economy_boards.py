"""Tests for economy ranking board images."""

from io import BytesIO
from time import monotonic
from collections.abc import Iterator

from PIL import Image
import pytest

from discordbot.typings.economy import LeaderboardEntry, LossLeaderboardEntry
from discordbot.cogs.economy.boards import (
    _BOARD_IMAGE_CACHE_TTL_SECONDS,
    _RankingBoardSpec,
    _board_image_cache,
    _ranking_amount_text,
    _render_ranking_board_image,
    build_loss_leaderboard_board_image,
    build_balance_leaderboard_board_image,
)


@pytest.fixture(autouse=True)
def _empty_board_cache() -> Iterator[None]:
    """Starts every test from an empty process-local board cache.

    Nothing clears it in production any more, so a leftover entry from a previous
    test would otherwise be indistinguishable from one this test's own eviction
    was supposed to remove.
    """
    _board_image_cache.clear()
    yield
    _board_image_cache.clear()


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
    first = build_balance_leaderboard_board_image(rows=rows)

    def fail_render(spec: _RankingBoardSpec) -> bytes:
        del spec
        raise AssertionError("render should be cached")

    monkeypatch.setattr("discordbot.cogs.economy.boards._render_ranking_board_image", fail_render)
    assert build_balance_leaderboard_board_image(rows=rows) == first


def test_an_expired_board_renders_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A board past the TTL is re-rendered rather than served from the cache."""
    rows = (LeaderboardEntry(user_id=1, name="alice", balance=100),)
    build_balance_leaderboard_board_image(rows=rows)
    _age_cached_boards_past_the_ttl()

    calls = 0

    def count_render(spec: _RankingBoardSpec) -> bytes:
        nonlocal calls
        calls += 1
        return _render_ranking_board_image(spec=spec)

    monkeypatch.setattr("discordbot.cogs.economy.boards._render_ranking_board_image", count_render)
    build_balance_leaderboard_board_image(rows=rows)
    assert calls == 1


def test_a_superseded_board_is_evicted_without_a_write_path() -> None:
    """Nothing outside this module reaps the cache, so expiry has to be the size bound.

    A balance change never poisons a cached board: the rows are part of the key, so
    it mints a new entry and abandons the old one. Growth is the only failure mode
    left, and it is what the ledger's invalidation call used to hold back.
    """
    build_balance_leaderboard_board_image(
        rows=(LeaderboardEntry(user_id=1, name="alice", balance=100),)
    )
    _age_cached_boards_past_the_ttl()
    build_balance_leaderboard_board_image(
        rows=(LeaderboardEntry(user_id=1, name="alice", balance=250),)
    )

    assert len(_board_image_cache) == 1


def _age_cached_boards_past_the_ttl() -> None:
    """Backdates every cached board so the next lookup treats it as expired."""
    aged = monotonic() - _BOARD_IMAGE_CACHE_TTL_SECONDS - 1
    for spec, (_, image) in list(_board_image_cache.items()):
        _board_image_cache[spec] = (aged, image)
