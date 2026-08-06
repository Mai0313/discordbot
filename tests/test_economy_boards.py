"""Pins the two economy ranking boards and the render cache that now bounds itself.

`cogs/economy/boards.py` draws `/leaderboard` and `/loss_leaderboard` as PNG tables, and the two
rendering tests here assert shape rather than pixels: the PNG magic, the fixed `_BOARD_WIDTH`, and
a height past the header block. That is deliberate. `load_font` degrades silently to a Latin-only
face and then to Pillow's own default when the host carries no CJK font, so anything measured off
glyphs would read differently in CI than on a developer's machine, while the layout arithmetic
this file actually owns holds either way: the canvas width is a constant whatever the name, and
the height follows the row count. The values fed in are the ones that pushed these tables out of
an embed in the first place, an overlong CJK name and an amount that only reads in scale units.

The `_ranking_amount_text` pair pins the cell both shipped boards ask for: `amount_label` empty,
so a row shows the compact amount alone rather than repeating the column caption into every line.

The last three are the point of the file. No ledger write path clears this cache and none may,
since keeping that import direction one-way is what keeps Pillow out of the ledger, so expiry
carries the whole job and each half of it is pinned here: an identical spec is served from memory
instead of re-rendered, the TTL is what forces a re-render, and a superseded entry is evicted
rather than left to accumulate. Staleness is not among the failure modes, because the rows travel
inside the cache key and a balance change therefore strands an entry instead of poisoning one;
growth is, and it is what the ledger's dropped invalidation call used to hold back.

The cache is a module-level dict living for the whole process, so the autouse fixture empties it
on both sides of every test and `_age_cached_boards_past_the_ttl` backdates entries instead of
sleeping out a real TTL.
"""

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

    Nothing clears it in production any more, so a leftover entry from a previous test would
    otherwise be indistinguishable from one this test's own eviction was supposed to remove.
    """
    _board_image_cache.clear()
    yield
    _board_image_cache.clear()


def test_balance_leaderboard_board_handles_large_balances_and_long_names() -> None:
    """An overlong CJK name beside a 兆-scale balance still renders to a fixed-width PNG."""
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
    """A daily loss well past the digits an embed row could align still renders to a PNG."""
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
    """An expired entry a later render superseded is dropped rather than accumulated."""
    build_balance_leaderboard_board_image(
        rows=(LeaderboardEntry(user_id=1, name="alice", balance=100),)
    )
    _age_cached_boards_past_the_ttl()
    build_balance_leaderboard_board_image(
        rows=(LeaderboardEntry(user_id=1, name="alice", balance=250),)
    )

    assert len(_board_image_cache) == 1


def _age_cached_boards_past_the_ttl() -> None:
    """Backdates every cached board so the next lookup treats it as expired.

    Rewriting the stored timestamp rather than sleeping, so an expiry test costs nothing instead
    of a whole TTL.
    """
    aged = monotonic() - _BOARD_IMAGE_CACHE_TTL_SECONDS - 1
    for spec, (_, image) in list(_board_image_cache.items()):
        _board_image_cache[spec] = (aged, image)
