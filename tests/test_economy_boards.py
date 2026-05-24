"""Tests for economy ranking board images."""

from io import BytesIO

from PIL import Image

from discordbot.typings.economy import LeaderboardEntry, LossLeaderboardEntry
from discordbot.cogs._economy.boards import (
    _ranking_amount_text,
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
            spec={
                "title": "今日輸錢榜",
                "subtitle": "",
                "amount_header": "累計輸",
                "amount_label": "",
                "accent": (0, 0, 0),
                "rows": (),
            },
            amount=9_876_543_210_000,
        )
        == "9.88兆"
    )


def test_balance_leaderboard_amount_text_has_no_prefix() -> None:
    """Balance leaderboard rows show only the compact amount."""
    assert (
        _ranking_amount_text(
            spec={
                "title": "虛擬歡樂豆 排行榜",
                "subtitle": "",
                "amount_header": "餘額",
                "amount_label": "",
                "accent": (0, 0, 0),
                "rows": (),
            },
            amount=27_0000_0000_0000,
        )
        == "27兆"
    )
