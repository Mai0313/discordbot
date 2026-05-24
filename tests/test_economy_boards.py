"""Tests for economy ranking board images."""

from io import BytesIO

from PIL import Image

from discordbot.typings.economy import LeaderboardEntry, LossLeaderboardEntry
from discordbot.cogs._economy.boards import (
    build_loss_leaderboard_board_image,
    build_balance_leaderboard_board_image,
)


def test_balance_leaderboard_board_handles_large_balances_and_long_names() -> None:
    """Balance leaderboard rendering stays image-backed for long table values."""
    image = build_balance_leaderboard_board_image(
        rows=(
            LeaderboardEntry(
                user_id=1,
                name="超級無敵長名字測試玩家股份有限公司",
                balance=123_456_789_000_000,
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
