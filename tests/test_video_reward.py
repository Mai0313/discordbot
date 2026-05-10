"""Tests for the /download_video reward calculation."""

import pytest

from discordbot.cogs.video import _REWARD_CAP, _BASE_REWARD, _reward_for, _success_text


@pytest.mark.parametrize(
    argnames=("file_size_mb", "expected"),
    argvalues=[
        (0.0, _BASE_REWARD),
        (0.4, _BASE_REWARD),
        (5.0, _BASE_REWARD + 5),
        (24.7, _BASE_REWARD + 25),
        (200.0, _REWARD_CAP),
    ],
)
def test_reward_for_includes_base_plus_size_capped(file_size_mb: float, expected: int) -> None:
    """Reward is base + rounded MB, never over the cap."""
    assert _reward_for(file_size_mb=file_size_mb) == expected


def test_success_text_omits_reward_suffix_on_db_failure() -> None:
    """When awarded is None, the body must not promise points."""
    text = _success_text(file_size_mb=12.3, awarded=None)
    assert "獲得" not in text
    assert "12.3MB" in text


def test_success_text_includes_reward_suffix_on_success() -> None:
    """When awarded is set, the body shows the rewarded amount."""
    text = _success_text(file_size_mb=12.3, awarded=22)
    assert "22 點數" in text
    assert "12.3MB" in text
