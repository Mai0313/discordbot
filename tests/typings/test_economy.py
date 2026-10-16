"""Tests for the economy rules in `typings/economy.py`."""

import pytest

from discordbot.typings.economy import apply_vip_blackjack_bonus, monthly_rate_percent_to_bps


@pytest.mark.parametrize(
    argnames=("delta", "is_vip", "expected"),
    argvalues=[
        (100, False, 100),
        (100, True, 120),
        (101, True, 121),
        (1, True, 1),
        (0, True, 0),
        (-50, True, -50),
    ],
)
def test_apply_vip_blackjack_bonus(delta: int, is_vip: bool, expected: int) -> None:
    """VIP bonus applies only to positive winnings and floors fractional fifths."""
    assert apply_vip_blackjack_bonus(delta=delta, is_vip=is_vip) == expected


@pytest.mark.parametrize(
    argnames=("monthly_rate_percent", "expected_bps"),
    argvalues=[(3.0, 300), (0.0, 0), (100.0, 10_000), (150.0, 10_000), (-1.0, 0)],
)
def test_a_monthly_rate_percent_converts_to_bps_inside_the_allowed_range(
    monthly_rate_percent: float, expected_bps: int
) -> None:
    """A percent becomes basis points, clamped into the range a loan may carry."""
    assert monthly_rate_percent_to_bps(monthly_rate_percent=monthly_rate_percent) == expected_bps
