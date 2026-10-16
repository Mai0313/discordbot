"""Tests for the economy's currency presentation helpers."""

from discordbot.services.economy.presentation import CURRENCY_NAME, amount_code, currency_text


def test_currency_helpers_can_opt_into_compact_amounts() -> None:
    """Economy presentation wrappers expose compact formatting as an option."""
    assert currency_text(amount=123_456_789, compact=True) == f"1.23億 {CURRENCY_NAME}"
    assert (
        currency_text(amount=123_456_789, signed=True, compact=True) == f"+1.23億 {CURRENCY_NAME}"
    )
    assert amount_code(amount=-10_000, signed=True, compact=True) == "`-1萬`"
