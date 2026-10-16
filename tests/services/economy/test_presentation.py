"""Tests for the economy's currency presentation helpers."""

from discordbot.services.economy.presentation import amount_code, currency_text


def test_currency_helpers_can_opt_into_compact_amounts() -> None:
    """Economy presentation wrappers expose compact formatting as an option."""
    assert currency_text(amount=123_456_789, compact=True) == "1.23億 虛擬歡樂豆"
    assert currency_text(amount=123_456_789, signed=True, compact=True) == "+1.23億 虛擬歡樂豆"
    assert amount_code(amount=-10_000, signed=True, compact=True) == "`-1萬`"
