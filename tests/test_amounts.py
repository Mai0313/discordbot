"""Tests for shared amount presentation helpers."""

from discordbot.utils.amounts import compact_amount
from discordbot.cogs._economy.presentation import amount_code, currency_text


def test_compact_amount_preserves_small_amounts() -> None:
    """Small values keep comma-grouped exact formatting."""
    assert compact_amount(amount=9_999) == "9,999"
    assert compact_amount(amount=-9_999) == "-9,999"
    assert compact_amount(amount=9_999, signed=True) == "+9,999"


def test_compact_amount_uses_traditional_chinese_scale_units() -> None:
    """Large values use 萬, 億, and 兆 suffixes."""
    assert compact_amount(amount=10_000) == "1萬"
    assert compact_amount(amount=1_000_000) == "100萬"
    assert compact_amount(amount=9_999_999) == "1,000萬"
    assert compact_amount(amount=123_456_789) == "1.23億"
    assert compact_amount(amount=9_876_543_210_000) == "9.88兆"
    assert compact_amount(amount=-27_0000_0000_0000) == "-27兆"


def test_compact_amount_rolls_up_rounded_unit_boundaries() -> None:
    """Rounded 10,000-unit displays roll into the next larger suffix."""
    assert compact_amount(amount=99_999_999) == "1億"
    assert compact_amount(amount=999_999_999_999) == "1兆"
    assert compact_amount(amount=99_999_999, signed=True) == "+1億"
    assert compact_amount(amount=-99_999_999) == "-1億"


def test_currency_helpers_can_opt_into_compact_amounts() -> None:
    """Economy presentation wrappers expose compact formatting as an option."""
    assert currency_text(amount=123_456_789, compact=True) == "1.23億 虛擬歡樂豆"
    assert currency_text(amount=123_456_789, signed=True, compact=True) == "+1.23億 虛擬歡樂豆"
    assert amount_code(amount=-10_000, signed=True, compact=True) == "`-1萬`"
