"""Pins the exact display text of the shared Traditional Chinese number helpers.

`utils/number_text.py` renders every user-visible figure in the project (an economy balance, a
jackpot, a Blackjack seat, a fishing payout, a stock position, a leaderboard PNG), so its output
is not one screen's formatting but the shared reading of every number the bot shows. This file is
its only guard, and the `compact=True` pass-through on `services/economy/presentation.py` has none
other either, so a silent change in either lands on every surface at once.

The cases assert exact string equality rather than a range or a sampled property, because the
contract IS the string: these helpers return rounded display text that nothing may settle
against, so there is no looser invariant left to assert. The boundaries are picked for the
arithmetic no other test observes. A scaled value keeps every digit before the point, so 9,999,999
reads `1,000萬` rather than being flattened to one unit's worth; a display that rounding pushed
all the way up to `10,000` of its unit is re-rendered one unit larger instead, so 99,999,999 reads
`1億` and never `10,000萬`. That rollover fires on a handful of values and, when it breaks, still
returns text that looks like a correct answer. Both signs are asserted on each helper, since every
one of them renders its magnitude from `abs()` and prepends the sign separately.

`bold_currency` is not exercised here; it delegates to `currency_text`, which is.
"""

from discordbot.utils.number_text import compact_amount, compact_number, share_quantity_text
from discordbot.services.economy.presentation import amount_code, currency_text


def test_compact_amount_preserves_small_amounts() -> None:
    """Amounts below 萬 keep their exact comma-grouped digits on either sign."""
    assert compact_amount(amount=9_999) == "9,999"
    assert compact_amount(amount=-9_999) == "-9,999"
    assert compact_amount(amount=9_999, signed=True) == "+9,999"


def test_compact_amount_uses_traditional_chinese_scale_units() -> None:
    """From 10,000 up an amount takes a 萬 / 億 / 兆 suffix, keeping every digit before the point."""
    assert compact_amount(amount=10_000) == "1萬"
    assert compact_amount(amount=1_000_000) == "100萬"
    assert compact_amount(amount=9_999_999) == "1,000萬"
    assert compact_amount(amount=123_456_789) == "1.23億"
    assert compact_amount(amount=9_876_543_210_000) == "9.88兆"
    assert compact_amount(amount=-27_0000_0000_0000) == "-27兆"


def test_compact_amount_rolls_up_rounded_unit_boundaries() -> None:
    """A display rounding pushed up to 10,000 of a unit re-renders one unit larger."""
    assert compact_amount(amount=99_999_999) == "1億"
    assert compact_amount(amount=999_999_999_999) == "1兆"
    assert compact_amount(amount=99_999_999, signed=True) == "+1億"
    assert compact_amount(amount=-99_999_999) == "-1億"


def test_currency_helpers_can_opt_into_compact_amounts() -> None:
    """`currency_text` and `amount_code` pass `signed` and `compact` on to the scale units."""
    assert currency_text(amount=123_456_789, compact=True) == "1.23億 虛擬歡樂豆"
    assert currency_text(amount=123_456_789, signed=True, compact=True) == "+1.23億 虛擬歡樂豆"
    assert amount_code(amount=-10_000, signed=True, compact=True) == "`-1萬`"


def test_compact_number_matches_amount_formatting() -> None:
    """`compact_number` renders a non-money value exactly as `compact_amount` does."""
    assert compact_number(number=123_456_789) == "1.23億"


def test_share_quantity_text_uses_lot_units_without_changing_small_shares() -> None:
    """Shares stay in 股 below one lot, then read as 張 with the lot count itself compacted."""
    assert share_quantity_text(shares=999) == "999股"
    assert share_quantity_text(shares=1_000) == "1張"
    assert share_quantity_text(shares=1_234) == "1張 234股"
    assert share_quantity_text(shares=-1_234) == "-1張 234股"
    assert share_quantity_text(shares=1_234, signed=True) == "+1張 234股"
    assert share_quantity_text(shares=10_000_000_000_000) == "100億張"
