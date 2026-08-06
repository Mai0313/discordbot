"""Pins the shared amount parser standing in front of every money and quantity slash option.

`utils.amount_parsing.parse_decimal_amount` is the one normalizer under `_parse_positive_amount`
/ `_parse_collect_amount` in the economy cog, `parse_wager_amount` in the games cog and
`parse_bait_quantity` in the fishing shop; each of those adds only its own range rule on top, so
what this parser accepts is what decides which text can reach a balance mutation at all. The
parametrized table pins both halves of that boundary: commas are stripped wherever they sit
(`"1,2,3"` is 123, since placement is deliberately not validated) and edge whitespace is
forgiven, while blank or absent text, a leading sign, a decimal point, an exponent and plain junk
each come back None. Widening the accept set is how unvalidated text would reach a ledger write;
narrowing it silently refuses amounts users do type. No 64-bit ceiling is pinned here on purpose,
since balances are `StoredInteger` decimal text precisely so an amount may exceed one.

The oversized digit string earns its own test because it is the only input that gets past the
`isdecimal()` gate and still fails: beyond CPython's int-conversion limit (4300 digits by
default) `int()` raises `ValueError`, and that has to degrade to None like any other unreadable
text rather than raise out of a slash command.
"""

import pytest

from discordbot.utils.amount_parsing import parse_decimal_amount


@pytest.mark.parametrize(
    argnames=("raw", "expected"),
    argvalues=[
        ("1,000", 1000),
        (" 42 ", 42),
        ("0", 0),
        ("1,2,3", 123),
        ("", None),
        ("   ", None),
        (None, None),
        ("-5", None),
        ("+5", None),
        ("1.5", None),
        ("abc", None),
        ("1e3", None),
    ],
)
def test_parse_decimal_amount(raw: str | None, expected: int | None) -> None:
    """Comma-formatted decimal text parses; blank, signed, fractional and junk text return None."""
    assert parse_decimal_amount(raw=raw) == expected


def test_parse_decimal_amount_rejects_oversized_digit_string() -> None:
    """A digit string past CPython's int-conversion limit is invalid, not a crash."""
    assert parse_decimal_amount(raw="9" * 5000) is None
