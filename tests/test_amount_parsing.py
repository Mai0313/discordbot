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
    """Comma-formatted decimal text parses; signs, fractions, and junk return None."""
    assert parse_decimal_amount(raw=raw) == expected


def test_parse_decimal_amount_rejects_oversized_digit_string() -> None:
    """A digit string past CPython's int-conversion limit is invalid, not a crash."""
    assert parse_decimal_amount(raw="9" * 5000) is None
