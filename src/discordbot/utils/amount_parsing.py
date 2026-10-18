"""Shared parsing for user-entered decimal amount text.

Money and quantity inputs are string slash options, since Discord's integer options
cap below the economy's balances. This normalizes their `"1,000"`-style text; each
caller applies its own range rules on top.
"""


def parse_decimal_amount(raw: str | None) -> int | None:
    """Returns the non-negative int parsed from comma-separated decimal text, or None if invalid.

    Args:
        raw (str | None): User-entered amount text.
    """
    normalized = (raw or "").replace(",", "").strip()
    if not normalized.isdecimal():
        return None
    try:
        return int(normalized)
    except ValueError:
        # A digit string past CPython's int-conversion limit passes isdecimal()
        # but int() rejects it; treat it as invalid input, not a crash.
        return None
