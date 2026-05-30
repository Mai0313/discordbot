"""Shared cent-to-integer-cash conversion helpers.

Stock execution and presentation convert cent-denominated prices into integer
`CURRENCY_NAME` cash. These rounding helpers live here so economy code can
reuse the same conversion without redefining it.
"""


def cash_ceil(cents: int) -> int:
    """Converts cents to integer cash with a ceiling."""
    return (cents + 99) // 100


def cash_floor(cents: int) -> int:
    """Converts cents to integer cash with a floor."""
    return cents // 100
