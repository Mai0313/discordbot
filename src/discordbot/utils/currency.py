"""Shared cent-to-integer-cash conversion helpers.

Stock prices are cent-denominated while wallets hold whole `CURRENCY_NAME` cash, so
stock execution and stock presentation round every settlement leg and every displayed
total through one of these two. Picking one is the call site's job: stock settlement
ceils what a user pays and floors what a user receives.
"""


def cash_ceil(cents: int) -> int:
    """Converts cents to integer cash with a ceiling."""
    return (cents + 99) // 100


def cash_floor(cents: int) -> int:
    """Converts cents to integer cash with a floor."""
    return cents // 100
