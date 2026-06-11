"""Shared parsing for user-entered decimal amount text.

Money and quantity inputs are string slash options (Discord's integer options cap
below the economy's balances), so several cogs parsed `"1,000"`-style text with the
same normalize / isdecimal / int sequence. This is the one normalizer; each caller
wraps it with its own range rules (positive-only, zero-means-all-in, purchase caps).
"""


def parse_decimal_amount(raw: str | None) -> int | None:
    """Parses decimal text with optional comma separators into an int.

    Args:
        raw: The user-entered amount text, possibly None or empty.

    Returns:
        The parsed non-negative integer, or None for empty or non-decimal text.
    """
    normalized = (raw or "").replace(",", "").strip()
    if not normalized.isdecimal():
        return None
    return int(normalized)
