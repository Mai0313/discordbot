"""Shared presentation helpers for economy currency labels."""

from discordbot.utils.number_text import compact_amount, grouped_amount

CURRENCY_NAME = "虛擬歡樂豆"


def _number_text(amount: int, signed: bool, compact: bool) -> str:
    """Formats a bare amount, in scale units when `compact`, else comma-grouped."""
    if compact:
        return compact_amount(amount=amount, signed=signed)
    return grouped_amount(amount=amount, signed=signed)


def currency_text(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Formats an economy amount with the shared currency name.

    Args:
        amount: Economy amount to display.
        signed: Whether positive non-zero amounts should include a leading `+`.
        compact: Whether large amounts should use Traditional Chinese scale units.

    Returns:
        A display string with the numeric amount and currency name.
    """
    return f"{_number_text(amount=amount, signed=signed, compact=compact)} {CURRENCY_NAME}"


def amount_code(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Formats a numeric amount as inline-code text.

    Args:
        amount: Economy amount to display.
        signed: Whether positive non-zero amounts should include a leading `+`.
        compact: Whether large amounts should use Traditional Chinese scale units.

    Returns:
        A Markdown inline-code numeric amount.
    """
    return f"`{_number_text(amount=amount, signed=signed, compact=compact)}`"


def bold_currency(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Formats a currency amount with bold Markdown emphasis."""
    return f"**{currency_text(amount=amount, signed=signed, compact=compact)}**"
