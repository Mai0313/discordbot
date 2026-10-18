"""Shared presentation helpers for economy currency labels."""

from discordbot.utils.number_text import compact_amount, grouped_amount

CURRENCY_NAME = "虛擬歡樂豆"


def _number_text(amount: int, signed: bool, compact: bool) -> str:
    """Formats a bare amount, in scale units when `compact`, else comma-grouped."""
    if compact:
        return compact_amount(amount=amount, signed=signed)
    return grouped_amount(amount=amount, signed=signed)


def currency_text(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Returns an economy amount formatted with the shared currency name.

    Args:
        amount (int): Economy amount to display.
        signed (bool): Whether positive non-zero amounts include a leading `+`.
        compact (bool): Whether large amounts use Traditional Chinese scale units.
    """
    return f"{_number_text(amount=amount, signed=signed, compact=compact)} {CURRENCY_NAME}"


def amount_code(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Returns a numeric amount formatted as Markdown inline code.

    Args:
        amount (int): Economy amount to display.
        signed (bool): Whether positive non-zero amounts include a leading `+`.
        compact (bool): Whether large amounts use Traditional Chinese scale units.
    """
    return f"`{_number_text(amount=amount, signed=signed, compact=compact)}`"


def bold_currency(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Returns a currency amount with bold Markdown emphasis.

    Args:
        amount (int): Economy amount to display.
        signed (bool): Whether positive non-zero amounts include a leading `+`.
        compact (bool): Whether large amounts use Traditional Chinese scale units.
    """
    return f"**{currency_text(amount=amount, signed=signed, compact=compact)}**"
