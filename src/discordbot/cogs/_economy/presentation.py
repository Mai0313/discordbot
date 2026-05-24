"""Shared presentation helpers for economy currency labels."""

from discordbot.utils.amounts import compact_amount

CURRENCY_NAME = "虛擬歡樂豆"


def currency_text(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Formats an economy amount with the shared currency name.

    Args:
        amount: Economy amount to display.
        signed: Whether positive non-zero amounts should include a leading `+`.
        compact: Whether large amounts should use Traditional Chinese scale units.

    Returns:
        A display string with the numeric amount and currency name.
    """
    number = (
        compact_amount(amount=amount, signed=signed)
        if compact
        else _amount_number(amount=amount, signed=signed)
    )
    return f"{number} {CURRENCY_NAME}"


def amount_code(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Formats a numeric amount as inline-code text.

    Args:
        amount: Economy amount to display.
        signed: Whether positive non-zero amounts should include a leading `+`.
        compact: Whether large amounts should use Traditional Chinese scale units.

    Returns:
        A Markdown inline-code numeric amount.
    """
    number = (
        compact_amount(amount=amount, signed=signed)
        if compact
        else _amount_number(amount=amount, signed=signed)
    )
    return f"`{number}`"


def bold_currency(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Formats a currency amount with bold Markdown emphasis."""
    return f"**{currency_text(amount=amount, signed=signed, compact=compact)}**"


def _amount_number(amount: int, signed: bool) -> str:
    """Formats the raw comma-grouped number for a currency amount."""
    return f"{amount:+,}" if signed and amount != 0 else f"{amount:,}"
