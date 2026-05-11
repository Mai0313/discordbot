"""Shared presentation helpers for economy currency labels."""

CURRENCY_NAME = "虛擬歡樂豆"


def currency_text(*, amount: int, signed: bool = False) -> str:
    """Formats an economy amount with the shared currency name.

    Args:
        amount: Economy amount to display.
        signed: Whether positive non-zero amounts should include a leading `+`.

    Returns:
        A display string with the numeric amount and currency name.
    """
    number = f"{amount:+,}" if signed and amount != 0 else f"{amount:,}"
    return f"{number} {CURRENCY_NAME}"
