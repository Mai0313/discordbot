"""Shared helpers for rendering numbers as readable text."""

from decimal import Decimal

_COMPACT_UNITS = ((1_0000_0000_0000, "兆"), (1_0000_0000, "億"), (1_0000, "萬"))


def compact_number(number: int, signed: bool = False) -> str:
    """Formats a large integer with Traditional Chinese scale units."""
    abs_number = abs(number)
    sign = _number_sign(number=number, signed=signed)
    for unit_index, (threshold, suffix) in enumerate(_COMPACT_UNITS):
        if abs_number >= threshold:
            value = Decimal(abs_number) / Decimal(threshold)
            formatted = _compact_decimal(value=value)
            display_suffix = suffix
            if formatted == "10,000" and unit_index > 0:
                rollover_threshold, rollover_suffix = _COMPACT_UNITS[unit_index - 1]
                value = Decimal(abs_number) / Decimal(rollover_threshold)
                formatted = _compact_decimal(value=value)
                display_suffix = rollover_suffix
            return f"{sign}{formatted}{display_suffix}"
    return f"{number:+,}" if signed and number != 0 else f"{number:,}"


def compact_amount(amount: int, signed: bool = False) -> str:
    """Formats a large amount with Traditional Chinese scale units."""
    return compact_number(number=amount, signed=signed)


def _number_sign(number: int, signed: bool) -> str:
    """Returns the display sign for a formatted number."""
    if number < 0:
        return "-"
    if signed and number > 0:
        return "+"
    return ""


def _compact_decimal(value: Decimal) -> str:
    """Formats a compact display number with bounded decimals."""
    if value >= 100:
        formatted = f"{value:,.0f}"
    elif value >= 10:
        formatted = f"{value:,.1f}"
    else:
        formatted = f"{value:,.2f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted
