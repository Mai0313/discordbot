"""Traditional Chinese number text: 萬 / 億 / 兆 scale units and Taiwan 張 / 股 lot units.

Every balance, jackpot, market cap and share count a user sees is rendered through one of the
three public helpers here, so the same integer reads identically in an economy embed, a stock
panel, a Dragon Gate button label and a leaderboard PNG.

Integers only, and deliberately so: economy and stock money / share columns are `StoredInteger`
decimal text parsed to Python `int` precisely so a balance may pass SQLite's 64-bit ceiling, so
the scale division runs on `Decimal` rather than `float` (float division loses digits well before
that ceiling, and a large enough value cannot be converted to one at all). What comes back is
rounded display text and never an intermediate a caller may settle against; settlement arithmetic
stays on the raw ints.

The units are fixed Traditional Chinese, matching the rest of the user-facing copy: there is no
locale switch here, and no currency name either — the `虛擬歡樂豆` label lives in
`services/economy/presentation.py`, which wraps `compact_amount`.

Sits in `utils/` because both layers above it render the same numbers: the cogs (`economy`,
`stock`, `games`) for their embeds and boards, and `services/stock/database.py` for the rejection
text it hands back with a refused order. So nothing here may know about either.
"""

from decimal import Decimal

# Descending, which `compact_number` relies on twice: it takes the first unit that fits, and rolls
# a display rounded up to 10,000 into the entry before it.
_COMPACT_UNITS = ((1_0000_0000_0000, "兆"), (1_0000_0000, "億"), (1_0000, "萬"))
_SHARES_PER_LOT = 1_000


def compact_number(number: int, signed: bool = False) -> str:
    """Formats an integer with Traditional Chinese scale units, to at most two decimal places.

    Anything below 萬 keeps its exact comma-grouped digits; above it the value is divided by the
    largest unit that fits and rendered with at most two decimals, narrowing to one at 10 of that
    unit and to none at 100. Digits before the point are never dropped, so 12,340,000 reads
    `1,234萬` and 9,999,999 reads `1,000萬`. A display that rounding pushed up to `10,000` of a
    unit is re-rendered one unit larger, so 99,999,999 reads `1億` rather than `10,000萬`. 兆 is
    the largest unit this table carries, so a value that outgrows it keeps counting in 兆 rather
    than inventing one: 10**16 reads `10,000兆`.

    Args:
        number (int): The value to render.
        signed (bool): Whether a positive non-zero value carries a leading `+`; a negative always
            carries `-` either way.

    Returns:
        Display text such as `9,999`, `1.23億` or `-27兆`.
    """
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
    """Formats a money amount with Traditional Chinese scale units.

    Identical to `compact_number`, and exists so money call sites read in the ledger's own
    vocabulary rather than passing a balance as a generic `number=`.

    Args:
        amount (int): The economy amount to render.
        signed (bool): Whether a positive non-zero amount carries a leading `+`; a negative always
            carries `-` either way.

    Returns:
        Display text such as `1.23億`.
    """
    return compact_number(number=amount, signed=signed)


def share_quantity_text(shares: int, signed: bool = False) -> str:
    """Formats a share count in the Taiwan lot convention, where one 張 is 1,000 股.

    Under one lot the count stays in 股. At or above it the lot count itself goes through
    `compact_number`, so ten trillion shares read `100億張`, while the leftover shares are printed
    exactly and ungrouped (they are always under 1,000). The sign leads the whole phrase instead
    of each half.

    Args:
        shares (int): The share count to render.
        signed (bool): Whether a positive non-zero count carries a leading `+`; a negative always
            carries `-` either way.

    Returns:
        Display text such as `999股`, `1張` or `-1張 234股`.
    """
    abs_shares = abs(shares)
    sign = _number_sign(number=shares, signed=signed)
    if abs_shares < _SHARES_PER_LOT:
        return f"{sign}{abs_shares:,}股"

    lots, remaining_shares = divmod(abs_shares, _SHARES_PER_LOT)
    lot_text = compact_number(number=lots)
    if remaining_shares:
        return f"{sign}{lot_text}張 {remaining_shares}股"
    return f"{sign}{lot_text}張"


def _number_sign(number: int, signed: bool) -> str:
    """Returns the leading sign for a number whose magnitude is rendered from `abs(number)`.

    Args:
        number (int): The signed value being rendered.
        signed (bool): Whether a positive non-zero value carries a leading `+`.

    Returns:
        `-`, `+`, or an empty string.
    """
    if number < 0:
        return "-"
    if signed and number > 0:
        return "+"
    return ""


def _compact_decimal(value: Decimal) -> str:
    """Formats an already-scaled value to at most two decimal places, trailing zeros dropped.

    The place count narrows as the value grows: two decimals under 10, one under 100, none at or
    above it. Nothing rounds the whole part, so a scaled value in the thousands keeps all four of
    its digits (`1,234`). The strip is gated on a decimal point actually being present, or
    `10,000` would be shredded to `10,` — and that exact string is what `compact_number` matches
    to detect a unit rollover.

    Args:
        value (Decimal): The value already divided by its scale unit.

    Returns:
        Text such as `1.23`, `9.9` or `10,000`.
    """
    if value >= 100:
        formatted = f"{value:,.0f}"
    elif value >= 10:
        formatted = f"{value:,.1f}"
    else:
        formatted = f"{value:,.2f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted
