"""The ledger's own display vocabulary: the `虛擬歡樂豆` label and the three ways an amount is shown.

`CURRENCY_NAME` is the single spelling of the currency, and every amount a user reads goes out as
one of three shapes built from it: the plain phrase (`currency_text`), the same phrase in bold
(`bold_currency`), and the bare figure in a Markdown code span with no currency name at all
(`amount_code`). Keeping the three here rather than in each surface is what makes a balance in an
economy embed, a Blackjack seat, a fishing panel and a stock position read identically.

The numeric half is not this module's: `utils/number_text.py` owns the Traditional Chinese
scale units, and `compact=True` is what reaches for them. So the split is deliberate — scale text
below, the currency label here — and both public helpers take exactly the same `signed` / `compact`
pair so a call site never has to know which branch it landed on. Everything produced is display
text: `compact=True` rounds, so nothing here may be fed back into settlement arithmetic, which
stays on the raw `StoredInteger`-backed ints.

It sits in `services/economy/` beside the ledger rather than inside `cogs/economy/` because the
callers span cogs and no cog may import from another: `cogs/economy/` (its commands, embeds and
ranking boards), `cogs/games/` (Blackjack and Dragon Gate views, the bot player, fishing),
`cogs/stock/presentation.py`, and the offline `scripts/modify_balance.py`. It is Discord-free —
what comes back is a string, and the Markdown is just characters.

One consequence of that reach: `CURRENCY_NAME` is interpolated into slash-command descriptions and
their localizations, so renaming the currency rewrites command metadata across the whole tree.
`tests/test_capabilities.py` reads those descriptions as unparsed source text for exactly that
reason.
"""

from discordbot.utils.number_text import compact_amount

CURRENCY_NAME = "虛擬歡樂豆"


def currency_text(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Formats an economy amount as `<number> 虛擬歡樂豆`.

    Args:
        amount (int): Economy amount to display.
        signed (bool): Whether a positive non-zero amount carries a leading `+`; a negative
            always carries `-` either way.
        compact (bool): Whether to render through the 萬 / 億 / 兆 scale units, which round, instead
            of the exact comma-grouped digits.

    Returns:
        Display text such as `9,999 虛擬歡樂豆` or `+1.23億 虛擬歡樂豆`.
    """
    number = (
        compact_amount(amount=amount, signed=signed)
        if compact
        else _amount_number(amount=amount, signed=signed)
    )
    return f"{number} {CURRENCY_NAME}"


def amount_code(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Formats an economy amount as a Markdown code span, without the currency name.

    The number is formatted exactly as `currency_text` would; only the wrapper and the missing
    label differ, for the embed fields and lines whose own heading already says what the figure is.

    Args:
        amount (int): Economy amount to display.
        signed (bool): Whether a positive non-zero amount carries a leading `+`; a negative
            always carries `-` either way.
        compact (bool): Whether to render through the 萬 / 億 / 兆 scale units, which round, instead
            of the exact comma-grouped digits.

    Returns:
        Display text such as `` `9,999` `` or `` `-1萬` ``.
    """
    number = (
        compact_amount(amount=amount, signed=signed)
        if compact
        else _amount_number(amount=amount, signed=signed)
    )
    return f"`{number}`"


def bold_currency(amount: int, signed: bool = False, compact: bool = False) -> str:
    """Formats an economy amount as `currency_text` does, in bold.

    The emphasis covers the whole phrase, currency name included, so the headline figure of a panel
    stays one visual unit.

    Args:
        amount (int): Economy amount to display.
        signed (bool): Whether a positive non-zero amount carries a leading `+`; a negative
            always carries `-` either way.
        compact (bool): Whether to render through the 萬 / 億 / 兆 scale units, which round, instead
            of the exact comma-grouped digits.

    Returns:
        Display text such as `**9,999 虛擬歡樂豆**`.
    """
    return f"**{currency_text(amount=amount, signed=signed, compact=compact)}**"


def _amount_number(amount: int, signed: bool) -> str:
    """Formats the exact comma-grouped digits of an amount, for the non-compact branch.

    Zero never takes a sign even under `signed`, so a delta that applied nothing reads `0` rather
    than `+0`. `compact_amount` suppresses it at zero the same way, which keeps the two branches
    interchangeable at a call site that only flips `compact`.

    Args:
        amount (int): Economy amount to display.
        signed (bool): Whether a positive non-zero amount carries a leading `+`.

    Returns:
        Text such as `9,999` or `+123,456`.
    """
    return f"{amount:+,}" if signed and amount != 0 else f"{amount:,}"
