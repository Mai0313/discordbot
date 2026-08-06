"""Cent-to-cash rounding, for every place a cent-denominated price becomes spendable money.

Stock prices are held in cents (`price_cents` on the profile and on every trade leg) while wallets
hold whole `CURRENCY_NAME`, so each settlement and each displayed valuation has to collapse a
`price_cents * shares` product by a factor of 100. Which way it rounds is the entire decision, and
it belongs to the call site rather than to this module. A cash flow rounds against the trader, so
the sub-unit remainder is never minted into the ledger: what they part with ceils (`cash_ceil` —
buy cost, short collateral, the cost of covering a short) and what they are credited floors
(`cash_floor` — sell proceeds, short entry value). A valuation rounds conservatively in the same
direction but moves no money at all: long market value and market cap floor, the mark a short
position is carried at ceils, and all three only ever reach an embed or a board.

Both stay pure integer arithmetic. Money columns are `StoredInteger` decimal text precisely so a
balance can pass SQLite's 64-bit ceiling, and float division loses digits well before that ceiling,
so a `/ 100` here would start silently rounding large accounts.

It sits in `utils/` rather than beside the settlement so economy code can reuse the conversion
without redefining it. That is intent rather than fact — the importers today are
`services/stock/database.py`, which settles with it, and `cogs/stock/presentation.py`, which values
a market cap with it — and nothing in the layering forces it either way, since a cog importing a
service is routine here. It carries no domain state that would give it a home in one of them.
"""


def cash_ceil(cents: int) -> int:
    """Converts cents to whole cash, rounding up.

    For what a trader is charged, or the liability a position is marked at, so a fractional unit is
    never given away. The `+ 99` form is an exact ceiling across the whole integer line, negatives
    included, because Python's `//` floors.

    Args:
        cents (int): The cent-denominated amount.

    Returns:
        The smallest whole cash amount not below `cents / 100`.
    """
    return (cents + 99) // 100


def cash_floor(cents: int) -> int:
    """Converts cents to whole cash, rounding down.

    For what a trader is credited or a position is valued at, so the sub-unit remainder is dropped
    rather than minted.

    Args:
        cents (int): The cent-denominated amount.

    Returns:
        The largest whole cash amount not above `cents / 100`.
    """
    return cents // 100
