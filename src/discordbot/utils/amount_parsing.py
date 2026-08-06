"""Shared parsing for user-entered decimal amount text.

Money and quantity inputs are string slash options (Discord's integer options cap below the
economy's balances), so several cogs parsed `"1,000"`-style text with the same normalize /
isdecimal / int sequence. This is the one normalizer; each caller keeps its own range rules on
top — `_parse_positive_amount` and `_parse_collect_amount` in the economy cog,
`parse_wager_amount` where zero means all-in, `parse_bait_quantity` under the fishing shop's
per-purchase cap.

The contract is deliberately narrow: digit text becomes a non-negative int well past 64-bit, up
to CPython's int-conversion limit, and everything else becomes None. It does not range-check,
does not accept a sign, a decimal point or an exponent, and does not verify where the commas sit
— they are stripped wherever they appear, so `"1,2,3"` is 123. Nothing here reports *why* the
text was rejected either; each caller surfaces its own rejection message before it touches a
balance (`build_invalid_amount_embed` in the economy cog, an invalid-bet embed or a re-rendered
shop notice in the games cog), so a malformed amount can never reach a mutation half-parsed.

It sits in `utils/` rather than inside the economy cog because the games cog parses the same kind
of text and may not import a peer cog to reach it.
"""


def parse_decimal_amount(raw: str | None) -> int | None:
    """Parses decimal text with optional comma separators into an int.

    `strip()` only touches the edges, so an inner space always fails; removing the commas first
    means those edges are measured afterwards, which forgives a comma sitting outside the digits
    too — `", 1,000 ,"` is 1000. `str.isdecimal()` is what gates the conversion, which rejects a
    sign, a decimal point and an exponent, while accepting every Unicode decimal digit, so an
    amount typed in full-width digits by a CJK IME parses instead of reading as junk.

    Args:
        raw (str | None): The user-entered amount text, possibly None or empty.

    Returns:
        The parsed non-negative int, unbounded by 64-bit, or None when the text is empty, not
        decimal, or too long for `int()` to convert.
    """
    normalized = (raw or "").replace(",", "").strip()
    if not normalized.isdecimal():
        return None
    try:
        return int(normalized)
    except ValueError:
        # A digit string past CPython's int-conversion limit (4300 digits by default) passes
        # isdecimal() but int() rejects it; treat it as invalid input, not a crash.
        return None
