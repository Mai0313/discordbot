"""Pure shop helpers for the fishing mini-game: input parsing, catalog ordering, option text.

`views.py` owns the shop's interaction plumbing and `presentation.py` its embeds; this file is
the half that touches neither nextcord nor the database, so both can call into it and a test can
exercise it with nothing but a catalog. It holds `parse_bait_quantity` (the fishing wrapper that
range-checks `utils.amount_parsing.parse_decimal_amount` against `MAX_BAIT_PER_PURCHASE`),
`partition_gear` (the catalog split that fixes the rod / bait order both surfaces render in), and
the two builders for a gear select option's label and description.

`rarity_bonus_text` sits here rather than in `presentation.py` because the shop embed line and
the select option's description have to word one gear's luck shift identically, and how that
number is worded is a decision with its own reasoning (see its docstring); `presentation.py`
imports it from here.

`SELECT_OPTION_LABEL_LIMIT` duplicates the identical constant in `cogs/stock/views.py` on
purpose: a cog may not import from a peer cog, so each spells Discord's limit itself rather than
reaching across for it.
"""

from discordbot.typings.fishing import MAX_BAIT_PER_PURCHASE, GearType, GearView
from discordbot.utils.amount_parsing import parse_decimal_amount

SELECT_OPTION_LABEL_LIMIT = 100


def rarity_bonus_text(rarity_shift_bps: int) -> str:
    """Formats a gear's luck shift as the per-rarity-step figure it actually is.

    Deliberately not a bare `稀有+40%`, which reads as a change in the odds and
    is not one: the shift applies once per step up the rarity ladder, so it
    compounds, and the real move at the top of the ladder is several times this.
    Deliberately not a multiplier either, tempting as `x1.40` looks: a rod and a
    bait ADD their shifts, so two multipliers shown side by side would invite a
    reader to multiply x1.40 by x1.25 and get x1.75 where the truth is x1.65.

    Args:
        rarity_shift_bps (int): The gear's own luck shift, in basis points.

    Returns:
        The rarity line shown both in the shop embed and in a gear select option.
    """
    return f"稀有度 每階+{rarity_shift_bps / 100:.0f}%"


def parse_bait_quantity(raw_quantity: str | None) -> int | None:
    """Parses a bait purchase quantity and holds it inside the per-purchase cap.

    Every rejection collapses to the same None, because the caller answers all of them with one
    notice and a re-rendered shop. `parse_decimal_amount` already refuses a sign, so the lower
    bound here only ever catches a typed zero.

    Args:
        raw_quantity (str | None): The quantity text typed into the bait modal.

    Returns:
        The quantity as an int between 1 and `MAX_BAIT_PER_PURCHASE`, or None when the text is
        not decimal or falls outside that range.
    """
    quantity = parse_decimal_amount(raw=raw_quantity)
    if quantity is None or quantity < 1 or quantity > MAX_BAIT_PER_PURCHASE:
        return None
    return quantity


def partition_gear(
    gear: tuple[GearView, ...],
) -> tuple[tuple[GearView, ...], tuple[GearView, ...]]:
    """Splits a gear catalog into (rods, baits), each sorted by tier.

    Tier is the order both shop surfaces render in, so whatever order the catalog rows were read
    in never reaches the user.

    Args:
        gear (tuple[GearView, ...]): The whole gear catalog, in any order.

    Returns:
        The rods and the baits, each ascending by tier.
    """
    rods = tuple(
        sorted(
            (item for item in gear if item.gear_type == GearType.ROD), key=lambda item: item.tier
        )
    )
    baits = tuple(
        sorted(
            (item for item in gear if item.gear_type == GearType.BAIT), key=lambda item: item.tier
        )
    )
    return rods, baits


def gear_option_label(gear: GearView) -> str:
    """Builds a select-option label for a gear item within Discord's length limit.

    An over-long label is trimmed to exactly the limit with a trailing ellipsis rather than
    refused: the option's value carries the gear id, so display text can be lost without losing
    which item the user picked.

    Args:
        gear (GearView): The catalog row this option buys.

    Returns:
        A label of at most `SELECT_OPTION_LABEL_LIMIT` characters.
    """
    label = f"{gear.emoji} {gear.name} · {gear.price:,}"
    if len(label) <= SELECT_OPTION_LABEL_LIMIT:
        return label
    return f"{label[: SELECT_OPTION_LABEL_LIMIT - 3]}..."


def gear_option_description(gear: GearView) -> str:
    """Builds a select-option description summarizing a gear item's stats.

    Each kind shows only the stat it uses: durability is always zero on a bait and the value bonus
    always zero on a rod, so listing both would put a permanent `0` on every option. Discord caps
    a description at the same 100 characters as a label, so the label constant bounds this too.

    Args:
        gear (GearView): The catalog row this option buys.

    Returns:
        A description of at most `SELECT_OPTION_LABEL_LIMIT` characters.
    """
    rarity = rarity_bonus_text(rarity_shift_bps=gear.rarity_shift_bps)
    if gear.gear_type == GearType.ROD:
        description = f"耐久 {gear.durability}・{rarity}"
    else:
        description = f"{rarity}・價值+{gear.value_bonus_bps / 100:.1f}%"
    return description[:SELECT_OPTION_LABEL_LIMIT]


__all__ = [
    "SELECT_OPTION_LABEL_LIMIT",
    "gear_option_description",
    "gear_option_label",
    "parse_bait_quantity",
    "partition_gear",
    "rarity_bonus_text",
]
