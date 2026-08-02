"""Pure helpers for the fishing shop: quantity parsing and gear partitioning."""

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
    """
    return f"稀有度 每階+{rarity_shift_bps / 100:.0f}%"


def parse_bait_quantity(raw_quantity: str | None) -> int | None:
    """Parses a bait purchase quantity with optional comma separators.

    Returns the integer quantity when it is a positive number within the per
    purchase cap, or None when the input is malformed or out of range.
    """
    quantity = parse_decimal_amount(raw=raw_quantity)
    if quantity is None or quantity < 1 or quantity > MAX_BAIT_PER_PURCHASE:
        return None
    return quantity


def partition_gear(
    gear: tuple[GearView, ...],
) -> tuple[tuple[GearView, ...], tuple[GearView, ...]]:
    """Splits a gear catalog into (rods, baits), each sorted by tier."""
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
    """Builds a select-option label for a gear item within Discord's length limit."""
    label = f"{gear.emoji} {gear.name} · {gear.price:,}"
    if len(label) <= SELECT_OPTION_LABEL_LIMIT:
        return label
    return f"{label[: SELECT_OPTION_LABEL_LIMIT - 3]}..."


def gear_option_description(gear: GearView) -> str:
    """Builds a select-option description summarizing a gear item's stats."""
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
