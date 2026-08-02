"""Pure, RNG-injected fish roll engine for the fishing mini-game.

This module performs no I/O. Production passes a `random.SystemRandom`; tests
pass a seeded `random.Random` so the same inputs always produce the same roll.
"""

from random import Random
from collections.abc import Sequence

from discordbot.typings.fishing import (
    LUCK_STEP_MAX_BPS,
    LUCK_STEP_MIN_BPS,
    FISHING_BPS_DENOMINATOR,
    GearView,
    CatchRoll,
    FishGrade,
    FishSpeciesView,
    FishGradeConfigView,
)


def _weighted_index(rng: Random, weights: Sequence[int]) -> int:
    """Returns a weighted random index via a cumulative scan over `rng.random()`."""
    total = sum(weights)
    if total <= 0:
        return 0
    target = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            return index
    return len(weights) - 1


def compose_grade_weights(
    grade_configs: Sequence[FishGradeConfigView],
    rod_rarity_shift_bps: int,
    bait_rarity_shift_bps: int,
) -> dict[FishGrade, int]:
    """Reweights grade roll weights by the combined rod and bait luck shift.

    Luck is additive across gear and compounds per rarity step: the combined
    shift describes one step up the ladder, and a grade is scaled by that step
    raised to the number of steps it sits above the commonest one. Compounding
    is what makes the knob bite at all. Base weights decay geometrically with
    rarity, so a factor growing only linearly cannot outrun them: it converges on
    weights proportional to `weight * rank`, which caps 神話 at 1.53% of rolls
    however large the shift gets and lands most of the moved mass on the
    second-commonest grade. Measured against the pre-#351 catalog WITH its clamp
    in force, the whole ladder's EV topped out at 2.45x its floor and then fell
    back to 1.87x as the shift grew (#351).

    The exponent is a grade's POSITION in `order_index` order, never the raw
    value, and ties share a position. `order_index` is also the display order, so
    an operator may well leave gaps in it, and a raw exponent turns a perfectly
    ordinary 0/10/20/30/40 catalog into 94% 神話; sharing a position on a tie
    keeps two grades an operator ranked equal from being separated by whatever
    row order SQLite happened to return. Position also means the commonest grade
    is untouched even when its own `order_index` is not zero, and bounds the
    arithmetic to the number of grades. The step itself is clamped to
    `[LUCK_STEP_MIN_BPS, LUCK_STEP_MAX_BPS]`, so together the two bound what a
    mis-typed catalog can do to the ladder. A grade whose base weight is zero
    stays disabled but keeps its position, so removing one never re-spaces the
    grades around it; the floor-to-1 below only protects a positive weight from
    rounding away, it must not resurrect a grade an operator removed.
    """
    total_shift = rod_rarity_shift_bps + bait_rarity_shift_bps
    step = max(LUCK_STEP_MIN_BPS, min(LUCK_STEP_MAX_BPS, FISHING_BPS_DENOMINATOR + total_shift))
    ranks: dict[int, int] = {}
    for config in sorted(grade_configs, key=lambda item: item.order_index):
        ranks.setdefault(config.order_index, len(ranks))
    adjusted: dict[FishGrade, int] = {}
    for config in grade_configs:
        if config.weight <= 0:
            adjusted[config.grade] = 0
            continue
        rank = ranks[config.order_index]
        scaled = config.weight * step**rank // FISHING_BPS_DENOMINATOR**rank
        adjusted[config.grade] = max(1, scaled)
    return adjusted


def _fallback_species(
    species: Sequence[FishSpeciesView],
    grade: FishGrade,
    grade_configs: Sequence[FishGradeConfigView],
) -> list[FishSpeciesView]:
    """Returns species from the populated grade nearest `grade`, preferring lower ranks.

    Defends against a mis-tuned catalog where the rolled grade has no species:
    the catch falls back to the closest grade at or below the rolled rank, or the
    lowest populated grade when none is at or below it. Grades an operator disabled
    (base weight zero) are never eligible here either, so the disabled-grade
    contract still holds on the fallback path; if every populated grade is disabled
    the roll fails rather than awarding a disabled grade.
    """
    rank_by_grade = {config.grade: config.order_index for config in grade_configs}
    disabled = {config.grade for config in grade_configs if config.weight <= 0}
    target_rank = rank_by_grade.get(grade, 0)
    populated = sorted(
        {item.grade for item in species if item.grade not in disabled},
        key=lambda candidate: rank_by_grade.get(candidate, 0),
    )
    if not populated:
        msg = "cannot roll a catch: every populated grade is disabled"
        raise ValueError(msg)
    lower_or_equal = [
        candidate for candidate in populated if rank_by_grade.get(candidate, 0) <= target_rank
    ]
    fallback_grade = lower_or_equal[-1] if lower_or_equal else populated[0]
    return sorted(
        (item for item in species if item.grade == fallback_grade),
        key=lambda item: item.species_id,
    )


def _select_species(
    rng: Random,
    species: Sequence[FishSpeciesView],
    grade: FishGrade,
    grade_configs: Sequence[FishGradeConfigView],
) -> FishSpeciesView:
    """Picks a species in the rolled grade, falling back to the nearest populated grade."""
    in_grade = sorted(
        (item for item in species if item.grade == grade), key=lambda item: item.species_id
    )
    if not in_grade:
        in_grade = _fallback_species(species=species, grade=grade, grade_configs=grade_configs)
    weights = [item.intra_grade_weight for item in in_grade]
    return in_grade[_weighted_index(rng=rng, weights=weights)]


def _size_rank_bps(species: FishSpeciesView, size_bps: int) -> int:
    """Returns where a rolled size sits inside its own species' band, in basis points.

    Size bands differ per grade, so the raw multiplier no longer says whether a
    catch was a big one for its kind; this does. A fixed-size species has no band
    to sit in and reports 0, since calling every one of its catches a big one is
    the more misleading of the two answers.
    """
    span = species.size_max_bps - species.size_min_bps
    if span <= 0:
        return 0
    return (size_bps - species.size_min_bps) * FISHING_BPS_DENOMINATOR // span


def roll_catch(  # noqa: PLR0913 -- a roll needs rng, configs, species, rod, bait, and the cap
    rng: Random,
    grade_configs: Sequence[FishGradeConfigView],
    species: Sequence[FishSpeciesView],
    rod: GearView,
    bait: GearView,
    max_value: int,
) -> CatchRoll:
    """Rolls a grade, then a species, then a size, returning a pure catch result.

    The grade is drawn from the luck-adjusted weights, the species from the
    intra-grade weights within that grade, and the size uniformly across the
    species' basis-point range. The final value applies the size multiplier and
    the bait value bonus, then clamps to `max_value`.
    """
    if not species:
        msg = "cannot roll a catch from an empty species catalog"
        raise ValueError(msg)
    ordered_configs = sorted(grade_configs, key=lambda config: config.order_index)
    weights = compose_grade_weights(
        grade_configs=ordered_configs,
        rod_rarity_shift_bps=rod.rarity_shift_bps,
        bait_rarity_shift_bps=bait.rarity_shift_bps,
    )
    grade_choices = [config.grade for config in ordered_configs]
    grade_weights = [weights[config.grade] for config in ordered_configs]
    # An all-disabled catalog would otherwise fall through _weighted_index's
    # total <= 0 branch to index 0 and award that disabled grade directly.
    if sum(grade_weights) <= 0:
        msg = "cannot roll a catch: every grade is disabled"
        raise ValueError(msg)
    chosen_grade = grade_choices[_weighted_index(rng=rng, weights=grade_weights)]
    chosen = _select_species(
        rng=rng, species=species, grade=chosen_grade, grade_configs=ordered_configs
    )
    size_bps = rng.randint(chosen.size_min_bps, chosen.size_max_bps)
    raw = chosen.base_value * size_bps // FISHING_BPS_DENOMINATOR
    raw = raw * (FISHING_BPS_DENOMINATOR + bait.value_bonus_bps) // FISHING_BPS_DENOMINATOR
    value = min(raw, max_value)
    return CatchRoll(
        species_id=chosen.species_id,
        species_name=chosen.name,
        grade=chosen.grade,
        emoji=chosen.emoji,
        size_bps=size_bps,
        size_rank_bps=_size_rank_bps(species=chosen, size_bps=size_bps),
        base_value=chosen.base_value,
        value=value,
        capped=raw > max_value,
    )


__all__ = ["compose_grade_weights", "roll_catch"]
