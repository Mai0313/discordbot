"""Pure, RNG-injected fish roll engine for the fishing mini-game.

This module performs no I/O. Production passes a `random.SystemRandom`; tests
pass a seeded `random.Random` so the same inputs always produce the same roll.
"""

from random import Random
from collections.abc import Sequence

from discordbot.typings.fishing import (
    LUCK_FACTOR_MAX_BPS,
    LUCK_FACTOR_MIN_BPS,
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

    Luck is additive across gear. Each grade's base weight is scaled by a factor
    that grows with the grade's rarity rank (`order_index`), so a positive shift
    moves roll mass monotonically from common grades toward rare ones. The factor
    is clamped to `[LUCK_FACTOR_MIN_BPS, LUCK_FACTOR_MAX_BPS]` so no gear
    combination can suppress or inflate a grade past those bounds. The most
    common grade (rank 0) is never affected.
    """
    total_shift = rod_rarity_shift_bps + bait_rarity_shift_bps
    adjusted: dict[FishGrade, int] = {}
    for config in grade_configs:
        raw_factor = FISHING_BPS_DENOMINATOR + total_shift * config.order_index
        factor = max(LUCK_FACTOR_MIN_BPS, min(LUCK_FACTOR_MAX_BPS, raw_factor))
        adjusted[config.grade] = max(1, config.weight * factor // FISHING_BPS_DENOMINATOR)
    return adjusted


def _fallback_species(
    species: Sequence[FishSpeciesView],
    grade: FishGrade,
    grade_configs: Sequence[FishGradeConfigView],
) -> list[FishSpeciesView]:
    """Returns species from the populated grade nearest `grade`, preferring lower ranks.

    Defends against a mis-tuned catalog where the rolled grade has no species:
    the catch falls back to the closest grade at or below the rolled rank, or the
    lowest populated grade when none is at or below it.
    """
    rank_by_grade = {config.grade: config.order_index for config in grade_configs}
    target_rank = rank_by_grade.get(grade, 0)
    populated = sorted(
        {item.grade for item in species}, key=lambda candidate: rank_by_grade.get(candidate, 0)
    )
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
        base_value=chosen.base_value,
        value=value,
        capped=raw > max_value,
    )


__all__ = ["compose_grade_weights", "roll_catch"]
