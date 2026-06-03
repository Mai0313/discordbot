"""Tests for the pure fishing roll engine."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random

import pytest

from discordbot.typings.fishing import (
    LUCK_FACTOR_MAX_BPS,
    LUCK_FACTOR_MIN_BPS,
    FISHING_BPS_DENOMINATOR,
    GearType,
    GearView,
    FishGrade,
    FishSpeciesView,
    FishGradeConfigView,
)
from discordbot.cogs._fishing.catch import roll_catch, compose_grade_weights
from discordbot.cogs._fishing.defaults import build_default_catalog


@pytest.fixture
def catalog() -> object:
    """Returns the default catalog views."""
    return build_default_catalog()


def _rod(rarity_shift_bps: int = 0) -> GearView:
    """Builds a test rod with a given luck shift."""
    return GearView(
        gear_id="rod",
        gear_type=GearType.ROD,
        name="rod",
        emoji="🎣",
        tier=0,
        price=1,
        rarity_shift_bps=rarity_shift_bps,
        durability=10,
    )


def _bait(rarity_shift_bps: int = 0, value_bonus_bps: int = 0) -> GearView:
    """Builds a test bait with given luck and value bonuses."""
    return GearView(
        gear_id="bait",
        gear_type=GearType.BAIT,
        name="bait",
        emoji="🪱",
        tier=0,
        price=1,
        rarity_shift_bps=rarity_shift_bps,
        value_bonus_bps=value_bonus_bps,
    )


def test_compose_grade_weights_no_shift_is_unchanged() -> None:
    """A zero luck shift leaves every grade weight unchanged."""
    catalog = build_default_catalog()
    weights = compose_grade_weights(
        grade_configs=catalog.grades, rod_rarity_shift_bps=0, bait_rarity_shift_bps=0
    )
    for config in catalog.grades:
        assert weights[config.grade] == config.weight


def test_compose_grade_weights_positive_shift_raises_rares_monotonically() -> None:
    """A positive shift never touches the common grade and grows with rarity rank."""
    catalog = build_default_catalog()
    base = {config.grade: config.weight for config in catalog.grades}
    weights = compose_grade_weights(
        grade_configs=catalog.grades, rod_rarity_shift_bps=400, bait_rarity_shift_bps=250
    )
    ordered = sorted(catalog.grades, key=lambda config: config.order_index)
    assert weights[FishGrade.N] == base[FishGrade.N]
    ratios = [weights[config.grade] / base[config.grade] for config in ordered]
    assert ratios == sorted(ratios)
    assert ratios[0] == pytest.approx(1.0)
    assert ratios[-1] > 1.0


def test_compose_grade_weights_clamps_extreme_shift() -> None:
    """An extreme positive shift clamps every rare grade to the max luck factor."""
    catalog = build_default_catalog()
    weights = compose_grade_weights(
        grade_configs=catalog.grades, rod_rarity_shift_bps=10_000_000, bait_rarity_shift_bps=0
    )
    for config in catalog.grades:
        if config.order_index == 0:
            assert weights[config.grade] == config.weight
        else:
            expected = config.weight * LUCK_FACTOR_MAX_BPS // FISHING_BPS_DENOMINATOR
            assert weights[config.grade] == expected


def test_compose_grade_weights_clamps_extreme_negative_shift() -> None:
    """An extreme negative shift floors every rare grade at the min luck factor."""
    catalog = build_default_catalog()
    weights = compose_grade_weights(
        grade_configs=catalog.grades, rod_rarity_shift_bps=-10_000_000, bait_rarity_shift_bps=0
    )
    for config in catalog.grades:
        if config.order_index == 0:
            assert weights[config.grade] == config.weight
        else:
            expected = max(1, config.weight * LUCK_FACTOR_MIN_BPS // FISHING_BPS_DENOMINATOR)
            assert weights[config.grade] == expected


def test_compose_grade_weights_honors_zero_weight() -> None:
    """A grade an operator zeroed out stays disabled instead of clamping back to 1."""
    grades = (
        FishGradeConfigView(
            grade=FishGrade.N, weight=100, color=0, emoji="⚪", label="普通", order_index=0
        ),
        FishGradeConfigView(
            grade=FishGrade.SSR, weight=0, color=0, emoji="🟡", label="傳說", order_index=3
        ),
    )
    weights = compose_grade_weights(
        grade_configs=grades, rod_rarity_shift_bps=400, bait_rarity_shift_bps=400
    )
    assert weights[FishGrade.N] == 100
    assert weights[FishGrade.SSR] == 0


def test_roll_catch_is_deterministic_under_seed() -> None:
    """The same seed and inputs always produce an identical roll."""
    catalog = build_default_catalog()
    first = roll_catch(
        rng=Random(99),
        grade_configs=catalog.grades,
        species=catalog.species,
        rod=_rod(),
        bait=_bait(),
        max_value=100_000,
    )
    second = roll_catch(
        rng=Random(99),
        grade_configs=catalog.grades,
        species=catalog.species,
        rod=_rod(),
        bait=_bait(),
        max_value=100_000,
    )
    assert first == second


def test_roll_distribution_matches_theory() -> None:
    """Observed grade frequencies match the base weights at zero shift."""
    catalog = build_default_catalog()
    rng = Random(0)
    counts = {config.grade: 0 for config in catalog.grades}
    rolls = 200_000
    for _ in range(rolls):
        roll = roll_catch(
            rng=rng,
            grade_configs=catalog.grades,
            species=catalog.species,
            rod=_rod(),
            bait=_bait(),
            max_value=100_000,
        )
        counts[roll.grade] += 1
    total_weight = sum(config.weight for config in catalog.grades)
    for config in catalog.grades:
        observed = counts[config.grade] / rolls
        expected = config.weight / total_weight
        assert observed == pytest.approx(expected, abs=0.01)


def test_roll_size_within_species_bounds() -> None:
    """Rolled size stays within the species size range."""
    catalog = build_default_catalog()
    rng = Random(3)
    for _ in range(2_000):
        roll = roll_catch(
            rng=rng,
            grade_configs=catalog.grades,
            species=catalog.species,
            rod=_rod(),
            bait=_bait(),
            max_value=100_000,
        )
        species = next(item for item in catalog.species if item.species_id == roll.species_id)
        assert species.size_min_bps <= roll.size_bps <= species.size_max_bps


def test_value_cap_binds() -> None:
    """The single-catch cap reduces an otherwise larger value and flags it capped."""
    grades = (
        FishGradeConfigView(
            grade=FishGrade.UR, weight=1, color=0, emoji="🐉", label="神話", order_index=0
        ),
    )
    species = (
        FishSpeciesView(
            species_id="leviathan",
            name="巨獸",
            grade=FishGrade.UR,
            emoji="🐉",
            intra_grade_weight=1,
            base_value=1_000_000_000,
            size_min_bps=10_000,
            size_max_bps=10_000,
        ),
    )
    roll = roll_catch(
        rng=Random(1),
        grade_configs=grades,
        species=species,
        rod=_rod(),
        bait=_bait(),
        max_value=100_000,
    )
    assert roll.capped is True
    assert roll.value == 100_000


def test_bait_value_bonus_raises_value() -> None:
    """A value-bonus bait yields a higher value than a plain bait for the same roll."""
    grades = (
        FishGradeConfigView(
            grade=FishGrade.N, weight=1, color=0, emoji="⚪", label="普通", order_index=0
        ),
    )
    species = (
        FishSpeciesView(
            species_id="fixed",
            name="定值魚",
            grade=FishGrade.N,
            emoji="🐟",
            intra_grade_weight=1,
            base_value=1_000,
            size_min_bps=10_000,
            size_max_bps=10_000,
        ),
    )
    plain = roll_catch(
        rng=Random(5),
        grade_configs=grades,
        species=species,
        rod=_rod(),
        bait=_bait(value_bonus_bps=0),
        max_value=10_000_000,
    )
    boosted = roll_catch(
        rng=Random(5),
        grade_configs=grades,
        species=species,
        rod=_rod(),
        bait=_bait(value_bonus_bps=5_000),
        max_value=10_000_000,
    )
    assert plain.value == 1_000
    assert boosted.value == 1_500


def test_empty_grade_falls_back_without_raising() -> None:
    """A rolled grade with no species falls back to a populated grade."""
    grades = (
        FishGradeConfigView(
            grade=FishGrade.N, weight=1, color=0, emoji="⚪", label="普通", order_index=0
        ),
        FishGradeConfigView(
            grade=FishGrade.UR, weight=10_000, color=0, emoji="🔴", label="神話", order_index=4
        ),
    )
    species = (
        FishSpeciesView(
            species_id="only_common",
            name="雜魚",
            grade=FishGrade.N,
            emoji="🐟",
            intra_grade_weight=1,
            base_value=1,
            size_min_bps=10_000,
            size_max_bps=10_000,
        ),
    )
    roll = roll_catch(
        rng=Random(7),
        grade_configs=grades,
        species=species,
        rod=_rod(),
        bait=_bait(),
        max_value=100_000,
    )
    assert roll.species_id == "only_common"


def test_empty_catalog_raises() -> None:
    """Rolling from an empty species catalog raises a clear error."""
    catalog = build_default_catalog()
    with pytest.raises(ValueError, match="empty species catalog"):
        roll_catch(
            rng=Random(0),
            grade_configs=catalog.grades,
            species=(),
            rod=_rod(),
            bait=_bait(),
            max_value=100_000,
        )


def _fish(species_id: str, grade: FishGrade) -> FishSpeciesView:
    """Builds a fixed-size, fixed-value test species in a grade."""
    return FishSpeciesView(
        species_id=species_id,
        name=species_id,
        grade=grade,
        emoji="🐟",
        intra_grade_weight=1,
        base_value=1,
        size_min_bps=10_000,
        size_max_bps=10_000,
    )


def test_fallback_skips_disabled_grade_with_species() -> None:
    """The empty-grade fallback never awards a grade an operator disabled."""
    grades = (
        FishGradeConfigView(
            grade=FishGrade.N, weight=100, color=0, emoji="⚪", label="普通", order_index=0
        ),
        FishGradeConfigView(
            grade=FishGrade.SSR, weight=0, color=0, emoji="🟡", label="傳說", order_index=3
        ),
        FishGradeConfigView(
            grade=FishGrade.UR, weight=10_000, color=0, emoji="🔴", label="神話", order_index=4
        ),
    )
    species = (
        _fish(species_id="common", grade=FishGrade.N),
        _fish(species_id="legend", grade=FishGrade.SSR),
    )
    # UR dominates the draw but has no species, so most rolls hit the fallback; it
    # must land on enabled N, never disabled SSR even though SSR has species.
    for seed in range(50):
        roll = roll_catch(
            rng=Random(seed),
            grade_configs=grades,
            species=species,
            rod=_rod(),
            bait=_bait(),
            max_value=100_000,
        )
        assert roll.species_id == "common"


def test_fallback_raises_when_all_populated_grades_disabled() -> None:
    """If every grade with species is disabled, the roll fails instead of awarding one."""
    grades = (
        FishGradeConfigView(
            grade=FishGrade.N, weight=0, color=0, emoji="⚪", label="普通", order_index=0
        ),
        FishGradeConfigView(
            grade=FishGrade.UR, weight=10_000, color=0, emoji="🔴", label="神話", order_index=4
        ),
    )
    with pytest.raises(ValueError, match="every populated grade is disabled"):
        roll_catch(
            rng=Random(0),
            grade_configs=grades,
            species=(_fish(species_id="only_common", grade=FishGrade.N),),
            rod=_rod(),
            bait=_bait(),
            max_value=100_000,
        )


def test_all_zero_catalog_raises_before_awarding() -> None:
    """A fully disabled catalog fails instead of awarding the index-0 grade directly."""
    grades = (
        FishGradeConfigView(
            grade=FishGrade.N, weight=0, color=0, emoji="⚪", label="普通", order_index=0
        ),
        FishGradeConfigView(
            grade=FishGrade.UR, weight=0, color=0, emoji="🔴", label="神話", order_index=4
        ),
    )
    # The rank-0 disabled grade has species, so without the guard _weighted_index's
    # total<=0 branch would return index 0 and award it directly.
    with pytest.raises(ValueError, match="every grade is disabled"):
        roll_catch(
            rng=Random(0),
            grade_configs=grades,
            species=(_fish(species_id="common", grade=FishGrade.N),),
            rod=_rod(),
            bait=_bait(),
            max_value=100_000,
        )
