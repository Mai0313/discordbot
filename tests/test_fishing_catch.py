"""Exact guards over the fishing roll engine and the tuning of the catalog it reads.

Pins `cogs/games/fishing/catch.py`, the pure RNG-injected engine one cast runs through, together
with the hand-written default catalog in `defaults.py` that feeds it. Two different things are
held in place here and they need different kinds of assertion, so the file reads in two halves.

The engine half covers what a cast produces and what a mis-tuned catalog must not be able to make
it do. A seeded roll is reproducible and observed grade frequencies match the base weights;
`compose_grade_weights` compounds the combined rod-and-bait luck shift once per rarity step, with
the exponent read off a grade's POSITION in `order_index` order rather than the raw value, so gaps
and ties in a hand-written catalog rank the grades without multiplying the step; the step itself
is clamped at both ends; a grade an operator zeroed out stays disabled, keeps its place in the
ladder, and is never resurrected by the empty-grade fallback; and a catalog with nothing left to
award raises rather than quietly handing back the first grade in the list. The size assertions pin
the band a roll stays inside plus `size_rank_bps`, the position within the species' own band that
the 大物 marker reads and that a fixed-size species has to report as 0.

The tuning half pins #351's design over the default catalog, which is data an operator retunes by
hand and seeds offline, so a retuning that breaks the design fails here instead of in a deployed
bot. Fishing is not a sink: every rod-and-bait pairing returns at least its own per-cast cost and
at most `_FAUCET_CEILING` of it, upgrading either axis never lowers the return and always raises
the rare-catch rate, the whole ladder at least doubles that rate, and on any one bait each grade's
payout band sits entirely above the grade below it.

Those figures are computed exactly rather than sampled, because adjacent rungs of the gear ladder
differ by as little as one point of return and Monte Carlo noise at any affordable sample size
would swamp that. The `_expected_*`, `_return_ratio` and `_rare_catch_rate` helpers below are
therefore a second, closed-form model of the same arithmetic `roll_catch` performs, and
`test_analytic_expected_value_matches_a_seeded_simulation` is what stops the two drifting apart.
"""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random
from functools import cache

import pytest

from discordbot.typings.fishing import (
    LUCK_STEP_MAX_BPS,
    LUCK_STEP_MIN_BPS,
    FISHING_BPS_DENOMINATOR,
    FISHING_MAX_SINGLE_CATCH,
    GearType,
    GearView,
    FishGrade,
    FishingCatalog,
    FishSpeciesView,
    FishGradeConfigView,
)
from discordbot.cogs.games.fishing.catch import roll_catch, compose_grade_weights
from discordbot.cogs.games.fishing.defaults import build_default_catalog


@pytest.fixture
def catalog() -> object:
    """Hands a test the default catalog without touching games.db.

    Returns:
        The default grades, species and gear bundled as one `FishingCatalog`.
    """
    return build_default_catalog()


def _rod(rarity_shift_bps: int = 0) -> GearView:
    """Builds a test rod carrying a given luck shift.

    That shift is all `roll_catch` reads off a rod, so the price and durability here are
    placeholders; the return-ratio helpers, which do read them, take the catalog's own rods.

    Returns:
        A rod view for a direct `roll_catch` call.
    """
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
    """Builds a test bait carrying a given luck shift and value bonus.

    Those two are all `roll_catch` reads off a bait; its price is a placeholder for the same
    reason a rod's is.

    Returns:
        A bait view for a direct `roll_catch` call.
    """
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


# Ceiling on a cast's expected return as a MULTIPLE OF ITS OWN COST. Fishing
# stopped being a sink in #351, so this replaces the old "EV must stay below
# cost" rule. It deliberately does not bound the absolute mint rate, which runs
# from +2.9 per cast on starter gear to +88 on top gear: nothing here caps that,
# and there is no cast cooldown either.
_FAUCET_CEILING = 1.7
# Grades a player would call a rare catch, used to measure whether better gear
# changes what comes out of the water rather than only what it is worth.
_RARE_GRADES = (FishGrade.SR, FishGrade.SSR, FishGrade.UR)


def _payout_band(species: FishSpeciesView) -> tuple[int, int]:
    """Prices a species' size band at both ends, before any bait bonus.

    Returns:
        The payout at `size_min_bps` and at `size_max_bps`, under the same integer division
        `roll_catch` applies, so the two agree on the rounding.
    """
    return (
        species.base_value * species.size_min_bps // FISHING_BPS_DENOMINATOR,
        species.base_value * species.size_max_bps // FISHING_BPS_DENOMINATOR,
    )


@cache
def _expected_species_value(species: FishSpeciesView, bait: GearView) -> float:
    """Averages one species' payout over every size in its band, exactly.

    Enumerated rather than sampled: the tuning guards compare rungs of the gear
    ladder that differ by as little as one point of return, which Monte Carlo
    noise at any affordable sample size would swamp. A band runs to 20001 sizes
    and every combo re-prices the whole species table, hence the cache, which the
    frozen views make possible by being hashable.

    Returns:
        The mean payout after the bait bonus and the single-catch cap.
    """
    total = 0
    for size_bps in range(species.size_min_bps, species.size_max_bps + 1):
        raw = species.base_value * size_bps // FISHING_BPS_DENOMINATOR
        raw = raw * (FISHING_BPS_DENOMINATOR + bait.value_bonus_bps) // FISHING_BPS_DENOMINATOR
        total += min(raw, FISHING_MAX_SINGLE_CATCH)
    return total / (species.size_max_bps - species.size_min_bps + 1)


def _grade_rates(catalog: FishingCatalog, rod: GearView, bait: GearView) -> dict[FishGrade, float]:
    """Normalizes the luck-adjusted grade weights into probabilities.

    Returns:
        One probability per grade in the catalog, summing to 1.
    """
    weights = compose_grade_weights(
        grade_configs=catalog.grades,
        rod_rarity_shift_bps=rod.rarity_shift_bps,
        bait_rarity_shift_bps=bait.rarity_shift_bps,
    )
    total = sum(weights.values())
    return {grade: weight / total for grade, weight in weights.items()}


def _expected_catch_value(catalog: FishingCatalog, rod: GearView, bait: GearView) -> float:
    """Prices one cast: each grade's rate against the mean value of that grade's species.

    Mirrors `roll_catch`'s arithmetic rather than calling it, which is what the closed form buys
    and also its one risk; the seeded cross-check test is what holds the two models together. A
    grade with no species fails the assertion here instead of pricing as zero, since the engine
    would fall back to another grade and the figure would then be wrong rather than missing.

    Returns:
        The expected payout of one cast.
    """
    expected = 0.0
    for grade, rate in _grade_rates(catalog=catalog, rod=rod, bait=bait).items():
        in_grade = [item for item in catalog.species if item.grade == grade]
        assert in_grade, f"default catalog has no species in grade {grade}"
        grade_weight = sum(item.intra_grade_weight for item in in_grade)
        expected += (
            rate
            * sum(
                item.intra_grade_weight * _expected_species_value(species=item, bait=bait)
                for item in in_grade
            )
            / grade_weight
        )
    return expected


def _return_ratio(catalog: FishingCatalog, rod: GearView, bait: GearView) -> float:
    """Divides a cast's expected payout by what that cast costs to make.

    A rod is bought once and spent over its durability, so its cost is amortized per cast and
    added to the bait's full price. That additive cost against a multiplicative payout is the
    shape the gear ladder has to be priced around.

    Returns:
        Expected payout as a multiple of per-cast cost, where 1.0 is break-even.
    """
    cost = bait.price + rod.price / rod.durability
    return _expected_catch_value(catalog=catalog, rod=rod, bait=bait) / cost


def _rare_catch_rate(catalog: FishingCatalog, rod: GearView, bait: GearView) -> float:
    """Sums the grade rates a player would call a rare catch.

    Returns:
        The chance one cast lands 史詩 or better.
    """
    rates = _grade_rates(catalog=catalog, rod=rod, bait=bait)
    return sum(rates[grade] for grade in _RARE_GRADES)


def _default_gear_ladders(catalog: FishingCatalog) -> tuple[list[GearView], list[GearView]]:
    """Splits the catalog's gear into the two upgrade ladders a player climbs.

    Returns:
        The rods and the baits, each sorted by `tier`, so list order is upgrade order.
    """
    rods = sorted(
        (gear for gear in catalog.gear if gear.gear_type is GearType.ROD),
        key=lambda gear: gear.tier,
    )
    baits = sorted(
        (gear for gear in catalog.gear if gear.gear_type is GearType.BAIT),
        key=lambda gear: gear.tier,
    )
    return rods, baits


def _default_combos(catalog: FishingCatalog) -> list[tuple[GearView, GearView]]:
    """Enumerates every rod and bait pairing a player can actually cast with.

    The per-combo guards have to hold on the mismatched pairings too, not only on matched tiers:
    the top rod with the cheapest bait is what caps rod amortization, and the cheapest rod with
    the top bait is what caps that bait's price.

    Returns:
        Every (rod, bait) pair, rods outermost.
    """
    rods, baits = _default_gear_ladders(catalog=catalog)
    return [(rod, bait) for rod in rods for bait in baits]


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
        grade_configs=catalog.grades, rod_rarity_shift_bps=4_000, bait_rarity_shift_bps=2_500
    )
    ordered = sorted(catalog.grades, key=lambda config: config.order_index)
    assert weights[FishGrade.N] == base[FishGrade.N]
    ratios = [weights[config.grade] / base[config.grade] for config in ordered]
    assert ratios == sorted(ratios)
    assert ratios[0] == pytest.approx(1.0)
    assert ratios[-1] > 1.0


def test_compose_grade_weights_compounds_the_shift_per_rarity_step() -> None:
    """Each rarity rank multiplies by the step again, rather than adding to it.

    This is the difference that makes gear felt at all: the base weights decay
    geometrically with rank, so a factor that only grew linearly could never move
    mass onto the top grades no matter how large the shift got.
    """
    catalog = build_default_catalog()
    base = {config.grade: config.weight for config in catalog.grades}
    weights = compose_grade_weights(
        grade_configs=catalog.grades, rod_rarity_shift_bps=3_000, bait_rarity_shift_bps=2_000
    )
    for config in catalog.grades:
        step = FISHING_BPS_DENOMINATOR + 5_000
        expected = (
            base[config.grade]
            * step**config.order_index
            // FISHING_BPS_DENOMINATOR**config.order_index
        )
        assert weights[config.grade] == expected


def test_compose_grade_weights_clamps_extreme_shift() -> None:
    """An extreme positive shift clamps the per-step multiplier to its ceiling."""
    catalog = build_default_catalog()
    weights = compose_grade_weights(
        grade_configs=catalog.grades, rod_rarity_shift_bps=10_000_000, bait_rarity_shift_bps=0
    )
    for config in catalog.grades:
        expected = (
            config.weight
            * LUCK_STEP_MAX_BPS**config.order_index
            // FISHING_BPS_DENOMINATOR**config.order_index
        )
        assert weights[config.grade] == max(1, expected)


def test_compose_grade_weights_clamps_extreme_negative_shift() -> None:
    """An extreme negative shift floors the per-step multiplier instead of inverting it."""
    catalog = build_default_catalog()
    weights = compose_grade_weights(
        grade_configs=catalog.grades, rod_rarity_shift_bps=-10_000_000, bait_rarity_shift_bps=0
    )
    for config in catalog.grades:
        expected = (
            config.weight
            * LUCK_STEP_MIN_BPS**config.order_index
            // FISHING_BPS_DENOMINATOR**config.order_index
        )
        assert weights[config.grade] == max(1, expected)


def _ladder(order_indices: tuple[int, ...]) -> tuple[FishGradeConfigView, ...]:
    """Rebuilds the default grade weights under a chosen set of `order_index` values.

    The weights are the default catalog's, so a test can vary the ranking on its own and compare
    the result against the contiguous 0..4 spelling of the same ladder.

    Returns:
        One config per grade, in the fixed N/R/SR/SSR/UR order the weights are written in.
    """
    weights = (6_000, 3_000, 800, 180, 20)
    grades = (FishGrade.N, FishGrade.R, FishGrade.SR, FishGrade.SSR, FishGrade.UR)
    return tuple(
        FishGradeConfigView(
            grade=grade, weight=weight, color=0, emoji="⚪", label=grade.value, order_index=index
        )
        for grade, weight, index in zip(grades, weights, order_indices, strict=True)
    )


def test_compose_grade_weights_reads_order_index_as_a_position_not_a_power() -> None:
    """Gaps in order_index order the grades; they do not multiply the luck step.

    order_index is also the display order, so an operator may reasonably leave
    room between grades. Raising the step to the raw value instead of the
    position turns that catalog into a 94%-神話 one.
    """
    contiguous = compose_grade_weights(
        grade_configs=_ladder(order_indices=(0, 1, 2, 3, 4)),
        rod_rarity_shift_bps=4_000,
        bait_rarity_shift_bps=2_500,
    )
    sparse = compose_grade_weights(
        grade_configs=_ladder(order_indices=(0, 10, 20, 30, 40)),
        rod_rarity_shift_bps=4_000,
        bait_rarity_shift_bps=2_500,
    )
    assert sparse == contiguous


def test_compose_grade_weights_gives_tied_grades_the_same_step() -> None:
    """Two grades an operator ranked equal are scaled equally, whatever order they arrive in.

    `list_grade_configs` orders by `order_index` alone, so a tie is broken by
    SQLite's row order; without a shared rank that would decide which of the two
    got the higher exponent.
    """
    declared = _ladder(order_indices=(0, 1, 1, 2, 3))
    swapped = (declared[0], declared[2], declared[1], declared[3], declared[4])
    weights = compose_grade_weights(
        grade_configs=declared, rod_rarity_shift_bps=4_000, bait_rarity_shift_bps=2_500
    )
    assert weights == compose_grade_weights(
        grade_configs=swapped, rod_rarity_shift_bps=4_000, bait_rarity_shift_bps=2_500
    )
    assert weights[FishGrade.R] / 3_000 == weights[FishGrade.SR] / 800


def test_compose_grade_weights_leaves_the_commonest_grade_alone_at_any_order_index() -> None:
    """The lowest-ranked grade is untouched even when its own order_index is not zero."""
    grades = _ladder(order_indices=(7, 8, 9, 10, 11))
    weights = compose_grade_weights(
        grade_configs=grades, rod_rarity_shift_bps=4_000, bait_rarity_shift_bps=2_500
    )
    assert weights[FishGrade.N] == 6_000
    assert weights == compose_grade_weights(
        grade_configs=_ladder(order_indices=(0, 1, 2, 3, 4)),
        rod_rarity_shift_bps=4_000,
        bait_rarity_shift_bps=2_500,
    )


def test_compose_grade_weights_keeps_a_disabled_grade_from_respacing_the_ladder() -> None:
    """A zero-weight grade stays disabled and still occupies its step of the ladder."""
    grades = (
        FishGradeConfigView(
            grade=FishGrade.N, weight=6_000, color=0, emoji="⚪", label="普通", order_index=0
        ),
        FishGradeConfigView(
            grade=FishGrade.R, weight=0, color=0, emoji="🔵", label="稀有", order_index=1
        ),
        FishGradeConfigView(
            grade=FishGrade.SR, weight=800, color=0, emoji="🟣", label="史詩", order_index=2
        ),
    )
    weights = compose_grade_weights(
        grade_configs=grades, rod_rarity_shift_bps=5_000, bait_rarity_shift_bps=0
    )
    assert weights[FishGrade.R] == 0
    assert weights[FishGrade.SR] == 800 * 15_000**2 // FISHING_BPS_DENOMINATOR**2


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


def test_every_default_combo_returns_at_least_its_cost() -> None:
    """Fishing is no longer a sink: no combo pays back less than it costs to cast."""
    catalog = build_default_catalog()
    for rod, bait in _default_combos(catalog=catalog):
        ratio = _return_ratio(catalog=catalog, rod=rod, bait=bait)
        assert ratio >= 1.0, (
            f"{rod.gear_id}+{bait.gear_id} is a sink: returns {ratio:.0%} of its cost"
        )


def test_no_default_combo_prints_past_the_faucet_ceiling() -> None:
    """No combo returns more than `_FAUCET_CEILING` times what its cast cost."""
    catalog = build_default_catalog()
    for rod, bait in _default_combos(catalog=catalog):
        ratio = _return_ratio(catalog=catalog, rod=rod, bait=bait)
        assert ratio <= _FAUCET_CEILING, (
            f"{rod.gear_id}+{bait.gear_id} returns {ratio:.0%}, "
            f"over the {_FAUCET_CEILING:.0%} ceiling"
        )


def test_upgrading_either_gear_axis_never_lowers_the_return() -> None:
    """Spending more never pays back less, the complaint that opened #351.

    Checked along each axis on its own, so a better rod with the same bait and a
    better bait with the same rod both have to hold.
    """
    catalog = build_default_catalog()
    rods, baits = _default_gear_ladders(catalog=catalog)
    for bait in baits:
        ratios = [_return_ratio(catalog=catalog, rod=rod, bait=bait) for rod in rods]
        assert ratios == sorted(ratios), f"rod ladder pays backwards with {bait.gear_id}: {ratios}"
    for rod in rods:
        ratios = [_return_ratio(catalog=catalog, rod=rod, bait=bait) for bait in baits]
        assert ratios == sorted(ratios), f"bait ladder pays backwards with {rod.gear_id}: {ratios}"


def test_upgrading_either_gear_axis_raises_the_rare_catch_rate() -> None:
    """Better gear changes what comes out of the water, not only what it is worth."""
    catalog = build_default_catalog()
    rods, baits = _default_gear_ladders(catalog=catalog)
    for bait in baits:
        rates = [_rare_catch_rate(catalog=catalog, rod=rod, bait=bait) for rod in rods]
        assert rates == sorted(rates)
        assert rates[-1] > rates[0]
    for rod in rods:
        rates = [_rare_catch_rate(catalog=catalog, rod=rod, bait=bait) for bait in baits]
        assert rates == sorted(rates)
        assert rates[-1] > rates[0]


def test_the_full_gear_ladder_at_least_doubles_the_rare_catch_rate() -> None:
    """The whole ladder buys a difference a player can feel, not a rounding one.

    The pre-#351 catalog moved 史詩以上 from 10.00% to 11.06% across the entire
    ladder, which is what "better gear barely changes rarity" measured.
    """
    catalog = build_default_catalog()
    rods, baits = _default_gear_ladders(catalog=catalog)
    worst = _rare_catch_rate(catalog=catalog, rod=rods[0], bait=baits[0])
    best = _rare_catch_rate(catalog=catalog, rod=rods[-1], bait=baits[-1])
    assert best >= worst * 2


def test_each_grade_pays_more_than_every_grade_below_it() -> None:
    """Grade payout bands do not overlap, so a rarer catch is worth more on the same bait.

    Measured before the bait bonus, which is the catalog property being guarded.
    The bonus scales every grade alike, so the ladder holds for whichever bait a
    player is casting with; it does NOT hold across two different baits, where a
    稀有 on 路亞 outpays a 史詩 on 蟲餌.
    """
    catalog = build_default_catalog()
    ordered = sorted(catalog.grades, key=lambda config: config.order_index)
    previous_best = 0
    for config in ordered:
        in_grade = [item for item in catalog.species if item.grade == config.grade]
        assert in_grade, f"default catalog has no species in grade {config.grade}"
        worst = min(_payout_band(species=item)[0] for item in in_grade)
        best = max(_payout_band(species=item)[1] for item in in_grade)
        assert worst > previous_best, (
            f"{config.grade} can pay {worst}, at or below the {previous_best} "
            "the grade under it can reach"
        )
        previous_best = best


def test_a_rare_catch_never_surfaces_below_its_promised_multiplier() -> None:
    """A 史詩 always reads 1.50x or better, a 傳說 2.00x, and a 神話 3.00x."""
    catalog = build_default_catalog()
    floors = {FishGrade.SR: 15_000, FishGrade.SSR: 20_000, FishGrade.UR: 30_000}
    for species in catalog.species:
        floor = floors.get(species.grade)
        if floor is None:
            continue
        assert species.size_min_bps >= floor, (
            f"{species.species_id} can surface at {species.size_min_bps / 10_000:.2f}x"
        )


@pytest.mark.parametrize(
    ("rod_id", "bait_id"), [("rod_bamboo", "bait_worm"), ("rod_legend", "bait_lure")]
)
def test_analytic_expected_value_matches_a_seeded_simulation(rod_id: str, bait_id: str) -> None:
    """Pins the exact EV arithmetic above against what roll_catch actually pays.

    The tuning guards read an analytic expectation because adjacent rungs of the
    ladder differ by as little as one point of return, which sampling noise
    cannot resolve. This is the test that stops the two models drifting apart,
    so it runs both ends of the ladder: the zero-shift, zero-bonus combo, and the
    one that exercises the compounded weights and the bait value bonus.
    """
    catalog = build_default_catalog()
    rod = next(gear for gear in catalog.gear if gear.gear_id == rod_id)
    bait = next(gear for gear in catalog.gear if gear.gear_id == bait_id)
    rng = Random(f"fishing-ev-cross-check:{rod_id}:{bait_id}")
    casts = 60_000
    total_value = 0
    for _ in range(casts):
        total_value += roll_catch(
            rng=rng,
            grade_configs=catalog.grades,
            species=catalog.species,
            rod=rod,
            bait=bait,
            max_value=FISHING_MAX_SINGLE_CATCH,
        ).value
    analytic = _expected_catch_value(catalog=catalog, rod=rod, bait=bait)
    assert total_value / casts == pytest.approx(analytic, rel=0.1)


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


def test_size_rank_reports_where_the_catch_sits_in_its_own_band() -> None:
    """A rolled size is reported as a position inside the species' band, not a raw x."""
    catalog = build_default_catalog()
    rng = Random(11)
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
        span = species.size_max_bps - species.size_min_bps
        expected = (roll.size_bps - species.size_min_bps) * FISHING_BPS_DENOMINATOR // span
        assert roll.size_rank_bps == expected
        assert 0 <= roll.size_rank_bps <= FISHING_BPS_DENOMINATOR


def test_size_rank_of_a_fixed_size_species_is_the_bottom_of_its_band() -> None:
    """A species with no size spread reports 0 rather than marking every catch a big one."""
    grades = (
        FishGradeConfigView(
            grade=FishGrade.N, weight=1, color=0, emoji="⚪", label="普通", order_index=0
        ),
    )
    roll = roll_catch(
        rng=Random(2),
        grade_configs=grades,
        species=(_fish(species_id="fixed", grade=FishGrade.N),),
        rod=_rod(),
        bait=_bait(),
        max_value=100_000,
    )
    assert roll.size_rank_bps == 0


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
    """Builds a fixed-size, fixed-value species for a hand-written catalog.

    A single-point size band and a base value of 1 keep the value arithmetic out of the way, so a
    test about grade selection or the empty-grade fallback asserts on nothing else.

    Returns:
        A species row whose only interesting field is its grade.
    """
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
