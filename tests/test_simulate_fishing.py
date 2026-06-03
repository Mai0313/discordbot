"""Tests for the fishing Monte Carlo simulator (net-deflation regression)."""

import pytest
from scripts import simulate_fishing

from discordbot.typings.fishing import GearType
from discordbot.cogs._fishing.defaults import build_default_catalog


def test_cheapest_combo_is_net_deflationary() -> None:
    """The cheapest rod+bait combo loses currency per cast on average."""
    catalog = build_default_catalog()
    rod = next(gear for gear in catalog.gear if gear.gear_id == "rod_bamboo")
    bait = next(gear for gear in catalog.gear if gear.gear_id == "bait_worm")
    ev, _counts, _capped = simulate_fishing._simulate_combo(
        catalog=catalog, rod=rod, bait=bait, casts=50_000, seed=0
    )
    assert ev < simulate_fishing._cost_per_cast(rod=rod, bait=bait)


def test_every_combo_is_net_deflationary() -> None:
    """Every rod+bait combo is a net sink, not a faucet."""
    catalog = build_default_catalog()
    rods = [gear for gear in catalog.gear if gear.gear_type == GearType.ROD]
    baits = [gear for gear in catalog.gear if gear.gear_type == GearType.BAIT]
    for rod in rods:
        for bait in baits:
            ev, _counts, _capped = simulate_fishing._simulate_combo(
                catalog=catalog, rod=rod, bait=bait, casts=30_000, seed=1
            )
            assert ev < simulate_fishing._cost_per_cast(rod=rod, bait=bait)


def test_grade_hit_rate_matches_theory() -> None:
    """Observed grade hit rates track the configured weights at zero shift."""
    catalog = build_default_catalog()
    rod = next(gear for gear in catalog.gear if gear.gear_id == "rod_bamboo")
    bait = next(gear for gear in catalog.gear if gear.gear_id == "bait_worm")
    _ev, counts, _capped = simulate_fishing._simulate_combo(
        catalog=catalog, rod=rod, bait=bait, casts=100_000, seed=0
    )
    total = sum(counts.values())
    total_weight = sum(grade.weight for grade in catalog.grades)
    for grade in catalog.grades:
        observed = counts[grade.grade] / total
        expected = grade.weight / total_weight
        assert observed == pytest.approx(expected, abs=0.01)


def test_report_runs_without_error() -> None:
    """The end-to-end report renders for a small run."""
    simulate_fishing._report(
        args=simulate_fishing._parse_args(argv=["--casts", "2000", "--seed", "0"])
    )
