"""Guards over what the offline fishing catalog seeder does to a `games.db`.

Pins `scripts/seed_fishing.py`, the only thing that carries the hand-tuned catalog in
`cogs/games/fishing/defaults.py` into a deployed database. Grades, species, and gear are seeded
offline and runtime never writes them, so retuning a default changes nothing in a running bot
until an operator runs that script; that is what makes the properties below worth holding.

A first run over an empty database creates every row of all three tables, and a second run writes
nothing, so a run interrupted part way is safe to repeat. A stored row that drifted from the
defaults is reported field by field as `field old -> new` and then rewritten, which is what the
operator reads before deciding to apply anything. `--dry-run` produces that same diff with the
writes withheld, so the preview can never become a separate, weaker code path than the apply. And
a row an operator added beyond the defaults survives a run untouched: this seeds, it does not
reconcile, and eating someone's hand-added gear on a routine reseed would be the worst outcome
the script could produce.

Every test runs under `fishing_isolated_db`, which deliberately leaves the catalog unseeded, so
the empty-database case is the starting state rather than something a test has to arrange.
"""

import pytest
from scripts import seed_fishing as seed_fishing_script

from discordbot.typings.fishing import GearType, GearUpsert
from discordbot.cogs.games.fishing.database import list_gear, upsert_gear, list_grade_configs
from discordbot.cogs.games.fishing.defaults import build_default_catalog

pytestmark = pytest.mark.usefixtures("fishing_isolated_db")


async def test_seeding_an_empty_database_creates_the_whole_catalog() -> None:
    """A first run writes every grade, species, and gear row."""
    catalog = build_default_catalog()
    expected = len(catalog.grades) + len(catalog.species) + len(catalog.gear)

    summary = await seed_fishing_script.seed_fishing_catalog()

    assert len(summary.changes) == expected
    assert summary.unchanged == 0
    assert all(change.created for change in summary.changes)
    assert {gear.gear_id for gear in await list_gear()} == {gear.gear_id for gear in catalog.gear}
    assert len(await list_grade_configs()) == len(catalog.grades)


async def test_seeding_twice_is_a_no_op() -> None:
    """Re-running against an already-current catalog writes nothing."""
    await seed_fishing_script.seed_fishing_catalog()

    summary = await seed_fishing_script.seed_fishing_catalog()

    assert summary.changes == ()
    assert summary.unchanged > 0


async def test_a_retuned_row_is_reported_field_by_field_and_rewritten() -> None:
    """A stored row that drifted from the defaults is named, diffed, and corrected."""
    await seed_fishing_script.seed_fishing_catalog()
    default_lure = next(
        gear for gear in build_default_catalog().gear if gear.gear_id == "bait_lure"
    )
    await upsert_gear(
        gear=GearUpsert(
            gear_id=default_lure.gear_id,
            gear_type=default_lure.gear_type,
            name=default_lure.name,
            emoji=default_lure.emoji,
            tier=default_lure.tier,
            price=200,
            rarity_shift_bps=default_lure.rarity_shift_bps,
            durability=default_lure.durability,
            value_bonus_bps=1_500,
        )
    )

    summary = await seed_fishing_script.seed_fishing_catalog()

    assert [change.row_id for change in summary.changes] == ["bait_lure"]
    change = summary.changes[0]
    assert change.created is False
    assert set(change.changes) == {
        f"price 200 -> {default_lure.price}",
        f"value_bonus_bps 1500 -> {default_lure.value_bonus_bps}",
    }
    stored = next(gear for gear in await list_gear() if gear.gear_id == "bait_lure")
    assert stored.price == default_lure.price
    assert stored.value_bonus_bps == default_lure.value_bonus_bps


async def test_a_dry_run_reports_the_same_changes_without_writing() -> None:
    """The dry run is the real diff, just with the writes withheld."""
    summary = await seed_fishing_script.seed_fishing_catalog(dry_run=True)

    assert summary.dry_run is True
    assert summary.changes != ()
    assert await list_gear() == ()


async def test_seeding_leaves_operator_added_gear_alone() -> None:
    """The seeder writes the defaults; it does not reconcile rows an operator added."""
    await upsert_gear(
        gear=GearUpsert(
            gear_id="rod_custom",
            gear_type=GearType.ROD,
            name="自訂竿",
            emoji="🎣",
            tier=9,
            price=1,
            rarity_shift_bps=0,
            durability=1,
            value_bonus_bps=0,
        )
    )

    await seed_fishing_script.seed_fishing_catalog()

    assert any(gear.gear_id == "rod_custom" for gear in await list_gear())
