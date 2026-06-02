"""Tests for the fishing database and settlement layer."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from random import Random
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from discordbot.cogs._fishing import database as fdb
from discordbot.typings.fishing import GearType, CastStatus, GearUpsert
from discordbot.cogs._economy.database import get_balance, adjust_balance
from discordbot.cogs._fishing.database import (
    CatchLog,
    settle_cast,
    purchase_gear,
    get_angler_state,
    fetch_top_catches,
    get_fishing_panel,
    reset_all_fishing,
    fetch_recent_catches,
    open_fishing_session,
)
from discordbot.cogs._fishing.defaults import (
    default_gear_upserts,
    default_grade_upserts,
    default_species_upserts,
)

pytestmark = pytest.mark.usefixtures("fishing_isolated_db")


async def _seed_catalog() -> None:
    """Seeds the default catalog into the isolated fishing database."""
    for grade in default_grade_upserts():
        await fdb.upsert_grade_config(config=grade)
    for species in default_species_upserts():
        await fdb.upsert_fish_species(species=species)
    for gear in default_gear_upserts():
        await fdb.upsert_gear(gear=gear)


async def _give(user_id: int, amount: int, name: str = "angler") -> None:
    """Credits a starting balance to a user."""
    await adjust_balance(user_id=user_id, name=name, delta=amount)


async def test_buy_bait_debits_and_stacks() -> None:
    """Buying bait burns price*quantity and stacks the inventory."""
    await _seed_catalog()
    await _give(user_id=1, amount=100_000)
    first = await purchase_gear(user_id=1, name="angler", gear_id="bait_worm", quantity=50)
    assert first.success is True
    assert first.total_cost == 1_500
    assert await get_balance(user_id=1) == 98_500
    await purchase_gear(user_id=1, name="angler", gear_id="bait_worm", quantity=10)
    panel = await get_fishing_panel(user_id=1)
    worm = next(stack for stack in panel.baits if stack.bait_id == "bait_worm")
    assert worm.quantity == 60


async def test_insufficient_purchase_is_rejected() -> None:
    """A purchase the user cannot afford is rejected with no gear granted."""
    await _seed_catalog()
    await _give(user_id=1, amount=100)
    result = await purchase_gear(user_id=1, name="angler", gear_id="rod_carbon", quantity=1)
    assert result.success is False
    assert result.reason == "insufficient"
    assert await get_balance(user_id=1) == 100
    angler = await get_angler_state(user_id=1)
    assert angler.rod is None


async def test_buy_rod_sets_then_replaces() -> None:
    """Buying a rod equips it; buying another replaces it and resets durability."""
    await _seed_catalog()
    await _give(user_id=1, amount=100_000)
    await purchase_gear(user_id=1, name="angler", gear_id="rod_bamboo", quantity=1)
    angler = await get_angler_state(user_id=1)
    assert angler.rod is not None
    assert angler.rod.gear_id == "rod_bamboo"
    assert angler.durability_remaining == 30
    await purchase_gear(user_id=1, name="angler", gear_id="rod_carbon", quantity=1)
    angler = await get_angler_state(user_id=1)
    assert angler.rod is not None
    assert angler.rod.gear_id == "rod_carbon"
    assert angler.durability_remaining == 80


async def test_purchase_grant_failure_refunds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A grant failure after the wallet debit refunds the full amount."""
    await _seed_catalog()
    await _give(user_id=1, amount=100_000)

    async def _boom(**_kwargs: object) -> None:
        raise RuntimeError("grant exploded")

    monkeypatch.setattr(fdb, "_grant_gear_in_session", _boom)
    result = await purchase_gear(user_id=1, name="angler", gear_id="rod_bamboo", quantity=1)
    assert result.success is False
    assert result.reason == "grant_failed"
    assert await get_balance(user_id=1) == 100_000


async def test_cast_consumes_and_credits() -> None:
    """A cast consumes one bait and one durability, logs the catch, and credits payout."""
    await _seed_catalog()
    await _give(user_id=1, amount=100_000)
    await purchase_gear(user_id=1, name="angler", gear_id="rod_bamboo", quantity=1)
    await purchase_gear(user_id=1, name="angler", gear_id="bait_worm", quantity=5)
    before = await get_balance(user_id=1)
    result = await settle_cast(user_id=1, name="angler", bait_id="bait_worm", rng=Random(7))
    assert result.status == CastStatus.SUCCESS
    assert result.roll is not None
    assert result.durability_remaining == 29
    assert result.bait_remaining == 4
    assert await get_balance(user_id=1) == before + result.payout
    angler = await get_angler_state(user_id=1)
    assert angler.total_casts == 1
    async with open_fishing_session() as session:
        logged = (await session.execute(statement=select(CatchLog))).scalars().all()
    assert len(logged) == 1


async def test_rod_breaks_then_blocks_next_cast() -> None:
    """A one-durability rod breaks on its first cast and blocks the next one."""
    await _seed_catalog()
    await fdb.upsert_gear(
        gear=GearUpsert(
            gear_id="rod_glass",
            gear_type=GearType.ROD,
            name="玻璃竿",
            emoji="🎣",
            tier=0,
            price=1,
            rarity_shift_bps=0,
            durability=1,
            value_bonus_bps=0,
        )
    )
    await _give(user_id=1, amount=100_000)
    await purchase_gear(user_id=1, name="angler", gear_id="rod_glass", quantity=1)
    await purchase_gear(user_id=1, name="angler", gear_id="bait_worm", quantity=5)
    first = await settle_cast(user_id=1, name="angler", bait_id="bait_worm", rng=Random(1))
    assert first.status == CastStatus.SUCCESS
    assert first.rod_broke is True
    assert first.durability_remaining == 0
    second = await settle_cast(user_id=1, name="angler", bait_id="bait_worm", rng=Random(1))
    assert second.status == CastStatus.BROKEN_ROD


async def test_cast_guards() -> None:
    """Casting without a rod or without the bait returns typed failures."""
    await _seed_catalog()
    await _give(user_id=1, amount=100_000)
    no_rod = await settle_cast(user_id=1, name="angler", bait_id="bait_worm", rng=Random(1))
    assert no_rod.status == CastStatus.NO_ROD
    await purchase_gear(user_id=1, name="angler", gear_id="rod_bamboo", quantity=1)
    no_bait = await settle_cast(user_id=1, name="angler", bait_id="bait_worm", rng=Random(1))
    assert no_bait.status == CastStatus.NO_BAIT
    assert await get_balance(user_id=1) == 99_700


async def test_leaderboard_is_integer_aware_descending() -> None:
    """Top catches order by numeric value, not lexicographic decimal text."""
    now = datetime.now(tz=UTC)
    async with open_fishing_session() as session:
        for index, value in enumerate([5, 100_000, 99_999, 1_000]):
            session.add(
                instance=CatchLog(
                    user_id=index,
                    user_name=f"angler{index}",
                    species_id="carp",
                    species_name="鯉魚",
                    grade="R",
                    emoji="🐠",
                    size_bps=10_000,
                    base_value=value,
                    value=value,
                    rod_id="rod_bamboo",
                    bait_id="bait_worm",
                    created_at=now,
                )
            )
        await session.commit()
    top = await fetch_top_catches(limit=10)
    assert [catch.value for catch in top] == [100_000, 99_999, 1_000, 5]


async def test_recent_catches_are_scoped_and_ordered() -> None:
    """Recent catches return only the requested user, newest first."""
    await _seed_catalog()
    await _give(user_id=1, amount=100_000)
    await _give(user_id=2, amount=100_000)
    await purchase_gear(user_id=1, name="a", gear_id="rod_bamboo", quantity=1)
    await purchase_gear(user_id=1, name="a", gear_id="bait_worm", quantity=5)
    await purchase_gear(user_id=2, name="b", gear_id="rod_bamboo", quantity=1)
    await purchase_gear(user_id=2, name="b", gear_id="bait_worm", quantity=5)
    for seed in range(3):
        await settle_cast(user_id=1, name="a", bait_id="bait_worm", rng=Random(seed))
    await settle_cast(user_id=2, name="b", bait_id="bait_worm", rng=Random(9))
    recent = await fetch_recent_catches(user_id=1, limit=10)
    assert len(recent) == 3
    assert all(catch.user_id == 1 for catch in recent)
    timestamps = [catch.created_at for catch in recent]
    assert timestamps == sorted(timestamps, reverse=True)


async def test_reset_clears_state_but_keeps_catalog() -> None:
    """The reset clears anglers, bait, and catches while leaving the catalog intact."""
    await _seed_catalog()
    await _give(user_id=1, amount=100_000)
    await purchase_gear(user_id=1, name="a", gear_id="rod_bamboo", quantity=1)
    await purchase_gear(user_id=1, name="a", gear_id="bait_worm", quantity=5)
    await settle_cast(user_id=1, name="a", bait_id="bait_worm", rng=Random(0))
    cleared = await reset_all_fishing()
    assert cleared == 1
    angler = await get_angler_state(user_id=1)
    assert angler.rod is None
    assert angler.total_casts == 0
    assert await fetch_top_catches(limit=10) == ()
    assert len(await fdb.list_fish_species()) == 10
    assert len(await fdb.list_gear()) == 6
