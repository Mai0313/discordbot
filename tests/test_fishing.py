"""Tests for the fishing mini-game: pure rules, EV sink, persistence, and views."""

# ruff: noqa: S311 -- seeded Random() in tests is for determinism, not cryptography

from types import SimpleNamespace
from random import Random
from typing import Any

import pytest
from nextcord import Embed

from discordbot.cogs._games import fishing_views
from discordbot.typings.fishing import (
    ROD_TIERS,
    BAIT_TYPES,
    ROD_BY_KEY,
    BAIT_BY_KEY,
    FISH_CATALOG,
    RARITY_COLOR,
    RARITY_ORDER,
    SPECIES_BY_RARITY,
    DexEntry,
    CastResult,
    SellResult,
    CastOutcome,
    LoadoutView,
    InventoryEntry,
    BiggestCatchRow,
    FishingLeaderboardRow,
)
from discordbot.cogs._games.fishing import (
    cast_fish,
    roll_size,
    sell_value,
    per_cast_ev,
    roll_rarity,
    loadout_cost,
    select_species,
    expected_value_for_rarity,
)
from discordbot.cogs._games.prompts import fishing_catch_fallback_line
from discordbot.cogs._economy.database import get_account, get_balance, credit_with_repayment
from discordbot.cogs._games.fishing_views import (
    FishingContext,
    FishingDexView,
    FishingShopView,
    BaitQuantityModal,
    FishingStationView,
    FishingPostCastView,
    FishingInventoryView,
    FishingResultNavView,
    FishingBaitSelectView,
    FishingLeaderboardView,
    build_dex_embeds,
    build_shop_embeds,
    open_fishing_panel,
    build_station_embeds,
    edit_fishing_message,
    _refresh_catch_banter,
    build_inventory_embeds,
    build_cast_result_embeds,
    build_leaderboard_embeds,
    build_sell_result_embeds,
)
from discordbot.cogs._games.fishing_database import (
    buy_rod,
    get_dex,
    buy_bait,
    grant_rod,
    sell_fish,
    grant_bait,
    reset_user,
    get_loadout,
    execute_cast,
    list_inventory,
    leaderboard_total_earned,
    leaderboard_biggest_catch,
)

_USER_ID = 1
_USER_NAME = "user1"


# --- deterministic rng doubles ----------------------------------------------


class _AlwaysCatchRandom(Random):
    """Forces a catch: max roll dodges the miss and picks the rarest tier and largest size."""

    def randint(self, a: int, b: int) -> int:
        return b

    def choice(self, seq: Any) -> Any:  # noqa: ANN401 -- mirrors random.Random.choice signature
        return seq[-1]


class _AlwaysMissRandom(Random):
    """Forces a 空竿: the zero roll is always below the miss threshold."""

    def randint(self, a: int, b: int) -> int:
        return a

    def choice(self, seq: Any) -> Any:  # noqa: ANN401 -- mirrors random.Random.choice signature
        return seq[0]


# --- interaction / message stubs --------------------------------------------


class MessageStub:
    """Records message edits."""

    def __init__(self) -> None:
        self.id = 999
        self.edits: list[dict[str, Any]] = []

    async def edit(self, **kwargs: Any) -> "MessageStub":  # noqa: ANN401 -- Discord kwargs
        self.edits.append(kwargs)
        return self


class ResponseStub:
    """Records interaction response state."""

    def __init__(self) -> None:
        self.deferred = False
        self.sent: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.modals: list[Any] = []

    async def defer(self) -> None:
        self.deferred = True

    async def send_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- Discord kwargs
        self.sent.append(kwargs)

    async def edit_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- Discord kwargs
        self.edits.append(kwargs)

    async def send_modal(self, modal: Any) -> None:  # noqa: ANN401 -- modal stub
        self.modals.append(modal)

    def is_done(self) -> bool:
        return self.deferred or bool(self.sent) or bool(self.modals) or bool(self.edits)


class FollowupStub:
    """Records followup sends and returns a fresh message stub."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> MessageStub:  # noqa: ANN401 -- Discord kwargs
        self.sent.append(kwargs)
        return MessageStub()


class InteractionStub:
    """Minimal interaction stub for fishing view callbacks."""

    def __init__(self, user_id: int = _USER_ID, message: MessageStub | None = None) -> None:
        self.user = SimpleNamespace(
            id=user_id,
            name=f"user{user_id}",
            display_name=f"User {user_id}",
            display_avatar=SimpleNamespace(url=f"https://example.test/{user_id}.png"),
        )
        self.message = message
        self.response = ResponseStub()
        self.followup = FollowupStub()
        self.guild = None


class NarratorStub:
    """Narrator double whose catch_fish returns a fixed upgraded line."""

    def __init__(self, line: str = "傳說漁獲現身於釣場") -> None:
        self.line = line
        self.calls: list[dict[str, Any]] = []

    async def catch_fish(self, **kwargs: Any) -> str:  # noqa: ANN401 -- narrator kwargs
        self.calls.append(kwargs)
        return self.line


def _item(view: Any, custom_id: str) -> Any:  # noqa: ANN401 -- nextcord Item lookup
    """Returns the view child with the given custom_id."""
    for child in view.children:
        if getattr(child, "custom_id", None) == custom_id:
            return child
    raise AssertionError(f"no item {custom_id}")


def _context(rng: Random | None = None, narrator: NarratorStub | None = None) -> FishingContext:
    """Builds a fishing context for view tests."""
    return FishingContext(
        owner_id=_USER_ID,
        owner_name=_USER_NAME,
        owner_avatar_url="https://example.test/1.png",
        rng=rng if rng is not None else Random(x=0),
        narrator=narrator,
        system_name="賭場系統",
        system_avatar_url="",
    )


async def _fund(amount: int) -> None:
    """Credits the test user with starting balance."""
    await credit_with_repayment(user_id=_USER_ID, name=_USER_NAME, amount=amount)


# --- pure rules --------------------------------------------------------------


def test_per_cast_ev_is_negative_for_every_loadout() -> None:
    for rod in ROD_TIERS:
        for bait in BAIT_TYPES:
            assert per_cast_ev(rod=rod, bait=bait) < 0, (rod.key, bait.key)


def test_monte_carlo_mean_net_is_negative_for_every_loadout() -> None:
    for rod in ROD_TIERS:
        for bait in BAIT_TYPES:
            rng = Random(x=2024)
            casts = 40000
            total = sum(cast_fish(rng=rng, rod=rod, bait=bait).sell_value for _ in range(casts))
            mean_net = total / casts - loadout_cost(rod=rod, bait=bait)
            assert mean_net < 0, (rod.key, bait.key, mean_net)


def test_roll_rarity_miss_and_catch() -> None:
    rod = ROD_BY_KEY["bamboo"]
    bait = BAIT_BY_KEY["worm"]
    assert roll_rarity(rng=_AlwaysMissRandom(x=0), rod=rod, bait=bait) is None
    assert roll_rarity(rng=_AlwaysCatchRandom(x=0), rod=rod, bait=bait) == "UR"


def test_roll_rarity_distribution_orders_by_commonness() -> None:
    rod = ROD_BY_KEY["bamboo"]
    bait = BAIT_BY_KEY["worm"]
    rng = Random(x=7)
    counts = dict.fromkeys(RARITY_ORDER, 0)
    for _ in range(20000):
        rarity = roll_rarity(rng=rng, rod=rod, bait=bait)
        if rarity is not None:
            counts[rarity] += 1
    assert counts["N"] > counts["R"] > counts["SR"]
    assert counts["UR"] < counts["SSR"]


def test_select_species_returns_only_that_rarity() -> None:
    rng = Random(x=3)
    for rarity in RARITY_ORDER:
        for _ in range(20):
            assert select_species(rng=rng, rarity=rarity).rarity == rarity


def test_roll_size_within_range_and_sell_value_monotonic() -> None:
    rng = Random(x=11)
    species = SPECIES_BY_RARITY["SSR"][0]
    for _ in range(50):
        size = roll_size(rng=rng, species=species)
        assert species.min_mm <= size <= species.max_mm
    assert sell_value(species=species, size_mm=species.min_mm) == species.base_value
    assert sell_value(species=species, size_mm=species.max_mm) > species.base_value


def test_cast_fish_miss_and_catch_outcomes() -> None:
    rod = ROD_BY_KEY["legend"]
    bait = BAIT_BY_KEY["lure"]
    miss = cast_fish(rng=_AlwaysMissRandom(x=0), rod=rod, bait=bait)
    assert miss.miss is True
    assert miss.rarity is None
    catch = cast_fish(rng=_AlwaysCatchRandom(x=0), rod=rod, bait=bait)
    assert catch.miss is False
    assert catch.species is not None
    assert catch.sell_value > 0


def test_expected_value_for_rarity_increases_with_rarity() -> None:
    values = [expected_value_for_rarity(rarity=rarity) for rarity in RARITY_ORDER]
    assert values == sorted(values)


# --- database: purchases -----------------------------------------------------


async def test_buy_rod_success_debits_and_equips(fishing_isolated_db: None) -> None:
    await _fund(amount=10000)
    result = await buy_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="bamboo")
    assert result.status == "ok"
    assert result.cost == ROD_BY_KEY["bamboo"].cost
    loadout = await get_loadout(user_id=_USER_ID)
    assert loadout.rod_key == "bamboo"
    assert loadout.rod_durability == ROD_BY_KEY["bamboo"].durability
    account = await get_account(user_id=_USER_ID)
    assert account is not None
    assert account.total_spent == ROD_BY_KEY["bamboo"].cost
    assert account.balance == account.total_earned - account.total_spent


async def test_buy_rod_insufficient_grants_nothing(fishing_isolated_db: None) -> None:
    result = await buy_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="legend")
    assert result.status == "insufficient"
    loadout = await get_loadout(user_id=_USER_ID)
    assert loadout.rod_key == ""
    assert await get_balance(user_id=_USER_ID) == 0


async def test_buy_unknown_rod_and_bait(fishing_isolated_db: None) -> None:
    assert (
        await buy_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="nope")
    ).status == "unknown_item"
    assert (
        await buy_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="nope", quantity=1)
    ).status == "unknown_item"
    assert (
        await buy_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", quantity=0)
    ).status == "invalid_quantity"


async def test_buy_bait_increments_quantity(fishing_isolated_db: None) -> None:
    await _fund(amount=10000)
    await buy_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", quantity=5)
    await buy_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", quantity=3)
    loadout = await get_loadout(user_id=_USER_ID)
    assert loadout.baits["worm"] == 8


# --- database: casting -------------------------------------------------------


async def test_execute_cast_without_rod_or_bait(fishing_isolated_db: None) -> None:
    assert (
        await execute_cast(
            user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", rng=Random(x=0)
        )
    ).status == "no_rod"
    await grant_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="bamboo")
    assert (
        await execute_cast(
            user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", rng=Random(x=0)
        )
    ).status == "no_bait"


async def test_execute_cast_catch_logs_and_consumes(fishing_isolated_db: None) -> None:
    await grant_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="bamboo")
    await grant_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", quantity=3)
    result = await execute_cast(
        user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", rng=_AlwaysCatchRandom(x=0)
    )
    assert result.status == "ok"
    assert result.outcome is not None
    assert result.outcome.miss is False
    assert result.catch_id is not None
    assert result.rod_durability_after == ROD_BY_KEY["bamboo"].durability - 1
    assert result.bait_remaining == 2
    inventory = await list_inventory(user_id=_USER_ID)
    assert len(inventory) == 1


async def test_execute_cast_miss_logs_nothing(fishing_isolated_db: None) -> None:
    await grant_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="bamboo")
    await grant_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", quantity=3)
    result = await execute_cast(
        user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", rng=_AlwaysMissRandom(x=0)
    )
    assert result.status == "ok"
    assert result.outcome is not None
    assert result.outcome.miss is True
    assert result.catch_id is None
    assert await list_inventory(user_id=_USER_ID) == ()


async def test_rod_breaks_at_zero_durability(fishing_isolated_db: None) -> None:
    await grant_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="bamboo")
    await grant_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", quantity=12)
    durability = ROD_BY_KEY["bamboo"].durability
    last = None
    for _ in range(durability):
        last = await execute_cast(
            user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", rng=_AlwaysMissRandom(x=0)
        )
    assert last is not None
    assert last.rod_broke is True
    assert last.rod_key == ""
    after = await execute_cast(
        user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", rng=_AlwaysMissRandom(x=0)
    )
    assert after.status == "no_rod"


# --- database: selling, dex, leaderboards ------------------------------------


async def _stock_two_catches() -> None:
    """Grants gear and lands two guaranteed catches for the test user."""
    await grant_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="legend")
    await grant_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="lure", quantity=5)
    for _ in range(2):
        await execute_cast(
            user_id=_USER_ID, user_name=_USER_NAME, bait_key="lure", rng=_AlwaysCatchRandom(x=0)
        )


async def test_sell_fish_credits_and_keeps_dex(fishing_isolated_db: None) -> None:
    await _stock_two_catches()
    inventory = await list_inventory(user_id=_USER_ID)
    expected = sum(entry.sell_value for entry in inventory)
    before = await get_balance(user_id=_USER_ID)
    result = await sell_fish(user_id=_USER_ID, user_name=_USER_NAME)
    assert result.status == "ok"
    assert result.sold_count == 2
    assert result.earned == expected
    assert await get_balance(user_id=_USER_ID) == before + expected
    assert await list_inventory(user_id=_USER_ID) == ()
    # Selling again finds nothing.
    assert (await sell_fish(user_id=_USER_ID, user_name=_USER_NAME)).status == "nothing"
    # Dex still records the caught species after selling.
    dex = {entry.species_key: entry for entry in await get_dex(user_id=_USER_ID)}
    caught = [entry for entry in dex.values() if entry.caught]
    assert caught
    account = await get_account(user_id=_USER_ID)
    assert account is not None
    assert account.balance == account.total_earned - account.total_spent


async def test_sell_one_fish_by_id(fishing_isolated_db: None) -> None:
    await _stock_two_catches()
    inventory = await list_inventory(user_id=_USER_ID)
    target = inventory[0]
    result = await sell_fish(user_id=_USER_ID, user_name=_USER_NAME, catch_ids=[target.catch_id])
    assert result.status == "ok"
    assert result.sold_count == 1
    remaining = await list_inventory(user_id=_USER_ID)
    assert len(remaining) == 1
    assert remaining[0].catch_id != target.catch_id


async def test_dex_and_leaderboards(fishing_isolated_db: None) -> None:
    await _stock_two_catches()
    dex = {entry.species_key: entry for entry in await get_dex(user_id=_USER_ID)}
    caught_entries = [entry for entry in dex.values() if entry.caught]
    assert len(caught_entries) == 1  # AlwaysCatch always lands the same UR species
    assert caught_entries[0].count == 2
    await sell_fish(user_id=_USER_ID, user_name=_USER_NAME)
    earned_board = await leaderboard_total_earned()
    assert earned_board
    assert earned_board[0].user_id == _USER_ID
    biggest_board = await leaderboard_biggest_catch()
    assert biggest_board
    assert biggest_board[0].user_id == _USER_ID
    assert biggest_board[0].rarity == "UR"


async def test_reset_user_clears_state(fishing_isolated_db: None) -> None:
    await _stock_two_catches()
    await reset_user(user_id=_USER_ID)
    loadout = await get_loadout(user_id=_USER_ID)
    assert loadout.rod_key == ""
    assert loadout.baits == {}
    assert await list_inventory(user_id=_USER_ID) == ()


# --- embed builders ----------------------------------------------------------


def test_build_station_embeds_with_and_without_gear() -> None:
    empty = build_station_embeds(loadout=_loadout(rod_key="", durability=0, baits={}))
    assert "尚未持有" in (empty[0].description or "")
    geared = build_station_embeds(
        loadout=_loadout(rod_key="carbon", durability=20, baits={"worm": 4}),
        avatar_url="https://example.test/1.png",
    )
    assert "碳纖竿" in (geared[0].description or "")


def test_build_shop_embeds_lists_rods_and_baits() -> None:
    embeds = build_shop_embeds(loadout=_loadout(rod_key="", durability=0, baits={}))
    field_text = "\n".join(str(field.value) for field in embeds[0].fields)
    assert "竹竿" in field_text
    assert "擬餌糖" in field_text


def test_build_cast_result_embeds_catch_and_miss() -> None:
    rod = ROD_BY_KEY["carbon"]
    species = SPECIES_BY_RARITY["SSR"][0]
    catch = build_cast_result_embeds(
        result=_cast_result(
            outcome=CastOutcome(miss=False, species=species, size_mm=600, sell_value=1400),
            durability_after=20,
            rod_broke=False,
        ),
        rod=rod,
        talk_line="banter",
        system_name="賭場系統",
    )
    assert species.name in (catch[0].title or "")
    assert catch[0].color is not None
    assert catch[0].color.value == RARITY_COLOR["SSR"]
    assert len(catch) == 2  # result + talk embed
    miss = build_cast_result_embeds(
        result=_cast_result(outcome=CastOutcome(miss=True), durability_after=0, rod_broke=True),
        rod=rod,
        talk_line="",
        system_name="賭場系統",
    )
    assert "空竿" in (miss[0].title or "")
    assert "斷了" in "\n".join(str(field.value) for field in miss[0].fields)
    assert len(miss) == 1


def test_build_inventory_and_sell_embeds() -> None:
    assert "空空" in (build_inventory_embeds(entries=())[0].description or "")
    entries = (_inventory_entry(catch_id=1), _inventory_entry(catch_id=2))
    assert build_inventory_embeds(entries=entries)[0].description
    sold = build_sell_result_embeds(
        result=SellResult(status="ok", sold_count=2, earned=50, new_balance=150)
    )
    assert "賣出" in (sold[0].description or "")
    nothing = build_sell_result_embeds(result=SellResult(status="nothing"))
    assert "沒有" in (nothing[0].description or "")


def test_build_dex_and_leaderboard_embeds() -> None:
    dex_entries = tuple(
        DexEntry(species_key=species.key, caught=(species.rarity == "N"), count=1, biggest_mm=120)
        for species in FISH_CATALOG
    )
    dex = build_dex_embeds(entries=dex_entries)
    assert "完成度" in (dex[0].description or "")
    earned = build_leaderboard_embeds(
        metric="earned", earned_rows=(FishingLeaderboardRow(user_id=1, user_name="a", value=999),)
    )
    assert "a" in (earned[0].description or "")
    biggest = build_leaderboard_embeds(
        metric="biggest",
        biggest_rows=(
            BiggestCatchRow(
                user_id=1, user_name="a", species_key="dragon", rarity="UR", size_mm=1500
            ),
        ),
    )
    assert "1500mm" in (biggest[0].description or "")
    assert "還沒有人" in (build_leaderboard_embeds(metric="earned")[0].description or "")


def test_durability_bar_bounds() -> None:
    assert fishing_views._durability_bar(remaining=0, total=10) == "⬜" * 10
    assert fishing_views._durability_bar(remaining=10, total=10) == "🟩" * 10
    assert fishing_views._durability_bar(remaining=1, total=100).startswith("🟩")


def test_fishing_catch_fallback_line() -> None:
    assert fishing_catch_fallback_line(rng=Random(x=0), rarity=None)
    assert fishing_catch_fallback_line(rng=Random(x=0), rarity="UR")


# --- views -------------------------------------------------------------------


async def test_interaction_check_is_owner_only() -> None:
    view = FishingStationView(ctx=_context())
    assert await view.interaction_check(interaction=InteractionStub(user_id=_USER_ID)) is True
    intruder = InteractionStub(user_id=2)
    assert await view.interaction_check(interaction=intruder) is False
    assert intruder.response.sent


async def test_edit_fishing_message_uses_response_edit_when_not_done() -> None:
    interaction = InteractionStub(message=MessageStub())
    await edit_fishing_message(interaction=interaction, embeds=[Embed(title="x")], view=None)
    assert interaction.response.edits


async def test_open_fishing_panel_sends_and_tracks(
    fishing_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked: list[Any] = []

    async def fake_track(message: Any, user_name: str | None = None) -> None:  # noqa: ANN401 -- stub
        tracked.append(message)

    monkeypatch.setattr(fishing_views, "track_public_message", fake_track)
    interaction = InteractionStub()
    await open_fishing_panel(interaction=interaction, ctx=_context())
    assert interaction.followup.sent
    assert tracked


async def test_station_cast_without_gear_notifies(fishing_isolated_db: None) -> None:
    view = FishingStationView(ctx=_context())
    view.bind_message(message=MessageStub())
    interaction = InteractionStub(message=view.message)
    await _item(view, "fishing:cast").callback(interaction)
    assert interaction.followup.sent  # ephemeral "no rod" notice


async def test_station_navigation_buttons(fishing_isolated_db: None) -> None:
    for custom_id in (
        "fishing:shop",
        "fishing:bag",
        "fishing:dex",
        "fishing:board",
        "fishing:refresh",
    ):
        view = FishingStationView(ctx=_context())
        message = MessageStub()
        view.bind_message(message=message)
        await _item(view, custom_id).callback(InteractionStub(message=message))
        assert message.edits


async def test_full_cast_flow_lands_a_catch(
    fishing_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fishing_views.asyncio, "sleep", _no_sleep)
    await grant_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="legend")
    await grant_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="lure", quantity=5)
    view = FishingStationView(ctx=_context(rng=_AlwaysCatchRandom(x=0)))
    message = MessageStub()
    view.bind_message(message=message)
    await _item(view, "fishing:cast").callback(InteractionStub(message=message))
    assert message.edits  # animation beats + result
    assert len(await list_inventory(user_id=_USER_ID)) == 1


async def test_cast_flow_with_multiple_baits_shows_picker(fishing_isolated_db: None) -> None:
    await grant_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="legend")
    await grant_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="worm", quantity=2)
    await grant_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="lure", quantity=2)
    view = FishingStationView(ctx=_context())
    message = MessageStub()
    view.bind_message(message=message)
    await _item(view, "fishing:cast").callback(InteractionStub(message=message))
    assert message.edits  # edited to the bait picker


async def test_bait_select_view_runs_cast(
    fishing_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fishing_views.asyncio, "sleep", _no_sleep)
    await grant_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="legend")
    await grant_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="lure", quantity=2)
    view = FishingBaitSelectView(
        ctx=_context(rng=_AlwaysCatchRandom(x=0)), rod=ROD_BY_KEY["legend"], owned={"lure": 2}
    )
    message = MessageStub()
    view.bind_message(message=message)
    select = _item(view, "fishing:cast:bait")
    select._selected_values = ["lure"]
    await select.callback(InteractionStub(message=message))
    assert len(await list_inventory(user_id=_USER_ID)) == 1
    back_view = FishingBaitSelectView(ctx=_context(), rod=ROD_BY_KEY["legend"], owned={"lure": 1})
    back_message = MessageStub()
    back_view.bind_message(message=back_message)
    await _item(back_view, "fishing:cast:back").callback(InteractionStub(message=back_message))
    assert back_message.edits


async def test_shop_rod_purchase_paths(fishing_isolated_db: None) -> None:
    await _fund(amount=10000)
    view = FishingShopView(ctx=_context())
    message = MessageStub()
    view.bind_message(message=message)
    rod_select = _item(view, "fishing:shop:rod")
    rod_select._selected_values = ["bamboo"]
    await rod_select.callback(InteractionStub(message=message))
    assert (await get_loadout(user_id=_USER_ID)).rod_key == "bamboo"
    # Insufficient path.
    poor_view = FishingShopView(ctx=_context())
    poor_message = MessageStub()
    poor_view.bind_message(message=poor_message)
    poor_select = _item(poor_view, "fishing:shop:rod")
    poor_select._selected_values = ["legend"]
    interaction = InteractionStub(message=poor_message)
    await poor_select.callback(interaction)
    assert interaction.followup.sent  # insufficient notice


async def test_shop_bait_select_opens_modal(fishing_isolated_db: None) -> None:
    view = FishingShopView(ctx=_context())
    message = MessageStub()
    view.bind_message(message=message)
    bait_select = _item(view, "fishing:shop:bait")
    bait_select._selected_values = ["worm"]
    interaction = InteractionStub(message=message)
    await bait_select.callback(interaction)
    assert interaction.response.modals


async def test_shop_back_button(fishing_isolated_db: None) -> None:
    view = FishingShopView(ctx=_context())
    message = MessageStub()
    view.bind_message(message=message)
    await _item(view, "fishing:shop:back").callback(InteractionStub(message=message))
    assert message.edits


async def test_bait_quantity_modal_paths(fishing_isolated_db: None) -> None:
    await _fund(amount=10000)
    message = MessageStub()
    # Bad quantity.
    bad = BaitQuantityModal(ctx=_context(), bait_key="worm", message=message)
    bad.quantity = SimpleNamespace(value="abc")
    interaction = InteractionStub(message=message)
    await bad.callback(interaction)
    assert interaction.followup.sent
    # Out of range.
    big = BaitQuantityModal(ctx=_context(), bait_key="worm", message=message)
    big.quantity = SimpleNamespace(value="9999")
    interaction = InteractionStub(message=message)
    await big.callback(interaction)
    assert interaction.followup.sent
    # Success.
    ok = BaitQuantityModal(ctx=_context(), bait_key="worm", message=message)
    ok.quantity = SimpleNamespace(value="5")
    await ok.callback(InteractionStub(message=message))
    assert (await get_loadout(user_id=_USER_ID)).baits["worm"] == 5


async def test_inventory_view_sell_all_and_one(fishing_isolated_db: None) -> None:
    await _stock_two_catches()
    entries = await list_inventory(user_id=_USER_ID)
    # Sell one.
    inv_view = FishingInventoryView(ctx=_context(), entries=entries)
    message = MessageStub()
    inv_view.bind_message(message=message)
    select = _item(inv_view, "fishing:bag:sellone")
    select._selected_values = [str(entries[0].catch_id)]
    await select.callback(InteractionStub(message=message))
    assert len(await list_inventory(user_id=_USER_ID)) == 1
    # Sell all.
    remaining = await list_inventory(user_id=_USER_ID)
    sell_view = FishingInventoryView(ctx=_context(), entries=remaining)
    sell_message = MessageStub()
    sell_view.bind_message(message=sell_message)
    await _item(sell_view, "fishing:bag:sellall").callback(InteractionStub(message=sell_message))
    assert await list_inventory(user_id=_USER_ID) == ()


async def test_inventory_view_empty_has_no_sell_controls(fishing_isolated_db: None) -> None:
    view = FishingInventoryView(ctx=_context(), entries=())
    custom_ids = {getattr(child, "custom_id", None) for child in view.children}
    assert "fishing:bag:sellall" not in custom_ids
    assert "fishing:bag:back" in custom_ids


async def test_post_cast_again_and_nav(
    fishing_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fishing_views.asyncio, "sleep", _no_sleep)
    await grant_rod(user_id=_USER_ID, user_name=_USER_NAME, rod_key="legend")
    await grant_bait(user_id=_USER_ID, user_name=_USER_NAME, bait_key="lure", quantity=5)
    view = FishingPostCastView(ctx=_context(rng=_AlwaysCatchRandom(x=0)), last_bait_key="lure")
    message = MessageStub()
    view.bind_message(message=message)
    await _item(view, "fishing:post:again").callback(InteractionStub(message=message))
    assert len(await list_inventory(user_id=_USER_ID)) == 1
    for custom_id in ("fishing:post:bag", "fishing:post:back"):
        nav = FishingPostCastView(ctx=_context(), last_bait_key="lure")
        nav_message = MessageStub()
        nav.bind_message(message=nav_message)
        await _item(nav, custom_id).callback(InteractionStub(message=nav_message))
        assert nav_message.edits


async def test_post_cast_again_without_gear_notifies(fishing_isolated_db: None) -> None:
    view = FishingPostCastView(ctx=_context(), last_bait_key="lure")
    message = MessageStub()
    view.bind_message(message=message)
    interaction = InteractionStub(message=message)
    await _item(view, "fishing:post:again").callback(interaction)
    assert interaction.followup.sent


async def test_result_nav_and_dex_and_leaderboard_views(fishing_isolated_db: None) -> None:
    for view, custom_id in (
        (FishingResultNavView(ctx=_context()), "fishing:sold:bag"),
        (FishingResultNavView(ctx=_context()), "fishing:sold:back"),
        (FishingDexView(ctx=_context()), "fishing:dex:back"),
        (FishingLeaderboardView(ctx=_context(), metric="earned"), "fishing:lb:earned"),
        (FishingLeaderboardView(ctx=_context(), metric="earned"), "fishing:lb:biggest"),
        (FishingLeaderboardView(ctx=_context(), metric="earned"), "fishing:lb:back"),
    ):
        message = MessageStub()
        view.bind_message(message=message)
        await _item(view, custom_id).callback(InteractionStub(message=message))
        assert message.edits


async def test_refresh_catch_banter_upgrades_line() -> None:
    species = SPECIES_BY_RARITY["UR"][0]
    result = _cast_result(
        outcome=CastOutcome(miss=False, species=species, size_mm=1500, sell_value=12000),
        durability_after=59,
        rod_broke=False,
    )
    narrator = NarratorStub(line="新的旁白台詞")
    ctx = _context(narrator=narrator)
    post_view = FishingPostCastView(ctx=ctx, last_bait_key="lure")
    message = MessageStub()
    post_view.bind_message(message=message)
    await _refresh_catch_banter(
        message=message,
        ctx=ctx,
        result=result,
        rod=ROD_BY_KEY["legend"],
        post_view=post_view,
        fallback_line="舊台詞",
    )
    assert narrator.calls
    assert message.edits


async def test_refresh_catch_banter_skips_when_finished() -> None:
    species = SPECIES_BY_RARITY["UR"][0]
    result = _cast_result(
        outcome=CastOutcome(miss=False, species=species, size_mm=1500, sell_value=12000),
        durability_after=59,
        rod_broke=False,
    )
    ctx = _context(narrator=NarratorStub(line="新的旁白台詞"))
    post_view = FishingPostCastView(ctx=ctx, last_bait_key="lure")
    post_view.stop()
    message = MessageStub()
    await _refresh_catch_banter(
        message=message,
        ctx=ctx,
        result=result,
        rod=ROD_BY_KEY["legend"],
        post_view=post_view,
        fallback_line="舊台詞",
    )
    assert message.edits == []


# --- local builders for embed tests -----------------------------------------


async def _no_sleep(_seconds: float) -> None:
    """Replaces asyncio.sleep during animation tests."""
    return


def _loadout(rod_key: str, durability: int, baits: dict[str, int]) -> LoadoutView:
    """Builds a LoadoutView for embed tests."""
    return LoadoutView(
        user_id=_USER_ID, balance=5000, rod_key=rod_key, rod_durability=durability, baits=baits
    )


def _cast_result(outcome: CastOutcome, durability_after: int, rod_broke: bool) -> CastResult:
    """Builds a CastResult for embed tests."""
    return CastResult(
        status="ok",
        outcome=outcome,
        rod_key="" if rod_broke else "legend",
        rod_durability_after=durability_after,
        rod_broke=rod_broke,
        bait_key="lure",
        bait_remaining=3,
        catch_id=1 if not outcome.miss else None,
    )


def _inventory_entry(catch_id: int) -> InventoryEntry:
    """Builds an InventoryEntry for embed tests."""
    return InventoryEntry(
        catch_id=catch_id, species_key="goldfish", rarity="SSR", size_mm=600, sell_value=1400
    )
