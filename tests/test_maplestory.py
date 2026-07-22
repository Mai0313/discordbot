"""Tests for MapleStory Artale data models, embeds, views, and commands."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Unpack, TypedDict

import pytest
from nextcord import File, Embed, Locale, Attachment, SelectOption

from discordbot.cogs import maplestory
from discordbot.cogs.maplestory import MapleStoryCogs
from discordbot.utils.discord_embeds import DEFAULT_EMBED_SPACER_FILENAME, embed_spacer_url
from discordbot.cogs._maplestory.views import _RESOLVERS, MapleDropSearchView
from discordbot.cogs._maplestory.embeds import (
    _truncate,
    create_map_embed,
    create_npc_embed,
    build_stats_embed,
    create_quest_embed,
    create_scroll_embed,
    create_monster_embed,
    create_equipment_embed,
    create_item_source_embed,
)
from discordbot.cogs._maplestory.models import (
    NPC,
    Quest,
    MapNPC,
    Scroll,
    Monster,
    Useable,
    DropItem,
    MapEntry,
    MiscItem,
    Equipment,
    QuestStep,
    StatValue,
    HuntTarget,
    MapMonster,
    RegionMaps,
    Acquisition,
    CollectItem,
    QuestReward,
    MonsterDrops,
    MonsterQuest,
    AcquisitionNPC,
    CraftingRecipe,
    EquipmentStats,
    AcquisitionQuest,
    CraftingMaterial,
    AcquisitionMonster,
    EquipmentRestriction,
)
from discordbot.cogs._maplestory.service import MapleStoryService, _load_json, _load_translations

if TYPE_CHECKING:
    from pathlib import Path

type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class InteractionPayload(TypedDict, total=False):
    """Payload captured from fake nextcord interaction sends."""

    content: str
    embed: Embed
    view: MapleDropSearchView | None
    message_id: int
    ephemeral: bool
    files: list[File]
    attachments: list[Attachment]


class _FakeResponse:
    """Minimal interaction response stub that records deferral and messages."""

    def __init__(self) -> None:
        """Initializes response state records."""
        self.deferred = False
        self.messages: list[InteractionPayload] = []

    async def defer(self) -> None:
        """Records that the interaction response was deferred."""
        self.deferred = True

    async def send_message(self, **kwargs: Unpack[InteractionPayload]) -> None:
        """Records a response message payload."""
        self.messages.append(kwargs)


class _FakeFollowup:
    """Minimal followup stub that records sends and message edits."""

    def __init__(self) -> None:
        """Initializes followup send and edit records."""
        self.sent: list[InteractionPayload] = []
        self.edited: list[InteractionPayload] = []

    async def send(
        self,
        content: str | None = None,
        embed: Embed | None = None,
        view: MapleDropSearchView | None = None,
        ephemeral: bool | None = None,
        files: list[File] | None = None,
    ) -> None:
        """Records a followup send payload."""
        payload = InteractionPayload()
        if content is not None:
            payload["content"] = content
        if embed is not None:
            payload["embed"] = embed
        if view is not None:
            payload["view"] = view
        if ephemeral is not None:
            payload["ephemeral"] = ephemeral
        if files is not None:
            payload["files"] = files
        self.sent.append(payload)

    async def edit_message(self, **kwargs: Unpack[InteractionPayload]) -> None:
        """Records a followup message edit payload."""
        self.edited.append(kwargs)


class _FakeInteraction:
    """Minimal interaction stub for MapleStory command and view tests."""

    def __init__(self, message_id: int = 777) -> None:
        """Initializes response, followup, and message identity fields."""
        self.response = _FakeResponse()
        self.followup = _FakeFollowup()
        self.message = SimpleNamespace(id=message_id)


def _write_json(path: Path, payload: JsonValue) -> None:
    """Writes a JSON fixture file."""
    path.write_text(data=json.dumps(obj=payload, ensure_ascii=False), encoding="utf-8")


def test_maplestory_region_recipe_defaults_are_isolated() -> None:
    """Region and recipe defaults are not shared between instances."""
    first_region = RegionMaps(region="Victoria")
    second_region = RegionMaps(region="Ossyria")
    first_region.maps.append("Henesys")
    assert second_region.maps == []

    first_recipe = CraftingRecipe()
    second_recipe = CraftingRecipe()
    first_recipe.materials.append(CraftingMaterial(item="Metal", quantity=1))
    assert second_recipe.materials == []


def test_maplestory_acquisition_defaults_are_isolated() -> None:
    """Acquisition list defaults are not shared between instances."""
    first_acquisition = Acquisition()
    second_acquisition = Acquisition()
    first_acquisition.monsters.append(AcquisitionMonster(name="Slime"))
    first_acquisition.npcs.append(AcquisitionNPC(name="Shop"))
    first_acquisition.quests.append(AcquisitionQuest(name="Quest"))
    first_acquisition.craftings.append(CraftingRecipe(output="Sword"))
    assert second_acquisition.monsters == []
    assert second_acquisition.npcs == []
    assert second_acquisition.quests == []
    assert second_acquisition.craftings == []


def test_maplestory_drop_monster_equipment_defaults_are_isolated() -> None:
    """Drop, monster, and equipment list defaults are isolated."""
    first_drop = DropItem(name="Sword")
    second_drop = DropItem(name="Shield")
    first_drop.jobs.append("Warrior")
    assert second_drop.jobs == []

    first_monster = Monster(name="Slime")
    second_monster = Monster(name="Pig")
    first_monster.modifiers.append("Fire")
    first_monster.quests.append(MonsterQuest(name="Helping Hand"))
    assert second_monster.modifiers == []
    assert second_monster.quests == []

    first_equipment = Equipment(name="Sword")
    second_equipment = Equipment(name="Shield")
    first_equipment.jobs.append("Warrior")
    assert second_equipment.jobs == []


def test_maplestory_scroll_npc_defaults_are_isolated() -> None:
    """Scroll dict and NPC list defaults are isolated."""
    first_scroll = Scroll(name="Scroll")
    second_scroll = Scroll(name="Other Scroll")
    first_scroll.stats["atk"] = 1
    assert second_scroll.stats == {}

    first_npc = NPC(name="Shopkeeper")
    second_npc = NPC(name="Guide")
    first_npc.quests.append(AcquisitionQuest(name="Quest"))
    first_npc.recipes.append(CraftingRecipe(output="Sword"))
    assert second_npc.quests == []
    assert second_npc.recipes == []


def test_maplestory_quest_reward_defaults_are_isolated() -> None:
    """Quest reward and quest list defaults are isolated."""
    first_reward = QuestReward()
    second_reward = QuestReward()
    first_reward.items["items"] = [CollectItem(name="Potion", quantity=1)]
    assert second_reward.items == {}

    first_quest = Quest(name="Quest")
    second_quest = Quest(name="Other Quest")
    first_quest.steps.append(QuestStep(startNPC="Guide"))
    first_quest.prerequisites.append("Prelude")
    assert second_quest.steps == []
    assert second_quest.prerequisites == []


def test_maplestory_map_defaults_are_isolated() -> None:
    """Map NPC and monster defaults are isolated."""
    first_map = MapEntry(name="Henesys")
    second_map = MapEntry(name="Kerning City")
    first_map.npcs.append(MapNPC(name="Guide"))
    first_map.monsters.append(MapMonster(name="Slime"))
    assert second_map.npcs == []
    assert second_map.monsters == []


@pytest.fixture
def maple_data_dir(tmp_path: Path) -> Path:
    """Creates a complete MapleStory data directory for integration-style tests."""
    data_dir = tmp_path / "maplestory"
    data_dir.mkdir()

    _write_json(
        path=data_dir / "monsters.json",
        payload=[
            {
                "name": "Slime",
                "nameZh": "綠水靈",
                "level": 5,
                "hp": 50,
                "mp": 10,
                "exp": 12,
                "def": {"weapon": 1, "magic": 2, "avoidability": 3},
                "accuracy": {"required": 1, "decrease": 0.1},
                "modifiers": ["Fire"],
                "regionToMapsList": [
                    {
                        "region": "Victoria",
                        "maps": [
                            "Henesys > Hunting Ground",
                            "Henesys > Hidden Street",
                            "Henesys > East Field",
                            "Henesys > West Field",
                            "Henesys > North Field",
                            "Henesys > South Field",
                        ],
                    }
                ],
                "drops": {
                    "equipmentItems": [
                        {
                            "name": "Wooden Sword",
                            "level": 10,
                            "type": "Weapon",
                            "jobs": ["Warrior"],
                        }
                    ],
                    "useableItems": [{"name": "Red Potion", "level": 0, "type": "Potion"}],
                    "scrolls": [
                        {"name": "Scroll for Gloves for ATT", "level": 0, "type": "Glove"}
                    ],
                    "miscItems": [{"name": "Slime Bubble", "level": 0, "type": "Etc"}],
                    "mesoRange": [4, 8],
                },
                "quests": [{"name": "Helping Hand", "level": 5}],
            },
            {
                "name": "Blue Slime",
                "nameZh": "藍水靈",
                "level": 7,
                "drops": {"miscItems": [{"name": "Blue Bubble"}]},
            },
        ],
    )
    _write_json(
        path=data_dir / "equipment.json",
        payload=[
            {
                "type": "Weapon",
                "name": "Wooden Sword",
                "nameZh": "木劍",
                "level": 10,
                "equipmentRestriction": {"str": 5, "dex": 2, "int": 0, "luk": 0},
                "stats": {"str": {"middle": 1}, "atk": {"middle": 12}, "upgradeSlots": 7},
                "jobs": ["Warrior"],
                "attackSpeed": "Fast",
                "tradeable": "Tradeable",
                "event": True,
                "unavailable": True,
                "acquisition": {
                    "monsters": [{"name": "Slime", "level": 5}],
                    "npcs": [{"name": "Shopkeeper", "price": 100}],
                    "quests": [{"name": "Helping Hand", "level": 5}],
                },
            },
            {"type": "Weapon", "name": "Wooden Club", "nameZh": "木棍", "level": 8},
        ],
    )
    _write_json(
        path=data_dir / "scrolls.json",
        payload=[
            {
                "name": "Scroll for Gloves for ATT",
                "nameZh": "手套攻擊卷軸",
                "type": "Glove",
                "stats": {"atk": 1, "speed": 2},
                "acquisition": {"monsters": [{"name": "Slime", "level": 5}]},
            },
            {"name": "Scroll for Shoes for Jump", "type": "Shoe"},
        ],
    )
    _write_json(
        path=data_dir / "useable.json",
        payload=[
            {
                "name": "Red Potion",
                "nameZh": "紅色藥水",
                "type": "Potion",
                "description": {"summary": "Restores HP"},
                "hp": {"amount": 50},
            }
        ],
    )
    _write_json(
        path=data_dir / "npcs.json",
        payload=[
            {
                "name": "Shopkeeper",
                "nameZh": "商店老闆",
                "type": "Shop",
                "regionToMapsList": [{"region": "Victoria", "maps": ["Henesys > Market"]}],
                "equipmentItems": [{"name": "equipment/Wooden Sword", "price": 100}],
                "useableItems": [{"name": "useable/Red Potion", "price": 50}],
                "quests": [{"name": "Helping Hand", "level": 5}],
                "recipes": [
                    {
                        "npc": "Shopkeeper",
                        "output": "Wooden Sword",
                        "materials": [{"item": "Slime Bubble", "quantity": 2}],
                    }
                ],
            },
            {"name": "Storage Keeper", "type": "Storage"},
        ],
    )
    _write_json(
        path=data_dir / "quests.json",
        payload=[
            {
                "name": "Helping Hand",
                "nameZh": "幫忙的手",
                "frequency": "daily",
                "lvLower": 5,
                "lvUpper": 20,
                "steps": [
                    {
                        "startNPC": "Shopkeeper",
                        "monstersToHunt": [{"name": "Slime", "quantity": 10}],
                        "itemsToCollect": {"misc": [{"name": "Slime Bubble", "quantity": 3}]},
                        "reward": {"exp": 100, "fame": 1},
                    }
                ],
            },
            {"name": "Another Help", "frequency": "one-time", "lvLower": 1},
        ],
    )
    _write_json(
        path=data_dir / "maps.json",
        payload=[
            {
                "region": "Victoria",
                "name": "Henesys > Hunting Ground",
                "nameZh": "弓箭手村狩獵場",
                "monsters": [{"name": "Slime", "level": 5}],
                "npcs": [{"name": "Shopkeeper", "type": "Shop", "subMap": "Market"}],
                "hidden": True,
            },
            {"region": "Victoria", "name": "Henesys > Market"},
        ],
    )
    _write_json(
        path=data_dir / "misc.json",
        payload=[{"name": "Slime Bubble", "nameZh": "綠水靈泡泡", "type": "Etc"}],
    )
    _write_json(
        path=data_dir / "translations.json",
        payload={
            "monsters": {"Slime": "綠水靈"},
            "equipment": {"Wooden Sword": "木劍"},
            "scrolls": {"Scroll for Gloves for ATT": "手套攻擊卷軸"},
            "useable": {"Red Potion": "紅色藥水"},
            "misc": {"Slime Bubble": "綠水靈泡泡"},
            "npcs": {"Shopkeeper": "商店老闆"},
            "quests": {"Helping Hand": "幫忙的手"},
            "maps": {"Henesys": "弓箭手村", "Hunting Ground": "狩獵場"},
            "region": {"Victoria": "維多利亞"},
            "eqType": {"Weapon": "武器", "Glove": "手套"},
            "job": {"Warrior": "劍士"},
            "npcType": {"Shop": "商店"},
            "modifiers": {"Fire": "火"},
        },
    )
    return data_dir


@pytest.fixture
def service(maple_data_dir: Path) -> MapleStoryService:
    """Returns a service loaded from the generated fixture data."""
    return MapleStoryService.from_directory(data_dir=maple_data_dir)


def test_maplestory_models_accept_aliases_and_helpers() -> None:
    """Verifies model aliases and computed helper properties."""
    drops = MonsterDrops(
        equipmentItems=[DropItem(name="Sword")],
        useableItems=[DropItem(name="Potion")],
        scrolls=[DropItem(name="Scroll")],
        miscItems=[DropItem(name="Etc")],
    )
    monster = Monster(
        name="Slime",
        nameZh="綠水靈",
        regionToMapsList=[RegionMaps(region="Victoria", maps=["A", "B"])],
        drops=drops,
    )
    equip = Equipment(
        name="Sword",
        nameZh="劍",
        equipmentRestriction=EquipmentRestriction(str=1, dex=2),
        stats=EquipmentStats(str=StatValue(middle=3), atk=StatValue(middle=5)),
    )
    quest = Quest(
        name="Quest",
        steps=[
            QuestStep(
                startNPC="NPC",
                monstersToHunt=[HuntTarget(name="Slime", quantity=2)],
                itemsToCollect={"misc": [CollectItem(name="Bubble", quantity=3)]},
                reward=QuestReward(exp=100, fame=1),
            )
        ],
    )
    assert monster.display_name == "綠水靈"
    assert monster.all_maps == ["A", "B"]
    assert [item.name for item in drops.all_items] == ["Sword", "Potion", "Scroll", "Etc"]
    assert equip.display_name == "劍"
    assert equip.equipment_restriction.has_requirements()
    assert equip.stats.non_zero_stats() == [
        ("STR", StatValue(middle=3, range=[])),
        ("ATK", StatValue(middle=5, range=[])),
    ]
    assert Scroll(name="Scroll", nameZh="卷").display_name == "卷"
    assert Useable(name="Potion", nameZh="藥水").display_name == "藥水"
    assert NPC(
        name="NPC", nameZh="店員", regionToMapsList=[RegionMaps(region="R", maps=["M"])]
    ).all_maps == ["M"]
    assert quest.display_name == "Quest"
    assert MapEntry(name="Map", nameZh="地圖").display_name == "地圖"
    assert MiscItem(name="Etc", nameZh="其他").display_name == "其他"
    assert (
        CraftingRecipe(
            npc="NPC", output="Sword", materials=[CraftingMaterial(item="Ores", quantity=1)]
        )
        .materials[0]
        .quantity
        == 1
    )


def test_maplestory_service_loads_searches_and_caches(service: MapleStoryService) -> None:
    """Verifies service loading, search helpers, translations, and stats caching."""
    assert service.has_data()
    assert service.translate(category="monsters", name="Slime") == "綠水靈"
    assert service.translate(category="missing", name="Slime") == "Slime"
    assert service.search_monsters_by_name(query="slime")[0].display_name == "綠水靈"
    assert service.search_monsters_by_name(query="綠")[0].name == "Slime"
    assert service.get_monster(name="綠水靈").name == "Slime"
    assert service.get_monster(name="missing") is None
    assert [
        monster.name for monster in service.get_monsters_by_drop(item_name="Wooden Sword")
    ] == ["Slime"]
    assert service.search_equipment_by_name(query="wooden")
    assert service.get_equipment(name="木劍").name == "Wooden Sword"
    assert service.get_equipment(name="missing") is None
    assert service.search_scrolls_by_name(query="gloves")
    assert service.search_npcs_by_name(query="shop")
    assert service.search_quests_by_name(query="help")
    assert service.search_maps_by_name(query="hunting")
    assert service.search_items_by_name(query="Bubble") == ["Blue Bubble", "Slime Bubble"]
    # Non-equipment drops must resolve through their own translation category.
    assert service.search_items_by_name(query="紅色藥水") == ["Red Potion"]
    assert service.search_items_by_name(query="手套攻擊卷軸") == ["Scroll for Gloves for ATT"]
    assert service.search_items_by_name(query="綠水靈泡泡") == ["Slime Bubble"]
    assert service.get_item_type(item_name="Wooden Sword") == "裝備"
    assert service.get_item_type(item_name="Scroll for Gloves for ATT") == "捲軸"
    assert service.get_item_type(item_name="Red Potion") == "消耗品"
    assert service.get_item_type(item_name="Slime Bubble") == "其它"
    assert service.get_item_type(item_name="Missing") == "未知"
    assert service.get_level_distribution() == {"0-9": 2}
    assert service.get_popular_items()[:2] == ["Wooden Sword", "Red Potion"]
    stats = service.get_stats()
    assert stats.total_monsters == 2
    assert service.get_stats() is stats


def test_maplestory_load_helpers_handle_missing_and_invalid_files(tmp_path: Path) -> None:
    """Verifies that malformed or missing data files degrade to empty data."""
    assert _load_json(path=tmp_path / "missing.json", model=Monster) == []
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(data="{bad", encoding="utf-8")
    assert _load_json(path=bad_json, model=Monster) == []
    assert _load_translations(data_dir=tmp_path) == {}
    # `_load_json` runs during the synchronous cog load, so every shape a hand-maintained
    # data file can take has to degrade instead of killing startup.
    off_schema = tmp_path / "off_schema.json"
    off_schema.write_text(data='[{"name": 1}]', encoding="utf-8")
    assert _load_json(path=off_schema, model=Monster) == []
    not_a_list = tmp_path / "not_a_list.json"
    not_a_list.write_text(data="12", encoding="utf-8")
    assert _load_json(path=not_a_list, model=Monster) == []


def test_maplestory_embeds_include_expected_sections(service: MapleStoryService) -> None:
    """Verifies that every MapleStory embed exposes its expected sections."""
    monster = service.get_monster(name="Slime")
    equip = service.get_equipment(name="Wooden Sword")
    scroll = service.search_scrolls_by_name(query="Gloves")[0]
    npc = service.search_npcs_by_name(query="Shopkeeper")[0]
    quest = service.search_quests_by_name(query="Helping")[0]
    map_entry = service.search_maps_by_name(query="Hunting")[0]
    assert monster is not None
    assert equip is not None

    embeds = [
        create_monster_embed(monster=monster, translate=service.translate),
        create_equipment_embed(equip=equip, translate=service.translate),
        create_scroll_embed(scroll=scroll, translate=service.translate),
        create_npc_embed(npc=npc, translate=service.translate),
        create_quest_embed(quest=quest, translate=service.translate),
        create_map_embed(map_entry=map_entry, translate=service.translate),
        create_item_source_embed(
            item_name="Wooden Sword", monsters=[monster], translate=service.translate
        ),
        build_stats_embed(stats=service.get_stats()),
    ]

    titles = [embed.title for embed in embeds]
    assert titles == [
        "🐲 綠水靈",
        "⚔️ 木劍",
        "📜 手套攻擊卷軸",
        "👤 商店老闆",
        "📋 幫忙的手",
        "🗺️ 弓箭手村狩獵場",
        "🎁 木劍",
        "📊 楓之谷資料庫統計",
    ]
    assert any(field.name == "⚔️ 裝備掉落" for field in embeds[0].fields)
    assert any(field.name == "📏 需求" for field in embeds[1].fields)
    assert any(field.name == "📊 屬性加成" for field in embeds[2].fields)
    assert any(field.name == "🗺️ 位置" for field in embeds[3].fields)
    assert any(field.name in {"步驟 1", "任務內容"} for field in embeds[4].fields)
    assert any(field.name == "🔐 隱藏地圖" for field in embeds[5].fields)
    assert _truncate(text="abcdef", limit=5) == "ab..."


async def test_maplestory_view_resolvers_and_selection(service: MapleStoryService) -> None:
    """Verifies resolver dispatch and select option truncation."""
    for search_type, name in [
        ("monster", "Slime"),
        ("item", "Wooden Sword"),
        ("equipment", "Wooden Sword"),
        ("scroll", "Scroll for Gloves for ATT"),
        ("npc", "Shopkeeper"),
        ("quest", "Helping Hand"),
        ("map", "Henesys > Hunting Ground"),
    ]:
        resolver = _RESOLVERS[search_type]
        embed = resolver(service=service, name=name, tr=service.translate)
        assert isinstance(embed, Embed)

    drop_view = MapleDropSearchView(service=service, search_type="monster", query="slime")
    drop_view.set_options(
        options=[SelectOption(label=f"Option {i}", value=str(i)) for i in range(30)]
    )
    assert len(drop_view.select_result.options) == 25


async def test_maplestory_view_select_result_handles_loading_and_valid_choice(
    service: MapleStoryService,
) -> None:
    """Verifies that the dropdown handles loading and valid selections."""
    view = MapleDropSearchView(service=service, search_type="monster", query="slime")
    interaction = _FakeInteraction()
    view.select_result._selected_values = ["loading"]
    await view.select_result.callback(interaction)
    assert interaction.response.deferred
    assert interaction.followup.sent[0]["content"] == "請先選擇有效的結果"

    valid_interaction = _FakeInteraction()
    view.select_result._selected_values = ["Slime"]
    await view.select_result.callback(valid_interaction)
    assert valid_interaction.followup.edited[0]["message_id"] == 777
    assert isinstance(valid_interaction.followup.edited[0]["embed"], Embed)
    assert valid_interaction.followup.edited[0]["view"] is None
    assert valid_interaction.followup.edited[0]["attachments"] == []
    assert (
        valid_interaction.followup.edited[0]["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    )
    assert valid_interaction.followup.edited[0]["embed"].image.url == embed_spacer_url()


@pytest.fixture
def maple_cog(maple_data_dir: Path) -> MapleStoryCogs:
    """Returns a MapleStory cog backed by the generated fixture data."""
    return MapleStoryCogs(bot=SimpleNamespace(), data_dir=maple_data_dir)


def test_maplestory_commands_are_grouped_under_maplestory() -> None:
    """Verifies MapleStory lookups are registered as /maplestory subcommands."""
    assert MapleStoryCogs.maplestory.name == "maplestory"
    assert MapleStoryCogs.maplestory.name_localizations[Locale.zh_TW] == "楓之谷"
    assert set(MapleStoryCogs.maplestory.children) == {
        "equip",
        "item",
        "map",
        "monster",
        "npc",
        "quest",
        "scroll",
        "stats",
    }
    assert MapleStoryCogs.maple_quest.name_localizations[Locale.zh_TW] == "任務"
    assert MapleStoryCogs.maple_map.name_localizations[Locale.zh_TW] == "地圖"
    assert MapleStoryCogs.maple_monster.name_localizations[Locale.zh_TW] == "怪物"
    assert MapleStoryCogs.maple_item.name_localizations[Locale.zh_TW] == "物品"
    assert MapleStoryCogs.maple_scroll.name_localizations[Locale.zh_TW] == "卷軸"
    assert MapleStoryCogs.maple_stats.name_localizations[Locale.zh_TW] == "統計"
    assert MapleStoryCogs.maple_equip.name_localizations[Locale.zh_TW] == "裝備"
    assert MapleStoryCogs.maple_npc.name_localizations[Locale.zh_TW] == "npc"


@pytest.mark.parametrize(
    argnames=("command_name", "query", "expected_title"),
    argvalues=[
        ("maple_monster", "綠水靈", "🐲 綠水靈"),
        ("maple_equip", "木劍", "⚔️ 木劍"),
        ("maple_scroll", "手套攻擊", "📜 手套攻擊卷軸"),
        ("maple_npc", "商店老闆", "👤 商店老闆"),
        ("maple_quest", "幫忙", "📋 幫忙的手"),
        ("maple_map", "狩獵場", "🗺️ 弓箭手村狩獵場"),
        ("maple_item", "Wooden Sword", "🎁 木劍"),
    ],
)
async def test_maplestory_commands_send_single_result_embed(
    maple_cog: MapleStoryCogs, command_name: str, query: str, expected_title: str
) -> None:
    """Verifies each single-result command sends the matching embed."""
    interaction = _FakeInteraction()
    command = getattr(MapleStoryCogs, command_name)
    await command.callback(maple_cog, interaction, name=query)
    embed = interaction.followup.sent[0]["embed"]
    assert interaction.response.deferred
    assert isinstance(embed, Embed)
    assert embed.title == expected_title


async def test_maplestory_commands_send_multi_result_view(maple_cog: MapleStoryCogs) -> None:
    """Verifies ambiguous monster search returns a selection view."""
    interaction = _FakeInteraction()
    await MapleStoryCogs.maple_monster.callback(maple_cog, interaction, name="Slime")
    payload = interaction.followup.sent[0]
    assert isinstance(payload["embed"], Embed)
    assert "找到 2 個相關怪物" in payload["embed"].description
    assert isinstance(payload["view"], MapleDropSearchView)
    assert payload["files"][0].filename == DEFAULT_EMBED_SPACER_FILENAME
    assert payload["embed"].image.url == embed_spacer_url()


async def test_maplestory_commands_send_not_found_and_stats(maple_cog: MapleStoryCogs) -> None:
    """Verifies not-found and stats command responses."""
    missing_interaction = _FakeInteraction()
    await MapleStoryCogs.maple_equip.callback(maple_cog, missing_interaction, name="missing")
    missing_embed = missing_interaction.followup.sent[0]["embed"]
    assert isinstance(missing_embed, Embed)
    assert "找不到名稱包含" in missing_embed.description

    stats_interaction = _FakeInteraction()
    await MapleStoryCogs.maple_stats.callback(maple_cog, stats_interaction)
    stats_embed = stats_interaction.followup.sent[0]["embed"]
    assert isinstance(stats_embed, Embed)
    assert stats_embed.title == "📊 楓之谷資料庫統計"


async def test_maplestory_command_error_path_when_data_missing(tmp_path: Path) -> None:
    """Verifies commands send the generic error embed when data is unavailable."""
    cog = MapleStoryCogs(bot=SimpleNamespace(), data_dir=tmp_path / "empty")
    interaction = _FakeInteraction()
    await MapleStoryCogs.maple_stats.callback(cog, interaction)
    embed = interaction.followup.sent[0]["embed"]
    assert isinstance(embed, Embed)
    assert embed.title == ":x: 錯誤"


def test_maplestory_setup_registers_cog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies setup registers the MapleStory cog synchronously."""
    added: list[tuple[MapleStoryCogs, bool | None]] = []
    monkeypatch.setattr(
        "discordbot.cogs.maplestory.MapleStoryService.from_directory",
        lambda data_dir: MapleStoryService(),
    )

    def record_cog(cog: MapleStoryCogs, override: bool | None = None) -> None:
        added.append((cog, override))

    bot = SimpleNamespace(add_cog=record_cog)

    maplestory.setup(bot=bot)

    assert isinstance(added[0][0], MapleStoryCogs)
    assert added[0][1] is True
