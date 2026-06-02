"""Tests for fishing embeds and interactive views."""

from types import SimpleNamespace
from typing import Any
from datetime import UTC, datetime

import pytest
from nextcord import Embed

from discordbot.cogs._fishing import views as fishing_views
from discordbot.cogs._fishing import database as fdb
from discordbot.typings.fishing import (
    GearView,
    CatchRoll,
    FishGrade,
    CastResult,
    CastStatus,
    CatchLogView,
    BaitStackView,
    AnglerStateView,
    FishingPanelData,
)
from discordbot.cogs._fishing.shop import partition_gear
from discordbot.cogs._fishing.views import (
    FishingPanelView,
    show_shop,
    begin_cast,
    show_panel,
    show_stats,
    show_leaderboard,
)
from discordbot.cogs._economy.database import adjust_balance
from discordbot.cogs._fishing.database import purchase_gear
from discordbot.cogs._fishing.defaults import (
    default_gear_upserts,
    build_default_catalog,
    default_grade_upserts,
    default_species_upserts,
)
from discordbot.cogs._fishing.presentation import (
    build_shop_embed,
    build_error_embed,
    build_panel_embed,
    build_stats_embed,
    build_reveal_embed,
    build_casting_embed,
    build_leaderboard_embed,
)

_GRADE_MAP = {grade.grade: grade for grade in build_default_catalog().grades}


class ResponseStub:
    """Minimal interaction response stub."""

    def __init__(self) -> None:
        """Initializes captured response state."""
        self.deferred = False
        self.sent: list[dict[str, Any]] = []
        self.modals: list[Any] = []

    async def defer(self, ephemeral: bool = False) -> None:
        """Records a deferred response."""
        self.deferred = True

    async def send_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records a sent response."""
        self.sent.append(kwargs)

    async def edit_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records an edited response."""
        self.sent.append(kwargs)

    async def send_modal(self, modal: Any) -> None:  # noqa: ANN401 -- test double
        """Records a launched modal."""
        self.modals.append(modal)

    def is_done(self) -> bool:
        """Returns whether this response has been used."""
        return self.deferred or bool(self.sent) or bool(self.modals)


class FollowupStub:
    """Minimal interaction followup stub."""

    def __init__(self) -> None:
        """Initializes captured followup payloads."""
        self.sent: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> "MessageStub":  # noqa: ANN401 -- test double
        """Records a followup send."""
        self.sent.append(kwargs)
        return MessageStub()


class MessageStub:
    """Minimal sent message stub."""

    def __init__(self) -> None:
        """Initializes fake message identity."""
        self.id = 123
        self.edits: list[dict[str, Any]] = []
        self.deleted = False

    async def edit(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records a message edit."""
        self.edits.append(kwargs)

    async def delete(self) -> None:
        """Records message deletion."""
        self.deleted = True


class InteractionStub:
    """Minimal interaction stub."""

    def __init__(self, user_id: int | None = 1, name: str = "alice") -> None:
        """Initializes fake Discord interaction pieces."""
        self.user = (
            SimpleNamespace(id=user_id, name=name, display_name=name)
            if user_id is not None
            else None
        )
        self.guild = None
        self.response = ResponseStub()
        self.followup = FollowupStub()
        self.message = MessageStub()


def _panel(
    rod: GearView | None = None, durability: int = 0, baits: tuple = ()
) -> FishingPanelData:
    """Builds a panel payload for embed tests."""
    return FishingPanelData(
        balance=12_345,
        angler=AnglerStateView(
            user_id=1, user_name="alice", rod=rod, durability_remaining=durability
        ),
        baits=baits,
        last_catch=None,
    )


def _assert_within_limits(embed: Embed) -> None:
    """Asserts an embed respects Discord's hard limits."""
    assert len(embed.title or "") <= 256
    assert len(embed.description or "") <= 4096
    assert len(getattr(embed.footer, "text", "") or "") <= 2048


def _rod_view() -> GearView:
    """Returns the default bamboo rod view."""
    return next(g for g in build_default_catalog().gear if g.gear_id == "rod_bamboo")


def test_panel_embed_branches_within_limits() -> None:
    """The panel renders within limits for no-rod, broken, and equipped states."""
    rod = _rod_view()
    baits = (BaitStackView(bait_id="bait_worm", name="蟲餌", emoji="🪱", quantity=12),)
    _assert_within_limits(build_panel_embed(panel=_panel(), grade_map=_GRADE_MAP))
    _assert_within_limits(
        build_panel_embed(panel=_panel(rod=rod, durability=0, baits=baits), grade_map=_GRADE_MAP)
    )
    equipped = build_panel_embed(
        panel=_panel(rod=rod, durability=18, baits=baits), grade_map=_GRADE_MAP
    )
    _assert_within_limits(equipped)
    assert "18/30" in (equipped.description or "")


def test_shop_embed_within_limits() -> None:
    """The shop embed lists rods and baits within limits."""
    rods, baits = partition_gear(gear=build_default_catalog().gear)
    embed = build_shop_embed(balance=999, rods=rods, baits=baits, notice="✅ ok")
    _assert_within_limits(embed)


def test_reveal_embed_jackpot_and_broken_within_limits() -> None:
    """The reveal embed handles a capped UR jackpot and a broken rod."""
    roll = CatchRoll(
        species_id="dragon",
        species_name="龍",
        grade=FishGrade.UR,
        emoji="🐉",
        size_bps=20_000,
        base_value=5_000,
        value=100_000,
        capped=True,
    )
    result = CastResult(
        status=CastStatus.PAYOUT_DEFERRED,
        roll=roll,
        payout=100_000,
        new_balance=100_000,
        rod_broke=True,
        durability_remaining=0,
        bait_id="bait_worm",
        bait_remaining=0,
    )
    embed = build_reveal_embed(result=result, panel=_panel(), grade_map=_GRADE_MAP)
    _assert_within_limits(embed)
    assert embed.color is not None
    assert embed.color.value == _GRADE_MAP[FishGrade.UR].color


def test_leaderboard_embed_full_within_limits() -> None:
    """A full 10-row leaderboard with large values stays within limits."""
    now = datetime.now(tz=UTC)
    catches = tuple(
        CatchLogView(
            user_id=i,
            user_name="長" * 32,
            species_id="dragon",
            species_name="龍" * 16,
            grade=FishGrade.UR,
            emoji="🐉",
            size_bps=20_000,
            value=100_000,
            created_at=now,
        )
        for i in range(10)
    )
    _assert_within_limits(build_leaderboard_embed(catches=catches, grade_map=_GRADE_MAP))
    _assert_within_limits(build_leaderboard_embed(catches=(), grade_map=_GRADE_MAP))


def test_casting_stats_error_embeds_within_limits() -> None:
    """The casting, stats, and error embeds stay within limits."""
    _assert_within_limits(build_casting_embed())
    _assert_within_limits(build_stats_embed(panel=_panel(), recent=()))
    _assert_within_limits(build_error_embed(message="x" * 200))


async def test_interaction_check_allows_owner_blocks_others() -> None:
    """Only the panel owner passes the interaction check."""
    view = FishingPanelView(owner_id=1)
    assert await view.interaction_check(interaction=InteractionStub(user_id=1)) is True
    intruder = InteractionStub(user_id=2, name="bob")
    assert await view.interaction_check(interaction=intruder) is False
    assert intruder.response.sent  # an ephemeral notice was sent


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_show_panel_builds_panel_view() -> None:
    """show_panel edits the message with a panel view owned by the caller."""
    interaction = InteractionStub(user_id=1)
    await show_panel(interaction=interaction, owner_id=1)
    assert interaction.response.sent
    assert interaction.response.sent[-1]["view"].owner_id == 1


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_show_shop_and_leaderboard_and_stats_render() -> None:
    """The shop, leaderboard, and stats navigation each render a view."""
    for grade in default_grade_upserts():
        await fdb.upsert_grade_config(config=grade)
    for species in default_species_upserts():
        await fdb.upsert_fish_species(species=species)
    for gear in default_gear_upserts():
        await fdb.upsert_gear(gear=gear)
    for nav in (show_shop, show_leaderboard, show_stats):
        interaction = InteractionStub(user_id=1)
        await nav(interaction=interaction, owner_id=1)
        assert interaction.response.sent[-1]["view"].owner_id == 1


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_begin_cast_runs_two_beat_animation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-bait cast plays the casting beat then reveals a post-cast view."""
    monkeypatch.setattr(fishing_views, "CAST_ANIMATION_SECONDS", 0.0)

    async def _no_avatar(**_kwargs: object) -> str:
        return ""

    monkeypatch.setattr(fishing_views, "guild_avatar_url", _no_avatar)
    for grade in default_grade_upserts():
        await fdb.upsert_grade_config(config=grade)
    for species in default_species_upserts():
        await fdb.upsert_fish_species(species=species)
    for gear in default_gear_upserts():
        await fdb.upsert_gear(gear=gear)
    await adjust_balance(user_id=1, name="alice", delta=100_000)
    await purchase_gear(user_id=1, name="alice", gear_id="rod_bamboo", quantity=1)
    await purchase_gear(user_id=1, name="alice", gear_id="bait_worm", quantity=5)

    interaction = InteractionStub(user_id=1)
    await begin_cast(interaction=interaction, owner_id=1)
    assert interaction.response.sent  # casting beat used the component response
    assert interaction.message.edits  # reveal edited the original message
    assert interaction.message.edits[-1]["view"].__class__.__name__ == "FishingPostCastView"


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_begin_cast_without_rod_shows_error() -> None:
    """Casting with no rod routes to the error view."""
    interaction = InteractionStub(user_id=1)
    await begin_cast(interaction=interaction, owner_id=1)
    assert interaction.response.sent[-1]["view"].__class__.__name__ == "FishingErrorView"
