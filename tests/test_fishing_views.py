"""Pins the fishing surface: what its embeds may not exceed, and where each control leads.

Two halves over `cogs/games/fishing/`, both about presentation rather than rules —
`tests/test_fishing_catch.py` owns the roll arithmetic and `tests/test_fishing_db.py` the
settlement, so nothing here asserts what a cast was worth.

The embed half calls `presentation.py`'s builders directly, which it can because they are pure:
frozen read models plus a grade map in, one `nextcord.Embed` out, no interaction and no database.
Each builder is driven through the branch that renders the most text — a rod at zero durability, a
capped `FishGrade.UR` jackpot whose payout also deferred, ten leaderboard rows of 32-character
names — and the result is measured against Discord's title / description / footer ceilings.
Everything a fishing embed shows is composed into its description out of catalog rows and
user-supplied display names, so a re-tuned `fish_grade_config` or a long enough name is what would
push one over, and Discord answers an oversize embed with a 400 rather than truncating it.
`_GRADE_MAP` comes from `defaults.py` instead of from a seeded database for the same reason this
half needs no fixture: a builder reads the catalog only through the map it is handed.

The view half drives `views.py`'s module-level transitions with a stub interaction. Every one of
them ends in `edit_owned_public_message`, which chooses its Discord call off
`interaction.response.is_done()`: an unanswered press is acknowledged and repainted in a single
`response.edit_message`, while an already-answered interaction is written to the message object
instead. The stubs therefore record per surface, which is what makes the two-beat cast assertable:
the casting beat spends the component response, then the reveal edits the original message.
Every navigation assertion reads the handed-over view rather than the embed: the view class is the
screen's identity, and its `owner_id` is the gate that keeps a stranger off someone else's panel.

The transitions that read state take `fishing_isolated_db` and carry the catalog in themselves,
since that fixture creates the schema and seeds nothing. The cast test buys its rod and bait
through the real store rather than faking panel state, so the animation runs over a state
settlement would accept; the no-rod test wants the opposite and seeds nothing at all.
"""

from types import SimpleNamespace
from typing import Any
from datetime import UTC, datetime

import pytest
from nextcord import Embed

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
from discordbot.cogs.games.fishing import views as fishing_views
from discordbot.cogs.games.fishing import database as fdb
from discordbot.cogs.games.fishing.shop import partition_gear
from discordbot.cogs.games.fishing.views import (
    FishingPanelView,
    show_shop,
    begin_cast,
    show_panel,
    show_stats,
    show_leaderboard,
)
from discordbot.services.economy.database import adjust_balance
from discordbot.cogs.games.fishing.database import purchase_gear
from discordbot.cogs.games.fishing.defaults import (
    default_gear_upserts,
    build_default_catalog,
    default_grade_upserts,
    default_species_upserts,
)
from discordbot.cogs.games.fishing.presentation import (
    build_shop_embed,
    build_error_embed,
    build_panel_embed,
    build_stats_embed,
    build_reveal_embed,
    build_casting_embed,
    build_leaderboard_embed,
)

from tests.helpers.casting import as_interaction

_GRADE_MAP = {grade.grade: grade for grade in build_default_catalog().grades}


class ResponseStub:
    """Stands in for `InteractionResponse`, recording what each transition answered with."""

    def __init__(self) -> None:
        """Initializes the empty defer, payload and modal records."""
        self.deferred = False
        self.sent: list[dict[str, Any]] = []
        self.modals: list[Any] = []

    async def defer(self, ephemeral: bool = False) -> None:
        """Marks the interaction answered without recording a payload."""
        self.deferred = True

    async def send_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records a first response, keeping whatever the caller passed."""
        self.sent.append(kwargs)

    async def edit_message(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records a repaint; both response paths share one list, which is what assertions read."""
        self.sent.append(kwargs)

    async def send_modal(self, modal: Any) -> None:  # noqa: ANN401 -- test double
        """Records a launched modal, which also counts as answering the interaction."""
        self.modals.append(modal)

    def is_done(self) -> bool:
        """Returns whether anything has answered this interaction yet.

        `edit_owned_public_message` branches on it: false repaints through the response, true
        writes to the message object instead, which is the difference the cast animation is read
        through.
        """
        return self.deferred or bool(self.sent) or bool(self.modals)


class FollowupStub:
    """Stands in for `Interaction.followup`, the path taken when the panel message is gone."""

    def __init__(self) -> None:
        """Initializes the empty followup record."""
        self.sent: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> "MessageStub":  # noqa: ANN401 -- test double
        """Returns a fresh message stub for the recorded followup, as `wait=True` does."""
        self.sent.append(kwargs)
        return MessageStub()


class MessageStub:
    """Stands in for the one public message a fishing panel lives on."""

    def __init__(self) -> None:
        """Initializes the fake message identity and its empty edit record."""
        self.id = 123
        self.edits: list[dict[str, Any]] = []
        self.deleted = False

    async def edit(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double
        """Records a repaint written straight to the message, the already-answered path."""
        self.edits.append(kwargs)

    async def delete(self) -> None:
        """Records the deletion an idle timeout would perform."""
        self.deleted = True


class InteractionStub:
    """Stands in for the interaction a fishing transition is handed."""

    def __init__(self, user_id: int | None = 1, name: str = "alice") -> None:
        """Initializes the identity plus the three surfaces a transition can answer through.

        A None `user_id` leaves `user` unset, which is the identity-less interaction
        `require_fishing_user` refuses before any fishing state is keyed on it.
        """
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
    rod: GearView | None = None, durability: int = 0, baits: tuple[BaitStackView, ...] = ()
) -> FishingPanelData:
    """Builds panel state for the embed tests, varying only what the panel branches on.

    Returns:
        A panel with a fixed balance and no last catch, carrying the given rod, the casts left on
        it, and the owned bait stacks.
    """
    return FishingPanelData(
        balance=12_345,
        angler=AnglerStateView(
            user_id=1, user_name="alice", rod=rod, durability_remaining=durability
        ),
        baits=baits,
        last_catch=None,
    )


def _assert_within_limits(embed: Embed) -> None:
    """Asserts an embed fits Discord's title, description and footer ceilings.

    Every fishing builder composes its whole body into the description, so that is the one of the
    three ceilings a long name or a re-tuned catalog can actually reach.
    """
    assert len(embed.title or "") <= 256
    assert len(embed.description or "") <= 4096
    assert len(getattr(embed.footer, "text", "") or "") <= 2048


def _rod_view() -> GearView:
    """Returns the starter rod from the shipped catalog.

    Returns:
        The `rod_bamboo` row, whose durability of 30 is the denominator the panel test reads back
        out of the rendered bar.
    """
    return next(g for g in build_default_catalog().gear if g.gear_id == "rod_bamboo")


def test_panel_embed_branches_within_limits() -> None:
    """The panel embed stays within limits with no rod, a broken rod, and casts left on one."""
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
    """The shop embed stays within limits listing the whole default catalog."""
    rods, baits = partition_gear(gear=build_default_catalog().gear)
    embed = build_shop_embed(balance=999, rods=rods, baits=baits, notice="✅ ok")
    _assert_within_limits(embed)


def test_reveal_embed_jackpot_and_broken_within_limits() -> None:
    """A capped UR jackpot on a broken rod stays within limits and takes the grade's color."""
    roll = CatchRoll(
        species_id="dragon",
        species_name="龍",
        grade=FishGrade.UR,
        emoji="🐉",
        size_bps=20_000,
        size_rank_bps=10_000,
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
    """A full ten-row leaderboard of long names stays within limits, as does an empty one."""
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
    """The casting, stats and error embeds stay within limits."""
    _assert_within_limits(build_casting_embed())
    _assert_within_limits(build_stats_embed(panel=_panel(), recent=()))
    _assert_within_limits(build_error_embed(message="x" * 200))


async def test_interaction_check_allows_owner_blocks_others() -> None:
    """Only the panel owner passes the check; anyone else is refused with an ephemeral notice."""
    view = FishingPanelView(owner_id=1)
    assert (
        await view.interaction_check(interaction=as_interaction(fake=InteractionStub(user_id=1)))
        is True
    )
    intruder = InteractionStub(user_id=2, name="bob")
    assert await view.interaction_check(interaction=as_interaction(fake=intruder)) is False
    assert intruder.response.sent  # an ephemeral notice was sent


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_show_panel_builds_panel_view() -> None:
    """The panel transition repaints through the response with a view locked to the opener."""
    interaction = InteractionStub(user_id=1)
    await show_panel(interaction=as_interaction(fake=interaction), owner_id=1)
    assert interaction.response.sent
    assert interaction.response.sent[-1]["view"].owner_id == 1


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_show_shop_and_leaderboard_and_stats_render() -> None:
    """The shop, leaderboard and stats transitions each hand over a view locked to the opener."""
    for grade in default_grade_upserts():
        await fdb.upsert_grade_config(config=grade)
    for species in default_species_upserts():
        await fdb.upsert_fish_species(species=species)
    for gear in default_gear_upserts():
        await fdb.upsert_gear(gear=gear)
    for nav in (show_shop, show_leaderboard, show_stats):
        interaction = InteractionStub(user_id=1)
        await nav(interaction=as_interaction(fake=interaction), owner_id=1)
        assert interaction.response.sent[-1]["view"].owner_id == 1


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_begin_cast_runs_two_beat_animation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-bait cast spends the response on the casting beat, then edits in the reveal."""
    monkeypatch.setattr(fishing_views, "CAST_ANIMATION_SECONDS", 0.0)

    # The stub user carries no display_avatar, which the real resolver reads before anything else.
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
    await begin_cast(interaction=as_interaction(fake=interaction), owner_id=1)
    assert interaction.response.sent  # casting beat used the component response
    assert interaction.message.edits  # reveal edited the original message
    assert interaction.message.edits[-1]["view"].__class__.__name__ == "FishingPostCastView"


@pytest.mark.usefixtures("fishing_isolated_db")
async def test_begin_cast_without_rod_shows_error() -> None:
    """Casting with no rod lands on the error view instead of spending a cast."""
    interaction = InteractionStub(user_id=1)
    await begin_cast(interaction=as_interaction(fake=interaction), owner_id=1)
    assert interaction.response.sent[-1]["view"].__class__.__name__ == "FishingErrorView"
