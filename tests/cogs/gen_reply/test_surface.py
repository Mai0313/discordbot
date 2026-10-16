"""Tests for the turn surface: where a reply goes and how long it has."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from datetime import timedelta

import pytest
from nextcord.enums import InteractionContextType
from nextcord.utils import utcnow
from nextcord.iterators import history_iterator

from discordbot.typings.timeouts import INTERACTION_DELIVERY_MARGIN_SECONDS
from discordbot.cogs.gen_reply.surface import INTERACTION_FOLLOWUP_LIMIT, TurnSurface
from discordbot.cogs.gen_reply.ask_store import load_ask_turns

from tests.helpers.casting import as_message
from tests.helpers.reply_fakes import FakeAuthor, FakeChannel, FakeMessage
from tests.helpers.ask_interaction import (
    ASKER_ID,
    GUILD_ID,
    CHANNEL_ID,
    BOT_USER_ID,
    _ask_message,
    _interaction,
)


async def test_channel_history_ends_at_the_message_just_before_past_one_page() -> None:
    """History longer than one 100-message page still reads oldest first, up to this message.

    Runs nextcord's own pagination, whose `oldest_first=True` reverses each page on its own and
    so put the newest page first, leaving a reply read against messages hundreds back.
    """
    channel_ids = list(range(1, 251))

    class FakeHttp:
        async def logs_from(
            self, channel_id: int, limit: int, before: int | None, after: None, around: None
        ) -> list[dict[str, str]]:
            """Discord's messages endpoint: the `limit` newest before `before`, newest first."""
            del channel_id, after, around
            older = [i for i in channel_ids if before is None or i < before]
            return [{"id": str(i)} for i in reversed(older[-limit:])]

    class FakeState:
        http = FakeHttp()

        def create_message(self, channel: object, data: dict[str, str]) -> int:
            """Stands a message in by its id."""
            del channel
            return int(data["id"])

    class FakeMessageable:
        id = 555
        _state = FakeState()

        async def _get_channel(self) -> FakeMessageable:
            """Resolves to itself, as a text channel does."""
            return self

    current = FakeMessage(content="current", author=FakeAuthor(user_id=1))
    current.id = 251
    current.channel = FakeChannel(
        history=lambda **kwargs: history_iterator(
            messageable=cast("Any", FakeMessageable()), **kwargs
        )
    )

    history = await TurnSurface.for_message(message=as_message(fake=current)).fetch_history(
        limit=250
    )

    assert history == channel_ids


@pytest.mark.parametrize(
    ("context", "guild_id", "expected_guild", "expected_direct"),
    [
        (InteractionContextType.guild, GUILD_ID, GUILD_ID, False),
        (InteractionContextType.bot_dm, None, None, True),
        (InteractionContextType.private_channel, None, None, False),
    ],
)
def test_only_a_dm_with_the_bot_counts_as_a_direct_message(
    context: InteractionContextType,
    guild_id: int | None,
    expected_guild: int | None,
    expected_direct: bool,
) -> None:
    """A group DM must not read as one, because `dm_partner_id` opens every compartment.

    `private_channel` covers a group DM and a DM between two other people alike, and the
    channel object is the same `PartialMessageable` for both plus for a real 1:1 DM, so the
    interaction's own context is the only thing that can tell them apart. Reading it wrong
    hands the asker's private tier and every server's facts to a channel full of strangers.
    """
    interaction = _interaction(guild_id=guild_id, context=context)

    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    assert surface.guild_id == expected_guild
    assert surface.is_direct_message is expected_direct


async def test_the_first_send_edits_the_deferred_response_then_follows_up() -> None:
    """The deferred response is one free message; everything after it spends the budget."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")
    assert surface.answer_capacity(has_landed_reply=False) == INTERACTION_FOLLOWUP_LIMIT + 1

    await surface.send(content="first")
    await surface.send(content="second")

    assert [edit["content"] for edit in interaction.edits] == ["first"]
    assert [sent["content"] for sent in interaction.followup.sent] == ["second"]
    # One follow-up spent, and the answer sitting on the second one can still be edited.
    assert surface.answer_capacity(has_landed_reply=True) == INTERACTION_FOLLOWUP_LIMIT


async def test_a_follow_up_never_replies_into_a_channel_the_bot_is_not_in() -> None:
    """`previous.reply` is a plain channel send, which is a 403 here and loses the answer's tail."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    def _explode(**kwargs: object) -> None:
        """Fails if the chunk ever goes out as a reply to the previous message."""
        del kwargs
        raise AssertionError("a follow-up must not reply into the channel")

    await surface.follow_up(previous=SimpleNamespace(reply=_explode), content="tail")  # ty: ignore[invalid-argument-type] -- only `.reply` is reachable here

    assert [sent["content"] for sent in interaction.followup.sent] == ["tail"]


async def test_a_dropped_clip_is_written_where_it_cannot_be_reacted() -> None:
    """The ⏱️ / ⚠️ is the only trace a dropped clip leaves, so it must survive having no message."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    await surface.hint(emoji="⚠️")
    await surface.hint(emoji="⚠️")
    await surface.hint(emoji="⏱️")

    assert surface.take_hints() == ["⚠️", "⏱️"]
    assert surface.take_hints() == []


@pytest.mark.usefixtures("ask_isolated_db")
async def test_the_surface_records_a_turn_and_replays_it_next_time() -> None:
    """One turn's answer is the next turn's history, since Discord keeps none of it for us."""
    interaction = _interaction()
    first = TurnSurface.for_interaction(interaction=interaction, question="你叫什麼")
    assert await first.fetch_history(limit=500) == []

    await first.record_turn(answer="我叫破貓")

    later = TurnSurface.for_interaction(interaction=interaction, question="剛剛說了什麼")
    history = await later.fetch_history(limit=500)
    assert [message.content for message in history] == ["你叫什麼", "我叫破貓"]
    assert [message.author.id for message in history] == [ASKER_ID, BOT_USER_ID]


@pytest.mark.usefixtures("ask_isolated_db")
async def test_a_gateway_turn_records_nothing() -> None:
    """Discord's own channel history is the record there, so the store must stay empty."""
    interaction = _interaction()
    surface = TurnSurface.for_message(message=_ask_message(interaction=interaction))

    await surface.record_turn(answer="ignored")

    assert await load_ask_turns(channel_id=CHANNEL_ID, user_id=ASKER_ID, limit=10) == []


async def test_a_gateway_surface_still_reacts_instead_of_collecting() -> None:
    """Nothing about the `on_message` path changes: the hint is the reaction it always was."""
    added: list[str] = []

    class _Reactable:
        """A message that records the reactions added to it."""

        guild = None
        channel = object()

        async def add_reaction(self, emoji: str) -> None:
            """Records one reaction."""
            added.append(emoji)

    surface = TurnSurface.for_message(message=_Reactable())  # ty: ignore[invalid-argument-type] -- only `.add_reaction` is reachable here

    await surface.hint(emoji="⚠️")

    assert added == ["⚠️"]
    assert surface.take_hints() == []


def _deferred_surface(**kwargs: Any) -> TurnSurface:  # noqa: ANN401 -- forwards to the untyped fake
    """The surface `/ask` builds, after the defer that opens the fifteen-minute window."""
    interaction = _interaction(**kwargs)
    interaction.deferred = True
    return TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")


def test_a_gateway_turn_has_nothing_running_out() -> None:
    """`on_message` answers into a channel that is simply there, so no route is bounded by it."""
    message = _ask_message(interaction=_interaction())

    assert TurnSurface.for_message(message=message).delivery_budget_seconds() is None


def test_the_delivery_budget_holds_back_what_answering_will_cost() -> None:
    """A `/ask` turn may spend its token up to the margin the answer itself still needs."""
    budget = _deferred_surface().delivery_budget_seconds()

    assert budget is not None
    # Fifteen minutes of token, less the margin, less however long the test took to get here.
    usable = 15 * 60 - INTERACTION_DELIVERY_MARGIN_SECONDS
    assert usable - 5.0 < budget <= usable


def test_a_window_already_gone_leaves_nothing_to_spend() -> None:
    """Past the token the budget floors at zero, so a route starts nothing it cannot finish."""
    interaction = _interaction()
    interaction.deferred = True
    interaction.created_at = utcnow() - timedelta(minutes=30)
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    assert surface.delivery_budget_seconds() == 0.0


def test_an_undeferred_invocation_has_nothing_to_spend_either() -> None:
    """Three seconds is what an unanswered interaction really holds, so it may start nothing slow.

    `/ask` defers before it builds its surface, so nothing reaches this today. Pinned because the
    budget reads `Interaction.expires_at`, which tells the truth about both of Discord's windows
    rather than assuming the deferred one.
    """
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    assert surface.delivery_budget_seconds() == 0.0
