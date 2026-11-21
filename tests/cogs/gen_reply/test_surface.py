"""Tests for the turn surface: where a reply goes and how long it has."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from datetime import timedelta

import pytest
from nextcord import File
from nextcord.enums import InteractionContextType
from nextcord.utils import utcnow
from nextcord.iterators import history_iterator

from discordbot.typings.timeouts import INTERACTION_DELIVERY_MARGIN_SECONDS
from discordbot.cogs.gen_reply.surface import INTERACTION_FOLLOWUP_LIMIT, TurnSurface
from discordbot.cogs.gen_reply.ask_store import load_ask_turns

from tests.helpers.casting import as_message, make_server_error, make_invalid_form_body
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


async def test_a_gateway_turn_records_nothing() -> None:
    """Discord's own channel history is the record there, so the store must stay empty."""
    interaction = _interaction()
    surface = TurnSurface.for_message(message=_ask_message(interaction=interaction))

    await surface.record_turn(answer="ignored")

    assert await load_ask_turns(channel_id=CHANNEL_ID, user_id=ASKER_ID, limit=10) == []


async def test_a_gateway_surface_still_reacts_instead_of_collecting() -> None:
    """Nothing about the `on_message` path changes: the hint is the reaction it always was."""
    message = FakeMessage()
    surface = TurnSurface.for_message(message=as_message(fake=message))

    await surface.hint(emoji="⚠️")

    assert message.added_reactions == ["⚠️"]
    assert surface.take_hints() == []


def test_a_gateway_turn_has_nothing_running_out() -> None:
    """`on_message` answers into a channel that is simply there, so no route is bounded by it."""
    message = _ask_message(interaction=_interaction())

    assert TurnSurface.for_message(message=message).delivery_budget_seconds() is None


def test_the_delivery_budget_holds_back_what_answering_will_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `/ask` turn may spend its token up to the margin the answer itself still needs."""
    fixed = utcnow()
    monkeypatch.setattr("discordbot.cogs.gen_reply.surface.utcnow", lambda: fixed)
    interaction = _interaction()
    interaction.deferred = True
    interaction.created_at = fixed
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")

    assert surface.delivery_budget_seconds() == 15 * 60 - INTERACTION_DELIVERY_MARGIN_SECONDS


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


async def test_failed_interaction_send_and_followup_do_not_consume_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed Discord calls leave `_original_consumed` and `_followups_spent` untouched."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")
    original_edit = interaction.edit_original_message
    original_followup = interaction.followup.send

    async def fail_once_edit(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("transient edit error")

    monkeypatch.setattr(interaction, "edit_original_message", fail_once_edit)
    with pytest.raises(RuntimeError, match="transient edit error"):
        await surface.send(content="first")
    assert surface.answer_capacity(has_landed_reply=False) == INTERACTION_FOLLOWUP_LIMIT + 1

    monkeypatch.setattr(interaction, "edit_original_message", original_edit)
    await surface.send(content="first")
    assert [edit["content"] for edit in interaction.edits] == ["first"]
    assert surface.answer_capacity(has_landed_reply=True) == INTERACTION_FOLLOWUP_LIMIT + 1

    async def fail_followup(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("transient followup error")

    monkeypatch.setattr(interaction.followup, "send", fail_followup)
    with pytest.raises(RuntimeError, match="transient followup error"):
        await surface.send(content="second")
    with pytest.raises(RuntimeError, match="transient followup error"):
        await surface.follow_up(previous=SimpleNamespace(), content="tail")  # ty: ignore[invalid-argument-type] -- never reaches previous.reply on /ask
    assert surface.answer_capacity(has_landed_reply=True) == INTERACTION_FOLLOWUP_LIMIT + 1

    monkeypatch.setattr(interaction.followup, "send", original_followup)
    await surface.follow_up(previous=SimpleNamespace(), content="tail")  # ty: ignore[invalid-argument-type] -- never reaches previous.reply on /ask
    assert surface.answer_capacity(has_landed_reply=True) == INTERACTION_FOLLOWUP_LIMIT


def _sent_files(kwargs: dict[str, object]) -> list[File]:
    """Extracts any `File` objects passed via `file` or `files` kwargs."""
    items: list[object] = []
    if (single := kwargs.get("file")) is not None:
        items.append(single)
    if isinstance(multi := kwargs.get("files"), list):
        items.extend(multi)
    return [item for item in items if isinstance(item, File)]


async def test_send_or_unparented_rebuilds_file_factories_and_reraises_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleted-target fallback rebuilds file factories after `File.close()` and re-raises 5xx."""
    reply_gone = make_invalid_form_body()
    channel_payloads: list[bytes] = []

    async def record_channel_send(**kwargs: object) -> SimpleNamespace:
        channel_payloads.extend(item.fp.read() for item in _sent_files(kwargs=kwargs))
        return SimpleNamespace(content=kwargs.get("content"))

    async def fail_reply_closing_files(**kwargs: object) -> object:
        # Mirror real `nextcord.abc.Messageable.send` closing file handles in `finally`.
        for item in _sent_files(kwargs=kwargs):
            item.fp.read()
            item.close()
        raise reply_gone

    # Case 1: single-file factory builds a fresh File per attempt.
    built_single = 0

    def _make_file() -> File:
        nonlocal built_single
        built_single += 1
        return File(fp=BytesIO(b"hello-factory"), filename="a.txt")

    msg1 = FakeMessage()
    monkeypatch.setattr(msg1.channel, "send", record_channel_send)
    monkeypatch.setattr(msg1, "reply", fail_reply_closing_files)
    surface1 = TurnSurface.for_message(message=as_message(fake=msg1))
    await surface1.send_or_unparented(content="hi", file=_make_file)
    assert built_single == 2
    assert channel_payloads[-1] == b"hello-factory"

    # Case 2: multi-file factory builds fresh Files per attempt.
    built_multi = 0

    def _make_files() -> list[File]:
        nonlocal built_multi
        built_multi += 1
        return [File(fp=BytesIO(b"hello-multi"), filename="b.txt")]

    msg2 = FakeMessage()
    monkeypatch.setattr(msg2.channel, "send", record_channel_send)
    monkeypatch.setattr(msg2, "reply", fail_reply_closing_files)
    surface2 = TurnSurface.for_message(message=as_message(fake=msg2))
    await surface2.send_or_unparented(content="hi", files=_make_files)
    assert built_multi == 2
    assert channel_payloads[-1] == b"hello-multi"

    # Case 3: non-deleted HTTP error propagates instead of sending unparented.
    msg3 = FakeMessage()
    msg3.reply_error = make_server_error()
    surface3 = TurnSurface.for_message(message=as_message(fake=msg3))
    with pytest.raises(type(msg3.reply_error)):
        await surface3.send_or_unparented(content="hi")
