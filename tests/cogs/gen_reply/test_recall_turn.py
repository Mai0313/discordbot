"""Tests for the memory a whole reply turn reads, and where it lands in the request."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from collections import Counter
from unittest.mock import MagicMock

import pytest
import nextcord
from nextcord import Message

from discordbot.typings.models import RouteClassification, RecallRouteClassification
from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.cogs.gen_reply.recall import NO_STORED_MEMORY
from discordbot.services.memory.store import user_scope, write_tone, server_scope
from discordbot.cogs.gen_reply.prompts import REPLY_PROMPT, ROUTE_RECALL_SECTION, route_prompt
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.typings.context_budgets import MEMORY_CONTEXT_TARGET_USERS
from discordbot.cogs.gen_reply.status_marks import MEMORY_READ_EMOJI

from tests.helpers.casting import as_message
from tests.helpers.llm_input import (
    block_index,
    request_index,
    request_input,
    extract_tone_block,
    has_memory_context_block,
    extract_callable_user_ids,
    extract_user_memory_blocks,
    extract_server_memory_block,
)
from tests.helpers.reply_fakes import (
    FakeAuthor,
    FakeChannel,
    FakeMessage,
    FakeReference,
    _cog,
    _recorded,
    _seed_fact,
    _seed_alias,
    _text_event,
    _config_stub,
    _run_pipeline,
    _completed_event,
    _install_streamer,
    _assert_runtime_time_context,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nextcord.ext import commands

    from discordbot.cogs.gen_reply.cog import ReplyGeneratorCogs
    from discordbot.cogs.gen_reply.context import ReplyContext


def _assert_route_offered(cog: ReplyGeneratorCogs, candidates: set[int]) -> None:
    """Asserts the turn made one route call, shaped for exactly these recall candidates.

    No candidates means the plain shape: the plain schema and prompt, with neither the candidate
    block nor the server memory it is read against.
    """
    responses = _recorded(cog).responses
    (route_input,) = responses.parse_inputs
    if candidates:
        assert responses.parse_text_formats == [RecallRouteClassification]
        assert responses.parse_instructions == [
            route_prompt(inline_image_enabled=True) + ROUTE_RECALL_SECTION
        ]
        assert extract_callable_user_ids(request=route_input) == candidates
    else:
        assert responses.parse_text_formats == [RouteClassification]
        assert responses.parse_instructions == [route_prompt(inline_image_enabled=True)]
        assert extract_callable_user_ids(request=route_input) == set()
        assert extract_server_memory_block(request=route_input) is None


@pytest.mark.usefixtures("no_memory_review")
async def test_handle_message_reply_orders_reference_after_memory_before_current() -> None:
    """The answer input puts memory first, then the reference message, then the current message.

    The reference (the message being replied to) rides just above the current message so the
    reply pair stays adjacent and reads as the primary context, and the strengthened headers
    spell out the reply relationship.
    """
    cog = _cog()
    _seed_fact(scope=user_scope(user_id=1), text="喜歡簡短回覆")

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    parent_author = FakeAuthor(user_id=4)
    parent_author.name, parent_author.display_name = "parent", "Parent"
    parent = FakeMessage(content="原訊息", author=parent_author)
    message.reference = FakeReference(resolved=parent)

    _recorded(cog).responses.stream_queue = [
        [_text_event(delta="好"), _completed_event(input_tokens=1, output_tokens=1)]
    ]

    await _run_pipeline(cog=cog, message=message)

    answer = request_input(responses=_recorded(cog).responses)
    reference_index = block_index(request=answer, kind="reference")
    current_index = block_index(request=answer, kind="current")
    assert block_index(request=answer, kind="memory") < reference_index < current_index


@pytest.mark.usefixtures("no_memory_review")
async def test_handle_message_reply_orders_server_memory_user_memory_then_tone() -> None:
    """The answer injects server memory, user memory, then the tone note before the current message."""
    cog = _cog()
    _seed_fact(scope=user_scope(user_id=1), text="喜歡簡短回覆")
    _seed_fact(scope=user_scope(user_id=42), text="第三人記憶")
    _seed_fact(scope=server_scope(server_id=1), text="社群風格", section="profile")
    _seed_alias(subject_id=42, text="Boss(社群暱稱:李董)")
    write_tone(scope=user_scope(user_id=1), content="語氣輕鬆,句子精簡")
    write_tone(scope=user_scope(user_id=42), content="第三人語氣不該出現")

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    _recorded(cog).responses.stream_queue = [
        [_text_event(delta="好"), _completed_event(input_tokens=1, output_tokens=1)]
    ]

    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", recall_user_ids=["42"]
    )

    await _run_pipeline(cog=cog, message=message)

    answer = request_input(responses=_recorded(cog).responses)
    tone = extract_tone_block(request=answer)
    assert tone is not None
    assert "語氣輕鬆" in tone
    assert "第三人語氣" not in tone
    assert (
        block_index(request=answer, kind="server_memory")
        < block_index(request=answer, kind="memory")
        < block_index(request=answer, kind="tone")
        < block_index(request=answer, kind="current")
    )


@pytest.mark.usefixtures("no_memory_review")
async def test_reply_context_always_injects_the_author_tone_block() -> None:
    """The author's tone note rides every reply, with no selection phase of its own."""
    cog = _cog()
    write_tone(scope=user_scope(user_id=1), content="語氣輕鬆")

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    _recorded(cog).responses.stream_queue = [
        [_text_event(delta="好"), _completed_event(input_tokens=1, output_tokens=1)]
    ]

    await _run_pipeline(cog=cog, message=message)

    answer = request_input(responses=_recorded(cog).responses)
    assert not has_memory_context_block(request=answer)
    tone = extract_tone_block(request=answer)
    assert tone is not None
    assert "語氣輕鬆" in tone


async def test_handle_message_reply_answers_with_builtins_and_deterministic_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An optional alias nobody picked stays out while the answer keeps built-ins."""
    cog = _cog()
    # Every appended marker off and voice, which rides inside REPLY_PROMPT, on, so the
    # instructions checked below are REPLY_PROMPT whole.
    cog.config = _config_stub(inline_voice_enabled=True)
    _seed_fact(scope=user_scope(user_id=1), text="喜歡簡短回覆")
    _seed_alias(subject_id=42, text="Boss(社群暱稱:老闆)")

    scheduled: list[dict[str, object]] = []

    def fake_schedule(**kwargs: object) -> None:
        """Records the scheduled memory update arguments."""
        scheduled.append(kwargs)

    _install_streamer(monkeypatch=monkeypatch)
    monkeypatch.setattr("discordbot.cogs.gen_reply.answer.schedule_memory_update", fake_schedule)

    # The route picks nobody for the optional alias. The author's memory is deterministic and
    # must still be injected.
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await _run_pipeline(cog=cog, message=message)

    # Only the answer pays for slow_model; memory adds no call of its own.
    assert _recorded(cog).responses.create_models == [cog.toolkit.runtime_models.slow_model.name]

    # Answer keeps the built-in tools and the deterministic author memory.
    answer_idx = request_index(responses=_recorded(cog).responses)
    assert _recorded(cog).responses.create_tools[answer_idx] == list(
        cog.toolkit.runtime_models.slow_model.tools
    )
    _assert_runtime_time_context(
        instructions=_recorded(cog).responses.create_instructions[answer_idx],
        system_prompt=REPLY_PROMPT,
    )
    answer = request_input(responses=_recorded(cog).responses)
    assert "喜歡簡短回覆" in (extract_user_memory_blocks(request=answer).get(1) or "")
    assert 42 not in extract_user_memory_blocks(request=answer)

    # The memory review still receives a memory-free transcript.
    scheduled_list = scheduled[0]["message_list"]
    assert isinstance(scheduled_list, list)
    assert "喜歡簡短回覆" not in str(scheduled_list)
    assert scheduled[0]["scope"] == user_scope(user_id=1)
    assert scheduled[0]["full_reply"] == "完整回覆"
    assert scheduled[0]["writer"] is cog.toolkit.memory_writer
    assert scheduled[0]["identity"] == "Tester (tester) [id: 1]"
    assert (
        cog.toolkit.memory_writer.model.name == cog.toolkit.runtime_models.memory_writer_model.name
    )


async def test_handle_message_reply_without_stored_memory_keeps_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies a memory-less user gets untouched instructions but still schedules."""
    cog = _cog()
    # Every appended marker off and voice, which rides inside REPLY_PROMPT, on, so the
    # instructions checked below are REPLY_PROMPT whole.
    cog.config = _config_stub(inline_voice_enabled=True)

    scheduled: list[object] = []

    def fake_schedule(**kwargs: object) -> None:
        """Records that a memory update was scheduled."""
        scheduled.append(kwargs["scope"])

    _install_streamer(monkeypatch=monkeypatch)
    monkeypatch.setattr("discordbot.cogs.gen_reply.answer.schedule_memory_update", fake_schedule)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await _run_pipeline(cog=cog, message=message)

    answer_idx = request_index(responses=_recorded(cog).responses)
    assert _recorded(cog).responses.create_streams == [True]
    _assert_runtime_time_context(
        instructions=_recorded(cog).responses.create_instructions[answer_idx],
        system_prompt=REPLY_PROMPT,
    )
    assert Counter(scheduled) == Counter((user_scope(user_id=1), server_scope(server_id=1)))


@pytest.mark.parametrize(
    (
        "seeded",
        "server_nick",
        "mention_ids",
        "reference_author_id",
        "channel_public",
        "picks",
        "expected_injected",
        "expected_candidates",
    ),
    [
        ({1: "作者記憶"}, None, [], None, True, [], {1}, set()),
        ({}, None, [], None, True, [], set(), set()),
        ({1: "作者記憶", 2: "mention 記憶"}, None, [2], None, True, [], {1, 2}, set()),
        ({1: "作者記憶", 7: "reply 記憶"}, None, [], 7, True, [], {1, 7}, set()),
        (
            {1: "作者記憶", 42: "李董記憶"},
            (42, "Boss", "李董"),
            [],
            None,
            True,
            ["42"],
            {1, 42},
            {42},
        ),
        ({1: "作者記憶", 42: "李董記憶"}, (42, "Boss", "李董"), [], None, True, [], {1}, {42}),
        (
            {1: "作者記憶", 42: "李董記憶", 99: "局外人記憶"},
            (42, "Boss", "李董"),
            [],
            None,
            True,
            ["99"],
            {1},
            {42},
        ),
        (
            {1: "作者記憶", 42: "李董記憶"},
            (42, "Boss", "李董"),
            [],
            None,
            False,
            ["42"],
            {1},
            set(),
        ),
        (
            {1: "作者記憶", 42: "李董記憶"},
            (42, "Boss", "李董"),
            [42],
            None,
            True,
            ["42"],
            {1, 42},
            set(),
        ),
        (
            {1: "作者記憶", 999: "bot 記憶"},
            (999, "Bot", "破貓"),
            [],
            None,
            True,
            ["999"],
            {1},
            set(),
        ),
        (
            {
                **{uid: f"記憶{uid}" for uid in range(1, MEMORY_CONTEXT_TARGET_USERS + 3)},
                42: "額外記憶",
            },
            (42, "Boss", "李董"),
            list(range(2, MEMORY_CONTEXT_TARGET_USERS + 3)),
            None,
            True,
            ["42"],
            set(range(1, MEMORY_CONTEXT_TARGET_USERS + 3)),
            set(),
        ),
    ],
    ids=[
        "author-is-deterministic",
        "no-stored-memory",
        "explicit-mention-is-deterministic",
        "reference-author-is-deterministic",
        "public-alias-picked",
        "public-alias-declined",
        "noncandidate-id-dropped",
        "private-channel-offers-nothing",
        "explicit-mention-removed-from-candidates",
        "bot-alias-removed-from-candidates",
        "deterministic-memories-not-displaced-by-budget",
    ],
)
@pytest.mark.usefixtures("no_memory_review")
async def test_handle_message_reply_user_memory_injection(  # noqa: PLR0913 -- parametrized columns
    seeded: dict[int, str],
    server_nick: tuple[int, str, str] | None,
    mention_ids: list[int],
    reference_author_id: int | None,
    channel_public: bool,
    picks: list[str],
    expected_injected: set[int],
    expected_candidates: set[int],
) -> None:
    """Deterministic participants and optional public aliases stay in disjoint sets.

    Driven through the whole turn, so what the route is offered and what its picks resolve to
    are both the pipeline's own. Injection is asserted by id and the offer structurally, never
    by a sentinel substring over a serialized request.
    """
    cog = _cog()
    for uid, body in seeded.items():
        _seed_fact(scope=user_scope(user_id=uid), text=body)
    if server_nick is not None:
        nick_id, nick_name, nick_alias = server_nick
        _seed_alias(subject_id=nick_id, text=f"{nick_name}(社群暱稱:{nick_alias})")
    message = FakeMessage(
        content="<@999> hi", author=FakeAuthor(user_id=1), channel_public=channel_public
    )
    message.mentions = [FakeAuthor(user_id=uid) for uid in mention_ids]
    if reference_author_id is not None:
        parent_author = FakeAuthor(user_id=reference_author_id)
        parent_author.name, parent_author.display_name = "parent", "Parent"
        parent = FakeMessage(content="原訊息", author=parent_author)
        message.reference = FakeReference(resolved=parent)

    # Staged on every case, so a pick only lands where the route was actually offered one.
    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", recall_user_ids=picks
    )

    await _run_pipeline(cog=cog, message=message)

    _assert_route_offered(cog=cog, candidates=expected_candidates)
    answer = request_input(responses=_recorded(cog).responses)
    # An allowlisted-but-memoryless user gets a placeholder block, not a leak; the boundary is
    # which ids' real memory reaches the model, so placeholder sections are filtered out.
    injected = {
        uid
        for uid, body in extract_user_memory_blocks(request=answer).items()
        if body != NO_STORED_MEMORY
    }
    assert injected == expected_injected
    # The current user message stays last so the model answers it.
    assert isinstance(answer, list)
    assert answer[-1].get("role") == "user"


@pytest.mark.usefixtures("no_memory_review")
async def test_deterministic_memories_are_author_reply_mentions_ordered_and_deduped() -> None:
    """Deterministic participants stay author-first and never include the bot twice."""
    cog = _cog()
    for user_id in (1, 2, 3, 999):
        _seed_fact(scope=user_scope(user_id=user_id), text=f"記憶{user_id}")

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    parent = FakeMessage(content="原訊息", author=FakeAuthor(user_id=2))
    message.reference = FakeReference(resolved=parent)
    message.mentions = [
        FakeAuthor(user_id=2),
        FakeAuthor(user_id=3),
        FakeAuthor(user_id=999),
        FakeAuthor(user_id=1),
        FakeAuthor(user_id=3),
    ]

    await _run_pipeline(cog=cog, message=message)

    answer = request_input(responses=_recorded(cog).responses)
    assert list(extract_user_memory_blocks(request=answer)) == [1, 2, 3]
    assert _recorded(cog).responses.create_streams == [True]


@pytest.mark.usefixtures("no_memory_review")
async def test_history_only_users_are_not_memory_candidates() -> None:
    """A history author is neither deterministic nor an optional nickname candidate."""
    cog = _cog()
    _seed_fact(scope=user_scope(user_id=1), text="作者記憶")
    _seed_fact(scope=user_scope(user_id=2), text="歷史使用者記憶")

    history_message = FakeMessage(content="之前說過", author=FakeAuthor(user_id=2))

    async def fake_history(limit: int, before: FakeMessage) -> AsyncIterator[FakeMessage]:
        """Yields one unrelated history participant."""
        del limit, before
        yield history_message

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    message.channel = FakeChannel(history=fake_history)
    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", recall_user_ids=["2"]
    )

    await _run_pipeline(cog=cog, message=message)

    _assert_route_offered(cog=cog, candidates=set())
    answer = request_input(responses=_recorded(cog).responses)
    assert set(extract_user_memory_blocks(request=answer)) == {1}


@pytest.mark.parametrize("where", ["private-thread", "group-dm", "dm"])
@pytest.mark.usefixtures("no_memory_review")
async def test_a_channel_that_is_not_public_offers_the_route_no_candidates(where: str) -> None:
    """Outside a public guild channel, an absent nickname-table member is never offered.

    The route gets the plain shape, so a pick staged for that member has nowhere to land.
    """
    cog = _cog()
    _seed_fact(scope=user_scope(user_id=1), text="作者記憶")
    _seed_fact(scope=user_scope(user_id=42), text="第三人記憶")
    _seed_alias(subject_id=42, text="Boss(社群暱稱:李董)")

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    if where == "private-thread":
        cast("Any", message.channel).is_private = lambda: True
        cast("Any", message.channel).parent = FakeChannel(
            history=message._history, view_channel=True
        )
    else:
        message.guild = None
    if where == "dm":
        dm_channel = MagicMock(spec=nextcord.DMChannel)
        dm_channel.id = 555
        dm_channel.history = message._history
        message.channel = dm_channel
    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", recall_user_ids=["42"]
    )

    await _run_pipeline(cog=cog, message=message)

    _assert_route_offered(cog=cog, candidates=set())
    answer = request_input(responses=_recorded(cog).responses)
    assert set(extract_user_memory_blocks(request=answer)) == {1}


@pytest.mark.usefixtures("no_memory_review")
async def test_optional_picks_use_only_the_remaining_memory_budget() -> None:
    """Deterministic users fill all but one slot, leaving one optional alias slot.

    The route names a deterministic participant first, which must not take that slot: only the
    offered candidates can fill it, in the order the route named them.
    """
    cog = _cog()
    deterministic = range(1, MEMORY_CONTEXT_TARGET_USERS)
    for user_id in (*deterministic, 42, 43):
        _seed_fact(scope=user_scope(user_id=user_id), text=f"記憶{user_id}")
    for user_id, name in ((42, "李董"), (43, "阿伯")):
        _seed_alias(subject_id=user_id, text=f"Member{user_id}(社群暱稱:{name})")

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    message.mentions = [FakeAuthor(user_id=user_id) for user_id in deterministic[1:]]
    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", recall_user_ids=["2", "42", "43"]
    )

    await _run_pipeline(cog=cog, message=message)

    _assert_route_offered(cog=cog, candidates={42, 43})
    answer = request_input(responses=_recorded(cog).responses)
    assert set(extract_user_memory_blocks(request=answer)) == {*deterministic, 42}


@pytest.mark.parametrize("picks", [["41", "42"], ["42", "41"]])
@pytest.mark.usefixtures("no_memory_review")
async def test_a_memoryless_pick_never_takes_a_stored_picks_slot(picks: list[str]) -> None:
    """The one slot left goes to the pick with stored memory, whichever the route names first."""
    cog = _cog()
    deterministic = range(1, MEMORY_CONTEXT_TARGET_USERS)
    for user_id in (*deterministic, 42):
        _seed_fact(scope=user_scope(user_id=user_id), text=f"記憶{user_id}")
    for user_id, name in ((41, "阿伯"), (42, "李董")):
        _seed_alias(subject_id=user_id, text=f"Member{user_id}(社群暱稱:{name})")

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    message.mentions = [FakeAuthor(user_id=user_id) for user_id in deterministic[1:]]
    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", recall_user_ids=picks
    )

    await _run_pipeline(cog=cog, message=message)

    _assert_route_offered(cog=cog, candidates={41, 42})
    answer = request_input(responses=_recorded(cog).responses)
    assert set(extract_user_memory_blocks(request=answer)) == {*deterministic, 42}


@pytest.mark.usefixtures("no_memory_review")
async def test_the_route_is_offered_candidates_with_no_deterministic_memory() -> None:
    """The oblique-reference offer must not be gated on the deterministic lookup finding something.

    A conversation where nobody present has a stored fact is exactly the one the code-resolved
    path has nothing to contribute to, so gating the offer on it would switch the feature off
    in the case it exists for. Nothing downstream needs a non-empty starting list: the
    picked memories build the block from scratch.
    """
    cog = _cog()
    _seed_fact(scope=user_scope(user_id=42), text="李董記憶")
    _seed_alias(subject_id=42, text="Boss(社群暱稱:李董)")
    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", recall_user_ids=["42"]
    )

    await _run_pipeline(
        cog=cog, message=FakeMessage(content="<@999> 李董在嗎", author=FakeAuthor(user_id=1))
    )

    _assert_route_offered(cog=cog, candidates={42})
    answer = request_input(responses=_recorded(cog).responses)
    assert set(extract_user_memory_blocks(request=answer)) == {42}


async def test_an_ask_turn_offers_the_route_no_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/ask` in a server with a nickname table still gets the plain route shape.

    Its synthesized message carries no guild while the surface knows which server it is in, and
    the table is read off the message: the write side can never refresh it from `/ask`, so the
    route there is offered nobody and a pick staged for it has nowhere to land.
    """
    cog = _cog()
    _seed_fact(scope=user_scope(user_id=1), text="作者記憶")
    _seed_fact(scope=user_scope(user_id=42), text="第三人記憶")
    _seed_alias(subject_id=42, text="Boss(社群暱稱:李董)")
    contexts: list[ReplyContext] = []

    async def capture_answer(self: object, context: ReplyContext, **kwargs: object) -> None:
        """Keeps the context the answer would have read, without an interaction to send on."""
        del self, kwargs
        contexts.append(context)

    async def no_history(self: object, limit: int) -> list[Message]:
        """An `/ask` conversation with nothing before this turn, read without the ask store."""
        del self, limit
        return []

    monkeypatch.setattr(AnswerTurn, "stream_answer", capture_answer)
    monkeypatch.setattr(TurnSurface, "fetch_history", no_history)
    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", recall_user_ids=["42"]
    )
    message = FakeMessage(content="李董在嗎", author=FakeAuthor(user_id=1))
    message.guild = None
    surface = TurnSurface(
        message=as_message(fake=message),
        interaction=cast("nextcord.Interaction[commands.Bot]", SimpleNamespace()),
        guild_id=1,
    )

    await _run_pipeline(cog=cog, message=message, surface=surface)

    _assert_route_offered(cog=cog, candidates=set())
    (context,) = contexts
    assert context.server_memory_block is None
    assert context.memory_block is not None
    assert set(extract_user_memory_blocks(request=[context.memory_block])) == {1}


@pytest.mark.parametrize(
    ("seeded_ids", "server_nick", "mentions", "picks", "expected_tokens", "absent"),
    [
        (
            [1, 2, 3],
            None,
            [(2, "alice", "Alice"), (3, "bob", "Bob")],
            [],
            ("Tester (tester)", "Alice (alice)", "3"),
            ("Bob (bob)",),
        ),
        (
            [1, 42],
            (42, "Boss", "李董"),
            [],
            ["42", "42"],
            ("Tester (tester)", "2"),
            ("社群暱稱", "李董", "42"),
        ),
        ([1], (1, "Tester", "李董"), [], [], ("Tester (tester)",), ("社群暱稱", "李董")),
        ([], None, [], ["42"], (), (MEMORY_READ_EMOJI,)),
        # Nobody present is nameable and the route picked a table-only member: the footer has
        # no name to print, so it reports the bare count. Only reachable because the optional
        # offer is not gated on a deterministic memory existing.
        ([42], (42, "Boss", "李董"), [], ["42"], ("1",), ("社群暱稱", "李董", "42")),
    ],
    ids=[
        "owners-collapse-past-two",
        "absent-member-counted-never-named",
        "participant-alias-row-stays-out-of-the-credit",
        "no-memory-no-credit",
        "only-an-absent-member-leaves-the-count-alone",
    ],
)
@pytest.mark.usefixtures("no_memory_review")
async def test_handle_message_reply_memory_footer(  # noqa: PLR0913 -- parametrized columns
    seeded_ids: list[int],
    server_nick: tuple[int, str, str] | None,
    mentions: list[tuple[int, str, str]],
    picks: list[str],
    expected_tokens: tuple[str, ...],
    absent: tuple[str, ...],
) -> None:
    """The footer credits the memory owners actually read.

    Reads the user-visible reply text (the feature's small, real output surface): the single-owner
    credit, the collapse to a count past two owners, repeat-pick de-duplication, and the
    no-credit case. Two of them also pin that the `## 成員稱呼` row never reaches this line
    from either side: an absent member is counted into the total and never named at all, and a
    participant is named by their Discord label.
    """
    cog = _cog()
    for uid in seeded_ids:
        _seed_fact(scope=user_scope(user_id=uid), text=f"記憶{uid}")
    if server_nick is not None:
        nick_id, nick_name, nick_alias = server_nick
        _seed_alias(subject_id=nick_id, text=f"{nick_name}(社群暱稱:{nick_alias})")

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    mention_authors: list[FakeAuthor] = []
    for uid, name, display in mentions:
        author = FakeAuthor(user_id=uid)
        author.name, author.display_name = name, display
        mention_authors.append(author)
    message.mentions = mention_authors
    _recorded(cog).responses.output_parsed = RecallRouteClassification(
        decision="QA", recall_user_ids=picks
    )

    await _run_pipeline(cog=cog, message=message)

    content = message.replies[0].content or ""
    if expected_tokens:
        last_line = content.splitlines()[-1]
        assert last_line.startswith(f"-# {MEMORY_READ_EMOJI}")
        positions = [last_line.index(token) for token in expected_tokens]
        assert positions == sorted(positions)
        for fragment in absent:
            assert fragment not in last_line
    else:
        for fragment in absent:
            assert fragment not in content


@pytest.mark.parametrize(
    ("has_guild", "channel_public", "expect_server_read", "expect_scopes"),
    [
        (True, True, True, ["user", "server"]),
        (True, False, True, ["user"]),
        (False, True, False, ["user"]),
    ],
    ids=["guild-public", "guild-private", "dm"],
)
async def test_handle_message_reply_server_memory_gating(
    monkeypatch: pytest.MonkeyPatch,
    has_guild: bool,
    channel_public: bool,
    expect_server_read: bool,
    expect_scopes: list[str],
) -> None:
    """Server memory is read on a guild QA turn and written only from a public guild channel.

    One matrix over (guild/DM, public/private): the read block rides the answer only on a
    guild turn, the per-user write always runs, and the per-server write additionally needs a
    public guild channel.
    """
    cog = _cog()
    _seed_fact(scope=server_scope(server_id=1), text="社群風格", section="profile")
    scheduled: list[dict[str, object]] = []

    def fake_schedule(**kwargs: object) -> None:
        """Records each scheduled memory update."""
        scheduled.append(kwargs)

    monkeypatch.setattr("discordbot.cogs.gen_reply.answer.schedule_memory_update", fake_schedule)

    message = FakeMessage(
        content="<@999> hi", author=FakeAuthor(user_id=1), channel_public=channel_public
    )
    if not has_guild:
        message.guild = None
    _recorded(cog).responses.stream_queue = [
        [_text_event(delta="好"), _completed_event(input_tokens=1, output_tokens=1)]
    ]

    await _run_pipeline(cog=cog, message=message)

    answer = request_input(responses=_recorded(cog).responses)
    assert (extract_server_memory_block(request=answer) is not None) == expect_server_read

    server_scope_value = server_scope(server_id=1)
    name_to_scope = {"user": user_scope(user_id=1), "server": server_scope_value}
    assert Counter(update["scope"] for update in scheduled) == Counter(
        name_to_scope[name] for name in expect_scopes
    )
    # The user subject carries a second line naming the conversation source (guild or DM)
    # so the pipeline can stamp each observation deterministically; the server flavor never does.
    user_source = "guild 1" if has_guild else "dm"
    for update in scheduled:
        if update["scope"] == name_to_scope["user"]:
            assert update["subject"] == f"target_user_id: 1\nsource: {user_source}"
        if update["scope"] == server_scope_value:
            assert update["subject"] == "target_server_id: 1"
            assert update["writer"] is cog.toolkit.memory_writer
            assert update["identity"] == "Test Guild [id: 1]"
