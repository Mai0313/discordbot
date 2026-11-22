"""Tests for `🔄` reply regeneration: permissions, locks, blind-test exclusion, deletion, rollback."""

from types import SimpleNamespace
import asyncio
from collections.abc import AsyncIterator

import pytest
from nextcord import ChannelType, PartialMessageable
from sqlalchemy import select
from nextcord.ui import Button
from openai.types.responses import ResponseStreamEvent

from discordbot.typings.models import RouteClassification
from discordbot.cogs.gen_reply.turn_state import current_answer_streamer
from discordbot.cogs.gen_reply.feedback_store import (
    ReplyTurnRow,
    ReplyFeedbackRow,
    open_session,
    load_reply_turn,
    record_reply_turn,
    set_feedback_reason,
    toggle_feedback_vote,
    load_regen_chain_state,
    parse_reply_message_ids,
)
from discordbot.cogs.gen_reply.feedback_views import FEEDBACK_REGEN_CUSTOM_ID, ReplyFeedbackView

from tests.helpers.casting import as_interaction, make_forbidden, make_not_found, make_server_error
from tests.helpers.reply_fakes import (
    FakeAuthor,
    FakeMessage,
    _cog,
    _recorded,
    _text_event,
    _completed_event,
    _stream_events_from,
    _stream_events_then_raise,
)
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage


class _FakePartialMessage:
    """Partial message double recording delete attempts or raising a configured error."""

    def __init__(self, message_id: int, deleted: list[int], errors: dict[int, Exception]) -> None:
        self._message_id = message_id
        self._deleted = deleted
        self._errors = errors

    async def delete(self) -> None:
        if self._message_id in self._errors:
            raise self._errors[self._message_id]
        self._deleted.append(self._message_id)


class _FakeRegenChannel:
    """Channel double supporting positional-only `fetch_message` and `get_partial_message`."""

    def __init__(
        self,
        messages: dict[int, FakeMessage] | None = None,
        fetch_error: Exception | None = None,
        delete_errors: dict[int, Exception] | None = None,
    ) -> None:
        self.id = 555
        self._messages = messages or {}
        self._fetch_error = fetch_error
        self._delete_errors = delete_errors or {}
        self.deleted_ids: list[int] = []
        self.fetched_ids: list[int] = []

    async def fetch_message(self, message_id: int, /) -> FakeMessage:
        self.fetched_ids.append(message_id)
        if self._fetch_error is not None:
            raise self._fetch_error
        if message_id not in self._messages:
            raise make_not_found(message="source gone")
        return self._messages[message_id]

    def get_partial_message(self, message_id: int, /) -> _FakePartialMessage:
        return _FakePartialMessage(
            message_id=message_id, deleted=self.deleted_ids, errors=self._delete_errors
        )


def _regen_interaction(
    reply_message_id: int,
    user_id: int = 100,
    channel: object | None = None,
    display_name: str = "Asker Nick",
) -> FakeInteraction:
    """Builds a component interaction clicking `🔄` on `reply_message_id`."""
    message = FakeDiscordMessage()
    message.id = reply_message_id
    return FakeInteraction(
        user=FakeUser(user_id=user_id, name=f"user{user_id}", display_name=display_name),
        message=message,
        channel=channel,
        custom_id=FEEDBACK_REGEN_CUSTOM_ID,
    )


def _has_regen_button(view: object) -> bool:
    """Returns whether `view` is a `ReplyFeedbackView` carrying the `🔄` button."""
    return isinstance(view, ReplyFeedbackView) and any(
        isinstance(child, Button) and child.custom_id == FEEDBACK_REGEN_CUSTOM_ID
        for child in view.children
    )


@pytest.mark.usefixtures("no_memory_review")
async def test_regen_happy_path_guarantees_qa_reuses_effort_deletes_chunks_in_reverse_order() -> (
    None
):
    """Regen reuses effort, forces QA even if classifier returns IMAGE, records implicit feedback, and deletes chunks reversed."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False

    source = FakeMessage(content="<@999> Explain WAL", author=FakeAuthor(user_id=100))
    channel = _FakeRegenChannel(messages={source.id: source})
    await record_reply_turn(
        reply_message_id=2002,
        source_message_id=source.id,
        guild_id=1,
        channel_id=channel.id,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="high",
        backend="responses",
        question="Explain WAL",
        answer="Old chunked answer",
        reply_message_ids=[2001, 2002],
    )

    # Even if classifier picks IMAGE + low, regen forces QA + preserved effort="high".
    _recorded(cog).responses.output_parsed = RouteClassification(decision="IMAGE", effort="low")
    completed = _completed_event(input_tokens=15, output_tokens=20)
    completed.response.model = "gemini-4-neon"
    _recorded(cog).responses.stream_queue = [[_text_event(delta="Fresh answer"), completed]]

    click = _regen_interaction(reply_message_id=2002, user_id=100, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=click))

    assert click.response.deferred is True
    assert click.response.deferred_ephemeral is True
    assert click.response.deferred_with_message is True
    assert [edit["content"] for edit in click.edits] == ["正在重新生成回覆中⋯", "已重新生成回覆"]
    assert channel.fetched_ids == [source.id]
    assert channel.deleted_ids == [2002, 2001]
    assert _recorded(cog).responses.create_reasonings == [{"effort": "high", "summary": "auto"}]

    assert len(source.replies) == 1
    new_reply = source.replies[0]
    assert _has_regen_button(view=new_reply.view) is False

    async with open_session() as session:
        old_turn = await session.get(entity=ReplyTurnRow, ident=2002)
        new_turn = await session.get(entity=ReplyTurnRow, ident=new_reply.id)
        implicit_vote = await session.get(entity=ReplyFeedbackRow, ident=(2002, 100))
    assert old_turn is not None
    assert new_turn is not None
    assert new_turn.regen_of_reply_message_id == 2002
    assert new_turn.effort == "high"
    assert parse_reply_message_ids(raw=new_turn.reply_message_ids, fallback_id=new_reply.id) == [
        new_reply.id
    ]
    assert implicit_vote is not None
    assert implicit_vote.vote == "down"
    assert implicit_vote.reason_tag == "regen"
    assert implicit_vote.reason_text is None


@pytest.mark.usefixtures("no_memory_review")
async def test_regen_permission_rejects_bystander_and_allows_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the original asker or bot owner may regenerate a reply."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False

    source = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=100))
    channel = _FakeRegenChannel(messages={source.id: source})
    await record_reply_turn(
        reply_message_id=2101,
        source_message_id=source.id,
        guild_id=1,
        channel_id=channel.id,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="hi",
        answer="old",
        reply_message_ids=[2101],
    )

    bystander = _regen_interaction(reply_message_id=2101, user_id=200, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=bystander))
    assert bystander.edits[-1]["content"] == "只有原發問者可以重新生成這則回覆"
    assert source.replies == []
    assert channel.deleted_ids == []

    async def owner_check(user: object) -> bool:
        return getattr(user, "id", None) == 9999

    monkeypatch.setattr(cog.bot, "is_owner", owner_check)
    completed = _completed_event(input_tokens=5, output_tokens=5)
    completed.response.model = "gemini-4-neon"
    _recorded(cog).responses.stream_queue = [[_text_event(delta="owner regen"), completed]]

    owner_click = _regen_interaction(reply_message_id=2101, user_id=9999, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=owner_click))
    assert owner_click.edits[-1]["content"] == "已重新生成回覆"
    assert channel.deleted_ids == [2101]


@pytest.mark.usefixtures("no_memory_review")
async def test_regen_blind_test_excludes_tried_models_tracks_remaining_and_caps_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blind-test regen excludes prior chain models, shows remaining count, and enforces max_regens."""
    cog = _cog()
    cog.config.slow_model_blind_test_enabled = True
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False

    source = FakeMessage(content="<@999> compare models", author=FakeAuthor(user_id=100))
    channel = _FakeRegenChannel(messages={source.id: source})
    await record_reply_turn(
        reply_message_id=2201,
        source_message_id=source.id,
        guild_id=1,
        channel_id=channel.id,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-carbon",
        effort="medium",
        backend="responses",
        question="compare models",
        answer="answer 1",
        reply_message_ids=[2201],
    )

    last_pool: tuple[str, ...] = ()

    def deterministic_choice(seq: list[str]) -> str:
        nonlocal last_pool
        last_pool = tuple(seq)
        return seq[0]

    monkeypatch.setattr("discordbot.cogs.gen_reply.answer.secrets.choice", deterministic_choice)

    # First regen (R1 -> R2): carbon excluded -> picks neon, remaining = 1, R2 keeps 🔄.
    c2 = _completed_event(input_tokens=5, output_tokens=5)
    c2.response.model = "gemini-4-neon"
    _recorded(cog).responses.stream_queue = [[_text_event(delta="answer 2"), c2]]
    click1 = _regen_interaction(reply_message_id=2201, user_id=100, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=click1))

    assert last_pool == ("gemini-4-neon", "gemini-4-hydrogen")
    assert click1.edits[0]["content"] == "正在換另一個模型重新生成中⋯（本次後剩餘 1 次重抽）"
    assert click1.edits[-1]["content"] == "已重新生成回覆"
    r2 = source.replies[0]
    assert _has_regen_button(view=r2.view) is True

    # Second regen (R2 -> R3): carbon & neon excluded -> picks hydrogen, remaining = 0, R3 hides 🔄.
    c3 = _completed_event(input_tokens=5, output_tokens=5)
    c3.response.model = "gemini-4-hydrogen"
    _recorded(cog).responses.stream_queue = [[_text_event(delta="answer 3"), c3]]
    click2 = _regen_interaction(reply_message_id=r2.id, user_id=100, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=click2))

    assert last_pool == ("gemini-4-hydrogen",)
    assert click2.edits[0]["content"] == "正在換另一個模型重新生成中⋯（最後一次重抽）"
    assert click2.edits[-1]["content"] == "已重新生成回覆"
    r3 = source.replies[1]
    assert _has_regen_button(view=r3.view) is False

    # Attempting a third regen on R3 is rejected at the chain-depth cap.
    click3 = _regen_interaction(reply_message_id=r3.id, user_id=100, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=click3))
    assert click3.edits[-1]["content"] == "已達重新生成次數上限"


@pytest.mark.usefixtures("no_memory_review")
async def test_regen_serializes_concurrent_clicks_on_same_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second click arriving while the first regen runs waits on the lock and is rejected."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False

    source = FakeMessage(content="<@999> race me", author=FakeAuthor(user_id=100))
    channel = _FakeRegenChannel(messages={source.id: source})
    await record_reply_turn(
        reply_message_id=2301,
        source_message_id=source.id,
        guild_id=1,
        channel_id=channel.id,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="race me",
        answer="old",
        reply_message_ids=[2301],
    )

    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def gated_create(**_kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        completed = _completed_event(input_tokens=5, output_tokens=5)
        completed.response.model = "gemini-4-neon"
        return _stream_events_from(events=[_text_event(delta="winner"), completed])

    monkeypatch.setattr(cog.openai_client.responses, "create", gated_create)

    first = _regen_interaction(reply_message_id=2301, user_id=100, channel=channel)
    second = _regen_interaction(reply_message_id=2301, user_id=100, channel=channel)
    task1 = asyncio.create_task(
        coro=cog.regenerator.handle(interaction=as_interaction(fake=first))
    )
    await entered.wait()
    task2 = asyncio.create_task(
        coro=cog.regenerator.handle(interaction=as_interaction(fake=second))
    )
    await asyncio.sleep(delay=0)
    release.set()
    await asyncio.gather(task1, task2)

    assert calls == 1
    assert first.edits[-1]["content"] == "已重新生成回覆"
    assert second.edits[-1]["content"] == "這則回覆已經重新生成過了"
    assert len(source.replies) == 1
    assert channel.deleted_ids == [2301]


@pytest.mark.usefixtures("no_memory_review")
async def test_regen_failure_leaves_old_reply_and_feedback_untouched_and_cleans_dangling_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When replacement streaming fails, old chunks and feedback stay untouched and partial preview is deleted."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False

    source = FakeMessage(content="<@999> fail me", author=FakeAuthor(user_id=100))
    channel = _FakeRegenChannel(messages={source.id: source})
    await record_reply_turn(
        reply_message_id=2401,
        source_message_id=source.id,
        guild_id=1,
        channel_id=channel.id,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="fail me",
        answer="keep this old answer",
        reply_message_ids=[2400, 2401],
    )

    async def failing_create(**_kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        return _stream_events_then_raise(
            events=[_text_event(delta="partial...")], error=RuntimeError("boom")
        )

    monkeypatch.setattr(cog.openai_client.responses, "create", failing_create)
    # Speed up preview flush so a partial reply message is actually sent before failing.
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.streaming.ResponseStreamer.preview_interval_seconds",
        0.0,
        raising=False,
    )

    click = _regen_interaction(reply_message_id=2401, user_id=100, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=click))

    assert "重新生成失敗" in click.edits[-1]["content"]
    assert channel.deleted_ids == []
    assert current_answer_streamer.get() is None
    for preview in source.replies:
        assert preview.deleted is True

    async with open_session() as session:
        votes = list((await session.execute(select(ReplyFeedbackRow))).scalars())
        turns = list((await session.execute(select(ReplyTurnRow))).scalars())
    assert votes == []
    assert [turn.reply_message_id for turn in turns] == [2401]


@pytest.mark.usefixtures("no_memory_review")
async def test_regen_preserves_prior_explicit_downvote_reason() -> None:
    """Regenerating after an explicit thumbs-down keeps the user's original downvote reason."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False

    source = FakeMessage(content="<@999> fix it", author=FakeAuthor(user_id=100))
    channel = _FakeRegenChannel(messages={source.id: source})
    await record_reply_turn(
        reply_message_id=2501,
        source_message_id=source.id,
        guild_id=1,
        channel_id=channel.id,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="fix it",
        answer="bad answer",
        reply_message_ids=[2501],
    )
    await toggle_feedback_vote(
        reply_message_id=2501, user_id=100, user_name="user100", vote="down"
    )
    await set_feedback_reason(reply_message_id=2501, user_id=100, reason_tag="inaccurate")

    completed = _completed_event(input_tokens=5, output_tokens=5)
    completed.response.model = "gemini-4-neon"
    _recorded(cog).responses.stream_queue = [[_text_event(delta="fixed"), completed]]

    click = _regen_interaction(reply_message_id=2501, user_id=100, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=click))

    assert click.edits[-1]["content"] == "已重新生成回覆"
    async with open_session() as session:
        row = await session.get(entity=ReplyFeedbackRow, ident=(2501, 100))
    assert row is not None
    assert row.vote == "down"
    assert row.reason_tag == "inaccurate"


@pytest.mark.usefixtures("no_memory_review")
async def test_regen_deletion_ignores_not_found_and_logs_forbidden_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deletion skips `NotFound`, logs `Forbidden` at `warn` with ids and no traceback, and stops."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False

    source = FakeMessage(content="<@999> long reply", author=FakeAuthor(user_id=100))
    channel = _FakeRegenChannel(
        messages={source.id: source},
        delete_errors={
            2603: make_not_found(message="already gone"),
            2602: make_forbidden(message="missing access"),
        },
    )
    await record_reply_turn(
        reply_message_id=2603,
        source_message_id=source.id,
        guild_id=1,
        channel_id=channel.id,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="long reply",
        answer="old",
        reply_message_ids=[2601, 2602, 2603],
    )

    warns: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.regen.logfire.warn",
        lambda msg, **kwargs: warns.append((msg, kwargs)),
    )

    completed = _completed_event(input_tokens=5, output_tokens=5)
    completed.response.model = "gemini-4-neon"
    _recorded(cog).responses.stream_queue = [[_text_event(delta="new"), completed]]

    click = _regen_interaction(reply_message_id=2603, user_id=100, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=click))

    assert click.edits[-1]["content"] == "已重新生成回覆"
    # 2603 was NotFound (ignored), 2602 raised Forbidden (logged once, broke loop before 2601).
    assert channel.deleted_ids == []
    assert len(warns) == 1
    assert warns[0][1]["channel_id"] == channel.id
    assert warns[0][1]["message_id"] == 2602
    assert warns[0][1]["code"] == 0
    assert warns[0][1]["error_type"] == "Forbidden"
    assert "_exc_info" not in warns[0][1]


async def test_regen_rejects_disabled_unknown_unsupported_and_unreachable_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation guards reject disabled feedback, unknown/non-QA/ask turns, and unreachable sources."""
    cog = _cog()

    # Feedback disabled.
    cog.config.feedback_enabled = False
    disabled_click = _regen_interaction(reply_message_id=2701)
    await cog.regenerator.handle(interaction=as_interaction(fake=disabled_click))
    assert disabled_click.edits[-1]["content"] == "目前已停用重新生成功能"
    cog.config.feedback_enabled = True

    # Unknown reply_message_id.
    unknown_click = _regen_interaction(reply_message_id=2799)
    await cog.regenerator.handle(interaction=as_interaction(fake=unknown_click))
    assert unknown_click.edits[-1]["content"] == "這則回覆已超過可重新生成範圍"

    # Unsupported surface (`ask`) and route (`IMAGE`).
    await record_reply_turn(
        reply_message_id=2702,
        source_message_id=8702,
        guild_id=1,
        channel_id=555,
        author_id=100,
        surface="ask",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="q",
        answer="a",
    )
    ask_click = _regen_interaction(reply_message_id=2702, user_id=100)
    await cog.regenerator.handle(interaction=as_interaction(fake=ask_click))
    assert ask_click.edits[-1]["content"] == "這則回覆不支援重新生成"

    await record_reply_turn(
        reply_message_id=2703,
        source_message_id=8703,
        guild_id=1,
        channel_id=555,
        author_id=100,
        surface="gateway",
        route="IMAGE",
        model="gemini-nano-banana-2.1",
        effort="",
        backend="responses",
        question="draw",
        answer="done",
    )
    img_click = _regen_interaction(reply_message_id=2703, user_id=100)
    await cog.regenerator.handle(interaction=as_interaction(fake=img_click))
    assert img_click.edits[-1]["content"] == "這則回覆不支援重新生成"

    # Source fetch NotFound, Forbidden (warn with no traceback), and 5xx HTTPException.
    await record_reply_turn(
        reply_message_id=2704,
        source_message_id=8704,
        guild_id=1,
        channel_id=555,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="q",
        answer="a",
    )
    nf_click = _regen_interaction(
        reply_message_id=2704, user_id=100, channel=_FakeRegenChannel(messages={})
    )
    await cog.regenerator.handle(interaction=as_interaction(fake=nf_click))
    assert nf_click.edits[-1]["content"] == "找不到原始發問訊息，無法重新生成"

    warns: list[dict[str, object]] = []
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.regen.logfire.warn", lambda _msg, **kwargs: warns.append(kwargs)
    )
    fb_click = _regen_interaction(
        reply_message_id=2704,
        user_id=100,
        channel=_FakeRegenChannel(fetch_error=make_forbidden(message="no read")),
    )
    await cog.regenerator.handle(interaction=as_interaction(fake=fb_click))
    assert fb_click.edits[-1]["content"] == "沒有權限讀取原始訊息，無法重新生成"
    assert warns[-1]["channel_id"] == 555
    assert warns[-1]["message_id"] == 8704
    assert warns[-1]["code"] == 0
    assert warns[-1]["error_type"] == "Forbidden"
    assert "_exc_info" not in warns[-1]

    http_click = _regen_interaction(
        reply_message_id=2704,
        user_id=100,
        channel=_FakeRegenChannel(fetch_error=make_server_error(message="503")),
    )
    await cog.regenerator.handle(interaction=as_interaction(fake=http_click))
    assert http_click.edits[-1]["content"] == "讀取原始訊息時發生錯誤，請稍後再試"


@pytest.mark.usefixtures("no_memory_review")
async def test_regen_resolves_partial_messageable_interaction_channel_via_fetch_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DM/uncached `PartialMessageable` interaction channels resolve to a full channel before fetch."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False

    source = FakeMessage(content="<@999> DM question", author=FakeAuthor(user_id=100))
    source.guild = None
    full_channel = _FakeRegenChannel(messages={source.id: source})
    full_channel.id = 888

    await record_reply_turn(
        reply_message_id=2801,
        source_message_id=source.id,
        guild_id=None,
        channel_id=full_channel.id,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="DM question",
        answer="Old DM answer",
        reply_message_ids=[2801],
    )

    partial = PartialMessageable(
        state=SimpleNamespace(),  # ty: ignore[invalid-argument-type] -- minimal state for PartialMessageable
        id=full_channel.id,
        type=ChannelType.private,
    )
    fetched_channel_ids: list[int] = []

    async def fake_fetch_channel(channel_id: int) -> _FakeRegenChannel:
        fetched_channel_ids.append(channel_id)
        return full_channel

    monkeypatch.setattr(cog.bot, "get_channel", lambda _cid: None)
    monkeypatch.setattr(cog.bot, "fetch_channel", fake_fetch_channel, raising=False)

    completed = _completed_event(input_tokens=5, output_tokens=5)
    completed.response.model = "gemini-4-neon"
    _recorded(cog).responses.stream_queue = [[_text_event(delta="New DM answer"), completed]]

    click = _regen_interaction(reply_message_id=2801, user_id=100, channel=partial)
    await cog.regenerator.handle(interaction=as_interaction(fake=click))

    assert fetched_channel_ids == [full_channel.id]
    assert full_channel.fetched_ids == [source.id]
    assert full_channel.deleted_ids == [2801]
    assert click.edits[-1]["content"] == "已重新生成回覆"


@pytest.mark.usefixtures("no_memory_review")
async def test_regen_validates_access_with_single_row_lookup_and_walks_chain_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-lock validation reads only the target row; chain walking runs once under the lock."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False

    source = FakeMessage(content="<@999> Explain WAL", author=FakeAuthor(user_id=100))
    channel = _FakeRegenChannel(messages={source.id: source})
    await record_reply_turn(
        reply_message_id=2901,
        source_message_id=source.id,
        guild_id=1,
        channel_id=channel.id,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="Explain WAL",
        answer="Old answer",
        reply_message_ids=[2901],
    )

    row_lookups: list[int] = []
    chain_walks: list[int] = []

    async def spy_load_reply_turn(reply_message_id: int) -> ReplyTurnRow | None:
        row_lookups.append(reply_message_id)
        return await load_reply_turn(reply_message_id=reply_message_id)

    async def spy_load_chain(reply_message_id: int) -> object:
        chain_walks.append(reply_message_id)
        return await load_regen_chain_state(reply_message_id=reply_message_id)

    monkeypatch.setattr("discordbot.cogs.gen_reply.regen.load_reply_turn", spy_load_reply_turn)
    monkeypatch.setattr("discordbot.cogs.gen_reply.regen.load_regen_chain_state", spy_load_chain)

    # Rejected bystander stops after single-row lookup without walking the chain.
    bystander = _regen_interaction(reply_message_id=2901, user_id=999, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=bystander))
    assert bystander.edits[-1]["content"] == "只有原發問者可以重新生成這則回覆"
    assert row_lookups == [2901]
    assert chain_walks == []

    # Allowed asker reads the row pre-lock and walks the chain exactly once under the lock.
    row_lookups.clear()
    completed = _completed_event(input_tokens=5, output_tokens=5)
    completed.response.model = "gemini-4-neon"
    _recorded(cog).responses.stream_queue = [[_text_event(delta="Fresh"), completed]]
    asker = _regen_interaction(reply_message_id=2901, user_id=100, channel=channel)
    await cog.regenerator.handle(interaction=as_interaction(fake=asker))
    assert asker.edits[-1]["content"] == "已重新生成回覆"
    assert row_lookups == [2901]
    assert chain_walks == [2901]
