"""Tests for persistent feedback buttons, ephemeral reason picker, modal, and pipeline wiring."""

from typing import cast
import asyncio
from collections.abc import AsyncIterator

import pytest
from nextcord import ButtonStyle
from sqlalchemy import select
from nextcord.ui import Button
from openai.types.responses import ResponseStreamEvent

from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.streaming import ResponseStreamer
from discordbot.cogs.gen_reply.feedback_store import (
    ReplyTurnRow,
    ReplyFeedbackRow,
    open_session,
    record_reply_turn,
    toggle_feedback_vote,
)
from discordbot.cogs.gen_reply.feedback_views import (
    FEEDBACK_UP_CUSTOM_ID,
    FEEDBACK_DOWN_CUSTOM_ID,
    ReplyFeedbackView,
    DownvoteReasonView,
    DownvoteReasonModal,
    _ReasonButton,
)

from tests.helpers.casting import as_message, as_interaction
from tests.helpers.reply_fakes import (
    FakeMessage,
    _cog,
    _stream,
    _streamer,
    _text_event,
    _completed_event,
    _stream_events_from,
    hosting_off_planner,
)
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage


def _feedback_interaction(reply_message_id: int, user_id: int = 100) -> FakeInteraction:
    """Builds a component interaction targeting `reply_message_id`."""
    message = FakeDiscordMessage()
    message.id = reply_message_id
    return FakeInteraction(user=FakeUser(user_id=user_id, name=f"user{user_id}"), message=message)


async def _seed_turn(reply_message_id: int = 7001, author_id: int = 100) -> None:
    """Seeds one `reply_turn` row so view interactions find it."""
    await record_reply_turn(
        reply_message_id=reply_message_id,
        source_message_id=6001,
        guild_id=10,
        channel_id=20,
        author_id=author_id,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="hi",
        answer="hello",
    )


def _button_by_id(view: ReplyFeedbackView, custom_id: str) -> Button[ReplyFeedbackView]:
    """Finds the registered button on `view` matching `custom_id`."""
    for child in view.children:
        if isinstance(child, Button) and child.custom_id == custom_id:
            return child
    raise AssertionError(f"missing button {custom_id}")


async def test_feedback_view_records_switches_and_cancels_votes() -> None:
    """Thumbs-up/down defer ephemerally with a message, update `reply.db`, and toggle cleanly."""
    await _seed_turn(reply_message_id=7001)
    view = ReplyFeedbackView()
    up_btn = _button_by_id(view=view, custom_id=FEEDBACK_UP_CUSTOM_ID)
    down_btn = _button_by_id(view=view, custom_id=FEEDBACK_DOWN_CUSTOM_ID)

    up = _feedback_interaction(reply_message_id=7001)
    await up_btn.callback(as_interaction(fake=up))
    assert up.response.deferred is True
    assert up.response.deferred_ephemeral is True
    assert up.response.deferred_with_message is True
    assert up.edits[-1]["content"] == "已記錄 👍"
    assert "view" not in up.edits[-1]

    down = _feedback_interaction(reply_message_id=7001)
    await down_btn.callback(as_interaction(fake=down))
    assert "已改為 👎" in down.edits[-1]["content"]
    reason_view = down.edits[-1].get("view")
    assert isinstance(reason_view, DownvoteReasonView)
    assert len(reason_view.children) == 5

    cancel = _feedback_interaction(reply_message_id=7001)
    await down_btn.callback(as_interaction(fake=cancel))
    assert cancel.edits[-1]["content"] == "已取消評分"


async def test_feedback_view_handles_unknown_reply_without_error() -> None:
    """Clicking buttons on an unrecorded reply reports expiry cleanly."""
    view = ReplyFeedbackView()
    interaction = _feedback_interaction(reply_message_id=404404)

    await _button_by_id(view=view, custom_id=FEEDBACK_UP_CUSTOM_ID).callback(
        as_interaction(fake=interaction)
    )

    assert interaction.edits[-1]["content"] == "這則回覆已超過可評分範圍"


async def test_downvote_reason_picker_highlights_switches_and_opens_modal() -> None:
    """Preset tags update in place; `其他` opens the modal without writing until submitted."""
    await _seed_turn(reply_message_id=7002)
    feedback_view = ReplyFeedbackView()
    down = _feedback_interaction(reply_message_id=7002)
    await _button_by_id(view=feedback_view, custom_id=FEEDBACK_DOWN_CUSTOM_ID).callback(
        as_interaction(fake=down)
    )
    reason_view = cast("DownvoteReasonView", down.edits[-1]["view"])
    buttons = {
        btn.reason_tag: btn for btn in reason_view.children if isinstance(btn, _ReasonButton)
    }

    # Select a preset tag -> highlights in primary style and updates the ephemeral copy.
    pick = _feedback_interaction(reply_message_id=7002)
    await buttons["inaccurate"].callback(as_interaction(fake=pick))
    assert pick.response.deferred is True
    assert pick.response.deferred_with_message is False
    assert buttons["inaccurate"].style is ButtonStyle.primary
    assert buttons["tone"].style is ButtonStyle.secondary
    assert pick.edits[-1]["content"] == "已記錄 👎（資訊錯誤）"

    # Click `其他` -> opens modal immediately without deferring or mutating DB.
    open_other = _feedback_interaction(reply_message_id=7002)
    await buttons["other"].callback(as_interaction(fake=open_other))
    assert open_other.response.deferred is False
    assert isinstance(open_other.response.opened_modal, DownvoteReasonModal)

    # Submit modal -> persists free-text reason and edits ephemeral message.
    modal = open_other.response.opened_modal
    modal.reason_input._inputed_value = "沒有看我附的圖片"
    submit = _feedback_interaction(reply_message_id=7002)
    await modal.callback(as_interaction(fake=submit))
    assert submit.edits[-1]["content"] == "已記錄 👎（其他：沒有看我附的圖片）"
    assert buttons["other"].style is ButtonStyle.primary
    assert buttons["inaccurate"].style is ButtonStyle.secondary

    async with open_session() as session:
        row = await session.get(entity=ReplyFeedbackRow, ident=(7002, 100))
    assert row is not None
    assert row.reason_tag == "other"
    assert row.reason_text == "沒有看我附的圖片"

    # Switching vote to 👍 makes a stale reason click clear the reason buttons cleanly.
    switch = _feedback_interaction(reply_message_id=7002)
    await _button_by_id(view=feedback_view, custom_id=FEEDBACK_UP_CUSTOM_ID).callback(
        as_interaction(fake=switch)
    )
    stale = _feedback_interaction(reply_message_id=7002)
    await buttons["tone"].callback(as_interaction(fake=stale))
    assert stale.edits[-1]["content"] == "你已變更或取消這則評分"
    assert stale.edits[-1]["view"] is None


async def test_streamer_attaches_feedback_view_only_on_final_chunk() -> None:
    """On a chunked reply, the feedback buttons land on the last chunk, never earlier ones."""
    message = FakeMessage(content="long question")
    view = ReplyFeedbackView()
    streamer = _streamer(message=message, feedback_view=view)
    completed = _completed_event(input_tokens=10, output_tokens=20)
    completed.response.model = "gemini-4-neon"

    await _stream(
        streamer=streamer,
        responses=_stream_events_from(events=[_text_event(delta="A" * 2400), completed]),
    )

    assert len(message.replies) == 1
    parent = message.replies[0]
    assert parent.view is None
    assert len(parent.replies) == 1
    assert parent.replies[0].view is view
    assert streamer.last_reply is parent.replies[0]


async def test_answer_turn_records_real_post_fallback_model_and_respects_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`reply_turn` records the actual answered model, and `feedback_enabled=False` turns both off."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False
    message = FakeMessage(content="<@999> 幫我解釋 WAL")

    async def fake_create(**_kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        completed = _completed_event(input_tokens=10, output_tokens=5)
        # Simulate a LiteLLM fallback answering from a different model than requested.
        completed.response.model = "gemini-4-hydrogen"
        return _stream_events_from(events=[_text_event(delta="WAL 是預寫日誌"), completed])

    monkeypatch.setattr(cog.openai_client.responses, "create", fake_create)
    monkeypatch.setattr(AnswerTurn, "_schedule_memory_updates", lambda *_args, **_kwargs: None)

    turn = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=message)),
    )
    await turn.stream_answer(
        system_prompt="Be concise.", context=ReplyContext(), effort="high", question="幫我解釋 WAL"
    )

    assert len(message.replies) == 1
    assert isinstance(message.replies[0].view, ReplyFeedbackView)
    async with open_session() as session:
        rows = list((await session.execute(select(ReplyTurnRow))).scalars())
    assert len(rows) == 1
    assert rows[0].route == "QA"
    assert rows[0].model == "gemini-4-hydrogen"
    assert rows[0].effort == "high"
    assert rows[0].question == "幫我解釋 WAL"
    assert rows[0].answer == "WAL 是預寫日誌"

    # With kill-switch off, no buttons are attached and no new reply_turn row is inserted.
    cog.config.feedback_enabled = False
    message_off = FakeMessage(content="<@999> 第二題")
    turn_off = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=message_off)),
    )
    await turn_off.stream_answer(
        system_prompt="Be concise.", context=ReplyContext(), effort="low", question="第二題"
    )
    assert message_off.replies[0].view is None
    async with open_session() as session:
        rows_after = list((await session.execute(select(ReplyTurnRow))).scalars())
    assert len(rows_after) == 1


async def test_media_persona_reply_records_media_route_and_generator_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IMAGE persona replies attach feedback buttons and record `route='IMAGE'` with `image_model`."""
    cog = _cog()
    message = FakeMessage(content="<@999> 畫一隻貓")
    delivered = await message.reply(content="<@12345>")

    async def fake_create(**_kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        completed = _completed_event(input_tokens=10, output_tokens=5)
        completed.response.model = "gemini-4-hydrogen"
        return _stream_events_from(events=[_text_event(delta="畫好了!"), completed])

    monkeypatch.setattr(cog.openai_client.responses, "create", fake_create)
    context_task: asyncio.Task[ReplyContext] = asyncio.create_task(
        coro=asyncio.sleep(0, result=ReplyContext())
    )
    turn = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=message)),
    )
    await turn.stream_media_persona_reply(
        reply=as_message(fake=delivered),
        context_task=context_task,
        system_prompt="Reply about the image.",
        focus_part={
            "type": "input_image",
            "image_url": "data:image/png;base64,AA==",
            "detail": "auto",
        },
        media_noun="image",
        media_model="gemini-nano-banana-2.1",
        question="畫一隻貓",
    )

    assert isinstance(delivered.view, ReplyFeedbackView)
    async with open_session() as session:
        row = await session.get(entity=ReplyTurnRow, ident=delivered.id)
    assert row is not None
    assert row.route == "IMAGE"
    assert row.model == "gemini-nano-banana-2.1"
    assert row.question == "畫一隻貓"
    assert row.answer == "畫好了!"


async def test_on_ready_registers_persistent_feedback_view_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent feedback view is registered once on first ready, surviving reconnects."""
    cog = _cog()
    monkeypatch.setattr(cog, "_resume_memory", lambda: asyncio.sleep(0))

    await cog.on_ready()
    await cog.on_ready()
    while cog._tasks:
        await asyncio.gather(*list(cog._tasks))

    added = getattr(cog.bot, "added_views", [])
    assert len(added) == 1
    assert isinstance(added[0], ReplyFeedbackView)


def test_response_streamer_defaults_feedback_view_to_none() -> None:
    """Callers that omit `feedback_view` keep default None without touching views."""
    message = FakeMessage(content="plain")
    streamer = ResponseStreamer(
        message=as_message(fake=message),
        surface=TurnSurface.for_message(message=as_message(fake=message)),
        media_delivery=hosting_off_planner(),
    )
    assert streamer.feedback_view is None
    assert streamer.last_reply is None


async def test_persistent_feedback_view_skips_per_message_view_store() -> None:
    """`ReplyFeedbackView` stays persistent for `bot.add_view` without leaking per-reply entries."""
    view = ReplyFeedbackView()
    assert view.is_persistent() is True
    assert view.prevent_update is False


async def test_reply_turn_persists_before_inline_media_attaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Buttons clicked while slow inline media runs already find a persisted `reply_turn` row."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False
    message = FakeMessage(content="<@999> 說個故事")
    vote_during_media: list[object] = []

    async def fake_create(**_kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        completed = _completed_event(input_tokens=5, output_tokens=5)
        completed.response.model = "gemini-4-neon"
        return _stream_events_from(events=[_text_event(delta="從前從前"), completed])

    async def fake_attach_media(self: ResponseStreamer) -> None:
        assert self.last_reply is not None
        vote_during_media.append(
            await toggle_feedback_vote(
                reply_message_id=self.last_reply.id, user_id=12345, user_name="tester", vote="up"
            )
        )

    monkeypatch.setattr(cog.openai_client.responses, "create", fake_create)
    monkeypatch.setattr(ResponseStreamer, "_attach_generated_media", fake_attach_media)
    monkeypatch.setattr(AnswerTurn, "_schedule_memory_updates", lambda *_args, **_kwargs: None)

    turn = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=message)),
    )
    await turn.stream_answer(
        system_prompt="Be concise.", context=ReplyContext(), effort="medium", question="說個故事"
    )

    assert len(vote_during_media) == 1
    assert vote_during_media[0] is not None
