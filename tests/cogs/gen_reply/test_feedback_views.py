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
    FEEDBACK_POLICY_NOTE,
    FEEDBACK_UP_CUSTOM_ID,
    FEEDBACK_DOWN_CUSTOM_ID,
    FEEDBACK_REGEN_CUSTOM_ID,
    ReplyFeedbackView,
    UpvoteConfirmView,
    DownvoteReasonView,
    DownvoteReasonModal,
    _ReasonButton,
    _CancelVoteButton,
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
from tests.helpers.ask_interaction import _interaction as _ask_interaction


def _feedback_interaction(
    reply_message_id: int, user_id: int = 100, display_name: str = "Alice"
) -> FakeInteraction:
    """Builds a component interaction targeting `reply_message_id`."""
    message = FakeDiscordMessage()
    message.id = reply_message_id
    return FakeInteraction(
        user=FakeUser(user_id=user_id, name=f"user{user_id}", display_name=display_name),
        message=message,
    )


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

    up = _feedback_interaction(reply_message_id=7001, display_name="First Nick")
    await up_btn.callback(as_interaction(fake=up))
    assert up.response.deferred is True
    assert up.response.deferred_ephemeral is True
    assert up.response.deferred_with_message is True
    assert up.edits[-1]["content"] == f"已記錄 👍\n{FEEDBACK_POLICY_NOTE}"
    up_view = up.edits[-1].get("view")
    assert isinstance(up_view, UpvoteConfirmView)
    assert len(up_view.children) == 1

    async with open_session() as session:
        up_row = await session.get(entity=ReplyFeedbackRow, ident=(7001, 100))
    assert up_row is not None
    assert up_row.user_name == "user100"
    assert up_row.user_display_name == "First Nick"

    down = _feedback_interaction(reply_message_id=7001, display_name="Second Nick")
    await down_btn.callback(as_interaction(fake=down))
    assert "已改為 👎" in down.edits[-1]["content"]
    assert FEEDBACK_POLICY_NOTE in down.edits[-1]["content"]
    reason_view = down.edits[-1].get("view")
    assert isinstance(reason_view, DownvoteReasonView)
    assert len(reason_view.children) == 6

    async with open_session() as session:
        down_row = await session.get(entity=ReplyFeedbackRow, ident=(7001, 100))
    assert down_row is not None
    assert down_row.user_display_name == "Second Nick"

    cancel = _feedback_interaction(reply_message_id=7001)
    await down_btn.callback(as_interaction(fake=cancel))
    assert cancel.edits[-1]["content"] == "已取消評分"
    assert FEEDBACK_POLICY_NOTE not in cancel.edits[-1]["content"]


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
    assert pick.edits[-1]["content"] == f"已記錄 👎（資訊錯誤）\n{FEEDBACK_POLICY_NOTE}"

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
    assert (
        submit.edits[-1]["content"]
        == f"已記錄 👎（其他：沒有看我附的圖片）\n{FEEDBACK_POLICY_NOTE}"
    )
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
    assert f"\n{FEEDBACK_POLICY_NOTE}" in (message.replies[0].content or "")
    async with open_session() as session:
        rows = list((await session.execute(select(ReplyTurnRow))).scalars())
    assert len(rows) == 1
    assert rows[0].route == "QA"
    assert rows[0].model == "gemini-4-hydrogen"
    assert rows[0].effort == "high"
    assert rows[0].author_name == message.author.name
    assert rows[0].author_display_name == message.author.display_name
    assert rows[0].question == "幫我解釋 WAL"
    assert rows[0].answer == "WAL 是預寫日誌"

    # With kill-switch off, no buttons or policy note are attached and no row is inserted.
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
    assert FEEDBACK_POLICY_NOTE not in (message_off.replies[0].content or "")
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
    assert f"\n{FEEDBACK_POLICY_NOTE}" in (delivered.content or "")
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
    custom_ids = [child.custom_id for child in added[0].children if isinstance(child, Button)]
    assert custom_ids == [FEEDBACK_UP_CUSTOM_ID, FEEDBACK_DOWN_CUSTOM_ID, FEEDBACK_REGEN_CUSTOM_ID]
    assert added[0]._on_regen is not None


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
    """`ReplyFeedbackView` stays persistent for `bot.add_view` with or without the regen button."""
    without_regen = ReplyFeedbackView(allow_regen=False)
    assert without_regen.is_persistent() is True
    assert without_regen.prevent_update is False
    assert [c.custom_id for c in without_regen.children if isinstance(c, Button)] == [
        FEEDBACK_UP_CUSTOM_ID,
        FEEDBACK_DOWN_CUSTOM_ID,
    ]

    called: list[int] = []

    async def on_regen(interaction: object) -> None:
        called.append(getattr(getattr(interaction, "message", None), "id", 0))

    with_regen = ReplyFeedbackView(allow_regen=True, on_regen=on_regen)
    assert with_regen.is_persistent() is True
    assert with_regen.prevent_update is False
    assert [c.custom_id for c in with_regen.children if isinstance(c, Button)] == [
        FEEDBACK_UP_CUSTOM_ID,
        FEEDBACK_DOWN_CUSTOM_ID,
        FEEDBACK_REGEN_CUSTOM_ID,
    ]

    regen_btn = _button_by_id(view=with_regen, custom_id=FEEDBACK_REGEN_CUSTOM_ID)
    click = _feedback_interaction(reply_message_id=7099)
    await regen_btn.callback(as_interaction(fake=click))
    assert called == [7099]

    unwired = ReplyFeedbackView(allow_regen=True, on_regen=None)
    unwired_click = _feedback_interaction(reply_message_id=7098)
    await _button_by_id(view=unwired, custom_id=FEEDBACK_REGEN_CUSTOM_ID).callback(
        as_interaction(fake=unwired_click)
    )
    assert unwired_click.response.sent == [
        {"content": "目前無法重新生成這則回覆", "ephemeral": True}
    ]


async def test_answer_turn_attaches_regen_button_only_for_gateway_qa_under_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway QA attaches `🔄` below `max_regens`; `/ask`, IMAGE, and capped turns omit it."""
    cog = _cog()
    cog.config.inline_voice_enabled = False
    cog.config.inline_image_enabled = False
    cog.config.inline_music_enabled = False
    cog.config.inline_video_enabled = False

    async def fake_create(**_kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        completed = _completed_event(input_tokens=10, output_tokens=5)
        completed.response.model = "gemini-4-neon"
        return _stream_events_from(events=[_text_event(delta="ok"), completed])

    monkeypatch.setattr(cog.openai_client.responses, "create", fake_create)
    monkeypatch.setattr(AnswerTurn, "_schedule_memory_updates", lambda *_args, **_kwargs: None)

    def _has_regen(view: object) -> bool:
        assert isinstance(view, ReplyFeedbackView)
        return any(
            isinstance(child, Button) and child.custom_id == FEEDBACK_REGEN_CUSTOM_ID
            for child in view.children
        )

    # Gateway QA regen_count=0 -> has 🔄; regen_count=1 (blind test off, max=1) -> omits 🔄.
    msg0 = FakeMessage(content="<@999> q0")
    turn0 = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=msg0)),
    )
    await turn0.stream_answer(system_prompt="Be concise.", context=ReplyContext(), regen_count=0)
    assert _has_regen(view=msg0.replies[0].view) is True

    msg1 = FakeMessage(content="<@999> q1")
    turn1 = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=msg1)),
    )
    await turn1.stream_answer(system_prompt="Be concise.", context=ReplyContext(), regen_count=1)
    assert _has_regen(view=msg1.replies[0].view) is False

    # Blind test on (3 candidates -> max_regens=2): regen_count=1 still shows 🔄, regen_count=2 hides it.
    cog.config.slow_model_blind_test_enabled = True
    msg_b1 = FakeMessage(content="<@999> qb1")
    turn_b1 = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=msg_b1)),
    )
    await turn_b1.stream_answer(system_prompt="Be concise.", context=ReplyContext(), regen_count=1)
    assert _has_regen(view=msg_b1.replies[0].view) is True

    msg_b2 = FakeMessage(content="<@999> qb2")
    turn_b2 = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=msg_b2)),
    )
    await turn_b2.stream_answer(system_prompt="Be concise.", context=ReplyContext(), regen_count=2)
    assert _has_regen(view=msg_b2.replies[0].view) is False

    # `/ask` omits 🔄 even at regen_count=0.
    ask_interaction = _ask_interaction()
    await ask_interaction.response.defer()
    ask_surface = TurnSurface.for_interaction(interaction=ask_interaction, question="ask q")
    ask_turn = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=ask_surface,
    )
    await ask_turn.stream_answer(
        system_prompt="Be concise.", context=ReplyContext(), regen_count=0
    )
    assert _has_regen(view=ask_interaction.edits[-1]["view"]) is False


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


async def test_cancel_vote_button_clears_matching_vote_and_rejects_stale_clicks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`取消評分` clears matching 👍/👎 votes, logs `action='clear'`, and rejects stale buttons."""
    await _seed_turn(reply_message_id=7010, author_id=100)
    feedback_logs: list[dict[str, object]] = []
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.feedback_views.logfire.info",
        lambda msg, **kwargs: (
            feedback_logs.append(kwargs) if msg == "gen_reply feedback" else None
        ),
    )
    view = ReplyFeedbackView()
    up_btn = _button_by_id(view=view, custom_id=FEEDBACK_UP_CUSTOM_ID)
    down_btn = _button_by_id(view=view, custom_id=FEEDBACK_DOWN_CUSTOM_ID)

    # Upvote -> click `取消評分` on row 0 clears the upvote and logs action="clear".
    up = _feedback_interaction(reply_message_id=7010)
    await up_btn.callback(as_interaction(fake=up))
    up_view = cast("UpvoteConfirmView", up.edits[-1]["view"])
    up_cancel = next(c for c in up_view.children if isinstance(c, _CancelVoteButton))
    assert up_cancel.row == 0
    assert up_cancel.label == "取消評分"
    cancel_up = _feedback_interaction(reply_message_id=7010)
    await up_cancel.callback(as_interaction(fake=cancel_up))
    assert cancel_up.response.deferred is True
    assert cancel_up.edits[-1]["content"] == "已取消評分"
    assert cancel_up.edits[-1]["view"] is None
    assert feedback_logs[-1]["action"] == "clear"
    assert feedback_logs[-1]["vote"] == "up"
    async with open_session() as session:
        assert await session.get(entity=ReplyFeedbackRow, ident=(7010, 100)) is None

    # Downvote -> click `取消評分` on row 1 clears the downvote.
    down = _feedback_interaction(reply_message_id=7010)
    await down_btn.callback(as_interaction(fake=down))
    down_view = cast("DownvoteReasonView", down.edits[-1]["view"])
    down_cancel = next(c for c in down_view.children if isinstance(c, _CancelVoteButton))
    assert down_cancel.row == 1
    cancel_down = _feedback_interaction(reply_message_id=7010)
    await down_cancel.callback(as_interaction(fake=cancel_down))
    assert cancel_down.edits[-1]["content"] == "已取消評分"
    assert cancel_down.edits[-1]["view"] is None
    assert feedback_logs[-1]["action"] == "clear"
    assert feedback_logs[-1]["vote"] == "down"

    # Stale cancel button after switching from 👍 to 👎 must not delete the new 👎 vote.
    up2 = _feedback_interaction(reply_message_id=7010)
    await up_btn.callback(as_interaction(fake=up2))
    stale_up_view = cast("UpvoteConfirmView", up2.edits[-1]["view"])
    stale_up_cancel = next(c for c in stale_up_view.children if isinstance(c, _CancelVoteButton))
    switch_down = _feedback_interaction(reply_message_id=7010)
    await down_btn.callback(as_interaction(fake=switch_down))
    stale_click = _feedback_interaction(reply_message_id=7010)
    await stale_up_cancel.callback(as_interaction(fake=stale_click))
    assert stale_click.edits[-1]["content"] == "你已變更或取消這則評分"
    assert stale_click.edits[-1]["view"] is None
    async with open_session() as session:
        row = await session.get(entity=ReplyFeedbackRow, ident=(7010, 100))
    assert row is not None
    assert row.vote == "down"
