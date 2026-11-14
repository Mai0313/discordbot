"""Tests for the context a reply reads: history, references and their budgets."""

from __future__ import annotations

from typing import cast
import asyncio
from datetime import timedelta

import pytest
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.cogs.gen_reply.input import shorten_earlier_reply
from discordbot.cogs.gen_reply.context import (
    ReplyContext,
    ReplyContextBuilder,
    gap_separator,
    latest_reply_ids,
    trim_history_to_budget,
    history_media_over_budget,
)
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.typings.context_budgets import (
    HISTORY_CHAR_BUDGET,
    MAX_HISTORY_MEDIA_PARTS,
    HISTORY_PER_MESSAGE_OVERHEAD,
    HISTORY_EARLIER_REPLY_MAX_CHARS,
)
from discordbot.cogs.gen_reply.ask_store import record_ask_turn
from discordbot.cogs.gen_reply.capabilities import render_capabilities_block

from tests.helpers.casting import as_message, step_dicts
from tests.helpers.reply_fakes import (
    FAKE_MESSAGE_CREATED_AT,
    FakeReply,
    FakeAuthor,
    FakeMessage,
    FakeReference,
    FakeAttachment,
    _cog,
    _answer,
    _recorded,
    _image_post,
    _config_stub,
    _context_builder,
    _install_streamer,
)
from tests.helpers.ask_interaction import ASKER_ID, CHANNEL_ID, ASK_SNOWFLAKE, _interaction
from tests.helpers.logfire_capture import capture_logs

BOT_ID = 999
# What every finished reply ends with on Discord, and what marks a bot post as a reply at all.
FOOTER = "\n\n-# gemini-test (high) · ⬆ 1 ⬇ 2"


def _said(content: str, author_id: int = 1, minutes: float = 0) -> FakeMessage:
    """A message `minutes` after `FAKE_MESSAGE_CREATED_AT`, from the bot when `author_id` is `BOT_ID`."""
    message = FakeMessage(content=content, author=FakeAuthor(user_id=author_id))
    message.created_at = FAKE_MESSAGE_CREATED_AT + timedelta(minutes=minutes)
    return message


def _trim(messages: list[FakeMessage], current_minutes: float = 0) -> list[FakeMessage]:
    """Trims `messages` as the gateway does for a message sent `current_minutes` in."""
    return cast(
        "list[FakeMessage]",
        trim_history_to_budget(
            messages=[as_message(fake=m) for m in messages],
            bot_user_id=BOT_ID,
            current_at=FAKE_MESSAGE_CREATED_AT + timedelta(minutes=current_minutes),
        ),
    )


def _text(block: EasyInputMessageParam) -> str:
    """One rendered block's text, whether it was rendered as a string or as parts."""
    content = block["content"]
    if isinstance(content, str):
        return content
    return "".join(str(part.get("text", "")) for part in step_dicts(steps=content))


@pytest.mark.usefixtures("no_memory_review")
async def test_the_answer_and_the_persona_reply_read_the_context_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each request lays the context out in its fixed order, the current message last.

    The media persona reply carries neither the feature reference, the server memory nor a
    linked post.
    """
    cog = _cog()
    cog.config = _config_stub()
    _install_streamer(monkeypatch=monkeypatch)
    context = ReplyContext(
        hist_messages=[EasyInputMessageParam(role="user", content="history")],
        reference_messages=[EasyInputMessageParam(role="user", content="reference")],
        current_message=[EasyInputMessageParam(role="user", content="current")],
        server_memory_block=EasyInputMessageParam(role="assistant", content="server memory"),
        memory_block=EasyInputMessageParam(role="assistant", content="user memory"),
        tone_block=EasyInputMessageParam(role="assistant", content="tone"),
        link_blocks=[EasyInputMessageParam(role="system", content="linked post")],
    )

    async def built() -> ReplyContext:
        """Hands over the full context at once."""
        return context

    message = as_message(fake=FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1)))
    answer = _answer(cog=cog, message=message)
    await answer.stream_answer(system_prompt="SYS", context=context)
    await answer.stream_media_persona_reply(
        reply=as_message(fake=FakeReply()),
        context_task=asyncio.create_task(coro=built()),
        system_prompt="SYS",
        focus_part=ResponseInputImageParam(
            image_url="data:image/png;base64,", detail="auto", type="input_image"
        ),
        media_noun="image",
    )

    qa_input, persona_input = _recorded(cog).responses.create_inputs
    assert [cast("EasyInputMessageParam", item)["content"] for item in qa_input] == [
        render_capabilities_block(config=cog.config)["content"],
        "history",
        "server memory",
        "user memory",
        "tone",
        "reference",
        "linked post",
        "current",
    ]
    assert [cast("EasyInputMessageParam", item)["content"] for item in persona_input[:-1]] == [
        "history",
        "user memory",
        "tone",
        "reference",
        "current",
    ]


def test_trim_history_keeps_the_newest_messages_within_the_budget() -> None:
    """The budget drops the oldest context first and never cuts inside a message.

    Ordering is the contract being checked here, not just the count: history is handed
    over oldest-first and the answer needs the conversation nearest the question, so a trim
    that kept the wrong end would still pass a length assertion.
    """
    per_message = 100
    body = "x" * (per_message - HISTORY_PER_MESSAGE_OVERHEAD)
    fits = HISTORY_CHAR_BUDGET // per_message
    messages = [FakeMessage(content=body, author=FakeAuthor(user_id=1)) for _ in range(fits + 20)]

    kept = _trim(messages=messages)

    # order-contract: history is fed to the model oldest-first, and the exact tail is what is kept.
    assert [m.id for m in kept] == [m.id for m in messages[-fits:]]


def test_trim_history_keeps_one_message_that_alone_exceeds_the_budget() -> None:
    """A single oversized post must not reduce history to nothing."""
    huge = FakeMessage(content="y" * (HISTORY_CHAR_BUDGET * 3), author=FakeAuthor(user_id=1))

    kept = _trim(messages=[huge])

    assert len(kept) == 1


def test_trim_history_charges_an_attachment_only_message() -> None:
    """Empty `content` still costs, so a run of image posts cannot overshoot the budget."""
    blanks = [FakeMessage(content="", author=FakeAuthor(user_id=1)) for _ in range(2000)]

    kept = _trim(messages=blanks)

    assert len(kept) <= HISTORY_CHAR_BUDGET // HISTORY_PER_MESSAGE_OVERHEAD


def test_history_media_budget_refuses_every_older_post_once_one_is_refused() -> None:
    """The files that survive are an unbroken run ending at the newest post.

    The oldest post here needs one part and would fit the single slot the newest two leave
    unspent, so a budget that kept looking for something small enough would admit it. That is
    the case being pinned: admitting it would show the model an older attachment while a newer
    one rendered as a marker, which reads as files going missing at random rather than as a cap.
    """
    posts = [
        _image_post(index=0, count=1),
        _image_post(index=1, count=5),
        _image_post(index=2, count=MAX_HISTORY_MEDIA_PARTS - 1),
    ]

    over = history_media_over_budget(
        builder=_cog().toolkit.input_builder, hist_messages=[as_message(fake=m) for m in posts]
    )

    assert over == {posts[0].id: 1, posts[1].id: 5}


def test_history_media_budget_exempts_the_newest_post_that_carries_attachments() -> None:
    """One post of many images keeps its files rather than spending nothing at all."""
    post = _image_post(index=0, count=MAX_HISTORY_MEDIA_PARTS + 5)

    over = history_media_over_budget(
        builder=_cog().toolkit.input_builder, hist_messages=[as_message(fake=post)]
    )

    assert over == {}


def _document_post(index: int, count: int) -> FakeMessage:
    """A history message carrying `count` office documents, which no model accepts."""
    message = FakeMessage(content=f"docs {index}", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(
            filename=f"{index}-{n}.docx",
            content_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            attachment_id=index * 100 + n,
        )
        for n in range(count)
    ]
    return message


def test_history_media_budget_is_not_spent_by_files_that_will_be_dropped() -> None:
    """An attachment the modality gate drops must not cost an older post its images.

    The newest post is exempt from the cap and is what sets the running total, so counting a post
    of `MAX_HISTORY_MEDIA_PARTS` office documents would record the budget full while nothing was
    uploaded, and every older message would then be refused, leaving the turn with no media at
    all and nothing in the logs saying the budget had been spent on nothing.
    """
    posts = [_image_post(index=0, count=4), _document_post(index=1, count=MAX_HISTORY_MEDIA_PARTS)]

    over = history_media_over_budget(
        builder=_cog().toolkit.input_builder, hist_messages=[as_message(fake=m) for m in posts]
    )

    assert over == {}


def test_history_media_budget_counts_only_the_supported_half_of_a_mixed_post() -> None:
    """A post carrying both kinds spends what it will upload, not what it holds.

    The all-or-nothing cases either side of this one both pass a gate that counts the whole
    message whenever any part of it survives, so this is what says the count is per source.
    """
    mixed = _image_post(index=0, count=MAX_HISTORY_MEDIA_PARTS - 1)
    mixed.attachments = [*mixed.attachments, *_document_post(index=9, count=6).attachments]
    older = _image_post(index=1, count=1)

    over = history_media_over_budget(
        builder=_cog().toolkit.input_builder,
        hist_messages=[as_message(fake=older), as_message(fake=mixed)],
    )

    # The newest post spends one part short of the cap rather than counting its documents
    # too, so the older one still fits.
    assert over == {}


@pytest.mark.parametrize(
    ("file_api_enabled", "budget", "kept"),
    [("true", 5_000, 3), ("false", 5_000, 1), ("false", 2_000, 0)],
    ids=["files-api", "inline", "inline-newest-alone-over"],
)
async def test_render_history_holds_inlined_attachment_bytes_to_the_budget(
    monkeypatch: pytest.MonkeyPatch, file_api_enabled: str, budget: int, kept: int
) -> None:
    """History files that fit one at a time may not add up past the inline byte budget.

    The newest keep their files, and the post that overflows renders as markers along with every
    older one, the newest included when it alone is over: exempting it would leave one post large
    enough to fail every reply. Under the Files API a part is a short handle, so nothing is held.
    """
    monkeypatch.setattr(
        RuntimeModelCatalog,
        "slow_model",
        property(lambda _self: ModelSettings(name="gemini-test", effort="high")),
    )
    monkeypatch.setenv(name="FILE_API_ENABLED", value=file_api_enabled)
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities",
        lambda model_name: {"text", "image"},
    )
    monkeypatch.setattr("discordbot.cogs.gen_reply.context.MAX_HISTORY_INLINE_BYTES", budget)
    cog = _cog()
    posts = [FakeMessage(content=f"notes {i}", author=FakeAuthor(user_id=1)) for i in range(3)]
    for index, post in enumerate(posts):
        post.attachments = [
            FakeAttachment(filename=f"{index}.txt", payload=b"x" * 3_000, attachment_id=index)
        ]
    logged = capture_logs(monkeypatch=monkeypatch, level="info")

    rendered = await _context_builder(
        cog=cog, message=as_message(fake=FakeMessage())
    ).render_history(hist_messages=[as_message(fake=m) for m in posts])

    # rendered[0] is the history header; the rest follow the posts, oldest first.
    assert [
        step_dicts(steps=message["content"])[1:]
        == [{"type": "input_text", "text": "[attachment: file]"}]
        for message in rendered[1:]
    ] == [True] * (3 - kept) + [False] * kept
    assert [
        fields["payload_capped_messages"]
        for text, fields in logged
        if text == "gen_reply history render done"
    ] == [3 - kept]


def test_reply_context_message_list_orders_hist_ref_current() -> None:
    """message_list keeps transcript order: history, reference, current."""
    context = ReplyContext(
        hist_messages=[{"role": "system", "content": "hist"}],
        reference_messages=[{"role": "system", "content": "ref"}],
        current_message=[{"role": "user", "content": "now"}],
    )
    assert [part["content"] for part in context.message_list] == ["hist", "ref", "now"]


def test_an_earlier_bot_reply_costs_only_its_opening_so_more_of_the_chat_fits() -> None:
    """Six long replies would fill the budget three times over at full length.

    At their shortened length every message fits, the most recent reply whole.
    """
    messages: list[FakeMessage] = []
    for index in range(6):
        messages.append(_said(content=f"question {index}"))
        messages.append(_said(content="答" * 1300 + FOOTER, author_id=BOT_ID))
    messages.append(_said(content="and now?"))

    kept = _trim(messages=messages)

    assert [m.id for m in kept] == [m.id for m in messages]


def test_the_most_recent_reply_is_kept_whole_across_its_chunks_past_the_budget() -> None:
    """A reply split across two messages is not cut in half by the budget, even when it overshoots."""
    older = _said(content="older")
    first = _said(content="a" * 1990, author_id=BOT_ID)
    second = _said(content="b" * 1990 + FOOTER, author_id=BOT_ID)
    second.reference = FakeReference(resolved=first)
    newest = _said(content="c" * 500)

    kept = _trim(messages=[older, first, second, newest])

    assert [m.id for m in kept] == [first.id, second.id, newest.id]


def test_the_latest_reply_is_found_by_its_footer_even_when_the_footer_followed_alone() -> None:
    """A newer command output is not a reply, and a footer sent on its own still marks one."""
    earlier = _said(content="earlier" + FOOTER, author_id=BOT_ID)
    answer = _said(content="a" * 1990, author_id=BOT_ID)
    footer_alone = _said(content=FOOTER.lstrip("\n"), author_id=BOT_ID)
    footer_alone.reference = FakeReference(resolved=answer)
    command_output = _said(content="🎰 slots", author_id=BOT_ID)

    latest = latest_reply_ids(
        messages=[as_message(fake=m) for m in (earlier, answer, footer_alone, command_output)],
        bot_user_id=BOT_ID,
    )

    assert latest == {answer.id, footer_alone.id}


def test_trim_charges_the_gap_separator_a_message_brings() -> None:
    """The separator is rendered, so a message that fits only without it is left out."""
    separator = gap_separator(
        earlier=FAKE_MESSAGE_CREATED_AT, later=FAKE_MESSAGE_CREATED_AT + timedelta(hours=3)
    )
    assert separator is not None
    newest = _said(content="n" * (2000 - HISTORY_PER_MESSAGE_OVERHEAD), minutes=180)
    fits_without = HISTORY_CHAR_BUDGET - 2000 - HISTORY_PER_MESSAGE_OVERHEAD - len(separator) // 2
    older = _said(content="o" * fits_without)

    kept = _trim(messages=[older, newest], current_minutes=180)

    assert [m.id for m in kept] == [newest.id]


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [
        (29, None),
        (30, "==== 30 minutes later ===="),
        (75, "==== 1 hour later ===="),
        (170, "==== 3 hours later ===="),
        (60 * 47, "==== 47 hours later ===="),
        (60 * 24 * 5, "==== 5 days later ===="),
    ],
)
def test_a_gap_separator_names_only_the_time_that_passed(
    minutes: int, expected: str | None
) -> None:
    """Under half an hour says nothing; past it, the gap in the unit a person would use."""
    assert (
        gap_separator(
            earlier=FAKE_MESSAGE_CREATED_AT,
            later=FAKE_MESSAGE_CREATED_AT + timedelta(minutes=minutes),
        )
        == expected
    )


async def test_render_history_shortens_earlier_replies_and_marks_where_time_passed() -> None:
    """Earlier replies render as their opening, the latest whole, and long gaps get a separator.

    A newer bot post that is not a reply (a command's output) does not take the latest reply's
    place, and the last separator measures the gap to the message being answered.
    """
    earlier_reply = "早" * (HISTORY_EARLIER_REPLY_MAX_CHARS + 200)
    latest_reply = "晚" * (HISTORY_EARLIER_REPLY_MAX_CHARS + 200)
    history = [
        _said(content=earlier_reply + FOOTER, author_id=BOT_ID),
        _said(content="ok", minutes=10),
        _said(content=latest_reply + FOOTER, author_id=BOT_ID, minutes=180),
        _said(
            content="🎰 " + "x" * HISTORY_EARLIER_REPLY_MAX_CHARS, author_id=BOT_ID, minutes=181
        ),
    ]
    current = _said(content="<@999> next", minutes=225)

    rendered = await _context_builder(
        cog=_cog(bot_user_id=BOT_ID), message=as_message(fake=current)
    ).render_history(hist_messages=[as_message(fake=m) for m in history])

    assert [_text(block=block) for block in rendered[1:]] == [
        shorten_earlier_reply(text=earlier_reply),
        "Tester (tester) [id: 1]: ok",
        "==== 3 hours later ====",
        latest_reply,
        shorten_earlier_reply(text="🎰 " + "x" * HISTORY_EARLIER_REPLY_MAX_CHARS),
        "==== 44 minutes later ====",
    ]
    assert [block["role"] for block in rendered[1:]] == [
        "assistant",
        "user",
        "user",
        "assistant",
        "assistant",
        "user",
    ]
    assert shorten_earlier_reply(text=earlier_reply).endswith("[… rest of this reply omitted]")


async def test_an_ask_conversation_gets_the_same_shortening_and_separators() -> None:
    """Stored `/ask` turns rebuild with real send times, so the window reads the same there."""
    interaction = _interaction()
    hour_ago = ASK_SNOWFLAKE - ((60 * 60 * 1000) << 22)
    two_hours_ago = ASK_SNOWFLAKE - ((2 * 60 * 60 * 1000) << 22)
    long_answer = "長" * (HISTORY_EARLIER_REPLY_MAX_CHARS + 50)
    for message_id in (two_hours_ago, hour_ago):
        await record_ask_turn(
            channel_id=CHANNEL_ID,
            user_id=ASKER_ID,
            message_id=message_id,
            question="q",
            answer=long_answer + FOOTER,
        )
    builder = ReplyContextBuilder(
        toolkit=_cog(bot_user_id=BOT_ID).toolkit,
        surface=TurnSurface.for_interaction(interaction=interaction, question="now"),
    )

    rendered = await builder.render_history(hist_messages=await builder.fetch_history(limit=50))

    assert [_text(block=block) for block in rendered[1:]] == [
        "Asker (asker) [id: 4242]: q",
        shorten_earlier_reply(text=long_answer),
        "==== 1 hour later ====",
        "Asker (asker) [id: 4242]: q",
        long_answer,
        "==== 1 hour later ====",
    ]


async def test_an_earlier_reply_degraded_to_markers_is_still_shortened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The text-only re-render a payload-capped post gets keeps the opening-only form."""
    monkeypatch.setattr(
        RuntimeModelCatalog,
        "slow_model",
        property(lambda _self: ModelSettings(name="gemini-test", effort="high")),
    )
    monkeypatch.setenv(name="FILE_API_ENABLED", value="false")
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.input.get_supported_modalities",
        lambda model_name: {"text", "image"},
    )
    monkeypatch.setattr("discordbot.cogs.gen_reply.context.MAX_HISTORY_INLINE_BYTES", 2_000)
    reply = "早" * (HISTORY_EARLIER_REPLY_MAX_CHARS + 200)
    earlier = _said(content=reply, author_id=BOT_ID)
    earlier.attachments = [FakeAttachment(filename="0.txt", payload=b"x" * 3_000, attachment_id=0)]
    latest = _said(content="latest" + FOOTER, author_id=BOT_ID)

    rendered = await _context_builder(
        cog=_cog(bot_user_id=BOT_ID), message=as_message(fake=_said(content="hi"))
    ).render_history(hist_messages=[as_message(fake=m) for m in (earlier, latest)])

    assert _text(block=rendered[1]) == (
        f"Tester (tester) [id: {BOT_ID}]: {shorten_earlier_reply(text=reply)}[attachment: file]"
    )
