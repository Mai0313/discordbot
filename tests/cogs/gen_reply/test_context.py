"""Tests for the context a reply reads: history, references and their budgets."""

from __future__ import annotations

from typing import cast
import asyncio

import pytest
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.cogs.gen_reply.context import (
    ReplyContext,
    reference_header,
    trim_history_to_budget,
    history_media_over_budget,
)
from discordbot.typings.context_budgets import (
    HISTORY_CHAR_BUDGET,
    MAX_HISTORY_MEDIA_PARTS,
    HISTORY_PER_MESSAGE_OVERHEAD,
)
from discordbot.cogs.gen_reply.capabilities import render_capabilities_block

from tests.helpers.casting import as_message, step_dicts
from tests.helpers.llm_input import iter_text_blocks
from tests.helpers.reply_fakes import (
    FakeReply,
    FakeAuthor,
    FakeMessage,
    FakeAttachment,
    _cog,
    _answer,
    _recorded,
    _image_post,
    _config_stub,
    _context_builder,
    _install_streamer,
)
from tests.helpers.logfire_capture import capture_logs


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
    messages = [
        FakeMessage(content=f"{i}:{body}", author=FakeAuthor(user_id=1)) for i in range(fits + 20)
    ]

    kept = trim_history_to_budget(messages=[as_message(fake=m) for m in messages])

    assert len(kept) <= fits
    # order-contract: history is fed to the model oldest-first, and the tail is what is kept.
    assert [m.content for m in kept] == [m.content for m in messages[len(messages) - len(kept) :]]


def test_trim_history_keeps_one_message_that_alone_exceeds_the_budget() -> None:
    """A single oversized post must not reduce history to nothing."""
    huge = FakeMessage(content="y" * (HISTORY_CHAR_BUDGET * 3), author=FakeAuthor(user_id=1))

    kept = trim_history_to_budget(messages=[as_message(fake=huge)])

    assert len(kept) == 1


def test_trim_history_charges_an_attachment_only_message() -> None:
    """Empty `content` still costs, so a run of image posts cannot overshoot the budget."""
    blanks = [FakeMessage(content="", author=FakeAuthor(user_id=1)) for _ in range(2000)]

    kept = trim_history_to_budget(messages=[as_message(fake=m) for m in blanks])

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


def test_only_the_replied_to_message_claims_the_current_message_is_about_it() -> None:
    """The one Reference Message block claims the Current Message, and nothing competes with it.

    A reply renders exactly one of these (`replied_to_message`), so the attachment sentence has
    no sibling. An ancestor block's `An earlier message in the reply thread` wording must not
    come back: a second block asserting it is what the Current Message is about is the
    ambiguity this sentence exists to remove.
    """
    header = reference_header(
        ref=as_message(fake=FakeMessage(content="原訊息", author=FakeAuthor(user_id=4)))
    )

    text = next(text for _role, text in iter_text_blocks(request=[header]))
    assert "it is the primary context for the Current Message below" in text
    assert "that something is here, this message's attachments included" in text
    assert "An earlier message in the reply thread" not in text
