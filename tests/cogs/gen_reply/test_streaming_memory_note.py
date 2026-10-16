"""Tests for the memory note under a reply."""

from __future__ import annotations

import pytest

from discordbot.typings.memory import MemoryWriteSummary
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.cogs.gen_reply.answer import memory_report_for
from discordbot.cogs.gen_reply.streaming import MEMORY_PENDING_NOTE, DISCORD_MESSAGE_LIMIT
from discordbot.cogs.gen_reply.references import message_link_texts, authored_link_texts

from tests.helpers.casting import as_message
from tests.helpers.reply_fakes import (
    TEST_LLM_MODEL,
    FakeAuthor,
    FakeMessage,
    _stream,
    _streamer,
    _text_event,
    _media_builder,
    _completed_event,
    _stream_events_from,
)


async def test_set_memory_note_splices_before_the_usage_footer() -> None:
    """The memory note lands seconds after the answer and must not break the footer.

    It goes before the usage footer for the same reason the hosted-URL line does: appended
    after it, `USAGE_FOOTER_RE` could no longer strip the footer, and every later history
    render would carry the model / token / cost line inside the bot's own answer. Directly above
    the footer, the note comes off with it (#865).
    """
    message = FakeMessage()
    streamer = _streamer(message=message)
    await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[_text_event(delta="好喔"), _completed_event(input_tokens=3, output_tokens=4)]
        ),
    )
    await streamer.set_memory_note(line="-# ✏️ 記下了 使用者偏好繁體中文")

    content = message.replies[0].content or ""
    assert "記下了 使用者偏好繁體中文" in content
    assert content.index("記下了") < content.index(f"-# {TEST_LLM_MODEL}")
    assert USAGE_FOOTER_RE.sub("", content) == "好喔"


async def test_set_memory_note_declines_when_the_reply_is_already_full() -> None:
    """A reply with no room left keeps what it has rather than being edited into an overflow.

    This is also the chunked case: a reply chunks precisely when its content plus footer
    already passes the limit, so one guard covers both and the streamer needs no separate
    chunked flag.
    """
    message = FakeMessage()
    streamer = _streamer(message=message)
    await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[
                _text_event(delta="x" * (DISCORD_MESSAGE_LIMIT - 5)),
                _completed_event(input_tokens=3, output_tokens=4),
            ]
        ),
    )
    before = message.replies[0].content
    await streamer.set_memory_note(line="-# ✏️ 記下了 使用者偏好繁體中文")
    assert message.replies[0].content == before


async def test_the_outcome_note_replaces_the_pending_one() -> None:
    """One turn shows one memory note, not a running log of one.

    The pending note is a promise the outcome takes back, so it has to be removed rather than
    written under: two `-#` lines saying different things about the same turn read as two
    separate pieces of memory work.
    """
    message = FakeMessage()
    streamer = _streamer(message=message)
    await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[
                _text_event(delta="好喔<write-memory>他喜歡繁體中文</write-memory>"),
                _completed_event(input_tokens=3, output_tokens=4),
            ]
        ),
    )
    assert MEMORY_PENDING_NOTE in (message.replies[0].content or "")

    await streamer.set_memory_note(line="-# ✏️ 記下了 使用者偏好繁體中文")

    content = message.replies[0].content or ""
    assert MEMORY_PENDING_NOTE not in content
    assert content.count("-# ✏️") == 1


async def test_the_pending_note_survives_a_hosted_media_splice() -> None:
    """A hosted-URL line lands between the note and the footer, so the note is no longer last.

    `_finalize_media_edit` rebuilds the content as body + hosted-URL line + footer, which puts
    the pending note mid-string. Removing it off the end would then leave both notes on screen.
    """
    message = FakeMessage()
    streamer = _streamer(message=message)
    await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[
                _text_event(delta="好喔<write-memory>他喜歡繁體中文</write-memory>"),
                _completed_event(input_tokens=3, output_tokens=4),
            ]
        ),
    )
    await streamer._finalize_media_edit(
        reply=as_message(fake=message.replies[0]),
        files=[],
        hosted_urls=["https://media.example/x.mp4"],
    )

    await streamer.set_memory_note(line="-# ✏️ 記下了 使用者偏好繁體中文")

    content = message.replies[0].content or ""
    assert MEMORY_PENDING_NOTE not in content
    assert "https://media.example/x.mp4" in content
    assert content.count("-# ✏️") == 1


async def test_the_pending_note_never_reaches_the_answer_text() -> None:
    """`full_reply` is the transcript the memory reviewer reads and history renders later.

    The note is chrome the bot added, not something it said. `USAGE_FOOTER_RE` cannot be relied
    on to remove it downstream either: it takes the note only while the note sits directly above
    the footer, which a hosted-URL line breaks.
    """
    message = FakeMessage()
    streamer = _streamer(message=message)
    full_reply = await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[
                _text_event(delta="好喔<write-memory>他喜歡繁體中文</write-memory>"),
                _completed_event(input_tokens=3, output_tokens=4),
            ]
        ),
    )

    assert MEMORY_PENDING_NOTE in (message.replies[0].content or "")
    assert MEMORY_PENDING_NOTE not in full_reply
    assert USAGE_FOOTER_RE.sub("", full_reply) == "好喔"


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ("好，我不會再提了<forget-memory>我住在台中</forget-memory>", "好，我不會再提了"),
        # No prose, so the message Discord stores opens on the note.
        ("<forget-memory>我住在台中</forget-memory>", ""),
    ],
    ids=["with_prose", "note_only"],
)
async def test_a_memory_note_never_reaches_a_later_turns_history(
    delta: str, expected: str
) -> None:
    """The memory note comes off wherever the bot's own reply is read back.

    A gateway turn re-reads the reply off Discord rather than `full_reply`, and the forget line
    quotes what the user asked to drop, so a note left on would hand the forgotten item back to
    the answer model and both memory reviews on every later turn (#865).
    """
    message = FakeMessage()
    streamer = _streamer(message=message)
    await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[_text_event(delta=delta), _completed_event(input_tokens=3, output_tokens=4)]
        ),
    )
    await memory_report_for(streamer=streamer)(
        MemoryWriteSummary(remembered=("使用者偏好繁體中文",), forgotten=("我住在台中",))
    )
    on_screen = message.replies[0].content or ""
    assert "我住在台中" in on_screen, "the note never landed, so this proves nothing"

    # Discord trims a message's leading and trailing whitespace before storing it.
    bot_reply = as_message(
        fake=FakeMessage(content=on_screen.strip(), author=FakeAuthor(bot=True, user_id=999))
    )
    assert await _media_builder().get_cleaned_content(message=bot_reply) == expected
    spans = [
        *message_link_texts(message=bot_reply, strip_usage_footer=True),
        *authored_link_texts(message=bot_reply),
    ]
    assert all("使用者偏好繁體中文" not in span and "台中" not in span for span in spans)


@pytest.mark.parametrize(
    ("carries_turn_notices", "delta", "expected"),
    [
        (True, "好喔<write-memory>他喜歡繁體中文</write-memory>", True),
        (True, "好喔<forget-memory>他討厭貓</forget-memory>", True),
        (True, "好喔", False),
        (False, "好喔<write-memory>他喜歡繁體中文</write-memory>", False),
    ],
    ids=["a-write-marker", "a-forget-marker", "no-marker", "a-media-persona-reply"],
)
async def test_the_pending_note_is_written_only_when_something_can_take_it_back(
    carries_turn_notices: bool, delta: str, expected: bool
) -> None:
    """A promise of more to come must never outlive the thing that would withdraw it.

    A media persona reply is the case that makes this a guard rather than a formality: markers
    are extracted on every route, but only the QA route schedules memory, so that streamer would
    caption a delivered image with `正在整理記憶⋯` and nothing would ever replace it.
    """
    message = FakeMessage()
    streamer = _streamer(message=message, carries_turn_notices=carries_turn_notices)
    await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[_text_event(delta=delta), _completed_event(input_tokens=3, output_tokens=4)]
        ),
    )

    assert (MEMORY_PENDING_NOTE in (message.replies[0].content or "")) is expected


async def test_an_outcome_too_long_to_splice_withdraws_the_pending_note() -> None:
    """A reply can have room for the promise and none for the answer that replaces it.

    The pending note is a dozen characters; the outcome names what was recorded and runs
    several times that. Declining the edit would leave `正在整理記憶⋯` standing for good, which
    is the one outcome worse than showing nothing at all.
    """
    message = FakeMessage()
    streamer = _streamer(message=message)
    await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[
                _text_event(delta="x" * (DISCORD_MESSAGE_LIMIT - 90)),
                _text_event(delta="<write-memory>他喜歡繁體中文</write-memory>"),
                _completed_event(input_tokens=3, output_tokens=4),
            ]
        ),
    )
    assert MEMORY_PENDING_NOTE in (message.replies[0].content or "")

    await streamer.set_memory_note(line=f"-# ✏️ 記下了 {'長' * 60}")

    content = message.replies[0].content or ""
    assert MEMORY_PENDING_NOTE not in content
    assert "記下了" not in content
    assert USAGE_FOOTER_RE.search(content) is not None


async def test_no_pending_note_on_a_reply_that_chunks() -> None:
    """A chunked reply's footer lives on a follow-up while the note would edit the parent.

    So the outcome could never replace it: `set_memory_note` edits `self.reply`, and the note
    it is looking for is on another message entirely.
    """
    message = FakeMessage()
    streamer = _streamer(message=message)
    await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[
                _text_event(delta="x" * DISCORD_MESSAGE_LIMIT),
                _text_event(delta="<write-memory>他喜歡繁體中文</write-memory>"),
                _completed_event(input_tokens=3, output_tokens=4),
            ]
        ),
    )

    # Each overflow chunk is a reply to the previous chunk, so the whole chain has to be walked:
    # `message.replies` alone holds the parent, and the note would be on the tail chunk with the
    # footer, where checking only the parent would never see it.
    chunks = [message.replies[0]]
    while chunks[-1].replies:
        chunks.append(chunks[-1].replies[-1])
    assert len(chunks) > 1, "the reply did not chunk, so this test proves nothing"
    assert all(MEMORY_PENDING_NOTE not in (chunk.content or "") for chunk in chunks)
