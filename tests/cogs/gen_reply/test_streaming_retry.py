"""Tests for retrying a failed answer stream and landing its failure."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
import asyncio

# openai 3.x builds its exceptions on httpx2, so a request or response handed to one has to
# come from there. google-genai is still on httpx 0.x, and both live in the environment.
import httpx2
from openai import APIError, BadRequestError
import pytest
from nextcord import Embed

from discordbot.cogs.gen_reply import streaming as streaming_module
from discordbot.typings.timeouts import ANSWER_STREAM_MAX_ATTEMPTS
from discordbot.cogs.gen_reply.surface import WINDOW_EXPIRED_NOTICE
from discordbot.cogs.gen_reply.pipeline import ReplyPipeline
from discordbot.cogs.gen_reply.streaming import ResponseStreamer, stream_answer_with_retry
from discordbot.cogs.gen_reply.status_marks import RETRY_HINT_EMOJI

from tests.helpers.casting import as_message, make_not_found
from tests.helpers.reply_fakes import (
    FakeReply,
    FakeAuthor,
    FakeMessage,
    _cog,
    _streamer,
    _text_event,
    _stream_events,
    _completed_event,
    _expiring_surface,
    _no_retry_backoff,
    _stream_events_from,
    _mid_stream_unavailable,
    _stream_events_then_raise,
)
from tests.helpers.link_sources import hosting_off_planner

if TYPE_CHECKING:
    from types import SimpleNamespace
    from collections.abc import AsyncIterator

    from openai.types.responses import ResponseStreamEvent


def _paced_stream_events(
    events: list[SimpleNamespace], pause: float
) -> AsyncIterator[ResponseStreamEvent]:
    """Yields the given events with a gap between them, so the preview editor gets a tick."""

    async def _iter() -> AsyncIterator[SimpleNamespace]:
        for index, event in enumerate(events):
            if index:
                await asyncio.sleep(pause)
            yield event

    return cast("AsyncIterator[ResponseStreamEvent]", _iter())


async def test_an_answer_stream_outliving_the_ask_window_says_so_while_it_can() -> None:
    """A stalled `/ask` answer gives up inside the margin, so the error embed still lands (#628).

    Past the token the answer and the notice both 404, so retrying on would only trade the
    reported failure for a thinking state that never resolves.
    """
    message = FakeMessage()
    streamer = ResponseStreamer(
        message=message,
        surface=_expiring_surface(message=as_message(fake=message), seconds_left=0.05),
        media_delivery=hosting_off_planner(),
    )

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        """Never answers, as a request stalled before its first delta does."""
        await asyncio.sleep(60)
        return _stream_events()

    with pytest.raises(TimeoutError) as raised:
        await asyncio.wait_for(
            stream_answer_with_retry(
                streamer=streamer, open_stream=open_stream, message_id=message.id
            ),
            timeout=5,
        )

    assert str(raised.value) == WINDOW_EXPIRED_NOTICE


async def test_an_answer_streams_own_timeout_is_not_blamed_on_the_ask_window() -> None:
    """A transport timeout inside a healthy window keeps its own failure."""
    message = FakeMessage()
    streamer = ResponseStreamer(
        message=message,
        surface=_expiring_surface(message=as_message(fake=message), seconds_left=300),
        media_delivery=hosting_off_planner(),
    )

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        """Fails the way a bound of the request's own does."""
        raise TimeoutError

    with pytest.raises(TimeoutError) as raised:
        await stream_answer_with_retry(
            streamer=streamer, open_stream=open_stream, message_id=message.id
        )

    assert str(raised.value) != WINDOW_EXPIRED_NOTICE


async def test_a_media_persona_reply_may_write_into_the_ask_margin() -> None:
    """The persona reply has no notice to save the margin for, so the window does not cut it."""
    message = FakeMessage()
    streamer = ResponseStreamer(
        message=message,
        surface=_expiring_surface(message=as_message(fake=message), seconds_left=0.0),
        media_delivery=hosting_off_planner(),
        reply=as_message(fake=FakeReply()),
        carries_turn_notices=False,
    )

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        """Answers after a pause the window would not have allowed."""
        await asyncio.sleep(0.05)
        return _stream_events()

    reply = await stream_answer_with_retry(
        streamer=streamer, open_stream=open_stream, message_id=message.id
    )

    assert reply


async def test_a_retried_answer_stream_replaces_the_dead_attempt_and_keeps_previewing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 503 mid-answer re-opens the stream onto the same message without doubling the text.

    Both halves matter. The dead attempt's partial text must not survive into the finished
    reply, and the preview editor must live through the reset -- `consume`'s finally stops it by
    SETTING an event that `_preview_editor` reads before its first tick, so without clearing it
    the retry streams blind and the stale preview sits frozen until the final write.
    """
    _no_retry_backoff(monkeypatch=monkeypatch)
    message = FakeMessage()
    streamer = _streamer(message=message, preview_interval_seconds=0.01)
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        if opened == 1:
            return _stream_events_then_raise(
                events=[_text_event(delta="half a sentence")], error=_mid_stream_unavailable()
            )
        # Paced so the editor gets at least one tick to write on the SECOND attempt.
        return _paced_stream_events(
            events=[
                _text_event(delta="the whole answer"),
                _completed_event(input_tokens=1, output_tokens=2),
            ],
            pause=0.05,
        )

    reply = await stream_answer_with_retry(
        streamer=streamer, open_stream=open_stream, message_id=message.id
    )

    assert opened == 2
    assert streamer.attempts == 2
    assert reply.startswith("the whole answer")
    assert "half a sentence" not in reply
    # One Discord message rather than a second one beside the dead attempt's text, and it was
    # created by a PREVIEW write and edited afterwards. An empty `edits` would mean the final
    # write created it, i.e. the retry streamed with a dead editor while the user watched a
    # frozen message the whole way through.
    assert len(message.replies) == 1
    assert message.replies[0].edits


async def test_a_non_retryable_answer_failure_never_re_opens_the_stream() -> None:
    """A refusal is the provider answering, so it surfaces on the first attempt."""
    message = FakeMessage()
    streamer = _streamer(message=message)
    opened = 0
    request = httpx2.Request(method="POST", url="http://proxy/v1/responses")
    refusal = BadRequestError(
        "blocked", response=httpx2.Response(status_code=400, request=request, json={}), body=None
    )

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        return _stream_events_then_raise(events=[], error=refusal)

    with pytest.raises(BadRequestError):
        await stream_answer_with_retry(
            streamer=streamer, open_stream=open_stream, message_id=message.id
        )

    assert opened == 1
    assert streamer.attempts == 1


async def test_an_exhausted_answer_retry_raises_the_provider_error_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`reraise` keeps the outer error path showing the provider failure, not a retry wrapper."""
    _no_retry_backoff(monkeypatch=monkeypatch)
    message = FakeMessage()
    streamer = _streamer(message=message)
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        return _stream_events_then_raise(events=[], error=_mid_stream_unavailable())

    with pytest.raises(APIError, match="MidStreamFallbackError"):
        await stream_answer_with_retry(
            streamer=streamer, open_stream=open_stream, message_id=message.id
        )

    assert opened == ANSWER_STREAM_MAX_ATTEMPTS
    assert streamer.attempts == ANSWER_STREAM_MAX_ATTEMPTS


async def test_a_retry_tells_the_user_it_is_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    """A silent retry is indistinguishable from a model that is just thinking slowly.

    The reaction is the half that always lands; the notice only takes over a reply that is
    already on screen, where what it replaces is the dead attempt's half-sentence.
    """
    _no_retry_backoff(monkeypatch=monkeypatch)
    message = FakeMessage()
    streamer = _streamer(message=message, reply=as_message(fake=FakeReply()))
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        if opened == 1:
            return _stream_events_then_raise(events=[], error=_mid_stream_unavailable())
        return _stream_events_from(
            events=[_text_event(delta="done"), _completed_event(input_tokens=1, output_tokens=2)]
        )

    await stream_answer_with_retry(
        streamer=streamer, open_stream=open_stream, message_id=message.id
    )

    assert RETRY_HINT_EMOJI in message.added_reactions
    reply = cast("FakeReply", streamer.reply)
    assert RETRY_HINT_EMOJI in reply.edits[0]
    # And the notice is transient: the finished answer takes the message back.
    assert (reply.content or "").startswith("done")


async def test_a_spent_retry_takes_its_own_notice_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Retrying...` promises another attempt; with none left it must not outlive the turn.

    Otherwise the turn ends with one message saying work is in flight beside the error embed
    saying it is not.
    """
    _no_retry_backoff(monkeypatch=monkeypatch)
    message = FakeMessage()
    reply = FakeReply()
    streamer = _streamer(message=message, reply=as_message(fake=reply))

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        return _stream_events_then_raise(events=[], error=_mid_stream_unavailable())

    with pytest.raises(APIError):
        await stream_answer_with_retry(
            streamer=streamer, open_stream=open_stream, message_id=message.id
        )

    assert reply.deleted is True
    assert streamer.reply is None
    # And with the notice gone there is nothing left to land the failure on, so the pipeline's
    # error path is told to post it fresh.
    assert await streamer.land_failure(embed=Embed(title="Something went wrong")) is False


async def test_a_spent_retry_keeps_text_the_last_attempt_managed_to_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a message that is still nothing but the notice goes; real text is the better residue."""
    _no_retry_backoff(monkeypatch=monkeypatch)
    message = FakeMessage()
    reply = FakeReply()
    streamer = _streamer(
        message=message, reply=as_message(fake=reply), preview_interval_seconds=0.01
    )

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        # Paints text, gives the editor a tick, then dies -- on every attempt.
        return _stream_events_then_raise(
            events=[_text_event(delta="partial answer")],
            error=_mid_stream_unavailable(),
            pause=0.05,
        )

    with pytest.raises(APIError):
        await stream_answer_with_retry(
            streamer=streamer, open_stream=open_stream, message_id=message.id
        )

    assert reply.deleted is False
    assert "partial answer" in (reply.edits[-1] if reply.edits else "")


async def test_a_retry_with_nothing_on_screen_yet_leaves_no_notice_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creating a reply just to say "Retrying" would orphan it on the turns that then fail."""
    _no_retry_backoff(monkeypatch=monkeypatch)
    message = FakeMessage()
    streamer = _streamer(message=message)

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        return _stream_events_then_raise(events=[], error=_mid_stream_unavailable())

    with pytest.raises(APIError):
        await stream_answer_with_retry(
            streamer=streamer, open_stream=open_stream, message_id=message.id
        )

    assert RETRY_HINT_EMOJI in message.added_reactions
    assert message.replies == []


async def test_a_failed_answer_lands_its_error_on_the_reply_it_was_streaming_into(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn that painted something before it died ends as ONE message, not two.

    The failure surfaces in `on_message`, several frames above the streamer that owns the
    reply handle, so this drives the real helper under the real error path: the pipeline is
    stubbed down to the one answer stream, and everything between the publish and the edit is
    the production code.
    """
    _no_retry_backoff(monkeypatch=monkeypatch)
    cog = _cog()
    message = FakeMessage(content="<@999> explain", author=FakeAuthor(user_id=1))
    reply = FakeReply()

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        # Paced so the preview editor gets a tick in: text nobody ever saw is a withdrawn
        # retry notice, which is the other case entirely.
        return _stream_events_then_raise(
            events=[_text_event(delta="half a sentence")],
            error=_mid_stream_unavailable(),
            pause=0.05,
        )

    async def failing_answer(self: object, **kwargs: object) -> None:
        """Streams half an answer onto a reply already on screen, then fails every attempt."""
        del kwargs
        streamer = _streamer(
            message=message, reply=as_message(fake=reply), preview_interval_seconds=0.01
        )
        await stream_answer_with_retry(
            streamer=streamer, open_stream=open_stream, message_id=message.id
        )

    monkeypatch.setattr(ReplyPipeline, "run", failing_answer)
    await cog.on_message(message=as_message(fake=message))

    # No second message beside the half-written one...
    assert message.replies == []
    # ...which keeps what the model managed to say, with the embed under it saying it is
    # incomplete -- the truncated text alone reads as an answer that simply stopped.
    assert reply.content == "half a sentence"
    assert reply.embed is not None
    assert reply.embed.footer.text == "APIError"


async def test_a_failure_over_a_thinking_preview_clears_it() -> None:
    """The preview is a live glance at a model that has now stopped thinking, so it goes.

    Frozen above the error it reads as work still in flight, and unlike a partial answer there
    is nothing in it the user was reading. The empty string is the load-bearing half: the
    error embed rides a spacer file, and nextcord drops a `content=None` out of a multipart
    edit instead of clearing it, which would leave the preview exactly where it was.
    """
    message = FakeMessage()
    reply = FakeReply()
    reply.content = "-# <:message:1517560873000898860> Thinking..."
    streamer = _streamer(message=message, reply=as_message(fake=reply))
    streamer.reasoning_content = "weighing the options"

    assert await streamer.land_failure(embed=Embed(title="Something went wrong")) is True

    assert reply.content == ""
    assert reply.embed is not None


async def test_a_reply_that_refuses_the_edit_sends_the_caller_back_to_a_fresh_message() -> None:
    """Discord turning the edit down must not cost the user the error entirely."""
    message = FakeMessage()
    reply = FakeReply()
    reply.edit_error = make_not_found()
    streamer = _streamer(message=message, reply=as_message(fake=reply))

    assert await streamer.land_failure(embed=Embed(title="Something went wrong")) is False


async def test_a_delivered_answer_stops_being_the_failure_paths_target() -> None:
    """A failure after the answer landed is a separate event, not the reason one is truncated.

    The take-back happens as the footer is written rather than when the stream helper returns,
    because everything past that point -- the inline media attach, a hosted-URL follow-up -- can
    still raise, and an error landing on the finished reply would take its attachments with it.
    """
    message = FakeMessage()
    streamer = _streamer(message=message)
    published: list[object] = []

    async def events() -> AsyncIterator[SimpleNamespace]:
        """Notes which streamer the failure path would find while the answer is in flight."""
        yield _text_event(delta="done")
        published.append(streaming_module.current_answer_streamer.get())
        yield _completed_event(input_tokens=1, output_tokens=2)

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        return cast("AsyncIterator[ResponseStreamEvent]", events())

    await stream_answer_with_retry(
        streamer=streamer, open_stream=open_stream, message_id=message.id
    )

    assert published == [streamer]
    assert streaming_module.current_answer_streamer.get() is None
