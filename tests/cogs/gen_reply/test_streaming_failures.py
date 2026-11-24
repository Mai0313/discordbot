"""Tests for answer-stream delivery windows, stream errors, and failure landing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
import asyncio

import httpx2
from openai import APIError, AsyncOpenAI, BadRequestError
import pytest
from nextcord import Embed

from discordbot.cogs.gen_reply import streaming as streaming_module
from discordbot.utils.llm_errors import (
    HIGH_DEMAND_ERROR_NOTICE,
    llm_status_code,
    extract_friendly_error,
)
from discordbot.cogs.gen_reply.surface import WINDOW_EXPIRED_NOTICE
from discordbot.cogs.gen_reply.pipeline import ReplyPipeline
from discordbot.cogs.gen_reply.streaming import (
    ResponseStreamer,
    AnswerStreamError,
    IncompleteAnswerError,
    stream_answer,
)
from discordbot.cogs.gen_reply.status_marks import ANSWER_EMOJI

from tests.helpers.casting import as_message, make_not_found
from tests.helpers.reply_fakes import (
    TEST_LLM_MODEL,
    FakeReply,
    FakeAuthor,
    FakeMessage,
    _cog,
    _streamer,
    _text_event,
    _stream_events,
    _completed_event,
    _expiring_surface,
    _incomplete_event,
    _stream_events_from,
    _mid_stream_unavailable,
    _stream_events_then_raise,
)
from tests.helpers.link_sources import hosting_off_planner
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
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

    Past the token the answer and the notice both 404, so waiting on would only trade the
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
            stream_answer(streamer=streamer, open_stream=open_stream), timeout=5
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
        await stream_answer(streamer=streamer, open_stream=open_stream)

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

    reply = await stream_answer(streamer=streamer, open_stream=open_stream)

    assert reply


async def test_a_provider_stream_error_propagates_without_retrying() -> None:
    """Both transient 5xx and 400 refusals surface directly on the single attempt."""
    request = httpx2.Request(method="POST", url="http://proxy/v1/responses")
    refusal = BadRequestError(
        "blocked", response=httpx2.Response(status_code=400, request=request, json={}), body=None
    )
    for error, expected_type in (
        (_mid_stream_unavailable(), APIError),
        (refusal, BadRequestError),
    ):
        message = FakeMessage()
        streamer = _streamer(message=message)
        opened = 0

        async def open_stream(err: Exception = error) -> AsyncIterator[ResponseStreamEvent]:
            nonlocal opened
            opened += 1
            return _stream_events_then_raise(events=[], error=err)

        with pytest.raises(expected_type):
            await stream_answer(streamer=streamer, open_stream=open_stream)

        assert opened == 1


async def test_a_failed_answer_lands_its_error_on_the_reply_it_was_streaming_into(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn that painted something before it died ends as ONE message, not two.

    The failure surfaces in `on_message`, several frames above the streamer that owns the
    reply handle, so this drives the real helper under the real error path: the pipeline is
    stubbed down to the one answer stream, and everything between the publish and the edit is
    the production code.
    """
    cog = _cog()
    message = FakeMessage(content="<@999> explain", author=FakeAuthor(user_id=1))
    reply = FakeReply()
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        return _stream_events_then_raise(
            events=[_text_event(delta="half a sentence")],
            error=_mid_stream_unavailable(),
            pause=0.05,
        )

    async def failing_answer(self: object, **kwargs: object) -> None:
        """Streams half an answer onto a reply already on screen, then fails."""
        del kwargs
        streamer = _streamer(
            message=message, reply=as_message(fake=reply), preview_interval_seconds=0.01
        )
        await stream_answer(streamer=streamer, open_stream=open_stream)

    monkeypatch.setattr(ReplyPipeline, "run", failing_answer)
    await cog.on_message(message=as_message(fake=message))

    assert opened == 1
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
    reply.content = f"-# {ANSWER_EMOJI} Thinking..."
    streamer = _streamer(message=message, reply=as_message(fake=reply))
    streamer.reasoning_content = "weighing the options"

    assert await streamer.land_failure(embed=Embed(title="Something went wrong")) is True

    assert reply.content == ""
    assert reply.embed is not None


async def test_a_failure_keeps_a_partial_answer_longer_than_one_message() -> None:
    """Text past one message continues in follow-ups, with the error under the last of them.

    The preview only ever showed one message's worth, so landing just that would drop the rest of
    what the model wrote without a word.
    """
    message = FakeMessage()
    reply = FakeReply()
    streamer = _streamer(message=message, reply=as_message(fake=reply))
    partial = " ".join(["word"] * 1000)
    streamer.stored_content = partial
    streamer.content_started = True

    assert await streamer.land_failure(embed=Embed(title="Something went wrong")) is True

    chain = [reply]
    while chain[-1].replies:
        chain.append(chain[-1].replies[0])
    assert len(chain) > 1
    assert "".join(sent.content or "" for sent in chain) == partial
    assert [sent.embed is not None for sent in chain] == [False] * (len(chain) - 1) + [True]


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

    await stream_answer(streamer=streamer, open_stream=open_stream)

    assert published == [streamer]
    assert streaming_module.current_answer_streamer.get() is None


def _proxy_sse(events: list[dict[str, object]]) -> bytes:
    """Frames Responses events the way the proxy streams them, one `data:` line each."""
    return b"".join(f"data: {json.dumps(obj=event)}\n\n".encode() for event in events)


def _proxy_response(status: str, usage: dict[str, int], **fields: object) -> dict[str, object]:
    """The `response` object a proxy terminal event carries, cut to what the stream reads."""
    return {
        "id": "resp_1",
        "object": "response",
        "model": TEST_LLM_MODEL,
        "status": status,
        "output": [],
        "usage": usage,
        **fields,
    }


def _proxy_sse_client(streams: list[bytes]) -> AsyncOpenAI:
    """Builds an `AsyncOpenAI` client whose `/v1/responses` calls replay `streams` in order."""
    return AsyncOpenAI(
        api_key="test",
        base_url="http://proxy.test/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                handler=lambda request: httpx2.Response(
                    status_code=200,
                    headers={"content-type": "text/event-stream"},
                    content=streams.pop(0),
                )
            )
        ),
    )


async def test_a_filtered_answer_raises_incomplete_without_retrying() -> None:
    """A proxy `response.incomplete` (`content_filter`) raises directly and records its usage.

    Driven through the real `openai` client over the proxy's framing, since what has to hold is
    that the SDK hands this event over at all and where the reason sits on it.
    """
    client = _proxy_sse_client(
        streams=[
            _proxy_sse(
                events=[
                    {"type": "response.output_text.delta", "delta": "half a sentence"},
                    {
                        "type": "response.incomplete",
                        "response": _proxy_response(
                            status="incomplete",
                            usage={"input_tokens": 100, "output_tokens": 55},
                            incomplete_details={"reason": "content_filter"},
                        ),
                    },
                ]
            )
        ]
    )
    message = FakeMessage()
    streamer = _streamer(message=message)
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        return await client.responses.create(model=TEST_LLM_MODEL, input="hi", stream=True)

    with pytest.raises(IncompleteAnswerError, match="content filter") as raised:
        await stream_answer(streamer=streamer, open_stream=open_stream)

    assert opened == 1
    assert raised.value.reason == "content_filter"
    assert (streamer.input_tokens, streamer.output_tokens) == (100, 55)


async def test_an_answer_that_spent_its_output_budget_logs_and_raises_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An answer that exhausted `max_output_tokens` logs the reason and raises directly.

    Its log record is what the rate of cut-off answers is read back from in `data/logs`.
    """
    logged = capture_logs(monkeypatch=monkeypatch, level="warn")
    message = FakeMessage()
    streamer = _streamer(message=message)
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        return _stream_events_from(
            events=[_text_event(delta="half"), _incomplete_event(reason="max_output_tokens")]
        )

    with pytest.raises(IncompleteAnswerError, match="output budget"):
        await stream_answer(streamer=streamer, open_stream=open_stream)

    assert opened == 1
    assert [fields for text, fields in logged if text == "gen_reply answer stream incomplete"] == [
        {
            "message_id": message.id,
            "model": TEST_LLM_MODEL,
            "backend": "responses",
            "reason": "max_output_tokens",
            "reply_chars": len("half"),
            "reasoning_chars": 0,
        }
    ]


async def test_a_stream_that_just_stops_raises_incomplete_without_retrying() -> None:
    """A stream with no terminal event at all raises `IncompleteAnswerError` on the first try."""
    message = FakeMessage()
    streamer = _streamer(message=message)
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        return _stream_events_from(events=[_text_event(delta="cut")])

    with pytest.raises(IncompleteAnswerError) as raised:
        await stream_answer(streamer=streamer, open_stream=open_stream)

    assert opened == 1
    assert raised.value.reason == "no_terminal_event"


async def test_a_cut_off_answer_says_why_under_the_text_it_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filtered reply keeps its half-sentence with the reason embed under it.

    What this replaced was the half-sentence over a usage footer, which reads as an answer that
    simply stopped.
    """
    cog = _cog()
    message = FakeMessage(content="<@999> explain", author=FakeAuthor(user_id=1))
    reply = FakeReply()

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        # Paced so the preview editor writes the half-sentence before the stream ends.
        return _paced_stream_events(
            events=[
                _text_event(delta="half a sentence"),
                _incomplete_event(reason="content_filter"),
            ],
            pause=0.05,
        )

    async def cut_off_answer(self: object, **kwargs: object) -> None:
        """Streams a half-sentence onto a reply already on screen, then stops on content_filter."""
        del kwargs
        streamer = _streamer(
            message=message, reply=as_message(fake=reply), preview_interval_seconds=0.01
        )
        await stream_answer(streamer=streamer, open_stream=open_stream)

    monkeypatch.setattr(ReplyPipeline, "run", cut_off_answer)
    await cog.on_message(message=as_message(fake=message))

    assert message.replies == []
    assert reply.content == "half a sentence"
    assert reply.embed is not None
    assert reply.embed.footer.text == "IncompleteAnswerError"
    assert "content filter" in (reply.embed.description or "")


async def test_response_failed_server_error_fails_without_retrying_and_surfaces_friendly_message() -> (
    None
):
    """A LiteLLM `response.failed` `server_error` fails directly and surfaces the masked notice."""
    high_demand = (
        "This model is currently experiencing high demand. Spikes in demand are usually "
        "temporary. Please try again later."
    )
    document = {"error": {"code": 503, "message": high_demand, "status": "UNAVAILABLE"}}
    wrapped = (
        "litellm.ServiceUnavailableError: Vertex_ai_betaException - "
        f"{json.dumps(obj=document).encode()!r}"
    )
    failed_sse = _proxy_sse(
        events=[
            {
                "type": "response.failed",
                "sequence_number": 1,
                "response": _proxy_response(
                    status="failed",
                    usage={"input_tokens": 11, "output_tokens": 0},
                    error={"code": "server_error", "message": wrapped},
                ),
            }
        ]
    )
    client = _proxy_sse_client(streams=[failed_sse])
    message = FakeMessage()
    streamer = _streamer(message=message)
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        return await client.responses.create(model=TEST_LLM_MODEL, input="hi", stream=True)

    with pytest.raises(AnswerStreamError) as raised:
        await stream_answer(streamer=streamer, open_stream=open_stream)

    assert opened == 1
    assert raised.value.code == "server_error"
    assert raised.value.status_code == 500
    assert llm_status_code(exc=raised.value) == 500
    assert extract_friendly_error(exc=raised.value) == HIGH_DEMAND_ERROR_NOTICE
    assert streamer.model_name == TEST_LLM_MODEL
    assert streamer.input_tokens == 11


@pytest.mark.parametrize(
    ("code", "expected_status", "upstream_message", "expected_friendly"),
    [
        ("server_error", 500, "temporary failure", HIGH_DEMAND_ERROR_NOTICE),
        ("rate_limit_exceeded", 429, "temporary failure", HIGH_DEMAND_ERROR_NOTICE),
        ("request_timeout", 408, "Request timed out", "Request timed out"),
        ("503", 503, "temporary failure", HIGH_DEMAND_ERROR_NOTICE),
        ("429", 429, "temporary failure", HIGH_DEMAND_ERROR_NOTICE),
        ("408", 408, "Request timed out", "Request timed out"),
        ("409", 409, "Conflict", "Conflict"),
        ("invalid_request_error", 400, "Unsupported parameter", "Unsupported parameter"),
        ("invalid_prompt", 400, "Invalid prompt", "Invalid prompt"),
        ("insufficient_quota", 402, "You exceeded your current quota", HIGH_DEMAND_ERROR_NOTICE),
        ("authentication_error", 401, "Invalid API key", HIGH_DEMAND_ERROR_NOTICE),
        ("permission_error", 403, "Forbidden", HIGH_DEMAND_ERROR_NOTICE),
        ("not_found_error", 404, "Model not found", HIGH_DEMAND_ERROR_NOTICE),
        ("400", 400, "Bad request", "Bad request"),
        (None, None, "Unknown stream failure", "Unknown stream failure"),
        ("bio_policy", None, "Policy violation", "Policy violation"),
    ],
)
async def test_response_failed_codes_map_status_and_fail_without_retrying(
    code: str | None, expected_status: int | None, upstream_message: str, expected_friendly: str
) -> None:
    """`response.failed` maps error codes to HTTP statuses and fails directly without retrying."""
    client = _proxy_sse_client(
        streams=[
            _proxy_sse(
                events=[
                    {
                        "type": "response.failed",
                        "sequence_number": 1,
                        "response": _proxy_response(
                            status="failed",
                            usage={"input_tokens": 5, "output_tokens": 0},
                            error={"code": code, "message": upstream_message},
                        ),
                    }
                ]
            )
        ]
    )
    message = FakeMessage()
    streamer = _streamer(message=message)
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        return await client.responses.create(model=TEST_LLM_MODEL, input="hi", stream=True)

    with pytest.raises(AnswerStreamError) as raised:
        await stream_answer(streamer=streamer, open_stream=open_stream)

    assert opened == 1
    assert raised.value.code == code
    assert raised.value.status_code == expected_status
    assert extract_friendly_error(exc=raised.value) == expected_friendly


def test_answer_stream_error_masks_overload_and_quota_payloads() -> None:
    """`AnswerStreamError` masks operator/backend codes while preserving 400/408 and `IncompleteAnswerError`."""
    quota_msg = "You exceeded your current quota, please check your plan and billing details."
    for masked_code in (
        "rate_limit_exceeded",
        "insufficient_quota",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "server_error",
        "500",
        "502",
        "503",
        "504",
    ):
        assert (
            extract_friendly_error(exc=AnswerStreamError(message=quota_msg, code=masked_code))
            == HIGH_DEMAND_ERROR_NOTICE
        )
    wrapped_429 = (
        "litellm.RateLimitError: Vertex_ai_betaException - "
        f"{json.dumps(obj={'error': {'code': 429, 'message': quota_msg}}).encode()!r}"
    )
    assert (
        extract_friendly_error(exc=AnswerStreamError(message=wrapped_429, code="server_error"))
        == HIGH_DEMAND_ERROR_NOTICE
    )
    wrapped_503 = (
        "litellm.ServiceUnavailableError: Vertex_ai_betaException - "
        f"{json.dumps(obj={'error': {'code': 503, 'message': 'Model overloaded'}}).encode()!r}"
    )
    assert (
        extract_friendly_error(exc=AnswerStreamError(message=wrapped_503, code="server_error"))
        == HIGH_DEMAND_ERROR_NOTICE
    )
    assert (
        extract_friendly_error(exc=AnswerStreamError(message="Request timed out", code="408"))
        == "Request timed out"
    )
    assert (
        extract_friendly_error(exc=AnswerStreamError(message="Bad request", code="400"))
        == "Bad request"
    )
    for reason in ("content_filter", "max_output_tokens", "no_terminal_event"):
        incomplete = IncompleteAnswerError(reason=reason)
        assert extract_friendly_error(exc=incomplete) == str(incomplete)


@pytest.mark.parametrize(
    ("code", "expected_status"),
    [
        ("rate_limit_exceeded", 429),
        ("503", 503),
        ("insufficient_quota", 402),
        ("invalid_request_error", 400),
    ],
)
async def test_response_error_event_raises_answer_stream_error_without_retrying(
    code: str, expected_status: int
) -> None:
    """`ResponseErrorEvent` (`type="error"`) surfaces as `AnswerStreamError` on a single attempt."""
    error_sse = _proxy_sse(
        events=[
            {
                "type": "error",
                "sequence_number": 1,
                "code": code,
                "message": f"upstream {code}",
                "param": None,
            }
        ]
    )
    client = _proxy_sse_client(streams=[error_sse])
    message = FakeMessage()
    streamer = _streamer(message=message)
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        return await client.responses.create(model=TEST_LLM_MODEL, input="hi", stream=True)

    with pytest.raises(AnswerStreamError) as raised:
        await stream_answer(streamer=streamer, open_stream=open_stream)

    assert opened == 1
    assert raised.value.code == code
    assert raised.value.status_code == expected_status
    assert str(raised.value) == f"upstream {code}"


async def test_response_failed_consume_finally_stamps_duration_and_stops_editor() -> None:
    """`consume`'s finally block stamps `answer_seconds`, stops the editor, and keeps created model."""
    message = FakeMessage()
    streamer = _streamer(message=message, preview_interval_seconds=0.01)

    async def _events() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(
            type="response.created", response=SimpleNamespace(model="captured-model")
        )
        yield _text_event(delta="partial")
        await asyncio.sleep(0.03)
        yield SimpleNamespace(
            type="response.failed", response=SimpleNamespace(model="", usage=None, error=None)
        )

    with pytest.raises(
        AnswerStreamError, match=r"The model failed to generate a response\."
    ) as raised:
        await streamer.consume(responses=cast("AsyncIterator[ResponseStreamEvent]", _events()))

    assert raised.value.code is None
    assert streamer.model_name == "captured-model"
    assert streamer.answer_seconds > 0
    assert streamer._editor is not None
    assert streamer._editor._task is None


async def test_empty_stream_without_terminal_event_still_raises_no_terminal_event() -> None:
    """A genuine empty stream with no terminal event raises `IncompleteAnswerError` directly."""
    message = FakeMessage()
    streamer = _streamer(message=message)
    opened = 0

    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
        nonlocal opened
        opened += 1
        return _stream_events_from(events=[])

    with pytest.raises(IncompleteAnswerError) as raised:
        await stream_answer(streamer=streamer, open_stream=open_stream)

    assert opened == 1
    assert raised.value.reason == "no_terminal_event"
