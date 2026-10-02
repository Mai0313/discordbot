"""Tests for YouTube URL detection and the Gemini Interactions answer-path adapters."""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import httpx
from google import genai
import pytest
from google.genai import types
from google.genai.errors import APIError

from discordbot.utils.llm_errors import extract_friendly_error, is_retryable_llm_error
from discordbot.cogs.gen_reply.streaming import stream_answer_with_retry
from discordbot.services.platforms.youtube import YOUTUBE_URL_RE
from discordbot.cogs.gen_reply.interactions import (
    to_interactions_input,
    adapt_interactions_stream,
    create_interactions_answer_stream,
)

from tests.helpers.casting import step_dicts, as_interaction_event_stream
from tests.helpers.gen_reply import event_stream, interactions_turn_events

if TYPE_CHECKING:
    from collections.abc import Callable, AsyncIterator

    from openai.types.responses import ResponseStreamEvent
    from openai.types.responses.response_input_param import ResponseInputParam

    from discordbot.cogs.gen_reply.streaming import ResponseStreamer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        ),
        ("https://youtube.com/watch?v=jNQXAC9IVRw", "https://youtube.com/watch?v=jNQXAC9IVRw"),
        ("https://youtu.be/jNQXAC9IVRw", "https://youtu.be/jNQXAC9IVRw"),
        (
            "https://www.youtube.com/shorts/abcdefghijk",
            "https://www.youtube.com/shorts/abcdefghijk",
        ),
        ("https://www.youtube.com/live/abcdefghijk", "https://www.youtube.com/live/abcdefghijk"),
        (
            "https://m.youtube.com/watch?v=jNQXAC9IVRw&t=30s",
            "https://m.youtube.com/watch?v=jNQXAC9IVRw&t=30s",
        ),
        (
            "https://www.youtube.com/watch?app=desktop&v=jNQXAC9IVRw",
            "https://www.youtube.com/watch?app=desktop&v=jNQXAC9IVRw",
        ),
        ("看這個 https://youtu.be/jNQXAC9IVRw。很讚", "https://youtu.be/jNQXAC9IVRw"),
        ("watch https://youtu.be/jNQXAC9IVRw, then react", "https://youtu.be/jNQXAC9IVRw"),
    ],
)
def test_youtube_url_re_matches_watchable_links(text: str, expected: str) -> None:
    """The shared regex extracts a watchable YouTube URL, trailing punctuation excluded."""
    match = YOUTUBE_URL_RE.search(string=text)
    assert match is not None
    assert match.group(0) == expected


@pytest.mark.parametrize(
    "text",
    [
        "no url here at all",
        "https://www.youtube.com/playlist?list=PL123",
        "https://www.youtube.com/@channelname",
        "https://example.com/watch?v=jNQXAC9IVRw",
        "https://vimeo.com/123456789",
    ],
)
def test_youtube_url_re_rejects_non_videos(text: str) -> None:
    """Channel / playlist / non-YouTube URLs and plain text are not matched."""
    assert YOUTUBE_URL_RE.search(string=text) is None


def test_to_interactions_input_maps_roles_and_appends_video() -> None:
    """System folds into user, assistant becomes model_output, and the video lands last."""
    answer_input = [
        {"role": "system", "content": "reference header"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": [{"type": "input_text", "text": "what happens here"}]},
    ]

    steps = step_dicts(
        steps=to_interactions_input(
            answer_input=cast("ResponseInputParam", answer_input),
            youtube_url="https://youtu.be/abcdefghijk",
        )
    )

    # system + user coalesce into one user_input step; assistant is its own model_output step.
    assert [s["type"] for s in steps] == ["user_input", "model_output", "user_input"]
    first_texts = [c["text"] for c in steps[0]["content"]]
    assert first_texts == ["reference header", "hello"]
    assert steps[1]["content"][0]["text"] == "hi there"
    last_parts = steps[-1]["content"]
    assert last_parts[0] == {"type": "text", "text": "what happens here"}
    assert last_parts[-1] == {"type": "video", "uri": "https://youtu.be/abcdefghijk"}


def test_to_interactions_input_maps_media_parts_by_kind() -> None:
    """Files map to video / image / audio / document params by extension; images keep their URL."""
    answer_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "compare these"},
                {"type": "input_file", "file_id": "https://x/files/v1", "filename": "clip.mp4"},
                {"type": "input_file", "file_id": "https://x/files/p1", "filename": "doc.pdf"},
                {"type": "input_file", "file_id": "https://x/files/i1", "filename": "shot.png"},
                {"type": "input_file", "file_id": "https://x/files/a1", "filename": "song.mp3"},
                {"type": "input_image", "image_url": "https://x/pic.jpg"},
            ],
        }
    ]

    steps = step_dicts(
        steps=to_interactions_input(
            answer_input=cast("ResponseInputParam", answer_input),
            youtube_url="https://youtu.be/abcdefghijk",
        )
    )

    parts = steps[-1]["content"]
    kinds = [p["type"] for p in parts]
    assert kinds == ["text", "video", "document", "image", "audio", "image", "video"]
    assert parts[1] == {"type": "video", "uri": "https://x/files/v1"}
    assert parts[2] == {"type": "document", "uri": "https://x/files/p1"}
    assert parts[4] == {"type": "audio", "uri": "https://x/files/a1"}
    assert parts[5] == {"type": "image", "uri": "https://x/pic.jpg"}


def test_to_interactions_input_sends_inlined_bytes_as_data_not_as_a_uri() -> None:
    """With the Files API off every attachment arrives inlined, and both halves must still land.

    `file_api_enabled=false` selects `InlineRenderer` for every provider, Gemini included, and
    it renders a base64 `data:` URI wherever the Files API path renders an https one. The two
    settings are independent — `use_interactions` never consults `file_api_enabled` — so a
    YouTube link plus an attachment reaches here inlined.

    The image must not go out in `uri`, which the SDK documents as a URI for the server to
    fetch, and the PDF arrives only in `file_data`, a field the Files API path never fills.
    """
    png = "data:image/png;base64,iVBORw0KGgo="
    pdf = "data:application/pdf;base64,JVBERi0xLjQK"
    answer_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": png},
                {"type": "input_file", "filename": "paper.pdf", "file_data": pdf},
            ],
        }
    ]

    steps = step_dicts(
        steps=to_interactions_input(
            answer_input=cast("ResponseInputParam", answer_input),
            youtube_url="https://youtu.be/abcdefghijk",
        )
    )

    parts = steps[-1]["content"]
    assert parts[0] == {"type": "image", "data": "iVBORw0KGgo=", "mime_type": "image/png"}
    assert parts[1] == {"type": "document", "data": "JVBERi0xLjQK", "mime_type": "application/pdf"}
    # The YouTube video itself is still a real URI and must not be inlined.
    assert parts[-1] == {"type": "video", "uri": "https://youtu.be/abcdefghijk"}


def _inlined_image_parts(reference: str) -> list[dict[str, object]]:
    """The content of the one user step for a turn carrying a single inlined image."""
    answer_input = [{"role": "user", "content": [{"type": "input_image", "image_url": reference}]}]
    steps = step_dicts(
        steps=to_interactions_input(
            answer_input=cast("ResponseInputParam", answer_input),
            youtube_url="https://youtu.be/abcdefghijk",
        )
    )
    return steps[-1]["content"]


def test_to_interactions_input_strips_a_mime_parameter_rather_than_losing_the_image() -> None:
    """The image path hands over whatever MIME Discord reported, parameters and all.

    `attachment_mime` strips them for the file path and `shrink_image_bytes` returns the raw
    `content_type` unchanged for a GIF, an animated image, a within-bounds JPEG, or any PIL
    failure. There are real bytes here, so the parameter is dropped and the image is carried.
    """
    parts = _inlined_image_parts(reference="data:image/gif;charset=binary;base64,R0lGOD==")

    assert parts[0] == {"type": "image", "data": "R0lGOD==", "mime_type": "image/gif"}


@pytest.mark.parametrize(
    ("reference", "why"),
    [
        ("data:application/pdf;base64,", "an empty attachment, so no payload at all"),
        ("data:image/png,notbase64", "a data URI that is not base64"),
        ("data:;base64,R0lGOD==", "no MIME to declare"),
    ],
)
def test_to_interactions_input_never_demotes_an_unusable_data_uri_to_a_uri(
    reference: str, why: str
) -> None:
    """An inlined reference with nothing to send is dropped, never put in `uri`.

    Recognising one exact shape and falling through on everything else would put the odd one
    back in the field the server fetches from.
    """
    parts = _inlined_image_parts(reference=reference)

    assert all(part.get("uri") != reference for part in parts), f"{why} was sent as a uri"
    # Only the YouTube video is left, so the part was dropped rather than mangled.
    assert parts == [{"type": "video", "uri": "https://youtu.be/abcdefghijk"}]


def test_to_interactions_input_skips_empty_and_handles_no_user_step() -> None:
    """An answer input with no messages still yields one user step carrying just the video."""
    steps = step_dicts(
        steps=to_interactions_input(answer_input=[], youtube_url="https://youtu.be/abcdefghijk")
    )
    assert len(steps) == 1
    assert steps[0]["type"] == "user_input"
    assert steps[0]["content"] == [{"type": "video", "uri": "https://youtu.be/abcdefghijk"}]


def _ns(event: object) -> SimpleNamespace:
    """Narrows an adapted event to the namespace shape the adapter fabricates."""
    assert isinstance(event, SimpleNamespace)
    return event


async def test_adapt_interactions_stream_remaps_to_responses_events() -> None:
    """Interactions events become Responses-shaped events the streamer consumes."""
    stream = adapt_interactions_stream(
        stream=as_interaction_event_stream(fake=event_stream(events=interactions_turn_events()))
    )
    out = [event async for event in stream]

    types = [event.type for event in out]
    assert types == [
        "response.created",
        "response.reasoning_summary_text.delta",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]
    assert _ns(event=out[0]).response.model == "gemini-3.1-pro-preview"
    assert _ns(event=out[1]).delta == "hmm"
    assert _ns(event=out[2]).delta == "Hello"
    # Usage is emitted once, on completion, with the Responses field names.
    assert _ns(event=out[-1]).response.usage.input_tokens == 12
    assert _ns(event=out[-1]).response.usage.output_tokens == 34
    # `output` is None rather than [], so the streamer logs grounding as "not reported" here.
    # An empty list would count as zero citations and read as an ungrounded answer.
    assert _ns(event=out[-1]).response.output is None
    assert _ns(event=out[0]).response.output is None


async def test_adapt_interactions_stream_falls_back_to_step_delta_usage() -> None:
    """A completed event with no usage of its own reports the last `total_usage` streamed.

    google-genai 2.22 moved `total_usage` onto `step.delta`, and `interaction.usage` is optional
    on a streaming payload, so without this the footer and the turn's token fields read zero.
    """
    events = interactions_turn_events()
    events[2].metadata = SimpleNamespace(
        total_usage=SimpleNamespace(total_input_tokens=7, total_output_tokens=9)
    )
    events[-1].interaction.usage = None

    out = [
        event
        async for event in adapt_interactions_stream(
            stream=as_interaction_event_stream(fake=event_stream(events=events))
        )
    ]

    assert _ns(event=out[-1]).response.usage.input_tokens == 7
    assert _ns(event=out[-1]).response.usage.output_tokens == 9


async def _raise_from_error_event(error: object) -> APIError:
    """Drives the adapter over one error event and returns what it raised."""
    events = [SimpleNamespace(event_type="error", error=error)]
    with pytest.raises(APIError) as raised:
        async for _ in adapt_interactions_stream(
            stream=as_interaction_event_stream(fake=event_stream(events=events))
        ):
            pass
    return raised.value


async def test_adapt_interactions_stream_raises_a_classifiable_error_event() -> None:
    """An in-band error surfaces as an SDK error the answer retry and the user can both read.

    A bare exception here would leave the YouTube answer backend sitting inside
    `stream_answer_with_retry` while never being retryable, and would show the user this
    event's repr instead of what the provider actually said.
    """
    transient = await _raise_from_error_event(
        error=SimpleNamespace(code="503", message="high demand")
    )
    assert extract_friendly_error(exc=transient) == "high demand"
    assert is_retryable_llm_error(exc=transient) is True

    # What the SDK actually documents that field as is a URI identifying the error type, so
    # this is the shape a real in-band failure takes: unclassifiable, and left alone rather
    # than guessed at. The decimal case above is a hedge, not an observed path.
    opaque = await _raise_from_error_event(
        error=SimpleNamespace(code="UNAVAILABLE", message="high demand")
    )
    assert is_retryable_llm_error(exc=opaque) is False
    assert is_retryable_llm_error(exc=await _raise_from_error_event(error=None)) is False


def _sse(events: list[dict[str, object]]) -> bytes:
    """Frames Interactions events the way the API streams them."""
    return b"".join(
        f"event: {event['event_type']}\ndata: {json.dumps(obj=event)}\n\n".encode()
        for event in events
    )


_INTERACTION_CREATED: dict[str, object] = {
    "event_type": "interaction.created",
    "interaction": {"id": "i1", "model": "gemini-3.1-pro-preview", "status": "in_progress"},
}

_ANSWER_STREAM = _sse(
    events=[
        _INTERACTION_CREATED,
        {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "the answer"}},
        {
            "event_type": "interaction.completed",
            "interaction": {"id": "i1", "model": "gemini-3.1-pro-preview", "status": "completed"},
        },
    ]
)


def _http_status(status: int) -> "Callable[[httpx.Request], httpx.Response]":
    """A failure the API answers with `status`."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status,
            json={"error": {"code": status, "message": f"provider answered {status}"}},
        )

    return respond


def _connection_refused(request: httpx.Request) -> httpx.Response:
    """A failure before any response arrives."""
    raise httpx.ConnectError("connection refused", request=request)


class _DroppedMidStream(httpx.AsyncByteStream):
    """An answer stream whose connection dies after its first event."""

    async def __aiter__(self) -> "AsyncIterator[bytes]":
        yield _sse(events=[_INTERACTION_CREATED])
        raise httpx.RemoteProtocolError("peer closed connection")


def _dropped_mid_stream(request: httpx.Request) -> httpx.Response:
    """A failure after the answer has started streaming."""
    return httpx.Response(
        status_code=200, headers={"content-type": "text/event-stream"}, stream=_DroppedMidStream()
    )


class _TextStreamer:
    """The part of `ResponseStreamer` the answer retry drives: it joins the text deltas."""

    carries_turn_notices = False

    def reset_for_retry(self) -> None:
        """Nothing to drop: each attempt's text is joined afresh."""

    async def stream(self, responses: "AsyncIterator[ResponseStreamEvent]") -> str:
        """Joins the answer's text, letting whatever the stream raises propagate."""
        return "".join([
            _ns(event=event).delta
            async for event in responses
            if event.type == "response.output_text.delta"
        ])


class _YouTubeAnswerTurn:
    """A YouTube answer turn on a real Gemini client whose first attempt meets `first_attempt`.

    Every request made while the first attempt is open gets that failure, however often the
    SDK re-sends it; every later attempt streams a complete answer.
    """

    def __init__(self, first_attempt: "Callable[[httpx.Request], httpx.Response]") -> None:
        self.first_attempt = first_attempt
        self.opened = 0
        self.client = genai.Client(
            api_key="test",
            http_options=types.HttpOptions(
                httpx_async_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(handler=self._respond)
                ),
                # Keeps the SDK's own re-sends but makes them sleepless.
                retry_options=types.HttpRetryOptions(initial_delay=0),
            ),
        )

    def _respond(self, request: httpx.Request) -> httpx.Response:
        if self.opened == 1:
            return self.first_attempt(request)
        return httpx.Response(
            status_code=200, headers={"content-type": "text/event-stream"}, content=_ANSWER_STREAM
        )

    async def _open_stream(self) -> "AsyncIterator[ResponseStreamEvent]":
        self.opened += 1
        return create_interactions_answer_stream(
            client=self.client,
            model="gemini-3.1-pro-preview",
            system_instruction="",
            steps=[],
            effort="high",
        )

    async def answer(self) -> str:
        """Runs the turn through the answer retry and returns the reply text."""
        return await stream_answer_with_retry(
            streamer=cast("ResponseStreamer", _TextStreamer()),
            open_stream=self._open_stream,
            message_id=1,
        )


@pytest.fixture
def sleepless_answer_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zeroes both halves of the wait between answer attempts (interval and jitter)."""
    monkeypatch.setattr("discordbot.cogs.gen_reply.streaming.ANSWER_RETRY_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr("discordbot.cogs.gen_reply.streaming.ANSWER_RETRY_JITTER_SECONDS", 0.0)


@pytest.mark.usefixtures("sleepless_answer_retry")
@pytest.mark.parametrize(
    "first_attempt",
    [
        pytest.param(_http_status(status=503), id="http-503"),
        pytest.param(_http_status(status=429), id="http-429"),
        pytest.param(_connection_refused, id="connection-refused"),
        pytest.param(_dropped_mid_stream, id="dropped-mid-stream"),
    ],
)
async def test_a_transient_interactions_failure_reopens_the_youtube_answer(
    first_attempt: "Callable[[httpx.Request], httpx.Response]",
) -> None:
    """A transient failure on the Interactions backend is retried like one on Responses.

    The client is real because what the retry reads is the exception class the SDK raises on
    this surface, which no hand-built error stands in for.
    """
    turn = _YouTubeAnswerTurn(first_attempt=first_attempt)

    assert await turn.answer() == "the answer"
    assert turn.opened == 2


@pytest.mark.usefixtures("sleepless_answer_retry")
async def test_an_interactions_refusal_is_not_retried() -> None:
    """A 400 is the provider refusing the request itself, so the turn fails on its first try."""
    turn = _YouTubeAnswerTurn(first_attempt=_http_status(status=400))

    with pytest.raises(Exception, match="provider answered 400"):
        await turn.answer()
    assert turn.opened == 1
