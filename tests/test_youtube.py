"""Tests for YouTube URL detection and the Gemini Interactions answer-path adapters."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from collections.abc import AsyncIterator

import pytest
from google.genai.errors import APIError

from discordbot.utils.youtube import YOUTUBE_URL_RE
from discordbot.utils.llm_errors import extract_friendly_error, is_retryable_llm_error
from discordbot.cogs.gen_reply.interactions import to_interactions_input, adapt_interactions_stream

from tests.helpers.casting import step_dicts, as_interaction_event_stream

if TYPE_CHECKING:
    from openai.types.responses.response_input_param import ResponseInputParam


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


def test_to_interactions_input_skips_empty_and_handles_no_user_step() -> None:
    """An answer input with no messages still yields one user step carrying just the video."""
    steps = step_dicts(
        steps=to_interactions_input(answer_input=[], youtube_url="https://youtu.be/abcdefghijk")
    )
    assert len(steps) == 1
    assert steps[0]["type"] == "user_input"
    assert steps[0]["content"] == [{"type": "video", "uri": "https://youtu.be/abcdefghijk"}]


def _interaction_events() -> list[SimpleNamespace]:
    """A minimal Interactions stream: created, a thought, two text deltas, completed+usage."""
    return [
        SimpleNamespace(
            event_type="interaction.created",
            interaction=SimpleNamespace(model="gemini-3.1-pro-preview"),
        ),
        SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="thought_summary", content=SimpleNamespace(text="hmm")),
        ),
        SimpleNamespace(event_type="step.delta", delta=SimpleNamespace(type="text", text="Hello")),
        SimpleNamespace(
            event_type="step.delta", delta=SimpleNamespace(type="text", text=" world")
        ),
        SimpleNamespace(
            event_type="interaction.completed",
            interaction=SimpleNamespace(
                model="gemini-3.1-pro-preview",
                usage=SimpleNamespace(total_input_tokens=12, total_output_tokens=34),
            ),
            metadata=None,
        ),
    ]


async def _aiter(events: list[SimpleNamespace]) -> AsyncIterator[SimpleNamespace]:
    """Yields fake Interactions events in order."""
    for event in events:
        yield event


def _ns(event: object) -> SimpleNamespace:
    """Narrows an adapted event to the namespace shape the adapter fabricates."""
    assert isinstance(event, SimpleNamespace)
    return event


async def test_adapt_interactions_stream_remaps_to_responses_events() -> None:
    """Interactions events become Responses-shaped events the streamer consumes."""
    stream = adapt_interactions_stream(
        stream=as_interaction_event_stream(fake=_aiter(events=_interaction_events()))
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


async def _raise_from_error_event(error: object) -> APIError:
    """Drives the adapter over one error event and returns what it raised."""
    events = [SimpleNamespace(event_type="error", error=error)]
    with pytest.raises(APIError) as raised:
        async for _ in adapt_interactions_stream(
            stream=as_interaction_event_stream(fake=_aiter(events=events))
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
