"""Tests for YouTube URL detection and the Gemini Interactions answer-path adapters."""

from types import SimpleNamespace
from collections.abc import AsyncIterator

import pytest

from discordbot.utils.youtube import YOUTUBE_URL_RE
from discordbot.cogs._gen_reply.interactions import (
    to_interactions_input,
    adapt_interactions_stream,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "https://www.youtube.com/watch?v=jNQXAC9IVRw"),
        ("https://youtube.com/watch?v=jNQXAC9IVRw", "https://youtube.com/watch?v=jNQXAC9IVRw"),
        ("https://youtu.be/jNQXAC9IVRw", "https://youtu.be/jNQXAC9IVRw"),
        ("https://www.youtube.com/shorts/abcdefghijk", "https://www.youtube.com/shorts/abcdefghijk"),
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

    steps = to_interactions_input(answer_input=answer_input, youtube_url="https://youtu.be/abcdefghijk")

    # system + user coalesce into one user_input step; assistant is its own model_output step.
    assert [s["type"] for s in steps] == ["user_input", "model_output", "user_input"]
    first_texts = [c["text"] for c in steps[0]["content"]]
    assert first_texts == ["reference header", "hello"]
    assert steps[1]["content"][0]["text"] == "hi there"
    last_parts = steps[-1]["content"]
    assert last_parts[0] == {"type": "text", "text": "what happens here"}
    assert last_parts[-1] == {"type": "video", "uri": "https://youtu.be/abcdefghijk"}


def test_to_interactions_input_maps_media_parts_by_kind() -> None:
    """Files map to video / image / document params by extension; images keep their URL."""
    answer_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "compare these"},
                {"type": "input_file", "file_id": "https://x/files/v1", "filename": "clip.mp4"},
                {"type": "input_file", "file_id": "https://x/files/p1", "filename": "doc.pdf"},
                {"type": "input_file", "file_id": "https://x/files/i1", "filename": "shot.png"},
                {"type": "input_image", "image_url": "https://x/pic.jpg"},
            ],
        },
    ]

    steps = to_interactions_input(answer_input=answer_input, youtube_url="https://youtu.be/abcdefghijk")

    parts = steps[-1]["content"]
    kinds = [p["type"] for p in parts]
    assert kinds == ["text", "video", "document", "image", "image", "video"]
    assert parts[1] == {"type": "video", "uri": "https://x/files/v1"}
    assert parts[2] == {"type": "document", "uri": "https://x/files/p1"}
    assert parts[4] == {"type": "image", "uri": "https://x/pic.jpg"}


def test_to_interactions_input_skips_empty_and_handles_no_user_step() -> None:
    """Empty content is dropped, and a video with no prior user step still gets one."""
    steps = to_interactions_input(answer_input=[], youtube_url="https://youtu.be/abcdefghijk")
    assert len(steps) == 1
    assert steps[0]["type"] == "user_input"
    assert steps[0]["content"] == [{"type": "video", "uri": "https://youtu.be/abcdefghijk"}]


def _interaction_events() -> list[SimpleNamespace]:
    """A minimal Interactions stream: created, a thought, two text deltas, completed+usage."""
    return [
        SimpleNamespace(
            event_type="interaction.created", interaction=SimpleNamespace(model="gemini-pro-latest")
        ),
        SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="thought_summary", content=SimpleNamespace(text="hmm")),
        ),
        SimpleNamespace(event_type="step.delta", delta=SimpleNamespace(type="text", text="Hello")),
        SimpleNamespace(event_type="step.delta", delta=SimpleNamespace(type="text", text=" world")),
        SimpleNamespace(
            event_type="interaction.completed",
            interaction=SimpleNamespace(model="gemini-pro-latest"),
            metadata=SimpleNamespace(
                usage=SimpleNamespace(total_input_tokens=12, total_output_tokens=34)
            ),
        ),
    ]


async def _aiter(events: list[SimpleNamespace]) -> AsyncIterator[SimpleNamespace]:
    """Yields fake Interactions events in order."""
    for event in events:
        yield event


async def test_adapt_interactions_stream_remaps_to_responses_events() -> None:
    """Interactions events become Responses-shaped events the streamer consumes."""
    out = [event async for event in adapt_interactions_stream(stream=_aiter(_interaction_events()))]

    types = [event.type for event in out]
    assert types == [
        "response.created",
        "response.reasoning_summary_text.delta",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]
    assert out[0].response.model == "gemini-pro-latest"
    assert out[1].delta == "hmm"
    assert out[2].delta == "Hello"
    # Usage is emitted once, on completion, with the Responses field names.
    assert out[-1].response.usage.input_tokens == 12
    assert out[-1].response.usage.output_tokens == 34


async def test_adapt_interactions_stream_raises_on_error_event() -> None:
    """An error event surfaces as an exception for the pipeline's outer handler."""
    events = [SimpleNamespace(event_type="error", error="boom")]

    with pytest.raises(RuntimeError):
        async for _ in adapt_interactions_stream(stream=_aiter(events)):
            pass
