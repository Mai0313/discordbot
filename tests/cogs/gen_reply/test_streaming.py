"""Tests for streaming an answer into Discord: previews, chunks and the footer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
import asyncio

import pytest
import nextcord
import requests

from discordbot.typings.memory import MemoryCredits
from discordbot.utils.model_pricing import ModelPriceEntry
from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.cogs.gen_reply.surface import INTERACTION_FOLLOWUP_LIMIT, TurnSurface
from discordbot.cogs.gen_reply.streaming import (
    TRUNCATED_NOTICE,
    REASONING_PREVIEW_MAX_CHARS,
    REASONING_PREVIEW_MAX_LINES,
    ResponseStreamer,
    _cut_cleanly,
)

from tests.helpers.casting import (
    as_message,
    make_forbidden,
    make_not_found,
    make_invalid_form_body,
)
from tests.helpers.reply_fakes import (
    TEST_LLM_MODEL,
    FakeReply,
    FakeMessage,
    _stream,
    _streamer,
    _text_event,
    _stream_events,
    _completed_event,
    _stream_events_from,
)
from tests.helpers.link_sources import hosting_off_planner
from tests.helpers.ask_interaction import _interaction
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai.types.responses import ResponseStreamEvent


def _annotated_completed_event(annotation_types: list[str]) -> SimpleNamespace:
    """Builds a completed event whose output text carries the given annotation types."""
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            model=TEST_LLM_MODEL,
            usage=SimpleNamespace(input_tokens=1, output_tokens=2),
            output=[
                SimpleNamespace(type="reasoning"),
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="refusal"),
                        SimpleNamespace(
                            type="output_text",
                            annotations=[SimpleNamespace(type=kind) for kind in annotation_types],
                        ),
                    ],
                ),
            ],
        ),
    )


async def test_streaming_counts_only_url_citation_annotations() -> None:
    """Grounding is counted off the completed output, past the reasoning and refusal shapes."""
    streamer = _streamer(message=FakeMessage())

    await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[
                _text_event(delta="grounded"),
                _annotated_completed_event(
                    annotation_types=["url_citation", "file_citation", "url_citation"]
                ),
            ]
        ),
    )

    assert streamer._url_citations == 2


async def test_streaming_leaves_grounding_unreported_when_the_backend_carries_no_output() -> None:
    """The Interactions path reports grounding in another shape, so it must not log a zero.

    A zero here would read as an ungrounded answer.
    """
    streamer = _streamer(message=FakeMessage(), backend="interactions")

    await _stream(
        streamer=streamer,
        responses=_stream_events_from(
            events=[
                _text_event(delta="watched"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        model=TEST_LLM_MODEL,
                        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
                        output=None,
                    ),
                ),
            ]
        ),
    )

    assert streamer._url_citations is None


@pytest.fixture
def price_table_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes every price-table fetch fail, restoring the process-wide held table after."""

    def refuse(url: str, timeout: int) -> None:
        """Fails the way an unreachable raw.githubusercontent.com does."""
        del url, timeout
        raise requests.ConnectionError("name or service not known")

    monkeypatch.setattr("discordbot.utils.model_pricing._LOADED_TABLE", None)
    monkeypatch.setattr("discordbot.utils.model_pricing.requests.get", refuse)


async def test_streaming_delivers_the_reply_when_the_price_table_is_unavailable(
    price_table_unavailable: None,
) -> None:
    """The footer loses its estimate; the reply that is already on screen is not lost with it."""
    del price_table_unavailable
    message = FakeMessage()

    result = await _stream(streamer=_streamer(message=message), responses=_stream_events())

    assert result == f"hello from stream\n\n-# {TEST_LLM_MODEL} · ⬆ 12 ⬇ 34"
    assert message.replies[0].content == result


async def test_handle_streaming_continues_long_reply_as_reply_chain() -> None:
    """Verifies replies over Discord's content limit continue as a reply chain."""
    message = FakeMessage(content="<@999> explain how long Discord replies are handled")
    body = "x" * 4500

    result = await _stream(
        streamer=_streamer(message=message),
        responses=_stream_events_from(
            events=[
                SimpleNamespace(type="response.output_text.delta", delta=body),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        model=TEST_LLM_MODEL,
                        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
                        output=[],
                    ),
                ),
            ]
        ),
    )

    usage_footer = f"\n\n-# {TEST_LLM_MODEL} · ⬆ 1 ⬇ 2"
    assert result == f"{body}{usage_footer}"

    parent = message.replies[0]
    assert parent.content == body[:DISCORD_MESSAGE_LIMIT]

    first_follow_up = parent.replies[0]
    assert first_follow_up.content == body[DISCORD_MESSAGE_LIMIT : DISCORD_MESSAGE_LIMIT * 2]

    second_follow_up = first_follow_up.replies[0]
    assert second_follow_up.content == f"{body[DISCORD_MESSAGE_LIMIT * 2 :]}{usage_footer}"
    assert second_follow_up.replies == []

    chain_chunks = [parent.content, first_follow_up.content, second_follow_up.content]
    assert all(len(chunk) <= DISCORD_MESSAGE_LIMIT for chunk in chain_chunks)


@pytest.mark.parametrize("error", [make_invalid_form_body(), make_not_found()])
async def test_streaming_falls_back_to_channel_send_when_source_deleted(
    error: nextcord.HTTPException,
) -> None:
    """A deleted source makes the final reply land unparented via channel.send, not crash."""
    message = FakeMessage()
    message.reply_error = error

    result = await _stream(streamer=_streamer(message=message), responses=_stream_events())

    assert message.replies == []  # reply() raised, so nothing was recorded there
    assert message.channel.sent[0].content == result


async def test_streaming_followup_chain_intact_after_channel_send_fallback() -> None:
    """Overflow follow-ups still chain off the unparented parent when the source is gone."""
    message = FakeMessage(content="<@999> explain")
    message.reply_error = make_invalid_form_body()
    body = "x" * 4500

    await _stream(
        streamer=_streamer(message=message),
        responses=_stream_events_from(
            events=[_text_event(delta=body), _completed_event(input_tokens=1, output_tokens=2)]
        ),
    )

    assert message.replies == []
    parent = message.channel.sent[0]
    assert parent.content == body[:DISCORD_MESSAGE_LIMIT]
    # The chain continues off the channel-sent parent, not the deleted source.
    assert parent.replies[0].content == body[DISCORD_MESSAGE_LIMIT : DISCORD_MESSAGE_LIMIT * 2]


async def test_streaming_reraises_non_deletion_http_errors() -> None:
    """A non-deletion HTTP error (e.g. Forbidden) propagates instead of silently channel.send."""
    message = FakeMessage()
    message.reply_error = make_forbidden()

    with pytest.raises(nextcord.HTTPException):
        await _stream(streamer=_streamer(message=message), responses=_stream_events())
    assert message.channel.sent == []


async def test_streaming_tolerates_reply_deleted_before_final_edit() -> None:
    """A reply deleted while streaming ends the turn quietly instead of raising to the cog."""
    message = FakeMessage()
    reply = FakeReply()
    reply.edit_error = make_not_found()
    streamer = _streamer(message=message, reply=reply)

    result = await _stream(streamer=streamer, responses=_stream_events())

    # The caller still gets the full text (memory / research follow-ups stay usable).
    assert result
    assert streamer.reply is None
    # Nothing is re-sent: the user removed that message on purpose.
    assert message.replies == []
    assert message.channel.sent == []


async def test_streaming_reraises_non_deletion_edit_errors() -> None:
    """A non-deletion edit failure (e.g. Forbidden) still propagates as a real error."""
    message = FakeMessage()
    reply = FakeReply()
    reply.edit_error = make_forbidden()

    with pytest.raises(nextcord.HTTPException):
        await _stream(streamer=_streamer(message=message, reply=reply), responses=_stream_events())


async def test_a_refused_preview_write_stops_previewing_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A channel that refuses the live preview is reported once by id and never asked again."""
    message = FakeMessage()
    reply = FakeReply()
    reply.edit_error = make_forbidden()
    streamer = _streamer(
        message=message, reply=as_message(fake=reply), preview_interval_seconds=0.01
    )
    streamer.content_started = True
    streamer.stored_content = "partial answer"
    warned = capture_logs(monkeypatch=monkeypatch, level="warn")

    await asyncio.wait_for(streamer._preview_editor(), timeout=1.0)

    assert warned == [
        (
            "Channel refused a preview write; stopping preview edits",
            {"channel_id": message.channel.id, "message_id": message.id},
        )
    ]


async def test_streamer_reasoning_preview_then_content_overwrites() -> None:
    """The reasoning preview renders as -# subtext and real content replaces it in place."""
    message = FakeMessage()
    streamer = _streamer(message=message)
    streamer.reasoning_content = "first thought\n\nsecond thought"

    await streamer._write_preview_snapshot()
    assert len(message.replies) == 1
    preview = message.replies[0].content
    assert isinstance(preview, str)
    assert preview.splitlines()[0] == "-# <:message:1517560873000898860> Thinking..."
    assert "-# first thought" in preview
    assert "-# second thought" in preview

    streamer.content_started = True
    streamer.stored_content = "real answer"
    await streamer._write_preview_snapshot()
    assert len(message.replies) == 1
    assert message.replies[0].content == "real answer"


def test_streamer_reasoning_preview_keeps_newest_lines_within_limit() -> None:
    """A long think keeps only its newest tail lines within the short preview window."""
    streamer = _streamer(message=FakeMessage())
    streamer.reasoning_content = "\n".join(f"thought line {i} " + "x" * 80 for i in range(60))

    preview = streamer._render_preview()

    assert len(preview) <= DISCORD_MESSAGE_LIMIT
    lines = preview.splitlines()
    assert lines[0] == "-# <:message:1517560873000898860> Thinking..."
    assert all(line.startswith("-# ") for line in lines)
    assert "thought line 59" in preview
    assert "thought line 9 " not in preview
    # Header plus at most the capped number of thought lines, and a short body overall.
    assert len(lines) <= REASONING_PREVIEW_MAX_LINES + 1
    assert len(preview) - len(lines[0]) <= REASONING_PREVIEW_MAX_CHARS + len("-# ") * len(lines)


def test_streamer_reasoning_preview_caps_short_line_count() -> None:
    """Many short thought lines are trimmed to the newest few, not stacked up."""
    streamer = _streamer(message=FakeMessage())
    streamer.reasoning_content = "\n".join(f"step {i}" for i in range(20))

    lines = streamer._render_preview().splitlines()

    assert len(lines) == REASONING_PREVIEW_MAX_LINES + 1
    assert lines[-1] == "-# step 19"
    assert "step 15" not in "\n".join(lines)


def test_streamer_reasoning_preview_keeps_tail_of_one_long_paragraph() -> None:
    """A single paragraph wider than the budget still shows its newest words."""
    streamer = _streamer(message=FakeMessage())
    streamer.reasoning_content = "a" * 900 + " ending words"

    lines = streamer._render_preview().splitlines()

    assert len(lines) == 2
    assert lines[1].startswith("-# …")
    assert lines[1].endswith("ending words")
    assert len(lines[1]) <= REASONING_PREVIEW_MAX_CHARS + len("-# …")


def test_streamer_reasoning_preview_escapes_mentions() -> None:
    """Transient thought text can never ping people or roles."""
    streamer = _streamer(message=FakeMessage())
    streamer.reasoning_content = "should I ping @everyone or <@123456789012345678>?"

    preview = streamer._render_preview()

    assert "@everyone" not in preview
    assert "<@123456789012345678>" not in preview


async def test_streamer_strips_leading_newlines_from_first_reasoning_delta() -> None:
    """Gemini's leading reasoning newlines are dropped like content newlines."""
    events = [
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="\n\n"),
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="\nthought"),
        _text_event(delta="answer"),
        _completed_event(input_tokens=1, output_tokens=1),
    ]
    streamer = _streamer(message=FakeMessage())

    await _stream(streamer=streamer, responses=_stream_events_from(events=events))

    assert streamer.reasoning_content == "thought"


async def test_streamer_edits_are_time_throttled() -> None:
    """The snapshot editor writes far fewer Discord edits than stream deltas."""
    message = FakeMessage()

    async def _events() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(type="response.reasoning_summary_text.delta", delta="thinking hard")
        await asyncio.sleep(0.06)
        for index in range(40):
            yield SimpleNamespace(type="response.output_text.delta", delta=f"chunk{index} ")
            await asyncio.sleep(0.002)
        yield _completed_event(input_tokens=1, output_tokens=1)

    streamer = _streamer(message=message, preview_interval_seconds=0.02)
    result = await _stream(
        streamer=streamer, responses=cast("AsyncIterator[ResponseStreamEvent]", _events())
    )

    assert len(message.replies) == 1
    reply = message.replies[0]
    assert 1 + len(reply.edits) < 40
    assert result.startswith("chunk0 ")
    assert isinstance(reply.content, str)
    assert reply.content.startswith("chunk0 ")


async def test_streamer_footer_shows_route_effort() -> None:
    """The usage footer labels the model with the route-decided effort."""
    message = FakeMessage()

    result = await _stream(
        streamer=_streamer(message=message, model_effort="low"), responses=_stream_events()
    )

    assert f"\n\n-# {TEST_LLM_MODEL} (low) · ⬆ 12 ⬇ 34" in result
    assert USAGE_FOOTER_RE.sub("", result) == "hello from stream"


async def test_streamer_footer_prices_the_model_it_labels_with_the_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The footer prices the bare model name while showing it with its effort, byte for byte.

    Pricing the label instead would find no entry and show no cost.
    """
    rates = {
        TEST_LLM_MODEL: ModelPriceEntry(input_cost_per_token=1e-6, output_cost_per_token=2e-6)
    }
    monkeypatch.setattr("discordbot.utils.model_pricing.load_model_info", lambda: rates)
    message = FakeMessage()

    result = await _stream(
        streamer=_streamer(
            message=message,
            model_effort="high",
            memory_lookups=MemoryCredits(named=("Tester (tester)",)),
        ),
        responses=_stream_events_from(
            events=[
                _text_event(delta="hello"),
                _completed_event(input_tokens=1234, output_tokens=567),
            ]
        ),
    )

    # 1,234 in at $1e-6 plus 567 out at $2e-6.
    assert result == (
        f"hello\n\n-# {TEST_LLM_MODEL} (high) · ⬆ 1,234 ⬇ 567 · $0.00236800"
        "\n-# 📖 讀了 Tester (tester) 的記憶"
    )
    assert message.replies[0].content == result


@pytest.mark.parametrize(
    ("listed", "footer_cost", "logged_cost"),
    [(TEST_LLM_MODEL, " · $0.00000000", 0.0), ("another-model", "", None)],
    ids=["listed-free", "unlisted"],
)
async def test_streamer_footer_shows_a_cost_only_for_a_model_the_table_lists(
    monkeypatch: pytest.MonkeyPatch, listed: str, footer_cost: str, logged_cost: float | None
) -> None:
    """A model listed at zero is free and says so; one the table lacks has no cost to show."""
    rates = {listed: ModelPriceEntry(input_cost_per_token=0.0, output_cost_per_token=0.0)}
    monkeypatch.setattr("discordbot.utils.model_pricing.load_model_info", lambda: rates)
    recorded = capture_logs(monkeypatch=monkeypatch, level="info")

    result = await _stream(streamer=_streamer(message=FakeMessage()), responses=_stream_events())

    assert result == f"hello from stream\n\n-# {TEST_LLM_MODEL} · ⬆ 12 ⬇ 34{footer_cost}"
    finalized = [fields for message, fields in recorded if message == "gen_reply reply finalized"]
    assert finalized[0]["cost"] == logged_cost


def test_an_answer_past_the_follow_up_budget_says_it_was_cut() -> None:
    """Silently losing the tail reads as the model being interrupted; the notice says otherwise."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    content = "字" * 9000

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=footer, max_messages=2
    )

    assert len(chunks) == 1
    assert len(parent) <= 2000
    assert len(chunks[0]) <= 2000
    assert TRUNCATED_NOTICE.strip() in f"{parent}{chunks[0]}"
    assert f"{parent}{chunks[0]}".endswith(footer)


@pytest.mark.parametrize("max_messages", [1, 2])
def test_only_an_answer_past_the_budget_is_cut(max_messages: int) -> None:
    """The notice's room is held back only once something must go, so a fitting answer is whole."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    fits = "字" * (max_messages * DISCORD_MESSAGE_LIMIT - len(footer))

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=fits, footer=footer, max_messages=max_messages
    )
    over_parent, over_chunks = ResponseStreamer._split_reply_for_discord(
        content=f"{fits}字", footer=footer, max_messages=max_messages
    )

    fit = [parent, *chunks]
    assert "".join(fit) == f"{fits}{footer}"
    assert len(fit) <= max_messages
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in fit)
    over = [over_parent, *over_chunks]
    assert len(over) == max_messages
    assert TRUNCATED_NOTICE.strip() in over[-1]


_SPLIT_FOOTER = "\n\n-# model · ⬆ 1 ⬇ 2 · $0.00000001"
_FAMILY_EMOJI = "\U0001f468\u200d\U0001f469\u200d\U0001f467"


def _code_lines(count: int) -> str:
    """Python source of `count` distinct lines, each with spaces a word break could take."""
    return "\n".join(
        f"value_{index:03d} = compute(index={index}, scale=2.5)" for index in range(count)
    )


def test_an_answer_that_fits_only_packed_full_is_not_cut_for_its_breaks() -> None:
    """Clean cuts leave room unused, which a capped surface cannot spare from an answer that fits."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    content = "\n\n".join(["字" * 1400, "字" * 1400, ""])
    content += "字" * (2 * DISCORD_MESSAGE_LIMIT - len(footer) - len(content))

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=footer, max_messages=2
    )
    _, uncapped_chunks = ResponseStreamer._split_reply_for_discord(content=content, footer=footer)

    assert len(uncapped_chunks) > 1, "clean cuts fit the cap anyway, so this test proves nothing"
    assert len(chunks) == 1
    assert f"{parent}{chunks[0]}" == f"{content}{footer}"
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in [parent, *chunks])


def test_an_answer_past_the_budget_is_still_cut_where_a_reader_would() -> None:
    """Running out of messages changes where the answer ends, not how its messages are cut."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    mention = "<@123456789012345678>"
    content = "b" * 1990 + f" {mention} thanks\n\n" + "\n\n".join(["word " * 100] * 25)

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=footer, max_messages=6
    )

    messages = [parent, *chunks]
    assert len(messages) == 6
    assert parent == "b" * 1990
    assert any(mention in message for message in messages)
    assert messages[-1].endswith(f"{TRUNCATED_NOTICE}{footer}")
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in messages)


def test_the_last_message_a_budget_allows_shows_what_it_can_of_a_code_block() -> None:
    """With no message after it, ending before the block would leave the room empty for nothing."""
    content = "prose " * 333 + f"\nHere it is:\n```python\n{_code_lines(count=150)}\n```"

    _parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=_SPLIT_FOOTER, max_messages=2
    )

    assert len(chunks) == 1
    assert chunks[0].count("```") == 2
    assert "value_000 = compute(index=0, scale=2.5)" in chunks[0]
    assert len(chunks[0]) > DISCORD_MESSAGE_LIMIT // 2
    assert chunks[0].endswith(f"{TRUNCATED_NOTICE}{_SPLIT_FOOTER}")


def test_an_uncapped_surface_splits_the_whole_answer() -> None:
    """The gateway path has no budget, so nothing about the existing split changes."""
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    content = "字" * 9000

    parent, chunks = ResponseStreamer._split_reply_for_discord(content=content, footer=footer)

    assert TRUNCATED_NOTICE.strip() not in "".join([parent, *chunks])
    assert "".join([parent, *chunks]) == f"{content}{footer}"


@pytest.mark.parametrize(
    ("content", "whole"),
    [
        pytest.param(
            "x" * 1990 + "\n```python\nprint('hello world')\n```\nafter",
            "```python\nprint('hello world')\n```",
            id="code-block",
        ),
        pytest.param("a" * 1995 + " hello wonderful world", "hello", id="word"),
        pytest.param(
            "b" * 1990 + " <@123456789012345678> thanks", "<@123456789012345678>", id="mention"
        ),
        pytest.param(
            "字" * 1990 + "<@123456789012345678>謝謝",
            "<@123456789012345678>",
            id="mention-no-space",
        ),
        pytest.param("\n\n".join(letter * 1500 for letter in "abc"), "b" * 1500, id="paragraph"),
        pytest.param("c" * 1999 + _FAMILY_EMOJI, _FAMILY_EMOJI, id="zwj-emoji"),
        pytest.param(
            "Type ``` to open a block. " + "a" * 1970 + " hello wonderful world",
            "hello",
            id="unpaired-fence-is-text",
        ),
        pytest.param(
            f"```python\n{_code_lines(count=48)}\n```",
            f"```python\n{_code_lines(count=48)}\n```",
            id="code-block-filling-the-message",
        ),
        pytest.param(
            "word " * 50 + f"\n```python\n{_code_lines(count=44)}\n```",
            f"```python\n{_code_lines(count=44)}\n```",
            id="code-block-after-short-intro",
        ),
    ],
)
def test_a_long_answer_is_split_where_a_reader_would(content: str, whole: str) -> None:
    """Whatever straddles the 2000th character lands whole in one message, and nothing is lost."""
    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=_SPLIT_FOOTER
    )

    messages = [parent, *chunks]
    assert len(messages) > 1, "the answer did not split, so this test proves nothing"
    assert any(whole in message for message in messages)
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in messages)
    assert "".join(messages) == f"{content}{_SPLIT_FOOTER}"


def test_the_footer_still_follows_answer_text_after_a_cut() -> None:
    """A footer alone loses its blank line to Discord's trim, and history then fails to strip it."""
    content = "para " * 300 + "\n\n" + "z" * 1980

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=_SPLIT_FOOTER
    )

    assert parent == "para " * 300
    assert USAGE_FOOTER_RE.search(chunks[-1].strip()) is not None
    assert "".join([parent, *chunks]) == f"{content}{_SPLIT_FOOTER}"


def test_a_code_block_longer_than_a_message_renders_as_code_in_each() -> None:
    """The block starts its own message, and each cut through it closes and reopens the fence."""
    code = _code_lines(count=90)
    content = f"Here is the code:\n```python\n{code}\n```\nThat is all."

    parent, chunks = ResponseStreamer._split_reply_for_discord(
        content=content, footer=_SPLIT_FOOTER
    )

    messages = [parent, *chunks]
    assert parent == "Here is the code:\n"
    assert len(chunks) > 1, "the block did not need cutting, so this test proves nothing"
    assert all(chunk.startswith("```python\n") and chunk.count("```") == 2 for chunk in chunks)
    assert all(any(line in chunk for chunk in chunks) for line in code.splitlines())
    assert all(len(message) <= DISCORD_MESSAGE_LIMIT for message in messages)
    assert "".join(messages).replace("\n``````python", "") == f"{content}{_SPLIT_FOOTER}"


def test_reopening_a_code_block_never_puts_back_what_the_cut_took() -> None:
    """A fence line too long to be a language tag is not carried over, so every cut shrinks the rest.

    Carried over whole, it would be re-added as fast as it is cut off, and the split, which runs
    on the event loop, would never end.
    """
    text = "```" + "a" * 1200 + "\n" + "b" * 1500 + "\n```\nok"

    head, rest = _cut_cleanly(text=text, budget=DISCORD_MESSAGE_LIMIT, earliest=0)

    assert len(head) <= DISCORD_MESSAGE_LIMIT
    assert len(rest) < len(text) - DISCORD_MESSAGE_LIMIT // 2


class _EditableReply:
    """A landed reply that records what each edit wrote to it."""

    def __init__(self) -> None:
        """Initializes the recorded edits."""
        self.contents: list[str] = []

    async def edit(self, **kwargs: Any) -> None:  # noqa: ANN401 -- passthrough Discord payload
        """Records one content edit."""
        self.contents.append(kwargs["content"])


async def test_the_hint_line_lands_on_the_reply_but_not_in_the_transcript() -> None:
    """It has to show, and it has to stay out of what the bot is later told it said."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    reply = _EditableReply()
    streamer = ResponseStreamer(
        message=surface.message, surface=surface, reply=reply, media_delivery=hosting_off_planner()
    )
    streamer.stored_content = f"答案{footer}"
    streamer._usage_footer = footer
    await surface.hint(emoji="⚠️")

    await streamer._write_hint_line()

    assert reply.contents == [f"答案\n-# ⚠️{footer}"]
    assert streamer._without_added_lines(text=streamer.stored_content) == f"答案{footer}"


async def test_a_capped_surface_stops_chunking_where_its_budget_ends() -> None:
    """The budget arithmetic, not just the split: five follow-ups and no sixth attempt."""
    interaction = _interaction()
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")
    footer = "\n\n-# model · ⬆ 1 ⬇ 1 · $0.00000000"
    reply = _EditableReply()
    streamer = ResponseStreamer(
        message=surface.message, surface=surface, reply=reply, media_delivery=hosting_off_planner()
    )

    await streamer._write_final_message(content="字" * 40000, footer=footer)

    assert len(interaction.followup.sent) == INTERACTION_FOLLOWUP_LIMIT
    written = reply.contents[-1] + "".join(sent["content"] for sent in interaction.followup.sent)
    assert TRUNCATED_NOTICE.strip() in written
    assert written.endswith(footer)
