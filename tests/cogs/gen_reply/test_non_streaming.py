"""Tests for the non-streaming answer path (`global.reply.streaming_enabled = false`)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
import asyncio

import pytest
from google.genai.errors import APIError
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.cogs.gen_reply import non_streaming as non_streaming_module
from discordbot.typings.models import SlowModelSettings, RuntimeModelCatalog
from discordbot.utils.llm_errors import HIGH_DEMAND_ERROR_NOTICE, extract_friendly_error
from discordbot.utils.preview_editor import CadencePreviewEditor
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.surface import WINDOW_EXPIRED_NOTICE, TurnSurface
from discordbot.cogs.gen_reply.streaming import (
    ResponseStreamer,
    AnswerStreamError,
    IncompleteAnswerError,
)
from discordbot.cogs.gen_reply.non_streaming import (
    complete_responses_answer,
    create_interactions_answer,
    complete_interactions_answer,
)

from tests.helpers.casting import as_message
from tests.helpers.gen_reply import FakeGeminiClient
from tests.helpers.reply_fakes import (
    TEST_LLM_MODEL,
    FakeReply,
    FakeAuthor,
    FakeMessage,
    _cog,
    _answer,
    _recorded,
    _streamer,
    _config_stub,
    _expiring_surface,
)
from tests.helpers.link_sources import hosting_off_planner
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from google import genai
    from openai.types.responses import Response

_REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"


def _responses_result(  # noqa: PLR0913 -- builder for test Response stubs
    text: str = "hello",
    status: str = "completed",
    model: str = TEST_LLM_MODEL,
    input_tokens: int = 10,
    output_tokens: int = 20,
    reasoning_summaries: list[str] | None = None,
    annotations: list[SimpleNamespace] | None = None,
    web_search_requests: int = 0,
    error: SimpleNamespace | None = None,
    incomplete_details: SimpleNamespace | None = None,
) -> Response:
    """Builds a minimal `Response`-shaped stub for non-streaming Responses tests."""
    output: list[SimpleNamespace] = []
    if reasoning_summaries:
        output.append(
            SimpleNamespace(
                type="reasoning",
                summary=[
                    SimpleNamespace(type="summary_text", text=s) for s in reasoning_summaries
                ],
            )
        )
    output.append(
        SimpleNamespace(
            type="message",
            content=[
                SimpleNamespace(type="output_text", text=text, annotations=annotations or [])
            ],
        )
    )
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(
            model_extra={"web_search_requests": web_search_requests}
        ),
    )
    return cast(
        "Response",
        SimpleNamespace(
            status=status,
            model=model,
            output=output,
            output_text=text,
            usage=usage,
            error=error,
            incomplete_details=incomplete_details,
        ),
    )


async def test_complete_responses_answer_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetches a completed Response, populates telemetry/tool use, and finalizes without previews."""
    logged = capture_logs(monkeypatch=monkeypatch, level="info")
    preview_starts = 0

    def forbid_preview_start(self: CadencePreviewEditor) -> None:
        nonlocal preview_starts
        del self
        preview_starts += 1

    monkeypatch.setattr(CadencePreviewEditor, "start", forbid_preview_start)

    captured_hook_seconds: list[float] = []

    async def on_final(active: ResponseStreamer) -> None:
        captured_hook_seconds.append(active.answer_seconds)

    message = FakeMessage()
    streamer = _streamer(message=message)
    streamer.streaming = False
    streamer.on_final_message = on_final

    result = _responses_result(
        text="\n\nGrounded answer",
        model="fallback-model",
        input_tokens=15,
        output_tokens=30,
        reasoning_summaries=["think one ", "think two"],
        annotations=[
            SimpleNamespace(type="url_citation", url=f"{_REDIRECT}abc", title="example.com"),
            SimpleNamespace(type="url_citation", url="https://news.org/story", title="Story"),
        ],
        web_search_requests=2,
    )

    async def fetch() -> Response:
        return result

    final = await complete_responses_answer(streamer=streamer, fetch=fetch)

    assert final.startswith("Grounded answer")
    assert "fallback-model" in final
    assert streamer.model_name == "fallback-model"
    assert streamer.input_tokens == 15
    assert streamer.output_tokens == 30
    assert streamer.tool_use.web_search_queries == 2
    assert streamer.tool_use.url_fetches is None
    assert streamer.tool_use.url_citations == 2
    assert streamer.tool_use.cited_domains == ["example.com", "news.org"]
    assert preview_starts == 0
    assert streamer._editor is None
    assert captured_hook_seconds == [streamer.answer_seconds]
    assert streamer.answer_seconds >= 0.0
    finalized = [fields for name, fields in logged if name == "gen_reply reply finalized"]
    assert len(finalized) == 1
    assert finalized[0]["streaming"] is False
    assert finalized[0]["model"] == "fallback-model"
    assert finalized[0]["reasoning_chars"] == len("think one think two")


async def test_complete_responses_answer_raises_on_failed_status() -> None:
    """`status='failed'` records model/usage and raises `AnswerStreamError` with mapped status."""
    streamer = _streamer(message=FakeMessage())
    streamer.streaming = False

    async def fetch() -> Response:
        return _responses_result(
            text="",
            status="failed",
            model="failed-model",
            input_tokens=7,
            output_tokens=0,
            error=SimpleNamespace(code="rate_limit_exceeded", message="quota hit"),
        )

    with pytest.raises(AnswerStreamError) as exc_info:
        await complete_responses_answer(streamer=streamer, fetch=fetch)

    assert exc_info.value.code == "rate_limit_exceeded"
    assert exc_info.value.status_code == 429
    assert extract_friendly_error(exc=exc_info.value) == HIGH_DEMAND_ERROR_NOTICE
    assert streamer.model_name == "failed-model"
    assert streamer.input_tokens == 7
    assert streamer.content_started is False
    assert streamer.reply is None


@pytest.mark.parametrize(
    ("status", "details", "expected_reason"),
    [
        ("incomplete", SimpleNamespace(reason="max_output_tokens"), "max_output_tokens"),
        ("incomplete", None, "incomplete"),
        ("cancelled", None, "cancelled"),
    ],
)
async def test_complete_responses_answer_raises_on_non_completed_status(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    details: SimpleNamespace | None,
    expected_reason: str,
) -> None:
    """Non-completed Responses log `gen_reply answer incomplete` and raise `IncompleteAnswerError`."""
    logged = capture_logs(monkeypatch=monkeypatch, level="warn")
    streamer = _streamer(message=FakeMessage())
    streamer.streaming = False

    async def fetch() -> Response:
        return _responses_result(
            text="\npartial text",
            status=status,
            reasoning_summaries=["partial thought"],
            incomplete_details=details,
        )

    with pytest.raises(IncompleteAnswerError) as exc_info:
        await complete_responses_answer(streamer=streamer, fetch=fetch)

    assert exc_info.value.reason == expected_reason
    assert streamer.content_started is False
    warnings = [fields for name, fields in logged if name == "gen_reply answer incomplete"]
    assert len(warnings) == 1
    assert warnings[0]["reason"] == expected_reason
    assert warnings[0]["reply_chars"] == len("partial text")
    assert warnings[0]["reasoning_chars"] == len("partial thought")


async def test_complete_responses_answer_honors_ask_delivery_window() -> None:
    """An `/ask` non-streaming Responses call expiring its delivery budget raises `WINDOW_EXPIRED_NOTICE`."""
    message = FakeMessage()
    typed = as_message(fake=message)
    streamer = ResponseStreamer(
        message=typed,
        surface=_expiring_surface(message=typed, seconds_left=0.05),
        media_delivery=hosting_off_planner(),
        streaming=False,
    )

    async def slow_fetch() -> Response:
        await asyncio.sleep(5)
        return _responses_result()

    with pytest.raises(TimeoutError) as raised:
        await asyncio.wait_for(
            complete_responses_answer(streamer=streamer, fetch=slow_fetch), timeout=5
        )

    assert str(raised.value) == WINDOW_EXPIRED_NOTICE


async def test_complete_responses_answer_own_timeout_is_not_blamed_on_ask_window() -> None:
    """A provider timeout inside a healthy `/ask` window keeps its own error."""
    message = FakeMessage()
    typed = as_message(fake=message)
    streamer = ResponseStreamer(
        message=typed,
        surface=_expiring_surface(message=typed, seconds_left=300),
        media_delivery=hosting_off_planner(),
        streaming=False,
    )

    async def timing_out_fetch() -> Response:
        raise TimeoutError

    with pytest.raises(TimeoutError) as raised:
        await complete_responses_answer(streamer=streamer, fetch=timing_out_fetch)

    assert str(raised.value) != WINDOW_EXPIRED_NOTICE


async def test_complete_interactions_answer_honors_ask_delivery_window() -> None:
    """An `/ask` non-streaming Interactions call expiring its delivery budget raises `WINDOW_EXPIRED_NOTICE`."""
    message = FakeMessage()
    typed = as_message(fake=message)
    streamer = ResponseStreamer(
        message=typed,
        surface=_expiring_surface(message=typed, seconds_left=0.05),
        media_delivery=hosting_off_planner(),
        backend="interactions",
        streaming=False,
    )

    async def slow_fetch() -> object:
        await asyncio.sleep(5)
        return SimpleNamespace(status="completed", output_text="late")

    with pytest.raises(TimeoutError) as raised:
        await asyncio.wait_for(
            complete_interactions_answer(streamer=streamer, fetch=slow_fetch), timeout=5
        )

    assert str(raised.value) == WINDOW_EXPIRED_NOTICE


async def test_complete_interactions_answer_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetches a non-streaming Interactions answer and extracts text, reasoning, usage, and tool use."""
    logged = capture_logs(monkeypatch=monkeypatch, level="info")
    recorded_calls: list[dict[str, object]] = []

    fake_interaction = SimpleNamespace(
        status="completed",
        model="gemini-3.1-pro-preview",
        output_text="\n\nWatched video summary",
        errors=None,
        usage=SimpleNamespace(total_input_tokens=120, total_output_tokens=45),
        steps=[
            SimpleNamespace(
                type="thought", summary=[SimpleNamespace(type="text", text="analyzing video")]
            ),
            SimpleNamespace(
                type="google_search_call", arguments=SimpleNamespace(queries=["q1", "q2", "q1"])
            ),
            SimpleNamespace(
                type="url_context_call", arguments=SimpleNamespace(urls=["https://docs.test/page"])
            ),
            SimpleNamespace(
                type="model_output",
                content=[
                    SimpleNamespace(
                        type="text",
                        text="\n\nWatched video summary",
                        annotations=[
                            SimpleNamespace(
                                type="url_citation", url="https://docs.test/page", title="Docs"
                            ),
                            SimpleNamespace(
                                type="url_citation", url=f"{_REDIRECT}xyz", title="wiki.example"
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    class _FakeInteractions:
        async def create(self, **kwargs: object) -> object:
            recorded_calls.append(kwargs)
            return fake_interaction

    gemini_client = cast("genai.Client", FakeGeminiClient(interactions=_FakeInteractions()))
    streamer = _streamer(message=FakeMessage())
    streamer.backend = "interactions"
    streamer.streaming = False

    async def fetch() -> object:
        return await create_interactions_answer(
            client=gemini_client,
            model="gemini-3.1-pro-preview",
            system_instruction="SYS",
            steps=[],
            effort="medium",
        )

    final = await complete_interactions_answer(streamer=streamer, fetch=fetch)

    assert len(recorded_calls) == 1
    assert recorded_calls[0]["stream"] is False
    assert final.startswith("Watched video summary")
    assert streamer.model_name == "gemini-3.1-pro-preview"
    assert streamer.input_tokens == 120
    assert streamer.output_tokens == 45
    assert streamer.tool_use.web_search_queries == 2
    assert streamer.tool_use.url_fetches == 1
    assert streamer.tool_use.url_citations == 2
    assert streamer.tool_use.cited_domains == ["docs.test", "wiki.example"]
    assert streamer._editor is None
    finalized = [fields for name, fields in logged if name == "gen_reply reply finalized"]
    assert len(finalized) == 1
    assert finalized[0]["streaming"] is False
    assert finalized[0]["backend"] == "interactions"
    assert finalized[0]["reasoning_chars"] == len("analyzing video")


async def test_complete_interactions_answer_falls_back_to_model_output_steps() -> None:
    """When `output_text` is None, text is assembled from `model_output` steps only."""
    streamer = _streamer(message=FakeMessage())
    streamer.backend = "interactions"
    streamer.streaming = False

    async def fetch() -> object:
        return SimpleNamespace(
            status="completed",
            model="gemini-3.1-pro-preview",
            output_text=None,
            errors=None,
            usage=None,
            steps=[
                SimpleNamespace(
                    type="user_input",
                    content=[SimpleNamespace(type="text", text="user prompt must be ignored")],
                ),
                SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text="From steps", annotations=[])],
                ),
            ],
        )

    final = await complete_interactions_answer(streamer=streamer, fetch=fetch)

    assert final.startswith("From steps")
    assert "user prompt must be ignored" not in final


async def test_complete_interactions_answer_raises_api_error_on_errors() -> None:
    """Non-empty `interaction.errors` on a non-completed interaction raises `APIError`."""
    streamer = _streamer(message=FakeMessage())
    streamer.backend = "interactions"
    streamer.streaming = False

    async def fetch() -> object:
        return SimpleNamespace(
            status="failed",
            model="gemini-3.1-pro-preview",
            output_text="",
            errors=[SimpleNamespace(code="503", message="backend busy")],
            usage=None,
            steps=None,
        )

    with pytest.raises(APIError) as exc_info:
        await complete_interactions_answer(streamer=streamer, fetch=fetch)

    assert exc_info.value.code == 503
    assert extract_friendly_error(exc=exc_info.value) == HIGH_DEMAND_ERROR_NOTICE


async def test_complete_interactions_answer_keeps_completed_reply_despite_diagnostic_errors() -> (
    None
):
    """A completed interaction carrying non-fatal diagnostic `errors` still delivers its answer."""
    streamer = _streamer(message=FakeMessage())
    streamer.backend = "interactions"
    streamer.streaming = False

    async def fetch() -> object:
        return SimpleNamespace(
            status="completed",
            model="gemini-3.1-pro-preview",
            output_text="Finished despite warning",
            errors=[SimpleNamespace(code="DIAGNOSTIC", message="non-fatal note")],
            usage=SimpleNamespace(total_input_tokens=4, total_output_tokens=6),
            steps=[],
        )

    final = await complete_interactions_answer(streamer=streamer, fetch=fetch)

    assert final.startswith("Finished despite warning")


async def test_create_interactions_answer_bounds_stalled_gateway_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled direct-to-Google non-streaming call times out even without an `/ask` deadline."""
    monkeypatch.setattr(non_streaming_module, "INTERACTIONS_ANSWER_TIMEOUT_SECONDS", 0.05)

    class _StalledInteractions:
        async def create(self, **kwargs: object) -> object:
            del kwargs
            await asyncio.sleep(5)
            return SimpleNamespace(status="completed", output_text="late")

    gemini_client = cast("genai.Client", FakeGeminiClient(interactions=_StalledInteractions()))
    streamer = _streamer(message=FakeMessage())
    streamer.backend = "interactions"
    streamer.streaming = False

    async def fetch() -> object:
        return await create_interactions_answer(
            client=gemini_client,
            model="gemini-3.1-pro-preview",
            system_instruction="SYS",
            steps=[],
            effort="medium",
        )

    with pytest.raises(TimeoutError) as raised:
        await asyncio.wait_for(
            complete_interactions_answer(streamer=streamer, fetch=fetch), timeout=5
        )

    assert str(raised.value) != WINDOW_EXPIRED_NOTICE


async def test_complete_interactions_answer_raises_incomplete_on_non_completed_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-completed status without `errors` logs warning and raises `IncompleteAnswerError`."""
    logged = capture_logs(monkeypatch=monkeypatch, level="warn")
    streamer = _streamer(message=FakeMessage())
    streamer.backend = "interactions"
    streamer.streaming = False

    async def fetch() -> object:
        return SimpleNamespace(
            status="budget_exceeded",
            model="gemini-3.1-pro-preview",
            output_text="halfway",
            errors=None,
            usage=SimpleNamespace(total_input_tokens=5, total_output_tokens=3),
            steps=[],
        )

    with pytest.raises(IncompleteAnswerError) as exc_info:
        await complete_interactions_answer(streamer=streamer, fetch=fetch)

    assert exc_info.value.reason == "budget_exceeded"
    warnings = [fields for name, fields in logged if name == "gen_reply answer incomplete"]
    assert len(warnings) == 1
    assert warnings[0]["reason"] == "budget_exceeded"
    assert warnings[0]["reply_chars"] == len("halfway")


@pytest.mark.parametrize("streaming_enabled", [False, True], ids=["non_streaming", "streaming"])
@pytest.mark.usefixtures("no_memory_review")
async def test_answer_turn_dispatches_qa_responses_by_streaming_switch(
    monkeypatch: pytest.MonkeyPatch, streaming_enabled: bool
) -> None:
    """`AnswerTurn.stream_answer` chooses `stream=False` or `stream=True` from `LLMConfig`."""
    logged = capture_logs(monkeypatch=monkeypatch, level="info")
    cog = _cog()
    cog.config = _config_stub(answer_streaming_enabled=streaming_enabled)
    if not streaming_enabled:
        _recorded(cog).responses.response_queue.append(
            cast("SimpleNamespace", _responses_result(text="Non-stream QA reply"))
        )
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    typed_message = as_message(fake=message)

    await _answer(cog=cog, message=typed_message).stream_answer(
        system_prompt="SYS", context=ReplyContext()
    )

    assert _recorded(cog).responses.create_streams == [streaming_enabled]
    dispatch = [fields for name, fields in logged if name == "gen_reply answer dispatch"]
    finalized = [fields for name, fields in logged if name == "gen_reply reply finalized"]
    assert dispatch[0]["streaming"] is streaming_enabled
    assert finalized[0]["streaming"] is streaming_enabled
    if not streaming_enabled:
        assert message.replies[0].content is not None
        assert message.replies[0].content.startswith("Non-stream QA reply")


@pytest.mark.usefixtures("no_memory_review")
async def test_answer_turn_dispatches_youtube_interactions_non_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YouTube QA turns dispatch `interactions.create(..., stream=False)` when streaming is off."""
    logged = capture_logs(monkeypatch=monkeypatch, level="info")
    monkeypatch.setattr(
        RuntimeModelCatalog,
        "slow_candidates",
        property(lambda _self: (SlowModelSettings(name="gemini-test", effort="high"),)),
    )
    recorded_streams: list[object] = []

    class _FakeInteractions:
        async def create(self, **kwargs: object) -> object:
            recorded_streams.append(kwargs.get("stream"))
            return SimpleNamespace(
                status="completed",
                model="gemini-test",
                output_text="Non-stream YT reply",
                errors=None,
                usage=SimpleNamespace(total_input_tokens=10, total_output_tokens=5),
                steps=[],
            )

    cog = _cog()
    cog.config = _config_stub(
        gemini_api_key="test-key", answer_streaming_enabled=False, feedback_enabled=False
    )
    cog.toolkit.__dict__["gemini_client"] = FakeGeminiClient(interactions=_FakeInteractions())
    message = FakeMessage(
        content="<@999> https://youtu.be/abcdefghijk", author=FakeAuthor(user_id=1)
    )
    typed_message = as_message(fake=message)

    await _answer(cog=cog, message=typed_message).stream_answer(
        system_prompt="SYS", context=ReplyContext(), yt_url="https://youtu.be/abcdefghijk"
    )

    assert recorded_streams == [False]
    assert _recorded(cog).responses.create_streams == []
    dispatch = [fields for name, fields in logged if name == "gen_reply answer dispatch"]
    assert dispatch[0]["backend"] == "interactions"
    assert dispatch[0]["streaming"] is False
    assert message.replies[0].content is not None
    assert message.replies[0].content.startswith("Non-stream YT reply")


@pytest.mark.usefixtures("no_memory_review")
async def test_answer_turn_dispatches_media_persona_non_streaming() -> None:
    """Media persona replies use `stream=False` when `answer_streaming_enabled=False`."""
    cog = _cog()
    cog.config = _config_stub(answer_streaming_enabled=False)
    _recorded(cog).responses.response_queue.append(
        cast("SimpleNamespace", _responses_result(text="Here is your drawing!"))
    )
    message = FakeMessage(content="<@999> draw a cat", author=FakeAuthor(user_id=1))
    typed_message = as_message(fake=message)
    delivered = FakeReply()
    context_task = asyncio.create_task(coro=asyncio.sleep(0, result=ReplyContext()))

    await _answer(cog=cog, message=typed_message).stream_media_persona_reply(
        reply=as_message(fake=delivered),
        context_task=context_task,
        system_prompt="SYS",
        focus_part=ResponseInputImageParam(
            type="input_image", image_url="https://files.test/img.png", detail="auto"
        ),
        media_noun="image",
    )

    assert _recorded(cog).responses.create_streams == [False]
    assert delivered.content is not None
    assert delivered.content.startswith("Here is your drawing!")
    assert len(delivered.edits) == 1


@pytest.mark.usefixtures("no_memory_review")
async def test_hosted_media_persona_non_streaming_failure_deletes_base_ping() -> None:
    """When a non-streaming hosted media persona reply fails, its unedited ping is deleted."""
    cog = _cog()
    cog.config = _config_stub(answer_streaming_enabled=False)
    _recorded(cog).responses.response_queue.append(
        cast("SimpleNamespace", _responses_result(text="partial", status="incomplete"))
    )
    message = FakeMessage(content="<@999> draw a cat", author=FakeAuthor(user_id=1))
    typed_message = as_message(fake=message)
    context_task = asyncio.create_task(coro=asyncio.sleep(0, result=ReplyContext()))

    await _answer(cog=cog, message=typed_message).stream_media_persona_reply(
        reply=None,
        context_task=context_task,
        system_prompt="SYS",
        focus_part=ResponseInputImageParam(
            type="input_image", image_url="https://files.test/img.png", detail="auto"
        ),
        media_noun="image",
    )

    assert len(message.replies) == 1
    assert message.replies[0].deleted is True


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (AnswerStreamError(message="upstream overloaded", code="server_error"), 500),
        (RuntimeError("unexpected failure"), None),
    ],
    ids=["mapped_status", "plain_error"],
)
@pytest.mark.usefixtures("no_memory_review")
async def test_cog_run_turn_logs_streaming_elapsed_and_status_on_failure(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected_status: int | None
) -> None:
    """`gen_reply failed` logs `streaming`, `elapsed_seconds`, and `status_code` for observability."""
    logged = capture_logs(monkeypatch=monkeypatch, level="error")
    cog = _cog()
    cog.config = _config_stub(answer_streaming_enabled=False)
    _recorded(cog).responses.response_queue.append(error)
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    surface = TurnSurface(message=as_message(fake=message))

    await cog._run_turn(surface=surface, content="hi")

    failed = [fields for name, fields in logged if name == "gen_reply failed"]
    assert len(failed) == 1
    assert failed[0]["streaming"] is False
    assert isinstance(failed[0]["elapsed_seconds"], float)
    assert failed[0]["elapsed_seconds"] >= 0.0
    assert failed[0]["status_code"] == expected_status
