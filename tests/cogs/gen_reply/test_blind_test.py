"""Tests for `[global.models.slow]` candidate sampling, label masking, per-model fallback, and mixed providers."""

import asyncio
from pathlib import Path
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.config import clear_bot_config_cache
from discordbot.typings.models import RouteClassification, RuntimeModelCatalog
from discordbot.utils.model_pricing import ModelPriceEntry, _LoadedTable
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.prompts import REPLY_PROMPT, IMAGE_REPLY_PROMPT
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.feedback_store import ReplyTurnRow, open_session
from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs.gen_reply.attachment.gemini_file_api import GeminiFileUploader

from tests.helpers.casting import as_message
from tests.helpers.gen_reply import FakeGeminiFiles
from tests.helpers.reply_fakes import (
    FakeClient,
    FakeMessage,
    FakeAttachment,
    FakeGeminiClient,
    _cog,
    _recorded,
    _text_event,
    _run_pipeline,
    _completed_event,
    _stream_events_from,
    hosting_off_planner,
)
from tests.helpers.discord_mocks import stub_bot

_GEMINI_NEXT_POOL_TOML = (
    '[global.models.slow]\ndisplay_name = "Gemini Next"\n'
    "candidates = [\n"
    '  { name = "gemini-4-neon", effort = "low" },\n'
    '  { name = "gemini-4-hydrogen", effort = "medium" },\n'
    "]\n"
)


async def test_slow_pool_masks_visible_label_and_records_real_model(
    bot_config_isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configured `display_name` masks prompt and footer while storing real model and effort."""
    bot_config_isolated.write_text(data=_GEMINI_NEXT_POOL_TOML, encoding="utf-8")  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO
    clear_bot_config_cache()
    cog = _cog()
    message = FakeMessage(content="<@999> 你是哪個模型?")
    seen_calls: list[dict[str, object]] = []
    finalized_logs: list[dict[str, object]] = []

    monkeypatch.setattr("discordbot.cogs.gen_reply.answer.secrets.choice", lambda seq: seq[1])
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.streaming.logfire.info",
        lambda msg, **kwargs: (
            finalized_logs.append(kwargs) if msg == "gen_reply reply finalized" else None
        ),
    )
    monkeypatch.setattr(
        "discordbot.utils.model_pricing._LOADED_TABLE",
        _LoadedTable(
            prices={
                "gemini/gemini-4-hydrogen": ModelPriceEntry(
                    input_cost_per_token=2e-6, output_cost_per_token=1e-5
                )
            },
            served_at=0.0,
        ),
    )

    async def fake_create(**kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        seen_calls.append(kwargs)
        completed = _completed_event(input_tokens=120, output_tokens=45)
        completed.response.model = "gemini-4-hydrogen"
        return _stream_events_from(events=[_text_event(delta="我是破貓"), completed])

    monkeypatch.setattr(cog.openai_client.responses, "create", fake_create)
    monkeypatch.setattr(AnswerTurn, "_schedule_memory_updates", lambda *_args, **_kwargs: None)

    turn = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=message)),
    )
    await turn.stream_answer(
        system_prompt=REPLY_PROMPT, context=ReplyContext(), effort="high", question="你是哪個模型?"
    )

    assert len(seen_calls) == 1
    assert seen_calls[0]["model"] == "gemini-4-hydrogen"
    assert seen_calls[0]["reasoning"] == {"effort": "high", "summary": "auto"}
    instructions = str(seen_calls[0]["instructions"])
    assert "(Gemini Next)" in instructions
    assert "{model_label}" not in instructions
    assert "gemini-4-neon" not in instructions
    assert "gemini-4-hydrogen" not in instructions

    reply_text = message.replies[0].content or ""
    assert "-# Gemini Next · ⬆ 120 ⬇ 45 · $0.00069000" in reply_text
    assert "(high)" not in reply_text
    assert "gemini-4-hydrogen" not in reply_text
    assert USAGE_FOOTER_RE.sub(repl="", string=reply_text).strip() == "我是破貓"

    async with open_session() as session:
        rows = list((await session.execute(select(ReplyTurnRow))).scalars())
    assert len(rows) == 1
    assert rows[0].model == "gemini-4-hydrogen"
    assert rows[0].effort == "high"
    assert len(finalized_logs) == 1
    assert finalized_logs[0]["model"] == "gemini-4-hydrogen"
    assert finalized_logs[0]["effort"] == "high"


async def test_slow_pool_records_real_post_fallback_model_while_hiding_footer(
    bot_config_isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even after a LiteLLM fallback, footer stays masked while `reply_turn` keeps the real model."""
    bot_config_isolated.write_text(data=_GEMINI_NEXT_POOL_TOML, encoding="utf-8")  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO
    clear_bot_config_cache()
    cog = _cog()
    message = FakeMessage(content="<@999> test fallback")

    monkeypatch.setattr("discordbot.cogs.gen_reply.answer.secrets.choice", lambda seq: seq[0])

    async def fake_create(**_kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        completed = _completed_event(input_tokens=10, output_tokens=5)
        completed.response.model = "gemini-4-hydrogen"
        return _stream_events_from(events=[_text_event(delta="ok"), completed])

    monkeypatch.setattr(cog.openai_client.responses, "create", fake_create)
    monkeypatch.setattr(AnswerTurn, "_schedule_memory_updates", lambda *_args, **_kwargs: None)

    turn = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=message)),
    )
    await turn.stream_answer(
        system_prompt=REPLY_PROMPT,
        context=ReplyContext(),
        effort="medium",
        question="test fallback",
    )

    reply_text = message.replies[0].content or ""
    assert "-# Gemini Next · ⬆ 10 ⬇ 5" in reply_text
    assert "(medium)" not in reply_text
    assert "gemini-4-hydrogen" not in reply_text

    async with open_session() as session:
        rows = list((await session.execute(select(ReplyTurnRow))).scalars())
    assert len(rows) == 1
    assert rows[0].model == "gemini-4-hydrogen"
    assert rows[0].effort == "medium"


async def test_unmasked_multi_candidate_pool_shows_drawn_model_in_prompt_and_footer(
    bot_config_isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With empty `display_name`, prompt and footer both identify the actual drawn candidate."""
    bot_config_isolated.write_text(  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO
        data=(
            '[global.models.slow]\ndisplay_name = ""\n'
            "candidates = [\n"
            '  { name = "gemini-4-neon", effort = "low" },\n'
            '  { name = "gemini-4-hydrogen", effort = "medium" },\n'
            "]\n"
        ),
        encoding="utf-8",
    )
    clear_bot_config_cache()
    cog = _cog()
    message = FakeMessage(content="<@999> who are you")
    seen_instructions: list[str] = []

    monkeypatch.setattr("discordbot.cogs.gen_reply.answer.secrets.choice", lambda seq: seq[1])

    async def fake_create(**kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        seen_instructions.append(str(kwargs["instructions"]))
        completed = _completed_event(input_tokens=10, output_tokens=5)
        completed.response.model = "gemini-4-hydrogen"
        return _stream_events_from(events=[_text_event(delta="hi"), completed])

    monkeypatch.setattr(cog.openai_client.responses, "create", fake_create)
    monkeypatch.setattr(AnswerTurn, "_schedule_memory_updates", lambda *_args, **_kwargs: None)

    turn = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=message)),
    )
    await turn.stream_answer(
        system_prompt=REPLY_PROMPT, context=ReplyContext(), effort="low", question="who are you"
    )

    assert "(gemini-4-hydrogen)" in seen_instructions[0]
    assert "gemini-4-neon" not in seen_instructions[0]
    reply_text = message.replies[0].content or ""
    assert "-# gemini-4-hydrogen (low)" in reply_text


async def test_default_single_candidate_dispatches_primary_model_with_visible_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packaged defaults dispatch the single slow model and display its real name and effort."""
    cog = _cog()
    message = FakeMessage(content="<@999> normal turn")
    expected_model = RuntimeModelCatalog().slow_model.name
    seen_models: list[object] = []

    async def fake_create(**kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        seen_models.append(kwargs["model"])
        completed = _completed_event(input_tokens=10, output_tokens=5)
        completed.response.model = expected_model
        return _stream_events_from(events=[_text_event(delta="normal"), completed])

    monkeypatch.setattr(cog.openai_client.responses, "create", fake_create)
    monkeypatch.setattr(AnswerTurn, "_schedule_memory_updates", lambda *_args, **_kwargs: None)

    turn = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=message)),
    )
    await turn.stream_answer(
        system_prompt=REPLY_PROMPT, context=ReplyContext(), effort="low", question="normal turn"
    )

    assert seen_models == [expected_model]
    reply_text = message.replies[0].content or ""
    assert f"-# {expected_model} (low)" in reply_text


@pytest.mark.parametrize(
    ("pick_index", "expected_model", "expected_fallback_effort"),
    [(0, "gemini-4-neon", "low"), (1, "gemini-4-hydrogen", "medium")],
    ids=["neon-falls-back-low", "hydrogen-falls-back-medium"],
)
@pytest.mark.usefixtures("no_memory_review")
async def test_pipeline_uses_selected_candidate_effort_only_on_router_failure(
    bot_config_isolated: Path,
    monkeypatch: pytest.MonkeyPatch,
    pick_index: int,
    expected_model: str,
    expected_fallback_effort: str,
) -> None:
    """Router failure falls back to the drawn candidate's effort; normal routing still overrides it."""
    bot_config_isolated.write_text(data=_GEMINI_NEXT_POOL_TOML, encoding="utf-8")  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO
    clear_bot_config_cache()
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.answer.secrets.choice", lambda seq: seq[pick_index]
    )

    # First turn: router returns no parsed output -> candidate's fallback effort applies.
    _recorded(cog).responses.output_parsed = None
    _recorded(cog).responses.stream_queue = [
        [_text_event(delta="fallback ok"), _completed_event(input_tokens=5, output_tokens=5)]
    ]
    await _run_pipeline(cog=cog, message=FakeMessage(content="<@999> fallback test"))

    assert _recorded(cog).responses.create_models[-1] == expected_model
    assert _recorded(cog).responses.create_reasonings[-1] == {
        "effort": expected_fallback_effort,
        "summary": "auto",
    }

    # Second turn: router succeeds with "high" -> overrides both candidates' fallback efforts.
    _recorded(cog).responses.output_parsed = RouteClassification(decision="QA", effort="high")
    _recorded(cog).responses.stream_queue = [
        [_text_event(delta="routed ok"), _completed_event(input_tokens=5, output_tokens=5)]
    ]
    await _run_pipeline(cog=cog, message=FakeMessage(content="<@999> routed test"))

    assert _recorded(cog).responses.create_models[-1] == expected_model
    assert _recorded(cog).responses.create_reasonings[-1] == {"effort": "high", "summary": "auto"}


async def test_toolkit_input_builder_for_reuses_gemini_uploads_and_isolates_mixed_providers(
    bot_config_isolated: Path,
) -> None:
    """Gemini slow candidates share one Files API upload cache; non-Gemini candidates stay isolated."""
    bot_config_isolated.write_text(  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO
        data=(
            "[global.models.slow]\n"
            "candidates = [\n"
            '  { name = "gemini-4-neon", effort = "low" },\n'
            '  { name = "gemini-4-hydrogen", effort = "medium" },\n'
            '  { name = "claude-sonnet-test", effort = "medium" },\n'
            "]\n"
        ),
        encoding="utf-8",
    )
    clear_bot_config_cache()

    toolkit = ReplyToolkit(
        bot=stub_bot(),
        openai_client=FakeClient(),
        config=LLMConfig.model_construct(gemini_api_key="test-key"),
    )
    fake_files = FakeGeminiFiles()
    toolkit.__dict__["gemini_client"] = FakeGeminiClient(files=fake_files)
    neon, hydrogen, claude = toolkit.runtime_models.slow_candidates

    neon_builder = toolkit.input_builder_for(model=neon)
    hydrogen_builder = toolkit.input_builder_for(model=hydrogen)
    claude_builder = toolkit.input_builder_for(model=claude)

    assert isinstance(neon_builder.attachment_handler, GeminiFileUploader)
    assert hydrogen_builder.attachment_handler is neon_builder.attachment_handler
    assert hydrogen_builder._attachment_cache is neon_builder._attachment_cache
    assert hydrogen_builder is not neon_builder
    assert isinstance(claude_builder.attachment_handler, InlineRenderer)
    assert claude_builder._attachment_cache is not neon_builder._attachment_cache
    assert toolkit.input_builder is neon_builder

    # Rendering the same attachment across two Gemini candidates performs only one Files API upload,
    # whereas the non-Gemini candidate renders inline without receiving the Gemini Files URI.
    message = FakeMessage(content="look")
    message.attachments = [
        FakeAttachment(filename="pic.png", content_type="image/png", payload=b"\x89PNG\r\n")
    ]
    discord_msg = as_message(fake=message)
    neon_parts = await neon_builder.get_attachment_parts(
        message=discord_msg, sources=neon_builder.supported_sources(message=discord_msg)
    )
    hydrogen_parts = await hydrogen_builder.get_attachment_parts(
        message=discord_msg, sources=hydrogen_builder.supported_sources(message=discord_msg)
    )
    claude_parts = await claude_builder.get_attachment_parts(
        message=discord_msg, sources=claude_builder.supported_sources(message=discord_msg)
    )

    assert len(fake_files.upload_calls) == 1
    assert neon_parts == hydrogen_parts
    assert neon_parts[0]["type"] == "input_file"
    assert claude_parts[0]["type"] == "input_image"


async def test_media_persona_reply_formats_fast_model_identity_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Media persona replies substitute `{model_label}` with `fast_model.name`, never leaking it."""
    cog = _cog()
    message = FakeMessage(content="<@999> draw a cat")
    seen_instructions: list[str] = []

    async def fake_create(**kwargs: object) -> AsyncIterator[ResponseStreamEvent]:
        seen_instructions.append(str(kwargs["instructions"]))
        completed = _completed_event(input_tokens=10, output_tokens=5)
        completed.response.model = cog.toolkit.runtime_models.fast_model.name
        return _stream_events_from(events=[_text_event(delta="畫好了"), completed])

    monkeypatch.setattr(cog.openai_client.responses, "create", fake_create)

    context_task = asyncio.create_task(coro=asyncio.sleep(0, result=ReplyContext()))
    turn = AnswerTurn(
        config=cog.config,
        media_delivery=hosting_off_planner(),
        toolkit=cog.toolkit,
        surface=TurnSurface.for_message(message=as_message(fake=message)),
    )
    await turn.stream_media_persona_reply(
        reply=None,
        context_task=context_task,
        system_prompt=IMAGE_REPLY_PROMPT,
        focus_part=ResponseInputImageParam(
            image_url="data:image/png;base64,AA==", detail="auto", type="input_image"
        ),
        media_noun="image",
        media_model=cog.toolkit.runtime_models.image_model.name,
    )

    assert len(seen_instructions) == 1
    assert "{model_label}" not in seen_instructions[0]
    assert f"({cog.toolkit.runtime_models.fast_model.name})" in seen_instructions[0]
