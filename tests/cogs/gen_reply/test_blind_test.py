"""Tests for randomized `slow_model` blind A/B testing and footer/prompt masking."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from openai.types.responses import ResponseStreamEvent

from discordbot.typings.models import BLIND_TEST_MODEL_LABEL, RuntimeModelCatalog
from discordbot.utils.model_pricing import ModelPriceEntry, _LoadedTable
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.prompts import REPLY_PROMPT
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.feedback_store import ReplyTurnRow, open_session

from tests.helpers.casting import as_message
from tests.helpers.reply_fakes import (
    FakeMessage,
    _cog,
    _text_event,
    _completed_event,
    _stream_events_from,
    hosting_off_planner,
)


async def test_blind_test_samples_candidate_masks_visible_label_and_records_real_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blind test masks prompt and footer as `blind-test-model` while storing real model in DB."""
    cog = _cog()
    cog.config.slow_model_blind_test_enabled = True
    message = FakeMessage(content="<@999> 你是哪個模型?")
    seen_calls: list[dict[str, object]] = []
    finalized_logs: list[dict[str, object]] = []

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.answer.secrets.choice", lambda seq: "gemini-4-hydrogen"
    )
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
    assert f"({BLIND_TEST_MODEL_LABEL})" in instructions
    assert RuntimeModelCatalog().slow_model.name not in instructions
    assert "gemini-4-hydrogen" not in instructions

    reply_text = message.replies[0].content or ""
    assert f"-# {BLIND_TEST_MODEL_LABEL} · ⬆ 120 ⬇ 45 · $0.00069000" in reply_text
    assert "(high)" not in reply_text
    assert "gemini-4-hydrogen" not in reply_text
    assert USAGE_FOOTER_RE.sub(repl="", string=reply_text).strip() == "我是破貓"

    async with open_session() as session:
        rows = list((await session.execute(select(ReplyTurnRow))).scalars())
    assert len(rows) == 1
    assert rows[0].model == "gemini-4-hydrogen"
    assert rows[0].effort == "high"
    assert rows[0].model != BLIND_TEST_MODEL_LABEL
    assert len(finalized_logs) == 1
    assert finalized_logs[0]["model"] == "gemini-4-hydrogen"
    assert finalized_logs[0]["effort"] == "high"


async def test_blind_test_records_real_post_fallback_model_while_hiding_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even after a LiteLLM fallback, footer stays masked while `reply_turn` keeps the real model."""
    cog = _cog()
    cog.config.slow_model_blind_test_enabled = True
    message = FakeMessage(content="<@999> test fallback")

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.answer.secrets.choice", lambda seq: "gemini-4-neon"
    )

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
    assert f"-# {BLIND_TEST_MODEL_LABEL} · ⬆ 10 ⬇ 5" in reply_text
    assert "(medium)" not in reply_text
    assert "gemini-4-hydrogen" not in reply_text

    async with open_session() as session:
        rows = list((await session.execute(select(ReplyTurnRow))).scalars())
    assert len(rows) == 1
    assert rows[0].model == "gemini-4-hydrogen"
    assert rows[0].effort == "medium"


async def test_blind_test_disabled_uses_default_slow_model_and_visible_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the switch off, QA dispatches `slow_model` and displays its real name in the footer."""
    cog = _cog()
    assert cog.config.slow_model_blind_test_enabled is False
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
    assert BLIND_TEST_MODEL_LABEL not in reply_text


def test_blind_test_candidates_are_gemini_4_models() -> None:
    """Blind-test candidate pool names neon and hydrogen inside `RuntimeModelCatalog`."""
    candidates = RuntimeModelCatalog().slow_model_blind_test_candidates
    assert candidates == ("gemini-4-neon", "gemini-4-hydrogen")
