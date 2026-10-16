"""Tests for the route and answer prompts."""

from __future__ import annotations

import pytest

from discordbot.cogs.gen_reply.recall import RecallCandidate
from discordbot.cogs.gen_reply.prompts import ROUTE_RECALL_SECTION, route_prompt
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES

from tests.helpers.casting import as_message
from tests.helpers.reply_fakes import (
    _ROUTE_CURRENT,
    _ROUTE_REFERENCE,
    FakeAuthor,
    FakeMessage,
    _cog,
    _recorded,
    _classifier,
    _run_pipeline,
)


def test_the_route_prompt_says_qa_draws_inline_only_while_it_can() -> None:
    """With inline images on, the route is told QA can draw inline; with it off, not."""
    on = route_prompt(inline_image_enabled=True)
    off = route_prompt(inline_image_enabled=False)

    assert "inline" in on
    assert "only possible on this route" not in on
    assert "inline" not in off


def test_the_route_prompt_names_the_link_sources_as_measured() -> None:
    """Every registered link source is named in the route prompt."""
    prompt = route_prompt(inline_image_enabled=True)

    for source in LINK_CONTEXT_SOURCES:
        assert f"`{source.name}`" in prompt


@pytest.mark.parametrize("inline_image_enabled", [True, False])
@pytest.mark.usefixtures("no_memory_review")
async def test_the_route_is_told_qa_draws_inline_only_while_the_answer_can(
    inline_image_enabled: bool,
) -> None:
    """A turn's route call reads the prompt for the deployment's inline-image switch."""
    cog = _cog()
    cog.config.inline_image_enabled = inline_image_enabled

    await _run_pipeline(
        cog=cog, message=FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    )

    assert _recorded(cog).responses.parse_instructions == [
        route_prompt(inline_image_enabled=inline_image_enabled)
    ]


@pytest.mark.parametrize("inline_image_enabled", [True, False])
async def test_a_recall_turn_appends_its_section_to_either_route_prompt(
    inline_image_enabled: bool,
) -> None:
    """The recall section follows the route prompt whether or not QA can draw."""
    cog = _cog()
    cog.config.inline_image_enabled = inline_image_enabled

    await _classifier(cog=cog, message=as_message(fake=FakeMessage())).classify(
        reference_messages=_ROUTE_REFERENCE,
        current_message=_ROUTE_CURRENT,
        recall_candidates={42: RecallCandidate(prompt_label="Boss(社群暱稱:李董)")},
        server_memory_block=None,
    )

    assert _recorded(cog).responses.parse_instructions == [
        route_prompt(inline_image_enabled=inline_image_enabled) + ROUTE_RECALL_SECTION
    ]
