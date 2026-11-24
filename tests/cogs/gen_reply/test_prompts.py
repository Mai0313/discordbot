"""Tests for the route and answer prompts."""

from __future__ import annotations

import pytest

from discordbot.cogs.gen_reply.recall import RecallCandidate
from discordbot.cogs.gen_reply.prompts import (
    IMAGE_PROMPT,
    VIDEO_PROMPT,
    ROUTE_RECALL_SECTION,
    route_prompt,
)
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


def test_the_route_prompt_says_qa_draws_inline() -> None:
    """The route prompt tells the classifier that QA can draw illustrations inline."""
    prompt = route_prompt()

    assert "inline" in prompt
    assert "only possible on this route" not in prompt


def test_the_route_prompt_names_the_link_sources_as_measured() -> None:
    """Every registered link source is named in the route prompt."""
    prompt = route_prompt()

    for source in LINK_CONTEXT_SOURCES:
        assert f"`{source.name}`" in prompt


@pytest.mark.usefixtures("no_memory_review")
async def test_the_route_call_uses_route_prompt() -> None:
    """A turn's route call sends `route_prompt()` as its instructions."""
    cog = _cog()

    await _run_pipeline(
        cog=cog, message=FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    )

    assert _recorded(cog).responses.parse_instructions == [route_prompt()]


async def test_a_recall_turn_appends_its_section_to_route_prompt() -> None:
    """The recall section follows the route prompt when recall candidates exist."""
    cog = _cog()

    await _classifier(cog=cog, message=as_message(fake=FakeMessage())).classify(
        reference_messages=_ROUTE_REFERENCE,
        current_message=_ROUTE_CURRENT,
        recall_candidates={42: RecallCandidate(prompt_label="Boss(社群暱稱:李董)")},
        server_memory_block=None,
    )

    assert _recorded(cog).responses.parse_instructions == [route_prompt() + ROUTE_RECALL_SECTION]


def test_route_prompt_marks_bare_youtube_links_as_watch_video() -> None:
    """A bare YouTube link (or short nudge) addressed to the bot asks the bot to watch it."""
    prompt = route_prompt()

    assert "contains ONLY a YouTube link" in prompt
    assert "set `watch_video` to `true`" in prompt


@pytest.mark.parametrize("prompt_text", [IMAGE_PROMPT, VIDEO_PROMPT], ids=["image", "video"])
def test_media_refine_prompts_scope_recent_conversation_to_references(prompt_text: str) -> None:
    """Both media refine prompts restrict Recent conversation to resolving references only."""
    assert "Recent conversation" in prompt_text
    assert "resolve references, pronouns, or follow-up edits" in prompt_text


def test_the_route_prompt_defines_all_three_effort_grades() -> None:
    """The route prompt names only the three grades (`low`, `medium`, `high`)."""
    prompt = route_prompt()
    for grade in ("`low`", "`medium`", "`high`"):
        assert grade in prompt
    assert "three grades" in prompt
    assert "xhigh" not in prompt
