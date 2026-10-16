"""Tests for the route and answer prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from discordbot.cogs.gen_reply.recall import RecallCandidate
from discordbot.cogs.gen_reply.prompts import REPLY_PROMPT, ROUTE_RECALL_SECTION, route_prompt

from tests.helpers.casting import as_message
from tests.helpers.llm_input import request_input, iter_text_blocks
from tests.helpers.reply_fakes import (
    _ROUTE_CURRENT,
    _ROUTE_REFERENCE,
    FakeAuthor,
    FakeChannel,
    FakeMessage,
    _cog,
    _recorded,
    _classifier,
    _text_event,
    _run_pipeline,
    _completed_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.mark.usefixtures("no_memory_review")
async def test_the_history_separator_names_the_block_without_inviting_an_answer_from_it() -> None:
    """The history separator is a label; where the subject may come from is a developer rule.

    A separator inviting an answer from the history would compete with the Reference Message's
    own claim to be the primary context. Behaviour rules belong in `instructions`, which
    outranks anything in `input`, so the separator carries only the naming. This render also
    feeds the media persona reply and the memory review's transcript, neither of which is
    answering a question, which is the second reason the rule cannot live on the block itself.
    """
    cog = _cog()

    older = FakeMessage(content="舊話題", author=FakeAuthor(user_id=2))

    async def fake_history(limit: int, before: FakeMessage) -> AsyncIterator[FakeMessage]:
        """Yields one older message so the history block is rendered at all."""
        yield older

    message = FakeMessage(content="<@999> 真假", author=FakeAuthor(user_id=1))
    message.channel = FakeChannel(history=fake_history)
    _recorded(cog).responses.stream_queue = [
        [_text_event(delta="好"), _completed_event(input_tokens=1, output_tokens=1)]
    ]

    await _run_pipeline(cog=cog, message=message)

    answer = request_input(responses=_recorded(cog).responses)
    history_header = next(
        text
        for _role, text in iter_text_blocks(request=answer)
        if text.startswith("==== Chat History")
    )
    assert history_header == "==== Chat History: earlier messages in this channel. ===="


def test_the_subject_rule_rides_the_developer_prompt_with_its_recap_exception() -> None:
    """The rule the history separator no longer carries lives in `REPLY_PROMPT`.

    `REPLY_PROMPT` reaches the answer through `instructions`, which has developer authority
    and outranks everything in `input`, so this is where a behaviour rule belongs. The recap
    carve-out is pinned with it: without that sentence the rule forbids answering the one
    question whose subject genuinely is the history.
    """
    assert "take the subject of your answer only from the Current Message" in REPLY_PROMPT
    assert "a question about the channel's own conversation" in REPLY_PROMPT
    # The invitation this replaced must not come back on the block itself.
    assert "might be helpful for answering" not in REPLY_PROMPT


# The two route-prompt claims that QA draws inline, as they read before #845.
_ROUTE_INLINE_IMAGE_NOTE = (
    "The bot has two ways to show a generated image. The QA path can already attach its own "
    "generated illustration inline whenever one would help its written answer, so an image "
    "alongside a reply is NOT by itself a reason to leave QA. Route to IMAGE only when a "
    "produced image is the whole point of the request, not a helpful add-on to an answer."
)
_ROUTE_QA_LINE = (
    "- QA: everything else — normal questions; image analysis; captioning; requests to "
    "summarize, recap, explain, or make a 懶人包 for ANYTHING, including a URL, webpage, "
    "article, referenced message, attachment, pasted content, and the channel's own recent "
    "conversation; discussions about art that do NOT ask the bot to actually generate or edit "
    "an image; and any message that is primarily a question, explanation, or conversation even "
    "when showing a picture alongside the answer would be nice (QA draws that picture inline "
    "itself). QA is also the default whenever no other category clearly applies."
)


def test_the_route_prompt_says_qa_draws_inline_only_while_it_can() -> None:
    """With inline images on, the route reads its old text less only the edit-only claim.

    The inline marker edits too, so "editing is only possible on this route" is false whenever
    QA can draw (#845). With inline images off the answer is never offered the marker, so both
    claims that QA draws go too, and nothing else changes.
    """
    on = route_prompt(inline_image_enabled=True)
    off = route_prompt(inline_image_enabled=False)

    assert f"rules below.\n\n{_ROUTE_INLINE_IMAGE_NOTE}\n\nClassification rules:\n" in on
    assert f"\n{_ROUTE_QA_LINE}\n" in on
    assert "only possible on this route" not in on
    assert "inline" not in off
    assert (
        on.replace(f"{_ROUTE_INLINE_IMAGE_NOTE}\n\n", "").replace(
            " (QA draws that picture inline itself)", ""
        )
        == off
    )


# The route prompt's link-source lines, byte for byte. They are part of the measured route
# prompt, so a new source, or one changing where its link may sit, fails here until these lines
# are rewritten on purpose.
_ROUTE_LINK_SOURCE_LINES = (
    "Also fill in the `link_context_sources` field for registered linked-post sources "
    "(`threads`, `facebook`, `instagram`, `twitter`, `douyin`, and `bilibili`):",
    "- Include a source only when a matching link is present AND the user wants the bot to read "
    "that post or video — for example summarizing, explaining, analyzing, comparing, reacting "
    "to, or answering a question about its actual content. Threads, Facebook and Instagram "
    "links may be in the latest message OR in the message it is replying to; Twitter, Douyin "
    "and Bilibili links must be in the latest message.",
)


def test_the_route_prompt_names_the_link_sources_as_measured() -> None:
    """Each registered link source, and where its link may be, reads exactly as measured."""
    lines = route_prompt(inline_image_enabled=True).splitlines()

    assert [line for line in _ROUTE_LINK_SOURCE_LINES if line not in lines] == []


# The route prompt's VIDEO rule, byte for byte. Unmeasured wording: how often it moves a clip edit
# off QA would take paid route calls.
_ROUTE_VIDEO_LINE = (
    "- VIDEO: two cases: (1) the user explicitly wants the bot to create, generate, or make a "
    "video or animation; (2) the user attached a video file, or replied to a message with one "
    "attached, and explicitly wants that clip modified, edited, altered, transformed, or "
    "restyled, with the edited clip as what they want back."
)


def test_the_route_prompt_sends_a_clip_edit_to_video() -> None:
    """The VIDEO rule names editing an attached or replied-to clip, as the IMAGE rule names its edit.

    Only the VIDEO route reads a source clip, and only an uploaded one in the message or the one it
    replies to; the QA answer's inline video marker never does, so an edit request left on QA comes
    back as an unrelated new clip.
    """
    lines = route_prompt(inline_image_enabled=True).splitlines()

    assert _ROUTE_VIDEO_LINE in lines


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
