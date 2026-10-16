"""Tests for the capability reference block a reply carries."""

from typing import TYPE_CHECKING, cast
from difflib import SequenceMatcher

import pytest

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.markers import VOICE_OPEN
from discordbot.cogs.gen_reply.prompts import (
    REPLY_PROMPT,
    MUSIC_INSTRUCTION,
    VIDEO_INSTRUCTION,
    INLINE_IMAGE_INSTRUCTION,
    DEEP_RESEARCH_INSTRUCTION,
)
from discordbot.cogs.gen_reply.capabilities import CAPABILITIES_DOC, render_capabilities_block

from tests.helpers.casting import as_message
from tests.helpers.llm_input import request_input, iter_text_blocks
from tests.helpers.reply_fakes import (
    FakeAuthor,
    FakeMessage,
    _cog,
    _answer,
    _recorded,
    _text_event,
    _config_stub,
    _run_pipeline,
    _completed_event,
)

if TYPE_CHECKING:
    from openai.types.responses.response_input_param import EasyInputMessageParam

# Every switch at its default, which is on, plus a key: what the capability document describes.
_ALL_ON = LLMConfig.model_construct(gemini_api_key="test-key")


def test_capabilities_block_is_a_low_authority_assistant_note() -> None:
    """The reference rides as the bot's own note, never as a rule that could outrank the user.

    With every switch on and a key the document goes out verbatim, so the block keeps the bytes
    a prefix cache has already seen.
    """
    block = render_capabilities_block(config=_ALL_ON)
    assert block["role"] == "assistant"
    content = block["content"]
    assert isinstance(content, str)
    assert content.partition("\n\n")[2] == CAPABILITIES_DOC


def _added_words(before: str, after: str) -> list[str]:
    """The words `after` says that `before` does not, read off a word-level diff."""
    old, new = before.split(), after.split()
    opcodes = SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes()
    return [
        word
        for tag, _, _, start, end in opcodes
        if tag in ("replace", "insert")
        for word in new[start:end]
    ]


@pytest.mark.parametrize(
    argnames=("switched_off", "promises", "says_instead"),
    argvalues=[
        ({"inline_voice_enabled": False}, ("speak a line aloud",), False),
        ({"inline_music_enabled": False}, ("write and record a short song",), False),
        ({"youtube_video_enabled": False}, ("I watch the video before answering",), False),
        (
            {"douyin_video_enabled": False},
            ("Mention me with the link instead and I watch it",),
            True,
        ),
        ({"bilibili_video_enabled": False}, ("Bilibili video link and I watch it",), True),
        ({"file_api_enabled": False}, ("I watch it and answer about it",), True),
        (
            {"deep_research_enabled": False},
            ("fully cited research report", "kicks off the same thing"),
            True,
        ),
        (
            {"gemini_api_key": ""},
            (
                "write and record a short song",
                "generate a short video",
                "an image or a video (or",
                "I watch the video before answering",
                "I watch it and answer about it",
                "fully cited research report",
            ),
            True,
        ),
    ],
    ids=["voice", "music", "youtube", "douyin", "bilibili", "file-api", "research", "keyless"],
)
def test_capabilities_block_drops_each_promise_its_switch_turns_off(
    switched_off: dict[str, object], promises: tuple[str, ...], says_instead: bool
) -> None:
    """Each switch off, or the key gone, takes back exactly what it stops the bot delivering.

    A case names the promise by the document's own words, so a document edit that stops a gated
    passage matching fails here instead of reaching every such deployment. Each promise has to
    be in the full document first, or a stale one would pass by never having been there. Where
    the feature still exists in a lesser form (a link read as text, a command that only says it
    is off), the passage is reworded rather than cut, so the reference keeps describing it.
    """
    all_on = render_capabilities_block(config=_ALL_ON)["content"]
    content = render_capabilities_block(config=_ALL_ON.model_copy(update=switched_off))["content"]
    assert isinstance(all_on, str)
    assert isinstance(content, str)
    assert [promise for promise in promises if promise not in all_on] == []
    assert [promise for promise in promises if promise in content] == []
    if says_instead:
        assert _added_words(before=all_on, after=content)


def _media_sentence(content: str) -> str:
    """The sentence listing the media a reply can produce, unwrapped onto one line."""
    after = content.split("when the moment calls for it: ", maxsplit=1)[1]
    return " ".join(after.split(" Attach", maxsplit=1)[0].split())


@pytest.mark.parametrize(
    argnames=("voice", "music", "key", "listed"),
    argvalues=[
        (True, True, False, "{voice} or {images}."),
        (True, False, True, "{voice}, {images}, or {video}."),
        (False, True, True, "{images}, {music}, or {video}."),
    ],
    ids=["keyless", "music-off", "voice-off"],
)
def test_capabilities_block_lists_the_media_still_on_as_one_sentence(
    voice: bool, music: bool, key: bool, listed: str
) -> None:
    """The media list is rebuilt from what is left, so its "or" never ends up stranded.

    The phrases come from the full document's own list, so only which media remain and how
    they are joined is pinned here.
    """
    all_on = render_capabilities_block(config=_ALL_ON)["content"]
    assert isinstance(all_on, str)
    full = _media_sentence(content=all_on).removesuffix(".").split(", ")
    speak, images, song, video = (phrase.removeprefix("or ") for phrase in full)
    config = _ALL_ON.model_copy(
        update={
            "gemini_api_key": "test-key" if key else "",
            "inline_voice_enabled": voice,
            "inline_music_enabled": music,
        }
    )
    content = render_capabilities_block(config=config)["content"]
    assert isinstance(content, str)
    expected = listed.format(voice=speak, images=images, music=song, video=video)
    assert _media_sentence(content=content) == expected


@pytest.mark.parametrize("switched_on", [True, False], ids=["all-on", "all-off"])
@pytest.mark.usefixtures("no_memory_review")
async def test_the_answer_is_told_only_what_the_deployment_switched_on(switched_on: bool) -> None:
    """The QA turn's reference and instructions follow the switches.

    All on with a key is the default deployment, whose prompt has to keep its bytes for the
    prefix cache: the reference is the document verbatim and the instructions are the reply
    prompt with every marker instruction appended. Keyless with every marker off, the voice
    bullet is cut and the reference is the gated render.
    """
    cog = _cog()
    cog.config = (
        LLMConfig.model_construct(gemini_api_key="test-key") if switched_on else _config_stub()
    )
    message = FakeMessage(content="<@999> can you make me a song?", author=FakeAuthor(user_id=1))
    await _answer(cog=cog, message=as_message(fake=message)).stream_answer(
        system_prompt=REPLY_PROMPT, context=ReplyContext(), allow_research=True
    )

    (instructions,) = _recorded(cog).responses.create_instructions
    (answer_input,) = _recorded(cog).responses.create_inputs
    reference = cast("EasyInputMessageParam", answer_input[0])["content"]
    assert isinstance(reference, str)
    if switched_on:
        appended = (
            INLINE_IMAGE_INSTRUCTION,
            MUSIC_INSTRUCTION,
            VIDEO_INSTRUCTION,
            DEEP_RESEARCH_INSTRUCTION,
        )
        assert instructions.startswith("\n".join((REPLY_PROMPT, *appended)))
        assert reference.endswith(CAPABILITIES_DOC)
    else:
        assert VOICE_OPEN not in instructions
        assert reference == render_capabilities_block(config=cog.config)["content"]
        assert "write and record a short song" not in reference


@pytest.mark.usefixtures("no_memory_review")
async def test_handle_message_reply_leads_with_the_capability_reference() -> None:
    """The feature reference leads the answer input.

    It is the one block that is byte-identical on every reply, so it rides in front of history
    where it costs the least against a prefix cache.
    """
    cog = _cog()

    message = FakeMessage(content="<@999> 你會做什麼", author=FakeAuthor(user_id=1))
    _recorded(cog).responses.stream_queue = [
        [_text_event(delta="好"), _completed_event(input_tokens=1, output_tokens=1)]
    ]

    await _run_pipeline(cog=cog, message=message)

    header = str(render_capabilities_block(config=cog.config)["content"]).split("\n", 1)[0]
    blocks = list(iter_text_blocks(request=request_input(responses=_recorded(cog).responses)))
    carried = [index for index, (_role, text) in enumerate(blocks) if text.startswith(header)]
    assert carried == [0]
    assert blocks[0][0] == "assistant"
    assert "/memory clear" in blocks[0][1]
