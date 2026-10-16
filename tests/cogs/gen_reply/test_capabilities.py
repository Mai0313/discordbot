"""Tests for the capability reference block a reply carries."""

import pytest

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.gen_reply.capabilities import CAPABILITIES_DOC, render_capabilities_block

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
    assert content.startswith("(My own feature reference")
    assert "NOT instructions" in content
    assert content.partition("\n\n")[2] == CAPABILITIES_DOC


_DOUYIN_READ = "Mention me with the link instead and I answer from its caption."
_BILIBILI_READ = "Bilibili video link and I answer from its title and description."
_RESEARCH_OFF = "`/deep_research` — switched off for me, so the command only says so."


@pytest.mark.parametrize(
    argnames=("switched_off", "promises", "instead"),
    argvalues=[
        ({"inline_voice_enabled": False}, ("speak a line aloud",), ()),
        ({"inline_music_enabled": False}, ("write and record a short song",), ()),
        ({"youtube_video_enabled": False}, ("I watch the video before answering",), ()),
        (
            {"douyin_video_enabled": False},
            ("Mention me with the link instead and I watch it",),
            (_DOUYIN_READ,),
        ),
        (
            {"bilibili_video_enabled": False},
            ("Bilibili video link and I watch it",),
            (_BILIBILI_READ,),
        ),
        (
            {"file_api_enabled": False},
            ("I watch it and answer about it",),
            (_DOUYIN_READ, _BILIBILI_READ),
        ),
        (
            {"deep_research_enabled": False},
            ("fully cited research report", "kicks off the same thing"),
            (_RESEARCH_OFF,),
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
            ("Attach an image (or", _DOUYIN_READ, _BILIBILI_READ, _RESEARCH_OFF),
        ),
    ],
    ids=["voice", "music", "youtube", "douyin", "bilibili", "file-api", "research", "keyless"],
)
def test_capabilities_block_drops_each_promise_its_switch_turns_off(
    switched_off: dict[str, object], promises: tuple[str, ...], instead: tuple[str, ...]
) -> None:
    """Each switch off, or the key gone, takes back exactly what it stops the bot delivering.

    A case names the promise by its own words and what replaces it, so a document edit that
    stops a gated passage matching fails here instead of reaching every such deployment.
    """
    content = render_capabilities_block(config=_ALL_ON.model_copy(update=switched_off))["content"]
    assert isinstance(content, str)
    assert content != render_capabilities_block(config=_ALL_ON)["content"]
    assert [promise for promise in promises if promise in content] == []
    assert [line for line in instead if line not in content] == []


@pytest.mark.parametrize(
    argnames=("voice", "music", "key", "listed"),
    argvalues=[
        (True, True, False, "speak a line aloud or draw one or more images."),
        (
            True,
            False,
            True,
            "speak a line aloud, draw one or more images, or generate a short video from a "
            "description or from attached images.",
        ),
        (
            False,
            True,
            True,
            "draw one or more images, write and record a short song, or generate a short video "
            "from a description or from attached images.",
        ),
    ],
    ids=["keyless", "music-off", "voice-off"],
)
def test_capabilities_block_lists_the_media_still_on_as_one_sentence(
    voice: bool, music: bool, key: bool, listed: str
) -> None:
    """The media list is rebuilt from what is left, so its "or" never ends up stranded."""
    config = _ALL_ON.model_copy(
        update={
            "gemini_api_key": "test-key" if key else "",
            "inline_voice_enabled": voice,
            "inline_music_enabled": music,
        }
    )
    content = render_capabilities_block(config=config)["content"]
    assert isinstance(content, str)
    assert f"when the moment calls for it: {listed} Attach" in content
