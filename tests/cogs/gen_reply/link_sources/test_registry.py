"""Tests for the link-source registry."""

from discordbot.typings.llm import LLMConfig

from tests.helpers.reply_fakes import _LINK_SOURCES, _link_source

_VIDEO_SOURCES = frozenset({"threads", "douyin", "bilibili"})


def test_the_file_api_kill_switch_stops_link_video_before_it_is_fetched() -> None:
    """The switch gates Gemini Files API video ingestion on video-capable sources only.

    Image-only sources render through the turn's `AttachmentRenderer` (which inlines when the
    Files API is disabled), while video-capable sources cannot inline clips and must skip
    downloading them when `file_api_enabled=False` or no Gemini key is configured.
    """
    on = LLMConfig.model_construct(gemini_api_key="test-key", file_api_enabled=True)
    for name in _LINK_SOURCES:
        assert _link_source(name=name).video_ingest_allowed(on) is (name in _VIDEO_SOURCES)

    off = on.model_copy(update={"file_api_enabled": False})
    assert not any(_link_source(name=name).video_ingest_allowed(off) for name in _LINK_SOURCES)

    keyless = on.model_copy(update={"gemini_api_key": ""})
    assert not any(_link_source(name=name).video_ingest_allowed(keyless) for name in _LINK_SOURCES)
