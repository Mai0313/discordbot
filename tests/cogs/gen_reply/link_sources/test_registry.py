"""Tests for the link-source registry."""

from discordbot.typings.llm import LLMConfig

from tests.helpers.reply_fakes import _LINK_SOURCES, _link_source


def test_the_file_api_kill_switch_stops_link_media_before_it_is_fetched() -> None:
    """The switch reaches every link source that fetches first and uploads after.

    Gating the upload alone would still spend a full Douyin / Bilibili download on media that
    can no longer reach the model, and Douyin's is the WAF-sensitive path an incident most
    wants left alone. Facebook, Instagram and Threads fetch and downscale their images before the
    upload those images could no longer feed, which is the same cost through a different door,
    and Threads downloads its clips whole besides. Read off the live registry so the wiring is
    what is pinned, and asserted over every source rather than the two it was written for.
    """
    on = LLMConfig.model_construct(gemini_api_key="test-key", file_api_enabled=True)
    assert all(_link_source(name=name).media_ingest_allowed(on) for name in _LINK_SOURCES)

    off = on.model_copy(update={"file_api_enabled": False})
    assert not any(_link_source(name=name).media_ingest_allowed(off) for name in _LINK_SOURCES)
