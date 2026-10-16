"""Tests for the link-source registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from discordbot.typings.llm import LLMConfig

from tests.helpers.reply_fakes import _LINK_SOURCES, _link_source

if TYPE_CHECKING:
    import pytest


def test_the_file_api_kill_switch_stops_link_media_before_it_is_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The switch reaches every link source that fetches first and uploads after.

    Gating the upload alone would still spend a full Douyin / Bilibili download on media that
    can no longer reach the model, and Douyin's is the WAF-sensitive path an incident most
    wants left alone. Facebook, Instagram and Threads fetch and downscale their images before the
    upload those images could no longer feed, which is the same cost through a different door,
    and Threads downloads its clips whole besides. Read off the live registry so the wiring is
    what is pinned, and asserted over every source rather than the two it was written for.
    """
    monkeypatch.setenv(name="GEMINI_API_KEY", value="test-key")
    monkeypatch.setenv(name="DOUYIN_VIDEO_ENABLED", value="true")
    monkeypatch.setenv(name="BILIBILI_VIDEO_ENABLED", value="true")

    monkeypatch.setenv(name="FILE_API_ENABLED", value="true")
    on = LLMConfig()
    assert all(_link_source(name=name).media_ingest_allowed(on) for name in _LINK_SOURCES)

    monkeypatch.setenv(name="FILE_API_ENABLED", value="false")
    off = LLMConfig()
    assert not any(_link_source(name=name).media_ingest_allowed(off) for name in _LINK_SOURCES)
