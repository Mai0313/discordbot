"""Tests for choosing the attachment handler for the answer model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from discordbot.typings.models import ModelSettings
from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs.gen_reply.attachment.select import build_attachment_handler
from discordbot.cogs.gen_reply.attachment.gemini_file_api import GeminiFileUploader

from tests.helpers.casting import as_client
from tests.helpers.gen_reply import FakeGeminiClient

if TYPE_CHECKING:
    import pytest


def test_gpt_attachment_handler_path_stays_disabled() -> None:
    """GPT models still use inline attachments until the OpenAI uploader branch is enabled."""
    client = as_client(fake=FakeGeminiClient())
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="gpt-5.1"), gemini_client=lambda: client
        ),
        InlineRenderer,
    )


def test_grok_attachment_handler_path_stays_disabled() -> None:
    """Grok models still use inline attachments until the xAI uploader branch is enabled."""
    client = as_client(fake=FakeGeminiClient())
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="grok-4.5"), gemini_client=lambda: client
        ),
        InlineRenderer,
    )


def test_gemini_attachments_upload_while_the_file_api_is_enabled() -> None:
    """The Gemini branch uploads to the Files API while the switch is on and a key is set."""
    client = as_client(fake=FakeGeminiClient())
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="gemini-3.8-flash"), gemini_client=lambda: client
        ),
        GeminiFileUploader,
    )


def test_the_file_api_kill_switch_inlines_gemini_attachments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the switch off, even a keyed Gemini answer model gets inlined attachments."""
    monkeypatch.setenv(name="FILE_API_ENABLED", value="false")
    client = as_client(fake=FakeGeminiClient())
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="gemini-3.8-flash"), gemini_client=lambda: client
        ),
        InlineRenderer,
    )
