"""Tests for choosing the attachment handler for the answer model."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from discordbot.typings.models import ModelSettings
from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs.gen_reply.attachment.select import build_attachment_handler
from discordbot.cogs.gen_reply.attachment.gemini_file_api import GeminiFileUploader
from discordbot.cogs.gen_reply.attachment.anthropic_file_api import AnthropicFileUploader

from tests.helpers.casting import as_client
from tests.helpers.gen_reply import FakeGeminiClient

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


def _stub_anthropic() -> AsyncAnthropic:
    return cast("AsyncAnthropic", SimpleNamespace())


def test_gpt_attachment_handler_path_stays_disabled() -> None:
    """GPT models still use inline attachments until the OpenAI uploader branch is enabled."""
    gemini = as_client(fake=FakeGeminiClient())
    anthropic = _stub_anthropic()
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="gpt-5.1"),
            gemini_client=lambda: gemini,
            anthropic_client=lambda: anthropic,
        ),
        InlineRenderer,
    )


def test_grok_attachment_handler_path_stays_disabled() -> None:
    """Grok models still use inline attachments until the xAI uploader branch is enabled."""
    gemini = as_client(fake=FakeGeminiClient())
    anthropic = _stub_anthropic()
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="grok-4.5"),
            gemini_client=lambda: gemini,
            anthropic_client=lambda: anthropic,
        ),
        InlineRenderer,
    )


def test_gemini_attachments_upload_while_the_file_api_is_enabled() -> None:
    """The Gemini branch uploads to the Files API while the switch is on and a key is set."""
    gemini = as_client(fake=FakeGeminiClient())
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="gemini-3.8-flash"),
            gemini_client=lambda: gemini,
            anthropic_client=lambda: None,
        ),
        GeminiFileUploader,
    )


def test_claude_attachments_upload_when_keyed_and_inline_when_keyless_or_disabled() -> None:
    """Claude uploads via Anthropic Files API when keyed and enabled, else falls back to inline."""
    anthropic = _stub_anthropic()
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="claude-sonnet-4-6"),
            gemini_client=lambda: None,
            anthropic_client=lambda: anthropic,
        ),
        AnthropicFileUploader,
    )
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="claude-sonnet-4-6"),
            gemini_client=lambda: None,
            anthropic_client=lambda: None,
        ),
        InlineRenderer,
    )
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="claude-sonnet-4-6"),
            gemini_client=lambda: None,
            anthropic_client=lambda: anthropic,
            file_api_enabled=False,
        ),
        InlineRenderer,
    )


def test_the_file_api_kill_switch_inlines_gemini_attachments() -> None:
    """With the switch off, even a keyed Gemini answer model gets inlined attachments."""
    gemini = as_client(fake=FakeGeminiClient())
    assert isinstance(
        build_attachment_handler(
            model=ModelSettings(name="gemini-3.8-flash"),
            gemini_client=lambda: gemini,
            anthropic_client=lambda: None,
            file_api_enabled=False,
        ),
        InlineRenderer,
    )
