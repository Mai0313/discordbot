"""Tests for the reply toolkit's clients."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.attachment.gemini_file_api import GeminiFileUploader

from tests.helpers.casting import as_bot
from tests.helpers.reply_fakes import FakeClient

if TYPE_CHECKING:
    import pytest


def test_gemini_uploader_uploads_through_the_toolkit_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attachments upload with the toolkit's own direct client.

    A file is readable only by the key that uploaded it, so the uploader holds no client of its
    own: the one the answer's direct paths use is the one it uploads with.
    """
    monkeypatch.setattr(
        RuntimeModelCatalog,
        "slow_model",
        property(lambda _self: ModelSettings(name="gemini-test", effort="high")),
    )
    bot = as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999, name="bot")))
    keyed = ReplyToolkit(bot=bot, openai_client=FakeClient(), gemini_api_key="test-key")
    keyed_handler = keyed.input_builder.attachment_handler
    assert isinstance(keyed_handler, GeminiFileUploader)
    assert keyed_handler.gemini_client() is keyed.gemini_client


def test_the_toolkit_memory_writer_knows_the_bots_own_id() -> None:
    """The note review leaves the bot out of the sharing gate's roster only if it knows its id."""
    bot = as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999, name="bot")))
    toolkit = ReplyToolkit(bot=bot, openai_client=FakeClient(), gemini_api_key="")
    assert toolkit.memory_writer.bot_user_id == 999
