"""Tests for the reply toolkit's clients."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx2
from google.genai import _api_client as genai_api_client

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs.gen_reply.attachment.gemini_file_api import GeminiFileUploader

from tests.helpers.casting import as_bot
from tests.helpers.reply_fakes import FakeClient
from tests.helpers.discord_mocks import stub_bot

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
    bot = stub_bot()
    keyed = ReplyToolkit(
        bot=bot,
        openai_client=FakeClient(),
        config=LLMConfig.model_construct(gemini_api_key="test-key"),
    )
    keyed_handler = keyed.input_builder.attachment_handler
    assert isinstance(keyed_handler, GeminiFileUploader)
    assert keyed_handler.gemini_client() is keyed.gemini_client

    # Turning the switch off on the toolkit's own config selects inline without touching env.
    switched_off = ReplyToolkit(
        bot=bot,
        openai_client=FakeClient(),
        config=LLMConfig.model_construct(gemini_api_key="test-key", file_api_enabled=False),
    )
    assert isinstance(switched_off.input_builder.attachment_handler, InlineRenderer)


def test_the_toolkit_memory_writer_knows_the_bots_own_id() -> None:
    """The note review leaves the bot out of the sharing gate's roster only if it knows its id."""
    bot = stub_bot()
    toolkit = ReplyToolkit(bot=bot, openai_client=FakeClient(), config=LLMConfig.model_construct())
    assert toolkit.memory_writer.bot_user_id == 999


def test_memory_writer_does_not_cache_none_bot_user_id_before_login() -> None:
    """Accessing `memory_writer` before login never pins `bot_user_id=None` permanently."""
    raw_bot = SimpleNamespace(user=None)
    toolkit = ReplyToolkit(
        bot=as_bot(fake=raw_bot), openai_client=FakeClient(), config=LLMConfig.model_construct()
    )

    pre_login = toolkit.memory_writer
    assert pre_login.bot_user_id is None

    raw_bot.user = SimpleNamespace(id=999, name="bot")
    post_login = toolkit.memory_writer
    assert post_login.bot_user_id == 999
    assert toolkit.memory_writer is post_login


async def test_gemini_client_pins_v1beta_api_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct Gemini calls use `v1beta` so omni/Lyria resolve and Files URIs match LiteLLM."""
    paths: list[str] = []

    async def fake_send(
        _self: httpx2.AsyncClient, request: httpx2.Request, **_kwargs: object
    ) -> httpx2.Response:
        paths.append(request.url.path)
        return httpx2.Response(
            status_code=200,
            request=request,
            json={
                "name": "files/abc",
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
                "state": "ACTIVE",
            },
        )

    monkeypatch.setattr(target=genai_api_client, name="has_aiohttp", value=False)
    monkeypatch.setattr(target=httpx2.AsyncClient, name="send", value=fake_send)
    bot = stub_bot()
    toolkit = ReplyToolkit(
        bot=bot,
        openai_client=FakeClient(),
        config=LLMConfig.model_construct(gemini_api_key="test-key"),
    )
    await toolkit.gemini_client.aio.files.get(name="files/abc")
    assert paths == ["/v1beta/files/abc"]
