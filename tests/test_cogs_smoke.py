"""Smoke tests for every cog's setup hook and the template cog."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from importlib import import_module

from nextcord import Embed

from discordbot import cli
from discordbot.cogs.template.cog import TemplateCogs

from tests.helpers.casting import as_bot, as_message, as_discord_bot
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage

if TYPE_CHECKING:
    import pytest
    from nextcord.ext import commands


async def test_template_on_message_and_ping() -> None:
    """Verifies template debug reaction and ping command response."""
    cog = TemplateCogs(bot=as_bot(fake=SimpleNamespace(latency=0.123)))
    message = FakeDiscordMessage(author=FakeUser(bot=False), content="debug")
    await cog.on_message(message=as_message(fake=message))
    assert message.reactions == ["🤬"]

    bot_message = FakeDiscordMessage(author=FakeUser(bot=True), content="debug")
    await cog.on_message(message=as_message(fake=bot_message))
    assert bot_message.reactions == []

    interaction = FakeInteraction(user=FakeUser(display_name="Alice"))
    await TemplateCogs.ping.callback(cog, interaction)
    embed = interaction.followup.sent[0]["embed"]
    assert isinstance(embed, Embed)
    assert embed.title == ":ping_pong: Pong!"


def test_setup_functions_register_cogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """EVERY cog module the loader picks up has a sync `setup` that adds its own cog.

    The modules come from `_load_cogs_sync` itself rather than a hand-written list, so a new
    cog cannot be left out and the real `cogs/` tree has to pass the loader's own checks:
    `setup` is the one function in a cog module the loader calls by name, and an
    `async def setup` never reaches `add_cog` here, while at boot it aborts `DiscordBot()`
    with `ExtensionFailed`.
    """
    added: list[tuple[commands.Cog, bool | None]] = []
    loaded: list[str] = []

    def record_cog(cog: commands.Cog, override: bool | None = None) -> None:
        """Records the cog instance and override flag passed to add_cog."""
        added.append((cog, override))

    def record_load_extensions(modules: list[str], stop_at_error: bool) -> None:
        """Records the cog modules the loader would hand to nextcord."""
        del stop_at_error
        loaded.extend(modules)

    bot = SimpleNamespace(add_cog=record_cog)
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    cli.DiscordBot._load_cogs_sync(
        as_discord_bot(fake=SimpleNamespace(load_extensions=record_load_extensions))
    )

    assert loaded  # a scan that found nothing would pass every assertion below
    for name in loaded:
        module = import_module(name=name)
        module.setup(bot=as_bot(fake=bot))
        cog, override = added[-1]
        # `override=True` is what lets a reload replace the cog instead of colliding, and
        # the module check is what stops a re-export standing in for a missing cog.
        assert override is True, name
        assert type(cog).__module__ == module.__name__, name
    assert len(added) == len(loaded)
