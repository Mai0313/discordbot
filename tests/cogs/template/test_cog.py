"""Tests for the template cog."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from nextcord import Embed

from discordbot.cogs.template.cog import TemplateCogs

from tests.helpers.casting import as_bot, as_message, make_forbidden
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    import pytest


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


async def test_template_logs_a_refused_reaction_at_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A channel that refuses the trigger reaction is a warn with the ids, not an escaped 403."""
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    cog = TemplateCogs(bot=as_bot(fake=SimpleNamespace(latency=0.123)))
    message = FakeDiscordMessage(author=FakeUser(bot=False), content="debug")

    async def refuse(emoji: str) -> None:
        """Refuses the reaction the way a channel overwrite does."""
        del emoji
        raise make_forbidden(message="Missing Permissions")

    monkeypatch.setattr(target=message, name="add_reaction", value=refuse)

    await cog.on_message(message=as_message(fake=message))

    assert warns == [("Discord refused a trigger reaction", {"message_id": 1, "channel_id": 2})]
