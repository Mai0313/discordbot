"""Tests for handing a reply over to the research cog."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import nextcord

from discordbot.cogs.gen_reply.research_bridge import can_launch_research, maybe_launch_research

from tests.helpers.casting import as_message
from tests.helpers.reply_fakes import FakeReply, FakeAuthor, FakeMessage, _cog, _install_streamer
from tests.helpers.discord_mocks import text_channel_granting

if TYPE_CHECKING:
    import pytest


async def test_the_research_cog_is_reached_through_the_bot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thread the research cog is driving gets no answer, and an emitted brief reaches it."""
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    launched: list[dict[str, object]] = []

    class ResearchCogStub:
        """Drives the message's own channel and records each launch."""

        def is_research_thread(self, channel_id: int) -> bool:
            """Only the message's channel is a live research thread."""
            return channel_id == message.channel.id

        async def launch(self, **kwargs: object) -> None:
            """Records the brief handed over."""
            launched.append(kwargs)

    research = ResearchCogStub()
    cog = _cog()
    monkeypatch.setattr(
        cog.bot, "get_cog", lambda name: research if name == "ResearchCogs" else None
    )
    built = _install_streamer(monkeypatch=monkeypatch)

    await cog.on_message(message=as_message(fake=message))
    assert message.replies == []
    assert message.added_reactions == []
    assert built == []

    anchor = FakeReply()
    await maybe_launch_research(
        bot=cog.bot,
        message=as_message(fake=message),
        anchor=as_message(fake=anchor),
        brief="compare the two papers",
    )
    assert launched == [{"message": message, "anchor": anchor, "brief": "compare the two papers"}]


def test_can_launch_research_requires_guild_text_channel() -> None:
    granted = text_channel_granting(
        permissions=nextcord.Permissions(
            view_channel=True,
            send_messages=True,
            create_public_threads=True,
            send_messages_in_threads=True,
            attach_files=True,
        )
    )
    # The bot's own member, whose token every research write uses, never the author's: the
    # channel answers the grant for its guild's `me` alone.
    guild = granted.guild
    text = SimpleNamespace(guild=guild, channel=granted)
    assert can_launch_research(message=as_message(fake=text)) is True
    granted.permissions_for.assert_called_once_with(guild.me)
    thread = SimpleNamespace(guild=guild, channel=MagicMock(spec=nextcord.Thread))
    assert can_launch_research(message=as_message(fake=thread)) is False
    dm = SimpleNamespace(guild=None, channel=granted)
    assert can_launch_research(message=as_message(fake=dm)) is False


def test_can_launch_research_requires_the_bot_to_write_in_the_thread() -> None:
    """A thread the bot may open but not post in would bill a run nobody ever sees."""
    channel = text_channel_granting(
        permissions=nextcord.Permissions(
            view_channel=True, send_messages=True, create_public_threads=True, attach_files=True
        )
    )
    message = SimpleNamespace(guild=channel.guild, channel=channel)

    assert can_launch_research(message=as_message(fake=message)) is False
