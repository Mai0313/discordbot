"""Guards the application-command registry against being left empty after a gateway READY.

`ConnectionState.parse_ready` clears that registry every time, and `DiscordBot` overrides the
one nextcord hook that rebuilds it. What an empty registry costs is not a missing command but a
deleted one: nextcord's lazy-load path answers an interaction it cannot resolve by deleting
every command Discord holds, so the failure is silent, global and outlives the process.

`on_connect` is called unbound on a stub rather than on a real bot: constructing one loads all
fifteen cogs, and the contract under test is one call in one method.
"""

from __future__ import annotations

from discordbot.cli import DiscordBot

from tests.helpers.casting import as_discord_bot


class _ConnectStub:
    """The two attributes `DiscordBot.on_connect` touches."""

    def __init__(self) -> None:
        """Starts with no logged-in user, so the method stops at its own guard."""
        self.user = None
        self.rebuilds = 0

    def add_all_application_commands(self) -> None:
        """Stands in for nextcord's registry rebuild."""
        self.rebuilds += 1


async def test_on_connect_rebuilds_the_application_command_registry() -> None:
    """Every READY has to repopulate the registry, including one with no user resolved yet."""
    stub = _ConnectStub()

    await DiscordBot.on_connect(as_discord_bot(fake=stub))

    assert stub.rebuilds == 1
