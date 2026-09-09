"""Guards the application-command registry against being left empty after a gateway READY.

`ConnectionState.parse_ready` clears that registry every time, and `DiscordBot` overrides the
one nextcord hook that rebuilds it. What an empty registry costs is not a missing command but a
deleted one: nextcord's lazy-load path answers an interaction it cannot resolve by deleting
every command Discord holds, so the failure is silent, global and outlives the process.

`on_connect` is called unbound on a stub rather than on a real bot: constructing one loads all
fifteen cogs, and the contract under test is one call in one method.
"""

from __future__ import annotations

import ast
import inspect

from discordbot import cli
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


def _on_ready_calls() -> list[str]:
    """Every attribute call in `DiscordBot.on_ready`, in source order.

    Read off the source rather than by driving `on_ready`, which would need a stub for the
    sync, both `tasks.Loop` starts and `application_info`, to pin one statement's position.
    """
    module = ast.parse(inspect.getsource(cli))
    bot = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DiscordBot"
    )
    on_ready = next(
        node
        for node in bot.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "on_ready"
    )
    calls = [
        (node.lineno, node.col_offset, node.func.attr)
        for node in ast.walk(on_ready)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    return [attr for _, _, attr in sorted(calls)]


def test_the_registered_command_count_is_read_before_the_sync_overwrites_it() -> None:
    """Reading it after the sync would report the repair rather than the damage.

    That ordering is the whole of what the diagnostic is worth, and it is invisible to every
    other test: nothing in the suite drives `on_ready`, so swapping the two lines runs green.
    """
    calls = _on_ready_calls()

    assert calls.index("_count_registered_commands") < calls.index("sync_all_application_commands")
