"""Tests for the slash-command listener that feeds the usage records."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from nextcord import InteractionType

from discordbot.cli import DiscordBot
from discordbot.cogs.usage.cog import UsageCogs, command_path

from tests.helpers.casting import as_bot, as_interaction, as_command_interaction_data
from tests.helpers.usage_log import usage_records

if TYPE_CHECKING:
    from pathlib import Path


class FakeCommandInteraction:
    """Interaction stub carrying the raw payload the listener reads."""

    def __init__(
        self,
        data: dict[str, Any] | None,
        interaction_type: InteractionType = InteractionType.application_command,
        guild_id: int | None = 55,
        channel_id: int | None = 77,
    ) -> None:
        """Initializes the payload, interaction type, invoker and location ids."""
        self.data = data
        self.type = interaction_type
        self.user: SimpleNamespace | None = SimpleNamespace(id=42, name="tester")
        self.guild_id = guild_id
        self.channel_id = channel_id


def test_command_path_walks_to_the_invoked_subcommand() -> None:
    """A subcommand is its own unit; folding it into its group is a `split` at read time."""
    assert command_path(data=as_command_interaction_data(fake={"name": "ping", "type": 1})) == (
        "ping"
    )
    assert (
        command_path(
            data=as_command_interaction_data(
                fake={
                    "name": "games",
                    "type": 1,
                    "options": [{"name": "blackjack", "type": 1, "options": []}],
                }
            )
        )
        == "games blackjack"
    )
    # A group holding a subcommand: `/memory server show` is two levels deep.
    assert (
        command_path(
            data=as_command_interaction_data(
                fake={
                    "name": "memory",
                    "type": 1,
                    "options": [
                        {
                            "name": "server",
                            "type": 2,
                            "options": [{"name": "show", "type": 1, "options": []}],
                        }
                    ],
                }
            )
        )
        == "memory server show"
    )
    # A plain option is not a subcommand, so the walk stops at the command itself.
    assert (
        command_path(
            data=as_command_interaction_data(
                fake={
                    "name": "download_video",
                    "type": 1,
                    "options": [{"name": "url", "type": 3, "value": "https://x.test"}],
                }
            )
        )
        == "download_video"
    )


async def test_the_listener_records_one_invocation(usage_log_isolated_dir: Path) -> None:
    """The listener writes a record for an application command, wherever it was run."""
    cog = UsageCogs(bot=as_bot(fake=SimpleNamespace()))

    await cog.on_interaction(
        interaction=as_interaction(
            fake=FakeCommandInteraction(
                data={
                    "name": "memory",
                    "type": 1,
                    "options": [{"name": "clear", "type": 1, "options": []}],
                }
            )
        )
    )
    await cog.on_interaction(
        interaction=as_interaction(
            fake=FakeCommandInteraction(
                data={"name": "ping", "type": 1}, guild_id=None, channel_id=None
            )
        )
    )

    records = usage_records(directory=usage_log_isolated_dir)
    assert [record["name"] for record in records] == ["memory clear", "ping"]
    assert all(record["kind"] == "slash" for record in records)
    assert (records[0]["user_id"], records[0]["user_name"]) == (42, "tester")
    assert records[0]["guild_id"] == 55
    assert (records[1]["guild_id"], records[1]["channel_id"]) == (None, None)


def test_the_recorder_hooks_in_as_a_listener_not_an_override() -> None:
    """It has to be additive, and only a cog listener is.

    `Client.on_interaction` is the method that calls `process_application_commands`, so a
    `DiscordBot.on_interaction` override would stop every slash command executing while
    the records themselves still looked healthy.
    """
    listeners = dict(UsageCogs(bot=as_bot(fake=SimpleNamespace())).get_listeners())

    assert "on_interaction" in listeners
    assert getattr(UsageCogs.on_interaction, "__cog_listener__", False) is True
    assert "on_interaction" not in vars(DiscordBot)


async def test_the_listener_ignores_everything_that_is_not_a_command(
    usage_log_isolated_dir: Path,
) -> None:
    """Buttons, autocomplete and modals share the event; only invocations are usage."""
    cog = UsageCogs(bot=as_bot(fake=SimpleNamespace()))

    for interaction_type in (
        InteractionType.component,
        InteractionType.application_command_autocomplete,
        InteractionType.modal_submit,
        InteractionType.ping,
    ):
        await cog.on_interaction(
            interaction=as_interaction(
                fake=FakeCommandInteraction(
                    data={"name": "blackjack_hit", "type": 1}, interaction_type=interaction_type
                )
            )
        )
    # A command payload that carries neither a name nor an invoker names no feature.
    await cog.on_interaction(interaction=as_interaction(fake=FakeCommandInteraction(data=None)))
    await cog.on_interaction(
        interaction=as_interaction(fake=FakeCommandInteraction(data={"type": 1}))
    )
    anonymous = FakeCommandInteraction(data={"name": "ping", "type": 1})
    anonymous.user = None
    await cog.on_interaction(interaction=as_interaction(fake=anonymous))

    assert not usage_log_isolated_dir.exists()  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO
