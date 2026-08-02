"""Tests for the append-only usage records and the slash-command listener that feeds them."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from pathlib import Path

from nextcord import InteractionType

from discordbot.cogs.usage.cog import UsageCogs, setup, command_path
from discordbot.utils.timezone import database_now
from discordbot.utils.usage_log import UsageRecorder, UsageLogConfig

from tests.helpers.casting import as_bot, as_interaction, as_command_interaction_data

if TYPE_CHECKING:
    import pytest
    from nextcord.ext import commands


def _recorder(directory: Path, enabled: bool = True) -> UsageRecorder:
    """Builds a recorder pointed at a throwaway directory.

    `model_validate` over the env-alias names keeps the alias spelling type-clean and,
    unlike `__init__`, never merges the ambient environment in.
    """
    return UsageRecorder(
        config=UsageLogConfig.model_validate({
            "USAGE_LOG_ENABLED": enabled,
            "USAGE_LOG_DIR": str(directory),
        })
    )


def _lines(directory: Path) -> list[dict[str, Any]]:
    """Reads every recorded line back, parsed as its own JSON object."""
    return [
        json.loads(line)
        for path in sorted(directory.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
        self.user: SimpleNamespace | None = SimpleNamespace(id=42)
        self.guild_id = guild_id
        self.channel_id = channel_id


async def test_a_record_lands_in_its_own_month_file(tmp_path: Path) -> None:
    """One use is one JSON line, in the file named after the month it happened in."""
    usage_dir = tmp_path / "usage"

    await _recorder(directory=usage_dir).record(
        kind="slash", name="games blackjack", user_id=1, guild_id=2, channel_id=3
    )

    month_file = usage_dir / f"{database_now():%Y-%m}.jsonl"
    assert [path.name for path in usage_dir.iterdir()] == [month_file.name]
    (record,) = _lines(directory=usage_dir)
    # The exact key set is the privacy decision: an id says who, and nothing here says
    # what they typed or what they are called. Nothing prunes these files.
    assert set(record) == {"at", "kind", "name", "user_id", "guild_id", "channel_id"}
    assert record["kind"] == "slash"
    assert record["name"] == "games blackjack"
    assert (record["user_id"], record["guild_id"], record["channel_id"]) == (1, 2, 3)
    # Stamped in Asia/Taipei with the offset present, so grouping by day is a string slice
    # and the value is still unambiguous.
    assert record["at"].endswith("+08:00")
    assert record["at"].startswith(f"{database_now():%Y-%m-%d}")


async def test_each_use_appends_its_own_line(tmp_path: Path) -> None:
    """A record never rewrites the ones before it."""
    usage_dir = tmp_path / "usage"
    recorder = _recorder(directory=usage_dir)

    await recorder.record(kind="slash", name="ping", user_id=1, guild_id=2, channel_id=3)
    await recorder.record(kind="reply", name="QA", user_id=1, guild_id=None, channel_id=3)

    records = _lines(directory=usage_dir)
    assert [(record["kind"], record["name"]) for record in records] == [
        ("slash", "ping"),
        ("reply", "QA"),
    ]
    # A DM has no guild, and the field says so rather than being dropped.
    assert records[1]["guild_id"] is None


async def test_the_kill_switch_writes_nothing(tmp_path: Path) -> None:
    """`USAGE_LOG_ENABLED=false` records nothing and creates no directory."""
    usage_dir = tmp_path / "usage"

    await _recorder(directory=usage_dir, enabled=False).record(
        kind="slash", name="ping", user_id=1, guild_id=2, channel_id=3
    )

    assert not usage_dir.exists()


async def test_a_write_failure_never_reaches_the_caller(tmp_path: Path) -> None:
    """Recording must never cost the thing it records."""
    blocked = tmp_path / "usage"
    blocked.write_text(data="not a directory", encoding="utf-8")

    await _recorder(directory=blocked).record(
        kind="slash", name="ping", user_id=1, guild_id=2, channel_id=3
    )

    assert blocked.read_text(encoding="utf-8") == "not a directory"


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


async def test_the_listener_records_one_invocation(tmp_path: Path) -> None:
    """The listener writes a record for an application command, wherever it was run."""
    usage_dir = tmp_path / "usage"
    cog = UsageCogs(bot=as_bot(fake=SimpleNamespace()))
    cog.usage_recorder = _recorder(directory=usage_dir)

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

    records = _lines(directory=usage_dir)
    assert [record["name"] for record in records] == ["memory clear", "ping"]
    assert all(record["kind"] == "slash" for record in records)
    assert (records[0]["user_id"], records[0]["guild_id"]) == (42, 55)
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


async def test_the_listener_ignores_everything_that_is_not_a_command(tmp_path: Path) -> None:
    """Buttons, autocomplete and modals share the event; only invocations are usage."""
    usage_dir = tmp_path / "usage"
    cog = UsageCogs(bot=as_bot(fake=SimpleNamespace()))
    cog.usage_recorder = _recorder(directory=usage_dir)

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

    assert not usage_dir.exists()


def test_setup_registers_the_cog(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cog loads through the sync `setup` every cog module exposes."""
    added: list[commands.Cog] = []
    bot = SimpleNamespace(add_cog=lambda cog, override: added.append(cog))

    setup(bot=as_bot(fake=bot))

    assert isinstance(added[0], UsageCogs)


def test_the_recorder_defaults_to_the_data_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """The records live beside the bot's other durable state, not in `data/logs`.

    The runtime log is debug-level, hand-cleaned and gated on `LOG_LEVEL`; a usage history
    kept inside it would die with it or silently stop recording.
    """
    monkeypatch.delenv(name="USAGE_LOG_DIR", raising=False)
    monkeypatch.delenv(name="USAGE_LOG_ENABLED", raising=False)

    config = UsageLogConfig()

    assert Path(config.directory) == Path("./data/usage")
    assert config.enabled is True
