"""Tests for the append-only usage records and the slash-command listener that feeds them."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from pathlib import Path

from nextcord import InteractionType

from discordbot.cli import DiscordBot
from discordbot.cogs.usage.cog import UsageCogs, command_path
from discordbot.utils.timezone import database_now
from discordbot.utils.usage_log import UsageRecord, UsageRecorder, UsageLogConfig

from tests.helpers.casting import as_bot, as_interaction, as_command_interaction_data
from tests.helpers.usage_log import usage_records

if TYPE_CHECKING:
    import pytest


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


async def test_a_record_lands_in_its_own_month_file(usage_log_isolated_dir: Path) -> None:
    """One use is one JSON line, in the file named after the month it happened in."""
    await UsageRecorder().record(
        kind="slash",
        name="games blackjack",
        user_id=1,
        user_name="tester",
        guild_id=2,
        channel_id=3,
    )

    month_file = usage_log_isolated_dir / f"{database_now():%Y-%m}.jsonl"
    assert [path.name for path in usage_log_isolated_dir.iterdir()] == [month_file.name]  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO
    (record,) = usage_records(directory=usage_log_isolated_dir)
    # The exact key set is the privacy decision: who and where, and nothing about what
    # they typed. The username rides along so an operator can read the file, but it is a
    # label beside the id rather than a second identifier. Nothing prunes these files.
    assert set(record) == {"at", "kind", "name", "user_id", "user_name", "guild_id", "channel_id"}
    assert record["kind"] == "slash"
    assert record["name"] == "games blackjack"
    assert (record["user_id"], record["user_name"]) == (1, "tester")
    assert (record["guild_id"], record["channel_id"]) == (2, 3)
    # Stamped in Asia/Taipei with the offset present, so grouping by day is a string slice
    # and the value is still unambiguous.
    assert record["at"].endswith("+08:00")
    assert record["at"].startswith(f"{database_now():%Y-%m-%d}")


async def test_each_use_appends_its_own_line(usage_log_isolated_dir: Path) -> None:
    """A record never rewrites the ones before it."""
    recorder = UsageRecorder()

    await recorder.record(
        kind="slash", name="ping", user_id=1, user_name="a", guild_id=2, channel_id=3
    )
    await recorder.record(
        kind="reply", name="QA", user_id=1, user_name="a", guild_id=None, channel_id=3
    )

    records = usage_records(directory=usage_log_isolated_dir)
    assert [(record["kind"], record["name"]) for record in records] == [
        ("slash", "ping"),
        ("reply", "QA"),
    ]
    # A DM has no guild, and the field says so rather than being dropped.
    assert records[1]["guild_id"] is None


async def test_the_kill_switch_writes_nothing(
    usage_log_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`USAGE_LOG_ENABLED=false` records nothing and creates no directory."""
    monkeypatch.setenv(name="USAGE_LOG_ENABLED", value="false")

    await UsageRecorder().record(
        kind="slash", name="ping", user_id=1, user_name="a", guild_id=2, channel_id=3
    )

    assert not usage_log_isolated_dir.exists()  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO


async def test_a_write_failure_never_reaches_the_caller(usage_log_isolated_dir: Path) -> None:
    """Recording must never cost the thing it records."""
    usage_log_isolated_dir.write_text(data="not a directory", encoding="utf-8")  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO

    await UsageRecorder().record(
        kind="slash", name="ping", user_id=1, user_name="a", guild_id=2, channel_id=3
    )

    assert usage_log_isolated_dir.read_text(encoding="utf-8") == "not a directory"  # noqa: ASYNC240 -- a tmp_path entry, not blocking IO


def test_a_record_written_before_usernames_still_parses() -> None:
    """`user_name` has to stay optional on the model, whatever the writer now sends.

    A month file is append-only and never rewritten, so the records from before the field
    existed keep their old shape forever. Making it required would not lose one line, it
    would make every reader reject every month written up to that point.
    """
    record = UsageRecord.model_validate_json(
        '{"at":"2026-08-03T03:29:00+08:00","kind":"reply","name":"QA","user_id":1,'
        '"guild_id":2,"channel_id":3}'
    )

    assert (record.user_id, record.user_name) == (1, "")


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
