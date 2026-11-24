"""Tests for `cli.DiscordBot`: cog discovery, the command registry, startup isolation, error logs.

`ConnectionState.parse_ready` clears the application-command registry on every gateway READY,
and `DiscordBot` overrides the one nextcord hook that rebuilds it. What an empty registry costs
is not a missing command but a deleted one: nextcord's lazy-load path answers an interaction it
cannot resolve by deleting every command Discord holds, so the failure is silent, global and
outlives the process.

Most methods are called unbound on a stub rather than on a real bot: each contract under test
lives in one method, while the subset-boot test instantiates a real `DiscordBot` over a stripped
`cogs/` tree.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
import asyncio
import logging
from importlib import import_module

import pytest
from logfire import LogfireLoggingHandler
from nextcord.errors import ApplicationInvokeError

from discordbot import cli
from discordbot.cli import DiscordBot
from discordbot.typings.config import clear_bot_config_cache
from discordbot.utils.timezone import database_now
from discordbot.utils.media_delivery import upload_limit_for

from tests.helpers.casting import as_bot, as_guild, as_discord_bot, make_invalid_webhook_token
from tests.helpers.source_tree import PACKAGE, REPO_ROOT
from tests.helpers.discord_mocks import FakeUser, FakeGuild, FakeInteraction, on_ready_bot
from tests.helpers.logfire_capture import capture_logs
from tests.helpers.message_cleanup import record_scheduled_deletes

if TYPE_CHECKING:
    from pathlib import Path
    from datetime import datetime

    from nextcord import Interaction
    from nextcord.ext import commands
    from nextcord.errors import ApplicationError


class _ConnectStub:
    """The attributes `DiscordBot.on_connect` touches."""

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


def test_interactions_carry_discords_attachment_size_limit_to_the_upload_ceiling() -> None:
    """A slash reply's ceiling is the one Discord sent, which counts the invoker's Nitro.

    nextcord's own parse drops the field, so without the override the guild's boost tier wins.
    """
    state = SimpleNamespace(http=SimpleNamespace(_HTTPClient__session=None))
    payload = {
        "id": "1",
        "type": 2,
        "token": "token",
        "version": 1,
        "application_id": "2",
        "attachment_size_limit": 1024**3,
    }

    interaction = DiscordBot.get_interaction(
        as_discord_bot(fake=SimpleNamespace(_connection=state)), data=cast("Any", payload)
    )

    guild = as_guild(fake=FakeGuild(filesize_limit=100 * 1024 * 1024))
    assert upload_limit_for(guild=guild, interaction=interaction) == 1024**3


async def test_the_registered_command_count_is_read_before_the_sync_overwrites_it() -> None:
    """Reading it after the sync would report the repair rather than the damage.

    That ordering is the whole of what the diagnostic is worth, and it is invisible to every
    other test: none records the order of the two calls, so swapping the two lines runs green.
    """
    order: list[str] = []

    async def count() -> int:
        """Records the read of what Discord holds."""
        order.append("count")
        return 0

    async def sync() -> None:
        """Records the sync, then stops `on_ready` before what the stub cannot answer."""
        order.append("sync")
        raise RuntimeError("stop after the sync")

    stub = on_ready_bot(started_at=database_now(), sync_all_application_commands=sync)
    stub._count_registered_commands = count

    with pytest.raises(RuntimeError, match="stop after the sync"):
        await DiscordBot.on_ready(as_discord_bot(fake=stub))
    await asyncio.gather(*stub._startup_tasks)

    # order-contract: the count is only worth reading before the sync overwrites it.
    assert order == ["count", "sync"]


async def test_the_stale_public_message_sweep_runs_once_even_when_the_sync_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep runs once per process while a failed command sync retries on the next ready."""
    swept: list[tuple[object, datetime]] = []
    sync_attempts = 0

    async def record_sweep(bot: object, tracked_before: datetime) -> None:
        """Stands in for the sweep, recording which bot it ran for and from when."""
        swept.append((bot, tracked_before))

    async def flaky_sync() -> None:
        """Fails on the first ready like a Discord outage, then succeeds on reconnect."""
        nonlocal sync_attempts
        sync_attempts += 1
        if sync_attempts == 1:
            raise RuntimeError("sync failed")

    monkeypatch.setattr(cli, "delete_tracked_public_messages", record_sweep)
    stub = on_ready_bot(started_at=database_now(), sync_all_application_commands=flaky_sync)
    bot = as_discord_bot(fake=stub)

    with pytest.raises(RuntimeError, match="sync failed"):
        await DiscordBot.on_ready(bot)
    await asyncio.gather(*stub._startup_tasks)
    assert stub._startup_sweeps_done is True
    assert stub._commands_synced is False

    # Reconnect retries command sync and sets `_commands_synced`, without re-running sweeps.
    await DiscordBot.on_ready(bot)
    assert stub._commands_synced is True
    assert sync_attempts == 2

    # Once synced, later reconnects run neither sweeps nor sync again.
    await DiscordBot.on_ready(bot)
    assert sync_attempts == 2
    # The process's start, not `on_ready`'s: whatever ran in between is this process's own.
    assert swept == [(bot, stub._started_at)]


async def test_on_ready_does_not_overlap_in_flight_command_sync() -> None:
    """A reconnect firing `on_ready` mid-sync must not start a second concurrent command sync."""
    sync_started = asyncio.Event()
    release_sync = asyncio.Event()
    sync_calls = 0

    async def slow_sync() -> None:
        nonlocal sync_calls
        sync_calls += 1
        sync_started.set()
        await release_sync.wait()

    stub = on_ready_bot(started_at=database_now(), sync_all_application_commands=slow_sync)
    bot = as_discord_bot(fake=stub)

    first = asyncio.create_task(coro=DiscordBot.on_ready(bot))
    await sync_started.wait()
    await DiscordBot.on_ready(bot)
    assert sync_calls == 1

    release_sync.set()
    await first
    await asyncio.gather(*stub._startup_tasks)
    assert stub._commands_synced is True
    assert stub._commands_syncing is False
    assert sync_calls == 1


def _load_cogs_from(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], bool]]:
    """Runs `_load_cogs_sync` over the `cogs/` tree under `root`, recording what it loads."""
    loaded: list[tuple[list[str], bool]] = []

    def record_load_extensions(modules: list[str], stop_at_error: bool) -> None:
        """Records modules passed to load_extensions."""
        loaded.append((modules, stop_at_error))

    monkeypatch.setattr(target=cli, name="__file__", value=str(root / "cli.py"))
    bot = SimpleNamespace(load_extensions=record_load_extensions)
    cli.DiscordBot._load_cogs_sync(as_discord_bot(fake=bot))
    return loaded


def _cog_entry(path: Path, files: tuple[str, ...]) -> None:
    """Creates one directory under `cogs/` holding the named empty files."""
    path.mkdir(parents=True)
    for name in files:
        (path / name).touch()


def test_cli_load_cogs_sync_loads_each_cog_directory_and_skips_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory holding `__init__.py` and `cog.py` loads; a plain file or `_` entry does not.

    The scan stays one level deep, so a cog's own helper subpackage is never handed to the
    loader, and one exact call pins both the list and `stop_at_error`.
    """
    cogs = tmp_path / "cogs"
    _cog_entry(path=cogs / "beta", files=("__init__.py", "cog.py"))
    _cog_entry(path=cogs / "alpha", files=("__init__.py", "cog.py", "views.py"))
    _cog_entry(path=cogs / "alpha" / "helpers", files=("__init__.py",))
    _cog_entry(path=cogs / "_draft", files=("__init__.py",))
    _cog_entry(path=cogs / "__pycache__", files=())
    (cogs / "stray.py").touch()

    loaded = _load_cogs_from(root=tmp_path, monkeypatch=monkeypatch)

    assert loaded == [(["discordbot.cogs.alpha.cog", "discordbot.cogs.beta.cog"], True)]


@pytest.mark.parametrize(argnames="present", argvalues=["__init__.py", "cog.py"])
def test_cli_load_cogs_sync_refuses_a_directory_that_is_not_a_cog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, present: str
) -> None:
    """A cog directory missing either file stops boot instead of silently never loading."""
    _cog_entry(path=tmp_path / "cogs" / "half_moved", files=(present,))

    with pytest.raises(RuntimeError, match="half_moved"):
        _load_cogs_from(root=tmp_path, monkeypatch=monkeypatch)


def test_setup_functions_register_cogs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """EVERY cog module the loader picks up has a sync `setup` that adds its own cog.

    The modules come from `_load_cogs_sync` itself rather than a hand-written list, so a new
    cog cannot be left out and the real `cogs/` tree has to pass the loader's own checks:
    `setup` is the one function in a cog module the loader calls by name, and an
    `async def setup` never reaches `add_cog` here, while at boot it aborts `DiscordBot()`
    with `ExtensionFailed`.
    """
    (tmp_path / "alembic.ini").symlink_to(target=REPO_ROOT / "alembic.ini")
    (tmp_path / "migrations").symlink_to(target=REPO_ROOT / "migrations")
    monkeypatch.chdir(path=tmp_path)
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


async def _fail_command(fake: FakeInteraction) -> None:
    """Hands the error handler a `/demo` that raised on `fake`'s interaction state."""
    interaction = SimpleNamespace(
        application_command=SimpleNamespace(qualified_name="demo"),
        guild_id=fake.guild_id,
        user=fake.user,
        response=fake.response,
        followup=fake.followup,
    )
    await cli.DiscordBot.on_application_command_error(
        as_discord_bot(fake=SimpleNamespace(user=FakeUser(user_id=999, bot=True))),
        cast("Interaction[commands.Bot]", interaction),
        cast("ApplicationError", ApplicationInvokeError(ValueError("boom"))),
    )


async def test_cli_reports_a_failing_slash_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising slash command reaches `./data/logs` naming the type nextcord wrapped.

    This is the only command-error surface the bot actually has: it registers no prefix
    commands and never passes `command_prefix`, so nextcord defaults it to `()` and
    `get_context` can never resolve one, which leaves no `on_command_*` handler able to fire.
    nextcord's own default prints to `sys.stderr`, which `_TeeStream` does not tee, so without
    this override the traceback reaches no file at all.
    """
    logged = capture_logs(monkeypatch=monkeypatch, level="error")
    fake = FakeInteraction(slash_command=True)
    await _fail_command(fake=fake)
    _message, fields = logged[-1]
    assert fields["error_type"] == "ValueError"
    assert fields["command"] == "demo"
    assert fields["guild_id"] == 100
    # Unanswered, so Discord's own failure notice is the caller's.
    assert fake.followup.sent == []


@pytest.mark.parametrize("ephemeral", [False, True])
async def test_cli_replaces_a_failed_commands_thinking_placeholder(
    monkeypatch: pytest.MonkeyPatch, ephemeral: bool
) -> None:
    """A command that raised after deferring has its placeholder filled with a failure notice.

    Discord shows nothing of its own once a command deferred, so the thinking placeholder would
    otherwise stay up for good. The notice keeps the defer's flag, so a public one expires like
    the commands' own public replies and an ephemeral one is the caller's alone.
    """
    capture_logs(monkeypatch=monkeypatch, level="error")
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    fake = FakeInteraction(slash_command=True)
    await fake.response.defer(ephemeral=ephemeral)

    await _fail_command(fake=fake)

    assert not fake.response.placeholder_pending
    assert len(fake.followup.sent) == 1
    assert len(scheduled.messages) == (0 if ephemeral else 1)


async def test_cli_tells_only_the_caller_when_an_answered_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raise after the command's own reply leaves it alone and tells the caller privately."""
    capture_logs(monkeypatch=monkeypatch, level="error")
    scheduled = record_scheduled_deletes(monkeypatch=monkeypatch)
    fake = FakeInteraction(slash_command=True)
    await fake.response.defer()
    await fake.followup.send(content="result")

    await _fail_command(fake=fake)

    assert len(fake.followup.sent) == 2
    assert fake.followup.sent[-1]["ephemeral"] is True
    assert scheduled.messages == []


async def test_cli_logs_a_failure_notice_discord_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A notice Discord refuses, such as on a token past its life, is logged instead of raised."""
    capture_logs(monkeypatch=monkeypatch, level="error")
    warned = capture_logs(monkeypatch=monkeypatch, level="warn")
    fake = FakeInteraction(slash_command=True)
    await fake.response.defer()

    async def refuse(**kwargs: object) -> None:
        """Answers the way Discord does for a dead interaction token."""
        del kwargs
        raise make_invalid_webhook_token()

    monkeypatch.setattr(target=fake.followup, name="send", value=refuse)

    await _fail_command(fake=fake)

    ((_message, fields),) = warned
    assert fields["command"] == "demo"


async def test_cli_reports_an_exception_from_any_event_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception no event handler caught reaches `./data/logs` with its traceback.

    `_run_event` funnels every unhandled exception from every event handler and every cog
    listener into `on_error`, whose default prints to `sys.stderr` — untee'd, so without this
    override an `on_message` or an expansion cog's `on_ready` sweep that raised leaves no line.
    """
    logged = capture_logs(monkeypatch=monkeypatch, level="error")
    try:
        raise ValueError("boom")
    except ValueError:
        # Dispatched with the failing handler's own arguments, so the call carries one:
        # a signature narrowed to the event name alone binds this test but not a real event.
        await cli.DiscordBot.on_error(
            as_discord_bot(fake=SimpleNamespace()), "on_message", SimpleNamespace()
        )

    _message, fields = logged[-1]
    assert fields["event_method"] == "on_message"
    assert isinstance(fields["_exc_info"], ValueError)


async def test_cli_counts_registered_commands_and_survives_a_failed_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero here against a non-zero local count is the wiped-registry signature.

    It is a diagnostic taken on the way into the sync, so a read that fails costs a log
    field rather than the boot.
    """
    warned = capture_logs(monkeypatch=monkeypatch, level="warn")

    async def two_registered(**_kwargs: Any) -> list[object]:  # noqa: ANN401 -- nextcord's own signature
        """Stands in for Discord answering with two registered commands."""
        return [object(), object()]

    async def refused(**_kwargs: Any) -> list[object]:  # noqa: ANN401 -- nextcord's own signature
        """Stands in for Discord refusing the read."""
        raise RuntimeError("refused")

    reading = SimpleNamespace(
        application_id=7, http=SimpleNamespace(get_global_commands=two_registered)
    )
    assert await cli.DiscordBot._count_registered_commands(as_discord_bot(fake=reading)) == 2

    failing = SimpleNamespace(application_id=7, http=SimpleNamespace(get_global_commands=refused))
    assert await cli.DiscordBot._count_registered_commands(as_discord_bot(fake=failing)) is None
    assert warned


def test_cli_imports_no_feature_modules() -> None:
    """`cli.py` stays feature-agnostic: zero imports from services, cogs, economy, or pricing."""
    tree = ast.parse(source=(PACKAGE / "cli.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = (
        "discordbot.services.",
        "discordbot.cogs.",
        "discordbot.typings.economy",
        "discordbot.utils.model_pricing",
    )
    offenders = sorted(name for name in imported if name.startswith(forbidden))
    assert not offenders, f"cli.py imports feature modules: {offenders}"


@pytest.mark.parametrize(
    ("kept_cogs", "expected_dbs"),
    [
        (("template",), {"cleanup.db"}),
        (("template", "economy"), {"cleanup.db", "economy.db"}),
        (("template", "games"), {"cleanup.db", "economy.db", "games.db"}),
        (("template", "memory"), {"cleanup.db", "reply.db"}),
        (("template", "parse_threads"), {"cleanup.db", "reply.db", "settings.db"}),
        (("template", "settings"), {"cleanup.db", "settings.db"}),
    ],
)
def test_discord_bot_boots_with_subsets_of_cogs_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kept_cogs: tuple[str, ...],
    expected_dbs: set[str],
) -> None:
    """`DiscordBot` boots cleanly over any subset of `cogs/` and upgrades only their databases."""
    (tmp_path / "alembic.ini").symlink_to(target=REPO_ROOT / "alembic.ini")
    (tmp_path / "migrations").symlink_to(target=REPO_ROOT / "migrations")
    fake_pkg = tmp_path / "pkg"
    for name in kept_cogs:
        _cog_entry(path=fake_pkg / "cogs" / name, files=("__init__.py", "cog.py"))
    monkeypatch.chdir(path=tmp_path)
    monkeypatch.setattr(target=cli, name="__file__", value=str(fake_pkg / "cli.py"))

    bot = DiscordBot()

    assert len(bot.cogs) == len(kept_cogs)
    assert {path.name for path in (tmp_path / "data" / "database").iterdir()} == expected_dbs
    assert not (tmp_path / "data" / "memories").exists()


def test_discord_bot_activity_defaults_and_reads_toml(
    tmp_path: Path, bot_config_isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`DiscordBot` presence defaults to `Discord` and honors `[global.bot].activity`."""
    monkeypatch.chdir(path=tmp_path)
    monkeypatch.setattr(target=cli, name="ensure_database_upgraded", value=lambda **_kwargs: None)
    monkeypatch.setattr(target=DiscordBot, name="_load_cogs_sync", value=lambda self: None)

    default_bot = DiscordBot()
    assert default_bot.activity is not None
    assert default_bot.activity.name == "Discord"

    bot_config_isolated.write_text(
        data='[global.bot]\nactivity = "custom status"\n', encoding="utf-8"
    )
    clear_bot_config_cache()
    custom_bot = DiscordBot()
    assert custom_bot.activity is not None
    assert custom_bot.activity.name == "custom status"


def test_discord_bot_init_does_not_duplicate_logfire_handlers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Instantiating `DiscordBot` repeatedly attaches at most one `LogfireLoggingHandler` per logger."""
    monkeypatch.chdir(path=tmp_path)
    monkeypatch.setattr(target=cli, name="ensure_database_upgraded", value=lambda **_kwargs: None)
    monkeypatch.setattr(target=DiscordBot, name="_load_cogs_sync", value=lambda self: None)
    for name in ("nextcord.state", "alembic"):
        logger = logging.getLogger(name=name)
        for handler in list(logger.handlers):
            if isinstance(handler, LogfireLoggingHandler):
                logger.removeHandler(hdlr=handler)

    DiscordBot()
    DiscordBot()

    for name in ("nextcord.state", "alembic"):
        logfire_handlers = [
            handler
            for handler in logging.getLogger(name=name).handlers
            if isinstance(handler, LogfireLoggingHandler)
        ]
        assert len(logfire_handlers) == 1
