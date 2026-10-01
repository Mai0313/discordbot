"""Tests for `cli.DiscordBot`: cog discovery, the command registry, the message reward, error logs.

`ConnectionState.parse_ready` clears the application-command registry on every gateway READY,
and `DiscordBot` overrides the one nextcord hook that rebuilds it. What an empty registry costs
is not a missing command but a deleted one: nextcord's lazy-load path answers an interaction it
cannot resolve by deleting every command Discord holds, so the failure is silent, global and
outlives the process.

Every method is called unbound on a stub rather than on a real bot: constructing one loads every
cog, and each contract under test lives in one method.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
import asyncio
import inspect
from functools import partial

import pytest
from nextcord.errors import ApplicationInvokeError

from discordbot import cli
from discordbot.cli import DiscordBot
from discordbot.services.economy.database import CreditResult

from tests.helpers.casting import as_message, as_discord_bot
from tests.helpers.discord_mocks import FakeUser

if TYPE_CHECKING:
    from pathlib import Path

    from nextcord import Interaction
    from nextcord.ext import commands
    from nextcord.errors import ApplicationError


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

    Read off the source, so a statement's position is pinned without stubbing every call
    `on_ready` makes.
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
    other test: none records the order of the two calls, so swapping the two lines runs green.
    """
    calls = _on_ready_calls()

    assert calls.index("_count_registered_commands") < calls.index("sync_all_application_commands")


async def test_the_stale_public_message_sweep_runs_once_even_when_the_sync_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep is the process's own, so nothing later in the first `on_ready` can cost it."""
    swept: list[object] = []

    async def record_sweep(bot: object) -> None:
        """Stands in for the sweep, recording which bot it ran for."""
        swept.append(bot)

    async def fail_sync() -> None:
        """Fails the way a Discord outage during the command sync does."""
        raise RuntimeError("sync failed")

    monkeypatch.setattr(cli, "delete_tracked_public_messages", record_sweep)
    stub = SimpleNamespace(
        _initial_setup_done=False,
        _startup_tasks=set(),
        user=FakeUser(user_id=999, bot=True),
        _count_registered_commands=partial(asyncio.sleep, delay=0),
        sync_all_application_commands=fail_sync,
    )
    bot = as_discord_bot(fake=stub)

    with pytest.raises(RuntimeError, match="sync failed"):
        await DiscordBot.on_ready(bot)
    await asyncio.gather(*stub._startup_tasks)
    await DiscordBot.on_ready(bot)

    assert swept == [bot]


def _reward_bot(**state: object) -> SimpleNamespace:
    """A bot double carrying everything `on_message`'s reward path reads off a real bot.

    These tests invoke `cli.DiscordBot.on_message` UNBOUND with this namespace as `self`, so the
    double has to answer every attribute the real method reaches for — the cooldown map, its
    prune stamp, and the prune helper itself. `on_message` calls that helper as an ordinary
    `self.` method, so it is bound here rather than being reached through the class.
    """
    bot = SimpleNamespace(
        user=FakeUser(user_id=999, bot=True), _message_reward_at={}, _message_reward_pruned_at=0.0
    )
    bot.__dict__.update(state)
    bot._prune_message_reward_cooldowns = partial(
        cli.DiscordBot._prune_message_reward_cooldowns, as_discord_bot(fake=bot)
    )
    return bot


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

    with pytest.raises(RuntimeError, match="half_moved is under cogs/ but is not a cog"):
        _load_cogs_from(root=tmp_path, monkeypatch=monkeypatch)


async def test_cli_message_reward_pays_a_member_and_never_the_bot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A member's message earns the base reward; the bot's own message earns nothing."""
    rewards: list[dict[str, Any]] = []

    async def record_reward(**kwargs: Any) -> CreditResult:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Records base reward arguments and returns a fake credit result."""
        rewards.append(kwargs)
        return CreditResult(new_balance=5_000)

    monkeypatch.setattr(target=cli, name="credit_with_repayment", value=record_reward)
    bot = _reward_bot()
    user_message = SimpleNamespace(author=FakeUser(user_id=1, bot=False), guild=None)
    await cli.DiscordBot.on_message(
        as_discord_bot(fake=bot), message=as_message(fake=user_message)
    )
    assert rewards[0]["amount"] == cli.BASE_MESSAGE_REWARD_AMOUNT
    await cli.DiscordBot.on_message(
        as_discord_bot(fake=bot),
        message=as_message(fake=SimpleNamespace(author=bot.user, guild=None)),
    )
    assert len(rewards) == 1


async def test_cli_message_reward_cooldown_suppresses_rapid_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second message within the cooldown earns nothing; a later one earns again."""
    rewards: list[dict[str, Any]] = []

    async def record_reward(**kwargs: Any) -> CreditResult:  # noqa: ANN401 -- command facade double
        rewards.append(kwargs)
        return CreditResult(new_balance=10)

    monkeypatch.setattr(target=cli, name="credit_with_repayment", value=record_reward)
    bot = _reward_bot()
    message = SimpleNamespace(author=FakeUser(user_id=1, bot=False), guild=None)

    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))
    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))
    assert len(rewards) == 1

    # Backdate the last-reward stamp so the cooldown window has elapsed.
    bot._message_reward_at[1] -= cli.MESSAGE_REWARD_COOLDOWN_SECONDS + 1
    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))
    assert len(rewards) == 2


async def test_cli_message_reward_cooldown_prunes_expired_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expired per-user cooldown slots are dropped lazily on later messages."""
    rewards: list[dict[str, Any]] = []

    async def record_reward(**kwargs: Any) -> CreditResult:  # noqa: ANN401 -- command facade double
        rewards.append(kwargs)
        return CreditResult(new_balance=10)

    monkeypatch.setattr(target=cli, name="credit_with_repayment", value=record_reward)
    monkeypatch.setattr(target=cli, name="monotonic", value=lambda: 1_000.0)
    bot = _reward_bot(_message_reward_at={1: 900.0, 2: 975.0})

    await cli.DiscordBot.on_message(
        as_discord_bot(fake=bot),
        message=as_message(
            fake=SimpleNamespace(author=FakeUser(user_id=3, bot=False), guild=None)
        ),
    )

    assert 1 not in bot._message_reward_at
    assert bot._message_reward_at[2] == 975.0
    assert bot._message_reward_at[3] == 1_000.0
    assert len(rewards) == 1


async def test_cli_message_reward_cooldown_rolls_back_on_credit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed credit must not leave the user on cooldown for the next message."""
    attempts = 0

    async def flaky_reward(**kwargs: Any) -> CreditResult:  # noqa: ANN401 -- command facade double
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient DB failure")
        return CreditResult(new_balance=10)

    monkeypatch.setattr(target=cli, name="credit_with_repayment", value=flaky_reward)
    bot = _reward_bot()
    message = SimpleNamespace(author=FakeUser(user_id=1, bot=False), guild=None)

    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))
    # The first credit failed, so the slot is rolled back and the next message retries.
    assert 1 not in bot._message_reward_at
    await cli.DiscordBot.on_message(as_discord_bot(fake=bot), message=as_message(fake=message))
    assert attempts == 2
    assert bot._message_reward_at.get(1) is not None


async def test_message_reward_stores_guild_avatar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Base message rewards pass the guild avatar into the economy DB facade."""
    captured_avatar_url = ""

    async def fake_credit_with_repayment(
        user_id: int, name: str, avatar_url: str, amount: int
    ) -> SimpleNamespace:
        """Records the avatar URL passed to the DB facade."""
        nonlocal captured_avatar_url
        del user_id, name, amount
        captured_avatar_url = avatar_url
        return SimpleNamespace(new_balance=0)

    recorded_participation: list[tuple[int, int]] = []

    async def fake_record_guild_participant(guild_id: int, user_id: int) -> None:
        """Records the participation upsert instead of writing to the live economy DB."""
        recorded_participation.append((guild_id, user_id))

    monkeypatch.setattr(cli, "credit_with_repayment", fake_credit_with_repayment)
    monkeypatch.setattr(cli, "record_guild_participant", fake_record_guild_participant)
    author = FakeUser(user_id=7, avatar_url="https://cdn.test/global.png")
    cached_member = FakeUser(user_id=7, avatar_url="https://cdn.test/global.png")
    cached_member.__dict__["guild_avatar"] = SimpleNamespace(url="https://cdn.test/server.png")
    guild = SimpleNamespace(id=100, get_member={cached_member.id: cached_member}.get)
    message = SimpleNamespace(author=author, guild=guild)

    await cli.DiscordBot.on_message(
        as_discord_bot(fake=_reward_bot()), message=as_message(fake=message)
    )

    assert captured_avatar_url == "https://cdn.test/server.png"
    # The faucet is the only bulk source of central-bank participation, and it rides the
    # reward rather than every message.
    assert recorded_participation == [(100, 7)]


async def test_cli_reports_a_failing_slash_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising slash command reaches `./data/logs` naming the type nextcord wrapped.

    This is the only command-error surface the bot actually has: it registers no prefix
    commands and never passes `command_prefix`, so nextcord defaults it to `()` and
    `get_context` can never resolve one — which is why the `on_command_*` pair that used to
    live here could not fire. nextcord's own default prints to `sys.stderr`, which
    `_TeeStream` does not tee, so before this the traceback reached no file at all.
    """
    logged: list[dict[str, Any]] = []

    def record_error(_message: str, **kwargs: Any) -> None:  # noqa: ANN401 -- logfire accepts arbitrary fields
        """Records the unhandled-command-error log."""
        logged.append(kwargs)

    monkeypatch.setattr(cli.logfire, "error", record_error)
    bot = SimpleNamespace(user=FakeUser(user_id=999, bot=True))
    interaction = SimpleNamespace(
        application_command=SimpleNamespace(qualified_name="demo"),
        guild_id=1,
        user=FakeUser(user_id=1),
    )
    await cli.DiscordBot.on_application_command_error(
        as_discord_bot(fake=bot),
        cast("Interaction[commands.Bot]", interaction),
        cast("ApplicationError", ApplicationInvokeError(ValueError("boom"))),
    )
    assert logged[-1]["error_type"] == "ValueError"
    assert logged[-1]["command"] == "demo"
    assert logged[-1]["guild_id"] == 1


async def test_cli_reports_an_exception_from_any_event_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wider half of the same gap the test above closes.

    `_run_event` funnels every unhandled exception from every event handler and every cog
    listener into `on_error`, whose default prints to `sys.stderr` — untee'd, so an
    `on_message` or an expansion cog's `on_ready` sweep that raised left no line at all.
    """
    logged: list[dict[str, Any]] = []

    def record_error(_message: str, **kwargs: Any) -> None:  # noqa: ANN401 -- logfire accepts arbitrary fields
        """Records the unhandled-event log."""
        logged.append(kwargs)

    monkeypatch.setattr(cli.logfire, "error", record_error)
    try:
        raise ValueError("boom")
    except ValueError:
        # Dispatched with the failing handler's own arguments, so the call carries one:
        # a signature narrowed to the event name alone binds this test but not a real event.
        await cli.DiscordBot.on_error(
            as_discord_bot(fake=SimpleNamespace()), "on_message", SimpleNamespace()
        )

    assert logged[-1]["event_method"] == "on_message"
    assert isinstance(logged[-1]["_exc_info"], ValueError)


async def test_cli_counts_registered_commands_and_survives_a_failed_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero here against a non-zero local count is the wiped-registry signature.

    It is a diagnostic taken on the way into the sync, so a read that fails costs a log
    field rather than the boot.
    """
    warned: list[dict[str, Any]] = []

    def record_warn(_message: str, **kwargs: Any) -> None:  # noqa: ANN401 -- logfire accepts arbitrary fields
        """Records the could-not-read warning."""
        warned.append(kwargs)

    monkeypatch.setattr(cli.logfire, "warn", record_warn)

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
