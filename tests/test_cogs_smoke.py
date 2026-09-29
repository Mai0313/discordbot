"""Smoke tests for cogs, setup hooks, and high-level Discord command branches."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_args
from pathlib import Path
from functools import partial
from importlib import import_module

from nextcord import Embed
from nextcord.errors import ApplicationInvokeError
from logfire._internal.constants import LEVEL_NUMBERS

from discordbot import cli
from discordbot.typings.config import LoggingConfig
from discordbot.cogs.template.cog import TemplateCogs
from discordbot.services.economy.database import CreditResult

from tests.helpers.casting import as_bot, as_message, as_discord_bot
from tests.helpers.discord_mocks import FakeUser, FakeInteraction, FakeDiscordMessage

if TYPE_CHECKING:
    import pytest
    from nextcord import Interaction
    from nextcord.ext import commands
    from nextcord.errors import ApplicationError


async def test_template_on_message_and_ping() -> None:
    """Verifies template debug reaction and ping command response."""
    cog = TemplateCogs(bot=as_bot(fake=SimpleNamespace(latency=0.123)))
    message = FakeDiscordMessage()
    message.__dict__["author"] = FakeUser(bot=False)
    message.__dict__["content"] = "debug"
    await cog.on_message(message=as_message(fake=message))
    assert message.reactions == ["🤬"]

    bot_message = FakeDiscordMessage()
    bot_message.__dict__["author"] = FakeUser(bot=True)
    bot_message.__dict__["content"] = "debug"
    await cog.on_message(message=as_message(fake=bot_message))
    assert bot_message.reactions == []

    interaction = FakeInteraction(user=FakeUser(display_name="Alice"))
    await TemplateCogs.ping.callback(cog, interaction)
    embed = interaction.followup.sent[0]["embed"]
    assert isinstance(embed, Embed)
    assert embed.title == ":ping_pong: Pong!"


def test_setup_functions_register_cogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """EVERY cog directory's sync `setup` adds its own cog, with `override=True`.

    Swept off the same directory scan `_load_cogs_sync` performs rather than a hand-written
    list, which is what left half the cogs uncovered before: `setup` is the one function in
    a cog module the loader calls by name, and an `async def setup` never reaches `add_cog`
    here, while at boot it aborts `DiscordBot()` with `ExtensionFailed`.
    """
    added: list[tuple[commands.Cog, bool | None]] = []

    def record_cog(cog: commands.Cog, override: bool | None = None) -> None:
        """Records the cog instance and override flag passed to add_cog."""
        added.append((cog, override))

    bot = SimpleNamespace(add_cog=record_cog)
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    cogs_dir = Path(cli.__file__).resolve().parent / "cogs"
    names = sorted(entry.name for entry in cogs_dir.iterdir() if (entry / "cog.py").is_file())

    assert names  # a scan that found nothing would pass every assertion below
    for name in names:
        module = import_module(f"discordbot.cogs.{name}.cog")
        module.setup(bot=as_bot(fake=bot))
        cog, override = added[-1]
        # `override=True` is what lets a reload replace the cog instead of colliding, and
        # the module check is what stops a re-export standing in for a missing cog.
        assert override is True, name
        assert type(cog).__module__ == module.__name__, name
    assert len(added) == len(names)


def test_cli_load_cogs_sync_discovers_exactly_the_cog_directories(tmp_path: Path) -> None:
    """Verifies synchronous cog loading discovers exactly the cog directories."""
    loaded: list[tuple[list[str], bool]] = []

    def record_load_extensions(modules: list[str], stop_at_error: bool) -> None:
        """Records modules passed to load_extensions."""
        loaded.append((modules, stop_at_error))

    bot = SimpleNamespace(load_extensions=record_load_extensions)
    cli.DiscordBot._load_cogs_sync(as_discord_bot(fake=bot))
    assert loaded[0][1] is True
    # An exact set, not a membership check: a discovery rule that grew a nested helper
    # package or lost a cog would still contain any single name you happened to test for.
    cogs_dir = Path(cli.__file__).resolve().parent / "cogs"
    expected = {
        f"discordbot.cogs.{entry.name}.cog"
        for entry in cogs_dir.iterdir()
        if entry.is_dir() and (entry / "cog.py").is_file()
    }
    assert set(loaded[0][0]) == expected
    assert "discordbot.cogs.template.cog" in expected


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


def test_log_level_setting_accepts_only_real_logfire_levels() -> None:
    """`LOG_LEVEL` is checked against logfire's own table, not a hand-copied list."""
    accepted = set(get_args(LoggingConfig.model_fields["log_level"].annotation))
    assert accepted == set(LEVEL_NUMBERS)


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
