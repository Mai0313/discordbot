"""What every auto-expansion cog owes, checked against all of them at once.

A pasted link is the same feature on every platform, and the whole point of it is that a reader
learns it once: the same reply slot under the link, the same reactions meaning the same things,
the same silence in the channel when it does not work out. Cogs drifting apart is what that
promise fails as, and it fails quietly — every cog passes its own tests while the set of them
stops agreeing.

Most of the shell now lives in `utils/expansion_cog.py`, so most of this file asserts that a cog
did NOT take a piece of it back: not its own status mark, not its own failure classification, not
its own listener. The one written-down list is `test_every_expansion_cog_is_accounted_for`, which
fails until a new source is named — that is what stops a source arriving with nobody having read
this file.

What deliberately is NOT here: how a post is rendered. A Threads chain, a Facebook comment
preload and a Douyin clip are different things and their cards should differ.
"""

from types import SimpleNamespace
from typing import Any
import inspect
from pathlib import Path
import importlib

import pytest
from nextcord.ext import commands

from discordbot.utils import expansion_placeholder as expansion_module
from discordbot.typings.emojis import LINK_SOURCE_EMOJIS
from discordbot.utils.link_errors import LinkRetryableError, LinkUnavailableError
from discordbot.utils.expansion_cog import ExpansionCog
from discordbot.utils.expansion_placeholder import (
    EXPANSION_DONE_EMOJI,
    EXPANSION_FAILED_EMOJI,
    EXPANSION_WORKING_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
    expansion_failure_emoji,
    report_expansion_read_failure,
)

from tests.helpers.casting import as_bot, as_message
from tests.helpers.discord_mocks import FakeUser, FakeDiscordMessage

_COGS_DIR = Path(__file__).resolve().parents[1] / "src" / "discordbot" / "cogs"

# Every status mark an expansion may answer with lives in one module, so a literal left in a cog
# is the drift this file exists to catch: it is what lets one platform quietly answer ⚠️ where the
# others answer ⏱️. The platform markers in `typings/emojis.py` are not status marks and stay
# where they are.
_STATUS_LITERALS = (
    EXPANSION_WORKING_EMOJI,
    EXPANSION_DONE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_FAILED_EMOJI,
)

# Calls that decide an outcome for every platform at once. A cog making one of them is deciding
# for itself again, which is how the marks and the log levels stopped agreeing before the shell
# existed.
_SHARED_DECISIONS = (
    "expansion_failure_emoji(",
    "report_expansion_read_failure(",
    "report_expansion_delivery_failure(",
    "send_expansion_placeholder(",
)


def _expansion_cog_classes() -> list[type[ExpansionCog[Any]]]:
    """Every cog class built on the shared expansion shell.

    Discovered by the base class rather than by a string in the source, so a cog that stops
    importing one helper cannot drop out of this file's coverage without anyone noticing.
    """
    found: list[type[ExpansionCog[Any]]] = []
    for entry in sorted(_COGS_DIR.iterdir()):
        source = entry / "cog.py"
        if entry.name.startswith("_") or not source.is_file():
            continue
        module = importlib.import_module(f"discordbot.cogs.{entry.name}.cog")
        for value in vars(module).values():
            if (
                inspect.isclass(value)
                and issubclass(value, ExpansionCog)
                and value is not ExpansionCog
                and value.__module__ == module.__name__
            ):
                found.append(value)
    return found


_COGS = _expansion_cog_classes()


def _cog_id(cog: type) -> str:
    """Names a parametrized case after the cog package it came from."""
    return cog.__module__.split(".")[-2]


def _cog_source(cog: type) -> str:
    """Reads a cog module's own source, for the checks that are about what it does not do."""
    return Path(inspect.getsourcefile(cog) or "").read_text(encoding="utf-8")


def test_every_expansion_cog_is_accounted_for() -> None:
    """The one written-down list, so a new source cannot arrive unread.

    Everything else here is discovered. This is the tripwire: a new expansion cog fails exactly
    one test, and the fix is to read this file and add its name.
    """
    assert {_cog_id(cog=cog) for cog in _COGS} == {
        "parse_douyin",
        "parse_facebook",
        "parse_instagram",
        "parse_threads",
        "parse_twitter",
    }


def test_the_discovery_finds_something() -> None:
    """An empty parameter set skips every test below it and reports success.

    That is how this file went quietly blank once already: the probe was a string in the source,
    the string moved into the shared shell, and eight parametrized tests turned into skips.
    """
    assert _COGS


@pytest.mark.parametrize("cog", _COGS, ids=_cog_id)
def test_an_expansion_cog_names_its_platform_the_shared_way(cog: type[ExpansionCog[Any]]) -> None:
    """`SOURCE` keys the pending-expansion rows, the marker lookup and the resume sweep.

    A key nothing else uses would look fine until a restart: the sweep reads its own rows by that
    string, so a private spelling resumes nothing and reports nothing either. It is also what the
    shell subscripts for the platform marker, so a stray spelling raises mid-expansion.
    """
    assert cog.SOURCE in LINK_SOURCE_EMOJIS


def test_no_two_expansion_cogs_share_a_source_key() -> None:
    """Two cogs on one key would each resume the other's interrupted expansions."""
    keys = [cog.SOURCE for cog in _COGS]

    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("cog", _COGS, ids=_cog_id)
def test_an_expansion_cog_declares_what_the_shell_asks_it_for(
    cog: type[ExpansionCog[Any]],
) -> None:
    """A cog that leaves a hook unfilled does nothing at all, and says nothing about it."""
    assert cog.PLATFORM
    assert cog.PLACEHOLDER_TEXT
    assert cog.URL_PATTERN.pattern
    assert cog.read is not ExpansionCog.read
    assert cog.build_delivery is not ExpansionCog.build_delivery


@pytest.mark.parametrize("cog", _COGS, ids=_cog_id)
def test_an_expansion_cog_keeps_the_shared_listener(cog: type[ExpansionCog[Any]]) -> None:
    """The listener, the restart sweep and the expansion body are the shell's, not a cog's.

    Overriding any of them is how the reply slot, the reaction order and the resume contract
    stopped agreeing when each cog held its own copy. `_expand` in particular is what
    `resume_expansion_placeholders` calls with the listener's own four arguments, so a cog
    redefining it can break a restart and nothing else.
    """
    assert cog.on_message is ExpansionCog.on_message
    assert cog.on_ready is ExpansionCog.on_ready
    assert cog._expand is ExpansionCog._expand
    assert cog._mark_failed is ExpansionCog._mark_failed


@pytest.mark.parametrize("cog", _COGS, ids=_cog_id)
def test_an_expansion_cog_spells_no_status_mark_of_its_own(cog: type[ExpansionCog[Any]]) -> None:
    """One symbol, one meaning, whichever platform was linked.

    The reaction is the entire report — an expansion that produced nothing says nothing in the
    channel — so a cog inventing its own mark, or reusing a shared one for a different outcome,
    is the whole feature's vocabulary coming apart.
    """
    body = "\n".join(
        line for line in _cog_source(cog=cog).split("\n") if not line.lstrip().startswith("#")
    )

    for literal in _STATUS_LITERALS:
        assert f'"{literal}"' not in body, f"{cog.__module__} spells {literal} itself"


@pytest.mark.parametrize("cog", _COGS, ids=_cog_id)
def test_an_expansion_cog_decides_no_shared_outcome_of_its_own(
    cog: type[ExpansionCog[Any]],
) -> None:
    """A cog reading the error's type itself is how the platforms stop agreeing.

    Every one of these calls answers the same question for every platform, so the shell makes
    them and a cog that makes one again has taken the decision back. `parse_douyin` used to own
    the logging split alone and was right; the others logged a deleted post at `warn` with a
    traceback, which is what makes a real regression unfindable.
    """
    source = _cog_source(cog=cog)

    for call in _SHARED_DECISIONS:
        assert call not in source, f"{cog.__module__} makes the shared decision {call}"
    assert "isinstance(error, TimeoutError)" not in source


@pytest.mark.parametrize("cog", _COGS, ids=_cog_id)
async def test_a_failure_with_nothing_on_the_message_still_names_the_platform(
    cog: type[ExpansionCog[Any]],
) -> None:
    """A cross on its own cannot say which link died, and a message can carry two.

    `current_emoji` is None exactly when claiming the reply slot failed, which is both the refused
    channel and the Discord 5xx that raises straight past it. Behavioural rather than a source
    scan, because what matters is that the marker lands whichever call site got there.
    """
    instance = cog(bot=as_bot(fake=SimpleNamespace(user=FakeUser(bot=True))))
    message = FakeDiscordMessage()

    await instance._mark_failed(message=as_message(fake=message), current_emoji=None)

    assert message.reactions == [LINK_SOURCE_EMOJIS[cog.SOURCE], EXPANSION_FAILED_EMOJI]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LinkRetryableError("429"), EXPANSION_RETRY_LATER_EMOJI),
        (TimeoutError(), EXPANSION_RETRY_LATER_EMOJI),
        (LinkUnavailableError("410"), EXPANSION_UNREADABLE_EMOJI),
        (RuntimeError("the parser blew up"), EXPANSION_FAILED_EMOJI),
    ],
    ids=["refused", "stalled", "gone", "broke"],
)
def test_one_failure_earns_the_same_mark_on_every_platform(
    error: Exception, expected: str
) -> None:
    """The vocabulary is only worth anything if the same failure reads the same everywhere.

    This pins the mapping itself; `test_an_expansion_cog_decides_no_shared_outcome_of_its_own` is
    what says every cog actually goes through it. Parametrizing this one over the cogs too would
    have looked like every platform was checked while testing one function repeatedly.
    """
    assert expansion_failure_emoji(error=error) == expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LinkUnavailableError("410"), "info"),
        (LinkRetryableError("429"), "warn"),
        (TimeoutError(), "warn"),
        (RuntimeError("the parser blew up"), "error"),
    ],
    ids=["gone", "refused", "stalled", "broke"],
)
def test_a_read_failure_is_logged_at_the_severity_its_outcome_earns(
    error: Exception, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ladder is keyed on how tolerable the failure is, not on how deep it happened.

    A deleted post logged at `warn` with a traceback is what makes a real regression unfindable,
    and it is `.github/CONTRIBUTING.md#logging`'s own example of `info`.
    """
    levels: list[str] = []
    for level in ("info", "warn", "error"):
        monkeypatch.setattr(
            target=expansion_module.logfire,
            name=level,
            value=lambda _message, level=level, **fields: levels.append(level),
        )

    report_expansion_read_failure(
        error=error, platform="Threads", url="https://example.test/p/1", message_id=7
    )

    assert levels == [expected]


def test_a_routine_remote_outcome_carries_its_reason_and_no_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A traceback for a deleted post is noise; the platform's own words are not.

    Douyin's filter reason exists in no other line, which is why the `info` branch keeps it while
    dropping the exception the ladder says that level usually does not carry.
    """
    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        target=expansion_module.logfire,
        name="info",
        value=lambda _message, **fields: recorded.update(fields),
    )

    report_expansion_read_failure(
        error=LinkUnavailableError("Douyin will not serve 123: filtered"),
        platform="Douyin",
        url="https://example.test/p/1",
        message_id=7,
    )

    assert "filtered" in str(recorded["reason"])
    assert "_exc_info" not in recorded
    assert recorded["message_id"] == 7


def test_the_shell_is_not_itself_a_loadable_cog() -> None:
    """`_load_cogs_sync` scans `cogs/` one level deep, so the base must not live there.

    It is a `commands.Cog` subclass with listeners of its own; a copy under `cogs/` would be
    loaded and would answer every message with a `NotImplementedError`.
    """
    assert issubclass(ExpansionCog, commands.Cog)
    assert not (_COGS_DIR / "expansion_cog").exists()
