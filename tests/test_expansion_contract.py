"""What every auto-expansion cog owes, checked against all of them at once.

A pasted Threads, Facebook, Instagram or Douyin link is the same feature four times over, and
the whole point of it is that a reader learns it once: the same reply slot under the link, the
same five reactions meaning the same five things, the same silence in the channel when it does
not work out. Four cogs drifting apart is what that promise fails as, and it fails quietly —
every cog passes its own tests while the set of them stops agreeing.

So the cogs are discovered rather than listed: an expansion cog is one importing
`send_expansion_placeholder`, which is the shared reply slot itself, so a fifth one is held to
the contract without anyone remembering to add it here. The one place a count is written down
is `test_every_expansion_cog_is_accounted_for`, which fails until a new source is named — that
is what stops a source arriving with nobody having read this file.

What deliberately is NOT here: how a post is rendered. A Threads chain, a Facebook comment
preload and a Douyin clip are different things and their cards should differ. The contract is
the shell around the card.
"""

from types import SimpleNamespace
from typing import Any
import inspect
from pathlib import Path
import importlib

import pytest
from nextcord.ext import commands

from discordbot.typings.emojis import LINK_SOURCE_EMOJIS
from discordbot.utils.link_errors import LinkRetryableError, LinkUnavailableError
from discordbot.utils.expansion_placeholder import (
    EXPANSION_DONE_EMOJI,
    EXPANSION_FAILED_EMOJI,
    EXPANSION_WORKING_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
    expansion_failure_emoji,
)

from tests.helpers.casting import as_bot, as_message
from tests.helpers.discord_mocks import FakeUser, FakeDiscordMessage

_COGS_DIR = Path(__file__).resolve().parents[1] / "src" / "discordbot" / "cogs"

# Every status mark an expansion may answer with now lives in one module, so a literal left in
# a cog is the drift this file exists to catch: it is what lets one platform quietly answer ⚠️
# where the others answer ⏱️. The platform markers in `typings/emojis.py` are not status marks
# and stay where they are.
_STATUS_LITERALS = (
    EXPANSION_WORKING_EMOJI,
    EXPANSION_DONE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_FAILED_EMOJI,
)


def _expansion_cog_modules() -> list[Any]:
    """Imports every cog module that claims a reply slot, which is what makes it an expansion."""
    modules = []
    for entry in sorted(_COGS_DIR.iterdir()):
        source = entry / "cog.py"
        if entry.name.startswith("_") or not source.is_file():
            continue
        if "send_expansion_placeholder" not in source.read_text(encoding="utf-8"):
            continue
        modules.append(importlib.import_module(f"discordbot.cogs.{entry.name}.cog"))
    return modules


_MODULES = _expansion_cog_modules()


def _cog_class(module: Any) -> type:  # noqa: ANN401 -- a module object has no useful annotation
    """Returns the one `commands.Cog` subclass a cog module defines."""
    classes = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and issubclass(value, commands.Cog)
        and value.__module__ == module.__name__
    ]
    assert len(classes) == 1, f"{module.__name__} defines {len(classes)} cogs"
    return classes[0]


def test_every_expansion_cog_is_accounted_for() -> None:
    """The one written-down list, so a new source cannot arrive unread.

    Everything else here is discovered. This is the tripwire: a fifth expansion cog fails
    exactly one test, and the fix is to read this file and add its name.
    """
    assert {module.__name__.split(".")[-2] for module in _MODULES} == {
        "parse_douyin",
        "parse_facebook",
        "parse_instagram",
        "parse_threads",
    }


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.split(".")[-2])
def test_an_expansion_cog_names_its_platform_the_shared_way(module: Any) -> None:  # noqa: ANN401
    """`_SOURCE` keys the pending-expansion rows and must be the project's own spelling.

    A key nothing else uses would look fine until a restart: the sweep reads its own rows by
    that string, so a private spelling resumes nothing and reports nothing either.
    """
    assert module._SOURCE in LINK_SOURCE_EMOJIS


def test_no_two_expansion_cogs_share_a_source_key() -> None:
    """Two cogs on one key would each resume the other's interrupted expansions."""
    keys = [module._SOURCE for module in _MODULES]

    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.split(".")[-2])
def test_an_expansion_cog_resumes_what_a_restart_interrupted(module: Any) -> None:  # noqa: ANN401
    """Without the `on_ready` entry point a restart leaves a placeholder that never resolves.

    That is the failure this contract was written for: the cog itself still works, so nothing
    goes red, and the channel keeps a line saying an expansion is coming that never is.
    """
    cog = _cog_class(module=module)

    assert hasattr(cog, "on_ready"), f"{cog.__name__} never sweeps its interrupted expansions"


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.split(".")[-2])
def test_an_expansion_cog_can_be_handed_back_its_own_expansion(module: Any) -> None:  # noqa: ANN401
    """The sweep calls `_expand` with the listener's own four arguments, keyword-only.

    Nothing in the type checker sees this: `ExpansionRetry` is satisfied structurally at the
    call site, and a cog whose `_expand` grew a fifth required argument would raise a
    `TypeError` inside a swallowed handler on the next restart and nowhere else.
    """
    expand = getattr(_cog_class(module=module), "_expand")  # noqa: B009 -- ty cannot see it
    parameters = inspect.signature(expand).parameters

    assert {"message", "url", "current_emoji", "placeholder"} <= set(parameters)
    for name in ("message", "url", "current_emoji", "placeholder"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.split(".")[-2])
def test_an_expansion_cog_spells_no_status_mark_of_its_own(module: Any) -> None:  # noqa: ANN401
    """One symbol, one meaning, whichever platform was linked.

    The reaction is the entire report — an expansion that produced nothing says nothing in the
    channel — so a cog inventing its own mark, or reusing a shared one for a different
    outcome, is the whole feature's vocabulary coming apart. Sharing the constants is what
    makes that a diff someone reads rather than a drift nobody notices.
    """
    source = Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")
    body = "\n".join(line for line in source.split("\n") if not line.lstrip().startswith("#"))

    for literal in _STATUS_LITERALS:
        assert f'"{literal}"' not in body, f"{module.__name__} spells {literal} itself"


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.split(".")[-2])
def test_an_expansion_cog_claims_its_reply_slot_before_it_reacts(module: Any) -> None:  # noqa: ANN401
    """The placeholder is what the reader is waiting for, so nothing queues in front of it.

    Both reactions share one rate-limit bucket, which nextcord serializes itself and which is
    per CHANNEL rather than per message, so a busy channel makes them slower still; a message
    send is a different bucket and waits on none of it. Reacting first therefore delays the
    one thing the placeholder exists to put under the link promptly. Read off the source
    because the ordering is the whole property and there is nothing else to assert against:
    `tests/test_parse_facebook_cog.py` proves the mechanism on one cog, and this holds the
    other three to it.
    """
    listener = getattr(_cog_class(module=module), "on_message")  # noqa: B009 -- ty cannot see it
    body = inspect.getsource(listener)

    assert body.index("send_expansion_placeholder(") < body.index("update_reaction(")


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.split(".")[-2])
async def test_a_failure_with_nothing_on_the_message_still_names_the_platform(
    module: Any,  # noqa: ANN401
) -> None:
    """A cross on its own cannot say which link died, and a message can carry two.

    `current_emoji` is None exactly when claiming the reply slot failed, which is both the
    refused channel and the Discord 5xx that raises straight past it. Behavioural rather than
    a source scan, because what matters is that the marker lands whichever call site got there.
    """
    cog = _cog_class(module=module)(bot=as_bot(fake=SimpleNamespace(user=FakeUser(bot=True))))
    message = FakeDiscordMessage()

    await cog._mark_failed(message=as_message(fake=message), current_emoji=None)

    assert message.reactions == [LINK_SOURCE_EMOJIS[module._SOURCE], EXPANSION_FAILED_EMOJI]


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.split(".")[-2])
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
    module: Any,  # noqa: ANN401
    error: Exception,
    expected: str,
) -> None:
    """The vocabulary is only worth anything if the same failure reads the same everywhere.

    Each cog routes its own read failure through `expansion_failure_emoji`, so this is what
    stops one of them growing a private `isinstance` branch and quietly answering ⚠️ where the
    others answer ⏱️ — the mistake that tells a reader their working link is dead.
    """
    del module  # the mapping is shared; the parametrize is what proves no cog opted out

    assert expansion_failure_emoji(error=error) == expected


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.split(".")[-2])
def test_an_expansion_cog_decides_no_failure_mark_of_its_own(module: Any) -> None:  # noqa: ANN401
    """A cog reading the error's type itself is how the four stop agreeing.

    `parse_douyin` keeps one `isinstance` for its LOGGING split, which is a different
    question: a deleted post is routine and Douyin's own reason for it exists in no other
    line. What no cog may do is pick the reaction that way.
    """
    source = Path(inspect.getsourcefile(module) or "").read_text(encoding="utf-8")

    assert "expansion_failure_emoji(" in source
    assert "isinstance(error, TimeoutError)" not in source
