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

from typing import Any
import inspect
from pathlib import Path
import importlib

import pytest
from nextcord.ext import commands

from discordbot.typings.emojis import LINK_SOURCE_EMOJIS
from discordbot.utils.expansion_placeholder import (
    EXPANSION_DONE_EMOJI,
    EXPANSION_FAILED_EMOJI,
    EXPANSION_WORKING_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
)

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
