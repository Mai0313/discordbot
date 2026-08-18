"""Pins the relationships between collected bounds that no single call site can see.

`typings/timeouts.py` exists so a bound can be picked with a view of the ones it interacts
with. Two of those interactions are load-bearing rather than cosmetic, and both used to live
only as prose in two files that could not see each other's number.
"""

import inspect

import pytest

from discordbot.typings import timeouts
from discordbot.typings.timeouts import (
    DOWNLOAD_TIMEOUT_SECONDS,
    LINK_CONTEXT_GRACE_SECONDS,
    LINK_MEDIA_TIMEOUT_SECONDS,
    LINK_MEDIA_DEGRADE_HEADROOM_SECONDS,
)


def test_link_media_finishes_inside_the_pipeline_grace() -> None:
    """The media step must degrade to text itself rather than be cancelled with nothing.

    That is the whole reason the media bound exists, and it holds only while it sits below the
    grace the pipeline gives the builder. Derived rather than restated so it cannot drift, so
    this pins the derivation's direction as much as its result: a headroom raised past the
    grace would otherwise silently invert the two.
    """
    assert LINK_MEDIA_TIMEOUT_SECONDS < LINK_CONTEXT_GRACE_SECONDS
    assert LINK_MEDIA_DEGRADE_HEADROOM_SECONDS > 0
    assert (
        LINK_MEDIA_TIMEOUT_SECONDS
        == LINK_CONTEXT_GRACE_SECONDS - LINK_MEDIA_DEGRADE_HEADROOM_SECONDS
    )


@pytest.mark.parametrize(
    "module_path",
    [
        "discordbot.cogs.video.cog",
        "discordbot.cogs.parse_threads.cog",
        "discordbot.cogs.parse_douyin.cog",
    ],
)
def test_every_link_download_path_shares_one_bound(module_path: str) -> None:
    """`/download_video` and both auto-expansions answer to the same number.

    They make the user the same promise — a paste or a command either produces media or reports
    a failure — and three of the four call sites had no bound at all before #529, which is how
    `/download_video` came to strand its caller on the progress message with no exit. Importing
    the shared constant is what keeps a later tweak from reaching one of them and not the rest.
    """
    module = __import__(module_path, fromlist=["cog"])
    assert getattr(module, "DOWNLOAD_TIMEOUT_SECONDS", None) is DOWNLOAD_TIMEOUT_SECONDS


def test_no_llm_call_bound_survived_the_collection() -> None:
    """The module holds no LLM call deadline, and that is a decision rather than an omission.

    Every LLM call in the tree lets the provider own its deadline (`AsyncOpenAI` defaults to
    connect 5s / read 600s and raises `APITimeoutError`, which each call site already degrades
    through on its broad `except`). A constant named for one of those calls reappearing here
    means an `asyncio.timeout` came back with it.
    """
    named = {
        name
        for name in dir(timeouts)
        if not name.startswith("_") and not inspect.ismodule(getattr(timeouts, name))
    }
    llm_shaped = {
        name
        for name in named
        if any(
            token in name
            for token in ("VOICE", "MUSIC", "REFINE", "EXTRACT", "CONSOLIDATE", "COMPARTMENT")
        )
    }
    assert llm_shaped == set()
