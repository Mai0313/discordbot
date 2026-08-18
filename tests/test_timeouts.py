"""Pins the relationships between collected bounds that no single call site can see.

`typings/timeouts.py` exists so a bound can be picked with a view of the ones it interacts
with. Two of those interactions are load-bearing rather than cosmetic, and both used to live
only as prose in two files that could not see each other's number.
"""

import ast
from pathlib import Path

import pytest

from discordbot.typings.timeouts import (
    DOWNLOAD_TIMEOUT_SECONDS,
    LINK_CONTEXT_GRACE_SECONDS,
    LINK_MEDIA_TIMEOUT_SECONDS,
    THREAD_TITLE_TIMEOUT_SECONDS,
    STOCK_NEWS_AI_TIMEOUT_SECONDS,
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


def test_every_direct_to_google_render_carries_its_own_deadline() -> None:
    """`genai.Client` is the one surface where "the provider owns it" needs a number from us.

    google-genai leaves `http_options.timeout` at None, so an unbounded `interactions.create` into
    a black-holed connection never returns and no `except` is ever reached — a hang rather than an
    error, which for the VIDEO route means no clip and a status reaction stuck forever. The bound
    rides as the SDK's own per-request `timeout=` rather than an `asyncio.timeout` around it, so
    this reads the call sites rather than the module.
    """
    source = (
        Path(__file__).resolve().parents[1] / "src/discordbot/cogs/gen_reply/generation.py"
    ).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source=source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "interactions"
    ]
    assert calls, "no interactions.create call found; this guard is reading the wrong module"
    unbounded = [
        call.lineno for call in calls if not any(kw.arg == "timeout" for kw in call.keywords)
    ]
    assert unbounded == []


def test_the_two_product_deadlines_stayed_short() -> None:
    """Neither bounds a transport; each bounds how long its feature may block something else.

    The stock one sits under the process-wide news generation lock that the stock views also take,
    and the research one runs before the thread exists so the caller sees nothing until it returns.
    Both would be pointless at the provider's own 600s, so the assertion is on the order of
    magnitude rather than the exact value.
    """
    assert STOCK_NEWS_AI_TIMEOUT_SECONDS < 10.0
    assert THREAD_TITLE_TIMEOUT_SECONDS < 60.0
