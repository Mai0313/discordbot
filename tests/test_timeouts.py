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
    PROMPT_REFINE_TIMEOUT_SECONDS,
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


# The one direct-to-Google path deliberately left unbounded. It can afford the hang the others
# cannot: `background=True` + `store=True` leave the agent settling server-side and `on_ready`
# re-attaches after a restart, so the cost is a resumable thread rather than a lost reply. Named
# here rather than assumed, so adding a render anywhere else fails until someone answers for it.
_UNBOUNDED_BY_DESIGN = {"src/discordbot/cogs/research/agent.py"}


def test_every_direct_to_google_render_carries_its_own_deadline() -> None:
    """`genai.Client` is the one surface where "the provider owns it" needs a number from us.

    google-genai leaves `http_options.timeout` at None and builds its httpx client with
    `timeout=None`, so an unbounded `interactions.create` into a black-holed connection never
    returns and no `except` is ever reached — a hang rather than an error, which for the VIDEO
    route means no clip and a status reaction stuck forever, and for the YouTube answer turn means
    the whole reply. The bound rides as the SDK's own per-request `timeout=` rather than an
    `asyncio.timeout` around it, so this reads the call sites rather than the module.

    It sweeps the whole package rather than the one file that holds two of them: a guard pointed
    at a single module reads as covering the rule while a render added next door goes unbounded,
    which is exactly what it did before #531's review.
    """
    root = Path(__file__).resolve().parents[1]
    checked: list[str] = []
    unbounded: list[str] = []
    for path in sorted((root / "src/discordbot").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        for node in ast.walk(ast.parse(source=path.read_text(encoding="utf-8"))):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr != "create"
                or not isinstance(node.func.value, ast.Attribute)
                or node.func.value.attr != "interactions"
            ):
                continue
            checked.append(f"{relative}:{node.lineno}")
            if relative in _UNBOUNDED_BY_DESIGN:
                continue
            if not any(keyword.arg == "timeout" for keyword in node.keywords):
                unbounded.append(f"{relative}:{node.lineno}")
    assert checked, "no interactions.create call found; this guard is reading the wrong tree"
    assert unbounded == []


def test_the_product_deadlines_stayed_short() -> None:
    """Neither bounds a transport; each bounds how long its feature may block something else.

    The stock one sits under the process-wide news generation lock that the stock views also take,
    the research one runs before the thread exists so the caller sees nothing until it returns, and
    the refine one sits serially ahead of the IMAGE/VIDEO render. All three would be pointless at
    what the provider actually allows (600s read, retried twice, so 1800s), so the assertion is on
    the order of magnitude rather than the exact value.
    """
    assert STOCK_NEWS_AI_TIMEOUT_SECONDS < 10.0
    assert THREAD_TITLE_TIMEOUT_SECONDS < 60.0
    assert PROMPT_REFINE_TIMEOUT_SECONDS < 300.0
