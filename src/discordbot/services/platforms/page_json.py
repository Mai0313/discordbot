"""Reading a post out of a page that embeds its data as JSON instead of serving a schema.

Some platforms answer a logged-out request with a browser page whose payload sits in
`<script type="application/json">` blocks, in a shape dominated by their own module loader
rather than by the post. There is nothing to mirror, so the parser walks for a node carrying the
fields it needs and models only what it extracted. The helpers here are that walk and the
narrowing that follows it.

A platform that mirrors a real schema does not use any of this.
"""

import re
from typing import Any, Final
from datetime import UTC, datetime
from collections.abc import Iterator

# The full set a browser sends. Anything less is served a different page: a refusal, or a shell
# carrying Open Graph tags and none of the post's own payload. Which of those, and how much is
# lost, is the platform's own business and is recorded at its fetch.
BROWSER_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

JSON_SCRIPT_RE: Final[re.Pattern[str]] = re.compile(
    r'<script type="application/json"[^>]*>(.*?)</script>', re.DOTALL
)

# What a parsed JSON payload can hold. Spelled out rather than left as a bare `Any`, which the
# project's checker refuses.
JsonValue = dict[str, Any] | list[Any] | str | float | None


def walk(*, node: JsonValue) -> Iterator[dict[str, Any]]:
    """Yields every dict in a parsed JSON tree, outermost first."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(node=value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(node=value)


def deep_get(node: JsonValue, *keys: str) -> JsonValue:
    """Follows a chain of dict keys, returning None as soon as one is missing."""
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def str_of(*, value: JsonValue) -> str:
    """A string field, or an empty one when the page served a null or another type."""
    return value if isinstance(value, str) else ""


def time_of(*, value: JsonValue) -> datetime | None:
    """A unix timestamp as an aware datetime, or None when the page omitted it."""
    return datetime.fromtimestamp(value, tz=UTC) if isinstance(value, int) and value else None
