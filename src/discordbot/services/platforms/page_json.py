"""Reading a post out of a page that embeds its data as JSON instead of serving a schema.

Some platforms answer a logged-out request with a browser page whose payload sits in
`<script type="application/json">` blocks, in a shape dominated by their own module loader
rather than by the post. There is nothing to mirror, so the parser walks for a node carrying the
fields it needs and models only what it extracted. The helpers here are that walk and the
narrowing that follows it, plus the page fetch itself, which a platform that reads a page for a
schema it does mirror shares too.
"""

import re
import json
from typing import Any, Final
from datetime import UTC, datetime
from urllib.parse import urlparse
from collections.abc import Iterator

import logfire
from pydantic import Field, BaseModel
import requests

from discordbot.utils.link_errors import link_fetch_error

# The full set a browser sends. Anything less is served a different page: a refusal, or a shell
# carrying Open Graph tags and none of the post's own payload. Each platform's module docstring
# says which of those it answers with, and when that was last checked — the answer is theirs to
# change, so a date is what says whether it is still true.
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


class FetchedPage(BaseModel):
    """One page as it came back: its HTML, and the URL the request actually ended on.

    Where it landed is part of the result because a share link can name its post only there,
    and because a redirect to a login page is how a platform says the post is not public.
    """

    html: str = Field(..., description="The fetched page's HTML body")
    final_url: str = Field(
        ..., description="The URL the request ended on, after every redirect it followed"
    )

    def landed_on(self, *, path_prefixes: tuple[str, ...]) -> bool:
        """Whether the request ended on a path starting with one of `path_prefixes`.

        Args:
            path_prefixes: Lower-case path prefixes, such as a platform's login pages.

        Returns:
            True when the final URL's lower-cased path starts with any of them.
        """
        return urlparse(self.final_url).path.lower().startswith(path_prefixes)


def fetch_page(*, url: str, headers: dict[str, str], timeout: float) -> FetchedPage:
    """Fetches one page, following redirects, and classifies a failed request.

    Args:
        url: The page to fetch.
        headers: The request headers the platform answers in full to.
        timeout: Per-request timeout in seconds.

    Returns:
        The page and where the request ended.

    Raises:
        LinkRetryableError: The platform refused the request or never answered.
        LinkUnavailableError: The platform answered that there is no such page.
        RuntimeError: The fetch failed in a way HTTP does not classify.
    """
    try:
        response = requests.get(url=url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return FetchedPage(html=response.text, final_url=response.url)
    except requests.RequestException as error:
        raise link_fetch_error(error=error, url=url) from error


def json_payloads(*, html: str, platform: str) -> Iterator[Any]:
    """Yields every embedded JSON block on the page, skipping the ones that do not parse.

    Skipping rather than failing is what keeps one truncated block, of the dozens a page
    carries, from costing the post.

    Args:
        html: The page.
        platform: The platform's display name, for the log line a skipped block leaves.

    Yields:
        Each block's parsed JSON, in page order.
    """
    for match in JSON_SCRIPT_RE.finditer(string=html):
        try:
            yield json.loads(s=match.group(1))
        except ValueError:
            # `json.JSONDecodeError` subclasses this, as does the int-string conversion limit a
            # very large embedded number can trip.
            logfire.debug(f"Skipped an unparsable {platform} JSON block", _exc_info=True)
            continue


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


def count_of(*, value: JsonValue) -> int:
    """Reads a count the page serves as a bare value, a `{"count": n}` wrapper, or "1,017".

    An int rather than the page's own formatted string, so every platform's counters are the
    same type and whoever renders them picks the formatting once. Zero for anything else.
    """
    if isinstance(value, dict):
        value = value.get("count")
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = value.replace(",", "").strip()
        return int(digits) if digits.isdigit() else 0
    return 0


def time_of(*, value: JsonValue) -> datetime | None:
    """A unix timestamp as an aware datetime, or None when the page omitted it."""
    return datetime.fromtimestamp(value, tz=UTC) if isinstance(value, int) and value else None
