"""Shared HTTP page fetching and embedded JSON-script traversal for browser-scraped platforms."""

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

# Full browser header set required by Facebook/Instagram; bare User-Agent gets an OG shell or 400.
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
    pattern=r'<script type="application/json"[^>]*>(.*?)</script>', flags=re.DOTALL
)

JsonValue = dict[str, Any] | list[Any] | str | float | None


class FetchedPage(BaseModel):
    """Fetched HTML body and the final URL after redirects."""

    html: str = Field(..., description="The fetched page's HTML body")
    final_url: str = Field(
        ..., description="The URL the request ended on, after every redirect it followed"
    )

    def landed_on(self, path_prefixes: tuple[str, ...]) -> bool:
        """Returns whether `final_url`'s lower-cased path starts with one of `path_prefixes`.

        Args:
            path_prefixes (tuple[str, ...]): Lower-case path prefixes, such as a platform's login pages.
        """
        return urlparse(self.final_url).path.lower().startswith(path_prefixes)


def fetch_page(url: str, headers: dict[str, str], timeout: float) -> FetchedPage:
    """Fetches one page, following redirects, and classifies a failed request.

    Args:
        url (str): The page to fetch.
        headers (dict[str, str]): The request headers the platform answers in full to.
        timeout (float): Per-request timeout in seconds.

    Returns:
        The page and where the request ended.

    Raises:
        LinkRetryableError: The platform refused the request, never answered, or lost the
            connection part-way through the body.
        LinkUnavailableError: The platform answered that there is no such page.
        RuntimeError: The fetch failed in a way HTTP does not classify.
    """
    try:
        response = requests.get(url=url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return FetchedPage(html=response.text, final_url=response.url)
    except requests.RequestException as error:
        raise link_fetch_error(error=error, url=url) from error


def json_payloads(html: str, platform: str) -> Iterator[Any]:
    """Yields each embedded JSON block's parsed value in page order, skipping unparsable blocks.

    Args:
        html (str): The page HTML.
        platform (str): The platform's display name, for the debug log when a block is skipped.
    """
    for match in JSON_SCRIPT_RE.finditer(string=html):
        try:
            yield json.loads(s=match.group(1))
        except ValueError:
            # Covers both `json.JSONDecodeError` and the int-string digit conversion limit.
            logfire.debug(f"Skipped an unparsable {platform} JSON block", _exc_info=True)
            continue


def walk(node: JsonValue) -> Iterator[dict[str, Any]]:
    """Yields every dict in `node`, outermost first.

    Args:
        node (JsonValue): Parsed JSON value to traverse.
    """
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(node=value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(node=value)


def deep_get(node: JsonValue, *keys: str) -> JsonValue:
    """Returns the nested value reached by `keys`, or `None` as soon as a key is missing.

    Args:
        node (JsonValue): Root JSON value.
        *keys (str): Sequence of dict keys to follow.
    """
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def str_of(value: JsonValue) -> str:
    """Returns `value` when it is a `str`, or an empty string otherwise.

    Args:
        value (JsonValue): Raw JSON field value.
    """
    return value if isinstance(value, str) else ""


def count_of(value: JsonValue) -> int:
    """Returns an integer count from a bare int, a `{"count": n}` wrapper, or a formatted string.

    Returns `0` for booleans and any other shape.

    Args:
        value (JsonValue): Raw counter value from the page.
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


def time_of(value: JsonValue) -> datetime | None:
    """Returns `value` as a UTC `datetime` when it is a positive Unix timestamp, or `None` otherwise.

    Args:
        value (JsonValue): Raw timestamp value from the page.
    """
    return datetime.fromtimestamp(value, tz=UTC) if isinstance(value, int) and value else None
