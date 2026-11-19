"""URL extraction and host-matching helpers for user-pasted text."""

import re
from urllib.parse import urlparse
from collections.abc import Sequence

# ASCII word-boundary lookbehind: Python's `\b` rejects URLs right after CJK text
# (`看這篇https://...`), while still excluding ASCII prefixes like `xhttps://...`.
URL_START_ANCHOR = r"(?<![A-Za-z0-9_])"

# Generic URL fallback; stops at whitespace and `<>` (does not stop at trailing CJK).
URL_RE = re.compile(rf"(?i){URL_START_ANCHOR}https?://[^\s<>]+")

# Sentence punctuation stripped from generic URL match tails.
_TRAILING_PUNCTUATION = ".,;:!?)]}'\"、。，！？）」』"  # noqa: RUF001 -- CJK sentence punctuation is the point


def normalized_url(url: str) -> str:
    """Returns `url` prefixed with `//` when scheme-less so `urlparse` extracts its hostname.

    Args:
        url (str): User-pasted URL string, with or without a scheme.
    """
    return url if "://" in url else f"//{url}"


def normalized_host(url: str) -> str:
    """Returns `url`'s lower-cased hostname, or an empty string if missing or malformed.

    Accepts scheme-less URLs via `normalized_url` and never raises.

    Args:
        url (str): User-pasted URL string.
    """
    try:
        return (urlparse(normalized_url(url=url)).hostname or "").lower()
    except ValueError:
        # urlparse raises on an unbalanced bracket ("https://[abc/x").
        return ""


def host_matches_domain(host: str, domain: str) -> bool:
    """Returns whether `host` is `domain` itself or a subdomain of it.

    Whole labels only; do not loosen.

    Args:
        host (str): A hostname, already lower-cased (see `normalized_host`).
        domain (str): The lower-cased domain to match against.
    """
    return host == domain or host.endswith(f".{domain}")


def extract_first_url(text: str, patterns: Sequence[re.Pattern[str]] = ()) -> str:
    """Returns the first URL in `text`, or `text.strip()` when no URL matches.

    Tries `patterns` in order before falling back to `URL_RE`.

    Args:
        text (str): User-supplied text containing a URL or raw input.
        patterns (Sequence[re.Pattern[str]]): Site-specific URL patterns preferred over `URL_RE`.
    """
    for pattern in patterns:
        match = pattern.search(string=text)
        if match:
            return match.group(0)
    match = URL_RE.search(string=text)
    if match:
        return match.group(0).rstrip(_TRAILING_PUNCTUATION)
    return text.strip()
