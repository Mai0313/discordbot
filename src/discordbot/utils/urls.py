"""Pulling a usable URL out of text a human pasted.

Share buttons rarely hand over a bare link. Douyin's, for one, wraps it in a blob of copy:
`7.64 gOX:/ ... https://v.douyin.com/iR2syBRn/ 复制此链接，打开Dou音搜索`. Pasting that whole
thing into a command that expects a URL is the natural thing to do, so the command should
find the link rather than fail on it.
"""

import re
from collections.abc import Sequence

# Where a URL is allowed to start. `\b` is the natural spelling of "not glued to the end of
# another token", but Python's `\w` counts every Unicode letter, so it also refuses the `h`
# after a Chinese character: `看這篇https://example.com` read as no URL at all, which is how a
# lot of people type (#492). Refusing only an ASCII word character keeps `xhttps://...` out and
# lets the Chinese in, widening the pattern by one class of characters rather than by a class of
# strings. Shared with `gen_reply/cog.py::_MESSAGE_URL_RE` so the two generic scanners cannot
# drift on where a URL begins.
URL_START_ANCHOR = r"(?<![A-Za-z0-9_])"

# Generic fallback. `[^\s<>]` stops at whitespace and at the angle brackets Discord and
# Markdown wrap links in; it deliberately does NOT stop at CJK, so Chinese written straight
# after a link is carried into the match. Where a link ENDS is what a site-specific pattern is
# the right tool for; where it starts is `URL_START_ANCHOR`'s job just above.
URL_RE = re.compile(rf"(?i){URL_START_ANCHOR}https?://[^\s<>]+")

# Sentence punctuation a URL never really ends on, stripped from the tail of a generic match
# so `see https://example.com/x.` does not carry the full stop into the URL.
_TRAILING_PUNCTUATION = ".,;:!?)]}'\"、。，！？）」』"  # noqa: RUF001 -- CJK sentence punctuation is the point


def extract_first_url(*, text: str, patterns: Sequence[re.Pattern[str]] = ()) -> str:
    """Returns the first URL in `text`, or the stripped text when there is none.

    `patterns` are tried in order before the generic one. A site-specific pattern knows where
    its own links end (Douyin's survives being butted against Chinese text, which the generic
    whitespace rule cannot), so it wins where it matches.

    Falling back to the raw text rather than an empty string keeps every existing caller
    working: a bare URL, or something this cannot parse, is passed through untouched and fails
    downstream exactly as it did before.

    Args:
        text: The text a user supplied, which may be a bare URL or a blob containing one.
        patterns: Site-specific URL patterns to prefer over the generic match.

    Returns:
        The extracted URL, or the input stripped of surrounding whitespace.
    """
    for pattern in patterns:
        match = pattern.search(string=text)
        if match:
            return match.group(0)
    match = URL_RE.search(string=text)
    if match:
        return match.group(0).rstrip(_TRAILING_PUNCTUATION)
    return text.strip()
