"""Pulling a usable URL out of text a human pasted.

Share buttons rarely hand over a bare link. Douyin's, for one, wraps it in a blob of copy:
`7.64 gOX:/ ... https://v.douyin.com/iR2syBRn/ 复制此链接，打开Dou音搜索`. Pasting that whole
thing into a command that expects a URL is the natural thing to do, so the command should
find the link rather than fail on it.
"""

import re
from collections.abc import Sequence

# Generic fallback. `[^\s<>]` stops at whitespace and at the angle brackets Discord and
# Markdown wrap links in; it deliberately does NOT stop at CJK, because a site-specific
# pattern is the right tool for text that runs straight into the link with no space.
URL_RE = re.compile(r"(?i)\bhttps?://[^\s<>]+")

# Sentence punctuation a URL never really ends on, stripped from the tail of a generic match
# so `see https://example.com/x.` does not carry the full stop into the URL.
_TRAILING_PUNCTUATION = ".,;:!?)]}'\"、。，！？）」』"


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
