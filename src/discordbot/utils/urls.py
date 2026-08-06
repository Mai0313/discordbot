"""Pulling a usable URL out of text a human pasted.

Share buttons rarely hand over a bare link. Douyin's, for one, wraps it in a blob of copy:
`7.64 gOX:/ ... https://v.douyin.com/iR2syBRn/ 复制此链接，打开Dou音搜索`. Pasting that whole
thing into a command that expects a URL is the natural thing to do, so the command should
find the link rather than fail on it.

What it promises is narrow, and it is not "the first URL in the text": the first match of a site
pattern the caller handed in, else the first generic `https?://` match, else the text stripped of
surrounding whitespace. The order is by pattern rather than by position, so a site pattern wins
over a generic URL sitting earlier in the text, and a link no pattern here can see (a scheme-less
`v.douyin.com/xxx`, which `is_douyin_url` itself accepts) takes the stripped-text branch. The one
guarantee is that the result is never empty, so a caller that used to be handed the raw text
still is. What it deliberately does not do is judge the result — no validation, no normalisation,
no resolving of a short link — because the caller already routes on what it gets back
(`/download_video` tests the returned URL with `is_douyin_url` and picks a downloader from it).

It carries no site knowledge either: a site's own pattern is handed in by the caller and lives
with that site's other code (`DOUYIN_URL_RE` in `utils/douyin.py`). That is what keeps this a
generic text helper below the cogs, rather than a second home for per-site regexes.
"""

import re
from collections.abc import Sequence

# Generic fallback. `[^\s<>]` stops at whitespace and at the angle brackets Discord and
# Markdown wrap links in; it deliberately does NOT stop at CJK, because a site-specific
# pattern is the right tool for text that runs straight into the link with no space.
URL_RE = re.compile(r"(?i)\bhttps?://[^\s<>]+")

# Sentence punctuation a URL never really ends on, stripped from the tail of a generic match
# so `see https://example.com/x.` does not carry the full stop into the URL.
_TRAILING_PUNCTUATION = ".,;:!?)]}'\"、。，！？）」』"  # noqa: RUF001 -- CJK sentence punctuation is the point


def extract_first_url(*, text: str, patterns: Sequence[re.Pattern[str]] = ()) -> str:
    """Returns the first URL any pattern here can see, or `text` stripped when none can.

    `patterns` are tried in order before the generic one. A site-specific pattern knows where
    its own links end (Douyin's survives being butted against Chinese text, which the generic
    whitespace rule cannot), so it wins where it matches and its match is returned verbatim;
    only the generic match is trimmed of trailing sentence punctuation. That precedence runs
    over the whole text rather than over one position: a site pattern matching at the very end
    beats a generic URL sitting at the start, so with `patterns=(DOUYIN_URL_RE,)` a blob naming
    a YouTube link first still yields the Douyin one.

    Falling back to the raw text rather than an empty string keeps every existing caller
    working: something this cannot parse is handed on and fails downstream exactly as it did
    before. That is the whole of the guarantee, and it is only that the result is never empty.
    Extraction is not otherwise free: the trailing trim is a guess, so a generic match whose own
    last character is one of `_TRAILING_PUNCTUATION` loses it, and a bare Wikipedia
    `..._(programming_language)` link that would have worked comes back a bracket short.

    Args:
        text (str): The text a user supplied, which may be a bare URL or a blob containing one.
        patterns (Sequence[re.Pattern[str]]): Site-specific URL patterns to prefer over the
            generic match.

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
