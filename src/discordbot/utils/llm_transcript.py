"""Conventions for rendering a Discord message into an LLM transcript.

The reply pipeline writes these markers and the memory pipeline reads them back, so they
live here rather than in either one: the author-identity prefix both treat as the trusted
authorship signal, the forwarded-content marker, and the usage footer the bot appends to
its own replies.

`sanitize_identity` plus the two `render_*_identity` helpers own the `<name> [id: <N>]` line in
both of its uses: the per-message sender prefix `gen_reply/input.py` renders, and the identity
stamped onto every memory fact and parsed back by `services/memory/facts.py`. Why each of the
other two shapes exists is in the comment above it.

The module is deliberately string-level and Discord-free (it imports only `re`), so the memory
service, which may not import a cog, and the cogs that produce the text share one definition of
the shape. Walking live messages into a transcript stays in `gen_reply/input.py`, and reading one
back stays in `services/memory/`.
"""

import re

# Strips the usage_footer appended by `streaming.ResponseStreamer.stream` from
# bot-authored messages before feeding them back as `role=assistant` history.
# Without this, the model performs in-context learning on its own past footers
# and starts hallucinating fake "-# model · ⬆ ... ⬇ ... · $... · ..." lines into
# fresh replies. Anchored on the `\n\n-# ` separator plus the ⬆/⬇ token-count
# icons, which never appear together in user-authored content. The optional
# trailing `\n-# ...` line matches the second subtext line that credits looked-up
# memory owners, so the whole footer is stripped as one unit.
USAGE_FOOTER_RE = re.compile(r"\n\n-#[^\n]*⬆[^\n]*⬇[^\n]*(?:\n-#[^\n]*)?$")

# A display name (or legacy username) containing an `[id: ...]`-shaped string
# could forge the sender-identity prefix the input builder prepends, which the
# reply persona prompt and the memory extraction prompt both treat as the trusted
# authorship signal. Neutralize the lookalike before rendering.
_ID_PREFIX_LOOKALIKE_RE = re.compile(r"\[\s*id\s*:", flags=re.IGNORECASE)

# Marker prefixing each forwarded snapshot span appended to a rendered message body. The
# answer model uses it to attribute forwarded content; the memory transcript strips from it
# to end-of-body so a forward of someone else's words is never recorded as the forwarder's
# own fact (forwarded text is always appended last, so the marker is the suffix boundary).
FORWARDED_MESSAGE_MARKER = "[forwarded message]"


def sanitize_identity(value: str) -> str:
    """Neutralizes authorship-prefix lookalikes in a user-controlled identity field.

    Only the `[id:` token is rewritten; the rest of the value, whitespace included, is left as
    the user wrote it, so callers that need one line collapse it themselves.

    Args:
        value (str): A display name, username or guild name about to be rendered into a
            transcript.

    Returns:
        The value with every `[id:` lookalike defused to `[id-`.
    """
    return _ID_PREFIX_LOOKALIKE_RE.sub("[id-", value)


def render_author_identity(display_name: str, username: str, user_id: int) -> str:
    """Renders the single-line author identity stamped into memory files.

    Whitespace runs (including any newline that slips past Discord's name rules) collapse to
    single spaces so the identity can never break the one-line header formats the memory store
    relies on: the line is threaded through the `memory_job` row and matched whole by
    `facts.parse_identity`, whose pattern cannot see past a newline and would fall back to the
    scope's own id while keeping the entire unparsed line, `[id: <N>]` token included, as the
    stored name.

    Args:
        display_name (str): The author's Discord display name.
        username (str): The author's account name.
        user_id (int): The author's Discord id, the trusted half of the line.

    Returns:
        `<display> (<username>) [id: <user_id>]`, sanitized and on one line.
    """
    safe_display = " ".join(sanitize_identity(value=display_name).split())
    safe_username = " ".join(sanitize_identity(value=username).split())
    return f"{safe_display} ({safe_username}) [id: {user_id}]"


def render_server_identity(server_name: str, server_id: int) -> str:
    """Renders the single-line server identity stamped into per-server memory files.

    Mirrors `render_author_identity`: the guild name is user-controlled, so it is sanitized
    against `[id:` lookalikes and collapsed to one line before the `[id: <server_id>]` suffix
    the memory store's identity regex expects.

    Args:
        server_name (str): The guild's name.
        server_id (int): The guild id, the trusted half of the line.

    Returns:
        `<server name> [id: <server_id>]`, sanitized and on one line.
    """
    safe_name = " ".join(sanitize_identity(value=server_name).split())
    return f"{safe_name} [id: {server_id}]"
