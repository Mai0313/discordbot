"""Shared transcript rendering, identity sanitization, fence defusing, and usage-footer helpers."""

import re

from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam

from discordbot.utils.model_pricing import get_token_rates

# Strips bot usage footers (plus optional leading ✏️/🩹 memory notes from
# `cogs/gen_reply/status_marks.py` and trailing subtext lines such as owner credit and feedback
# policy) before re-reading replies as assistant history so the model does not hallucinate footers.
# `^` covers note-only replies.
USAGE_FOOTER_RE = re.compile(
    r"(?:(?:^|\n)-# (?:✏️|🩹) [^\n]*)*\n\n-#[^\n]*⬆[^\n]*⬇[^\n]*(?:\n-#[^\n]*)*$"
)

# Neutralizes `[id:` lookalikes in display names that could spoof trusted author prefixes.
_ID_PREFIX_LOOKALIKE_RE = re.compile(r"\[\s*id\s*:", flags=re.IGNORECASE)

# Suffix boundary for appended forwarded snapshots; memory transcripts strip from here to end.
FORWARDED_MESSAGE_MARKER = "[forwarded message]"


def render_usage_footer(
    model_name: str, label: str, input_tokens: int, output_tokens: int
) -> tuple[str, float | None]:
    """Returns the usage footer line and its USD cost, or None when `model_name` has no rate.

    Omits the cost suffix when `model_name` is absent from the price table. `USAGE_FOOTER_RE`
    anchors on the `⬆` / `⬇` icons in the returned line.

    Args:
        model_name (str): Model identifier used to look up token rates.
        label (str): Display model label shown in the footer line.
        input_tokens (int): Input tokens reported by the request.
        output_tokens (int): Output tokens reported by the request.
    """
    line = f"-# {label} · ⬆ {input_tokens:,} ⬇ {output_tokens:,}"
    rates = get_token_rates(model_name=model_name)
    if rates is None:
        return line, None
    input_rate, output_rate = rates
    cost = input_rate * input_tokens + output_rate * output_tokens
    return f"{line} · ${cost:.8f}", cost


def defuse_separator_fences(text: str) -> str:
    """Returns `text` with exact four-character runs of `=` or full-width equals reduced to `==`.

    Leaves longer runs untouched.

    Args:
        text (str): User- or external-sourced text to defuse.
    """
    return re.sub(pattern=r"(?<![=\uff1d])[=\uff1d]{4}(?![=\uff1d])", repl="==", string=text)


def sanitize_identity(value: str) -> str:
    """Returns `value` with authorship-prefix lookalikes (`[id:`) and marker fences neutralized.

    Args:
        value (str): User-controlled identity string to sanitize.
    """
    return defuse_separator_fences(text=_ID_PREFIX_LOOKALIKE_RE.sub("[id-", value))


def render_author_identity(display_name: str, username: str, user_id: int) -> str:
    """Returns the sanitized single-line author identity stamped into memory files.

    Neutralizes `[id:` lookalikes and marker fences, and collapses whitespace to single spaces.

    Args:
        display_name (str): User's display name or server nickname.
        username (str): User's Discord account handle.
        user_id (int): User's Discord numeric id.
    """
    safe_display = " ".join(sanitize_identity(value=display_name).split())
    safe_username = " ".join(sanitize_identity(value=username).split())
    return f"{safe_display} ({safe_username}) [id: {user_id}]"


def render_server_identity(server_name: str, server_id: int) -> str:
    """Returns the single-line server identity stamped into per-server memory files.

    Sanitizes `server_name` against `[id:` lookalikes and collapses whitespace to one line before
    the `[id: <server_id>]` suffix.

    Args:
        server_name (str): Guild display name.
        server_id (int): Guild Discord numeric id.
    """
    safe_name = " ".join(sanitize_identity(value=server_name).split())
    return f"{safe_name} [id: {server_id}]"


def marker_block(text: str) -> EasyInputMessageParam:
    """Returns `text` wrapped as a `role="user"` input block marking a section boundary.

    Uses `role="user"` rather than `role="system"` so markers stay in position; `====` fences
    distinguish them from user text sanitized by `defuse_separator_fences`.

    Args:
        text (str): Separator or notice text to wrap.
    """
    return EasyInputMessageParam(
        role="user", content=[ResponseInputTextParam(text=text, type="input_text")]
    )
