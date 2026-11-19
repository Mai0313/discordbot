"""Provider error message extraction and retryability classification for LLM SDK exceptions."""

import re
import ast
import json

from openai import APIError, APIConnectionError
from google.genai.errors import APIError as GenAIAPIError

# `interactions.create` raises its own hierarchy (not a subclass of `google.genai.errors`), with
# no public export; importing directly fails loudly if an SDK upgrade moves it.
from google.genai._gaos.lib.compat_errors import APIError as InteractionsAPIError
from google.genai._gaos.lib.compat_errors import APIConnectionError as InteractionsConnectionError

# Matches provider JSON embedded as a Python bytes literal in LiteLLM error strings.
_BYTES_LITERAL_RE = re.compile(pattern=r"b'((?:[^'\\]|\\.)*)'", flags=re.DOTALL)

# Strips anchored LiteLLM/provider wrapper prefixes (`litellm.X: `, `<Provider>Exception - `,
# `Timeout Error: `, `UNAVAILABLE - `) while leaving prose error messages intact.
_WRAPPER_PREFIX_RE = re.compile(
    pattern=r"^(?:(?:[A-Z][\w.]* )?[\w.]*(?:Error|Exception|Timeout)(?:: | - )|[A-Z][A-Z0-9_]+ - )+"
)

# Decoded error payloads on SDK exceptions: openai uses `.body`, google.genai uses `.details`.
_DECODED_BODY_ATTRIBUTES = ("body", "details")

# Retryable 4xx statuses matching openai._base_client._should_retry.
_RETRYABLE_CLIENT_STATUSES = frozenset({408, 409, 429})


def _provider_message(payload: object) -> str | None:
    """Returns the provider's own message from a decoded JSON error body, or None if absent."""
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        inner = error.get("message")
        if isinstance(inner, str) and inner:
            return inner
    top = payload.get("message")
    if isinstance(top, str) and top:
        return top
    return None


def extract_friendly_error(exc: BaseException) -> str:
    """Returns the innermost provider error message from `exc`, or `str(exc)` if none parses.

    Reads the decoded body off the exception first, peels any LiteLLM wrapper prefix chain, and
    then parses any embedded `b'...'` JSON bytes literal.

    Args:
        exc (BaseException): Exception carrying a decoded error body, wrapper chain, or embedded
            provider JSON.
    """
    raw = str(exc)
    for attribute in _DECODED_BODY_ATTRIBUTES:
        if (message := _provider_message(payload=getattr(exc, attribute, None))) is not None:
            raw = message
            break
    if (prefix := _WRAPPER_PREFIX_RE.match(string=raw)) and (
        peeled := raw[prefix.end() :].strip()
    ):
        raw = peeled
    for match in _BYTES_LITERAL_RE.finditer(string=raw):
        try:
            decoded = ast.literal_eval(node_or_string=match.group(0)).decode(
                encoding="utf-8", errors="replace"
            )
            data = json.loads(s=decoded)
        except (SyntaxError, ValueError, TypeError, AttributeError):
            continue
        if (message := _provider_message(payload=data)) is not None:
            return message
    return raw


def llm_status_code(exc: BaseException) -> int | None:
    """Returns the HTTP status code carried by `exc`, or None when it carries none.

    Reads `status_code` (`openai` and `google-genai` Interactions), then an integer `code`
    (`google.genai.errors`), and finally a numeric or digit-string `code` inside the decoded
    `body` dict (LiteLLM mid-stream SSE errors).

    Args:
        exc (BaseException): Exception raised by a failed LLM call.
    """
    if isinstance(status := getattr(exc, "status_code", None), int):
        return status
    if isinstance(code := getattr(exc, "code", None), int):
        return code
    if isinstance(body := getattr(exc, "body", None), dict):
        body_code = body.get("code")
        if isinstance(body_code, int):
            return body_code
        if isinstance(body_code, str) and body_code.isdigit():
            return int(body_code)
    return None


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Returns whether re-sending the failed LLM request could plausibly succeed.

    True only for an LLM SDK's own `APIConnectionError` (including timeouts) or an LLM SDK
    exception carrying a 5xx or retryable 4xx status (`408`, `409`, `429`). Non-LLM exceptions
    such as `nextcord.HTTPException` (whose `.code` is a Discord JSON error code like `50035`)
    and unclassifiable statuses return False.

    Args:
        exc (BaseException): Exception raised by a failed LLM call.
    """
    if not isinstance(exc, (APIError, GenAIAPIError, InteractionsAPIError)):
        return False
    if isinstance(exc, (APIConnectionError, InteractionsConnectionError)):
        return True
    status = llm_status_code(exc=exc)
    if status is None:
        return False
    return status >= 500 or status in _RETRYABLE_CLIENT_STATUSES
