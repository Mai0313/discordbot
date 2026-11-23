"""Provider error message extraction and retryability classification for LLM SDK exceptions."""

import re
import ast
import json

from openai import APIError, APITimeoutError, APIConnectionError
from google.genai.errors import APIError as GenAIAPIError

# `interactions.create` raises its own hierarchy (not a subclass of `google.genai.errors`), with
# no public export; importing directly fails loudly if an SDK upgrade moves it.
from google.genai._gaos.lib.compat_errors import APIError as InteractionsAPIError
from google.genai._gaos.lib.compat_errors import APITimeoutError as InteractionsTimeoutError
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

# Statuses, error codes, and non-timeout connection errors whose raw provider message (auth,
# billing, quota, capacity, backend 5xx) is masked behind a generic high-demand notice for
# Discord users; callers log the full exception before calling `extract_friendly_error`.
HIGH_DEMAND_ERROR_NOTICE = (
    "This model is currently experiencing high demand. Spikes in demand are usually "
    "temporary. Please try again later."
)
_MASKED_CLIENT_STATUSES = frozenset({401, 402, 403, 404, 429})
_MASKED_OVERLOAD_CODES = frozenset({
    "authentication_error",
    "permission_error",
    "not_found_error",
    "rate_limit_exceeded",
    "insufficient_quota",
    "server_error",
})


def _is_masked_status(status: int | None) -> bool:
    """Returns whether `status` is an operator/backend HTTP status masked behind `HIGH_DEMAND_ERROR_NOTICE`."""
    if status is None:
        return False
    return status in _MASKED_CLIENT_STATUSES or 500 <= status <= 599


def _parse_int_status(value: object) -> int | None:
    """Returns `value` as an integer status when it is a non-bool int or ASCII decimal string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


def _payload_status_code(payload: object) -> int | None:
    """Returns the integer status code from `payload['code']` or `payload['error']['code']`."""
    if not isinstance(payload, dict):
        return None
    if (status := _parse_int_status(value=payload.get("code"))) is not None:
        return status
    error = payload.get("error")
    if isinstance(error, dict):
        return _parse_int_status(value=error.get("code"))
    return None


def _is_masked_payload(payload: object) -> bool:
    """Returns whether a decoded error payload carries a masked status or error code."""
    if _is_masked_status(status=_payload_status_code(payload=payload)):
        return True
    if not isinstance(payload, dict):
        return False
    top_code = payload.get("code")
    if isinstance(top_code, str) and top_code in _MASKED_OVERLOAD_CODES:
        return True
    error = payload.get("error")
    if isinstance(error, dict):
        inner_code = error.get("code")
        return isinstance(inner_code, str) and inner_code in _MASKED_OVERLOAD_CODES
    return False


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


def _extract_bytes_literal_error(raw: str) -> str | None:
    """Returns a masked notice or provider message from an embedded `b'...'` JSON literal."""
    for match in _BYTES_LITERAL_RE.finditer(string=raw):
        try:
            decoded = ast.literal_eval(node_or_string=match.group(0)).decode(
                encoding="utf-8", errors="replace"
            )
            data = json.loads(s=decoded)
        except (SyntaxError, ValueError, TypeError, AttributeError):
            continue
        if _is_masked_payload(payload=data):
            return HIGH_DEMAND_ERROR_NOTICE
        if (message := _provider_message(payload=data)) is not None:
            return message
    return None


def _is_masked_connection_error(exc: BaseException) -> bool:
    """Returns whether `exc` is a non-timeout LLM SDK connection failure."""
    return isinstance(exc, (APIConnectionError, InteractionsConnectionError)) and not isinstance(
        exc, (APITimeoutError, InteractionsTimeoutError)
    )


def extract_friendly_error(exc: BaseException) -> str:
    """Returns the user-facing provider error message for `exc`, masking operator/backend errors.

    Masks `401`/`402`/`403`/`404`/`429`/`5xx` statuses, matching error codes, and non-timeout
    connection errors behind `HIGH_DEMAND_ERROR_NOTICE` so auth, billing, or backend details never
    reach Discord. Otherwise reads the decoded body off the exception, peels any LiteLLM wrapper
    prefix chain, and parses any embedded `b'...'` JSON bytes literal, falling back to `str(exc)`.

    Args:
        exc (BaseException): Exception carrying a decoded error body, wrapper chain, or embedded
            provider JSON.
    """
    if _is_masked_status(status=llm_status_code(exc=exc)) or _is_masked_connection_error(exc=exc):
        return HIGH_DEMAND_ERROR_NOTICE
    outer_code = getattr(exc, "code", None)
    if isinstance(outer_code, str) and outer_code in _MASKED_OVERLOAD_CODES:
        return HIGH_DEMAND_ERROR_NOTICE
    raw = str(exc)
    for attribute in _DECODED_BODY_ATTRIBUTES:
        payload = getattr(exc, attribute, None)
        if _is_masked_payload(payload=payload):
            return HIGH_DEMAND_ERROR_NOTICE
        if (message := _provider_message(payload=payload)) is not None:
            raw = message
            break
    if (prefix := _WRAPPER_PREFIX_RE.match(string=raw)) and (
        peeled := raw[prefix.end() :].strip()
    ):
        raw = peeled
    if (embedded := _extract_bytes_literal_error(raw=raw)) is not None:
        return embedded
    return raw


def llm_status_code(exc: BaseException) -> int | None:
    """Returns the HTTP status code carried by `exc`, or None when it carries none.

    Reads `status_code` (`openai` and `google-genai` Interactions), then an integer `code`
    (`google.genai.errors`), and finally a numeric or ASCII digit-string `code` inside the
    decoded `body` dict (top-level or nested `error.code`, e.g. LiteLLM mid-stream SSE errors).

    Args:
        exc (BaseException): Exception raised by a failed LLM call.
    """
    if (status := _parse_int_status(value=getattr(exc, "status_code", None))) is not None:
        return status
    if isinstance(code := getattr(exc, "code", None), int) and not isinstance(code, bool):
        return code
    return _payload_status_code(payload=getattr(exc, "body", None))


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
