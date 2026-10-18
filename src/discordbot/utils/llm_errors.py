"""Reads what a caller needs off an SDK-wrapped provider error.

Two questions are asked of the same object and both are answered by digging past the SDK
wrapper: the message a user should see, and whether the failure is worth another attempt.
"""

import re
import ast
import json

from openai import APIError, APIConnectionError
from google.genai.errors import APIError as GenAIAPIError

# `interactions.create` raises its own hierarchy, unrelated to `google.genai.errors`, and the
# SDK exports it from no public module; an upgrade that moves it fails this import loudly
# instead of leaving every Interactions failure unretried.
from google.genai._gaos.lib.compat_errors import APIError as InteractionsAPIError
from google.genai._gaos.lib.compat_errors import APIConnectionError as InteractionsConnectionError

# LiteLLM surfaces upstream provider errors as a chain like
# `litellm.X: litellm.Y: VertexException - b'{"error": {"message": "..."}}'`,
# where the provider's actual JSON body is embedded as a Python bytes literal.
_BYTES_LITERAL_RE = re.compile(pattern=r"b'((?:[^'\\]|\\.)*)'", flags=re.DOTALL)

# The same chain, read as the run of prefixes it is: every LiteLLM exception class prepends
# `litellm.<ClassName>: ` (`exceptions.py`), the provider mapping prepends
# `<Provider>Exception - ` and on some branches a second label with it
# (`exception_mapping_utils.py`, `Timeout Error: ` and `...Exception BadRequestError - `), and a
# failure that arrives once the stream is open prepends the provider's canonical status
# (`_check_streaming_error`, `UNAVAILABLE - `). None of those names is written down here: a
# segment is recognised by its shape alone -- a code-shaped word ending in `Error`, `Exception`
# or `Timeout`, or an ALL-CAPS status -- so the wording upstream can move without this going
# stale, and a provider's own sentence, which is prose, matches none of it. `Timeout` is there
# because the run is anchored: it is the one exception class in `exceptions.py` named after
# neither of the other two, so leaving it out costs not that segment but every segment behind
# it, on the paths that already made the user wait longest.
_WRAPPER_PREFIX_RE = re.compile(
    pattern=r"^(?:(?:[A-Z][\w.]* )?[\w.]*(?:Error|Exception|Timeout)(?:: | - )|[A-Z][A-Z0-9_]+ - )+"
)

# Both SDKs keep the decoded error body on the exception, so it never has to be read back out
# of the repr their `__str__` builds: `openai` as `.body` (already unwrapped to the `error`
# object), `google.genai` as `.details` (the whole document).
_DECODED_BODY_ATTRIBUTES = ("body", "details")

# Statuses below 500 that are still worth another attempt: the set `openai`'s own client
# retries (`_base_client.py::_should_retry`) and the one the provider rate-limit guides
# prescribe. Every other 4xx is the provider refusing the request itself, so re-sending it
# only spends the user's wait twice.
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
