"""Reads what a caller needs off an SDK-wrapped provider error.

Two questions are asked of the same object and both are answered by digging past the SDK
wrapper: the message a user should see, and whether the failure is worth another attempt.
"""

import re
import ast
import json

from openai import APIError, APIConnectionError
from google.genai.errors import APIError as GenAIAPIError

# LiteLLM surfaces upstream provider errors as a chain like
# `litellm.X: litellm.Y: VertexException - b'{"error": {"message": "..."}}'`,
# where the provider's actual JSON body is embedded as a Python bytes literal.
_BYTES_LITERAL_RE = re.compile(pattern=r"b'((?:[^'\\]|\\.)*)'", flags=re.DOTALL)

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
    """Read the provider's own message out of one decoded error body.

    Args:
        payload: A decoded JSON error body, either the whole document or the `error`
            object an SDK already unwrapped out of it.

    Returns:
        The provider message, or None when `payload` carries none.
    """
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
    """Surface the innermost provider error message from an SDK-wrapped APIError.

    Two shapes reach here. A refusal the provider answered as a plain 400 keeps its
    body on the exception, but both SDKs render it into their `__str__` as a Python
    dict repr (`Error code: 400 - {'error': ...}`), which is why the decoded body is
    read off the exception instead of parsed back out of that text. A LiteLLM-wrapped
    one nests further: OpenAI's streaming layer constructs
    `APIError(message=error["message"], ...)` from the upstream SSE event, and that
    `message` is the wrapped exception chain with the provider response stuffed inside
    as a `b'...'` Python literal, so every embedded bytes literal is walked and parsed
    as JSON too. Fall back to `str(exc)` when nothing parses, so we never lose the
    original signal.

    Args:
        exc: The exception carrying a decoded error body, or whose string form may
            contain embedded provider JSON.

    Returns:
        The provider message from the decoded body or from an embedded JSON bytes
        literal, or `str(exc)` if no provider message can be extracted.
    """
    raw = str(exc)
    for attribute in _DECODED_BODY_ATTRIBUTES:
        if (message := _provider_message(payload=getattr(exc, attribute, None))) is not None:
            raw = message
            break
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
    """The HTTP status a failed LLM call carried, or None when it carries none.

    Three shapes, read in order. An `openai` failure the SDK typed keeps the status on
    `status_code`. A `google.genai` one keeps it on `code` as an int, which is why that
    attribute is read before the body rather than after: `openai` also has a `code`, but its
    own is the body's semantic string (`rate_limit_exceeded`), so the int test is what tells
    the two apart. And the one this project actually has to classify has neither, because
    LiteLLM reports a mid-stream provider failure as an SSE error frame holding
    `ProxyException.to_dict()` and `openai`'s streaming layer turns that into a bare
    `APIError` whose only trace of the status is `code` inside the decoded body it attaches.
    That one is a decimal STRING, since `ProxyException` stringifies it to match the OpenAI
    error schema (verified against the running proxy, not read off its docs).

    Args:
        exc: The exception a failed LLM call raised.

    Returns:
        The status, or None when the exception carries none of the three.
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
    """Whether re-sending the same request could plausibly succeed.

    True for a transport failure (which `APIConnectionError` covers along with its
    `APITimeoutError` subclass) and for a status the provider guides call transient. An
    unclassifiable failure is NOT retried: a status that cannot be read is as likely to be a
    refusal of the request itself as an outage, and re-sending a refusal only makes the user
    wait for it repeatedly.

    Only an LLM SDK's own exception is even considered, and that gate is load-bearing rather
    than tidiness: `llm_status_code` reads a plain int `code`, and a `nextcord.HTTPException`
    carries one too -- the Discord JSON error code, where 50035 (Invalid Form Body, what an
    oversized final write raises) would read as a 5xx and re-run the whole answer to fail the
    same way. A Discord write failing is not a reason to ask the model again.

    Args:
        exc: The exception a failed LLM call raised.

    Returns:
        Whether the caller should try again.
    """
    if not isinstance(exc, (APIError, GenAIAPIError)):
        return False
    if isinstance(exc, APIConnectionError):
        return True
    status = llm_status_code(exc=exc)
    if status is None:
        return False
    return status >= 500 or status in _RETRYABLE_CLIENT_STATUSES
