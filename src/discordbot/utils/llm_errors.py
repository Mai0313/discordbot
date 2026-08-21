"""Unwraps an SDK-wrapped provider error into the message a user should see."""

import re
import ast
import json

# LiteLLM surfaces upstream provider errors as a chain like
# `litellm.X: litellm.Y: VertexException - b'{"error": {"message": "..."}}'`,
# where the provider's actual JSON body is embedded as a Python bytes literal.
_BYTES_LITERAL_RE = re.compile(pattern=r"b'((?:[^'\\]|\\.)*)'", flags=re.DOTALL)

# Both SDKs keep the decoded error body on the exception, so it never has to be read back out
# of the repr their `__str__` builds: `openai` as `.body` (already unwrapped to the `error`
# object), `google.genai` as `.details` (the whole document).
_DECODED_BODY_ATTRIBUTES = ("body", "details")


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
