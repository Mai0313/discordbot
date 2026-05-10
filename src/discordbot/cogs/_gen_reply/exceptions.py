import re
import ast
import json

# LiteLLM surfaces upstream provider errors as a chain like
# `litellm.X: litellm.Y: VertexException - b'{"error": {"message": "..."}}'`,
# where the provider's actual JSON body is embedded as a Python bytes literal.
_BYTES_LITERAL_RE = re.compile(pattern=r"b'((?:[^'\\]|\\.)*)'", flags=re.DOTALL)


def extract_friendly_error(exc: BaseException) -> str:
    """Surface the innermost provider error message from a LiteLLM-wrapped APIError.

    OpenAI's streaming layer constructs `APIError(message=error["message"], ...)`
    from the upstream SSE event; when LiteLLM is the upstream, that `message` is
    the wrapped exception chain with the provider response stuffed inside as a
    `b'...'` Python literal. Walk every embedded bytes literal, parse it as
    JSON, and return `error.message` (or top-level `message`). Fall back to
    `str(exc)` when nothing parses, so we never lose the original signal.

    Args:
        exc: The exception whose string form may contain embedded provider JSON.

    Returns:
        The first nested provider message found in an embedded JSON bytes literal,
        or `str(exc)` if no provider message can be extracted.
    """
    raw = str(exc)
    for match in _BYTES_LITERAL_RE.finditer(string=raw):
        try:
            decoded = ast.literal_eval(node_or_string=match.group(0)).decode(
                encoding="utf-8", errors="replace"
            )
            data = json.loads(s=decoded)
        except (SyntaxError, ValueError, TypeError, AttributeError):
            continue
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                inner = error.get("message")
                if isinstance(inner, str) and inner:
                    return inner
            top = data.get("message")
            if isinstance(top, str) and top:
                return top
    return raw
