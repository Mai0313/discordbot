"""Turns a failed LLM call's exception into the one line worth showing a user.

An OpenAI-SDK streaming failure that came back through the LiteLLM proxy carries the provider's
own explanation nested inside its message rather than beside it: the SDK's streaming layer
constructs `APIError(message=error["message"], ...)` straight from the upstream SSE error event,
and LiteLLM has already written its own wrapper chain into that field with the provider's raw
response body embedded as a Python bytes literal
(`litellm.X: litellm.Y: VertexException - b'{"error": {"message": "quota exceeded"}}'`), so that
chain is the entirety of `str(exc)`. The sentence a user can act on is the innermost one, so this
module is the string surgery that digs it back out.

The contract is deliberately narrow. It reads the exception's own text and nothing else — no
`__cause__` / `__context__` walk, because the nesting here happens inside one message rather
than across a chain of exceptions — and it never raises and never returns an empty EXTRACTED
message: anything it cannot parse falls back to `str(exc)`, which is itself empty for an
exception carrying no args (an ordinary timeout shape), so an empty result is the caller's to
handle — `cogs/research` substitutes its own wording. That fallback is also the whole of the
direct-to-Google paths' behavior (deep research and the native Interactions calls), where no
proxy wrapped the error in the first place, which is what lets every failure surface call this
unconditionally. An extracted message is provider-authored and comes back as the provider wrote
it, bar undecodable bytes turned into U+FFFD by the decode; neither branch shortens or escapes
anything, so length limits and fencing stay the caller's problem.

It lives here rather than in a cog because both failure surfaces need it — `gen_reply`'s outer
error embed and `cogs/research`'s failure notice — and a cog may not import a peer to reach one.
"""

import re
import ast
import json

# LiteLLM surfaces upstream provider errors as a chain like
# `litellm.X: litellm.Y: VertexException - b'{"error": {"message": "..."}}'`,
# where the provider's actual JSON body is embedded as a Python bytes literal.
_BYTES_LITERAL_RE = re.compile(pattern=r"b'((?:[^'\\]|\\.)*)'", flags=re.DOTALL)


def extract_friendly_error(exc: BaseException) -> str:
    """Digs the provider's own error message out of a LiteLLM wrapper chain.

    Scans the exception's text for embedded `b'...'` literals, decoding and JSON-parsing each in
    turn, and takes the first message it finds: `error.message`, then a top-level `message` for a
    provider that flattens its body. A literal that fails to decode, or decodes to something other
    than that shape, is skipped without ending the scan, so a stray `b'...'` in the text can
    neither hijack the result nor cost a real message sitting later in the same string.

    Args:
        exc (BaseException): The exception whose text may carry an embedded provider response.

    Returns:
        The first non-empty provider message found in an embedded JSON bytes literal, else
        `str(exc)`, which is itself empty for an exception carrying no args.
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
