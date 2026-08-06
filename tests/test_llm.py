"""Pins `utils/llm.py::output_text_or_empty` against the one SDK quirk it exists for.

The SDK's own `Response.output_text` bare-joins every output_text part's `text`, so a single part
arriving with `text=None` — a Gemini-via-proxy shape seen on some grounded / refused turns —
raises `TypeError` from inside the property, where the usual `(responses.output_text or "")` guard
can never catch it. `output_text_or_empty` is the replacement aggregation, and both of its runtime
callers sit on best-effort paths whose broad `except` would absorb that `TypeError` into a silent
degrade: `create_text_or_none` would hand its caller None and a fallback line would go out
instead, and `gen_reply/generation.py`'s prompt director would send the raw request unrefined.
Neither loss is reported anywhere, which is why the shapes are pinned here rather than left to
review.

Three tests cover the whole contract: a lone None part yields the empty string instead of raising,
a None part between valid ones is skipped rather than truncating the join, and a non-message
output item (reasoning) plus a non-text content part (refusal) contribute nothing.

The doubles are `SimpleNamespace` casts rather than real SDK models, and have to be: the SDK types
`ResponseOutputText.text` as a required `str`, so the very shape under test cannot be built
through validation. The cast is honest at runtime because the helper discriminates output items
and content parts on their `.type` literal rather than by isinstance, the same way
`gen_reply/streaming.py::_consume` reads its stream events.
"""

from types import SimpleNamespace
from typing import cast

from openai.types.responses import Response

from discordbot.utils.llm import output_text_or_empty


def _message(*parts: object) -> SimpleNamespace:
    """Builds a fake message output item carrying the given content parts.

    Returns:
        A double whose `.type` is "message" and whose `content` holds `parts` in order.
    """
    return SimpleNamespace(type="message", content=list(parts))


def _text_part(text: str | None) -> SimpleNamespace:
    """Builds a fake output_text content part whose `text` may be None.

    Returns:
        A double the SDK's own models cannot express, since `ResponseOutputText.text` is a
        required `str`.
    """
    return SimpleNamespace(type="output_text", text=text)


def _as_response(fake: SimpleNamespace) -> Response:
    """Views a fake output-bearing double as the Response `output_text_or_empty` expects.

    Production discriminates output/content items on `.type` string, not isinstance
    (see `gen_reply/streaming.py::_consume`), so a SimpleNamespace stand-in is valid at runtime.

    Returns:
        The same object, retyped so the call reads as the production one it stands in for.
    """
    return cast("Response", fake)


def test_output_text_or_empty_tolerates_none_text_part() -> None:
    """A lone output_text part with text=None yields "" instead of raising (the reported bug)."""
    responses = SimpleNamespace(output=[_message(_text_part(None))])
    assert output_text_or_empty(responses=_as_response(fake=responses)) == ""


def test_output_text_or_empty_joins_valid_parts_and_skips_none() -> None:
    """Valid text parts are joined; a None part in the middle is skipped, never raising."""
    responses = SimpleNamespace(
        output=[_message(_text_part("hello "), _text_part(None), _text_part("world"))]
    )
    assert output_text_or_empty(responses=_as_response(fake=responses)) == "hello world"


def test_output_text_or_empty_ignores_non_text_content_and_items() -> None:
    """A reasoning output item and a refusal content part contribute nothing."""
    responses = SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning", content=[]),
            _message(SimpleNamespace(type="refusal", refusal="no"), _text_part("kept")),
        ]
    )
    assert output_text_or_empty(responses=_as_response(fake=responses)) == "kept"
