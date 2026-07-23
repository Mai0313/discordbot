"""Tests for the shared best-effort Responses helpers in utils/llm."""

from types import SimpleNamespace
from typing import cast

from openai.types.responses import Response

from discordbot.utils.llm import output_text_or_empty


def _message(*parts: object) -> SimpleNamespace:
    """A fake message output item carrying the given content parts."""
    return SimpleNamespace(type="message", content=list(parts))


def _text_part(text: str | None) -> SimpleNamespace:
    """A fake output_text content part with the given (possibly None) text."""
    return SimpleNamespace(type="output_text", text=text)


def _as_response(fake: SimpleNamespace) -> Response:
    """Views a fake output-bearing double as the Response `output_text_or_empty` expects.

    Production discriminates output/content items on `.type` string, not isinstance
    (see `_gen_reply/streaming.py::_consume`), so a SimpleNamespace stand-in is valid at runtime.
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
