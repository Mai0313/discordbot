"""Tests for cross-provider server-side tool-use accounting on Responses API answers."""

from types import SimpleNamespace
from typing import cast

from openai.types.responses import Response

from discordbot.cogs.gen_reply.tool_use import AnswerToolUse


def _response(output: list[object] | None = None, usage: object = None) -> Response:
    """Builds a minimal `Response`-shaped stub for `AnswerToolUse.record_response`."""
    return cast("Response", SimpleNamespace(output=output or [], usage=usage))


def _usage(extra: dict[str, object] | None = None, with_details: bool = True) -> SimpleNamespace:
    """Builds a minimal `ResponseUsage`-shaped stub."""
    if not with_details:
        return SimpleNamespace(input_tokens=1, output_tokens=2)
    return SimpleNamespace(
        input_tokens=1,
        output_tokens=2,
        input_tokens_details=SimpleNamespace(model_extra=extra or {}),
    )


def _message_with_citations(citations: list[tuple[str, str]]) -> SimpleNamespace:
    """Builds a `message` output item carrying `url_citation` annotations."""
    return SimpleNamespace(
        type="message",
        content=[
            SimpleNamespace(
                type="output_text",
                annotations=[
                    SimpleNamespace(type="url_citation", url=url, title=title)
                    for url, title in citations
                ],
            )
        ],
    )


def _web_search_call(action: object = None) -> SimpleNamespace:
    """Builds a `web_search_call` output item."""
    return SimpleNamespace(type="web_search_call", action=action)


def test_openai_web_search_calls_count_distinct_queries_and_keep_output_citations() -> None:
    """OpenAI `web_search_call` queries are deduplicated and text `url_citation`s take precedence."""
    tool_use = AnswerToolUse()
    response = _response(
        output=[
            _web_search_call(
                action=SimpleNamespace(
                    type="search",
                    queries=["python 3.14", "openai responses"],
                    query=None,
                    sources=[
                        SimpleNamespace(type="url", url="https://unused.example.com/1"),
                        SimpleNamespace(type="url", url="https://unused.example.com/2"),
                    ],
                )
            ),
            _web_search_call(
                action=SimpleNamespace(
                    type="search",
                    queries=["python 3.14", "litellm bridge"],
                    query=None,
                    sources=[],
                )
            ),
            _message_with_citations(citations=[("https://docs.python.org/3/", "Python Docs")]),
        ],
        usage=_usage(extra={}),
    )

    tool_use.record_response(response=response)

    assert tool_use.web_search_queries == 3
    assert tool_use.url_fetches is None
    assert tool_use.url_citations == 1
    assert tool_use.cited_domains == ["docs.python.org"]


def test_web_search_call_precedence_sources_fallback_and_open_page_fetches() -> None:
    """Prefers `web_search_call` counts when both appear, falls back to `sources`, and counts `open_page`."""
    tool_use = AnswerToolUse()
    response = _response(
        output=[
            _web_search_call(
                action=SimpleNamespace(
                    type="search",
                    query="weather taipei",
                    queries=["weather taipei"],
                    sources=[
                        SimpleNamespace(type="url", url="https://cwa.gov.tw/a"),
                        SimpleNamespace(type="url", url="https://news.example.org/b"),
                        SimpleNamespace(type="url", url="https://cwa.gov.tw/a"),
                    ],
                )
            ),
            _web_search_call(
                action=SimpleNamespace(
                    type="search",
                    query="typhoon alert",
                    queries=["typhoon alert"],
                    sources=[SimpleNamespace(type="url", url="https://cwa.gov.tw/c")],
                )
            ),
            _web_search_call(action=SimpleNamespace(type="open_page", url="https://cwa.gov.tw/a")),
            _web_search_call(action=SimpleNamespace(type="open_page", url="https://cwa.gov.tw/a")),
            _web_search_call(
                action=SimpleNamespace(type="open_page", url="https://news.example.org/b")
            ),
            _message_with_citations(citations=[]),
        ],
        usage=_usage(extra={"web_search_requests": 2}),
    )

    tool_use.record_response(response=response)

    assert tool_use.web_search_queries == 2
    assert tool_use.url_fetches == 2
    assert tool_use.url_citations == 3
    assert tool_use.cited_domains == ["cwa.gov.tw", "news.example.org"]


def test_web_search_call_falls_back_to_single_query_and_unnamed_call() -> None:
    """Handles `action.query` without `queries` and `web_search_call` with no query strings."""
    tool_use = AnswerToolUse()
    response = _response(
        output=[
            _web_search_call(
                action=SimpleNamespace(type="search", queries=None, query="only one")
            ),
            _web_search_call(action=SimpleNamespace(type="search", queries=[], query="")),
            _web_search_call(action=None),
            _web_search_call(
                action=SimpleNamespace(
                    type="find_in_page", pattern="price", url="https://shop.example.com"
                )
            ),
        ],
        usage=None,
    )

    tool_use.record_response(response=response)

    assert tool_use.web_search_queries == 3
    assert tool_use.url_fetches is None
    assert tool_use.url_citations == 0
    assert tool_use.cited_domains == []


def test_usage_web_search_requests_guards_none_and_bool() -> None:
    """Explicit None or bool `web_search_requests` normalizes to 0; missing usage stays None."""
    for bad_value in (None, True, "3"):
        tool_use = AnswerToolUse()
        tool_use.record_response(
            response=_response(output=[], usage=_usage(extra={"web_search_requests": bad_value}))
        )
        assert tool_use.web_search_queries == 0

    minimal_usage = AnswerToolUse()
    minimal_usage.record_response(response=_response(output=[], usage=_usage(with_details=False)))
    assert minimal_usage.web_search_queries == 0

    no_usage = AnswerToolUse()
    no_usage.record_response(response=_response(output=[], usage=None))
    assert no_usage.web_search_queries is None
