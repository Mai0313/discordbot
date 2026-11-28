"""Server-side tool-use accounting for streaming answer turns."""

import re

from pydantic import Field, BaseModel
from openai.types.responses import Response, ResponseOutputItem

from discordbot.utils.urls import normalized_host


def _citation_domain(url: str, title: str) -> str | None:
    """Returns the cited domain from `url`, falling back to `title` for Vertex redirects."""
    host = normalized_host(url=url)
    if host != "vertexaisearch.cloud.google.com":
        return host or None
    if re.fullmatch(pattern=r"(?:[a-z0-9-]+\.)+[a-z]{2,}", string=title, flags=re.IGNORECASE):
        return title.lower()
    return None


def _url_citations(output: list[ResponseOutputItem]) -> list[tuple[str, str]]:
    """The `(url, title)` of every `url_citation` annotation across message output text parts."""
    return [
        (annotation.url, annotation.title)
        for item in output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
        for annotation in part.annotations
        if annotation.type == "url_citation"
    ]


def _search_action_queries(action: object) -> list[str]:
    """Returns non-empty query strings carried by a `web_search_call` search `action`."""
    raw_queries = getattr(action, "queries", None)
    queries = [q for q in (raw_queries or ()) if isinstance(q, str) and q]
    if queries:
        return queries
    raw_query = getattr(action, "query", None)
    return [raw_query] if isinstance(raw_query, str) and raw_query else []


def _web_search_call_stats(output: list[ResponseOutputItem]) -> tuple[int, list[str], set[str]]:
    """Extracts `(search_count, source_urls, opened_urls)` from `web_search_call` output items."""
    queries: set[str] = set()
    unnamed_calls = 0
    source_urls: list[str] = []
    opened_urls: set[str] = set()
    for item in output:
        if item.type != "web_search_call":
            continue
        action = getattr(item, "action", None)
        action_type = getattr(action, "type", None) if action is not None else None
        if action is None or action_type == "search":
            call_queries = _search_action_queries(action=action) if action is not None else []
            if call_queries:
                queries.update(call_queries)
            else:
                unnamed_calls += 1
            for source in getattr(action, "sources", None) or ():
                url = getattr(source, "url", None)
                if isinstance(url, str) and url:
                    source_urls.append(url)
        elif action_type == "open_page":
            url = getattr(action, "url", None)
            if isinstance(url, str) and url:
                opened_urls.add(url)
    return len(queries) + unnamed_calls, list(dict.fromkeys(source_urls)), opened_urls


class AnswerToolUse(BaseModel):
    """What the answer's server-side tools did, as far as its backend reports it.

    None on a field means the backend does not report it, never zero: a zero there would read as
    an answer that did not search or cite.
    """

    web_search_queries: int | None = Field(
        default=None, description="Distinct web search queries (or unnamed search calls) run."
    )
    url_fetches: int | None = Field(
        default=None,
        description=(
            "Distinct URLs retrieved via server-side URL fetch (`open_page` or Interactions "
            "`url_context_call`). None when the backend does not report URL fetches."
        ),
    )
    url_citations: int | None = Field(
        default=None,
        description=(
            "`url_citation` annotations on the answer text, or distinct `web_search_call` "
            "source URLs when the response carries no text annotations."
        ),
    )
    cited_domains: list[str] | None = Field(
        default=None,
        description="Distinct domains those citations point at, in first-cited order.",
        examples=[["wikipedia.org", "nownews.com"]],
    )

    def record_citations(self, citations: list[tuple[str, str]]) -> None:
        """Records the answer's `(url, title)` citations as a count and distinct domains.

        Args:
            citations (list[tuple[str, str]]): Citation `(url, title)` pairs.
        """
        self.url_citations = len(citations)
        domains = (_citation_domain(url=url, title=title) for url, title in citations)
        self.cited_domains = list(dict.fromkeys(domain for domain in domains if domain))

    def record_response(self, response: Response) -> None:
        """Records citations, search queries, and page fetches from a completed Responses answer.

        Args:
            response (Response): Completed Responses API response.
        """
        output_search_count, source_urls, opened_urls = _web_search_call_stats(
            output=response.output
        )
        citations = _url_citations(output=response.output)
        if not citations and source_urls:
            citations = [(url, "") for url in source_urls]
        self.record_citations(citations=citations)
        if opened_urls:
            self.url_fetches = len(opened_urls)
        usage_search_count: int | None = None
        if response.usage is not None:
            details = getattr(response.usage, "input_tokens_details", None)
            extra = (getattr(details, "model_extra", None) if details is not None else None) or {}
            raw_requests = extra.get("web_search_requests")
            usage_search_count = (
                raw_requests
                if isinstance(raw_requests, int) and not isinstance(raw_requests, bool)
                else 0
            )
        if output_search_count > 0:
            self.web_search_queries = output_search_count
        elif usage_search_count is not None:
            self.web_search_queries = usage_search_count
