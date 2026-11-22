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


class AnswerToolUse(BaseModel):
    """What the answer's server-side tools did, as far as its backend reports it.

    None on a field means the backend does not report it, never zero: a zero there would read as
    an answer that did not search or cite.
    """

    web_search_queries: int | None = Field(
        default=None, description="Distinct Google Search queries the answer ran."
    )
    url_fetches: int | None = Field(
        default=None,
        description="Distinct URLs URL context retrieved. The Responses bridge never reports it.",
    )
    url_citations: int | None = Field(
        default=None, description="`url_citation` annotations on the answer text."
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
        """Records citations and search-query counts reported on a completed Responses API answer.

        Args:
            response (Response): Completed Responses API response.
        """
        self.record_citations(citations=_url_citations(output=response.output))
        if response.usage:
            details = response.usage.input_tokens_details
            extra = (details.model_extra if details is not None else None) or {}
            self.web_search_queries = extra.get("web_search_requests", 0)
