"""Descriptor for one linked-content source `gen_reply` reads into answer context.

Each source (Threads, Douyin, Bilibili, ...) keeps its own builder module beside this one in
this package; the model here only carries the wiring `gen_reply` needs to treat them
uniformly: spot the URL, start the speculative build, gate its media ingestion, and inject a
deterministic notice when the build outruns the post-route grace. The registry instances live
in `gen_reply.py` (`LINK_CONTEXT_SOURCES`) as thin adapters over the builder functions: an
adapter body resolves the builder name from that module's globals at call time, so a test
monkeypatching `discordbot.cogs.gen_reply.build_*_context_messages` still intercepts the
call. Adding a source is one builder module here, a `utils/` URL regex, and one registry
entry.
"""

import re
from typing import Any, Protocol
from collections.abc import Callable, Coroutine

from google import genai
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.llm import LLMConfig


class LinkUrlFilter(Protocol):
    """Post-match guard rejecting a matched URL the source cannot read (e.g. a profile)."""

    def __call__(self, url: str) -> bool: ...


class LinkContextBuilder(Protocol):
    """Normalized builder signature every source adapter satisfies."""

    def __call__(
        self,
        *,
        url: str,
        answer_model_is_gemini: bool,
        gemini_client: genai.Client | None,
        allow_media_ingest: bool,
    ) -> Coroutine[Any, Any, list[EasyInputMessageParam]]: ...


class MediaIngestPredicate(Protocol):
    """Config predicate deciding whether the source may download and upload media."""

    def __call__(self, config: LLMConfig) -> bool: ...


class LinkContextSource(BaseModel):
    """One linked-content source: how to spot its URL, build its blocks, and gate its media."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        ...,
        description="Source label used for logs, task-discard labels, and the splice order.",
        examples=["douyin"],
    )
    url_pattern: SkipValidation[re.Pattern[str]] = Field(
        ..., description="The first match in the message selects the URL to read."
    )
    url_filter: SkipValidation[LinkUrlFilter | None] = Field(
        default=None, description="Optional post-match guard; None accepts every pattern match."
    )
    build: SkipValidation[LinkContextBuilder] = Field(
        ..., description="Adapter starting the context build with the normalized keyword set."
    )
    on_timeout: SkipValidation[Callable[[], list[EasyInputMessageParam]]] = Field(
        ...,
        description="Deterministic notice blocks for a build that outruns the post-route grace.",
    )
    media_ingest_allowed: SkipValidation[MediaIngestPredicate] = Field(
        ...,
        description="Kill-switch predicate for media ingestion; a switchless source returns True.",
    )
