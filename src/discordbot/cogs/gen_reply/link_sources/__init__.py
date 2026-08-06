"""Descriptor for one linked-content source `gen_reply` reads into answer context.

Each source (Threads, Douyin, Bilibili, ...) keeps its own builder module beside this one in this
package; the model here only carries the wiring `gen_reply` needs to treat them uniformly: spot
the URL, decide how far to look for it, start the intent-selected build, gate its media ingestion,
and inject a deterministic notice when the build outruns the post-route grace. A build starts only
after the router selects that source for QA, so an incidental URL never reaches its
network-capable builder. How far to look is per-source rather than global
(`search_reference_chain`): Threads also reads a link the user only replied to, while Douyin and
Bilibili stay on the triggering message, since their value is the clip rather than a discussion
and both are rate-limit sensitive. The builders live here rather than in the `parse_threads` /
`parse_douyin` expansion cogs because `gen_reply` is their only consumer and no cog may import a
peer cog; what Douyin genuinely shares with both surfaces sits in `utils/douyin.py` instead. The
registry instances live in `gen_reply/cog.py` (`LINK_CONTEXT_SOURCES`) as thin adapters over the
builder functions: an adapter body resolves the builder name from that module's globals at call
time, so a test monkeypatching `discordbot.cogs.gen_reply.cog.build_*_context_messages` still
intercepts the call. Adding a source is one builder module here, a `utils/` URL regex, a
route-schema and prompt source name, and one registry entry.
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

    def __call__(self, url: str) -> bool:
        """Whether the source should spend a request reading this URL.

        Judged on the URL alone, before any fetch, since the point is to avoid spending a
        rate-limited request finding out. It sees only the match the scan already chose, so a
        rejection drops the source for this message rather than hunting for a second URL.

        Args:
            url (str): The URL `url_pattern` matched, exactly as it appeared in the message.

        Returns:
            True when the source should read this URL.
        """
        ...


class LinkContextBuilder(Protocol):
    """Normalized builder signature every source adapter satisfies."""

    def __call__(
        self,
        *,
        url: str,
        answer_model_is_gemini: bool,
        gemini_client: genai.Client | None,
        allow_media_ingest: bool,
    ) -> Coroutine[Any, Any, list[EasyInputMessageParam]]:
        """Reads the linked post into blocks ready to splice into the answer input.

        An implementation must never raise: every failure degrades to its own notice block, so
        the model says what it could not read instead of hallucinating the post, and anything
        the pipeline's resolver does catch is a bug rather than a failed fetch. Cancellation is
        the exception and must propagate, since the shared post-route deadline is what stops a
        slow build and the pipeline waits on that cancellation to finish.

        Args:
            url (str): The linked-post URL this source selected.
            answer_model_is_gemini (bool): Whether the answer model can resolve a Files API uri;
                anything else gets text plus URLs rather than uploaded media.
            gemini_client (genai.Client | None): Direct-to-Google client for the media upload,
                None when no key is configured.
            allow_media_ingest (bool): The source's `media_ingest_allowed` verdict; false still
                reads the post's text and only skips its media.

        Returns:
            The pending build, resolving to this source's separator plus the post's content, or
            to its own notice block when the post could not be read.
        """
        ...


class MediaIngestPredicate(Protocol):
    """Config predicate deciding whether the source may download and upload media."""

    def __call__(self, config: LLMConfig) -> bool:
        """Whether this source may fetch the linked media and upload it to the Files API.

        Called once when the build starts, before any network call, so a source switched off
        spends no download or upload. A source with no kill-switch of its own answers True and
        leaves the gating to the builder's own Gemini checks.

        Args:
            config (LLMConfig): The kill-switches and the direct-to-Google key.

        Returns:
            True when the build may ingest media; false leaves it with the post's text.
        """
        ...


class LinkContextSource(BaseModel):
    """One linked-content source: how to spot its URL, build its blocks, and gate its media."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        ...,
        description="Source label used for logs, task-discard labels, and the splice order.",
        examples=["douyin"],
    )
    url_pattern: SkipValidation[re.Pattern[str]] = Field(
        ..., description="The first match in the scanned message selects the URL to read."
    )
    url_filter: SkipValidation[LinkUrlFilter | None] = Field(
        default=None, description="Optional post-match guard; None accepts every pattern match."
    )
    search_reference_chain: bool = Field(
        default=False,
        description="Whether a link in the reply-reference chain also selects this source.",
        examples=[True],
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
