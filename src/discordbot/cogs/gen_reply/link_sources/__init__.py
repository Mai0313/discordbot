"""Descriptor for one linked-content source `gen_reply` reads into answer context.

Each source (Threads, Douyin, Bilibili, ...) keeps its own builder module beside this one in
this package; the model here only carries the wiring `gen_reply` needs to treat them
uniformly: spot the URL, decide how far to look for it, start the intent-selected build, gate its
media ingestion, and inject a deterministic notice when the build outruns the post-route
grace. A build starts only after the router selects that source for QA, so an incidental URL
never reaches its network-capable builder. How far to look is per-source rather than global
(`search_replied_to_message`): Threads also reads a link the user only replied to, while Douyin
and Bilibili stay on the triggering message, since their value is the clip rather than a
discussion and both are rate-limit sensitive. The registry instances live in `registry.py` beside
this file (`LINK_CONTEXT_SOURCES`) as thin adapters over the builder functions: an adapter body
resolves the builder name from that module's globals at call time, so a test monkeypatching
`discordbot.cogs.gen_reply.link_sources.registry.build_*_context_messages` still intercepts the
call. Adding a source is one builder module here, a `utils/` URL regex, a route-schema and prompt
source name, and one registry entry.
"""

import re
from typing import Any, Protocol
from collections.abc import Callable, Sequence, Coroutine

from google import genai
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam

from discordbot.typings.llm import LLMConfig


def system_block(*, text: str) -> EasyInputMessageParam:
    """Wraps one separator or notice string as a low-authority system block."""
    return EasyInputMessageParam(
        role="system", content=[ResponseInputTextParam(text=text, type="input_text")]
    )


def link_context_blocks(
    *, separator: str, text: str, media_parts: Sequence[ResponseInputFileParam] = ()
) -> list[EasyInputMessageParam]:
    """The separator plus the post itself, the shape a readable source returns.

    The separator carries the source's own claim about what is attached, so it is the caller's
    to choose: the same post renders under a "here is its media" wording when the upload landed
    and under a "text only" one when it did not, and the difference is exactly what stops the
    model describing media it never received.
    """
    return [
        system_block(text=separator),
        EasyInputMessageParam(
            role="user",
            content=[ResponseInputTextParam(text=text, type="input_text"), *media_parts],
        ),
    ]


# The pipeline's own inline markers, opening or closing. Quoted post text is the one place they
# can arrive written by someone else; `defuse_markers` has the why. Case-insensitive because
# `markers.py` extracts case-insensitively, and a defusing pass that is stricter than the
# extraction it defends against is no defence at all.
_MARKER_TAG_RE = re.compile(
    r"</?(generate-(?:voice|image|music|video)|deep-research|write-memory|forget-memory"
    r"|write-server-memory)>",
    flags=re.IGNORECASE,
)


def defuse_markers(*, text: str) -> str:
    """Breaks the pipeline's own inline markers where they appear inside quoted post text.

    `extract_inline_markers` reads the answer model's OWN output, so a `<generate-video>` tag
    written into a linked post or one of its comments becomes a real render the moment the model
    quotes it back — which is exactly what "what does this comment say" asks it to do. Extraction
    runs regardless of the kill-switches, so the tag has to stop being a tag here. Cheap to write
    and cheap to abuse otherwise: a comment on a viral post costs an attacker nothing.

    The memory tags are defused for a different cost. A quoted `<forget-memory>` fires no render
    and spends nothing, so nothing in the logs looks wrong; it writes into the replied-to user's
    own long-term memory, and what it can reach there survives every later conversation.

    Shared by the sources that carry a DISCUSSION rather than a caption — Threads and Facebook,
    each of which hands the model thousands of characters written by strangers. Douyin and
    Bilibili do not use it: a caption or a video title is one line by its own author, and
    `tests/test_prompt_guards.py` owns the prompt rule that covers every undefused path.
    """
    return _MARKER_TAG_RE.sub(repl=lambda match: f"({match.group(1)})", string=text)


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
        ..., description="The first match in the scanned message selects the URL to read."
    )
    url_filter: SkipValidation[LinkUrlFilter | None] = Field(
        default=None, description="Optional post-match guard; None accepts every pattern match."
    )
    search_replied_to_message: bool = Field(
        default=False,
        description="Whether a link in the message being replied to also selects this source.",
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
