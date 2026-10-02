"""Descriptor for one linked-content source `gen_reply` reads into answer context.

Each source keeps its own builder module beside this one; the model here carries only the wiring
`gen_reply` needs to treat them uniformly: spot the URL, decide how far to look for it, start the
intent-selected build, gate its media ingestion, and inject a deterministic notice when the build
outruns the post-route grace.

A build starts only after the router selects that source for QA, so an incidental URL never
reaches a network-capable builder.

How far to look is per source rather than global (`search_replied_to_message`). A source opts in
when what it fetches includes something its own expansion does not show — the comments under the
post — so a mention on someone else's link has something new to answer from. A source whose value
is a single clip stays on the triggering message: a second read finds what the first did, and
those platforms are the rate-limit sensitive ones.

`registry.py` holds the instances and says why each entry's `build` is an adapter rather than the
builder itself.

What the builders share lives here too: the block shapes, the marker defusing, the comment lines,
the clip quality, `read_post`, the read every conversation-shaped post source starts with, and
`build_post_context`, the read-render-upload flow every one of them runs except Threads, whose
media step splits one budget across two posts.
"""

import re
from typing import Any, Protocol
import asyncio
from collections.abc import Callable, Sequence, Coroutine

from google import genai
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.video import VideoQuality
from discordbot.typings.emojis import LinkSourceName
from discordbot.cogs.gen_reply.markers import MARKER_TAG_NAMES
from discordbot.services.platforms.base import (
    PlatformOutput,
    LinkableComments,
    PlatformConversation,
    LinkableCommentOutput,
)
from discordbot.cogs.gen_reply.link_sources.image_ingest import upload_post_images

# Resolution asked for a clip the model reads: the lowest preset. The model samples frames at its
# own media resolution, so extra source pixels buy it nothing while costing download and upload
# time on the reply's critical path, which on long-form video scales with duration first (and
# anonymous Bilibili access mostly tops out around 480p regardless). Deliberately below what the
# bot asks for a clip it posts to Discord, where a human watching does notice.
AI_INGEST_QUALITY: VideoQuality = "low"


def system_block(text: str) -> EasyInputMessageParam:
    """Wraps one separator or notice string as a low-authority system block."""
    return EasyInputMessageParam(
        role="system", content=[ResponseInputTextParam(text=text, type="input_text")]
    )


def link_context_blocks(
    separator: str, text: str, media_parts: Sequence[ResponseInputFileParam] = ()
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


class PostSeparators(BaseModel):
    """What a source says around a post it could read, in that source's own words.

    Three strings rather than one, because which of the first two opens the block is decided by
    what actually arrived rather than by what the post has.
    """

    attached: str = Field(
        ..., description="Opens the block when the post's media really is below it."
    )
    text_only: str = Field(
        ...,
        description=(
            "Opens it instead when media EXISTS and did not arrive, so the model says what it "
            "has rather than inventing a scene. Never used for a post that simply carries none, "
            "which would have it apologise for nothing."
        ),
    )
    trailer: str = Field(..., description="Closes the quoted block, and is always its last part.")


def post_context_blocks(
    text: str,
    media_parts: Sequence[ResponseInputFileParam],
    post_carries_media: bool,
    separators: PostSeparators,
) -> list[EasyInputMessageParam]:
    """Assembles the blocks for a post that could be read.

    The trailer rides AFTER the attachments rather than at the end of the text: the images are
    the one part of this block nothing ever looked inside, so a fence that closed before them
    would leave an instruction-shaped screenshot sitting past the end-of-data marker. That is
    the reason this lives in one place rather than once per source.

    Args:
        text: The rendered post.
        media_parts: Whatever the upload actually produced, which is not what the post carries.
        post_carries_media: Whether the post has media at all, deciding which separator opens it.
        separators: The source's own wording.

    Returns:
        Input blocks ready to splice into the answer input.
    """
    if media_parts:
        return [
            system_block(text=separators.attached),
            EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(text=text, type="input_text"),
                    *media_parts,
                    ResponseInputTextParam(text=separators.trailer, type="input_text"),
                ],
            ),
        ]
    opener = separators.text_only if post_carries_media else separators.attached
    return link_context_blocks(separator=opener, text=f"{text}\n\n{separators.trailer}")


class PostReader[ConversationT](Protocol):
    """Reads one post URL into its conversation; a platform downloader satisfies it."""

    def parse_metadata(self, url: str) -> ConversationT: ...


class PostRenderer[OutputT, ConversationT](Protocol):
    """Renders a readable post and the conversation around it as the text the model reads."""

    def __call__(
        self, post: OutputT, conversation: ConversationT, attached_images: int
    ) -> str: ...


async def read_post[ConversationT: PlatformConversation[Any]](
    platform: str,
    url: str,
    reader: Callable[[], PostReader[ConversationT]],
    readable: Callable[[ConversationT], bool],
) -> ConversationT | None:
    """Reads a post URL into its conversation, or None when there is nothing to show.

    Never raises, and logs whichever way it fails, so a caller only has to inject its notice.

    Args:
        platform: The platform's display name, for the log lines.
        url: The post URL found in the conversation.
        reader: Builds the platform's downloader.
        readable: Whether the conversation holds a post worth showing. The source's own rule,
            since what counts as readable differs per platform.

    Returns:
        The conversation, or None when the read failed or `readable` refused it.
    """
    try:
        conversation = await asyncio.to_thread(reader().parse_metadata, url=url)
    # Broad on purpose: a parse error must degrade to the unavailable notice rather than break
    # the reply pipeline, which relies on every builder never raising.
    except Exception as error:
        logfire.warn(
            f"{platform} post read failed; injecting unavailable notice",
            url=url,
            error_type=type(error).__name__,
            _exc_info=error,
        )
        return None
    if not readable(conversation):
        logfire.info(
            f"{platform} post unavailable for context; injecting unavailable notice", url=url
        )
        return None
    return conversation


async def build_post_context[OutputT: PlatformOutput, ConversationT: PlatformConversation[Any]](  # noqa: PLR0913 -- a source's reader and wording plus the per-call inputs every builder takes
    platform: str,
    url: str,
    reader: Callable[[], PostReader[ConversationT]],
    render: PostRenderer[OutputT, ConversationT],
    separators: PostSeparators,
    unavailable_notice: str,
    image_cap: int,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
    deadline: float,
) -> list[EasyInputMessageParam]:
    """Reads a post URL into answer-model input blocks, uploading up to `image_cap` of its images.

    Returns `[separator, user-content]` for a readable post, or a single notice block saying it
    could not be read. Never raises: every failure degrades to a deterministic notice so the
    reply pipeline is never broken by it.

    Args:
        platform: The platform's display name, for the span and the log lines.
        url: The post URL found in the conversation.
        reader: Builds the platform's downloader.
        render: Renders the readable post; handed the number of images that actually rode in.
        separators: The source's own wording around the post.
        unavailable_notice: What the model is told instead when the post could not be read.
        image_cap: How many of the post's images one reply may pay a fetch and an upload for.
        answer_model_is_gemini: Whether the answer model can resolve a Files API uri.
        gemini_client: Direct-to-Google client used for the image upload, or None when no key
            is configured, which reads the post as text just like a non-Gemini answer model.
        allow_media_ingest: Kill-switch plus key check; when false only the text is read.
        deadline: Event-loop time the pipeline cancels this build at, which the image upload
            stops short of so the text still comes back.

    Returns:
        Input blocks ready to splice into the answer input before the current message.
    """
    with logfire.span(f"gen_reply {platform.lower()} context"):
        conversation = await read_post(
            platform=platform,
            url=url,
            reader=reader,
            readable=lambda conversation: (
                conversation.target is not None and conversation.target.is_readable
            ),
        )
        if conversation is None or conversation.target is None:
            return [system_block(text=unavailable_notice)]

        target = conversation.target
        media_parts: list[ResponseInputFileParam] = []
        if answer_model_is_gemini and allow_media_ingest and gemini_client is not None:
            media_parts = await upload_post_images(
                platform=platform,
                post_url=target.url,
                image_urls=target.image_urls,
                cap=image_cap,
                gemini_client=gemini_client,
                deadline=deadline,
            )

    text = render(post=target, conversation=conversation, attached_images=len(media_parts))
    return post_context_blocks(
        text=text,
        media_parts=media_parts,
        post_carries_media=bool(target.image_urls or target.video_urls),
        separators=separators,
    )


# The pipeline's own inline markers, opening or closing, derived from the extractor's own set so
# a tag added there cannot be missed here. Case-insensitive because the extraction is, and a
# defusing pass stricter than what it defends against is no defence at all.
_MARKER_TAG_RE = re.compile(
    pattern=rf"</?({'|'.join(re.escape(pattern=name) for name in MARKER_TAG_NAMES)})>",
    flags=re.IGNORECASE,
)


def defuse_markers(text: str) -> str:
    """Breaks the pipeline's own inline markers where they appear inside quoted post text.

    `extract_inline_markers` reads the answer model's OWN output, so a `<generate-video>` tag
    written into a linked post or one of its comments becomes a real render the moment the model
    quotes it back — which is exactly what "what does this comment say" asks it to do. Extraction
    runs regardless of the kill-switches, so the tag has to stop being a tag here. Cheap to write
    and cheap to abuse otherwise: a comment on a viral post costs an attacker nothing.

    The memory tags are defused for a different cost. A quoted `<forget-memory>` fires no render
    and spends nothing, so nothing in the logs looks wrong; it writes into the replied-to user's
    own long-term memory, and what it can reach there survives every later conversation.

    Used by every source that hands the model posts around the linked one as well: its comments,
    or the post it replies to or quotes. Douyin and Bilibili do not use it: a caption or a video
    title is one line by its own author, and `tests/test_prompt_guards.py` owns the prompt rule
    that covers every undefused path.
    """
    return _MARKER_TAG_RE.sub(repl=lambda match: f"({match.group(1)})", string=text)


def comment_lines[OutputT: LinkableCommentOutput](
    conversation: LinkableComments[OutputT], cap: int, served_as: str, handle_prefix: str
) -> list[str]:
    """Renders up to `cap` comments under a header counting them, marking the one the link named.

    The named comment is labelled rather than moved to the front: its position in the thread is
    part of reading it, and a model told which one was linked can answer about it without losing
    what came before. One sitting past the cap rides after the others as one more, since it is the
    comment the user is asking about, and the header counts it.

    Args:
        conversation: The post's conversation.
        cap: How many comments ride, not counting a named one past it.
        served_as: Closes the header, saying in the source's own words how much of the
            discussion the page served.
        handle_prefix: Written before each author's name, e.g. `@` where names are handles.

    Returns:
        Lines to append to the rendered post, none when it has no comments.
    """
    comments = conversation.comments[:cap]
    selected = conversation.selected_comment
    if selected is not None and selected not in comments:
        comments.append(selected)
    if not comments:
        return []
    lines = [f"\n[{len(comments)} of the post's comments, {served_as}]"]
    for comment in comments:
        marker = (
            " (this is the comment the link points at)"
            if comment.comment_id == conversation.selected_comment_id
            else ""
        )
        author = defuse_markers(text=comment.author_name)
        lines.append(f"- {handle_prefix}{author}{marker}: {defuse_markers(text=comment.text)}")
    return lines


class LinkUrlFilter(Protocol):
    """Post-match guard rejecting a matched URL the source cannot read (e.g. a profile)."""

    def __call__(self, url: str) -> bool: ...


class LinkContextBuilder(Protocol):
    """Normalized builder signature every source adapter satisfies."""

    def __call__(
        self,
        url: str,
        answer_model_is_gemini: bool,
        gemini_client: genai.Client | None,
        allow_media_ingest: bool,
        deadline: float,
    ) -> Coroutine[Any, Any, list[EasyInputMessageParam]]: ...


class MediaIngestPredicate(Protocol):
    """Config predicate deciding whether the source may download and upload media."""

    def __call__(self, config: LLMConfig) -> bool: ...


class LinkContextSource(BaseModel):
    """One linked-content source: how to spot its URL, build its blocks, and gate its media."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: LinkSourceName = Field(
        ...,
        description="Source label used for logs, task-discard labels, and the splice order.",
        examples=["douyin"],
    )
    url_pattern: SkipValidation[re.Pattern[str]] = Field(
        ..., description="The first match `url_filter` accepts selects the URL to read."
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
    timeout_notice: str = Field(
        ...,
        description=(
            "Deterministic notice injected for a build that outruns the post-route grace, so a "
            "slow fetch does not leave the model with only the raw URL."
        ),
    )
    media_ingest_allowed: SkipValidation[MediaIngestPredicate] = Field(
        ..., description="Kill-switch predicate for media ingestion."
    )

    def timeout_blocks(self) -> list[EasyInputMessageParam]:
        """The blocks injected in place of a build that outran the post-route grace.

        Returns:
            The timeout notice as a single system block.
        """
        return [system_block(text=self.timeout_notice)]
