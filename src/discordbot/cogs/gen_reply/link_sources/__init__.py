"""Descriptor and shared helpers for linked-content sources read into `gen_reply` answer context.

`LinkContextSource` defines uniform wiring for each source (URL matching, optional replied-to
message lookup via `search_replied_to_message`, builder adapter, media-ingest gate, and timeout
notice). Sources are built only after the router selects them for QA.

Also provides the shared block builders (`marker_block`, `link_context_blocks`,
`post_context_blocks`), inline-marker defusing (`defuse_markers`), comment formatting
(`comment_lines`), and the shared `read_post` / `build_post_context` flow. Instances live in
`registry.py`.
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
from discordbot.utils.link_errors import LinkUnavailableError
from discordbot.utils.llm_transcript import defuse_separator_fences
from discordbot.cogs.gen_reply.markers import MARKER_TAG_NAMES
from discordbot.services.platforms.base import (
    PlatformOutput,
    LinkableComments,
    PlatformConversation,
    LinkableCommentOutput,
)
from discordbot.cogs.gen_reply.link_sources.image_ingest import upload_post_images

# Lowest preset: the model downsamples frames, so higher resolution only adds transfer latency.
AI_INGEST_QUALITY: VideoQuality = "low"


def marker_block(text: str) -> EasyInputMessageParam:
    """Returns `text` wrapped as a `role="user"` input block marking a section boundary.

    Uses `role="user"` rather than `role="system"` so markers stay in position; `====` fences
    distinguish them from user text sanitized by `defuse_separator_fences`.

    Args:
        text (str): Separator or notice text to wrap.
    """
    return EasyInputMessageParam(
        role="user", content=[ResponseInputTextParam(text=text, type="input_text")]
    )


def link_context_blocks(
    separator: str, text: str, media_parts: Sequence[ResponseInputFileParam] = ()
) -> list[EasyInputMessageParam]:
    """Returns the `[separator, user-content]` input blocks for a readable linked post.

    The caller chooses `separator` to match what `media_parts` actually holds so the model never
    claims to see media that was not attached.

    Args:
        separator (str): Marker stating what is attached below.
        text (str): Rendered post text.
        media_parts (Sequence[ResponseInputFileParam]): Uploaded media parts to attach.
    """
    return [
        marker_block(text=separator),
        EasyInputMessageParam(
            role="user",
            content=[
                ResponseInputTextParam(text=defuse_separator_fences(text=text), type="input_text"),
                *media_parts,
            ],
        ),
    ]


class PostSeparators(BaseModel):
    """Opening markers and closing trailer around a readable post.

    Whether `attached` or `text_only` opens the block depends on whether media parts actually
    arrived, not on whether the post carries media.
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
    """Returns the answer-model input blocks for a readable post.

    `separators.trailer` is placed after `media_parts` rather than at the end of `text` so any
    instruction-shaped text inside an attached image stays inside the quoted-data fence. It is
    joined after the post's text is defused, so its own fence stands.

    Args:
        text (str): The rendered post text.
        media_parts (Sequence[ResponseInputFileParam]): Uploaded media parts that arrived.
        post_carries_media (bool): Whether the post has any media, selecting `separators.text_only`
            when `media_parts` is empty.
        separators (PostSeparators): The source's opener and trailer strings.
    """
    text = defuse_separator_fences(text=text)
    if media_parts:
        return [
            marker_block(text=separators.attached),
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
    return [
        marker_block(text=opener),
        EasyInputMessageParam(
            role="user",
            content=[
                ResponseInputTextParam(text=f"{text}\n\n{separators.trailer}", type="input_text")
            ],
        ),
    ]


class PostReader[ConversationT](Protocol):
    """Reads one post URL into its conversation; a platform downloader satisfies it."""

    def parse_metadata(self, url: str) -> ConversationT: ...


class PostRenderer[OutputT, ConversationT](Protocol):
    """Renders a readable post and the conversation around it as the text the model reads."""

    def __call__(
        self, post: OutputT, conversation: ConversationT, attached_images: int
    ) -> str: ...


async def read_post[ConversationT: PlatformConversation[Any]](  # noqa: PLR0913 -- a source's reader and readability rule plus the two notices it chooses between
    platform: str,
    url: str,
    reader: Callable[[], PostReader[ConversationT]],
    readable: Callable[[ConversationT], bool],
    unavailable_notice: str,
    unreadable_notice: str,
) -> ConversationT | str:
    """Returns the parsed conversation for `url`, or the notice string to inject when it fails.

    Never raises; logs the failure at the severity matching its cause.

    Args:
        platform (str): Display name of the platform for log messages.
        url (str): The post URL found in the conversation.
        reader (Callable[[], PostReader[ConversationT]]): Zero-argument factory for the platform
            downloader.
        readable (Callable[[ConversationT], bool]): Predicate deciding whether the parsed
            conversation holds a showable post.
        unavailable_notice (str): Notice returned when the platform reports the post is gone or
            private.
        unreadable_notice (str): Notice returned when the read fails without proving the post is
            gone.
    """
    try:
        conversation = await asyncio.to_thread(reader().parse_metadata, url=url)
    except LinkUnavailableError as error:
        # Routine remote outcome, not a defect; log at info with the platform's reason.
        logfire.info(
            f"{platform} post is gone or private; injecting unavailable notice",
            url=url,
            reason=str(error),
        )
        return unavailable_notice
    # Broad on purpose: a parse error must degrade to a notice rather than break the reply
    # pipeline, which relies on every builder never raising. Anything but the platform saying
    # there is no post says nothing about the post, so it never gets the unavailable notice.
    except Exception as error:
        logfire.warn(
            f"{platform} post read failed; injecting unreadable notice",
            url=url,
            error_type=type(error).__name__,
            _exc_info=error,
        )
        return unreadable_notice
    if not readable(conversation):
        logfire.info(
            f"{platform} post unavailable for context; injecting unavailable notice", url=url
        )
        return unavailable_notice
    return conversation


async def build_post_context[OutputT: PlatformOutput, ConversationT: PlatformConversation[Any]](  # noqa: PLR0913 -- a source's reader and wording plus the per-call inputs every builder takes
    platform: str,
    url: str,
    reader: Callable[[], PostReader[ConversationT]],
    render: PostRenderer[OutputT, ConversationT],
    separators: PostSeparators,
    unavailable_notice: str,
    unreadable_notice: str,
    image_cap: int,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
    deadline: float,
) -> list[EasyInputMessageParam]:
    """Returns answer-model input blocks for a post URL, uploading up to `image_cap` images.

    Returns `[separator, user-content]` for a readable post, or a single notice marker when
    the post cannot be read. Never raises.

    Args:
        platform (str): Display name of the platform for the span and log messages.
        url (str): The post URL found in the conversation.
        reader (Callable[[], PostReader[ConversationT]]): Zero-argument factory for the platform
            downloader.
        render (PostRenderer[OutputT, ConversationT]): Renders the post and conversation text
            given the count of attached images.
        separators (PostSeparators): Opening markers and closing trailer for the platform.
        unavailable_notice (str): Notice returned when the platform reports the post is gone or
            private.
        unreadable_notice (str): Notice returned when the read fails without proving the post is
            gone.
        image_cap (int): Maximum number of post images to fetch and upload.
        answer_model_is_gemini (bool): Whether the answer model can resolve a Files API URI.
        gemini_client (genai.Client | None): Direct-to-Google client for Files API uploads, or
            `None` when unconfigured.
        allow_media_ingest (bool): Whether media ingestion is enabled for this source.
        deadline (float): Event-loop timestamp at which the pipeline cancels this build.
    """
    with logfire.span(f"gen_reply {platform.lower()} context"):
        conversation = await read_post(
            platform=platform,
            url=url,
            reader=reader,
            readable=lambda conversation: (
                conversation.target is not None and conversation.target.is_readable
            ),
            unavailable_notice=unavailable_notice,
            unreadable_notice=unreadable_notice,
        )
        if isinstance(conversation, str):
            return [marker_block(text=conversation)]
        if conversation.target is None:
            return [marker_block(text=unavailable_notice)]

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


# Derived from `MARKER_TAG_NAMES` and case-insensitive to match `extract_inline_markers`.
_MARKER_TAG_RE = re.compile(
    pattern=rf"</?({'|'.join(re.escape(pattern=name) for name in MARKER_TAG_NAMES)})>",
    flags=re.IGNORECASE,
)


def defuse_markers(text: str) -> str:
    """Returns `text` with inline pipeline marker tags neutralized into parenthesized names.

    Prevents quoted external text from triggering `extract_inline_markers` if echoed by the
    answer model.

    Args:
        text (str): Quoted post, author, or comment text to sanitize.
    """
    return _MARKER_TAG_RE.sub(repl=lambda match: f"({match.group(1)})", string=text)


def comment_lines[OutputT: LinkableCommentOutput](
    conversation: LinkableComments[OutputT], cap: int, served_as: str, handle_prefix: str
) -> list[str]:
    """Returns up to `cap` rendered comment lines under a count header, plus any linked comment.

    A comment matching `conversation.selected_comment_id` is marked in place rather than moved
    to the front so thread order holds, and is appended past `cap` when it sits deeper in the
    list.

    Args:
        conversation (LinkableComments[OutputT]): Parsed conversation holding the comments.
        cap (int): Maximum number of leading comments to include before the linked-comment check.
        served_as (str): Header clause stating how much of the discussion the page served.
        handle_prefix (str): Prefix placed before each author name, such as `"@"`.
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
        """Returns `self.timeout_notice` wrapped as a single marker block."""
        return [marker_block(text=self.timeout_notice)]
