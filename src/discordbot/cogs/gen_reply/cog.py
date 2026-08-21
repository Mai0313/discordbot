"""Cog that routes Discord messages through the AI reply pipeline."""

import re
import time
import base64
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
import asyncio
from functools import cached_property
import contextlib
from contextvars import ContextVar

from google import genai
from openai import AsyncOpenAI
import logfire
from nextcord import Embed, Message, NotFound, TextChannel, HTTPException, AllowedMentions
from pydantic import ValidationError
from nextcord.ext import commands
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.utils.douyin import DOUYIN_URL_RE, is_douyin_post_url
from discordbot.utils.images import convert_base64_to_data_uri
from discordbot.utils.threads import THREADS_URL_RE
from discordbot.utils.youtube import YOUTUBE_URL_RE
from discordbot.typings.colors import DISCORD_RED
from discordbot.typings.models import (
    EffortGrade,
    ModelSettings,
    RouteClassification,
    RuntimeModelCatalog,
)
from discordbot.utils.bilibili import BILIBILI_URL_RE
from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.utils.reactions import ReactionStatusChain, update_reaction
from discordbot.utils.usage_log import UsageRecorder
from discordbot.typings.timeouts import (
    EFFORT_GRACE_SECONDS,
    LINK_CONTEXT_GRACE_SECONDS,
    MEMORY_SELECT_GRACE_SECONDS,
    GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS,
)
from discordbot.utils.llm_errors import extract_friendly_error
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.llm_transcript import (
    USAGE_FOOTER_RE,
    sanitize_identity,
    render_author_identity,
    render_server_identity,
)
from discordbot.utils.media_delivery import (
    MediaItem,
    MediaDeliveryPlanner,
    upload_limit_for,
    build_media_delivery_planner,
)
from discordbot.services.memory.facts import render_owner_identity
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    read_tone,
    read_owner,
    user_scope,
    iter_scopes,
    server_scope,
    read_memory_document,
)
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.prompts import (
    IMAGE_PROMPT,
    REPLY_PROMPT,
    ROUTE_PROMPT,
    VIDEO_PROMPT,
    EFFORT_PROMPT,
    MUSIC_INSTRUCTION,
    VIDEO_INSTRUCTION,
    IMAGE_REPLY_PROMPT,
    VIDEO_REPLY_PROMPT,
    MEMORY_SELECT_PROMPT,
    INLINE_IMAGE_INSTRUCTION,
    DEEP_RESEARCH_INSTRUCTION,
    REQUEST_TIME_CONTEXT_PROMPT,
    REQUEST_LOCATION_CONTEXT_PROMPT,
)
from discordbot.cogs.gen_reply.files_api import upload_to_files_api
from discordbot.cogs.gen_reply.streaming import ResponseStreamer
from discordbot.services.memory.pipeline import (
    flavor_of,
    needs_consolidation,
    safe_list_resumable,
    resume_memory_update,
    consolidate_if_needed,
    schedule_memory_update,
)
from discordbot.cogs.gen_reply.generation import (
    MAX_VIDEO_REFERENCE_IMAGES,
    ImageGenerator,
    MusicGenerator,
    VideoGenerator,
    VoiceGenerator,
    PromptGenerator,
)
from discordbot.cogs.gen_reply.memory_tool import (
    NO_STORED_MEMORY,
    GET_USER_MEMORY_TOOL,
    UserMemory,
    MemoryCandidate,
    MemorySelection,
    MemoryReadContext,
    render_tone_block,
    parse_user_id_list,
    memory_read_context,
    memory_lookup_labels,
    resolve_user_memories,
    build_memory_allowlist,
    render_server_memory_block,
    render_callable_users_block,
    render_memory_context_block,
    widen_allowlist_with_aliases,
    allowlist_ids_from_server_memory,
)
from discordbot.services.memory.extraction import (
    MemoryExtractorAI,
    subject_source_line,
    target_centered_memory_messages,
)
from discordbot.cogs.gen_reply.capabilities import render_capabilities_block
from discordbot.cogs.gen_reply.interactions import (
    to_interactions_input,
    create_interactions_answer_stream,
)
from discordbot.cogs.gen_reply.link_sources import LinkContextSource
from discordbot.services.memory.git_history import memory_git
from discordbot.services.memory.server_prompts import (
    SERVER_PHASE1_PROMPT,
    SERVER_PHASE2_PROMPT,
    SERVER_PHASE1_EVALUATOR_PROMPT,
)
from discordbot.cogs.gen_reply.attachment.select import build_attachment_handler
from discordbot.cogs.gen_reply.link_sources.douyin import (
    build_douyin_context_messages,
    douyin_timeout_context_messages,
)
from discordbot.cogs.gen_reply.link_sources.threads import (
    build_threads_context_messages,
    threads_timeout_context_messages,
)
from discordbot.cogs.gen_reply.link_sources.bilibili import (
    build_bilibili_context_messages,
    bilibili_timeout_context_messages,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable, Coroutine, AsyncIterator

    from openai.types.responses import ResponseStreamEvent


# Preserve the existing eight-user context target for optional model-selected additions.
# Deterministic participants are never displaced: if they fill or exceed the target, the
# selector is skipped; otherwise it can use only the remaining slots.
MEMORY_CONTEXT_TARGET_USERS = 8

# Recorded as a reply's route when the pipeline failed before the router returned one.
UNROUTED_REPLY = "unrouted"

# How much channel history one answer reads, bounded on two axes because either alone is the
# wrong shape. Discord conversation here is overwhelmingly one-line messages — measured across
# 10M logged messages, the median is 6 characters and the busiest channel's last 200 come to
# 1.5k — so a message count alone lets a chatty channel hand the model almost nothing while a
# channel of long posts blows the input up. Whichever bound binds first wins.
HISTORY_MESSAGE_LIMIT = 500
HISTORY_CHAR_BUDGET = 8000

# What a history message costs beyond its own text: the rendered form carries an author header,
# and an attachment-only message has empty `content` but still renders a marker standing in for
# it. Without a floor per message a run of image posts would count as free and overshoot.
HISTORY_PER_MESSAGE_OVERHEAD = 40

# The model this turn most recently dispatched on, so `gen_reply failed` can name it: the failure
# surfaces in `on_message`, several frames above every place that picks a model, and a provider
# error rarely says which model it refused. A ContextVar rather than an attribute on the cog
# because many turns are in flight at once — nextcord dispatches each `on_message` as its own
# task, which copies the context, so a write here can never be read by another user's turn. Set
# only where the turn itself dispatches (route, answer, image, video); a generator that swallows
# its own failure logs its own model and never reaches the reader, and None means the turn failed
# before any model was asked for anything.
_dispatched_model: ContextVar[str | None] = ContextVar("gen_reply_dispatched_model", default=None)


def _message_link_texts(message: Message, strip_usage_footer: bool) -> list[str]:
    """The text spans a message actually renders to the model, for URL detection.

    Mirrors `get_cleaned_content` / `snapshot_text`: content takes precedence and an embed is
    rendered (and thus scanned) only when its content is empty. So a URL scanner never fires on a
    link the answer model was not shown, e.g. a captioned forwarded link card whose URL lives only
    in the embed. A forward puts its payload in `message.snapshots`, scanned via `snapshot_text`.

    `strip_usage_footer` removes the bot-authored footer from every span when a caller scans a
    reply-reference chain. The triggering message keeps its complete author-controlled text.
    """
    content = message.content or ""
    content_present = bool(content.strip())
    if strip_usage_footer:
        content = USAGE_FOOTER_RE.sub("", content)
    content = content.strip()
    texts = [content]
    if not content_present:
        texts.append(MessageInputBuilder.extract_embed_text(embeds=list(message.embeds)))
    for snapshot in message.snapshots:
        texts.append(MessageInputBuilder.snapshot_text(snapshot=snapshot))
    if strip_usage_footer:
        return [USAGE_FOOTER_RE.sub("", text).strip() for text in texts]
    return texts


def _authored_link_texts(message: Message) -> list[str]:
    """The text spans a message's author actually wrote, for scanning a message replied to.

    Narrower than `_message_link_texts` by exactly one thing: an embed card never counts,
    neither the message's own nor a forwarded snapshot's. One hop out an embed is a card the
    author did not write, and the bot's own Threads expansion is the common one:
    `parse_threads._build_embed_plan` emits one permalink per post in the reply chain, ROOT first,
    so a scan keyed on it would read the thread's top post rather than the one the human
    linked — and it disappears entirely when an oversize video pushes hosted URLs into
    `content`. A link a person typed always lives in `content` (or in the content of what they
    forwarded), so nothing human-written is lost. The bot's own replies pass through here too,
    so every span gets the `get_cleaned_content` / `snapshot_text` usage-footer strip: the
    footer carries the memory labels, which are display names their owners choose.
    """
    spans = [message.content or "", *(snapshot.content for snapshot in message.snapshots)]
    return [USAGE_FOOTER_RE.sub("", span).strip() for span in spans]


def _trim_history_to_budget(messages: list[Message]) -> list[Message]:
    """Keeps the newest history messages that fit `HISTORY_CHAR_BUDGET`, cut on a boundary.

    `_fetch_history` returns oldest-first, so this walks from the end and reverses back: what
    survives is the conversation closest to the question being answered, and the oldest context
    is what gets dropped. Cutting between messages rather than mid-text is the point — half a
    sentence with no author and no end reads as corrupted context rather than as less of it.

    The newest message is always kept even when it alone exceeds the budget, so a single long
    post can never reduce history to nothing.
    """
    kept: list[Message] = []
    spent = 0
    for candidate in reversed(messages):
        spent += len(candidate.content or "") + HISTORY_PER_MESSAGE_OVERHEAD
        if spent > HISTORY_CHAR_BUDGET and kept:
            break
        kept.append(candidate)
    kept.reverse()
    return kept


def _first_url_match(pattern: re.Pattern[str], texts: list[str]) -> re.Match[str] | None:
    """First match of a URL pattern across one message's already-rendered text spans."""
    for text in texts:
        match = pattern.search(string=text)
        if match:
            return match
    return None


def _link_url_for_source(source: LinkContextSource, message: Message) -> str | None:
    """The URL one link source should read: the current message's, else the replied-to one's.

    The current message always wins. A source that opts into `search_reference_chain` then
    falls back to the reply-reference chain, the same walk `_find_youtube_url` does, so
    "@bot 這篇底下在吵什麼" sent as a reply to someone else's link still reads the post; one
    that does not opt in never looks past the triggering message. The chain is scanned with
    `_authored_link_texts`, which is what keeps the bot's own expansion from triggering a read
    of the wrong post.

    A source's `url_filter` rejects a matched link it cannot read (e.g. a Douyin profile or
    live room, whose regex matches the host, not the path), which would only spend a
    rate-limited request to say so. It applies to the chosen match alone: a rejected link
    drops the source rather than sending the scan hunting for a second URL.
    """
    match = _first_url_match(
        pattern=source.url_pattern,
        texts=_message_link_texts(message=message, strip_usage_footer=False),
    )
    if match is None and source.search_reference_chain:
        for ref in _walk_reference_chain(message=message):
            match = _first_url_match(
                pattern=source.url_pattern, texts=_authored_link_texts(message=ref)
            )
            if match is not None:
                break
    if match is None:
        return None
    url = match.group(0)
    if source.url_filter is not None and not source.url_filter(url=url):
        return None
    return url


def _source_channel_is_public(message: Message) -> bool:
    """Whether @everyone can view the message's channel, so its content is not private.

    `message.channel` is a heterogeneous messageable union, so visibility is read
    defensively (mirrors `utils.discord_embeds`): a private thread is never public; a
    thread otherwise inherits its parent channel's `@everyone` visibility; a regular
    guild channel uses its own. A non-guild message, or any channel whose permissions
    cannot be resolved, counts as non-public — so content from channels members cannot
    see never enters the server-wide memory any member can read via `/memory server show`.
    """
    guild = message.guild
    if guild is None:
        return False
    channel = message.channel
    is_private = getattr(channel, "is_private", None)
    if callable(is_private) and is_private():
        return False
    source = getattr(channel, "parent", None) or channel
    permissions_for = getattr(source, "permissions_for", None)
    if not callable(permissions_for):
        return False
    return bool(getattr(permissions_for(guild.default_role), "view_channel", False))


def _build_runtime_instructions(system_prompt: str, message: Message) -> str:
    """Prepends per-request time and conversation-location context to the model instructions.

    The location line names the current guild (or DM) with developer authority so the
    model can reason about where it is speaking; the memory rules lean on it as the
    anchor for never attributing a remembered fact to another server.
    """
    message_created_at_asia_taipei = message.created_at.astimezone(tz=TAIWAN_TIMEZONE)
    request_time_context = REQUEST_TIME_CONTEXT_PROMPT.format(
        message_created_at_asia_taipei=message_created_at_asia_taipei.isoformat(timespec="seconds")
    ).strip()
    if message.guild is not None:
        # Deliberately id-only: the guild NAME is owner-controlled text and this block
        # rides the developer-authority `instructions` parameter, so embedding it would
        # hand a server owner an instruction-injection surface. The id anchors the
        # location just as well and cannot carry instructions.
        conversation_location = f"a Discord server (guild id {message.guild.id})"
    else:
        conversation_location = "a Discord direct message (DM)"
    request_location_context = REQUEST_LOCATION_CONTEXT_PROMPT.format(
        conversation_location=conversation_location
    ).strip()
    return f"{request_time_context}\n\n{request_location_context}\n\n{system_prompt}"


def _youtube_url_in_message(message: Message, strip_usage_footer: bool) -> str | None:
    """Returns the first YouTube URL in a message's text, embeds, or forwarded snapshots, if any."""
    match = _first_url_match(
        pattern=YOUTUBE_URL_RE,
        texts=_message_link_texts(message=message, strip_usage_footer=strip_usage_footer),
    )
    return match.group(0) if match else None


def _find_youtube_url(message: Message) -> str | None:
    """Finds a YouTube URL in the current message or the reply-reference chain.

    A reply to a message that merely links a video would otherwise be missed, so the chain is
    searched too and "summarize this" on a replied-to video still watches it. The current
    message wins, then the nearest reference outward. Threads reaches one hop the same way
    (`_link_url_for_source`, `search_reference_chain`); Douyin and Bilibili deliberately do
    not, since their value is the clip rather than a discussion and both are rate-limit
    sensitive. This one keeps scanning embeds out there — a YouTube link card is the link
    itself, not a rendering of some other post the way a Threads expansion is.
    """
    found = _youtube_url_in_message(message=message, strip_usage_footer=False)
    if found is not None:
        return found
    for ref in _walk_reference_chain(message=message):
        found = _youtube_url_in_message(message=ref, strip_usage_footer=True)
        if found is not None:
            return found
    return None


def _walk_reference_chain(message: Message) -> list[Message]:
    """Walks the reply-reference chain up to depth 3, oldest link last."""
    chain: list[Message] = []
    visited: set[int] = {message.id}
    current = message
    while (
        len(chain) < 3
        and current.reference
        and isinstance(current.reference.resolved, Message)
        and current.reference.resolved.id not in visited
    ):
        ref = current.reference.resolved
        visited.add(ref.id)
        chain.append(ref)
        current = ref
    return chain


def _reference_header(ref: Message, is_direct: bool) -> EasyInputMessageParam:
    """Builds the system separator that precedes one reference-chain message.

    `is_direct` marks the message the user is actually replying to (the immediate parent);
    older ancestors in the chain are labelled as thread context so only the real reply
    target reads as the primary context.
    """
    relation = (
        "The user is directly replying to this message; it is the primary context for the "
        "Current Message below."
        if is_direct
        else "An earlier message in the reply thread, for context."
    )
    return EasyInputMessageParam(
        role="system",
        content=[
            ResponseInputTextParam(
                text=(
                    f"==== Reference Message from {sanitize_identity(value=ref.author.display_name)} "
                    f"({sanitize_identity(value=ref.author.name)}) [id: {ref.author.id}]. {relation} ===="
                ),
                type="input_text",
            )
        ],
    )


def _current_header(message: Message, has_reference: bool) -> EasyInputMessageParam:
    """Builds the system separator that precedes the current message.

    When the message is a reply, the header points back to the Reference Message block
    (rendered just above) so the model reads the reply pair as one unit.
    """
    reply_note = " It is the user's reply to the Reference Message above." if has_reference else ""
    return EasyInputMessageParam(
        role="system",
        content=[
            ResponseInputTextParam(
                text=f"==== Current Message that needs to be answered from {sanitize_identity(value=message.author.display_name)} ({sanitize_identity(value=message.author.name)}) [id: {message.author.id}].{reply_note} ====",
                type="input_text",
            )
        ],
    )


async def _discard_task[TaskResultT](
    *, task: asyncio.Task[TaskResultT], label: str = "speculative", message_id: int | None = None
) -> None:
    """Cancels and drains a speculative task so its exception is retrieved.

    The except is deliberately broad: this drains unrelated subsystems (prep, effort, parts,
    memory selection), so anything they can raise must be swallowed here rather than surfacing on
    a route that already decided it does not need the result. `label` names which one failed,
    since the tasks are otherwise indistinguishable at this point. A link-context build is drained
    by `_drain_deadline_bound_task` instead, which must not steal its own deadline's cancellation.
    """
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logfire.warn(
            "Speculative reply context build failed off-route",
            task_label=label,
            error_type=type(exc).__name__,
            message_id=message_id,
            _exc_info=exc,
        )


async def _await_gated[GatedT](
    *, task: asyncio.Task[GatedT], label: str, route_done: asyncio.Event, grace_seconds: float
) -> GatedT:
    """Awaits a side task with a grace period beginning when routing finishes.

    A task started before routing completes overlaps it without consuming the grace. A task
    started after routing receives the grace immediately. The task is always cancelled on exit
    so it never orphans.
    """
    route_wait = asyncio.create_task(coro=route_done.wait())
    try:
        await asyncio.wait({task, route_wait}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return task.result()
        return await asyncio.wait_for(fut=task, timeout=grace_seconds)
    finally:
        route_wait.cancel()
        # `route_done.wait()` has no other terminal state, so nothing else is worth catching.
        with contextlib.suppress(asyncio.CancelledError):
            await route_wait
        if not task.done():
            await _discard_task(task=task, label=label)


async def _await_deadline_bound_task[DeadlineT](
    *, task: asyncio.Task[DeadlineT], deadline: float, label: str
) -> DeadlineT:
    """Awaits a self-deadline-bound task while preserving its cancellation cleanup ownership."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _drain_deadline_bound_task(task=task, deadline=deadline, label=label)
        raise


async def _drain_deadline_bound_task[DeadlineT](
    *, task: asyncio.Task[DeadlineT], deadline: float, label: str, message_id: int | None = None
) -> None:
    """Cancels before a task's deadline or preserves its in-progress deadline cleanup."""
    if not task.done() and asyncio.get_running_loop().time() < deadline:
        task.cancel()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                break
        except Exception:
            break
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logfire.warn(
            "Speculative reply context build failed off-route",
            task_label=label,
            error_type=type(exc).__name__,
            message_id=message_id,
            _exc_info=exc,
        )


async def _run_until_deadline[DeadlineT](
    *, awaitable: "Awaitable[DeadlineT]", deadline: float
) -> DeadlineT:
    """Runs a cancellation-propagating builder until its fixed event-loop deadline.

    Registered builders all propagate `CancelledError`, so `wait_for` alone owns the boundary.
    A clock check after this await would reject a pre-deadline result when a busy event loop only
    resumes this wrapper after the deadline.
    """
    event_loop = asyncio.get_running_loop()
    remaining_seconds = max(0.0, deadline - event_loop.time())
    return await asyncio.wait_for(fut=awaitable, timeout=remaining_seconds)


async def _build_threads_link_context(
    *,
    url: str,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
) -> list[EasyInputMessageParam]:
    """Adapts the Threads builder to the registry signature.

    Threads media ingestion has no kill-switch, so the flag is accepted and dropped. The
    builder name resolves from this module's globals at call time, which is what keeps a test
    monkeypatching `discordbot.cogs.gen_reply.cog.build_threads_context_messages` effective.
    """
    del allow_media_ingest
    return await build_threads_context_messages(
        url=url, answer_model_is_gemini=answer_model_is_gemini, gemini_client=gemini_client
    )


async def _build_douyin_link_context(
    *,
    url: str,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
) -> list[EasyInputMessageParam]:
    """Adapts the Douyin builder to the registry signature (a straight pass-through)."""
    return await build_douyin_context_messages(
        url=url,
        answer_model_is_gemini=answer_model_is_gemini,
        gemini_client=gemini_client,
        allow_media_ingest=allow_media_ingest,
    )


async def _build_bilibili_link_context(
    *,
    url: str,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
) -> list[EasyInputMessageParam]:
    """Adapts the Bilibili builder to the registry signature (a straight pass-through)."""
    return await build_bilibili_context_messages(
        url=url,
        answer_model_is_gemini=answer_model_is_gemini,
        gemini_client=gemini_client,
        allow_media_ingest=allow_media_ingest,
    )


def _threads_media_ingest_allowed(config: LLMConfig) -> bool:
    """Threads media ingestion has no kill-switch; the Gemini checks alone gate it."""
    del config
    return True


def _douyin_media_ingest_allowed(config: LLMConfig) -> bool:
    """The Douyin kill-switch plus the direct-Gemini key its Files API upload needs.

    `file_api_enabled` belongs here rather than only at the upload: the clip is downloaded
    first and the upload skipped after, so gating it there alone would still spend the whole
    fetch on a WAF-sensitive path for media that can no longer reach the model.
    """
    return (
        config.douyin_video_enabled
        and config.file_api_enabled
        and bool(config.gemini_api_key.strip())
    )


def _bilibili_media_ingest_allowed(config: LLMConfig) -> bool:
    """The Bilibili kill-switch plus the direct-Gemini key its Files API upload needs.

    Carries `file_api_enabled` for the reason the Douyin predicate does, minus the WAF: a
    30-minute video is downloaded in full before the upload it can no longer feed.
    """
    return (
        config.bilibili_video_enabled
        and config.file_api_enabled
        and bool(config.gemini_api_key.strip())
    )


# The linked-content sources gen_reply reads into answer context, in splice order: the blocks
# land in the answer input in this order, just before the current message. Adding a source is
# one entry here, its builder module, and its name in `RouteClassification.link_context_sources`
# (a source the router cannot name is never selected, so its builder never starts); the pipeline
# loops stay untouched.
LINK_CONTEXT_SOURCES: tuple[LinkContextSource, ...] = (
    LinkContextSource(
        name="threads",
        url_pattern=THREADS_URL_RE,
        # The one source that reads a link the user only replied to: what it fetches is the
        # discussion under the post, which the `parse_threads` expansion deliberately does not
        # show, so "@bot 這篇底下在吵什麼" on someone else's link has nothing else to answer from.
        search_reference_chain=True,
        build=_build_threads_link_context,
        on_timeout=threads_timeout_context_messages,
        media_ingest_allowed=_threads_media_ingest_allowed,
    ),
    LinkContextSource(
        name="douyin",
        url_pattern=DOUYIN_URL_RE,
        # The regex matches the host, not the path: a profile or live-room link is not a post,
        # so reading it would only spend a rate-limited Douyin request to say so.
        url_filter=is_douyin_post_url,
        build=_build_douyin_link_context,
        on_timeout=douyin_timeout_context_messages,
        media_ingest_allowed=_douyin_media_ingest_allowed,
    ),
    LinkContextSource(
        name="bilibili",
        # Path-anchored to the watchable /video/ forms (plus b23.tv short links), so unlike
        # Douyin no url_filter is needed on top.
        url_pattern=BILIBILI_URL_RE,
        build=_build_bilibili_link_context,
        on_timeout=bilibili_timeout_context_messages,
        media_ingest_allowed=_bilibili_media_ingest_allowed,
    ),
)


async def _discard_link_tasks(
    *,
    link_tasks: dict[str, "asyncio.Task[list[EasyInputMessageParam]]"],
    deadline: float | None,
    message_id: int,
) -> None:
    """Drains link builds without stealing cancellation from their shared deadline."""
    if link_tasks and deadline is None:
        raise RuntimeError("Selected link tasks have no route deadline")
    if deadline is None:
        return
    for name, task in link_tasks.items():
        await _drain_deadline_bound_task(
            task=task, deadline=deadline, label=name, message_id=message_id
        )
    link_tasks.clear()


def _log_pre_answer_latency(started: float, decision: str, message_id: int) -> None:
    """Logs total time from pipeline start to answer dispatch (the user's 'router stage')."""
    logfire.info(
        "gen_reply pre-answer latency",
        elapsed_seconds=time.monotonic() - started,
        decision=decision,
        message_id=message_id,
    )


def _count_media_parts(*, answer_input: ResponseInputParam) -> int:
    """Counts the media parts riding in an assembled answer request.

    An `input_file` (a Files API handle) or an `input_image` (inline base64) is the only shape a
    picture, clip or document reaches the model in, so this one number answers whether an
    attachment survived collection, the modality gate, the upload and the render.
    """
    total = 0
    for item in answer_input:
        content = cast("EasyInputMessageParam", item).get("content", "")
        if isinstance(content, str):
            continue
        total += sum(1 for part in content if part["type"] in ("input_file", "input_image"))
    return total


class _MessageLogFields(TypedDict):
    """Exact key set for `_message_log_fields`, so `**`-spreading it into a logfire call
    keeps statically known keys (none underscore-prefixed) and never collides with logfire's
    `_tags` / `_exc_info` keyword-only parameters.
    """

    user_id: int
    user_name: str
    display_name: str
    message_id: int
    channel_id: int
    guild_id: int | None
    guild_name: str | None


def _message_log_fields(message: Message) -> _MessageLogFields:
    """Standard Discord identifying fields for correlating one reply's logs.

    The pipeline-entry log carries the full set; every downstream log carries only
    `message_id` as the correlation key, so a whole turn reconstructs by grepping it.
    `user_name` is the stable handle, `display_name` the per-guild nickname;
    `guild_id` / `guild_name` are None in a DM.
    """
    guild = message.guild
    return {
        "user_id": message.author.id,
        "user_name": message.author.name,
        "display_name": message.author.display_name,
        "message_id": message.id,
        "channel_id": message.channel.id,
        "guild_id": guild.id if guild else None,
        "guild_name": guild.name if guild else None,
    }


class ReplyGeneratorCogs(commands.Cog):
    """Generates AI replies for Discord messages.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: The LLM client configuration loaded for reply generation.
        runtime_models: The model strings and per-tier settings every call here dispatches on.
        usage_recorder: The per-reply usage-record writer read by `scripts/usage_report.py`.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the ReplyGeneratorCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()
        self.runtime_models = RuntimeModelCatalog()
        self.usage_recorder = UsageRecorder()
        # Tracked background tasks for the one-shot restart memory resume.
        self._tasks: set[asyncio.Task[None]] = set()
        self._resume_started = False

    def _spawn(self, coro: "Coroutine[Any, Any, None]") -> None:
        """Runs `coro` as a tracked background task so the gateway never blocks on it."""
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @cached_property
    def openai_client(self) -> AsyncOpenAI:
        """The cached AsyncOpenAI client for all LiteLLM-proxy Responses / audio / image calls.

        Returns:
            A configured AsyncOpenAI client reused across reply requests.
        """
        return AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    @cached_property
    def gemini_client(self) -> genai.Client:
        """The cached native Gemini client for every DIRECT-to-Google runtime path.

        DIRECT to Google (`gemini_api_key`, no proxy): it serves the runtime paths the LiteLLM
        proxy cannot.

        - native omni video generation / editing (`interactions.create`, delivery=uri + Files
          download), for both the VIDEO route and the inline `<generate-video>` marker;
        - the inline `<generate-music>` Lyria render, also on the Interactions API;
        - Files API uploads, so a generated clip (and, through
          `gemini_client_if_configured`, a linked post's media) can be referenced by uri; and
        - the YouTube-aware QA answer turn that streams through the native Interactions API (the
          only path that can actually watch a linked video). That last swap only ever fires when
          the answer model is already Gemini, so the direct credential is always the right one.

        All of them forgo proxy-side cost/usage tracking, like the deep-research direct path. An
        empty key raises at construction, so a caller that is reachable without one must go
        through `gemini_client_if_configured` instead of touching this.

        Returns:
            A Gemini client for native media generation and the Interactions answer turn.
        """
        return genai.Client(api_key=self.config.gemini_api_key)

    @property
    def gemini_client_if_configured(self) -> genai.Client | None:
        """The direct Gemini client, or None when no key is configured.

        For the paths that stay useful without a key: a linked post still contributes its text,
        it just carries no uploaded media. Reading `gemini_client` there would raise before the
        feature's own kill-switch was ever consulted.

        Returns:
            The client, or None when `GEMINI_API_KEY` is unset.
        """
        if not self.config.gemini_api_key.strip():
            return None
        return self.gemini_client

    @cached_property
    def voice_generator(self) -> VoiceGenerator:
        """The cached text-to-speech engine for spoken QA replies.

        Returns:
            A generator bound to this cog's proxy client and the catalog's TTS model; the
            caller still gates it on `allow_voice` and `config.inline_voice_enabled`.
        """
        return VoiceGenerator(
            client=self.openai_client, model_name=self.runtime_models.tts_model.deployment_name
        )

    @cached_property
    def image_generator(self) -> ImageGenerator:
        """The cached image renderer shared by the IMAGE route and the QA-route `<generate-image>` marker.

        Returns:
            A generator bound to this cog's proxy client and the image model; the route calls
            `render` (raises) while the inline path calls `generate` (best-effort, gated on
            `allow_image` and `config.inline_image_enabled`).
        """
        return ImageGenerator(
            client=self.openai_client, image_model=self.runtime_models.image_model
        )

    @cached_property
    def prompt_generator(self) -> PromptGenerator:
        """The cached prompt director for the IMAGE and VIDEO routes.

        Returns:
            A director bound to this cog's proxy client and the grounding-capable
            `fast_model`; each `refine` call is gated by the caller's per-route flag
            (`config.image_refine_prompt_enabled` / `config.video_refine_prompt_enabled`) and
            expands the raw request before `render`, best-effort (raw prompt on disable / empty /
            error).
        """
        return PromptGenerator(
            client=self.openai_client, prompt_model=self.runtime_models.fast_model
        )

    @cached_property
    def video_generator(self) -> VideoGenerator:
        """The cached video renderer shared by the VIDEO route and the QA-route `<generate-video>` marker.

        Returns:
            A generator bound to this cog's DIRECT-to-Google Gemini client and the video model
            (the Interactions API is Gemini-only, not reachable via the proxy); the route calls
            `render` (raises) while the inline path calls `generate` (best-effort, gated on
            `allow_video` and `config.video_available`).
        """
        return VideoGenerator(
            client=self.gemini_client, video_model=self.runtime_models.video_model
        )

    @cached_property
    def music_generator(self) -> MusicGenerator:
        """The cached music renderer for the QA-route `<generate-music>` marker.

        Returns:
            A generator bound to this cog's DIRECT-to-Google Gemini client (Lyria runs on the
            Interactions API, not the proxy) and the music model; the inline path calls
            `generate` (best-effort, gated on `allow_music` and `config.music_available`).
        """
        return MusicGenerator(
            client=self.gemini_client, music_model=self.runtime_models.music_model
        )

    @cached_property
    def media_delivery(self) -> MediaDeliveryPlanner:
        """The cached media-delivery planner shared by the IMAGE / VIDEO routes and QA streamer.

        Returns:
            A planner that decides which media attach natively and which are hosted as a public
            URL (media too big for Discord's upload limit); its host self-disables when
            unconfigured, so every oversize item then degrades to the route's host-free path.
        """
        return build_media_delivery_planner()

    @cached_property
    def memory_extractor(self) -> MemoryExtractorAI:
        """The cached per-user memory extraction service.

        Returns:
            An extractor bound to this cog's client and the phase-1/phase-2
            memory models.
        """
        return MemoryExtractorAI(
            client=self.openai_client,
            extract_model=self.runtime_models.memory_extractor_model,
            evaluate_model=self.runtime_models.memory_writer_model,
            consolidate_model=self.runtime_models.memory_writer_model,
        )

    @cached_property
    def server_memory_extractor(self) -> MemoryExtractorAI:
        """The cached per-server (bot self) memory extraction service.

        Returns:
            An extractor sharing the per-user models and client but driving the
            server-flavor prompts, so the bot builds community-level memory per
            guild through the same engine.
        """
        return MemoryExtractorAI(
            client=self.openai_client,
            extract_model=self.runtime_models.memory_extractor_model,
            evaluate_model=self.runtime_models.memory_writer_model,
            consolidate_model=self.runtime_models.memory_writer_model,
            phase1_prompt=SERVER_PHASE1_PROMPT,
            evaluator_prompt=SERVER_PHASE1_EVALUATOR_PROMPT,
            consolidate_prompt=SERVER_PHASE2_PROMPT,
        )

    @cached_property
    def input_builder(self) -> MessageInputBuilder:
        """The cached Discord-message-to-Responses-API input builder.

        Returns:
            A builder bound to this bot, runtime model catalog, and the attachment
            handler matching the answer model's provider.
        """
        return MessageInputBuilder(
            bot=self.bot,
            runtime_models=self.runtime_models,
            attachment_handler=build_attachment_handler(
                model_name=self.runtime_models.slow_model.name
            ),
        )

    async def _fetch_history(self, message: Message, limit: int) -> list[Message]:
        """Fetches up to `limit` channel-history messages once, trimmed to the char budget.

        Returned raw so both the optional selector's text-only render and the answer's
        uploaded render derive from one fetch, without a second walk of history.
        """
        hist_messages: list[Message] = []
        async for m in message.channel.history(limit=limit, before=message, oldest_first=True):
            hist_messages.append(m)
        return _trim_history_to_budget(messages=hist_messages)

    async def _render_history(
        self, hist_messages: list[Message], *, text_only: bool, message_id: int
    ) -> list[EasyInputMessageParam]:
        """Renders fetched history in one mode: text-only markers, or full uploaded parts.

        Both modes derive from the same `_fetch_history` result, so history is walked once
        however many renders are asked for. The text-only twin (no upload) feeds optional
        memory selection without waiting on the Files API; the full render uploads attachment
        parts for the answer. History is the only render that opts into the dead-source skip:
        an expired CDN attachment here re-fails every turn (current / reference do not; see
        GeminiFileUploader._resolve_file_upload).
        """
        if not hist_messages:
            return []
        tasks: list[Awaitable[EasyInputMessageParam]] = [
            self.input_builder.process_single_message_text_only(message=m)
            if text_only
            else self.input_builder.process_single_message(message=m, allow_dead_cache=True)
            for m in hist_messages
        ]
        started = time.monotonic()
        processed = await asyncio.gather(*tasks)
        if not text_only:
            logfire.info(
                "gen_reply history render done",
                elapsed_seconds=time.monotonic() - started,
                message_count=len(hist_messages),
                message_id=message_id,
            )
        header = EasyInputMessageParam(
            role="system",
            content=[
                ResponseInputTextParam(
                    text="==== Chat History that might be helpful for answering. ====",
                    type="input_text",
                )
            ],
        )
        return [header, *processed]

    async def _get_reference_message(
        self, message: Message, text_only: bool = False
    ) -> list[EasyInputMessageParam]:
        """Walks the reference chain up to depth 3 and renders each link as context.

        `text_only` emits attachment markers instead of uploaded file parts, for the
        route and memory-selection calls that must not wait on the Files API.
        """
        chain = _walk_reference_chain(message=message)
        if not chain:
            return []

        tasks: list[Awaitable[EasyInputMessageParam]] = []
        for ref in chain:
            if text_only:
                tasks.append(self.input_builder.process_single_message_text_only(message=ref))
            else:
                tasks.append(self.input_builder.process_single_message(message=ref))
        processed: list[EasyInputMessageParam] = await asyncio.gather(*tasks)

        messages: list[EasyInputMessageParam] = []
        for ref, processed_ref in zip(reversed(chain), reversed(processed), strict=True):
            messages.append(_reference_header(ref=ref, is_direct=ref is chain[0]))
            messages.append(processed_ref)
        return messages

    async def _get_current_message(
        self, message: Message, text_only: bool = False
    ) -> list[EasyInputMessageParam]:
        """Processes the current message that needs to be answered."""
        has_reference = bool(_walk_reference_chain(message=message))
        messages: list[EasyInputMessageParam] = [
            _current_header(message=message, has_reference=has_reference)
        ]
        if text_only:
            current_msg = await self.input_builder.process_single_message_text_only(
                message=message
            )
        else:
            current_msg = await self.input_builder.process_single_message(message=message)
        messages.append(current_msg)
        return messages

    async def _deliver_generated_media(
        self, *, message: Message, data: bytes, filename: str
    ) -> Message | None:
        """Delivers generated image/video bytes, hosting a URL when too big to upload natively.

        Returns the delivered media message the persona reply should stream onto, or None when the
        bytes were too big and hosted as a standalone URL reply instead. On None the caller posts
        the persona reply on a fresh non-pinging message (via `_persona_base_reply`) only if it
        proceeds, so the hosted-URL message is never clobbered and no stray persona-base is left if
        the persona reply bails. If hosting is unavailable the native attach is attempted anyway,
        raising on oversize so the route stays on its existing hard-fail error path.
        """
        item = MediaItem(source=data, filename=filename)
        plan = await self.media_delivery.plan(
            items=[item], upload_limit=upload_limit_for(guild=message.guild)
        )
        if plan.native:
            return await message.reply(
                content=message.author.mention, file=plan.native[0].to_file()
            )
        if not plan.hosted_urls:
            # Hosting off/failed: attempt the native attach, which raises on oversize and keeps
            # the route on the outer error path exactly as before.
            return await message.reply(content=message.author.mention, file=item.to_file())
        # Too big to attach: the hosted URL is the deliverable (pings the author once). The persona
        # reply, if it runs, streams onto its own fresh message so it never clobbers this link.
        await message.reply(content=f"{message.author.mention}\n{plan.hosted_urls[0]}")
        return None

    async def _persona_base_reply(self, *, message: Message, reply: Message | None) -> Message:
        """The message the persona stream edits: the delivered media message, or a fresh reply.

        When the media rode as a native attachment, that same message is reused (its content edits
        keep the attachment). When the media was hosted as a separate URL (`reply is None`), a fresh
        non-pinging reply is created here, lazily, only when the persona reply actually proceeds —
        so the hosted-URL message keeps the sole author ping and no empty message is ever orphaned.
        """
        if reply is not None:
            return reply
        return await message.reply(
            content=message.author.mention, allowed_mentions=AllowedMentions.none()
        )

    async def _handle_video_reply(
        self, message: Message, user_prompt: str, context_task: "asyncio.Task[ReplyContext]"
    ) -> None:
        """Generates a video via the native Gemini (omni) Interactions API, delivers it, then replies.

        Runs direct to Google via `interactions.create`. If the message (or the replied-to message,
        mirroring the IMAGE route) carries a video, omni edits that actual clip in place
        (`task="edit"`, the literal request as the edit instruction, no prompt director); otherwise
        the request is expanded by the prompt director and any images ride as subject reference
        frames (up to `MAX_VIDEO_REFERENCE_IMAGES`). The clip is delivered first; then, best-effort,
        the bot watches the video it just made (uploaded to the Gemini Files API) and streams a
        persona reply onto the same message, mirroring `_handle_image_reply` and consuming the
        speculative `ReplyContext` (history + the requester's memory) only after the video is on
        screen so its build overlaps generation.
        """
        started = time.monotonic()
        logfire.info(
            "gen_reply video generation start",
            message_id=message.id,
            model=self.runtime_models.video_model.name,
        )
        try:
            source_messages = [message]
            if message.reference and isinstance(message.reference.resolved, Message):
                source_messages.append(message.reference.resolved)
            # Find the source video first, by priority (current message, then replied-to); each
            # message reads at most its first clip. Only when there is no source video do we
            # download reference images, so an edit is never delayed by media it discards.
            source_video: tuple[bytes, str] | None = None
            for source_message in source_messages:
                videos = await self.input_builder.get_video_sources(message=source_message)
                if videos:
                    source_video = videos[0]
                    break
            # Both branches end in the same omni render, and the director the else branch runs
            # first is best-effort, so this is the model a failure past here belongs to.
            _dispatched_model.set(self.runtime_models.video_model.name)
            if source_video is not None:
                # A source video is edited in place (task=edit): omni ingests the actual clip, so
                # the prompt is the literal edit instruction. The director is skipped here — it
                # only grounds on image parts (a video-only edit would run it blind) and it sits
                # serially on the time-to-video path; the user's edit request is already specific.
                # omni takes a single input here, so any accompanying reference images are dropped.
                video_bytes = await self.video_generator.render(
                    prompt=user_prompt, reference_image_sources=[], source_video=source_video
                )
            else:
                # No source video: gather the message + replied-to images as subject references,
                # capped to the same set render sends (omni takes a few), so the director grounds
                # on exactly those frames and no unused bytes ride the path.
                image_groups = await asyncio.gather(
                    *(
                        self.input_builder.get_image_sources_with_mime(message=m)
                        for m in source_messages
                    )
                )
                images = [pair for group in image_groups for pair in group][
                    :MAX_VIDEO_REFERENCE_IMAGES
                ]
                # Refine the raw request into a full motion/camera prompt first (best-effort, raw
                # prompt on disable / failure); the reference frames ride along as grounding.
                refined_prompt = await self.prompt_generator.refine(
                    user_prompt=user_prompt,
                    instructions=VIDEO_PROMPT,
                    end_user_id=message.author.name,
                    enabled=self.config.video_refine_prompt_enabled,
                    image_bytes_list=[raw for raw, _ in images] or None,
                )
                video_bytes = await self.video_generator.render(
                    prompt=refined_prompt, reference_image_sources=images
                )
            reply = await self._deliver_generated_media(
                message=message, data=video_bytes, filename="generated.mp4"
            )
            logfire.info(
                "gen_reply video delivered",
                message_id=message.id,
                model=self.runtime_models.video_model.name,
                total_elapsed_seconds=time.monotonic() - started,
                bytes=len(video_bytes),
            )
        except Exception:
            # Generation failing IS a real error and stays on the outer error path, but the
            # speculative context must not leak when we bail before consuming it.
            await _discard_task(task=context_task, label="prep", message_id=message.id)
            raise

        # The video is already delivered, so from here a failure must never surface as an error:
        # the conversational reply is best-effort and leaves the delivered video untouched.
        await self._reply_about_video(
            message=message, reply=reply, video_bytes=video_bytes, context_task=context_task
        )

    async def _upload_video_for_reply(self, data: bytes) -> str | None:
        """Uploads a generated video to the Gemini Files API, polling to ACTIVE; None on failure.

        The bound is generous because video processing is slower than an image's. The reply
        then references the full `uri` through the proxy; see `files_api` for why a uri and
        not the clip's own URL.
        """
        return await upload_to_files_api(
            client=self.gemini_client,
            source=data,
            mime_type="video/mp4",
            display_name="generated.mp4",
            timeout_seconds=GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS,
        )

    async def _reply_about_video(
        self,
        message: Message,
        reply: Message | None,
        video_bytes: bytes,
        context_task: "asyncio.Task[ReplyContext]",
    ) -> None:
        """Best-effort: watches the just-made video and streams a persona reply onto its message.

        Feeds the generated video as an uploaded Files API `input_file` (video cannot be
        inlined), then delegates to the shared media-persona-reply streamer. `reply` is None when
        the clip was hosted as a URL; the persona-base message is only created once the Files API
        upload succeeds, so a failed upload leaves no orphaned message. Any failure leaves the
        delivered video untouched.
        """
        file_uri = await self._upload_video_for_reply(data=video_bytes)
        if file_uri is None:
            await _discard_task(task=context_task, label="prep", message_id=message.id)
            return
        await self._stream_media_persona_reply(
            message=message,
            reply=reply,
            context_task=context_task,
            model=self.runtime_models.fast_model,
            system_prompt=VIDEO_REPLY_PROMPT,
            focus_part=ResponseInputFileParam(type="input_file", file_id=file_uri),
            media_noun="video",
            span_name="gen_reply video reply",
        )

    async def _stream_media_persona_reply(  # noqa: PLR0913 -- shared by IMAGE/VIDEO; the prompt / focus part / noun / span differ per route
        self,
        *,
        message: Message,
        reply: Message | None,
        context_task: "asyncio.Task[ReplyContext]",
        model: ModelSettings,
        system_prompt: str,
        focus_part: ResponseInputFileParam | ResponseInputImageParam,
        media_noun: str,
        span_name: str,
    ) -> None:
        """Best-effort: streams a persona reply onto an already-delivered generated image/video.

        Shared by the IMAGE and VIDEO routes' post-delivery reply. `reply` is the delivered media
        message (native attachment) or None when the media was hosted as a separate URL; the
        persona-base message is built from it INSIDE the protected flow (`_persona_base_reply`), so a
        base-creation or streaming failure is swallowed here instead of surfacing to the outer error
        path, and a fresh hosted-case base that never received content is deleted (never an orphan).
        Builds the answer-path input (history, selected user memory, tone note, reference, current),
        appends the just-made media as the focus, and streams onto the base (its content edits keep an
        attached media). Injects only the selected user memory (already read through the
        compartments this conversation may open) plus the author's tone note, never the server
        memory block, and seeds the
        selection-call usage / memory labels so the footer matches the QA path. Consumes the
        speculative `context_task` (awaited here so its build overlaps generation); any failure
        leaves the delivered media untouched.
        """
        base: Message | None = None
        streamer: ResponseStreamer | None = None
        try:
            context = await context_task
            base = await self._persona_base_reply(message=message, reply=reply)
            # Mirror the answer path's order (history, memory, tone, reference, current),
            # injecting only the selected user memory (already compartment-scoped by
            # `resolve_user_memories`) and the author's tone note, never the server memory block.
            response_input: ResponseInputParam = [*context.hist_messages]
            response_input.extend(
                block for block in (context.memory_block, context.tone_block) if block is not None
            )
            response_input.extend(context.reference_messages)
            response_input.extend(context.current_message)
            # The generated media is the focus, appended last right after the request it answers.
            response_input.append(
                EasyInputMessageParam(
                    role="user",
                    content=[
                        ResponseInputTextParam(
                            text=(
                                f"This is the {media_noun} you just made for them in response "
                                "to the request above. Reply to them about it."
                            ),
                            type="input_text",
                        ),
                        focus_part,
                    ],
                )
            )
            streamer = ResponseStreamer(
                message=message,
                reply=base,
                memory_lookups=context.memory_labels,
                input_tokens=context.selection_input_tokens,
                output_tokens=context.selection_output_tokens,
                model_effort=model.effort or "",
            )
            with logfire.span(span_name, model=model.name, message_id=message.id):
                responses = await self.openai_client.responses.create(
                    model=model.deployment_name,
                    instructions=_build_runtime_instructions(
                        system_prompt=system_prompt, message=message
                    ),
                    input=response_input,
                    reasoning=model.reasoning,
                    stream=True,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": message.author.name},
                )
                await streamer.stream(responses=responses)
        except Exception as exc:
            logfire.warn(
                "Media persona reply failed; leaving the delivered media without a reply",
                media=media_noun,
                message_id=message.id,
                model=model.name,
                error_type=type(exc).__name__,
                _exc_info=True,
            )
            # A fresh hosted-case base (reply was None) that never received content is a bare ping;
            # delete it so a failed persona reply leaves no orphan. A native media message
            # (reply is not None) is the deliverable itself and is always kept.
            if (
                reply is None
                and base is not None
                and (streamer is None or not streamer.content_started)
            ):
                with contextlib.suppress(Exception):
                    await base.delete()

    async def _handle_image_reply(
        self, message: Message, user_prompt: str, context_task: "asyncio.Task[ReplyContext]"
    ) -> None:
        """Generates or edits an image, then replies about it in persona.

        The image is delivered first so the user sees it without waiting; the conversational
        reply then streams onto that same message, so the bot answers while holding the image
        it just made rather than coldly describing it. The reply brings in conversation
        history and the selected user memory (never server memory), using the speculative
        `context_task` that built in parallel with the route, awaited only after
        the image is on screen so the context build overlaps generation. Once the image is
        delivered the reply is best-effort: any failure leaves the delivered image untouched.
        """
        started = time.monotonic()
        logfire.info(
            "gen_reply image generation start",
            message_id=message.id,
            model=self.runtime_models.image_model.name,
            has_source_images=bool(
                message.reference and isinstance(message.reference.resolved, Message)
            ),
        )
        try:
            if message.reference and isinstance(message.reference.resolved, Message):
                own_bytes, ref_bytes = await asyncio.gather(
                    self.input_builder.get_image_source_bytes(message=message),
                    self.input_builder.get_image_source_bytes(message=message.reference.resolved),
                )
                image_bytes_list = own_bytes + ref_bytes
            else:
                image_bytes_list = await self.input_builder.get_image_source_bytes(message=message)

            # Refine the raw request into a full generation/edit prompt first (best-effort, raw
            # prompt on disable / failure); the source bytes ride along so an edit prompt is
            # grounded in the actual image without a re-download.
            refined_prompt = await self.prompt_generator.refine(
                user_prompt=user_prompt,
                instructions=IMAGE_PROMPT,
                end_user_id=message.author.name,
                enabled=self.config.image_refine_prompt_enabled,
                image_bytes_list=image_bytes_list or None,
            )
            # The director above is best-effort and swallows its own failures, so from here the
            # image model is the only one a failure can be reported against.
            _dispatched_model.set(self.runtime_models.image_model.name)
            image_bytes = await self.image_generator.render(
                prompt=refined_prompt,
                end_user_id=message.author.name,
                image_bytes_list=image_bytes_list or None,
            )
            # Send the generated image immediately so the user sees it without waiting on the
            # conversational reply; the reply text streams onto this same message right after.
            reply = await self._deliver_generated_media(
                message=message, data=image_bytes, filename="generated.png"
            )
            logfire.info(
                "gen_reply image delivered",
                message_id=message.id,
                model=self.runtime_models.image_model.name,
                elapsed_seconds=time.monotonic() - started,
            )
        except Exception:
            # Generation failing IS a real error and stays on the outer error path, but the
            # speculative context must not leak when we bail before consuming it.
            await _discard_task(task=context_task, label="prep", message_id=message.id)
            raise

        # The image is already delivered, so from here a failure must never surface as an
        # error: the conversational reply is best-effort and leaves the image untouched. The
        # image rides as inline base64 (provider-agnostic), unlike the video's Files API handle.
        await self._stream_media_persona_reply(
            message=message,
            reply=reply,
            context_task=context_task,
            model=self.runtime_models.fast_model,
            system_prompt=IMAGE_REPLY_PROMPT,
            focus_part=ResponseInputImageParam(
                image_url=convert_base64_to_data_uri(
                    base64_image=base64.b64encode(image_bytes).decode()
                ),
                detail="auto",
                type="input_image",
            ),
            media_noun="image",
            span_name="gen_reply image reply",
        )

    async def _get_reference_and_current(
        self, message: Message, text_only: bool = False
    ) -> tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]]:
        """Renders the reference chain and the current message together.

        With `text_only` they render as attachment markers (no upload) for the route and memory
        selection; otherwise this is the answer-path render (uploads + activation poll to ACTIVE)
        that runs in the background so only the answer awaits the Files API. The render-timing log
        fires only for the upload-bearing render, the latency-critical one.
        """
        started = time.monotonic()
        reference_messages, current_message = await asyncio.gather(
            self._get_reference_message(message=message, text_only=text_only),
            self._get_current_message(message=message, text_only=text_only),
        )
        if not text_only:
            logfire.info(
                "gen_reply attachment render done",
                elapsed_seconds=time.monotonic() - started,
                reference_count=len(reference_messages),
                current_count=len(current_message),
                message_id=message.id,
            )
        return reference_messages, current_message

    async def _route_classify(
        self,
        message: Message,
        reference_messages: list[EasyInputMessageParam],
        current_message: list[EasyInputMessageParam],
    ) -> RouteClassification:
        """Classifies the message into a reply mode using pre-built context parts.

        The handler choice and the two content-read decisions that ride with it
        (`watch_video`, `link_context_sources`) all come from this one call; the answer
        effort is graded by `_grade_effort` in a parallel call, so this stays short on the
        critical path. The reference + current parts arrive already text-only (attachment
        markers, no file ids), so the route classifies on the text without reading or
        waiting on uploads.
        """
        message_list = [*reference_messages, *current_message]

        triage_model = self.runtime_models.triage_model
        _dispatched_model.set(triage_model.name)
        started = time.monotonic()
        try:
            with logfire.span("gen_reply route", message_id=message.id):
                responses = await self.openai_client.responses.parse(
                    model=triage_model.deployment_name,
                    instructions=ROUTE_PROMPT,
                    input=cast("ResponseInputParam", message_list),
                    text_format=RouteClassification,
                    reasoning=triage_model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": message.author.name},
                )
            parsed = responses.output_parsed
            route = parsed if parsed is not None else RouteClassification(decision="QA")
        except ValidationError as exc:
            # `responses.parse` validates before `output_parsed` is reachable, so an empty /
            # safety-filtered response and a genuine schema mismatch both land here; the
            # attached exception is the only way to tell them apart.
            logfire.warn(
                "RouteClassification parse failed; defaulting to QA",
                message_id=message.id,
                model=triage_model.name,
                _exc_info=exc,
            )
            route = RouteClassification(decision="QA")
        # Route-call latency is logged on every path: this is the prime suspect for slow
        # replies, so the log file must show its duration directly, not just a span start.
        logfire.info(
            "gen_reply route done",
            elapsed_seconds=time.monotonic() - started,
            model=triage_model.name,
            decision=route.decision,
            link_context_sources=route.link_context_sources,
            watch_video=route.watch_video,
            message_id=message.id,
        )
        return route

    async def _grade_effort(
        self,
        message: Message,
        reference_messages: list[EasyInputMessageParam],
        current_message: list[EasyInputMessageParam],
    ) -> EffortGrade:
        """Grades how much reasoning effort the answer model should spend on this message.

        Runs in parallel with the route under the shared `route_done` gate (`_await_gated`);
        the grade is consumed only on the QA path, while IMAGE and VIDEO cancel
        this task. The parts arrive already text-only, so grading never waits on uploads.
        Raises on any provider/parse failure so the caller (`_resolve_effort`) can fall back.

        Every message is graded by the model, including one carrying an attachment or a link:
        #491's code-decided "high" for those was measured against the live grader in #493 and
        bought nothing the prompt does not already deliver (19-20 of 20 on a screenshot, a thin
        caption over an image, a bare URL and a casual line beside a link), while costing the
        one case it reads wrong, a sticker-only reaction. What that case actually needed was for
        the marker to name a sticker instead of calling it an image (`render_text_only`).
        """
        message_list = [*reference_messages, *current_message]

        triage_model = self.runtime_models.triage_model
        started = time.monotonic()
        with logfire.span("gen_reply effort", message_id=message.id):
            responses = await self.openai_client.responses.parse(
                model=triage_model.deployment_name,
                instructions=EFFORT_PROMPT,
                input=cast("ResponseInputParam", message_list),
                text_format=EffortGrade,
                reasoning=triage_model.reasoning,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )
        parsed = responses.output_parsed
        grade = parsed if parsed is not None else EffortGrade(effort="high")
        logfire.info(
            "gen_reply effort done",
            elapsed_seconds=time.monotonic() - started,
            model=triage_model.name,
            effort=grade.effort,
            message_id=message.id,
        )
        return grade

    async def _resolve_effort(
        self,
        *,
        message: Message,
        effort_task: "asyncio.Task[EffortGrade]",
        route_done: asyncio.Event,
    ) -> Literal["low", "high"]:
        """Resolves the parallel effort grade, bounded by the route like memory selection.

        Falls back to "high" on the post-route grace timeout or any grading error, so a slow
        or failed effort call never stalls or silently degrades the reply.
        """
        try:
            grade = await _await_gated(
                task=effort_task,
                label="effort",
                route_done=route_done,
                grace_seconds=EFFORT_GRACE_SECONDS,
            )
        except TimeoutError as exc:
            logfire.warn(
                "Effort grading exceeded the post-route grace; defaulting to high effort",
                grace_seconds=EFFORT_GRACE_SECONDS,
                message_id=message.id,
                model=self.runtime_models.triage_model.name,
                _exc_info=exc,
            )
            return "high"
        except Exception as e:
            logfire.warn(
                "Effort grading failed; defaulting to high effort",
                message_id=message.id,
                model=self.runtime_models.triage_model.name,
                error_type=type(e).__name__,
                _exc_info=True,
            )
            return "high"
        return grade.effort

    async def _resolve_link_block(
        self,
        *,
        message: Message,
        source: str,
        link_task: "asyncio.Task[list[EasyInputMessageParam]]",
        deadline: float,
        on_timeout: "Callable[[], list[EasyInputMessageParam]]",
    ) -> list[EasyInputMessageParam]:
        """Resolves an intent-selected linked-post build before the shared route deadline.

        Each selected builder owns the deadline fixed when routing finishes. This resolver only
        retrieves that task, so the builder's cancellation cleanup cannot be interrupted by a
        second timeout. On expiry it injects a short "could not read it in time" notice instead
        of nothing; on any other unexpected error it returns [] (cancellation propagates). The
        builders themselves never raise (they degrade to their own notices).
        """
        started = time.monotonic()
        try:
            blocks = await _await_deadline_bound_task(
                task=link_task, deadline=deadline, label=source
            )
        except TimeoutError as exc:
            logfire.warn(
                "Linked-post context exceeded the post-route grace; injecting timeout notice",
                source=source,
                grace_seconds=LINK_CONTEXT_GRACE_SECONDS,
                message_id=message.id,
                _exc_info=exc,
            )
            return on_timeout()
        except Exception as exc:
            # Broad on purpose: the builders are documented never to raise, so anything landing
            # here is unexpected (a builder bug, or a fetch/WAF failure that escaped its own
            # notice); error_type is what tells those apart in logs. CancelledError is a
            # BaseException and deliberately propagates instead of being swallowed as "no link".
            logfire.warn(
                "Linked-post context failed; answering without it",
                source=source,
                message_id=message.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return []
        logfire.info(
            "gen_reply link context done",
            source=source,
            elapsed_seconds=time.monotonic() - started,
            blocks=len(blocks),
            message_id=message.id,
        )
        return blocks

    async def _select_user_memories(
        self,
        *,
        message: Message,
        message_list: list[EasyInputMessageParam],
        allowed: dict[int, MemoryCandidate],
        read_context: MemoryReadContext,
        server_memory_block: EasyInputMessageParam | None = None,
    ) -> MemorySelection:
        """Lets the model choose optional third-party memories for an oblique reference.

        Runs an isolated request offering only the get_user_memory tool, then resolves the
        chosen ids server-side against an allowlist containing only absent members from a
        public server nickname table. The server memory rides in front as background context
        so a spoken or misspelled nickname can be mapped to its id. Returns the memories plus
        this request's token usage so the reply footer and chat reward account for the call.
        """
        triage_model = self.runtime_models.triage_model
        # The optional-candidates block stays last so the model reads it right before deciding;
        # the server-memory block (if any) leads as earlier background context. The caller
        # passes an already text-only transcript (attachment markers, no file ids), so this
        # request neither re-reads the uploaded payloads nor waits on their upload.
        selection_input: ResponseInputParam = [
            *([server_memory_block] if server_memory_block is not None else []),
            *message_list,
            render_callable_users_block(allowed=allowed),
        ]
        responses = await self.openai_client.responses.create(
            model=triage_model.deployment_name,
            instructions=MEMORY_SELECT_PROMPT,
            input=selection_input,
            reasoning=triage_model.reasoning,
            tools=[GET_USER_MEMORY_TOOL],
            stream=False,
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": message.author.name},
        )
        memories: list[UserMemory] = []
        seen: set[str] = set()
        for item in responses.output:
            if item.type != "function_call":
                continue
            if item.name != "get_user_memory":
                continue
            for memory in resolve_user_memories(
                user_id_list=parse_user_id_list(arguments=item.arguments),
                allowed=allowed,
                context=read_context,
            ):
                if memory.user_id not in seen:
                    seen.add(memory.user_id)
                    memories.append(memory)
        input_tokens = responses.usage.input_tokens if responses.usage else 0
        output_tokens = responses.usage.output_tokens if responses.usage else 0
        return MemorySelection(
            memories=memories, input_tokens=input_tokens, output_tokens=output_tokens
        )

    def _read_server_memory(self, *, message: Message) -> str:
        """Reads the current guild's raw server memory, or "" when there is none.

        Unlike user memory there is exactly one server memory per guild, so it needs no
        selection phase, allowlist, or function tool: it is read directly with zero extra
        LLM latency. Returns "" for a DM (no guild) or an empty memory. Read once per reply
        and shared by the selection and answer phases.
        """
        if message.guild is None:
            return ""
        return read_memory_document(
            scope=server_scope(server_id=message.guild.id),
            compartments=[GLOBAL_COMPARTMENT],
            flavor="server",
        )

    def _resolve_reply_memory_candidates(
        self, *, message: Message, server_memory: str, read_context: MemoryReadContext
    ) -> tuple[list[UserMemory], dict[int, MemoryCandidate], int]:
        """Resolves deterministic memories and derives disjoint optional alias candidates."""
        bot_user = self.bot.user
        if bot_user is None:
            return [], {}, 0

        reference_chain = _walk_reference_chain(message=message)
        deterministic_allowed = build_memory_allowlist(
            users=[message.author, *(ref.author for ref in reference_chain), *message.mentions],
            bot_user_id=bot_user.id,
        )
        optional_allowed: dict[int, MemoryCandidate] = {}
        # Existing participant labels keep their community aliases even in a private
        # channel because that grants no new access. Only a public channel may offer absent
        # nickname-table members to the selector.
        if server_memory and message.guild is not None:
            widen_allowlist_with_aliases(
                allowed=deterministic_allowed, memory=server_memory, include_absent=False
            )
            if _source_channel_is_public(message=message):
                # No credit label: the conversation never names these members, so
                # `resolve_user_memories` credits them by their bare id. Deliberately NOT the
                # identity their own memory carries, which is the display name of whichever
                # guild's consolidation last wrote that fact (see `MemoryCandidate`).
                optional_allowed = {
                    user_id: MemoryCandidate(prompt_label=label)
                    for user_id, label in allowlist_ids_from_server_memory(
                        memory=server_memory
                    ).items()
                    if user_id not in deterministic_allowed and user_id != bot_user.id
                }

        memories = [
            memory
            for memory in resolve_user_memories(
                user_id_list=[str(user_id) for user_id in deterministic_allowed],
                allowed=deterministic_allowed,
                context=read_context,
            )
            if memory.memory != NO_STORED_MEMORY
        ]
        return memories, optional_allowed, len(deterministic_allowed)

    def _schedule_server_memory_update(
        self, *, message: Message, message_list: list[EasyInputMessageParam], full_reply: str
    ) -> None:
        """Schedules the bot's per-server memory update for a guild message.

        Server memory learns community-level signal from the whole conversation (no
        target-centering, since every message is server context). Skipped for DMs and
        for channels not visible to `@everyone`, so private / restricted-channel content
        never enters the server-wide memory any member can read.
        """
        if message.guild is None:
            return
        if not _source_channel_is_public(message=message):
            return
        schedule_memory_update(
            scope=server_scope(server_id=message.guild.id),
            subject=f"target_server_id: {message.guild.id}",
            message_list=message_list,
            full_reply=full_reply,
            extractor=self.server_memory_extractor,
            identity=render_server_identity(
                server_name=message.guild.name, server_id=message.guild.id
            ),
        )

    async def _await_optional_memory_selection(
        self, *, task: asyncio.Task[MemorySelection], message: Message, route_done: asyncio.Event
    ) -> tuple[MemorySelection, float] | None:
        """Awaits the optional selector without letting its failure affect direct memories."""
        started = time.monotonic()
        try:
            with logfire.span("gen_reply memory selection", message_id=message.id):
                selection = await _await_gated(
                    task=task,
                    label="memory selection",
                    route_done=route_done,
                    grace_seconds=MEMORY_SELECT_GRACE_SECONDS,
                )
        except TimeoutError as exc:
            logfire.warn(
                "Optional memory selection exceeded the post-route grace; retaining deterministic memories",
                grace_seconds=MEMORY_SELECT_GRACE_SECONDS,
                message_id=message.id,
                model=self.runtime_models.triage_model.name,
                _exc_info=exc,
            )
            return None
        except Exception:
            logfire.warn(
                "Optional memory selection failed; retaining deterministic memories",
                message_id=message.id,
                model=self.runtime_models.triage_model.name,
                _exc_info=True,
            )
            return None
        return selection, time.monotonic() - started

    async def _prepare_reply_context(
        self,
        message: Message,
        history_limit: int,
        parts_task: asyncio.Task[tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]]],
        text_parts: tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]],
        route_done: asyncio.Event,
    ) -> ReplyContext:
        """Builds history, shared parts, server memory, and the memory selection result.

        Runs speculatively as its own task concurrent with routing: everything here only
        reads (channel history, memory files, the selection request), so a non-QA route
        can discard it safely. `parts_task` carries the answer-path reference/current
        renders (uploaded files); `text_parts` carries their text-only twins so the memory
        selection call never re-reads or waits on the uploads.
        """
        text_reference, text_current = text_parts
        build_started = time.monotonic()

        # Fetch channel history once. Its text-only twin is rendered below only if a narrowed
        # selector request is actually needed; the upload-bearing full render is always awaited
        # later because the answer consumes it.
        raw_history = await self._fetch_history(message=message, limit=history_limit)

        # The bot's own per-server memory is read once here and shared by both phases: it
        # primes selection (a `## 成員稱呼` nickname table maps spoken aliases to ids) and
        # rides into the answer as background context. One file read, no extra LLM call.
        server_memory = self._read_server_memory(message=message)
        server_memory_block = (
            render_server_memory_block(memory=server_memory) if server_memory else None
        )

        # Where this reply is happening, for compartment scoping of every user-memory read.
        read_context = memory_read_context(message=message)

        # The message author's tone-preference note is read directly for that one author
        # (their own preference for how the bot should sound, cross-server safe by
        # construction) and injected on every reply with no selection phase, including one
        # that runs with user memory off. One file read, no extra LLM call.
        author_tone = read_tone(scope=user_scope(user_id=message.author.id))
        tone_block = render_tone_block(tone=author_tone) if author_tone else None

        # Code always resolves the current author, reply-chain authors, and current-message
        # mentions. A separate model call is reserved for the one non-mechanical question: does
        # the latest message obliquely refer to an absent member in a public nickname table? Both
        # paths stay behind resolve_user_memories, the shared permission and compartment boundary.
        memory_labels: list[str] = []
        selection_input_tokens = 0
        selection_output_tokens = 0
        memory_block: EasyInputMessageParam | None = None
        remaining_slots = 0
        selection_task: asyncio.Task[MemorySelection] | None = None
        memories, optional_allowed, deterministic_candidate_count = (
            self._resolve_reply_memory_candidates(
                message=message, server_memory=server_memory, read_context=read_context
            )
        )
        deterministic_memory_count = len(memories)
        if memories:
            memory_block = render_memory_context_block(memories=memories)
            memory_labels = memory_lookup_labels(memories=memories)

            remaining_slots = max(0, MEMORY_CONTEXT_TARGET_USERS - len(memories))
            logfire.debug(
                "gen_reply memory candidates built",
                deterministic_candidates=deterministic_candidate_count,
                deterministic_memories=len(memories),
                optional_candidates=len(optional_allowed),
                optional_slots=remaining_slots,
                message_id=message.id,
            )
            if optional_allowed and remaining_slots:
                # Render the text-only history only for a real optional lookup. This request
                # carries markers instead of file ids, so it never re-reads uploaded payloads.
                history_text_only = await self._render_history(
                    raw_history, text_only=True, message_id=message.id
                )
                selection_message_list: list[EasyInputMessageParam] = [
                    *history_text_only,
                    *text_reference,
                    *text_current,
                ]
                selection_task = asyncio.create_task(
                    coro=self._select_user_memories(
                        message=message,
                        message_list=selection_message_list,
                        allowed=optional_allowed,
                        read_context=read_context,
                        server_memory_block=server_memory_block,
                    )
                )

        try:
            # The answer needs the uploaded renders; await the full history render and the shared
            # reference/current uploads here, concurrently with any in-flight selection above.
            # `parts_task` is shielded so cancelling this speculative prep (IMAGE / VIDEO) never
            # cancels the shared upload task those routes still reuse; the full history render
            # rides as an ordinary gather child, so it is cancelled together with prep.
            with logfire.span("gen_reply context build", message_id=message.id):
                hist_messages, (reference_messages, current_message) = await asyncio.gather(
                    self._render_history(raw_history, text_only=False, message_id=message.id),
                    asyncio.shield(parts_task),
                )
            # Covers the history fetch/render plus waiting on the shared attachment upload, so
            # the log separates pre-answer attachment cost from the route-call cost.
            logfire.info(
                "gen_reply context build done",
                elapsed_seconds=time.monotonic() - build_started,
                message_id=message.id,
            )

            if selection_task is not None:
                # Memory selection is an optional preflight; a provider/proxy hiccup here must
                # never turn an answerable message into the generic error path. Resolved under the
                # route_done gate: it usually already finished during the upload wait above, so
                # this returns immediately; a slow one gets only the post-route grace.
                selection_result = await self._await_optional_memory_selection(
                    task=selection_task, message=message, route_done=route_done
                )
                if selection_result is not None:
                    selection, selection_elapsed = selection_result
                    selection_input_tokens = selection.input_tokens
                    selection_output_tokens = selection.output_tokens
                    selected_memories = selection.memories[:remaining_slots]
                    if len(selection.memories) > len(selected_memories):
                        logfire.warn(
                            "Capping optional memories to the remaining per-reply budget",
                            requested=len(selection.memories),
                            kept=len(selected_memories),
                            message_id=message.id,
                        )
                    if selected_memories:
                        memories.extend(selected_memories)
                        memory_block = render_memory_context_block(memories=memories)
                        memory_labels = memory_lookup_labels(memories=memories)
                    logfire.info(
                        "gen_reply memory selection done",
                        elapsed_seconds=selection_elapsed,
                        model=self.runtime_models.triage_model.name,
                        selected=len(selected_memories),
                        selected_ids=[memory.user_id for memory in selected_memories],
                        labels=memory_lookup_labels(memories=selected_memories),
                        candidate_count=len(optional_allowed),
                        deterministic_count=deterministic_memory_count,
                        message_id=message.id,
                    )
        finally:
            # If this prep is cancelled during the upload wait (a non-QA route discarding it)
            # before the gate resolves it, cancel the in-flight selection so it never orphans.
            if selection_task is not None and not selection_task.done():
                await _discard_task(
                    task=selection_task, label="memory selection", message_id=message.id
                )

        return ReplyContext(
            hist_messages=hist_messages,
            reference_messages=reference_messages,
            current_message=current_message,
            server_memory_block=server_memory_block,
            memory_block=memory_block,
            tone_block=tone_block,
            memory_labels=memory_labels,
            selection_input_tokens=selection_input_tokens,
            selection_output_tokens=selection_output_tokens,
        )

    async def _handle_message_reply(  # noqa: PLR0913 -- per-call reply inputs plus the route's memory/effort/voice gates
        self,
        message: Message,
        system_prompt: str,
        context: ReplyContext,
        effort: Literal["low", "high"] = "high",
        allow_voice: bool = False,
        allow_image: bool = False,
        allow_music: bool = False,
        allow_video: bool = False,
        allow_research: bool = False,
        describe_capabilities: bool = False,
        yt_url: str | None = None,
    ) -> None:
        """Streams the answer from a pre-built reply context, then schedules memory updates.

        Both the per-user and the per-server update are scheduled here; the per-server one
        carries its own guild / public-channel guards. `allow_voice` enables a
        spoken clip, `allow_image` an inline generated image, `allow_music` an inline generated
        music clip, and `allow_video` an inline generated video clip when the answer model marks
        the reply for it (image / music / video are QA only; the media persona replies get voice
        alone). `describe_capabilities` injects the feature reference that replaced
        `/help`, carried by QA alone since a persona reply riding generated media is not fielding
        a question about the bot. `yt_url`, set only when the router asked
        to watch a linked YouTube video, swaps the answer turn onto the Gemini Interactions API
        (which can ingest the video) while reusing the same streamer / footer / memory path.
        """
        voice_generator = (
            self.voice_generator if allow_voice and self.config.inline_voice_enabled else None
        )
        image_generator = (
            self.image_generator if allow_image and self.config.inline_image_enabled else None
        )
        music_generator = (
            self.music_generator if allow_music and self.config.music_available else None
        )
        video_generator = (
            self.video_generator if allow_video and self.config.video_available else None
        )
        # Only advertise the inline `<generate-image>` marker when the renderer is actually active; with
        # it disabled the streamer would strip the block and produce nothing, silently dropping
        # the visual request from the reply, so a disabled deployment must not be told about it.
        if image_generator is not None:
            system_prompt = f"{system_prompt}\n{INLINE_IMAGE_INSTRUCTION}"
        # Advertise the inline `<generate-music>` marker only when the generator is actually active, same
        # reasoning as the image marker: a disabled deployment (kill-switch off or no Gemini key)
        # must not be told about a marker the streamer would strip without producing anything.
        if music_generator is not None:
            system_prompt = f"{system_prompt}\n{MUSIC_INSTRUCTION}"
        # Advertise the inline `<generate-video>` marker only when the generator is actually active, same
        # reasoning as the image/music markers: a disabled deployment (kill-switch off or no Gemini
        # key) must not be told about a marker the streamer would strip without producing anything.
        if video_generator is not None:
            system_prompt = f"{system_prompt}\n{VIDEO_INSTRUCTION}"
        # Advertise the <deep-research> marker only when the feature is on, same reasoning as the
        # image marker: a disabled deployment must not be told about a marker the streamer would
        # strip without producing anything.
        research_offered = allow_research and self.config.deep_research_available
        if research_offered:
            system_prompt = f"{system_prompt}\n{DEEP_RESEARCH_INSTRUCTION}"
        slow_model = self.runtime_models.slow_model.model_copy(update={"effort": effort})
        _dispatched_model.set(slow_model.name)
        # Keep the current user message LAST so the model answers it. Memory rides earliest as
        # low-authority background; the reference message then sits just above the current
        # message so the reply pair (reference -> current) stays adjacent and reads as the
        # primary context rather than getting buried up near history. The feature reference
        # leads: it is the one block that is byte-identical on every reply, so the front is
        # where it costs the least against a prefix cache.
        answer_input: ResponseInputParam = (
            [render_capabilities_block()] if describe_capabilities else []
        )
        answer_input.extend(context.hist_messages)
        answer_input.extend(
            block
            for block in (context.server_memory_block, context.memory_block, context.tone_block)
            if block is not None
        )
        answer_input.extend(context.reference_messages)
        # The linked post(s) the user pointed at ride just before the current message, each
        # block led by its own separator; empty unless a registered source found a link to read
        # (in this message, or for Threads the one it replies to). The order inside is
        # LINK_CONTEXT_SOURCES order.
        answer_input.extend(context.link_blocks)
        answer_input.extend(context.current_message)

        # A linked YouTube video the router asked to watch swaps the answer turn onto the Gemini
        # Interactions API: the Responses bridge cannot make Gemini watch the video, so this is
        # the one backend swap. It is Gemini-only and kill-switchable; otherwise (no video, a
        # non-Gemini answer model, the switch off, or no direct key to swap with) the turn falls
        # back to the Responses path, which never errors. Both feed the same streamer so footer /
        # memory / preview are shared.
        use_interactions = (
            yt_url is not None
            and "gemini" in slow_model.name
            and self.config.youtube_video_enabled
            and bool(self.config.gemini_api_key.strip())
        )
        backend = "interactions" if use_interactions else "responses"
        if yt_url is not None and not use_interactions:
            # The swap is silent to the user, so without this the log shows a Responses answer to
            # a message the router explicitly asked to watch, with nothing saying which gate said
            # no. The fallback itself is correct; only the reason was unrecoverable.
            logfire.info(
                "gen_reply youtube watch declined; answering on the responses backend",
                message_id=message.id,
                reason="model"
                if "gemini" not in slow_model.name
                else "kill-switch"
                if not self.config.youtube_video_enabled
                else "no-gemini-key",
                model=slow_model.name,
            )
        if use_interactions:
            # Persistent marker (added directly, not via the status chain) so it stays after the
            # chain's final reaction to show the reply was grounded in the watched video. The bot's
            # own application emoji `youtube`, usable as a reaction in any guild the bot is in.
            # Added BEFORE the streamer is built: its `created_at` is what the answer latency is
            # measured from, so leaving this REST round trip inside that window would bias the
            # figure against the one backend that pays for it.
            await update_reaction(
                message=message, bot_user=self.bot.user, emoji="<:youtube:1517546722535018596>"
            )
        # Seed the streamer with the selection request's usage so the footer and chat reward
        # reflect both LLM calls; the answer stream sums its own usage on top.
        streamer = ResponseStreamer(
            message=message,
            memory_lookups=context.memory_labels,
            input_tokens=context.selection_input_tokens,
            output_tokens=context.selection_output_tokens,
            model_effort=effort,
            backend=backend,
            voice_generator=voice_generator,
            image_generator=image_generator,
            music_generator=music_generator,
            video_generator=video_generator,
            media_delivery=self.media_delivery,
            input_builder=self.input_builder,
        )
        # The one record of what the answer model was actually handed. Everything here is a count
        # or a flag: a reply that behaves as if it never saw an attachment, a memory or a linked
        # post is otherwise indistinguishable in the log from one that had them.
        logfire.info(
            "gen_reply answer dispatch",
            message_id=message.id,
            model=slow_model.name,
            backend=backend,
            effort=effort,
            input_blocks=len(answer_input),
            history=len(context.hist_messages),
            reference=len(context.reference_messages),
            link_blocks=len(context.link_blocks),
            media_parts=_count_media_parts(answer_input=answer_input),
            capabilities=describe_capabilities,
            server_memory=context.server_memory_block is not None,
            user_memory=context.memory_block is not None,
            tone=context.tone_block is not None,
            # Joined rather than a list: the console exporter renders a list one element per
            # line, and this record fires on every reply.
            markers=",".join(
                name
                for name, offered in (
                    ("voice", voice_generator is not None),
                    ("image", image_generator is not None),
                    ("music", music_generator is not None),
                    ("video", video_generator is not None),
                    ("research", research_offered),
                )
                if offered
            ),
        )
        with logfire.span(
            "gen_reply answer", model=slow_model.name, backend=backend, message_id=message.id
        ):
            responses: AsyncIterator[ResponseStreamEvent]
            if use_interactions and yt_url is not None:
                responses = create_interactions_answer_stream(
                    client=self.gemini_client,
                    model=slow_model.name,
                    system_instruction=_build_runtime_instructions(
                        system_prompt=system_prompt, message=message
                    ),
                    steps=to_interactions_input(answer_input=answer_input, youtube_url=yt_url),
                    effort=slow_model.effort,
                )
            else:
                responses = await self.openai_client.responses.create(
                    model=slow_model.deployment_name,
                    instructions=_build_runtime_instructions(
                        system_prompt=system_prompt, message=message
                    ),
                    input=answer_input,
                    reasoning=slow_model.reasoning,
                    tools=list(slow_model.tools),
                    stream=True,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": message.author.name},
                )
            full_reply = await streamer.stream(responses=responses)
        # A <deep-research> brief the answer model emitted launches a research thread. Done after
        # the stream (and its single media edit) so it never touches the reply's attachment edit;
        # best-effort, gated, and a no-op when the feature is off or no brief was emitted.
        if research_offered and streamer.research_brief:
            await _maybe_launch_research(
                bot=self.bot, message=message, anchor=streamer.reply, brief=streamer.research_brief
            )
        memory_message_list = target_centered_memory_messages(
            hist_messages=context.hist_messages,
            reference_messages=context.reference_messages,
            current_message=context.current_message,
            target_user_id=message.author.id,
        )
        # The second subject line names where this conversation happened (guild id
        # or DM); it survives the memory_job round-trip so the pipeline can stamp
        # each observation's source deterministically.
        source_line = subject_source_line(guild_id=message.guild.id if message.guild else None)
        schedule_memory_update(
            scope=user_scope(user_id=message.author.id),
            subject=f"target_user_id: {message.author.id}\n{source_line}",
            message_list=memory_message_list,
            full_reply=full_reply,
            extractor=self.memory_extractor,
            identity=render_author_identity(
                display_name=message.author.display_name,
                username=message.author.name,
                user_id=message.author.id,
            ),
        )
        # The per-server update carries its own guards rather than riding the per-user one:
        # DMs and non-public channels are dropped inside `_schedule_server_memory_update`.
        self._schedule_server_memory_update(
            message=message, message_list=context.message_list, full_reply=full_reply
        )

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Resumes persisted memory work after a restart (runs once).

        `on_ready` fires on every gateway reconnect, so `_resume_started` guards it
        to a single sweep per process. The sweep is spawned, never awaited, so the
        gateway is not blocked while it digests in the background.
        """
        if self._resume_started:
            return
        # Bound to this loop, so it starts here rather than at import: an unstarted
        # service drops every commit request instead of binding its queue to whichever
        # loop happened to enqueue first.
        memory_git.start()
        self._resume_started = True
        self._spawn(self._resume_memory())

    async def _resume_memory(self) -> None:
        """Re-enqueues persisted phase-1 jobs and consolidates over-threshold scopes.

        Two paths, both riding the existing per-scope lock + global concurrency
        semaphore: persisted `pending`/`failed` jobs are re-run (transcript intact),
        and every scope whose raw backlog is over threshold is swept. The sweep
        covers scopes with a resumed job too: the per-scope lock plus the under-lock
        `_should_consolidate` re-check make the resumed extraction and the sweep
        idempotent, so a consolidation interrupted by the restart still finishes
        even when the resumed extraction early-returns (failed, no signal, or all
        duplicates) before it would reach the consolidation check.
        """
        jobs = await safe_list_resumable()
        for job in jobs:
            if job.transcript is None:
                continue
            extractor = (
                self.server_memory_extractor if job.flavor == "server" else self.memory_extractor
            )
            resume_memory_update(
                scope=job.scope,
                subject=job.subject,
                transcript=job.transcript,
                extractor=extractor,
                identity=job.identity,
                token=job.token,
            )
        if jobs:
            logfire.info("resumed persisted memory jobs", count=len(jobs))
        swept = 0
        for scope in iter_scopes():
            if not needs_consolidation(scope=scope):
                continue
            extractor = (
                self.server_memory_extractor
                if flavor_of(scope=scope) == "server"
                else self.memory_extractor
            )
            self._spawn(
                consolidate_if_needed(
                    scope=scope,
                    extractor=extractor,
                    identity=render_owner_identity(owner=read_owner(scope=scope)),
                )
            )
            swept += 1
        if swept:
            logfire.info("scheduled memory consolidation sweep", count=swept)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and handles AI reply generation.

        Args:
            message: The message that was sent.
        """
        # Ignore messages from bots.
        if message.author.bot:
            return

        # Match <@ID> in content, not message.mentions: reply notifications add
        # the bot to mentions and would trigger on replies to functional bot
        # posts (e.g. Threads embeds, video downloads).
        is_dm = message.guild is None
        if not is_dm and not self.input_builder.has_bot_mention(content=message.content):
            return

        # Skip a (mentioned) message typed inside a research thread the ResearchCogs cog is
        # actively driving: the thread is its workspace until the report lands, so QA must not
        # answer over the live status edits. The skip lifts the moment the run finishes.
        if _in_active_research_thread(bot=self.bot, channel_id=message.channel.id):
            logfire.debug(
                "gen_reply skipped: the research cog is still writing into this thread",
                **_message_log_fields(message=message),
            )
            return

        user_prompt = await self.input_builder.get_user_prompt(content=message.content)
        has_attachment = bool(message.attachments or message.stickers)
        # A forward leaves content/attachments/stickers empty and puts the payload in
        # `message.snapshots`, so it must not be gated out as an empty message here, or the
        # snapshot text/media render in `input.py` never runs.
        is_forward = bool(message.snapshots)
        # A forward puts its request in `message.snapshots`, not content, so merge the forwarded
        # text into the prompt (after the forwarder's own comment, if any). A guild forward can
        # only trigger via a `<@bot>` comment, so the comment is usually non-empty: merging (not
        # just an empty fallback) is what lets an IMAGE/VIDEO route render the forwarded "draw a
        # cat" even when the trigger comment ("@bot please") survives mention-stripping.
        if is_forward and (
            forwarded := self.input_builder.forwarded_request_text(message=message)
        ):
            user_prompt = f"{user_prompt}\n{forwarded}".strip() if user_prompt else forwarded

        if not user_prompt and not has_attachment and not is_forward:
            logfire.debug(
                "gen_reply empty prompt; replied with ?", **_message_log_fields(message=message)
            )
            await update_reaction(message=message, bot_user=self.bot.user, emoji="❓")
            await message.reply(content="?")
            return

        logfire.info(
            "gen_reply received",
            **_message_log_fields(message=message),
            prompt_chars=len(user_prompt),
            has_attachment=has_attachment,
            attachment_count=len(message.attachments),
            sticker_count=len(message.stickers),
            is_dm=is_dm,
        )

        reactions = ReactionStatusChain(message=message, bot_user=self.bot.user)
        try:
            await self._run_reply_pipeline(
                message=message, user_prompt=user_prompt, reactions=reactions
            )
        except Exception as e:
            logfire.error(
                "gen_reply failed",
                **_message_log_fields(message=message),
                model=_dispatched_model.get(),
                error_type=type(e).__name__,
                _exc_info=True,
            )
            try:
                reactions.advance(emoji="<:redcross:1517565100838355016>")
                error_embed = Embed(
                    title="Something went wrong",
                    description=f"```\n{extract_friendly_error(exc=e)}\n```",
                    color=DISCORD_RED,
                )
                error_embed.set_footer(text=type(e).__name__)
                spacer = embed_spacer_payload(embeds=[error_embed], is_edit=False, target=message)
                try:
                    await message.reply(content=None, embed=error_embed, **spacer)
                except HTTPException as send_error:
                    # Source deleted before the error landed (50035): send it unparented. Rebuild
                    # the spacer; the failed reply already consumed the single-use spacer file.
                    if send_error.code != 50035 and not isinstance(send_error, NotFound):
                        raise
                    fresh_spacer = embed_spacer_payload(
                        embeds=[error_embed], is_edit=False, target=message
                    )
                    await message.channel.send(content=None, embed=error_embed, **fresh_spacer)
            except Exception as report_error:
                # Broad on purpose: this is the last-resort user notice; nothing above it can
                # recover, and it must not displace the original failure.
                logfire.warn(
                    "failed to deliver the pipeline failure notice",
                    message_id=message.id,
                    error_type=type(report_error).__name__,
                    _exc_info=report_error,
                )
        finally:
            await reactions.flush()

    async def _run_reply_pipeline(  # noqa: PLR0915, C901, PLR0912 -- orchestrates route, speculative prep, threads context, and per-route dispatch in sequence
        self, message: Message, user_prompt: str, reactions: ReactionStatusChain
    ) -> None:
        """Routes the message and dispatches the matching handler with speculative QA context."""
        # Named in the usage record below, which is written from this method's `finally`
        # because this is the one scope that has both outcomes and the route in hand.
        route_decision: str | None = None
        prep_task: asyncio.Task[ReplyContext] | None = None
        parts_task: (
            asyncio.Task[tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]]] | None
        ) = None
        effort_task: asyncio.Task[EffortGrade] | None = None
        link_tasks: dict[str, asyncio.Task[list[EasyInputMessageParam]]] = {}
        link_context_deadline: float | None = None
        try:
            with logfire.span("gen_reply pipeline", message_id=message.id) as pipeline_span:
                pipeline_started = time.monotonic()
                reactions.advance(emoji="<:flowchart:1517561877973045349>")
                # The reference + current attachment uploads (and their activation polls)
                # run in the background and only the answer awaits them. The route and the
                # optional memory selection use the text-only renders, so neither waits on the Files
                # API. The QA context builds speculatively in parallel with the route call
                # since QA is the dominant route — non-QA routes discard it.
                parts_task = asyncio.create_task(
                    coro=self._get_reference_and_current(message=message)
                )
                text_reference, text_current = await self._get_reference_and_current(
                    message=message, text_only=True
                )
                # Signals optional memory selection that the route has returned: selection runs
                # unbounded while this is clear and gets only a short grace once it is set.
                route_done = asyncio.Event()
                prep_task = asyncio.create_task(
                    coro=self._prepare_reply_context(
                        message=message,
                        history_limit=HISTORY_MESSAGE_LIMIT,
                        parts_task=parts_task,
                        text_parts=(text_reference, text_current),
                        route_done=route_done,
                    )
                )
                # Effort grading rides the same route_done gate as memory selection: it runs
                # in parallel with the route and only the QA answer model consumes it, so
                # IMAGE/VIDEO cancel it below.
                effort_task = asyncio.create_task(
                    coro=self._grade_effort(
                        message=message,
                        reference_messages=text_reference,
                        current_message=text_current,
                    )
                )
                route = await self._route_classify(
                    message=message,
                    reference_messages=text_reference,
                    current_message=text_current,
                )
                if route.decision == "QA" and route.link_context_sources:
                    link_context_deadline = (
                        asyncio.get_running_loop().time() + LINK_CONTEXT_GRACE_SECONDS
                    )
                route_done.set()
                route_decision = route.decision
                pipeline_span.set_attribute(key="route", value=route.decision)
                if route.decision == "QA" and route.link_context_sources:
                    # The router selects only source names; URL ownership stays local and the
                    # registry still applies every URL filter and reply-chain rule. Start each
                    # selected builder only now, after intent is known, so an incidental link
                    # never begins a metadata fetch, media download, or Files API upload.
                    selected_sources = set(route.link_context_sources)
                    if link_context_deadline is None:
                        raise RuntimeError("Selected link sources have no route deadline")
                    for link_source in LINK_CONTEXT_SOURCES:
                        if link_source.name not in selected_sources:
                            continue
                        link_url = _link_url_for_source(source=link_source, message=message)
                        if link_url is None:
                            # The router named this source, but its URL is not where the source is
                            # allowed to look (Threads alone walks the reply chain), so the answer
                            # silently goes without the post the user was pointing at.
                            logfire.info(
                                "gen_reply selected link source has no readable URL; skipping it",
                                source=link_source.name,
                                message_id=message.id,
                            )
                            continue
                        link_tasks[link_source.name] = asyncio.create_task(
                            coro=_run_until_deadline(
                                awaitable=link_source.build(
                                    url=link_url,
                                    answer_model_is_gemini=(
                                        "gemini" in self.runtime_models.slow_model.name
                                    ),
                                    gemini_client=self.gemini_client_if_configured,
                                    allow_media_ingest=link_source.media_ingest_allowed(
                                        config=self.config
                                    ),
                                ),
                                deadline=link_context_deadline,
                            )
                        )
                    if "threads" in link_tasks:
                        # Persistent marker (added directly, not via the status chain) saying a
                        # Threads post was read, the same one `parse_threads` adds when it expands
                        # a link instead. Added once every builder is started so the REST call
                        # never sits between two of them.
                        await update_reaction(
                            message=message,
                            bot_user=self.bot.user,
                            emoji="<:threads:1535657820668559380>",
                        )
                if route.decision in ("IMAGE", "VIDEO"):
                    # IMAGE and VIDEO share identical speculative-task teardown; they differ only
                    # in the status emoji and which media handler runs. Effort is answer-only,
                    # while intent-gated link builders never start for these routes.
                    await _discard_task(task=effort_task, label="effort", message_id=message.id)
                    effort_task = None
                    reactions.advance(
                        emoji="<:image:1517559727880667226>"
                        if route.decision == "IMAGE"
                        else "<:video:1517560671913377842>"
                    )
                    # The media reply consumes (not discards) the speculative context: the handler
                    # awaits it only after the media is on screen so the build overlaps generation.
                    # `parts_task` is left for the finally backstop — prep awaits it via
                    # asyncio.shield, so if the handler discards prep on a generation failure the
                    # shielded upload keeps running and the finally must drain it.
                    media_context_task = prep_task
                    prep_task = None
                    if route.decision == "IMAGE":
                        await self._handle_image_reply(
                            message=message,
                            user_prompt=user_prompt,
                            context_task=media_context_task,
                        )
                    else:
                        await self._handle_video_reply(
                            message=message,
                            user_prompt=user_prompt,
                            context_task=media_context_task,
                        )
                else:
                    reactions.advance(emoji="<:message:1517560873000898860>")
                    # Selection still gates the answer here; if this wait ever needs to go,
                    # the answer could speculatively start without memory and refire when
                    # selection picks some.
                    context = await prep_task
                    prep_task = None
                    parts_task = None
                    effort = await self._resolve_effort(
                        message=message, effort_task=effort_task, route_done=route_done
                    )
                    effort_task = None
                    # The selected builds overlapped the remaining reply preparation. Resolve
                    # each under the same grace and fold the post blocks into the answer context
                    # in registry order so the splice stays deterministic.
                    if link_tasks:
                        if link_context_deadline is None:
                            raise RuntimeError("Selected link tasks have no route deadline")
                        link_blocks: list[EasyInputMessageParam] = []
                        for link_source in LINK_CONTEXT_SOURCES:
                            link_task = link_tasks.pop(link_source.name, None)
                            if link_task is None:
                                continue
                            link_blocks.extend(
                                await self._resolve_link_block(
                                    message=message,
                                    source=link_source.name,
                                    link_task=link_task,
                                    deadline=link_context_deadline,
                                    on_timeout=link_source.on_timeout,
                                )
                            )
                        context = context.model_copy(update={"link_blocks": link_blocks})
                    pipeline_span.set_attribute(key="effort", value=effort)
                    # Watch a linked YouTube video only when the router judged the user is asking
                    # about it; the URL itself is taken from the message text or the replied-to
                    # message (never the model) so the answer turn ingests the exact link posted.
                    yt_url = _find_youtube_url(message=message) if route.watch_video else None
                    if route.watch_video and yt_url is None:
                        # The router judged the user is asking about a video, but the URL scan
                        # found none where it is allowed to look, so the answer is written
                        # without watching anything and nothing else records that.
                        logfire.info(
                            "gen_reply watch_video requested but no YouTube URL was found",
                            message_id=message.id,
                        )
                    _log_pre_answer_latency(
                        started=pipeline_started, decision=route.decision, message_id=message.id
                    )
                    await self._handle_message_reply(
                        message=message,
                        system_prompt=REPLY_PROMPT,
                        context=context,
                        effort=effort,
                        allow_voice=True,
                        allow_image=True,
                        allow_music=True,
                        allow_video=True,
                        allow_research=_can_launch_research(message=message),
                        describe_capabilities=True,
                        yt_url=yt_url,
                    )
                reactions.advance(emoji="<:greencheck:1517565102424068226>")
                # End of the turn on the success path; the failure path is `gen_reply failed`,
                # which carries the traceback. The console exporter prints no span-end line, so
                # without this the file holds no total for the turn at all.
                logfire.info(
                    "gen_reply pipeline done",
                    elapsed_seconds=time.monotonic() - pipeline_started,
                    decision=route.decision,
                    message_id=message.id,
                )
        finally:
            if prep_task is not None:
                await _discard_task(task=prep_task, label="prep", message_id=message.id)
            if effort_task is not None:
                await _discard_task(task=effort_task, label="effort", message_id=message.id)
            if parts_task is not None:
                await _discard_task(task=parts_task, label="parts", message_id=message.id)
            await _discard_link_tasks(
                link_tasks=link_tasks, deadline=link_context_deadline, message_id=message.id
            )
            # One record per triggering message, not per delivered artifact: the inline
            # clips and the media persona reply are all parts of this same turn. A failure
            # is recorded too — someone still talked to the bot — under the route it had
            # reached, or `unrouted` when it did not get that far.
            await self.usage_recorder.record(
                kind="reply",
                name=route_decision or UNROUTED_REPLY,
                user_id=message.author.id,
                user_name=message.author.name,
                guild_id=message.guild.id if message.guild else None,
                channel_id=message.channel.id,
            )


def _can_launch_research(*, message: Message) -> bool:
    """Whether a research thread can be opened from this message.

    Only a guild text channel can host a nested thread; in a DM or inside an existing thread the
    `<deep-research>` marker is suppressed so the answer model never promises a run that cannot
    actually start (the launch would otherwise return the no-thread path and contradict itself).
    """
    return message.guild is not None and isinstance(message.channel, TextChannel)


def _in_active_research_thread(*, bot: commands.Bot, channel_id: int) -> bool:
    """Whether a channel id is a research thread the ResearchCogs cog is actively driving."""
    get_cog = getattr(bot, "get_cog", None)
    cog = get_cog("ResearchCogs") if callable(get_cog) else None
    checker = getattr(cog, "is_research_thread", None)
    return bool(checker(channel_id=channel_id)) if checker is not None else False


async def _maybe_launch_research(
    *, bot: commands.Bot, message: Message, anchor: Message | None, brief: str
) -> None:
    """Hands a QA-emitted research brief to the ResearchCogs cog when it is loaded and enabled.

    `anchor` is the bot's own reply message; the research thread hangs off it (more intuitive than
    the user's message), falling back to the user's message inside the cog when it is None.
    """
    get_cog = getattr(bot, "get_cog", None)
    cog = get_cog("ResearchCogs") if callable(get_cog) else None
    launcher = getattr(cog, "launch", None)
    if launcher is None:
        return
    # Best-effort boundary: the research launch must never break an already-delivered reply, so
    # the except stays broad. ResearchCogs.launch handles its expected outcomes by return value,
    # so anything raising here is unexpected and the emitted brief is lost — markers.py already
    # stripped it from the visible text.
    try:
        await launcher(message=message, anchor=anchor, brief=brief)
    except Exception as exc:
        logfire.warn(
            "deep research launch failed; the emitted brief was dropped",
            message_id=message.id,
            anchor_id=anchor.id if anchor is not None else None,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )


def setup(bot: commands.Bot) -> None:
    """Adds the ReplyGeneratorCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
