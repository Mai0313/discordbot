"""Cog that routes Discord messages through the AI reply pipeline."""

import re
import time
import base64
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast
import asyncio
from functools import cached_property
import contextlib

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
from discordbot.cogs._memory.store import (
    read_tone,
    user_scope,
    iter_scopes,
    server_scope,
    read_main_memory,
    read_main_identity,
)
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.media_delivery import (
    MediaItem,
    MediaDeliveryPlanner,
    upload_limit_for,
    build_media_delivery_planner,
)
from discordbot.cogs._gen_reply.input import (
    MessageInputBuilder,
    sanitize_identity,
    render_author_identity,
    render_server_identity,
)
from discordbot.cogs._memory.pipeline import (
    flavor_of,
    needs_consolidation,
    safe_list_resumable,
    resume_memory_update,
    consolidate_if_needed,
    schedule_memory_update,
)
from discordbot.cogs._gen_reply.context import ReplyContext
from discordbot.cogs._gen_reply.prompts import (
    IMAGE_PROMPT,
    REPLY_PROMPT,
    ROUTE_PROMPT,
    VIDEO_PROMPT,
    EFFORT_PROMPT,
    SUMMARY_PROMPT,
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
from discordbot.cogs._memory.extraction import (
    MemoryExtractorAI,
    subject_source_line,
    target_centered_memory_messages,
)
from discordbot.cogs._gen_reply.files_api import upload_to_files_api
from discordbot.cogs._gen_reply.streaming import ResponseStreamer
from discordbot.cogs._gen_reply.exceptions import extract_friendly_error
from discordbot.cogs._gen_reply.generation import (
    MAX_VIDEO_REFERENCE_IMAGES,
    ImageGenerator,
    MusicGenerator,
    VideoGenerator,
    VoiceGenerator,
    PromptGenerator,
)
from discordbot.cogs._gen_reply.memory_tool import (
    NO_STORED_MEMORY,
    GET_USER_MEMORY_TOOL,
    UserMemory,
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
)
from discordbot.cogs._memory.server_prompts import (
    SERVER_PHASE1_PROMPT,
    SERVER_PHASE2_PROMPT,
    SERVER_PHASE1_EVALUATOR_PROMPT,
)
from discordbot.cogs._gen_reply.interactions import (
    to_interactions_input,
    create_interactions_answer_stream,
)
from discordbot.cogs._gen_reply.link_sources import LinkContextSource
from discordbot.cogs._gen_reply.attachment.select import build_attachment_handler
from discordbot.cogs._gen_reply.link_sources.douyin import (
    build_douyin_context_messages,
    douyin_timeout_context_messages,
)
from discordbot.cogs._gen_reply.link_sources.threads import (
    build_threads_context_messages,
    threads_timeout_context_messages,
)
from discordbot.cogs._gen_reply.link_sources.bilibili import (
    build_bilibili_context_messages,
    bilibili_timeout_context_messages,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable, Coroutine, AsyncIterator

    from openai.types.responses import ResponseStreamEvent


_MESSAGE_URL_RE = re.compile(pattern=r"(?i)\b(?:https?://|www\.)\S+")

# Memory selection overlaps the route call for free: the QA path joins the speculative
# prep task only after the route returns, so selection runs unbounded while the route is
# still in flight. Once the route completes, a still-running selection gets only this grace
# before the reply answers without memory, so a slow selection can never stall the pipeline
# yet a selection that finishes within the (route-dominant) window is never thrown away.
# Tune against the `gen_reply memory selection done` latency log.
MEMORY_SELECT_GRACE_SECONDS = 5.0

# Effort grading runs in parallel with the route under the same `route_done` gate as
# memory selection: it runs unbounded while the route is in flight and gets only this
# grace once the route returns before the reply falls back to "high" effort. The grade
# is consumed only just before the answer model starts, so this latency hides behind the
# route. Tune against the `gen_reply effort done` latency log.
EFFORT_GRACE_SECONDS = 5.0

# A linked-post context build rides the same route_done gate: it runs unbounded while the
# route is in flight and gets only this grace once the route returns. Far wider than
# memory/effort because it fetches the post's media and uploads it to the Files API, and
# because answering blind about a link the user explicitly pointed at is the failure this
# feature exists to prevent. The builder bounds its own media step just under this and
# degrades to text, so the grace is a backstop rather than the usual exit. The build overlaps
# the route window for free. Tune against the `gen_reply link context done` latency log.
LINK_CONTEXT_GRACE_SECONDS = 180.0

# Bound on the ACTIVE poll for a generated clip uploaded so the persona reply can watch it.
# Generous relative to an image because video sits in PROCESSING longer, but far under the
# link-media bound: the clip was just produced here, so it is small and known-good.
GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS = 60.0


def _message_link_texts(message: Message) -> list[str]:
    """The text spans a message actually renders to the model, for URL detection.

    Mirrors `get_cleaned_content` / `snapshot_text`: content takes precedence and an embed is
    rendered (and thus scanned) only when its content is empty. So a URL scanner never fires on a
    link the answer model was not shown, e.g. a captioned forwarded link card whose URL lives only
    in the embed. A forward puts its payload in `message.snapshots`, scanned via `snapshot_text`.
    """
    content = (message.content or "").strip()
    texts = [content]
    if not content:
        texts.append(MessageInputBuilder.extract_embed_text(embeds=list(message.embeds)))
    for snapshot in message.snapshots:
        texts.append(MessageInputBuilder.snapshot_text(snapshot=snapshot))
    return texts


def _first_url_match(pattern: re.Pattern[str], message: Message) -> re.Match[str] | None:
    """First match of a URL pattern across a message's content, embeds, and forwarded snapshots."""
    for text in _message_link_texts(message=message):
        match = pattern.search(string=text)
        if match:
            return match
    return None


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


def _youtube_url_in_message(message: Message) -> str | None:
    """Returns the first YouTube URL in a message's text, embeds, or forwarded snapshots, if any."""
    match = _first_url_match(pattern=YOUTUBE_URL_RE, message=message)
    return match.group(0) if match else None


def _find_youtube_url(message: Message) -> str | None:
    """Finds a YouTube URL in the current message or the reply-reference chain.

    Unlike Threads (whose `parse_threads` cog re-injects a replied-to post as an embed),
    a YouTube link has no such cog, so a reply to a message that merely links a video would
    otherwise be missed; the reference chain is searched so "summarize this" on a replied-to
    video still watches it. The current message wins, then the nearest reference outward.
    """
    found = _youtube_url_in_message(message=message)
    if found is not None:
        return found
    for ref in _walk_reference_chain(message=message):
        found = _youtube_url_in_message(message=ref)
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
    threads, douyin, memory selection), so anything they can raise must be swallowed here rather
    than surfacing on a route that already decided it does not need the result. `label` names
    which one failed, since the tasks are otherwise indistinguishable at this point.
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
    """Awaits a speculative side task, bounded by the route call instead of a fixed timeout.

    The task overlaps the route for free: while the route is still in flight it may run
    unbounded; once the route completes (`route_done` set) a still-running task gets only
    `grace_seconds` more before this raises TimeoutError. The task is always cancelled on
    exit so it never orphans (e.g. when the speculative prep task is discarded on a non-QA
    route). Shared by memory selection and effort grading, which both ride this gate.
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
    monkeypatching `discordbot.cogs.gen_reply.build_threads_context_messages` effective.
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
    """The Douyin kill-switch plus the direct-Gemini key its Files API upload needs."""
    return config.douyin_video_enabled and bool(config.gemini_api_key.strip())


def _bilibili_media_ingest_allowed(config: LLMConfig) -> bool:
    """The Bilibili kill-switch plus the direct-Gemini key its Files API upload needs."""
    return config.bilibili_video_enabled and bool(config.gemini_api_key.strip())


# The linked-content sources gen_reply reads into answer context, in splice order: the blocks
# land in the answer input in this order, just before the current message. Adding a source is
# one entry here plus its builder module; the pipeline loops stay untouched.
LINK_CONTEXT_SOURCES: tuple[LinkContextSource, ...] = (
    LinkContextSource(
        name="threads",
        url_pattern=THREADS_URL_RE,
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
    *, link_tasks: dict[str, "asyncio.Task[list[EasyInputMessageParam]]"], message_id: int
) -> None:
    """Discards every in-flight link-context build (non-QA routes and the finally backstop)."""
    for name, task in link_tasks.items():
        await _discard_task(task=task, label=name, message_id=message_id)
    link_tasks.clear()


def _log_pre_answer_latency(started: float, decision: str) -> None:
    """Logs total time from pipeline start to answer dispatch (the user's 'router stage')."""
    logfire.info(
        "gen_reply pre-answer latency",
        elapsed_seconds=time.monotonic() - started,
        decision=decision,
    )


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
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the ReplyGeneratorCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()
        self.runtime_models = RuntimeModelCatalog()
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

        DIRECT to Google (`gemini_api_key`, no proxy): it serves the two runtime paths the
        LiteLLM proxy cannot, and the swap only ever fires when the answer model is already
        Gemini so the direct credential is always the right one:
        - native omni video generation / editing (`interactions.create`, delivery=uri + Files
          download); and
        - the YouTube-aware QA answer turn that streams through the native Interactions API (the
          only path that can actually watch a linked video).
        Both forgo proxy-side cost/usage tracking, like the deep-research direct path. An empty
        key raises at construction, so a caller that is reachable without one must go through
        `gemini_client_if_configured` instead of touching this.

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
            client=self.openai_client, model_name=self.runtime_models.tts_model.name
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
            A director bound to this cog's proxy client and the flash + high + grounding
            `prompt_model`; each `refine` call is gated by the caller's per-route flag
            (`config.image_refine_prompt_enabled` / `config.video_refine_prompt_enabled`) and
            expands the raw request before `render`, best-effort (raw prompt on disable / empty /
            error).
        """
        return PromptGenerator(
            client=self.openai_client, prompt_model=self.runtime_models.prompt_model
        )

    @cached_property
    def video_generator(self) -> VideoGenerator:
        """The cached video renderer for the VIDEO route.

        Returns:
            A generator bound to this cog's DIRECT-to-Google Gemini client and the video model
            (the Interactions API is Gemini-only, not reachable via the proxy); the route calls
            `render` (raises).
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
            evaluate_model=self.runtime_models.memory_evaluator_model,
            consolidate_model=self.runtime_models.memory_consolidator_model,
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
            evaluate_model=self.runtime_models.memory_evaluator_model,
            consolidate_model=self.runtime_models.memory_consolidator_model,
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
        """Fetches up to `limit` channel-history messages once (a single Discord API call).

        Returned raw so both the text-only and the uploaded render derive from one fetch, and
        the memory allowlist reads the same messages, without a second history round-trip.
        """
        hist_messages: list[Message] = []
        async for m in message.channel.history(limit=limit, before=message, oldest_first=True):
            hist_messages.append(m)
        return hist_messages

    async def _render_history(
        self, hist_messages: list[Message], *, text_only: bool
    ) -> list[EasyInputMessageParam]:
        """Renders fetched history in one mode: text-only markers, or full uploaded parts.

        Both modes derive from the same `_fetch_history` result (one Discord call). The
        text-only twin (no upload) feeds routing + memory selection so neither waits on the
        Files API; the full render uploads attachment parts for the answer. History is the only
        render that opts into the dead-source skip: an expired CDN attachment here re-fails every
        turn (current / reference do not; see GeminiFileUploader._resolve_file_upload).
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
        logfire.info("gen_reply video generation start", message_id=message.id)
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
            model=self.runtime_models.media_reply_model,
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
        attached media). Injects only the selected user memory (already source-filtered) plus the
        author's tone note, never the server memory block, and seeds the
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
            # injecting only the selected user memory (already source-filtered) and the
            # author's tone note, never the server memory block.
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
                model_effort=model.effort,
            )
            with logfire.span(span_name, model=model.name):
                responses = await self.openai_client.responses.create(
                    model=model.name,
                    instructions=_build_runtime_instructions(
                        system_prompt=system_prompt, message=message
                    ),
                    input=response_input,
                    reasoning=model.reasoning,
                    stream=True,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": message.author.name},
                    extra_body={"mock_testing_fallbacks": False},
                )
                await streamer.stream(responses=responses)
        except Exception as exc:
            logfire.warn(
                "Media persona reply failed; leaving the delivered media without a reply",
                media=media_noun,
                message_id=message.id,
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
            model=self.runtime_models.media_reply_model,
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
            )
        return reference_messages, current_message

    async def _route_classify(
        self,
        message: Message,
        reference_messages: list[EasyInputMessageParam],
        current_message: list[EasyInputMessageParam],
    ) -> RouteClassification:
        """Classifies the message into a reply mode using pre-built context parts.

        Only the handler choice is decided here; the answer effort is graded by
        `_grade_effort` in a parallel call, so this stays a short single-purpose
        classification on the critical path. The reference + current parts arrive already
        text-only (attachment markers, no file ids), so the route classifies on the text
        without reading or waiting on uploads.
        """
        message_list = [*reference_messages, *current_message]

        fast_model = self.runtime_models.fast_model
        started = time.monotonic()
        try:
            with logfire.span("gen_reply route"):
                responses = await self.openai_client.responses.parse(
                    model=fast_model.name,
                    instructions=ROUTE_PROMPT,
                    input=cast("ResponseInputParam", message_list),
                    text_format=RouteClassification,
                    reasoning=fast_model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": message.author.name},
                    extra_body={"mock_testing_fallbacks": False},
                )
            parsed = responses.output_parsed
            if parsed is None:
                route = RouteClassification(decision="QA")
            elif parsed.decision == "SUMMARY" and (
                _first_url_match(pattern=_MESSAGE_URL_RE, message=message) is not None
            ):
                # A summary request carrying a URL is really a QA recap of that link, not a
                # recap of channel history, so steer it back to QA. Preserve watch_video so a
                # "summarize this YouTube link" still reaches the video-watching path.
                route = RouteClassification(decision="QA", watch_video=parsed.watch_video)
            else:
                route = parsed
        except ValidationError as exc:
            # `responses.parse` validates before `output_parsed` is reachable, so an empty /
            # safety-filtered response and a genuine schema mismatch both land here; the
            # attached exception is the only way to tell them apart.
            logfire.warn(
                "RouteClassification parse failed; defaulting to QA",
                message_id=message.id,
                _exc_info=exc,
            )
            route = RouteClassification(decision="QA")
        # Route-call latency is logged on every path: this is the prime suspect for slow
        # replies, so the log file must show its duration directly, not just a span start.
        logfire.info(
            "gen_reply route done",
            elapsed_seconds=time.monotonic() - started,
            decision=route.decision,
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
        the grade is consumed only on the QA and SUMMARY paths, while IMAGE and VIDEO cancel
        this task. The parts arrive already text-only, so grading never waits on uploads.
        Raises on any provider/parse failure so the caller (`_resolve_effort`) can fall back.
        """
        message_list = [*reference_messages, *current_message]

        fast_model = self.runtime_models.fast_model
        started = time.monotonic()
        with logfire.span("gen_reply effort"):
            responses = await self.openai_client.responses.parse(
                model=fast_model.name,
                instructions=EFFORT_PROMPT,
                input=cast("ResponseInputParam", message_list),
                text_format=EffortGrade,
                reasoning=fast_model.reasoning,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
                extra_body={"mock_testing_fallbacks": False},
            )
        parsed = responses.output_parsed
        grade = parsed if parsed is not None else EffortGrade(effort="high")
        logfire.info(
            "gen_reply effort done",
            elapsed_seconds=time.monotonic() - started,
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
    ) -> Literal["low", "medium", "high"]:
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
                _exc_info=exc,
            )
            return "high"
        except Exception as e:
            logfire.warn(
                "Effort grading failed; defaulting to high effort",
                message_id=message.id,
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
        route_done: asyncio.Event,
        on_timeout: "Callable[[], list[EasyInputMessageParam]]",
    ) -> list[EasyInputMessageParam]:
        """Resolves a parallel linked-post build, bounded by the route like effort.

        On the post-route grace timeout it injects a short "could not read it in time" notice
        instead of nothing, so a slow build keeps deterministic context rather than re-exposing
        the "I cannot open this link" fallback; on any other unexpected error it returns []
        (cancellation propagates). The builders themselves never raise (they degrade to their
        own notices).
        """
        started = time.monotonic()
        try:
            blocks = await _await_gated(
                task=link_task,
                label=source,
                route_done=route_done,
                grace_seconds=LINK_CONTEXT_GRACE_SECONDS,
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
        allowed: dict[int, str],
        read_context: MemoryReadContext,
        server_memory_block: EasyInputMessageParam | None = None,
    ) -> MemorySelection:
        """Phase 1 of a reply: lets the model choose whose long-term memory to read.

        Runs an isolated request offering only the get_user_memory tool (the read path is split
        into a selection phase and an answer phase on purpose, not a hard limit), then resolves
        the chosen ids server-side against the allowlist. The current guild's server memory rides in front as
        background context so a spoken nickname can be mapped to its user id. Returns the
        memories plus this request's token usage so the reply footer and chat reward account
        for the selection call too.
        """
        tool_model = self.runtime_models.tool_model
        # The callable-users block stays last so the model reads it right before deciding;
        # the server-memory block (if any) leads as earlier background context. The caller
        # passes an already text-only transcript (attachment markers, no file ids), so this
        # request neither re-reads the uploaded payloads nor waits on their upload.
        selection_input: ResponseInputParam = [
            *([server_memory_block] if server_memory_block is not None else []),
            *message_list,
            render_callable_users_block(allowed=allowed),
        ]
        responses = await self.openai_client.responses.create(
            model=tool_model.name,
            instructions=MEMORY_SELECT_PROMPT,
            input=selection_input,
            reasoning=tool_model.reasoning,
            tools=[GET_USER_MEMORY_TOOL],
            stream=False,
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"mock_testing_fallbacks": False},
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
        # Bound how many memories ride into the answer request so a pathological multi-user
        # lookup (e.g. a message mentioning many people) can't bloat or overrun it. Each
        # main.md can be tens of KB before compaction; keep the first few in selection order.
        max_memories = 8
        if len(memories) > max_memories:
            logfire.warn(
                "Capping selected memories to the per-reply limit",
                requested=len(memories),
                kept=max_memories,
                message_id=message.id,
            )
            memories = memories[:max_memories]
        input_tokens = responses.usage.input_tokens if responses.usage else 0
        output_tokens = responses.usage.output_tokens if responses.usage else 0
        return MemorySelection(
            memories=memories, input_tokens=input_tokens, output_tokens=output_tokens
        )

    def _participant_memory_fallback(
        self, *, message: Message, allowed: dict[int, str], read_context: MemoryReadContext
    ) -> tuple[EasyInputMessageParam | None, list[str]]:
        """Builds the fallback memory block from the author plus any reply-reference authors.

        A selection timeout or error never returned a decision, so instead of dropping
        memory the reply falls back to the long-term memory of the most relevant
        participants: the message author (always allowlisted as a conversation author)
        and, when the message is a reply, whoever it replies to up the reference chain.
        Replying to someone is a strong signal their memory is relevant, so a failed
        selection should still surface it. Ids are deduped, kept in author-first order,
        and gated through `allowed` (the permission boundary, with the bot already
        removed, so a reply to the bot's own message reads no memory for it). Only
        participants with stored memory contribute, so a fallback never injects an empty
        block; returns (None, []) when none of them have memory. A completed selection
        that deliberately picked nobody is different and is still honored (that path does
        not call this).
        """
        candidate_ids = [
            message.author.id,
            *(ref.author.id for ref in _walk_reference_chain(message=message)),
        ]
        # Delegates to the selection path's resolver so the allowlist gate, the
        # per-bullet source filter, and dedupe live in exactly one place; only
        # candidates whose memory survives filtering contribute.
        memories = [
            memory
            for memory in resolve_user_memories(
                user_id_list=[str(user_id) for user_id in candidate_ids],
                allowed=allowed,
                context=read_context,
            )
            if memory.memory != NO_STORED_MEMORY
        ]
        if not memories:
            return None, []
        return render_memory_context_block(memories=memories), memory_lookup_labels(
            memories=memories
        )

    def _read_server_memory(self, *, message: Message, memory_enabled: bool) -> str:
        """Reads the current guild's raw server memory, or "" when there is none.

        Unlike user memory there is exactly one server memory per guild, so it needs no
        selection phase, allowlist, or function tool: it is read directly with zero extra
        LLM latency. Returns "" for DMs (no guild), the SUMMARY route, or an empty memory.
        Read once per reply and shared by the selection and answer phases.
        """
        if not memory_enabled or self.bot.user is None or message.guild is None:
            return ""
        return read_main_memory(
            scope=server_scope(bot_id=self.bot.user.id, server_id=message.guild.id)
        )

    def _schedule_server_memory_update(
        self, *, message: Message, message_list: list[EasyInputMessageParam], full_reply: str
    ) -> None:
        """Schedules the bot's per-server memory update for a guild message.

        Server memory learns community-level signal from the whole conversation (no
        target-centering, since every message is server context). Skipped for DMs and
        for channels not visible to `@everyone`, so private / restricted-channel content
        never enters the server-wide memory any member can read.
        """
        if self.bot.user is None or message.guild is None:
            return
        if not _source_channel_is_public(message=message):
            return
        schedule_memory_update(
            scope=server_scope(bot_id=self.bot.user.id, server_id=message.guild.id),
            subject=f"target_server_id: {message.guild.id}",
            message_list=message_list,
            full_reply=full_reply,
            extractor=self.server_memory_extractor,
            identity=render_server_identity(
                server_name=message.guild.name, server_id=message.guild.id
            ),
        )

    async def _prepare_reply_context(  # noqa: PLR0913, PLR0915 -- speculative prep needs the turn payload plus the route-done signal, and builds history/memory/tone/selection in sequence
        self,
        message: Message,
        history_limit: int,
        memory_enabled: bool,
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

        # Fetch channel history once (one Discord call), then render its cheap text-only twin up
        # front so memory selection can start on text-only renders alone. The upload-bearing full
        # render is awaited later, concurrently with the in-flight selection, so selection
        # overlaps the Files-API upload window (which the answer must wait on regardless) instead
        # of running serially after it.
        raw_history = await self._fetch_history(message=message, limit=history_limit)
        history_text_only = (
            await self._render_history(raw_history, text_only=True) if memory_enabled else []
        )

        # The bot's own per-server memory is read once here and shared by both phases: it
        # primes selection (a `## 成員稱呼` nickname table maps spoken aliases to ids) and
        # rides into the answer as background context. One file read, no extra LLM call.
        server_memory = self._read_server_memory(message=message, memory_enabled=memory_enabled)
        server_memory_block = (
            render_server_memory_block(memory=server_memory) if server_memory else None
        )

        # Where this reply is happening, for per-bullet source scoping of every user
        # memory read (selection resolution and the participant fallback alike).
        read_context = memory_read_context(message=message)

        # The message author's tone-preference note is read directly for that one author
        # (their own preference for how the bot should sound, cross-server safe by
        # construction) and injected on every reply with no selection phase — even on the
        # SUMMARY route, which skips user memory. One file read, no extra LLM call.
        author_tone = read_tone(scope=user_scope(user_id=message.author.id))
        tone_block = render_tone_block(tone=author_tone) if author_tone else None

        # Memory retrieval is two-phase: phase 1 lets the model pick whose long-term memory to
        # read via get_user_memory (no built-in tools), and phase 2 streams the answer with the
        # built-in tools always available and any selected memory injected as context. The
        # allowlist (conversation authors + mentioned users, minus the bot) is the permission
        # boundary.
        # The split is deliberate, not a hard limit: by default LiteLLM silently drops grounding
        # when a function tool and built-in search/url tools mix, and the Gemini 3
        # include_server_side_tool_invocations opt-out that lifts it is Preview-only. Splitting
        # also keeps selection on a cheaper/faster model off the answer's critical path and stays
        # provider-neutral (OpenAI / Claude mix tools fine), so it stays correct if the answer
        # model changes.
        memory_labels: list[str] = []
        selection_input_tokens = 0
        selection_output_tokens = 0
        memory_block: EasyInputMessageParam | None = None
        allowed: dict[int, str] = {}
        selection_task: asyncio.Task[MemorySelection] | None = None
        if memory_enabled and self.bot.user:
            # The allowlist needs raw Message objects (authors + mentions): the current
            # message, its reference chain, and the raw side of the shared history fetch.
            allowed = build_memory_allowlist(
                messages=[message, *_walk_reference_chain(message=message), *raw_history],
                bot_user_id=self.bot.user.id,
            )
            # Enrich participant labels with their community aliases in every guild channel,
            # but only widen the boundary with absent members' ids in public channels: the
            # nickname table is public, yet an absent member's personal memory is not, so
            # widening in a private channel would leak it. DMs have no guild and keep the
            # conversation-only boundary.
            if server_memory and message.guild is not None:
                widen_allowlist_with_aliases(
                    allowed=allowed,
                    memory=server_memory,
                    include_absent=_source_channel_is_public(message=message),
                )
            logfire.debug(
                "gen_reply memory allowlist built",
                allowlist_size=len(allowed),
                widened=bool(server_memory and message.guild is not None),
                message_id=message.id,
            )
            if allowed:
                # Start selection now (the text-only renders are ready) so it overlaps the upload
                # wait below instead of running serially after it. It runs on the text-only
                # transcript (markers, no file ids), so it never re-reads or blocks on the uploads.
                selection_message_list: list[EasyInputMessageParam] = [
                    *history_text_only,
                    *text_reference,
                    *text_current,
                ]
                selection_task = asyncio.create_task(
                    coro=self._select_user_memories(
                        message=message,
                        message_list=selection_message_list,
                        allowed=allowed,
                        read_context=read_context,
                        server_memory_block=server_memory_block,
                    )
                )

        try:
            # The answer needs the uploaded renders; await the full history render and the shared
            # reference/current uploads here, concurrently with any in-flight selection above.
            # `parts_task` is shielded so cancelling this speculative prep (non-QA routes) never
            # cancels the shared upload task a SUMMARY route still reuses; the full history render
            # rides as an ordinary gather child, so it is cancelled together with prep.
            with logfire.span("gen_reply context build"):
                hist_messages, (reference_messages, current_message) = await asyncio.gather(
                    self._render_history(raw_history, text_only=False), asyncio.shield(parts_task)
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
                selection_started = time.monotonic()
                try:
                    with logfire.span("gen_reply memory selection"):
                        selection = await _await_gated(
                            task=selection_task,
                            label="memory selection",
                            route_done=route_done,
                            grace_seconds=MEMORY_SELECT_GRACE_SECONDS,
                        )
                except TimeoutError as exc:
                    logfire.warn(
                        "Memory selection exceeded the post-route grace; falling back to participant memory",
                        grace_seconds=MEMORY_SELECT_GRACE_SECONDS,
                        message_id=message.id,
                        _exc_info=exc,
                    )
                    memory_block, memory_labels = self._participant_memory_fallback(
                        message=message, allowed=allowed, read_context=read_context
                    )
                except Exception:
                    logfire.warn(
                        "Memory selection failed; falling back to participant memory",
                        message_id=message.id,
                        _exc_info=True,
                    )
                    memory_block, memory_labels = self._participant_memory_fallback(
                        message=message, allowed=allowed, read_context=read_context
                    )
                else:
                    selection_input_tokens = selection.input_tokens
                    selection_output_tokens = selection.output_tokens
                    if selection.memories:
                        memory_block = render_memory_context_block(memories=selection.memories)
                        memory_labels = memory_lookup_labels(memories=selection.memories)
                    logfire.info(
                        "gen_reply memory selection done",
                        elapsed_seconds=time.monotonic() - selection_started,
                        selected=len(selection.memories),
                        selected_ids=[memory.user_id for memory in selection.memories],
                        labels=memory_lookup_labels(memories=selection.memories),
                        allowlist_size=len(allowed),
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
        memory_enabled: bool = True,
        effort: Literal["low", "medium", "high"] = "high",
        allow_voice: bool = False,
        allow_image: bool = False,
        allow_music: bool = False,
        allow_video: bool = False,
        allow_research: bool = False,
        yt_url: str | None = None,
    ) -> None:
        """Streams the answer from a pre-built reply context, then schedules memory updates.

        The per-user update is gated by `memory_enabled`; the per-server update always runs
        (subject to its own guild / public-channel guards), so the SUMMARY route still records
        community memory even though it carries `memory_enabled=False`. `allow_voice` enables a
        spoken clip, `allow_image` an inline generated image, `allow_music` an inline generated
        music clip, and `allow_video` an inline generated video clip when the answer model marks
        the reply for it (image / music / video are QA only; voice also rides SUMMARY, which
        otherwise stays text). `yt_url`, set only when the router asked
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
        if allow_research and self.config.deep_research_available:
            system_prompt = f"{system_prompt}\n{DEEP_RESEARCH_INSTRUCTION}"
        slow_model = self.runtime_models.slow_model.model_copy(update={"effort": effort})
        # Keep the current user message LAST so the model answers it. Memory rides earliest as
        # low-authority background; the reference message then sits just above the current
        # message so the reply pair (reference -> current) stays adjacent and reads as the
        # primary context rather than getting buried up near history.
        answer_input: ResponseInputParam = [*context.hist_messages]
        answer_input.extend(
            block
            for block in (context.server_memory_block, context.memory_block, context.tone_block)
            if block is not None
        )
        answer_input.extend(context.reference_messages)
        # The linked post(s) the user pointed at ride just before the current message, each
        # block led by its own separator; empty unless the message carried a link a registered
        # source reads. The order inside is LINK_CONTEXT_SOURCES order.
        answer_input.extend(context.link_blocks)
        answer_input.extend(context.current_message)

        # Seed the streamer with the selection request's usage so the footer and chat reward
        # reflect both LLM calls; the answer stream sums its own usage on top.
        streamer = ResponseStreamer(
            message=message,
            memory_lookups=context.memory_labels,
            input_tokens=context.selection_input_tokens,
            output_tokens=context.selection_output_tokens,
            model_effort=effort,
            voice_generator=voice_generator,
            image_generator=image_generator,
            music_generator=music_generator,
            video_generator=video_generator,
            media_delivery=self.media_delivery,
            input_builder=self.input_builder,
        )
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
        if use_interactions:
            # Persistent marker (added directly, not via the status chain) so it stays after the
            # chain's final reaction to show the reply was grounded in the watched video. The bot's
            # own application emoji `youtube`, usable as a reaction in any guild the bot is in.
            await update_reaction(
                message=message, bot_user=self.bot.user, emoji="<:youtube:1517546722535018596>"
            )
        with logfire.span(
            "gen_reply answer",
            model=slow_model.name,
            backend="interactions" if use_interactions else "responses",
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
                    model=slow_model.name,
                    instructions=_build_runtime_instructions(
                        system_prompt=system_prompt, message=message
                    ),
                    input=answer_input,
                    reasoning=slow_model.reasoning,
                    tools=list(slow_model.tools),
                    stream=True,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": message.author.name},
                    extra_body={"mock_testing_fallbacks": False},
                )
            full_reply = await streamer.stream(responses=responses)
        # A <deep-research> brief the answer model emitted launches a research thread. Done after
        # the stream (and its single media edit) so it never touches the reply's attachment edit;
        # best-effort, gated, and a no-op when the feature is off or no brief was emitted.
        if allow_research and self.config.deep_research_available and streamer.research_brief:
            await _maybe_launch_research(
                bot=self.bot, message=message, anchor=streamer.reply, brief=streamer.research_brief
            )
        if memory_enabled:
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
        # Server memory is not gated by `memory_enabled`: the SUMMARY route runs with it
        # off (no per-user memory) yet its ~100-message digest is high-quality community
        # signal worth recording. DMs and non-public channels are dropped by the guards
        # inside `_schedule_server_memory_update`.
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
                    scope=scope, extractor=extractor, identity=read_main_identity(scope=scope)
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
        # actively driving, so QA does not double-handle a plan-refinement turn there.
        if _in_active_research_thread(bot=self.bot, channel_id=message.channel.id):
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
        prep_task: asyncio.Task[ReplyContext] | None = None
        parts_task: (
            asyncio.Task[tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]]] | None
        ) = None
        effort_task: asyncio.Task[EffortGrade] | None = None
        link_tasks: dict[str, asyncio.Task[list[EasyInputMessageParam]]] = {}
        try:
            with logfire.span("gen_reply pipeline") as pipeline_span:
                pipeline_started = time.monotonic()
                reactions.advance(emoji="<:flowchart:1517561877973045349>")
                # The reference + current attachment uploads (and their activation polls)
                # run in the background and only the answer awaits them. The route and the
                # memory selection use the text-only renders, so neither waits on the Files
                # API. The QA context builds speculatively in parallel with the route call
                # since QA is the dominant route — non-QA routes discard it.
                parts_task = asyncio.create_task(
                    coro=self._get_reference_and_current(message=message)
                )
                # A link a registered source can read (Threads, Douyin) is self-parsed into
                # answer-context blocks: metadata text always, the media downloaded and
                # uploaded to the Files API when the source allows it. Started here so the
                # fetch (the slow half) overlaps the whole route/prep window for free; only
                # the QA route consumes the blocks, other routes cancel the tasks. Resolution
                # is route_done-gated like effort, never a fixed wait.
                for link_source in LINK_CONTEXT_SOURCES:
                    link_match = _first_url_match(pattern=link_source.url_pattern, message=message)
                    if link_match is None:
                        continue
                    link_url = link_match.group(0)
                    # A source's url_filter rejects a matched link it cannot read (e.g. a
                    # Douyin profile or live room, whose regex matches the host, not the
                    # path), which would only spend a rate-limited request to say so.
                    if link_source.url_filter is not None and not link_source.url_filter(
                        url=link_url
                    ):
                        continue
                    link_tasks[link_source.name] = asyncio.create_task(
                        coro=link_source.build(
                            url=link_url,
                            answer_model_is_gemini="gemini" in self.runtime_models.slow_model.name,
                            gemini_client=self.gemini_client_if_configured,
                            allow_media_ingest=link_source.media_ingest_allowed(
                                config=self.config
                            ),
                        )
                    )
                text_reference, text_current = await self._get_reference_and_current(
                    message=message, text_only=True
                )
                # Signals memory selection that the route has returned: selection runs
                # unbounded while this is clear and gets only a short grace once it is set.
                route_done = asyncio.Event()
                prep_task = asyncio.create_task(
                    coro=self._prepare_reply_context(
                        message=message,
                        history_limit=30,
                        memory_enabled=True,
                        parts_task=parts_task,
                        text_parts=(text_reference, text_current),
                        route_done=route_done,
                    )
                )
                # Effort grading rides the same route_done gate as memory selection: it runs
                # in parallel with the route and only the answer model (QA/SUMMARY) consumes
                # it, so IMAGE/VIDEO cancel it below.
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
                route_done.set()
                pipeline_span.set_attribute(key="route", value=route.decision)
                if route.decision in ("IMAGE", "VIDEO"):
                    # IMAGE and VIDEO share identical speculative-task teardown; they differ only
                    # in the status emoji and which media handler runs. Effort and link context
                    # are answer-only, so both are discarded here.
                    await _discard_task(task=effort_task, label="effort", message_id=message.id)
                    effort_task = None
                    await _discard_link_tasks(link_tasks=link_tasks, message_id=message.id)
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
                elif route.decision == "SUMMARY":
                    await _discard_task(task=prep_task, label="prep", message_id=message.id)
                    prep_task = None
                    # A digest recaps channel history, not one linked post, so the linked-post
                    # blocks are not injected here. A URL-bearing SUMMARY is already rerouted
                    # to QA in `_route_classify`, so these are normally absent; discard
                    # defensively.
                    await _discard_link_tasks(link_tasks=link_tasks, message_id=message.id)
                    reactions.advance(emoji="<:stacks:1517562531365912607>")
                    # so it neither biases the digest nor floods extraction, but the
                    # per-server memory is still recorded since the digest is rich
                    # community signal. Cancelling the speculative prep leaves `parts_task`
                    # running (prep awaits it through asyncio.shield), so the shared
                    # reference/current parts are still reused here.
                    context = await self._prepare_reply_context(
                        message=message,
                        history_limit=100,
                        memory_enabled=False,
                        parts_task=parts_task,
                        text_parts=(text_reference, text_current),
                        route_done=route_done,
                    )
                    parts_task = None
                    effort = await self._resolve_effort(
                        message=message, effort_task=effort_task, route_done=route_done
                    )
                    effort_task = None
                    pipeline_span.set_attribute(key="effort", value=effort)
                    _log_pre_answer_latency(started=pipeline_started, decision=route.decision)
                    await self._handle_message_reply(
                        message=message,
                        system_prompt=SUMMARY_PROMPT,
                        context=context,
                        memory_enabled=False,
                        effort=effort,
                        allow_voice=True,
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
                    # The parses ran in parallel since before the route; resolve each under
                    # the same route_done gate and fold the post blocks into the answer
                    # context, in registry order so the splice stays deterministic.
                    if link_tasks:
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
                                    route_done=route_done,
                                    on_timeout=link_source.on_timeout,
                                )
                            )
                        context = context.model_copy(update={"link_blocks": link_blocks})
                    pipeline_span.set_attribute(key="effort", value=effort)
                    # Watch a linked YouTube video only when the router judged the user is asking
                    # about it; the URL itself is taken from the message text or the replied-to
                    # message (never the model) so the answer turn ingests the exact link posted.
                    yt_url = _find_youtube_url(message=message) if route.watch_video else None
                    _log_pre_answer_latency(started=pipeline_started, decision=route.decision)
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
                        yt_url=yt_url,
                    )
                reactions.advance(emoji="<:greencheck:1517565102424068226>")
        finally:
            if prep_task is not None:
                await _discard_task(task=prep_task, label="prep", message_id=message.id)
            if effort_task is not None:
                await _discard_task(task=effort_task, label="effort", message_id=message.id)
            if parts_task is not None:
                await _discard_task(task=parts_task, label="parts", message_id=message.id)
            await _discard_link_tasks(link_tasks=link_tasks, message_id=message.id)


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
