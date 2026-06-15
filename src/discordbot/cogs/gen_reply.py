"""Cog that routes Discord messages through the AI reply pipeline."""

from io import BytesIO
import re
import time
import base64
from typing import TYPE_CHECKING, Literal, cast
import asyncio
from functools import cached_property
import contextlib

from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import File, Embed, Message
from pydantic import ValidationError
from nextcord.ext import commands
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.llm import litellm_call_kwargs, create_litellm_client
from discordbot.typings.llm import LLMConfig
from discordbot.utils.images import convert_base64_to_data_uri
from discordbot.typings.models import RouteDecision, RuntimeModelCatalog
from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.utils.reactions import ReactionStatusChain, update_reaction
from discordbot.cogs._memory.store import user_scope, server_scope, read_main_memory
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.cogs._gen_reply.input import (
    MessageInputBuilder,
    sanitize_identity,
    render_author_identity,
    render_server_identity,
)
from discordbot.cogs._gen_reply.voice import VoiceSynthesizer
from discordbot.cogs._memory.pipeline import schedule_memory_update
from discordbot.cogs._gen_reply.context import ReplyContext, RenderedHistory
from discordbot.cogs._gen_reply.prompts import (
    IMAGE_PROMPT,
    REPLY_PROMPT,
    ROUTE_PROMPT,
    SUMMARY_PROMPT,
    MEMORY_SELECT_PROMPT,
    REQUEST_TIME_CONTEXT_PROMPT,
)
from discordbot.cogs._memory.extraction import MemoryExtractorAI, target_centered_memory_messages
from discordbot.cogs._gen_reply.streaming import ResponseStreamer
from discordbot.cogs._gen_reply.exceptions import extract_friendly_error
from discordbot.cogs._gen_reply.memory_tool import (
    GET_USER_MEMORY_TOOL,
    UserMemory,
    MemorySelection,
    parse_user_id_list,
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
from discordbot.cogs._gen_reply.attachment.select import build_attachment_handler

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from openai.types.responses import ResponseFunctionToolCall


_MESSAGE_URL_RE = re.compile(pattern=r"(?i)\b(?:https?://|www\.)\S+")

# Memory selection overlaps the route call for free: the QA path joins the speculative
# prep task only after the route returns, so selection runs unbounded while the route is
# still in flight. Once the route completes, a still-running selection gets only this grace
# before the reply answers without memory, so a slow selection can never stall the pipeline
# yet a selection that finishes within the (route-dominant) window is never thrown away.
# Tune against the `gen_reply memory selection done` latency log.
MEMORY_SELECT_GRACE_SECONDS = 5.0

# Hard ceiling on the video-generation polling loop so a hung provider job cannot
# leave the message handler waiting forever.
VIDEO_GENERATION_TIMEOUT_SECONDS = 600.0


def _message_has_url(content: str) -> bool:
    """Returns whether the current message carries an explicit URL."""
    return _MESSAGE_URL_RE.search(string=content) is not None


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
    """Prepends per-request time context to the model instructions."""
    message_created_at_asia_taipei = message.created_at.astimezone(tz=TAIWAN_TIMEZONE)
    request_time_context = REQUEST_TIME_CONTEXT_PROMPT.format(
        message_created_at_asia_taipei=message_created_at_asia_taipei.isoformat(timespec="seconds")
    ).strip()
    return f"{request_time_context}\n\n{system_prompt}"


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


def _reference_header(ref: Message) -> EasyInputMessageParam:
    """Builds the system separator that precedes one reference-chain message."""
    return EasyInputMessageParam(
        role="system",
        content=[
            ResponseInputTextParam(
                text=(
                    f"==== Reference Message from {sanitize_identity(value=ref.author.display_name)} "
                    f"({sanitize_identity(value=ref.author.name)}) [id: {ref.author.id}] that might be helpful "
                    "for answering. ===="
                ),
                type="input_text",
            )
        ],
    )


def _current_header(message: Message) -> EasyInputMessageParam:
    """Builds the system separator that precedes the current message."""
    return EasyInputMessageParam(
        role="system",
        content=[
            ResponseInputTextParam(
                text=f"==== Current Message that needs to be answered from {sanitize_identity(value=message.author.display_name)} ({sanitize_identity(value=message.author.name)}) [id: {message.author.id}]. ====",
                type="input_text",
            )
        ],
    )


async def _discard_task[TaskResultT](task: asyncio.Task[TaskResultT]) -> None:
    """Cancels and drains a speculative task so its exception is retrieved."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logfire.warn("Speculative reply context build failed off-route", _exc_info=True)


def _log_pre_answer_latency(started: float, decision: str) -> None:
    """Logs total time from pipeline start to answer dispatch (the user's 'router stage')."""
    logfire.info(
        "gen_reply pre-answer latency",
        elapsed_seconds=time.monotonic() - started,
        decision=decision,
    )


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

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The cached AsyncOpenAI client instance.

        Returns:
            A configured AsyncOpenAI client reused across reply requests.
        """
        return create_litellm_client(config=self.config)

    @cached_property
    def voice_synthesizer(self) -> VoiceSynthesizer:
        """The cached text-to-speech engine for spoken QA replies.

        Returns:
            A synthesizer bound to this cog's client and the catalog's TTS model; the
            caller still gates it on `allow_voice` and `config.voice_reply_enabled`.
        """
        return VoiceSynthesizer(client=self.client, model_name=self.runtime_models.tts_model.name)

    @cached_property
    def memory_extractor(self) -> MemoryExtractorAI:
        """The cached per-user memory extraction service.

        Returns:
            An extractor bound to this cog's client and the phase-1/phase-2
            memory models.
        """
        return MemoryExtractorAI(
            client=self.client,
            extract_model=self.runtime_models.extract_model,
            evaluate_model=self.runtime_models.memory_evaluator_model,
            consolidate_model=self.runtime_models.memories_model,
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
            client=self.client,
            extract_model=self.runtime_models.extract_model,
            evaluate_model=self.runtime_models.memory_evaluator_model,
            consolidate_model=self.runtime_models.memories_model,
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

    async def _get_history_message(
        self, message: Message, limit: int, with_text_only: bool = False
    ) -> RenderedHistory:
        """Retrieves channel history once, returning rendered context plus raw messages.

        The full render (uploaded attachment parts) always feeds the answer; the
        text-only render is built only when `with_text_only`, since just the memory
        selection call reads it and the SUMMARY route should not pay for a second pass.
        """
        started = time.monotonic()
        messages: list[EasyInputMessageParam] = []
        text_only_messages: list[EasyInputMessageParam] = []
        hist_messages: list[Message] = []
        async for m in message.channel.history(limit=limit, before=message, oldest_first=True):
            hist_messages.append(m)

        if hist_messages:
            full_tasks: list[Awaitable[EasyInputMessageParam]] = []
            for hist_msg in hist_messages:
                # History is the only render that opts into the dead-source skip: an expired
                # CDN attachment here re-fails every turn. Current/reference do not (see
                # GeminiFileUploader._resolve_file_upload).
                full_tasks.append(
                    self.input_builder.process_single_message(
                        message=hist_msg, allow_dead_cache=True
                    )
                )
            text_tasks: list[Awaitable[EasyInputMessageParam]] = []
            if with_text_only:
                for hist_msg in hist_messages:
                    text_tasks.append(
                        self.input_builder.process_single_message_text_only(message=hist_msg)
                    )
            processed, processed_text = await asyncio.gather(
                asyncio.gather(*full_tasks), asyncio.gather(*text_tasks)
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
            messages.append(header)
            messages.extend(processed)
            if with_text_only:
                text_only_messages.append(header)
                text_only_messages.extend(processed_text)

        logfire.info(
            "gen_reply history render done",
            elapsed_seconds=time.monotonic() - started,
            message_count=len(hist_messages),
        )
        return RenderedHistory(
            rendered=messages, rendered_text_only=text_only_messages, raw=hist_messages
        )

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
            messages.append(_reference_header(ref=ref))
            messages.append(processed_ref)
        return messages

    async def _get_current_message(
        self, message: Message, text_only: bool = False
    ) -> list[EasyInputMessageParam]:
        """Processes the current message that needs to be answered."""
        messages: list[EasyInputMessageParam] = [_current_header(message=message)]
        if text_only:
            current_msg = await self.input_builder.process_single_message_text_only(
                message=message
            )
        else:
            current_msg = await self.input_builder.process_single_message(message=message)
        messages.append(current_msg)
        return messages

    async def _handle_video_reply(self, message: Message, user_prompt: str) -> None:
        """Handles video generation requests."""
        video_model = self.runtime_models.video_model
        video = await self.client.videos.create(
            model=video_model.name,
            prompt=user_prompt or "請依照訊息內容生成一段影片。",
            extra_headers={"x-litellm-end-user-id": message.author.name},
        )
        async with asyncio.timeout(delay=VIDEO_GENERATION_TIMEOUT_SECONDS):
            while video.status not in ("completed", "failed"):
                await asyncio.sleep(5)
                video = await self.client.videos.retrieve(
                    video_id=video.id, extra_headers={"x-litellm-end-user-id": message.author.name}
                )
        if video.status != "completed":
            raise RuntimeError(f"Video generation failed: {video.error}")
        video_content = await self.client.videos.download_content(
            video_id=video.id, extra_headers={"x-litellm-end-user-id": message.author.name}
        )
        video_file = File(fp=BytesIO(video_content.content), filename="generated.mp4")
        await message.reply(content=f"{message.author.mention}", file=video_file)

    async def _handle_image_reply(self, message: Message, user_prompt: str) -> None:
        """Handles image generation or editing requests."""
        image_model = self.runtime_models.image_model
        if message.reference and isinstance(message.reference.resolved, Message):
            own_bytes, ref_bytes = await asyncio.gather(
                self.input_builder.get_image_source_bytes(message=message),
                self.input_builder.get_image_source_bytes(message=message.reference.resolved),
            )
            image_bytes_list = own_bytes + ref_bytes
        else:
            image_bytes_list = await self.input_builder.get_image_source_bytes(message=message)

        if image_bytes_list:
            result = await self.client.images.edit(
                image=image_bytes_list,
                prompt=user_prompt or "請依照附件內容進行編輯或優化。",
                model=image_model.name,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )
        else:
            result = await self.client.images.generate(
                prompt=user_prompt or "請生成一張圖片。",
                model=image_model.name,
                n=1,
                response_format="b64_json",
                quality="auto",
                size="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
            )

        if not result.data:
            raise ValueError("Image operation returned no results")
        image_b64 = result.data[0].b64_json
        if image_b64 is None:
            raise ValueError("Image operation returned no b64_json")
        # Send the generated image immediately so the user sees it without waiting on the
        # caption round-trip; the caption is edited into the same message once it returns.
        image_file = File(fp=BytesIO(base64.b64decode(image_b64)), filename="generated.png")
        reply = await message.reply(content=message.author.mention, file=image_file)

        image_description_input: list[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(
                        text="Describe this generated image briefly for the Discord reply.",
                        type="input_text",
                    ),
                    ResponseInputImageParam(
                        image_url=convert_base64_to_data_uri(image_b64),
                        detail="auto",
                        type="input_image",
                    ),
                ],
            )
        ]
        fast_model = self.runtime_models.fast_model
        try:
            image_responses = await self.client.responses.create(
                model=fast_model.name,
                instructions=IMAGE_PROMPT,
                input=cast("ResponseInputParam", image_description_input),
                reasoning=fast_model.reasoning,
                **litellm_call_kwargs(end_user_id=message.author.name),
            )
            image_description = (image_responses.output_text or "").strip()
        except Exception:
            # The image is already delivered; a caption failure must not surface as an error.
            logfire.warn("Image caption failed; leaving image uncaptioned", _exc_info=True)
            image_description = ""
        if image_description:
            await reply.edit(content=f"{message.author.mention} {image_description}")

    async def _get_reference_and_current(
        self, message: Message
    ) -> tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]]:
        """Renders the reference chain and current message with uploaded attachment parts.

        This is the answer-path render (uploads + activation poll to ACTIVE); it runs in
        the background so only the answer awaits the Files API.
        """
        started = time.monotonic()
        reference_messages, current_message = await asyncio.gather(
            self._get_reference_message(message=message),
            self._get_current_message(message=message),
        )
        logfire.info(
            "gen_reply attachment render done",
            elapsed_seconds=time.monotonic() - started,
            reference_count=len(reference_messages),
            current_count=len(current_message),
        )
        return reference_messages, current_message

    async def _get_reference_and_current_text_only(
        self, message: Message
    ) -> tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]]:
        """Renders reference + current as attachment markers for route and memory selection."""
        reference_messages, current_message = await asyncio.gather(
            self._get_reference_message(message=message, text_only=True),
            self._get_current_message(message=message, text_only=True),
        )
        return reference_messages, current_message

    async def _route_message(
        self,
        message: Message,
        reference_messages: list[EasyInputMessageParam],
        current_message: list[EasyInputMessageParam],
    ) -> RouteDecision:
        """Routes the message to the appropriate handler using pre-built context parts.

        Besides the handler choice, the route also grades how much reasoning effort the
        answer deserves; QA and SUMMARY override the slow model's effort with it. The
        reference + current parts arrive already text-only (attachment markers, no file
        ids), so the route classifies on the text without reading or waiting on uploads.
        """
        message_list = [*reference_messages, *current_message]

        route_model = self.runtime_models.route_model
        started = time.monotonic()
        try:
            with logfire.span("gen_reply route"):
                responses = await self.client.responses.parse(
                    model=route_model.name,
                    instructions=ROUTE_PROMPT,
                    input=cast("ResponseInputParam", message_list),
                    text_format=RouteDecision,
                    reasoning=route_model.reasoning,
                    **litellm_call_kwargs(end_user_id=message.author.name),
                )
            parsed = responses.output_parsed
            if parsed is None:
                route = RouteDecision(decision="QA")
            elif parsed.decision == "SUMMARY" and _message_has_url(content=message.content):
                route = RouteDecision(decision="QA", effort=parsed.effort)
            else:
                route = parsed
        except ValidationError:
            # The model returned no text output (e.g. safety filter, empty response);
            # model_validate_json(None) raises ValidationError before we can inspect output_parsed.
            logfire.warn("RouteDecision parse failed, model returned no text; defaulting to QA")
            route = RouteDecision(decision="QA")
        # Route-call latency is logged on every path: this is the prime suspect for slow
        # replies, so the log file must show its duration directly, not just a span start.
        logfire.info(
            "gen_reply route done",
            elapsed_seconds=time.monotonic() - started,
            decision=route.decision,
            effort=route.effort,
        )
        return route

    async def _select_user_memories(
        self,
        *,
        message: Message,
        message_list: list[EasyInputMessageParam],
        allowed: dict[int, str],
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
        responses = await self.client.responses.create(
            model=tool_model.name,
            instructions=MEMORY_SELECT_PROMPT,
            input=selection_input,
            reasoning=tool_model.reasoning,
            tools=[GET_USER_MEMORY_TOOL],
            stream=False,
            **litellm_call_kwargs(end_user_id=message.author.name),
        )
        memories: list[UserMemory] = []
        seen: set[str] = set()
        for item in responses.output:
            if item.type != "function_call":
                continue
            call = cast("ResponseFunctionToolCall", item)
            if call.name != "get_user_memory":
                continue
            for memory in resolve_user_memories(
                user_id_list=parse_user_id_list(arguments=call.arguments), allowed=allowed
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
            )
            memories = memories[:max_memories]
        input_tokens = responses.usage.input_tokens if responses.usage else 0
        output_tokens = responses.usage.output_tokens if responses.usage else 0
        return MemorySelection(
            memories=memories, input_tokens=input_tokens, output_tokens=output_tokens
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

    async def _await_selection_gated(
        self, *, selection_task: "asyncio.Task[MemorySelection]", route_done: asyncio.Event
    ) -> MemorySelection:
        """Awaits memory selection, bounded by the route call instead of a fixed timeout.

        Selection overlaps the route for free, so while the route is still in flight it may
        run unbounded; once the route completes a still-running selection gets only
        `MEMORY_SELECT_GRACE_SECONDS` more before this raises TimeoutError and the reply
        answers without memory. The selection task is always cancelled on exit so it never
        orphans (e.g. when the speculative prep task is discarded on a non-QA route).
        """
        route_wait = asyncio.create_task(coro=route_done.wait())
        try:
            await asyncio.wait({selection_task, route_wait}, return_when=asyncio.FIRST_COMPLETED)
            if selection_task.done():
                return selection_task.result()
            return await asyncio.wait_for(fut=selection_task, timeout=MEMORY_SELECT_GRACE_SECONDS)
        finally:
            route_wait.cancel()
            with contextlib.suppress(BaseException):
                await route_wait
            if not selection_task.done():
                selection_task.cancel()
                with contextlib.suppress(BaseException):
                    await selection_task

    async def _prepare_reply_context(  # noqa: PLR0913 -- speculative prep needs the turn payload plus the route-done signal
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
        with logfire.span("gen_reply context build"):
            history = await self._get_history_message(
                message=message, limit=history_limit, with_text_only=memory_enabled
            )
            # Shielded so cancelling this speculative prep (non-QA routes) does not
            # propagate into the shared upload task: a SUMMARY route cancels prep while
            # still reusing `parts_task`, and an unshielded `await` would cancel it too.
            reference_messages, current_message = await asyncio.shield(parts_task)
        # Covers the history fetch/render plus waiting on the shared attachment upload, so
        # the log separates pre-answer attachment cost from the route-call cost.
        logfire.info(
            "gen_reply context build done", elapsed_seconds=time.monotonic() - build_started
        )
        hist_messages = history.rendered

        # The bot's own per-server memory is read once here and shared by both phases: it
        # primes selection (a `## 成員稱呼` nickname table maps spoken aliases to ids) and
        # rides into the answer as background context. One file read, no extra LLM call.
        server_memory = self._read_server_memory(message=message, memory_enabled=memory_enabled)
        server_memory_block = (
            render_server_memory_block(memory=server_memory) if server_memory else None
        )

        # Memory retrieval is two-phase: phase 1 lets the model pick whose long-term
        # memory to read via get_user_memory (no built-in tools), and phase 2 streams the
        # answer with the built-in tools always available and any selected memory injected
        # as context. The allowlist (conversation authors + mentioned users, minus the bot)
        # is the permission boundary.
        # The split is deliberate, not a hard limit: by default LiteLLM silently drops
        # grounding when a function tool and built-in search/url tools mix, and the Gemini 3
        # include_server_side_tool_invocations opt-out that lifts it is Preview-only.
        # Splitting also keeps selection on a cheaper/faster model off the answer's critical
        # path and stays provider-neutral (OpenAI / Claude mix tools fine), so it stays
        # correct if the answer model changes.
        memory_labels: list[str] = []
        selection_input_tokens = 0
        selection_output_tokens = 0
        memory_block: EasyInputMessageParam | None = None
        if memory_enabled and self.bot.user:
            # The allowlist needs raw Message objects (authors + mentions): the current
            # message, its reference chain, and the raw side of the shared history fetch.
            allowed = build_memory_allowlist(
                messages=[message, *_walk_reference_chain(message=message), *history.raw],
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
            if allowed:
                # Selection runs on the text-only transcript (markers, no file ids) so it
                # neither re-reads the uploaded files nor blocks on their upload.
                selection_message_list: list[EasyInputMessageParam] = [
                    *history.rendered_text_only,
                    *text_reference,
                    *text_current,
                ]
                # Memory selection is an optional preflight; a provider/proxy hiccup here must
                # never turn an answerable message into the generic error path.
                selection_started = time.monotonic()
                try:
                    with logfire.span("gen_reply memory selection"):
                        selection_task = asyncio.create_task(
                            coro=self._select_user_memories(
                                message=message,
                                message_list=selection_message_list,
                                allowed=allowed,
                                server_memory_block=server_memory_block,
                            )
                        )
                        selection = await self._await_selection_gated(
                            selection_task=selection_task, route_done=route_done
                        )
                except TimeoutError:
                    logfire.warn(
                        "Memory selection exceeded the post-route grace; answering without memory",
                        grace_seconds=MEMORY_SELECT_GRACE_SECONDS,
                    )
                except Exception:
                    logfire.warn(
                        "Memory selection failed; answering without memory", _exc_info=True
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
                    )

        return ReplyContext(
            hist_messages=hist_messages,
            reference_messages=reference_messages,
            current_message=current_message,
            server_memory_block=server_memory_block,
            memory_block=memory_block,
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
    ) -> None:
        """Streams the answer from a pre-built reply context, then schedules memory updates.

        The per-user update is gated by `memory_enabled`; the per-server update always runs
        (subject to its own guild / public-channel guards), so the SUMMARY route still records
        community memory even though it carries `memory_enabled=False`. `allow_voice` enables a
        spoken clip when the answer model marks the reply for it (QA only; SUMMARY stays text).
        """
        voice_synthesizer = (
            self.voice_synthesizer if allow_voice and self.config.voice_reply_enabled else None
        )
        slow_model = self.runtime_models.slow_model.model_copy(update={"effort": effort})
        # Keep the current user message LAST so the model answers it rather than continuing
        # the assistant memory note: the memory rides as earlier context, after history and
        # reference but before the current message.
        answer_input: ResponseInputParam = [*context.hist_messages, *context.reference_messages]
        answer_input.extend(
            block
            for block in (context.server_memory_block, context.memory_block)
            if block is not None
        )
        answer_input.extend(context.current_message)

        # Seed the streamer with the selection request's usage so the footer and chat reward
        # reflect both LLM calls; the answer stream sums its own usage on top.
        streamer = ResponseStreamer(
            message=message,
            memory_lookups=context.memory_labels,
            input_tokens=context.selection_input_tokens,
            output_tokens=context.selection_output_tokens,
            model_effort=effort,
            voice_synthesizer=voice_synthesizer,
        )
        with logfire.span("gen_reply answer", model=slow_model.name):
            responses = await self.client.responses.create(
                model=slow_model.name,
                instructions=_build_runtime_instructions(
                    system_prompt=system_prompt, message=message
                ),
                input=answer_input,
                reasoning=slow_model.reasoning,
                tools=list(slow_model.tools),
                stream=True,
                **litellm_call_kwargs(end_user_id=message.author.name),
            )
            full_reply = await streamer.stream(responses=responses)
        if memory_enabled:
            memory_message_list = target_centered_memory_messages(
                hist_messages=context.hist_messages,
                reference_messages=context.reference_messages,
                current_message=context.current_message,
                target_user_id=message.author.id,
            )
            schedule_memory_update(
                scope=user_scope(user_id=message.author.id),
                subject=f"target_user_id: {message.author.id}",
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

        user_prompt = await self.input_builder.get_user_prompt(content=message.content)
        has_attachment = bool(message.attachments or message.stickers)

        if not user_prompt and not has_attachment:
            await update_reaction(message=message, bot_user=self.bot.user, emoji="❓")
            await message.reply(content="?")
            return

        reactions = ReactionStatusChain(message=message, bot_user=self.bot.user)
        try:
            await self._run_reply_pipeline(
                message=message, user_prompt=user_prompt, reactions=reactions
            )
        except Exception as e:
            logfire.error("Failed to generate reply", user_id=message.author.name, _exc_info=True)
            with contextlib.suppress(Exception):
                reactions.advance(emoji="❌")
                error_embed = Embed(
                    title="Something went wrong",
                    description=f"```\n{extract_friendly_error(exc=e)}\n```",
                    color=0xED4245,
                )
                error_embed.set_footer(text=type(e).__name__)
                spacer = embed_spacer_payload(embeds=[error_embed], is_edit=False, target=message)
                try:
                    await message.reply(content=None, embed=error_embed, **spacer)
                except nextcord.HTTPException as send_error:
                    # Source deleted before the error landed (50035): send it unparented. Rebuild
                    # the spacer; the failed reply already consumed the single-use spacer file.
                    if send_error.code != 50035 and not isinstance(send_error, nextcord.NotFound):
                        raise
                    fresh_spacer = embed_spacer_payload(
                        embeds=[error_embed], is_edit=False, target=message
                    )
                    await message.channel.send(content=None, embed=error_embed, **fresh_spacer)
        finally:
            await reactions.flush()

    async def _run_reply_pipeline(  # noqa: PLR0915 -- orchestrates route, speculative prep, and per-route dispatch in sequence
        self, message: Message, user_prompt: str, reactions: ReactionStatusChain
    ) -> None:
        """Routes the message and dispatches the matching handler with speculative QA context."""
        prep_task: asyncio.Task[ReplyContext] | None = None
        parts_task: (
            asyncio.Task[tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]]] | None
        ) = None
        try:
            with logfire.span("gen_reply pipeline") as pipeline_span:
                pipeline_started = time.monotonic()
                reactions.advance(emoji="🔀")
                # The reference + current attachment uploads (and their activation polls)
                # run in the background and only the answer awaits them. The route and the
                # memory selection use the text-only renders, so neither waits on the Files
                # API. The QA context builds speculatively in parallel with the route call
                # since QA is the dominant route — non-QA routes discard it.
                parts_task = asyncio.create_task(
                    coro=self._get_reference_and_current(message=message)
                )
                text_reference, text_current = await self._get_reference_and_current_text_only(
                    message=message
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
                route = await self._route_message(
                    message=message,
                    reference_messages=text_reference,
                    current_message=text_current,
                )
                route_done.set()
                pipeline_span.set_attribute(key="route", value=route.decision)
                pipeline_span.set_attribute(key="effort", value=route.effort)
                if route.decision == "IMAGE":
                    await _discard_task(task=prep_task)
                    prep_task = None
                    # IMAGE loads raw bytes itself, so the background uploads are wasted.
                    await _discard_task(task=parts_task)
                    parts_task = None
                    reactions.advance(emoji="🎨")
                    await self._handle_image_reply(message=message, user_prompt=user_prompt)
                elif route.decision == "VIDEO":
                    await _discard_task(task=prep_task)
                    prep_task = None
                    await _discard_task(task=parts_task)
                    parts_task = None
                    reactions.advance(emoji="🎬")
                    await self._handle_video_reply(message=message, user_prompt=user_prompt)
                elif route.decision == "SUMMARY":
                    await _discard_task(task=prep_task)
                    prep_task = None
                    reactions.advance(emoji="📖")
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
                    _log_pre_answer_latency(started=pipeline_started, decision=route.decision)
                    await self._handle_message_reply(
                        message=message,
                        system_prompt=SUMMARY_PROMPT,
                        context=context,
                        memory_enabled=False,
                        effort=route.effort,
                        allow_voice=True,
                    )
                else:
                    reactions.advance(emoji="💭")
                    # Selection still gates the answer here; if this wait ever needs to go,
                    # the answer could speculatively start without memory and refire when
                    # selection picks some.
                    context = await prep_task
                    prep_task = None
                    parts_task = None
                    _log_pre_answer_latency(started=pipeline_started, decision=route.decision)
                    await self._handle_message_reply(
                        message=message,
                        system_prompt=REPLY_PROMPT,
                        context=context,
                        effort=route.effort,
                        allow_voice=True,
                    )
                reactions.advance(emoji="🆗")
        finally:
            if prep_task is not None:
                await _discard_task(task=prep_task)
            if parts_task is not None:
                await _discard_task(task=parts_task)


def setup(bot: commands.Bot) -> None:
    """Adds the ReplyGeneratorCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
