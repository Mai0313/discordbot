"""Cog that routes Discord messages through the AI reply pipeline."""

from io import BytesIO
import re
import base64
from typing import TYPE_CHECKING, Literal, cast
import asyncio
from functools import cached_property
import contextlib

from openai import AsyncOpenAI
import logfire
from nextcord import File, Embed, Message
from pydantic import ValidationError
from nextcord.ext import commands
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.llm import create_litellm_client
from discordbot.typings.llm import LLMConfig
from discordbot.utils.images import get_image_data, convert_base64_to_data_uri
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
    strip_attachment_parts,
)
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

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from openai.types.responses import ResponseFunctionToolCall


_MESSAGE_URL_RE = re.compile(pattern=r"(?i)\b(?:https?://|www\.)\S+")

# Memory selection is a lightweight tool-only preflight; past this deadline the reply
# answers without memory instead of letting a slow proxy stall the whole pipeline.
MEMORY_SELECT_TIMEOUT_SECONDS = 3.0

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


async def _discard_task(task: asyncio.Task[ReplyContext]) -> None:
    """Cancels and drains a speculative context task so its exception is retrieved."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logfire.warn("Speculative reply context build failed off-route", _exc_info=True)


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
            A builder bound to this bot and runtime model catalog.
        """
        return MessageInputBuilder(bot=self.bot, runtime_models=self.runtime_models)

    async def _get_history_message(self, message: Message, limit: int) -> RenderedHistory:
        """Retrieves channel history once, returning rendered context plus raw messages."""
        messages: list[EasyInputMessageParam] = []
        hist_messages: list[Message] = []
        async for m in message.channel.history(limit=limit, before=message, oldest_first=True):
            hist_messages.append(m)

        if hist_messages:
            tasks: list[Awaitable[EasyInputMessageParam]] = []
            for hist_msg in hist_messages:
                task = self.input_builder.process_single_message(message=hist_msg)
                tasks.append(task)
            processed: list[EasyInputMessageParam] = await asyncio.gather(*tasks)

            messages.append(
                EasyInputMessageParam(
                    role="system",
                    content=[
                        ResponseInputTextParam(
                            text="==== Chat History that might be helpful for answering. ====",
                            type="input_text",
                        )
                    ],
                )
            )
            messages.extend(processed)

        return RenderedHistory(rendered=messages, raw=hist_messages)

    async def _get_reference_message(self, message: Message) -> list[EasyInputMessageParam]:
        """Walks the reference chain up to depth 3 and renders each link as context."""
        chain = _walk_reference_chain(message=message)
        if not chain:
            return []

        tasks: list[Awaitable[EasyInputMessageParam]] = []
        for ref in chain:
            task = self.input_builder.process_single_message(message=ref)
            tasks.append(task)
        processed: list[EasyInputMessageParam] = await asyncio.gather(*tasks)

        messages: list[EasyInputMessageParam] = []
        for ref, processed_ref in zip(reversed(chain), reversed(processed), strict=True):
            messages.append(
                EasyInputMessageParam(
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
            )
            messages.append(processed_ref)
        return messages

    async def _get_current_message(self, message: Message) -> list[EasyInputMessageParam]:
        """Processes the current message that needs to be answered."""
        messages: list[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role="system",
                content=[
                    ResponseInputTextParam(
                        text=f"==== Current Message that needs to be answered from {sanitize_identity(value=message.author.display_name)} ({sanitize_identity(value=message.author.name)}) [id: {message.author.id}]. ====",
                        type="input_text",
                    )
                ],
            )
        ]
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
            own_parts, ref_parts = await asyncio.gather(
                self.input_builder.get_attachment_parts(message=message),
                self.input_builder.get_attachment_parts(message=message.reference.resolved),
            )
            attachment_parts = own_parts + ref_parts
        else:
            attachment_parts = await self.input_builder.get_attachment_parts(message=message)

        data_uris: list[str] = []
        for part in attachment_parts:
            if part.get("type") == "input_image" and (image_url := part.get("image_url")):
                data_uris.append(image_url)

        if data_uris:
            tasks = []
            for uri in data_uris:
                tasks.append(asyncio.to_thread(get_image_data, image_file=uri, use_b64=False))
            image_bytes_list: list[bytes] = list(await asyncio.gather(*tasks))
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
        image_url = convert_base64_to_data_uri(image_b64)
        image_description_input: list[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(
                        text="Describe this generated image briefly for the Discord reply.",
                        type="input_text",
                    ),
                    ResponseInputImageParam(
                        image_url=image_url, detail="auto", type="input_image"
                    ),
                ],
            )
        ]
        fast_model = self.runtime_models.fast_model
        image_responses = await self.client.responses.create(
            model=fast_model.name,
            instructions=IMAGE_PROMPT,
            input=cast("ResponseInputParam", image_description_input),
            reasoning=fast_model.reasoning,
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": message.author.name},
            extra_body={"mock_testing_fallbacks": False},
        )
        image_description = (image_responses.output_text or "").strip()
        image_bytes = BytesIO(base64.b64decode(image_b64))
        image_file = File(fp=image_bytes, filename="generated.png")
        final_content = f"{message.author.mention} {image_description}"
        await message.reply(content=final_content, file=image_file)

    async def _get_reference_and_current(
        self, message: Message
    ) -> tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]]:
        """Processes the reference chain and current message once, shared by routing and the answer."""
        reference_messages, current_message = await asyncio.gather(
            self._get_reference_message(message=message),
            self._get_current_message(message=message),
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
        answer deserves; QA and SUMMARY override the slow model's effort with it.
        Attachment parts are reduced to text markers: classification needs the text and
        the fact that an attachment exists, not the payload bytes.
        """
        message_list = strip_attachment_parts(messages=[*reference_messages, *current_message])

        try:
            route_model = self.runtime_models.route_model
            with logfire.span("gen_reply route"):
                responses = await self.client.responses.parse(
                    model=route_model.name,
                    instructions=ROUTE_PROMPT,
                    input=cast("ResponseInputParam", message_list),
                    text_format=RouteDecision,
                    reasoning=route_model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": message.author.name},
                    extra_body={"mock_testing_fallbacks": False},
                )
            if responses.output_parsed is None:
                return RouteDecision(decision="QA")
            route = responses.output_parsed
            if route.decision == "SUMMARY" and _message_has_url(content=message.content):
                return RouteDecision(decision="QA", effort=route.effort)
            return route
        except ValidationError:
            # The model returned no text output (e.g. safety filter, empty response);
            # model_validate_json(None) raises ValidationError before we can inspect output_parsed.
            logfire.warn("RouteDecision parse failed, model returned no text; defaulting to QA")
            return RouteDecision(decision="QA")

    async def _select_user_memories(
        self,
        *,
        message: Message,
        message_list: list[EasyInputMessageParam],
        allowed: dict[int, str],
        server_memory_block: EasyInputMessageParam | None = None,
    ) -> MemorySelection:
        """Phase 1 of a reply: lets the model choose whose long-term memory to read.

        Runs an isolated request offering only the get_user_memory tool (Gemini cannot mix a
        custom function tool with its built-in search/url tools), then resolves the chosen ids
        server-side against the allowlist. The current guild's server memory rides in front as
        background context so a spoken nickname can be mapped to its user id. Returns the
        memories plus this request's token usage so the reply footer and chat reward account
        for the selection call too.
        """
        tool_model = self.runtime_models.tool_model
        # The callable-users block stays last so the model reads it right before deciding;
        # the server-memory block (if any) leads as earlier background context. Attachment
        # parts are reduced to text markers: picking whose memory to read needs the text,
        # not the payload bytes, and the smaller request returns faster.
        selection_input: ResponseInputParam = [
            *([server_memory_block] if server_memory_block is not None else []),
            *strip_attachment_parts(messages=message_list),
            render_callable_users_block(allowed=allowed),
        ]
        responses = await self.client.responses.create(
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

    async def _prepare_reply_context(
        self,
        message: Message,
        history_limit: int,
        memory_enabled: bool,
        parts_task: asyncio.Task[tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]]],
    ) -> ReplyContext:
        """Builds history, shared parts, server memory, and the memory selection result.

        Runs speculatively as its own task concurrent with routing: everything here only
        reads (channel history, memory files, the selection request), so a non-QA route
        can discard it safely.
        """
        with logfire.span("gen_reply context build"):
            history = await self._get_history_message(message=message, limit=history_limit)
            reference_messages, current_message = await parts_task
        hist_messages = history.rendered
        message_list: list[EasyInputMessageParam] = [
            *hist_messages,
            *reference_messages,
            *current_message,
        ]

        # The bot's own per-server memory is read once here and shared by both phases: it
        # primes selection (a `## 成員稱呼` nickname table maps spoken aliases to ids) and
        # rides into the answer as background context. One file read, no extra LLM call.
        server_memory = self._read_server_memory(message=message, memory_enabled=memory_enabled)
        server_memory_block = (
            render_server_memory_block(memory=server_memory) if server_memory else None
        )

        # Gemini cannot use a custom function tool together with its built-in search/url
        # tools, so memory retrieval is two-phase: phase 1 lets the model pick whose
        # long-term memory to read via get_user_memory (no built-in tools), and phase 2
        # streams the answer with the built-in tools always available and any selected
        # memory injected as context. The allowlist (conversation authors + mentioned
        # users, minus the bot) is the permission boundary.
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
                # Memory selection is an optional preflight; a provider/proxy hiccup here must
                # never turn an answerable message into the generic error path.
                try:
                    with logfire.span("gen_reply memory selection"):
                        async with asyncio.timeout(delay=MEMORY_SELECT_TIMEOUT_SECONDS):
                            selection = await self._select_user_memories(
                                message=message,
                                message_list=message_list,
                                allowed=allowed,
                                server_memory_block=server_memory_block,
                            )
                except TimeoutError:
                    logfire.warn(
                        "Memory selection timed out; answering without memory",
                        timeout_seconds=MEMORY_SELECT_TIMEOUT_SECONDS,
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

        return ReplyContext(
            hist_messages=hist_messages,
            reference_messages=reference_messages,
            current_message=current_message,
            server_memory=server_memory,
            server_memory_block=server_memory_block,
            memory_block=memory_block,
            memory_labels=memory_labels,
            selection_input_tokens=selection_input_tokens,
            selection_output_tokens=selection_output_tokens,
        )

    async def _handle_message_reply(
        self,
        message: Message,
        system_prompt: str,
        context: ReplyContext,
        memory_enabled: bool = True,
        effort: Literal["low", "medium", "high"] = "high",
    ) -> None:
        """Streams the answer from a pre-built reply context, then schedules memory updates."""
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
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": message.author.name},
                extra_body={"mock_testing_fallbacks": False},
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
                await message.reply(
                    content=None,
                    embed=error_embed,
                    **embed_spacer_payload(embeds=[error_embed], is_edit=False, target=message),
                )
        finally:
            await reactions.flush()

    async def _run_reply_pipeline(
        self, message: Message, user_prompt: str, reactions: ReactionStatusChain
    ) -> None:
        """Routes the message and dispatches the matching handler with speculative QA context."""
        prep_task: asyncio.Task[ReplyContext] | None = None
        try:
            with logfire.span("gen_reply pipeline") as pipeline_span:
                reactions.advance(emoji="🔀")
                # Reference + current are processed once and shared by routing and the
                # answer; the QA context (history, server memory, memory selection) builds
                # speculatively in parallel with the route call since QA is the dominant
                # route — non-QA routes discard it.
                parts_task = asyncio.create_task(
                    coro=self._get_reference_and_current(message=message)
                )
                prep_task = asyncio.create_task(
                    coro=self._prepare_reply_context(
                        message=message,
                        history_limit=30,
                        memory_enabled=True,
                        parts_task=parts_task,
                    )
                )
                reference_messages, current_message = await parts_task
                route = await self._route_message(
                    message=message,
                    reference_messages=reference_messages,
                    current_message=current_message,
                )
                pipeline_span.set_attribute(key="route", value=route.decision)
                pipeline_span.set_attribute(key="effort", value=route.effort)
                if route.decision == "IMAGE":
                    await _discard_task(task=prep_task)
                    prep_task = None
                    reactions.advance(emoji="🎨")
                    await self._handle_image_reply(message=message, user_prompt=user_prompt)
                elif route.decision == "VIDEO":
                    await _discard_task(task=prep_task)
                    prep_task = None
                    reactions.advance(emoji="🎬")
                    await self._handle_video_reply(message=message, user_prompt=user_prompt)
                elif route.decision == "SUMMARY":
                    await _discard_task(task=prep_task)
                    prep_task = None
                    reactions.advance(emoji="📖")
                    # Summaries digest ~100 channel messages: skip per-user memory
                    # so it neither biases the digest nor floods extraction. The
                    # cancelled speculative prep does not cancel `parts_task`, so the
                    # shared reference/current parts are still reused here.
                    context = await self._prepare_reply_context(
                        message=message,
                        history_limit=100,
                        memory_enabled=False,
                        parts_task=parts_task,
                    )
                    await self._handle_message_reply(
                        message=message,
                        system_prompt=SUMMARY_PROMPT,
                        context=context,
                        memory_enabled=False,
                        effort=route.effort,
                    )
                else:
                    reactions.advance(emoji="💭")
                    # Selection still gates the answer here; if this wait ever needs to go,
                    # the answer could speculatively start without memory and refire when
                    # selection picks some.
                    context = await prep_task
                    prep_task = None
                    await self._handle_message_reply(
                        message=message,
                        system_prompt=REPLY_PROMPT,
                        context=context,
                        effort=route.effort,
                    )
                reactions.advance(emoji="🆗")
        finally:
            if prep_task is not None:
                await _discard_task(task=prep_task)


def setup(bot: commands.Bot) -> None:
    """Adds the ReplyGeneratorCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
