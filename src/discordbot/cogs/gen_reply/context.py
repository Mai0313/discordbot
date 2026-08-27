"""The reply context: what one turn hands the answer model, and how it is built.

`ReplyContext` is the value; `ReplyContextBuilder` is the single speculative build that produces
it while the route call is still in flight. Everything the builder does only READS — channel
history, memory files, one optional selection request — so a non-QA route can discard it safely,
and the IMAGE / VIDEO routes consume it after their media is on screen instead.
"""

import time
from typing import TYPE_CHECKING
import asyncio

from openai import AsyncOpenAI
import logfire
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.ext import commands
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam

from discordbot.typings.memory import MemoryCredits
from discordbot.typings.timeouts import RECALL_SELECT_GRACE_SECONDS
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.llm_transcript import sanitize_identity
from discordbot.cogs.gen_reply.recall import (
    NO_STORED_MEMORY,
    GET_USER_MEMORY_TOOL,
    UserMemory,
    RecallContext,
    RecallCandidate,
    RecallSelection,
    render_tone_block,
    parse_user_id_list,
    build_recall_context,
    recall_user_memories,
    memory_lookup_credits,
    build_recall_allowlist,
    render_server_memory_block,
    render_callable_users_block,
    render_memory_context_block,
    widen_allowlist_with_aliases,
    allowlist_ids_from_server_memory,
)
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    read_tone,
    user_scope,
    server_scope,
    read_memory_document,
)
from discordbot.cogs.gen_reply.prompts import RECALL_SELECT_PROMPT
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import GeminiKeyToolkit
from discordbot.typings.context_budgets import (
    HISTORY_CHAR_BUDGET,
    MAX_HISTORY_MEDIA_PARTS,
    MEMORY_CONTEXT_TARGET_USERS,
    HISTORY_PER_MESSAGE_OVERHEAD,
)
from discordbot.cogs.gen_reply.references import replied_to_message, source_channel_is_public
from discordbot.cogs.gen_reply.speculation import await_gated, discard_task

if TYPE_CHECKING:
    from collections.abc import Awaitable

type MessageParts = tuple[list[EasyInputMessageParam], list[EasyInputMessageParam]]


class ReplyContext(BaseModel):
    """Reply inputs built once per message and shared across pipeline phases.

    Built speculatively by `ReplyContextBuilder.build` while the route decision is
    still in flight; it carries everything the answer phase needs so that phase adds
    no further context work.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hist_messages: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list, description="Rendered channel-history context blocks."
    )
    reference_messages: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list,
        description="Rendered blocks for the message being replied to; empty when it is not a reply.",
    )
    current_message: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list,
        description="Header plus the processed current message; stays last in the answer input.",
    )
    server_memory_block: SkipValidation[EasyInputMessageParam | None] = Field(
        default=None, description="Rendered server-memory context block, if any."
    )
    memory_block: SkipValidation[EasyInputMessageParam | None] = Field(
        default=None, description="Rendered deterministic and optional user-memory block."
    )
    tone_block: SkipValidation[EasyInputMessageParam | None] = Field(
        default=None, description="Rendered tone-preference block for the message author, if any."
    )
    link_blocks: SkipValidation[list[EasyInputMessageParam]] = Field(
        default_factory=list,
        description=(
            "Rendered linked-post context blocks in LINK_CONTEXT_SOURCES order, "
            "injected before the current message."
        ),
    )
    memory_credits: MemoryCredits = Field(
        default_factory=MemoryCredits,
        description="Footer credits for the users whose memory was injected.",
    )
    selection_input_tokens: int = Field(
        default=0, description="Input tokens spent by the memory selection request."
    )
    selection_output_tokens: int = Field(
        default=0, description="Output tokens spent by the memory selection request."
    )

    @property
    def message_list(self) -> list[EasyInputMessageParam]:
        """History, reference, and current blocks in transcript order."""
        return [*self.hist_messages, *self.reference_messages, *self.current_message]


def trim_history_to_budget(*, messages: list[Message]) -> list[Message]:
    """Keeps the newest history messages that fit `HISTORY_CHAR_BUDGET`, cut on a boundary.

    `ReplyContextBuilder.fetch_history` returns oldest-first, so this walks from the end and
    reverses back: what survives is the conversation closest to the question being answered, and
    the oldest context is what gets dropped. Cutting between messages rather than mid-text is the
    point — half a sentence with no author and no end reads as corrupted context rather than as
    less of it.

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


def history_media_over_budget(
    *, builder: MessageInputBuilder, hist_messages: list[Message]
) -> dict[int, int]:
    """History message ids to how many attachments each renders as markers, newest kept first.

    The count rides along because it is what tells an operator the cap did anything: `media_parts`
    on the dispatch record stops at the cap by construction, so only the number held back says
    whether this turn was trimmed by one file or by thirty.

    Walks from the newest message back, the same direction and for the same reason as
    `trim_history_to_budget`: what keeps its real files is the conversation closest to the
    question being asked. Once one message is refused every older one is too, so the files the
    model gets are always an unbroken run ending at the present. Letting a later small message
    slip into the leftover budget would put an older attachment on screen while a newer one
    showed only a marker, which reads as the pipeline losing files at random.

    Counting is off `collect_attachment_sources` rather than the modality-gated list, so it
    stays free of the per-message log the gate emits; the gate dropped nothing at all across
    the day this was measured, and over-counting can only refuse an attachment the answer was
    never going to be sent.

    The newest message carrying attachments is exempt, so a single post of many images is
    never reduced to nothing but markers while the budget sits unspent. That makes the cap a
    soft one on exactly that message: a source is not only an upload, it is also every sticker
    and every embed image and thumbnail, snapshots included, so one post of ten files carrying
    a few unfurled link cards can exempt well past `MAX_HISTORY_MEDIA_PARTS`. Everything older
    than it is still bounded.
    """
    over: dict[int, int] = {}
    spent = 0
    for candidate in reversed(hist_messages):
        try:
            count = len(builder.collect_attachment_sources(message=candidate))
        except Exception:  # noqa: S112
            # Broad for the same reason `process_single_message` is, and load-bearing here for a
            # different one: this runs inside `ReplyContextBuilder.build`'s gather, which has no
            # except of its own, so an unexpected nextcord shape would take the whole reply out
            # through the generic error path rather than costing one message its attachments.
            # Silent against S112 on purpose: the message is left out of the refusal set, so its
            # own render re-collects a moment later, fails the same way, and that handler logs it
            # with the message id and the traceback. Logging here would double every such failure.
            continue
        if not count:
            continue
        if over or (spent and spent + count > MAX_HISTORY_MEDIA_PARTS):
            over[candidate.id] = count
            continue
        spent += count
    return over


def reference_header(*, ref: Message) -> EasyInputMessageParam:
    """Builds the system separator that precedes the message being replied to.

    Exactly one of these is ever rendered, so it is always the primary context and says so
    plainly. The attachment sentence is the load-bearing half: a Current Message that points at
    something without naming it is pointing here, this message's files included.
    """
    return EasyInputMessageParam(
        role="system",
        content=[
            ResponseInputTextParam(
                text=(
                    f"==== Reference Message from {sanitize_identity(value=ref.author.display_name)} "
                    f"({sanitize_identity(value=ref.author.name)}) [id: {ref.author.id}]. "
                    "The user is directly replying to this message; it is the primary context for "
                    "the Current Message below. When the Current Message points at something "
                    "without naming it, that something is here, this message's attachments "
                    "included. ===="
                ),
                type="input_text",
            )
        ],
    )


def current_header(*, message: Message, has_reference: bool) -> EasyInputMessageParam:
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


class ReplyContextBuilder(BaseModel):
    """Builds one turn's `ReplyContext` from Discord history plus stored memory.

    Everything here reads and nothing writes, which is what lets the pipeline start the build
    speculatively alongside the route call and throw it away when the route turns out not to
    need it.

    Attributes:
        client: The shared LiteLLM-proxy client, for the optional memory-selection request.
        bot: The Discord bot instance, whose user id is excluded from every memory allowlist.
        toolkit: The leased Gemini key's toolkit, which owns the input builder and model tiers.
        message: The message being answered.
        surface: Where this turn is happening, for the history read and the memory compartments.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client used for the memory-selection request."
    )
    bot: SkipValidation[commands.Bot] = Field(
        ..., description="The Discord bot instance, excluded from every memory allowlist."
    )
    toolkit: GeminiKeyToolkit = Field(
        ..., description="The leased Gemini key's clients, model catalog and input builder."
    )
    message: SkipValidation[Message] = Field(..., description="The message being answered.")
    surface: TurnSurface = Field(
        ..., description="Where this turn is happening: its history source and its guild."
    )

    async def fetch_history(self, *, limit: int) -> list[Message]:
        """Fetches up to `limit` history messages once, trimmed to the char budget.

        Returned raw so both the optional selector's text-only render and the answer's
        uploaded render derive from one fetch, without a second walk of history. Where they
        come from is the surface's question: a channel walk on the gateway path, and the
        conversation store on the `/ask` one, which is the only history a user-installed app
        has (it is not a member of the channel and holds no `READ_MESSAGE_HISTORY`).
        """
        return trim_history_to_budget(messages=await self.surface.fetch_history(limit=limit))

    async def render_history(
        self, *, hist_messages: list[Message], text_only: bool
    ) -> list[EasyInputMessageParam]:
        """Renders fetched history in one mode: text-only markers, or full uploaded parts.

        Both modes derive from the same `fetch_history` result, so history is walked once
        however many renders are asked for. The text-only twin (no upload) feeds optional
        memory selection without waiting on the Files API; the full render uploads attachment
        parts for the answer. History is the only render that opts into the dead-source skip:
        an expired CDN attachment here re-fails every turn (current / reference do not; see
        GeminiFileUploader._resolve_file_upload).

        The full render is additionally capped at `MAX_HISTORY_MEDIA_PARTS` uploaded files: a
        message past the cap takes the text-only render, which is exactly the marker form the
        route already reads, so the degradation needs no second render path of its own.
        """
        if not hist_messages:
            return []
        input_builder = self.toolkit.input_builder
        over_budget = (
            {}
            if text_only
            else history_media_over_budget(builder=input_builder, hist_messages=hist_messages)
        )
        tasks: list[Awaitable[EasyInputMessageParam]] = [
            input_builder.process_single_message_text_only(message=m)
            if text_only or m.id in over_budget
            else input_builder.process_single_message(message=m, allow_dead_cache=True)
            for m in hist_messages
        ]
        started = time.monotonic()
        processed = await asyncio.gather(*tasks)
        if not text_only:
            logfire.info(
                "gen_reply history render done",
                elapsed_seconds=time.monotonic() - started,
                message_count=len(hist_messages),
                media_capped=sum(over_budget.values()),
                message_id=self.message.id,
            )
        # Names the block and stops there. The old wording invited the model to answer FROM the
        # history ("that might be helpful for answering"), which competed with the Reference
        # Message's own claim to be the primary context and lost the reply's subject to whatever
        # in the window read as the most answerable thing. Where the subject may come from is a
        # behaviour rule, so it lives in `REPLY_PROMPT` at developer authority instead. Keeping it
        # out of here also keeps it out of the three other calls this render feeds, none of which
        # is answering a question: memory selection, the media persona reply, and the memory
        # review transcript, whose first message is this header verbatim.
        header = EasyInputMessageParam(
            role="system",
            content=[
                ResponseInputTextParam(
                    text="==== Chat History: earlier messages in this channel. ====",
                    type="input_text",
                )
            ],
        )
        return [header, *processed]

    async def render_reference_message(
        self, *, text_only: bool = False
    ) -> list[EasyInputMessageParam]:
        """Renders the message being replied to, or nothing when this is not a reply.

        `text_only` emits attachment markers instead of uploaded file parts, for the
        route and memory-selection calls that must not wait on the Files API.
        """
        replied_to = replied_to_message(message=self.message)
        if replied_to is None:
            return []
        input_builder = self.toolkit.input_builder
        if text_only:
            processed = await input_builder.process_single_message_text_only(message=replied_to)
        else:
            processed = await input_builder.process_single_message(message=replied_to)
        return [reference_header(ref=replied_to), processed]

    async def render_current_message(
        self, *, text_only: bool = False
    ) -> list[EasyInputMessageParam]:
        """Processes the current message that needs to be answered."""
        has_reference = replied_to_message(message=self.message) is not None
        messages: list[EasyInputMessageParam] = [
            current_header(message=self.message, has_reference=has_reference)
        ]
        input_builder = self.toolkit.input_builder
        if text_only:
            current_msg = await input_builder.process_single_message_text_only(
                message=self.message
            )
        else:
            current_msg = await input_builder.process_single_message(message=self.message)
        messages.append(current_msg)
        return messages

    async def render_parts(self, *, text_only: bool = False) -> MessageParts:
        """Renders the message being replied to and the current message together.

        With `text_only` they render as attachment markers (no upload) for the route and memory
        selection; otherwise this is the answer-path render (uploads + activation poll to ACTIVE)
        that runs in the background so only the answer awaits the Files API. The render-timing log
        fires only for the upload-bearing render, the latency-critical one.
        """
        started = time.monotonic()
        reference_messages, current_message = await asyncio.gather(
            self.render_reference_message(text_only=text_only),
            self.render_current_message(text_only=text_only),
        )
        if not text_only:
            logfire.info(
                "gen_reply attachment render done",
                elapsed_seconds=time.monotonic() - started,
                reference_count=len(reference_messages),
                current_count=len(current_message),
                message_id=self.message.id,
            )
        return reference_messages, current_message

    def read_server_memory(self) -> str:
        """Reads the current guild's raw server memory, or "" when there is none.

        Unlike user memory there is exactly one server memory per guild, so it needs no
        selection phase, allowlist, or function tool: it is read directly with zero extra
        LLM latency. Returns "" for a DM (no guild) or an empty memory. Read once per reply
        and shared by the selection and answer phases.

        A `/ask` turn always takes the "" branch, since its synthesized message carries no
        guild — deliberately, because the write side is gated on a public channel this route can
        never satisfy and a memory nothing writes back to is one the bot slowly goes stale on.
        What goes with it is the `## 成員稱呼` table, so the alias widening and the optional
        third-party selector below have nothing to work from either, exactly as in a DM today.
        """
        if self.message.guild is None:
            return ""
        return read_memory_document(
            scope=server_scope(server_id=self.message.guild.id),
            compartments=[GLOBAL_COMPARTMENT],
            flavor="server",
        )

    def _resolve_recall_candidates(
        self, *, server_memory: str, recall_context: RecallContext
    ) -> tuple[list[UserMemory], dict[int, RecallCandidate], int]:
        """Resolves deterministic memories and derives disjoint optional alias candidates."""
        bot_user = self.bot.user
        if bot_user is None:
            return [], {}, 0

        replied_to = replied_to_message(message=self.message)
        deterministic_allowed = build_recall_allowlist(
            users=[
                self.message.author,
                *([replied_to.author] if replied_to is not None else []),
                *self.message.mentions,
            ],
            bot_user_id=bot_user.id,
        )
        optional_allowed: dict[int, RecallCandidate] = {}
        # Existing participant labels keep their community aliases even in a private
        # channel because that grants no new access. Only a public channel may offer absent
        # nickname-table members to the selector.
        if server_memory and self.message.guild is not None:
            widen_allowlist_with_aliases(
                allowed=deterministic_allowed, memory=server_memory, include_absent=False
            )
            if source_channel_is_public(message=self.message):
                # No credit label: the conversation never names these members, so
                # `recall_user_memories` credits them by their bare id. Deliberately NOT the
                # identity their own memory carries, which is the display name of whichever
                # guild's consolidation last wrote that fact (see `RecallCandidate`).
                optional_allowed = {
                    user_id: RecallCandidate(prompt_label=label)
                    for user_id, label in allowlist_ids_from_server_memory(
                        memory=server_memory
                    ).items()
                    if user_id not in deterministic_allowed and user_id != bot_user.id
                }

        memories = [
            memory
            for memory in recall_user_memories(
                user_id_list=[str(user_id) for user_id in deterministic_allowed],
                allowed=deterministic_allowed,
                context=recall_context,
            )
            if memory.memory != NO_STORED_MEMORY
        ]
        return memories, optional_allowed, len(deterministic_allowed)

    async def select_recalled_memories(
        self,
        *,
        message_list: list[EasyInputMessageParam],
        allowed: dict[int, RecallCandidate],
        recall_context: RecallContext,
        server_memory_block: EasyInputMessageParam | None = None,
    ) -> RecallSelection:
        """Lets the model choose optional third-party memories for an oblique reference.

        Runs an isolated request offering only the get_user_memory tool, then resolves the
        chosen ids server-side against an allowlist containing only absent members from a
        public server nickname table. The server memory rides in front as background context
        so a spoken or misspelled nickname can be mapped to its id. Returns the memories plus
        this request's token usage so the reply footer and chat reward account for the call.
        """
        triage_model = self.toolkit.runtime_models.triage_model
        # The optional-candidates block stays last so the model reads it right before deciding;
        # the server-memory block (if any) leads as earlier background context. The caller
        # passes an already text-only transcript (attachment markers, no file ids), so this
        # request neither re-reads the uploaded payloads nor waits on their upload.
        selection_input: ResponseInputParam = [
            *([server_memory_block] if server_memory_block is not None else []),
            *message_list,
            render_callable_users_block(allowed=allowed),
        ]
        responses = await self.client.responses.create(
            model=triage_model.deployment_name,
            instructions=RECALL_SELECT_PROMPT,
            input=selection_input,
            reasoning=triage_model.reasoning,
            tools=[GET_USER_MEMORY_TOOL],
            stream=False,
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": self.message.author.name},
        )
        memories: list[UserMemory] = []
        seen: set[str] = set()
        for item in responses.output:
            if item.type != "function_call":
                continue
            if item.name != "get_user_memory":
                continue
            for memory in recall_user_memories(
                user_id_list=parse_user_id_list(arguments=item.arguments),
                allowed=allowed,
                context=recall_context,
            ):
                if memory.user_id not in seen:
                    seen.add(memory.user_id)
                    memories.append(memory)
        input_tokens = responses.usage.input_tokens if responses.usage else 0
        output_tokens = responses.usage.output_tokens if responses.usage else 0
        return RecallSelection(
            memories=memories, input_tokens=input_tokens, output_tokens=output_tokens
        )

    async def _await_optional_selection(
        self, *, task: asyncio.Task[RecallSelection], route_done: asyncio.Event
    ) -> tuple[RecallSelection, float] | None:
        """Awaits the optional selector without letting its failure affect direct memories."""
        started = time.monotonic()
        try:
            with logfire.span("gen_reply memory selection", message_id=self.message.id):
                selection = await await_gated(
                    task=task,
                    label="memory selection",
                    route_done=route_done,
                    grace_seconds=RECALL_SELECT_GRACE_SECONDS,
                )
        except TimeoutError as exc:
            logfire.warn(
                "Optional memory selection exceeded the post-route grace; retaining deterministic memories",
                grace_seconds=RECALL_SELECT_GRACE_SECONDS,
                message_id=self.message.id,
                model=self.toolkit.runtime_models.triage_model.name,
                _exc_info=exc,
            )
            return None
        except Exception:
            logfire.warn(
                "Optional memory selection failed; retaining deterministic memories",
                message_id=self.message.id,
                model=self.toolkit.runtime_models.triage_model.name,
                _exc_info=True,
            )
            return None
        return selection, time.monotonic() - started

    async def build(
        self,
        *,
        history_limit: int,
        parts_task: asyncio.Task[MessageParts],
        text_parts: MessageParts,
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
        raw_history = await self.fetch_history(limit=history_limit)

        # The bot's own per-server memory is read once here and shared by both phases: it
        # primes selection (a `## 成員稱呼` nickname table maps spoken aliases to ids) and
        # rides into the answer as background context. One file read, no extra LLM call.
        server_memory = self.read_server_memory()
        server_memory_block = (
            render_server_memory_block(memory=server_memory) if server_memory else None
        )

        # Where this reply is happening, for compartment scoping of every user-memory read.
        recall_context = build_recall_context(
            author_id=self.message.author.id,
            guild_id=self.surface.guild_id,
            is_direct_message=self.surface.is_direct_message,
        )

        # The message author's tone-preference note is read directly for that one author
        # (their own preference for how the bot should sound, cross-server safe by
        # construction) and injected on every reply with no selection phase, including one
        # that runs with user memory off. One file read, no extra LLM call.
        author_tone = read_tone(scope=user_scope(user_id=self.message.author.id))
        tone_block = render_tone_block(tone=author_tone) if author_tone else None

        # Code always resolves the current author, reply-chain authors, and current-message
        # mentions. A separate model call is reserved for the one non-mechanical question: does
        # the latest message obliquely refer to an absent member in a public nickname table? Both
        # paths stay behind recall_user_memories, the shared permission and compartment boundary.
        memory_credits = MemoryCredits()
        selection_input_tokens = 0
        selection_output_tokens = 0
        memory_block: EasyInputMessageParam | None = None
        remaining_slots = 0
        selection_task: asyncio.Task[RecallSelection] | None = None
        memories, optional_allowed, deterministic_candidate_count = (
            self._resolve_recall_candidates(
                server_memory=server_memory, recall_context=recall_context
            )
        )
        deterministic_memory_count = len(memories)
        if memories:
            memory_block = render_memory_context_block(memories=memories)
            memory_credits = memory_lookup_credits(memories=memories)

            remaining_slots = max(0, MEMORY_CONTEXT_TARGET_USERS - len(memories))
            logfire.debug(
                "gen_reply memory candidates built",
                deterministic_candidates=deterministic_candidate_count,
                deterministic_memories=len(memories),
                optional_candidates=len(optional_allowed),
                optional_slots=remaining_slots,
                message_id=self.message.id,
            )
            if optional_allowed and remaining_slots:
                # Render the text-only history only for a real optional lookup. This request
                # carries markers instead of file ids, so it never re-reads uploaded payloads.
                history_text_only = await self.render_history(
                    hist_messages=raw_history, text_only=True
                )
                selection_message_list: list[EasyInputMessageParam] = [
                    *history_text_only,
                    *text_reference,
                    *text_current,
                ]
                selection_task = asyncio.create_task(
                    coro=self.select_recalled_memories(
                        message_list=selection_message_list,
                        allowed=optional_allowed,
                        recall_context=recall_context,
                        server_memory_block=server_memory_block,
                    )
                )

        try:
            # The answer needs the uploaded renders; await the full history render and the shared
            # reference/current uploads here, concurrently with any in-flight selection above.
            # `parts_task` is shielded so cancelling this speculative prep (IMAGE / VIDEO) never
            # cancels the shared upload task those routes still reuse; the full history render
            # rides as an ordinary gather child, so it is cancelled together with prep.
            with logfire.span("gen_reply context build", message_id=self.message.id):
                hist_messages, (reference_messages, current_message) = await asyncio.gather(
                    self.render_history(hist_messages=raw_history, text_only=False),
                    asyncio.shield(parts_task),
                )
            # Covers the history fetch/render plus waiting on the shared attachment upload, so
            # the log separates pre-answer attachment cost from the route-call cost.
            logfire.info(
                "gen_reply context build done",
                elapsed_seconds=time.monotonic() - build_started,
                message_id=self.message.id,
            )

            if selection_task is not None:
                # Memory selection is an optional preflight; a provider/proxy hiccup here must
                # never turn an answerable message into the generic error path. Resolved under the
                # route_done gate: it usually already finished during the upload wait above, so
                # this returns immediately; a slow one gets only the post-route grace.
                selection_result = await self._await_optional_selection(
                    task=selection_task, route_done=route_done
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
                            message_id=self.message.id,
                        )
                    if selected_memories:
                        memories.extend(selected_memories)
                        memory_block = render_memory_context_block(memories=memories)
                        memory_credits = memory_lookup_credits(memories=memories)
                    logfire.info(
                        "gen_reply memory selection done",
                        elapsed_seconds=selection_elapsed,
                        model=self.toolkit.runtime_models.triage_model.name,
                        selected=len(selected_memories),
                        selected_ids=[memory.user_id for memory in selected_memories],
                        # The model-facing label, not the footer credit: this is an operator
                        # record, so the community nickname the selector matched on is exactly
                        # what makes the row readable, and it is never None.
                        labels=[memory.prompt_label for memory in selected_memories],
                        candidate_count=len(optional_allowed),
                        deterministic_count=deterministic_memory_count,
                        message_id=self.message.id,
                    )
        finally:
            # If this prep is cancelled during the upload wait (a non-QA route discarding it)
            # before the gate resolves it, cancel the in-flight selection so it never orphans.
            if selection_task is not None and not selection_task.done():
                await discard_task(
                    task=selection_task, label="memory selection", message_id=self.message.id
                )

        return ReplyContext(
            hist_messages=hist_messages,
            reference_messages=reference_messages,
            current_message=current_message,
            server_memory_block=server_memory_block,
            memory_block=memory_block,
            tone_block=tone_block,
            memory_credits=memory_credits,
            selection_input_tokens=selection_input_tokens,
            selection_output_tokens=selection_output_tokens,
        )
