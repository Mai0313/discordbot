"""The reply context: what one turn hands the answer model, and how it is built.

`ReplyContext` is the value; `ReplyContextBuilder` is the single speculative build that produces
it while the route call is still in flight. Everything the builder does only READS — channel
history and memory files — so a non-QA route can discard it safely, and the IMAGE / VIDEO routes
consume it after their media is on screen instead. Whose memory the turn may carry is settled
before the route call (`plan_recall`), since that call is what picks the optional members.
"""

import time
from typing import TYPE_CHECKING
import asyncio

import logfire
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.memory import MemoryCredits
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.llm_transcript import sanitize_identity
from discordbot.cogs.gen_reply.recall import (
    NO_STORED_MEMORY,
    UserMemory,
    RecallContext,
    RecallCandidate,
    render_tone_block,
    build_recall_context,
    recall_user_memories,
    memory_lookup_credits,
    build_recall_allowlist,
    render_server_memory_block,
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
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.typings.context_budgets import (
    HISTORY_CHAR_BUDGET,
    MAX_HISTORY_MEDIA_PARTS,
    MEMORY_CONTEXT_TARGET_USERS,
    HISTORY_PER_MESSAGE_OVERHEAD,
)
from discordbot.cogs.gen_reply.references import replied_to_message, source_channel_is_public
from discordbot.cogs.gen_reply.link_sources import system_block

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

    @property
    def message_list(self) -> list[EasyInputMessageParam]:
        """History, reference, and current blocks in transcript order."""
        return [*self.hist_messages, *self.reference_messages, *self.current_message]


class RecallPlan(BaseModel):
    """Whose memory one turn may carry, settled before the route call so the route can pick.

    Built by `ReplyContextBuilder.plan_recall`; the route reads `optional_candidates` and the
    server memory block, and `ReplyContextBuilder.build` joins the route's picks to `memories`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    server_memory_block: SkipValidation[EasyInputMessageParam | None] = Field(
        ..., description="Rendered server-memory block, or None outside a guild or without one."
    )
    recall_context: RecallContext = Field(
        ..., description="Where the turn happens, which decides the compartments every read opens."
    )
    memories: list[UserMemory] = Field(
        ..., description="Stored memory of the deterministic participants, in resolution order."
    )
    optional_candidates: dict[int, RecallCandidate] = Field(
        ...,
        description=(
            "Absent nickname-table members the route may pick from; empty in a private channel, "
            "without a table, or when the deterministic participants fill the budget."
        ),
    )
    remaining_slots: int = Field(
        ..., description="How many optional memories still fit the per-reply budget."
    )


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

    Counting is off `count_supported_sources`, so a source the modality gate is going to drop
    spends nothing. Counting the raw list instead took that budget from an older message whose
    images WOULD have been sent, and since the newest message is exempt, ten office documents
    on it could record the budget full while the turn carried no media at all (#660).

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
            count = builder.count_supported_sources(message=candidate)
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
    return system_block(
        text=(
            f"==== Reference Message from {sanitize_identity(value=ref.author.display_name)} "
            f"({sanitize_identity(value=ref.author.name)}) [id: {ref.author.id}]. "
            "The user is directly replying to this message; it is the primary context for "
            "the Current Message below. When the Current Message points at something "
            "without naming it, that something is here, this message's attachments "
            "included. ===="
        )
    )


def current_header(*, message: Message, has_reference: bool) -> EasyInputMessageParam:
    """Builds the system separator that precedes the current message.

    When the message is a reply, the header points back to the Reference Message block
    (rendered just above) so the model reads the reply pair as one unit.
    """
    reply_note = " It is the user's reply to the Reference Message above." if has_reference else ""
    return system_block(
        text=f"==== Current Message that needs to be answered from {sanitize_identity(value=message.author.display_name)} ({sanitize_identity(value=message.author.name)}) [id: {message.author.id}].{reply_note} ===="
    )


class ReplyContextBuilder(BaseModel):
    """Builds one turn's `ReplyContext` from Discord history plus stored memory.

    Everything here reads and nothing writes, which is what lets the pipeline start the build
    speculatively alongside the route call and throw it away when the route turns out not to
    need it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    toolkit: ReplyToolkit = Field(
        ...,
        description=(
            "The reply toolkit: the bot, whose user id is excluded from every memory allowlist, "
            "and the input builder."
        ),
    )
    surface: TurnSurface = Field(
        ...,
        description=(
            "Where this turn is happening: its history source, and the guild its memory "
            "compartments are scoped to."
        ),
    )

    @property
    def message(self) -> Message:
        """The message being answered, read off the surface that carries it."""
        return self.surface.message

    async def fetch_history(self, *, limit: int) -> list[Message]:
        """Fetches up to `limit` history messages once, trimmed to the char budget.

        Returned raw for the answer's uploaded render. Where they come from is the surface's
        question: a channel walk on the gateway path, and the
        conversation store on the `/ask` one, which is the only history a user-installed app
        has (it is not a member of the channel and holds no `READ_MESSAGE_HISTORY`).
        """
        return trim_history_to_budget(messages=await self.surface.fetch_history(limit=limit))

    async def render_history(self, *, hist_messages: list[Message]) -> list[EasyInputMessageParam]:
        """Renders fetched history with its uploaded attachment parts, for the answer.

        History is the only render that opts into the dead-source skip:
        an expired CDN attachment here re-fails every turn (current / reference do not; see
        GeminiFileUploader._resolve_file_upload).

        The full render is additionally capped at `MAX_HISTORY_MEDIA_PARTS` uploaded files: a
        message past the cap takes the text-only render, which is exactly the marker form the
        route already reads, so the degradation needs no second render path of its own.
        """
        if not hist_messages:
            return []
        input_builder = self.toolkit.input_builder
        over_budget = history_media_over_budget(builder=input_builder, hist_messages=hist_messages)
        tasks: list[Awaitable[EasyInputMessageParam]] = [
            input_builder.process_single_message_text_only(message=m)
            if m.id in over_budget
            else input_builder.process_single_message(message=m, allow_dead_cache=True)
            for m in hist_messages
        ]
        started = time.monotonic()
        processed = await asyncio.gather(*tasks)
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
        # out of here also keeps it out of the two other calls this render feeds, neither of which
        # is answering a question: the media persona reply, and the memory review transcript,
        # whose first message is this header verbatim.
        header = system_block(text="==== Chat History: earlier messages in this channel. ====")
        return [header, *processed]

    async def render_reference_message(
        self, *, text_only: bool = False
    ) -> list[EasyInputMessageParam]:
        """Renders the message being replied to, or nothing when this is not a reply.

        `text_only` emits attachment markers instead of uploaded file parts, for the
        route call that must not wait on the Files API.
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

        With `text_only` they render as attachment markers (no upload) for the route call;
        otherwise this is the answer-path render (uploads + activation poll to ACTIVE)
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
        selection or allowlist: it is read directly with zero extra LLM latency. Returns "" for
        a DM (no guild) or an empty memory. Read once per reply and shared by the route call,
        whose recall picks its nickname table maps, and the answer.

        A `/ask` turn always takes the "" branch, since its synthesized message carries no
        guild — deliberately, because the write side is gated on a public channel this route can
        never satisfy and a memory nothing writes back to is one the bot slowly goes stale on.
        What goes with it is the `## 成員稱呼` table, so the alias widening and the optional
        third-party candidates below have nothing to work from either, exactly as in a DM today.
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
        bot_user = self.toolkit.bot.user
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
        # nickname-table members to the route call.
        if server_memory and self.message.guild is not None:
            widen_allowlist_with_aliases(allowed=deterministic_allowed, memory=server_memory)
            if source_channel_is_public(message=self.message):
                # No credit label, because nothing in this channel names these members;
                # `RecallCandidate` owns what the footer does about that and why no name is
                # pulled from anywhere else.
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

    def plan_recall(self) -> RecallPlan:
        """Settles whose memory this turn may carry, before the route call that picks from it.

        File reads only (the server memory and the deterministic participants' memory), so the
        route never waits on the history fetch. Optional candidates are offered only when the
        deterministic participants leave a slot, so a pick can never displace one of them.
        """
        server_memory = self.read_server_memory()
        # Where this reply is happening, for compartment scoping of every user-memory read.
        recall_context = build_recall_context(
            author_id=self.message.author.id,
            guild_id=self.surface.guild_id,
            is_direct_message=self.surface.is_direct_message,
        )
        memories, optional_allowed, deterministic_candidate_count = (
            self._resolve_recall_candidates(
                server_memory=server_memory, recall_context=recall_context
            )
        )
        # The optional offer does not depend on the deterministic lookup having found anything:
        # a conversation where nobody present has a stored fact is exactly the one the code path
        # has nothing of its own to contribute to.
        remaining_slots = max(0, MEMORY_CONTEXT_TARGET_USERS - len(memories))
        logfire.debug(
            "gen_reply memory candidates built",
            deterministic_candidates=deterministic_candidate_count,
            deterministic_memories=len(memories),
            optional_candidates=len(optional_allowed),
            optional_slots=remaining_slots,
            message_id=self.message.id,
        )
        return RecallPlan(
            server_memory_block=(
                render_server_memory_block(memory=server_memory) if server_memory else None
            ),
            recall_context=recall_context,
            memories=memories,
            optional_candidates=optional_allowed if remaining_slots else {},
            remaining_slots=remaining_slots,
        )

    def _resolve_picks(self, *, recall: RecallPlan, picked_ids: list[str]) -> list[UserMemory]:
        """Reads the members the route picked, held to the offered candidates and the budget."""
        picked = recall_user_memories(
            user_id_list=picked_ids,
            allowed=recall.optional_candidates,
            context=recall.recall_context,
        )
        kept = picked[: recall.remaining_slots]
        if len(picked) > len(kept):
            logfire.warn(
                "Capping optional memories to the remaining per-reply budget",
                requested=len(picked),
                kept=len(kept),
                message_id=self.message.id,
            )
        logfire.info(
            "gen_reply optional recall resolved",
            selected=len(kept),
            selected_ids=[memory.user_id for memory in kept],
            # The model-facing label, not the footer credit: this is an operator record, so the
            # community nickname the route matched on is exactly what makes the row readable, and
            # it is never None.
            labels=[memory.prompt_label for memory in kept],
            candidate_count=len(recall.optional_candidates),
            deterministic_count=len(recall.memories),
            message_id=self.message.id,
        )
        return kept

    async def build(
        self,
        *,
        history_limit: int,
        parts_task: asyncio.Task[MessageParts],
        recall: RecallPlan,
        recall_picks: asyncio.Future[list[str]],
    ) -> ReplyContext:
        """Builds history, shared parts, server memory, and the memory the turn carries.

        Runs speculatively as its own task concurrent with routing: everything here only reads
        (channel history, memory files), so a non-QA route can discard it safely. `parts_task`
        carries the answer-path reference/current renders (uploaded files). `recall_picks` is
        the route's optional recall picks, which the pipeline resolves as soon as the route
        returns; it is awaited only after the history and uploads, which almost always outlast
        the route.
        """
        build_started = time.monotonic()

        raw_history = await self.fetch_history(limit=history_limit)

        # The message author's tone-preference note is read directly for that one author
        # (their own preference for how the bot should sound, cross-server safe by
        # construction) and injected on every reply with no selection phase, including one
        # that runs with user memory off. One file read, no extra LLM call.
        author_tone = read_tone(scope=user_scope(user_id=self.message.author.id))
        tone_block = render_tone_block(tone=author_tone) if author_tone else None

        # The answer needs the uploaded renders; await the full history render and the shared
        # reference/current uploads here. `parts_task` is shielded so cancelling this speculative
        # prep (IMAGE / VIDEO) never cancels the shared upload task those routes still reuse; the
        # full history render rides as an ordinary gather child, so it is cancelled together with
        # prep.
        with logfire.span("gen_reply context build", message_id=self.message.id):
            hist_messages, (reference_messages, current_message) = await asyncio.gather(
                self.render_history(hist_messages=raw_history), asyncio.shield(parts_task)
            )
        # Covers the history fetch/render plus waiting on the shared attachment upload, so
        # the log separates pre-answer attachment cost from the route-call cost.
        logfire.info(
            "gen_reply context build done",
            elapsed_seconds=time.monotonic() - build_started,
            message_id=self.message.id,
        )

        picked_ids = await recall_picks
        memories = list(recall.memories)
        if recall.optional_candidates:
            memories.extend(self._resolve_picks(recall=recall, picked_ids=picked_ids))
        return ReplyContext(
            hist_messages=hist_messages,
            reference_messages=reference_messages,
            current_message=current_message,
            server_memory_block=recall.server_memory_block,
            memory_block=render_memory_context_block(memories=memories) if memories else None,
            tone_block=tone_block,
            memory_credits=memory_lookup_credits(memories=memories)
            if memories
            else MemoryCredits(),
        )
