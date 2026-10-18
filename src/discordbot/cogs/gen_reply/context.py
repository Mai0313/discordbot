"""The reply context: what one turn hands the answer model, and how it is built.

`ReplyContext` is the value; `ReplyContextBuilder` is the single speculative build that produces
it while the route call is still in flight. Every route consumes it, the `IMAGE` / `VIDEO` routes
once their media is on screen. Everything the builder does only reads channel history and memory
files, so a turn that fails before consuming it can discard it safely. Whose memory the turn may
carry is settled before the route call (`plan_recall`), since that call picks the optional
members.
"""

import time
from typing import TYPE_CHECKING
import asyncio

import logfire
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
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
    MAX_HISTORY_INLINE_BYTES,
    MEMORY_CONTEXT_TARGET_USERS,
    HISTORY_PER_MESSAGE_OVERHEAD,
)
from discordbot.utils.channel_visibility import channel_is_public
from discordbot.cogs.gen_reply.references import replied_to_message
from discordbot.cogs.gen_reply.capabilities import render_capabilities_block
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
        """Returns history, reference, and current blocks in transcript order."""
        return [*self.hist_messages, *self.reference_messages, *self.current_message]

    def answer_input(self, config: LLMConfig) -> ResponseInputParam:
        """Returns the QA answer's input in the order the answer model reads it.

        Places the byte-static capability reference first for prefix caching, followed by
        history, memory blocks, the reference message, linked posts, and the current message last.

        Args:
            config (LLMConfig): Deployment switches passed to `render_capabilities_block`.
        """
        memory = (self.server_memory_block, self.memory_block, self.tone_block)
        return [
            render_capabilities_block(config=config),
            *self.hist_messages,
            *(block for block in memory if block is not None),
            *self.reference_messages,
            *self.link_blocks,
            *self.current_message,
        ]

    def persona_input(self) -> ResponseInputParam:
        """Returns the media persona reply's input, omitting capabilities, server memory, and link blocks."""
        memory = (self.memory_block, self.tone_block)
        return [
            *self.hist_messages,
            *(block for block in memory if block is not None),
            *self.reference_messages,
            *self.current_message,
        ]


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


def trim_history_to_budget(messages: list[Message]) -> list[Message]:
    """Returns the newest history messages that fit `HISTORY_CHAR_BUDGET`, cut on a message boundary.

    Always keeps the newest message even when it alone exceeds `HISTORY_CHAR_BUDGET`.

    Args:
        messages (list[Message]): Candidate history messages ordered oldest-first.
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
    builder: MessageInputBuilder, hist_messages: list[Message]
) -> dict[int, int]:
    """Returns a mapping of over-budget history message ids to their supported attachment counts.

    Walks newest-first against `MAX_HISTORY_MEDIA_PARTS` using `count_supported_sources`. The
    newest attachment-bearing message is exempt, and once one older message overflows, it and
    every earlier message are marked to render as text-only markers so kept media forms an
    unbroken suffix.

    Args:
        builder (MessageInputBuilder): Input builder used to count supported attachment sources.
        hist_messages (list[Message]): History messages ordered oldest-first.
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


def history_payload_overflow(rendered: list[EasyInputMessageParam]) -> int:
    """Returns the count of oldest rendered history messages that overflow `MAX_HISTORY_INLINE_BYTES`.

    Walks newest-first without exempting the newest message, returning `index + 1` at the first
    message that pushes cumulative inline bytes past `MAX_HISTORY_INLINE_BYTES`.

    Args:
        rendered (list[EasyInputMessageParam]): Rendered history messages ordered oldest-first.
    """
    spent = 0
    for index in range(len(rendered) - 1, -1, -1):
        content = rendered[index]["content"]
        if isinstance(content, str):
            continue
        spent += sum(
            len(value.encode())
            for part in content
            for value in part.values()
            if isinstance(value, str)
        )
        if spent > MAX_HISTORY_INLINE_BYTES:
            return index + 1
    return 0


def reference_header(ref: Message) -> EasyInputMessageParam:
    """Returns the system separator block that precedes the message being replied to.

    Args:
        ref (Message): The referenced message being replied to.
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


def current_header(message: Message, has_reference: bool) -> EasyInputMessageParam:
    """Returns the system separator block that precedes the current message.

    Args:
        message (Message): The current message being answered.
        has_reference (bool): Whether a Reference Message block is rendered above it.
    """
    reply_note = " It is the user's reply to the Reference Message above." if has_reference else ""
    return system_block(
        text=f"==== Current Message that needs to be answered from {sanitize_identity(value=message.author.display_name)} ({sanitize_identity(value=message.author.name)}) [id: {message.author.id}].{reply_note} ===="
    )


class ReplyContextBuilder(BaseModel):
    """Builds one turn's `ReplyContext` from Discord history plus stored memory.

    Everything here reads and nothing writes, which is what lets the pipeline start the build
    speculatively alongside the route call and throw it away when the turn fails before
    consuming it.
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
        """Returns the message being answered, read off `self.surface`."""
        return self.surface.message

    async def fetch_history(self, limit: int) -> list[Message]:
        """Returns up to `limit` history messages from `self.surface`, trimmed to `HISTORY_CHAR_BUDGET`.

        Args:
            limit (int): Maximum number of history messages to request from the surface.
        """
        return trim_history_to_budget(messages=await self.surface.fetch_history(limit=limit))

    async def render_history(self, hist_messages: list[Message]) -> list[EasyInputMessageParam]:
        """Returns rendered history blocks with uploaded attachment parts, capped by the media budgets.

        Opts into `allow_dead_cache=True` and degrades messages past `MAX_HISTORY_MEDIA_PARTS` or
        `MAX_HISTORY_INLINE_BYTES` to `text_only=True` markers.

        Args:
            hist_messages (list[Message]): Trimmed history messages ordered oldest-first.
        """
        if not hist_messages:
            return []
        input_builder = self.toolkit.input_builder
        over_budget = history_media_over_budget(builder=input_builder, hist_messages=hist_messages)
        tasks: list[Awaitable[EasyInputMessageParam]] = [
            input_builder.process_single_message(
                message=m, text_only=m.id in over_budget, allow_dead_cache=True
            )
            for m in hist_messages
        ]
        started = time.monotonic()
        processed = await asyncio.gather(*tasks)
        payload_capped = [
            index
            for index in range(history_payload_overflow(rendered=processed))
            if hist_messages[index].id not in over_budget
            and isinstance(processed[index]["content"], list)
        ]
        for index in payload_capped:
            processed[index] = await input_builder.process_single_message(
                message=hist_messages[index], text_only=True
            )
        logfire.info(
            "gen_reply history render done",
            elapsed_seconds=time.monotonic() - started,
            message_count=len(hist_messages),
            media_capped=sum(over_budget.values()),
            payload_capped_messages=len(payload_capped),
            message_id=self.message.id,
        )
        # Names the block and stops there. Wording that invites the model to answer from the
        # history competes with the Reference Message's own claim to be the primary context and
        # loses the reply's subject to whatever in the window reads as the most answerable thing.
        # Where the subject may come from is a behaviour rule, so it lives in `REPLY_PROMPT` at
        # developer authority instead. Keeping it out of here also keeps it out of every other
        # call this render feeds, none of which is answering a question; the memory review
        # transcript opens with this header verbatim.
        header = system_block(text="==== Chat History: earlier messages in this channel. ====")
        return [header, *processed]

    async def render_reference_message(
        self, text_only: bool = False
    ) -> list[EasyInputMessageParam]:
        """Returns the rendered reference-message header and body, or an empty list when not a reply.

        Args:
            text_only (bool): Whether to emit attachment markers instead of uploading files.
        """
        replied_to = replied_to_message(message=self.message)
        if replied_to is None:
            return []
        input_builder = self.toolkit.input_builder
        processed = await input_builder.process_single_message(
            message=replied_to, text_only=text_only
        )
        return [reference_header(ref=replied_to), processed]

    async def render_current_message(self, text_only: bool = False) -> list[EasyInputMessageParam]:
        """Returns the rendered current-message header and body.

        Args:
            text_only (bool): Whether to emit attachment markers instead of uploading files.
        """
        has_reference = replied_to_message(message=self.message) is not None
        messages: list[EasyInputMessageParam] = [
            current_header(message=self.message, has_reference=has_reference)
        ]
        input_builder = self.toolkit.input_builder
        current_msg = await input_builder.process_single_message(
            message=self.message, text_only=text_only
        )
        messages.append(current_msg)
        return messages

    async def render_parts(self, text_only: bool = False) -> MessageParts:
        """Returns the rendered `(reference_messages, current_message)` pair.

        Args:
            text_only (bool): Whether to render attachment markers (`True`) or upload files (`False`).
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
        """Returns the current guild's raw server memory, or `""` when there is none.

        Reads `self.message.guild` directly, so a `/ask` turn (whose synthesized message carries
        no `guild`) always takes the `""` branch: the write side is gated on a public channel
        that `/ask` can never verify, and a server memory that is never written back to would go
        stale. `## 成員稱呼` alias widening and optional third-party candidates stay off with it,
        matching DM behavior.
        """
        if self.message.guild is None:
            return ""
        return read_memory_document(
            scope=server_scope(server_id=self.message.guild.id),
            compartments=[GLOBAL_COMPARTMENT],
            flavor="server",
        )

    def _resolve_recall_candidates(
        self, server_memory: str, recall_context: RecallContext
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
            if channel_is_public(guild=self.message.guild, channel=self.message.channel):
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
        """Returns the `RecallPlan` for this turn before the route call selects optional members.

        Performs local file reads only so routing never waits on the history fetch, and offers
        optional candidates only when deterministic participants leave remaining slots.
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

    def _resolve_picks(self, recall: RecallPlan, picked_ids: list[str]) -> list[UserMemory]:
        """Reads the members the route picked, held to the offered candidates and the budget.

        A pick with no stored memory ranks after every pick that has some, so it only takes a
        slot no stored pick wanted.
        """
        picked = sorted(
            recall_user_memories(
                user_id_list=picked_ids,
                allowed=recall.optional_candidates,
                context=recall.recall_context,
            ),
            key=lambda memory: memory.memory == NO_STORED_MEMORY,
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
        history_limit: int,
        parts_task: asyncio.Task[MessageParts],
        recall: RecallPlan,
        recall_picks: asyncio.Future[list[str]],
    ) -> ReplyContext:
        """Returns the assembled `ReplyContext` from history, rendered parts, and resolved memory.

        Runs speculatively alongside routing using read-only history and memory lookups, awaiting
        `recall_picks` after history rendering and `parts_task` complete.

        Args:
            history_limit (int): Maximum number of channel history messages to fetch.
            parts_task (asyncio.Task[MessageParts]): Task rendering the reference and current
                messages with uploaded attachment parts.
            recall (RecallPlan): Pre-route deterministic memories and optional candidate pool.
            recall_picks (asyncio.Future[list[str]]): Future resolved with the router's optional
                user-id picks.
        """
        build_started = time.monotonic()

        raw_history = await self.fetch_history(limit=history_limit)

        # The message author's tone-preference note is read directly for that one author
        # (their own preference for how the bot should sound, cross-server safe by
        # construction) and injected on every reply with no selection phase. One file read, no
        # extra LLM call.
        author_tone = read_tone(scope=user_scope(user_id=self.message.author.id))
        tone_block = render_tone_block(tone=author_tone) if author_tone else None

        # The answer needs the uploaded renders; await the full history render and the shared
        # reference/current uploads here. Cancelling this build cancels the upload too, which is
        # safe only while nothing else reads `parts_task`.
        with logfire.span("gen_reply context build", message_id=self.message.id):
            hist_messages, (reference_messages, current_message) = await asyncio.gather(
                self.render_history(hist_messages=raw_history), parts_task
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
