"""Streams a Responses API reply onto a Discord message."""

import re

from openai import AsyncStream
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses import ResponseStreamEvent, ResponseFunctionToolCall

from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.economy import CHAT_REWARD_MAX_PER_REPLY, CHAT_REWARD_TOKEN_DIVISOR
from discordbot.utils.reactions import update_reaction
from discordbot.utils.model_pricing import get_token_rates
from discordbot.cogs._economy.database import credit_with_repayment
from discordbot.cogs._economy.presentation import currency_text

# Gemini occasionally wraps Discord mention syntax in backticks (inline code),
# which stops Discord from rendering the actual mention. Strip those wrappers
# before sending; matches user (<@id>, <@!id>), role (<@&id>) and channel (<#id>) mentions.
CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")
DISCORD_MESSAGE_LIMIT = 2000


class ResponseStreamer(BaseModel):
    """Renders a (possibly multi-turn) streaming Responses API reply onto a message.

    One instance spans the whole agentic reply: the cog calls `consume_turn` for each
    Responses stream (re-calling the model after a `get_user_memory` tool turn) and
    `finalize` once at the end. Display, usage, and the lazily-created Discord reply
    accumulate across turns so intermediate tool-only turns add nothing visible.

    Attributes:
        message: The Discord message being answered and replied to.
        stored_content: The reply text accumulated across all turns.
        reply: The Discord reply message, created lazily on the first text delta.
        displayed_content: The text last written to the Discord reply.
        content_started: Whether the first non-newline text delta has been seen.
        model_name: The model name reported by the latest turn, for the usage footer.
        input_tokens: Input tokens summed across all turns.
        output_tokens: Output tokens summed across all turns.
        used_web_search: Whether any turn used a native web-search / grounding tool.
        memory_lookups: Labels of users whose stored memory was read this reply, for the footer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message]
    stored_content: str = ""
    reply: SkipValidation[Message | None] = None
    displayed_content: str = ""
    content_started: bool = False
    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    used_web_search: bool = False
    memory_lookups: list[str] = Field(default_factory=list)

    @staticmethod
    def _split_reply_for_discord(content: str, footer: str) -> tuple[str, list[str]]:
        """Splits a completed reply into one parent message plus follow-up chunks."""
        if len(f"{content}{footer}") <= DISCORD_MESSAGE_LIMIT:
            return f"{content}{footer}", []

        tail_capacity = DISCORD_MESSAGE_LIMIT - len(footer)
        if tail_capacity <= 0:
            raise ValueError("Usage footer is too long for Discord message content")

        parent_content = content[:DISCORD_MESSAGE_LIMIT]
        remaining = content[DISCORD_MESSAGE_LIMIT:]
        follow_up_chunks: list[str] = []

        while len(remaining) > DISCORD_MESSAGE_LIMIT:
            follow_up_chunks.append(remaining[:DISCORD_MESSAGE_LIMIT])
            remaining = remaining[DISCORD_MESSAGE_LIMIT:]

        if len(remaining) <= tail_capacity:
            follow_up_chunks.append(f"{remaining}{footer}")
        else:
            follow_up_chunks.append(remaining[:tail_capacity])
            follow_up_chunks.append(f"{remaining[tail_capacity:]}{footer}")
        return parent_content, follow_up_chunks

    async def _write_preview(
        self, reply: Message | None, content: str, displayed_content: str
    ) -> tuple[Message | None, str]:
        """Writes at most one Discord message worth of streaming preview text."""
        preview = content[:DISCORD_MESSAGE_LIMIT]
        if preview == displayed_content:
            return reply, displayed_content
        if reply is None:
            reply = await self.message.reply(content=preview)
        else:
            await reply.edit(content=preview)
        return reply, preview

    async def _finalize(self, reply: Message | None, content: str, footer: str) -> Message:
        """Writes the final reply, continuing overflow as follow-up replies in the same channel."""
        parent_content, follow_up_chunks = self._split_reply_for_discord(
            content=content, footer=footer
        )
        if reply is None:
            reply = await self.message.reply(content=parent_content)
        else:
            await reply.edit(content=parent_content)
        previous = reply
        for chunk in follow_up_chunks:
            previous = await previous.reply(content=chunk)
        return reply

    async def consume_turn(  # noqa: C901 -- dispatches on multiple Responses API stream event types
        self, *, responses: AsyncStream[ResponseStreamEvent]
    ) -> list[ResponseFunctionToolCall]:
        """Streams one turn into the accumulated state and returns its tool calls.

        Text deltas extend the persistent reply, usage is summed onto the instance,
        and any `get_user_memory` function calls are returned so the cog can resolve
        them and re-call the model. An empty list means this was the final turn.
        """
        function_calls: list[ResponseFunctionToolCall] = []
        counted_content = 0
        # Snapshot so a turn that resolves into a get_user_memory call can roll back any
        # preamble text it streamed: tool turns carry no user-facing answer, so committing
        # their text would prepend it to the real answer from the next turn. Gemini emits no
        # text on tool turns today; this keeps that an invariant rather than an assumption.
        text_before_turn = self.stored_content
        content_started_before_turn = self.content_started
        displayed_before_turn = self.displayed_content
        reply_before_turn = self.reply

        async for response in responses:
            if response.type in {"response.created", "response.completed"}:
                # Capture the model on `created` too so the usage footer never
                # falls back to an empty model name (and $0.00000000) when a
                # stream ends without a clean `completed` event. Usage only
                # arrives on `completed`, and is summed across turns.
                self.model_name = response.response.model
                if response.response.usage:
                    self.input_tokens += response.response.usage.input_tokens
                    self.output_tokens += response.response.usage.output_tokens
            elif response.type in {
                "response.web_search_call.in_progress",
                "response.web_search_call.searching",
                "response.web_search_call.completed",
                "response.output_text.annotation.added",
            }:
                self.used_web_search = True
            elif response.type == "response.output_text.delta":
                delta = response.delta
                if not self.content_started:
                    delta = delta.lstrip("\n")
                    if not delta:
                        continue
                    self.content_started = True
                self.stored_content += delta
                counted_content += len(delta)

                if counted_content >= 30:
                    self.reply, self.displayed_content = await self._write_preview(
                        reply=self.reply,
                        content=self.stored_content,
                        displayed_content=self.displayed_content,
                    )
                    counted_content = 0
            elif response.type == "response.output_item.done":
                item = response.item
                if item.type == "function_call" and item.name == "get_user_memory":
                    function_calls.append(item)

        if function_calls:
            self.stored_content = text_before_turn
            self.content_started = content_started_before_turn
            # If this tool turn streamed a visible preview before the call arrived, undo
            # it too, so the preamble never lingers in the channel (e.g. if the follow-up
            # turn errors out). Delete a reply created this turn; otherwise restore the
            # prior answer text rather than editing to empty (Discord rejects empty edits).
            if self.reply is not None and self.displayed_content != displayed_before_turn:
                if reply_before_turn is None:
                    await self.reply.delete()
                    self.reply = None
                else:
                    await self.reply.edit(content=displayed_before_turn)
                self.displayed_content = displayed_before_turn

        return function_calls

    async def finalize(self) -> str:
        """Writes the reward footer and final reply once all turns are consumed."""
        input_rate, output_rate = get_token_rates(model_name=self.model_name)
        cost = input_rate * self.input_tokens + output_rate * self.output_tokens

        # Award chat points from token usage, divided down and capped per reply so a
        # single long (e.g. web-search) reply cannot mint a huge balance. We await this
        # (rather than fire-and-forget) so the resulting balance can land in the footer.
        # On DB failure, it returns None and the footer falls back to the delta-only format.
        total_tokens = self.input_tokens + self.output_tokens
        reward = min(total_tokens // CHAT_REWARD_TOKEN_DIVISOR, CHAT_REWARD_MAX_PER_REPLY)
        avatar_url = await guild_avatar_url(
            user=self.message.author, guild=getattr(self.message, "guild", None)
        )
        result = await credit_with_repayment(
            user_id=self.message.author.id,
            name=self.message.author.name,
            avatar_url=avatar_url,
            amount=reward,
        )

        self.stored_content = CODED_MENTION_RE.sub(r"\1", self.stored_content)
        if result.new_balance is not None:
            balance_text = f"{currency_text(amount=result.new_balance, compact=True)} ({currency_text(amount=reward, signed=True, compact=True)})"
        else:
            balance_text = currency_text(amount=reward, signed=True, compact=True)
        # Credit looked-up memory owners on a second -# subtext line. Dedupe while
        # preserving lookup order; past two names collapse to "等 N 人" so a busy
        # lookup stays short. USAGE_FOOTER_RE matches this optional second line too.
        memory_line = ""
        if self.memory_lookups:
            names = list(dict.fromkeys(self.memory_lookups))
            if len(names) > 2:
                memory_line = f"\n-# 🧠 已讀取 {', '.join(names[:2])} 等 {len(names)} 人的記憶"
            else:
                memory_line = f"\n-# 🧠 已讀取 {', '.join(names)} 的記憶"
        # Footer format must stay matchable by `input.USAGE_FOOTER_RE`; the ⬆/⬇ icons are its anchor.
        usage_footer = f"\n\n-# {self.model_name} · ⬆ {self.input_tokens:,} ⬇ {self.output_tokens:,} · ${cost:.8f} · {balance_text}{memory_line}"

        # Final update to ensure complete message is displayed.
        await self._finalize(reply=self.reply, content=self.stored_content, footer=usage_footer)
        self.stored_content += usage_footer

        if self.used_web_search:
            await update_reaction(message=self.message, bot_user=None, emoji="🌐")

        return self.stored_content

    async def stream(self, *, responses: AsyncStream[ResponseStreamEvent]) -> str:
        """Consumes a single-turn stream and finalizes; convenience for tool-free paths."""
        await self.consume_turn(responses=responses)
        return await self.finalize()
