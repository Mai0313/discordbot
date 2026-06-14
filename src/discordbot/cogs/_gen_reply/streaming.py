"""Streams a Responses API reply onto a Discord message."""

import re
import time
from typing import cast
import asyncio
import contextlib

from openai import AsyncStream
import logfire
import nextcord
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.utils import escape_mentions
from openai.types.responses import (
    ResponseStreamEvent,
    ResponseCreatedEvent,
    ResponseCompletedEvent,
)

from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.economy import CHAT_REWARD_MAX_PER_REPLY, CHAT_REWARD_TOKEN_DIVISOR
from discordbot.utils.model_pricing import get_token_rates
from discordbot.cogs._economy.database import credit_with_repayment
from discordbot.cogs._economy.presentation import currency_text

# Gemini occasionally wraps Discord mention syntax in backticks (inline code),
# which stops Discord from rendering the actual mention. Strip those wrappers
# before sending; matches user (<@id>, <@!id>), role (<@&id>) and channel (<#id>) mentions.
CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")
DISCORD_MESSAGE_LIMIT = 2000


class ResponseStreamer(BaseModel):
    """Renders one streaming Responses API reply onto a Discord message.

    The cog calls `stream` once with the answer-turn stream; reasoning summaries are
    previewed as `-#` subtext while the model thinks, the real text replaces them as it
    arrives, then a usage footer (and an optional memory-credit line) is written. Memory
    lookups are decided in a separate request before streaming, so the labels are passed
    in via `memory_lookups` rather than discovered here. Discord edits run on a
    time-based snapshot editor task so consuming the stream never waits on Discord.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message] = Field(
        ..., description="The Discord message being answered and replied to."
    )
    stored_content: str = Field(default="", description="The accumulated reply text.")
    reasoning_content: str = Field(
        default="", description="The accumulated reasoning-summary text shown before content."
    )
    reply: SkipValidation[Message | None] = Field(
        default=None, description="The Discord reply message, created lazily on the first delta."
    )
    displayed_content: str = Field(
        default="", description="The text last written to the Discord reply."
    )
    content_started: bool = Field(
        default=False, description="Whether the first non-newline text delta has been seen."
    )
    preview_interval_seconds: float = Field(
        default=1.0, description="Cadence of the snapshot editor's Discord edits while streaming."
    )
    model_name: str = Field(
        default="", description="The model name reported by the stream, for the usage footer."
    )
    model_effort: str = Field(
        default="",
        description="Route-decided reasoning effort shown next to the model in the footer.",
    )
    input_tokens: int = Field(default=0, description="Input tokens reported by the stream.")
    output_tokens: int = Field(default=0, description="Output tokens reported by the stream.")
    memory_lookups: list[str] = Field(
        default_factory=list,
        description="Labels of users whose stored memory was injected, for the footer.",
    )
    created_at: float = Field(
        default_factory=time.monotonic,
        description=(
            "Monotonic creation time; the streamer is constructed right before the answer "
            "request, so the first-content-delta log measures answer-call-to-first-token."
        ),
    )
    _editor_task: asyncio.Task[None] | None = PrivateAttr(default=None)
    _editor_stop: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

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

    def _render_preview(self) -> str:
        """Builds the current streaming preview: real content once started, else reasoning.

        The reasoning preview shows the tail of the model's thought summary as `-#`
        subtext lines under a 💭 header, so the user watches the thinking process until
        the first real content delta replaces it. The window keeps the newest lines that
        fit one Discord message, so a long think never hits the 2000-char limit.
        """
        if self.content_started:
            return self.stored_content[:DISCORD_MESSAGE_LIMIT]
        if not self.reasoning_content:
            return ""
        # Mentions are escaped because this transient text is never meant to ping;
        # the real reply may mention people, the thought process must not.
        tail = escape_mentions(self.reasoning_content[-1500:])
        lines = [f"-# {line}" for line in tail.splitlines() if line.strip()]
        header = "-# 💭 思考中..."
        budget = DISCORD_MESSAGE_LIMIT - len(header)
        kept: list[str] = []
        for line in reversed(lines):
            if budget - (len(line) + 1) < 0:
                break
            kept.append(line)
            budget -= len(line) + 1
        kept.reverse()
        return "\n".join([header, *kept])

    async def _reply_or_send(self, content: str) -> Message:
        """Replies to the source message, sending unparented if it was deleted.

        Deleting the source before the reply lands makes Discord 400 with code 50035
        (unknown message_reference); we log it and send into the same channel instead of
        wasting the whole pipeline. Other HTTP errors still propagate to the caller.
        """
        try:
            return await self.message.reply(content=content)
        except nextcord.HTTPException as exc:
            if exc.code != 50035 and not isinstance(exc, nextcord.NotFound):
                raise
            logfire.info(
                "Source message deleted before reply; sending unparented",
                message_id=self.message.id,
            )
            return await self.message.channel.send(content=content)

    async def _write_preview_snapshot(self) -> None:
        """Writes the latest preview snapshot to the Discord reply, skipping no-ops."""
        preview = self._render_preview()
        if not preview or preview == self.displayed_content:
            return
        if self.reply is None:
            self.reply = await self._reply_or_send(content=preview)
        else:
            await self.reply.edit(content=preview)
        self.displayed_content = preview

    async def _preview_editor(self) -> None:
        """Edits the reply with the latest snapshot on a fixed cadence until stopped.

        Stopping uses the event rather than task cancellation so an in-flight Discord
        write always completes before `_finalize_reply` runs; a cancel landing inside
        the first `message.reply` could otherwise orphan the created message and let
        the finalizer create a duplicate.
        """
        while True:
            try:
                await asyncio.wait_for(
                    self._editor_stop.wait(), timeout=self.preview_interval_seconds
                )
            except TimeoutError:
                with contextlib.suppress(Exception):
                    await self._write_preview_snapshot()
            else:
                return

    def _ensure_editor_started(self) -> None:
        """Starts the snapshot editor task on the first delta that gives it work."""
        if self._editor_task is None:
            self._editor_task = asyncio.create_task(coro=self._preview_editor())

    async def _stop_editor(self) -> None:
        """Signals the editor to stop and waits out any in-flight Discord write."""
        if self._editor_task is None:
            return
        self._editor_stop.set()
        with contextlib.suppress(Exception):
            await self._editor_task
        self._editor_task = None

    async def _write_final_message(self, reply: Message | None, content: str, footer: str) -> None:
        """Writes the final reply, continuing overflow as follow-up replies in the same channel."""
        parent_content, follow_up_chunks = self._split_reply_for_discord(
            content=content, footer=footer
        )
        if reply is None:
            reply = await self._reply_or_send(content=parent_content)
        else:
            await reply.edit(content=parent_content)
        previous = reply
        for chunk in follow_up_chunks:
            previous = await previous.reply(content=chunk)

    def _on_reasoning_delta(self, delta: str) -> None:
        """Accumulates one reasoning-summary delta, logging the first one's latency."""
        if not self.reasoning_content:
            # Gemini may prepend newlines to the first reasoning delta too.
            delta = delta.lstrip("\n")
            if not delta:
                return
            logfire.info(
                "gen_reply first reasoning delta",
                elapsed_seconds=time.monotonic() - self.created_at,
                model=self.model_name,
            )
        self.reasoning_content += delta
        self._ensure_editor_started()

    def _on_content_delta(self, delta: str) -> None:
        """Accumulates one content delta, logging the first one's latency."""
        if not self.content_started:
            delta = delta.lstrip("\n")
            if not delta:
                return
            self.content_started = True
            logfire.info(
                "gen_reply first content delta",
                elapsed_seconds=time.monotonic() - self.created_at,
                model=self.model_name,
            )
        self.stored_content += delta
        self._ensure_editor_started()

    async def _consume(self, *, responses: AsyncStream[ResponseStreamEvent]) -> None:
        """Streams the reply, accumulating text and usage onto the instance.

        Only accumulates state; the snapshot editor task renders it to Discord, so this
        loop never blocks on a Discord edit between deltas.
        """
        async for response in responses:
            if response.type in {"response.created", "response.completed"}:
                # Capture the model on `created` too so the usage footer never falls back
                # to an empty model name (and $0.00000000) when a stream ends without a
                # clean `completed` event. Usage only arrives on `completed`. Both events
                # carry `.response`, but mypy cannot narrow the ~50-member event union on a
                # set-membership test, so cast to the two that do before reading it.
                event = cast("ResponseCreatedEvent | ResponseCompletedEvent", response)
                self.model_name = event.response.model
                if event.response.usage:
                    self.input_tokens += event.response.usage.input_tokens
                    self.output_tokens += event.response.usage.output_tokens
            elif response.type == "response.reasoning_summary_text.delta":
                self._on_reasoning_delta(delta=response.delta)
            elif response.type == "response.output_text.delta":
                self._on_content_delta(delta=response.delta)

    async def _finalize_reply(self) -> str:
        """Writes the reward footer and final reply once the stream is consumed."""
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
        model_label = (
            f"{self.model_name} ({self.model_effort})" if self.model_effort else self.model_name
        )
        usage_footer = f"\n\n-# {model_label} · ⬆ {self.input_tokens:,} ⬇ {self.output_tokens:,} · ${cost:.8f} · {balance_text}{memory_line}"

        # Final update to ensure complete message is displayed.
        await self._write_final_message(
            reply=self.reply, content=self.stored_content, footer=usage_footer
        )
        self.stored_content += usage_footer

        return self.stored_content

    async def stream(self, *, responses: AsyncStream[ResponseStreamEvent]) -> str:
        """Streams the reply onto the message and writes the usage footer; returns the full text."""
        try:
            await self._consume(responses=responses)
        finally:
            await self._stop_editor()
        return await self._finalize_reply()
