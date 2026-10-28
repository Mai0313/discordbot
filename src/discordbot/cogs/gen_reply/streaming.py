"""Streams a Responses API reply onto a Discord message."""

import re
import time
from typing import Literal
import asyncio
import contextlib
import unicodedata
from collections.abc import Callable, Awaitable, AsyncIterator

import logfire
from nextcord import File, Embed, Message, NotFound, Forbidden, HTTPException, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_fixed, wait_random
from nextcord.utils import escape_mentions
from openai.types.responses import ResponseOutputItem, ResponseStreamEvent

from discordbot.typings.media import LoadedMedia
from discordbot.typings.memory import MemoryCredits
from discordbot.typings.timeouts import ANSWER_STREAM_MAX_ATTEMPTS
from discordbot.utils.llm_errors import llm_status_code, is_retryable_llm_error
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT, embed_spacer_payload
from discordbot.utils.discord_errors import is_reply_target_gone
from discordbot.utils.llm_transcript import render_usage_footer
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaDeliveryPlanner,
    upload_limit_for,
)
from discordbot.cogs.gen_reply.markers import (
    MAX_INLINE_IMAGES,
    InlineMarkers,
    extract_inline_markers,
    scrub_markers_for_preview,
)
from discordbot.cogs.gen_reply.surface import WINDOW_EXPIRED_NOTICE, TurnSurface
from discordbot.cogs.gen_reply.generation import (
    VOICE_REPLY_FILENAME,
    INLINE_IMAGE_FILENAME,
    INLINE_VIDEO_FILENAME,
    ImageGenerator,
    MusicGenerator,
    VideoGenerator,
    VoiceGenerator,
    music_filename,
    speechify_discord_markup,
)
from discordbot.cogs.gen_reply.references import replied_to_message
from discordbot.cogs.gen_reply.turn_state import current_answer_streamer
from discordbot.cogs.gen_reply.status_marks import (
    IMAGE_EMOJI,
    MUSIC_EMOJI,
    VIDEO_EMOJI,
    VOICE_EMOJI,
    ANSWER_EMOJI,
    RETRY_HINT_EMOJI,
    MEMORY_READ_EMOJI,
    DROPPED_HINT_EMOJI,
    MEMORY_WRITE_EMOJI,
    TIMEOUT_HINT_EMOJI,
)

# Gemini occasionally wraps Discord mention syntax in backticks (inline code),
# which stops Discord from rendering the actual mention. Strip those wrappers
# before sending; matches user (<@id>, <@!id>), role (<@&id>) and channel (<#id>) mentions.
CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")

# Flat spacing between answer-stream attempts so a provider 5xx burst is outlasted rather than
# immediately hit again. `wait_random` is added beside `wait_fixed` to keep concurrent replies
# from retrying in lockstep without relying on a library-default jitter parameter.
ANSWER_RETRY_INTERVAL_SECONDS = 5.0
ANSWER_RETRY_JITTER_SECONDS = 1.0

# Shown when a reply with personal memory markers lands, and replaced once background review
# finishes, without costing an extra Discord edit.
MEMORY_PENDING_NOTE = f"-# {MEMORY_WRITE_EMOJI} 正在整理記憶⋯"

# Closes a reply the surface could not carry to its end. Only `/ask` can reach it, and only on an
# answer past roughly twelve thousand characters; saying so is what keeps it from reading as the
# model getting cut off mid-sentence.
TRUNCATED_NOTICE = "\n-# 回覆太長，這裡放不下後面的內容了"

# Discord pairs these in order: each opens a code block the next one closes, and one with nothing
# after it to close it shows as plain text.
_CODE_FENCE = "```"
# A fence's language tag as Discord reads one. A block reopened past a cut carries it over, so the
# second half keeps its highlighting. Bounded so a reopened block can never put back as much as a
# cut took off, which would split the same text forever.
_FENCE_LANGUAGE_RE = re.compile(pattern=r"[A-Za-z0-9_+\-.#]{0,20}")
# A hard cut never lands beside one of these: combining and enclosing marks, format characters
# (the zero-width joiner, emoji tags), emoji and their modifiers. Each belongs with its neighbour,
# so a cut there tears an emoji sequence or a marked letter in two.
_CLINGING_CATEGORIES = frozenset({"Mn", "Mc", "Me", "Cf", "Sk", "So"})

# The thinking preview is a live glance, not a transcript: keep only the newest few subtext
# lines so a long think never grows into a wall of text above the reply. The char budget is
# the load-bearing half, since one thought line is often a whole paragraph that Discord wraps
# into several rendered lines; the line cap only stops many short lines from stacking up.
REASONING_PREVIEW_MAX_LINES = 4
REASONING_PREVIEW_MAX_CHARS = 320


def _count_url_citations(output: list[ResponseOutputItem]) -> int:
    """Counts `url_citation` annotations across message output text parts in `output`.

    The Responses bridge reports grounding solely as `url_citation` annotations on `output_text`
    parts and never populates `groundingMetadata`.
    """
    return sum(
        1
        for item in output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
        for annotation in part.annotations
        if annotation.type == "url_citation"
    )


def _in_code_block(text: str, cut: int) -> bool:
    """Whether `cut` falls inside a code block of `text`, pairing fences the way Discord does."""
    return text.count(_CODE_FENCE, 0, cut) % 2 == 1 and text.find(_CODE_FENCE, cut) != -1


def _last_break(text: str, end: int, floor: int, outside_code: bool) -> int:
    """Where a message holding at most `text[:end]` should end, never before `floor`.

    The last paragraph break, else line break, else space, skipping any inside a code block when
    `outside_code`. With none, the cut falls at `end`, moved back before a `<...>` it would split
    (a mention, a channel, a custom emoji) and off any emoji or mark it would tear.
    """
    for separator in ("\n\n", "\n", " "):
        cut = text.rfind(separator, 0, end)
        while outside_code and cut >= floor and _in_code_block(text=text, cut=cut):
            cut = text.rfind(separator, 0, cut)
        if cut >= floor:
            return cut
    cut = end
    opening = text.rfind("<", 0, end)
    if opening >= floor and opening > text.rfind(">", 0, end):
        cut = opening
    while cut > floor and _CLINGING_CATEGORIES & {
        unicodedata.category(text[cut - 1]),
        unicodedata.category(text[cut]),
    }:
        cut -= 1
    return cut


def _cut_cleanly(text: str, budget: int, earliest: int) -> tuple[str, str]:
    """Splits `text` into a leading chunk of at most `budget` characters (not before `earliest`) and the remainder.

    Prefers breaking before a code block when text precedes it; otherwise closes an overrun code
    block at the cut and reopens it with its language tag on the remainder.
    """
    floor = max(budget // 2, earliest)
    cut = _last_break(text=text, end=budget, floor=floor, outside_code=True)
    if not _in_code_block(text=text, cut=cut):
        return text[:cut], text[cut:]
    opener = text.rfind(_CODE_FENCE, 0, cut)
    if opener >= earliest and text[:opener].strip():
        return text[:opener], text[opener:]
    language = text[opener + len(_CODE_FENCE) :].split("\n", 1)[0]
    fence = (
        _CODE_FENCE + language if _FENCE_LANGUAGE_RE.fullmatch(string=language) else _CODE_FENCE
    )
    # The closing fence takes room from this message and the reopened one adds to the rest,
    # which still has to fit where `earliest` left room for it.
    end = budget - len(_CODE_FENCE) - 1
    fenced_floor = max(floor, earliest + len(fence) + 1)
    if fenced_floor > end:
        return text[:cut], text[cut:]
    cut = _last_break(text=text, end=end, floor=fenced_floor, outside_code=False)
    head, rest = text[:cut], text[cut:]
    if not _in_code_block(text=text, cut=cut):
        return head, rest
    reopened = f"{fence}{rest}" if rest.startswith("\n") else f"{fence}\n{rest}"
    return f"{head}\n{_CODE_FENCE}", reopened


class ResponseStreamer(BaseModel):
    """Renders one streaming Responses API reply onto a Discord message.

    `consume` is called once per attempt with the answer stream; reasoning summaries are
    previewed as `-#` subtext while the model thinks, the real text replaces them as it
    arrives, then `finalize_reply` writes a usage footer (and an optional memory-credit
    line). Whose memory the answer read is decided before streaming, so the credits are passed
    in via `memory_lookups` rather than discovered here. Discord edits run on a time-based snapshot
    editor task so consuming the stream never waits on Discord.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message] = Field(
        ..., description="The Discord message being answered and replied to."
    )
    surface: TurnSurface = Field(
        ...,
        description="Where this reply goes, and where its reactions and dropped-media hints land.",
    )
    stored_content: str = Field(default="", description="The accumulated reply text.")
    reasoning_content: str = Field(
        default="", description="The accumulated reasoning-summary text shown before content."
    )
    reply: SkipValidation[Message | None] = Field(
        default=None, description="The Discord reply message, created lazily on the first delta."
    )
    carries_turn_notices: bool = Field(
        default=True,
        description=(
            "Whether `reply` is the turn's own surface, so the turn's notices belong on it: a "
            "`Retrying...` notice while an attempt is in flight, and the error embed once every "
            "attempt is spent. False for a media persona reply, which renders onto the DELIVERED "
            "image or video -- a notice there would caption a finished picture with a promise of "
            "more to come and nothing would take it back on the failing path, and the "
            "deliverable itself must never be overwritten by an error."
        ),
    )
    displayed_content: str = Field(
        default="", description="The text last written to the Discord reply."
    )
    content_started: bool = Field(
        default=False,
        description="Whether THIS attempt has seen its first non-newline text delta.",
    )
    content_ever_started: bool = Field(
        default=False,
        description=(
            "Whether any attempt put reply text on screen. Survives a retry reset, so it "
            "answers 'was anything ever written here' where `content_started` answers 'is "
            "this attempt past its leading newlines'."
        ),
    )
    attempts: int = Field(
        default=1, description="How many times the answer stream was opened for this reply."
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
    backend: Literal["responses", "interactions"] = Field(
        default="responses",
        description="Which answer surface produced this stream, logged so a metric only one of "
        "them reports is not read as an absence on the other.",
        examples=["responses", "interactions"],
    )
    input_tokens: int = Field(default=0, description="Input tokens reported by the stream.")
    output_tokens: int = Field(default=0, description="Output tokens reported by the stream.")
    memory_lookups: MemoryCredits = Field(
        default_factory=MemoryCredits,
        description="Credits for the users whose stored memory was injected, for the footer.",
    )
    markers: InlineMarkers = Field(
        default_factory=lambda: InlineMarkers(cleaned_text=""),
        description=(
            "What the finished reply's inline markers asked for, extracted once the stream "
            "ends. The streamer renders the media itself; the research brief and the memory "
            "notes are the caller's, which launches research only after the reply's one media "
            "edit and owns whose memory each note is written to. A media persona reply never "
            "sees the marker instructions, so those stay empty there."
        ),
    )
    voice_generator: SkipValidation[VoiceGenerator | None] = Field(
        default=None,
        description="TTS engine for spoken replies; None disables voice for this reply.",
    )
    voice_text: str = Field(
        default="",
        description="Speechified text of the <generate-voice> segment used as the spoken-clip input.",
    )
    image_generator: SkipValidation[ImageGenerator | None] = Field(
        default=None,
        description="Inline-image renderer; None disables inline <generate-image> for this reply.",
    )
    input_builder: SkipValidation[MessageInputBuilder | None] = Field(
        default=None,
        description=(
            "Loads the message's uploaded images so an inline <generate-image> edits them and an "
            "inline <generate-video> references them."
        ),
    )
    music_generator: SkipValidation[MusicGenerator | None] = Field(
        default=None,
        description="Inline-music renderer; None disables inline <generate-music> for this reply.",
    )
    video_generator: SkipValidation[VideoGenerator | None] = Field(
        default=None,
        description="Inline-video renderer; None disables inline <generate-video> for this reply.",
    )
    media_delivery: MediaDeliveryPlanner = Field(
        ..., description="Decides attach-vs-host-vs-drop for generated media."
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
    # The usage footer appended to stored_content, kept so the media edit can splice any hosted-URL
    # line before it (`USAGE_FOOTER_RE` strips only a footer at end-of-message).
    _usage_footer: str = PrivateAttr(default="")
    # The memory note currently on the reply, so the outcome can replace the pending one rather
    # than stack under it, and so the answer text handed back to the caller can drop it again.
    _memory_note: str = PrivateAttr(default="")
    # The dropped-media hint line currently on the reply, tracked for the same second reason:
    # it is chrome the bot added, so it must not reach the transcript the memory reviewer reads
    # back or the `/ask` history the next turn replays.
    _hint_line: str = PrivateAttr(default="")
    # Set when the reply message was deleted while streaming, so the media step knows the
    # difference between "never sent" (a real problem, worth a hint) and "sent then deleted".
    _reply_deleted: bool = PrivateAttr(default=False)
    # Set once the preview editor has reported a failed snapshot write, so the 1s cadence does
    # not repeat the same warn for the rest of the stream. Deliberately survives a retry reset:
    # a Discord edit that failed on one attempt fails on the next for the same reason.
    _preview_error_logged: bool = PrivateAttr(default=False)
    # Set once the first-reasoning-delta latency has been logged. Separate from `reasoning_content`
    # (which a retry clears) so a retried turn logs first-reasoning latency only once.
    _reasoning_logged: bool = PrivateAttr(default=False)
    # How long the answer call itself took, stamped when the stream is consumed.
    _answer_seconds: float = PrivateAttr(default=0.0)
    # Grounding citations the completed response carried, or None when this backend never
    # reported any. The distinction is the point: only the Responses bridge reports grounding,
    # and only as `url_citation` annotations, so a zero written for the Interactions path (or for
    # a stream that ended without `response.completed`) would read as a genuinely ungrounded
    # answer. See the grounding note in `.agents/rules/responses-api.md`.
    _url_citations: int | None = PrivateAttr(default=None)

    @staticmethod
    def _split_reply_for_discord(
        content: str, footer: str, max_messages: int | None = None
    ) -> tuple[str, list[str]]:
        """Splits `content` plus `footer` into a parent message and follow-up chunks within `max_messages`.

        Each chunk is cut via `_cut_cleanly`. When `max_messages` is reached (such as `/ask`'s
        follow-up limit), the final allowed message truncates remaining content and appends
        `TRUNCATED_NOTICE`.
        """
        if len(f"{content}{footer}") <= DISCORD_MESSAGE_LIMIT:
            return f"{content}{footer}", []
        if len(footer) >= DISCORD_MESSAGE_LIMIT:
            raise ValueError("Usage footer is too long for Discord message content")

        messages: list[str] = []
        remaining = content
        while len(remaining) + len(footer) > DISCORD_MESSAGE_LIMIT:
            left = None if max_messages is None else max_messages - len(messages)
            if left == 1:
                budget = DISCORD_MESSAGE_LIMIT - len(footer) - len(TRUNCATED_NOTICE)
                # No message follows, so ending before a code block that opens early would only
                # waste the room: such a block is closed where the cut falls instead.
                head, _ = _cut_cleanly(text=remaining, budget=budget, earliest=budget // 2)
                remaining = f"{head}{TRUNCATED_NOTICE}"
                break
            if not messages and len(remaining) <= DISCORD_MESSAGE_LIMIT:
                # An answer that fits one message is not split for its footer, which follows alone.
                messages.append(remaining)
                remaining = ""
                break
            # Past the first message the footer keeps some text in front of it: alone, Discord
            # trims the blank line `USAGE_FOOTER_RE` anchors on and the footer rides into history.
            budget = (
                DISCORD_MESSAGE_LIMIT
                if len(remaining) > DISCORD_MESSAGE_LIMIT
                else DISCORD_MESSAGE_LIMIT - len(footer)
            )
            # The earliest cut that leaves the rest room in the messages still allowed after this
            # one. Past this whole message the answer overflows anyway, and the last one cuts it.
            earliest = 0
            if left is not None:
                needed = len(remaining) + len(footer) - (left - 1) * DISCORD_MESSAGE_LIMIT
                earliest = needed if needed <= budget else 0
            head, remaining = _cut_cleanly(text=remaining, budget=budget, earliest=earliest)
            messages.append(head)
        messages.append(f"{remaining}{footer}")
        return messages[0], messages[1:]

    def _render_preview(self) -> str:
        """Builds the current streaming preview: real content once started, else reasoning.

        The reasoning preview shows the tail of the model's thought summary as `-#`
        subtext lines under a `message` app-emoji header, so the user watches the thinking until
        the first real content delta replaces it. The window keeps only the newest lines within
        `REASONING_PREVIEW_MAX_LINES` / `REASONING_PREVIEW_MAX_CHARS`, so a long think stays a
        few rendered lines tall instead of filling the message. A single paragraph wider than the
        budget keeps its own tail behind an ellipsis, so the newest thought always shows.
        """
        if self.content_started:
            return scrub_markers_for_preview(text=self.stored_content)[:DISCORD_MESSAGE_LIMIT]
        if not self.reasoning_content:
            return ""
        # Mentions are escaped because this transient text is never meant to ping;
        # the real reply may mention people, the thought process must not.
        tail = escape_mentions(self.reasoning_content[-1500:])
        lines = [line for line in tail.splitlines() if line.strip()]
        header = f"-# {ANSWER_EMOJI} Thinking..."
        budget = REASONING_PREVIEW_MAX_CHARS
        kept: list[str] = []
        for line in reversed(lines[-REASONING_PREVIEW_MAX_LINES:]):
            if len(line) > budget:
                if not kept:
                    kept.append(f"…{line[-budget:]}")
                break
            kept.append(line)
            budget -= len(line) + 1
        kept.reverse()
        return "\n".join([header, *(f"-# {line}" for line in kept)])

    async def _reply_or_send(self, content: str) -> Message:
        """Replies to the source message, sending unparented if it was deleted.

        A source deleted before the reply lands is logged and the reply sent into the same
        channel instead of wasting the whole pipeline. Other HTTP errors still propagate to the
        caller.
        """
        try:
            return await self.surface.send(content=content)
        except HTTPException as exc:
            if not is_reply_target_gone(error=exc):
                raise
            logfire.info(
                "Source message deleted before reply; sending unparented",
                message_id=self.message.id,
            )
            return await self.surface.send_unparented(content=content)

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
        write always completes before `finalize_reply` runs; a cancel landing inside
        the first `message.reply` could otherwise orphan the created message and let
        the finalizer create a duplicate.
        """
        while True:
            try:
                await asyncio.wait_for(
                    self._editor_stop.wait(), timeout=self.preview_interval_seconds
                )
            except TimeoutError:
                try:
                    await self._write_preview_snapshot()
                except NotFound:
                    # The reply was deleted mid-stream; a normal end, handled again in
                    # _write_final_message. Nothing to repair, so stop previewing.
                    logfire.info(
                        "Reply deleted while streaming; stopping preview edits",
                        message_id=self.message.id,
                    )
                    return
                except Forbidden:
                    # A refusal repeats on every tick, and the ids are the whole finding.
                    logfire.warn(
                        "Channel refused a preview write; stopping preview edits",
                        channel_id=self.message.channel.id,
                        message_id=self.message.id,
                    )
                    return
                except Exception as exc:
                    # Broad on purpose: the preview is best-effort and must never break the
                    # stream, but a persistent failure kills the whole live-preview UX, so the
                    # first one is recorded. Logged once because displayed_content is not
                    # advanced on failure, so the same error repeats every tick.
                    if not self._preview_error_logged:
                        self._preview_error_logged = True
                        logfire.warn(
                            "Preview snapshot edit failed; continuing to stream",
                            message_id=self.message.id,
                            error_type=type(exc).__name__,
                            _exc_info=exc,
                        )
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
        try:
            await self._editor_task
        except Exception as exc:
            # Broad on purpose: the editor is a best-effort UX task, its death must not sink
            # the finished reply. Cancellation still propagates (it is a BaseException).
            logfire.warn(
                "Preview editor task crashed; the reply still finalizes without live preview",
                message_id=self.message.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        self._editor_task = None

    async def _write_final_message(self, content: str, footer: str) -> None:
        """Writes the final reply, continuing overflow as follow-ups on the same surface.

        A reply deleted while it streamed (author delete, moderator purge) makes the final edit
        404 with code 10008. That is a normal end, not a failure: the answer is complete and the
        message it belonged to is gone on purpose, so the handle is dropped and nothing is
        re-sent, since re-sending would resurrect exactly what someone just removed.

        The overflow goes through the surface rather than `previous.reply` because on the `/ask`
        route it must not be a channel send at all: a follow-up there carries a
        `PartialMessageable`, so replying to it would post into a channel the bot is not in and
        every answer over the limit would lose its tail.
        """
        parent_content, follow_up_chunks = self._split_reply_for_discord(
            content=content,
            footer=footer,
            max_messages=self.surface.answer_capacity(has_landed_reply=self.reply is not None),
        )
        # Track the parent reply so a later voice attach edits the right message even when
        # the reply is created here (no preview snapshot ran before finalize).
        if self.reply is None:
            self.reply = await self._reply_or_send(content=parent_content)
        else:
            try:
                await self.reply.edit(content=parent_content)
            except NotFound:
                logfire.info(
                    "Reply deleted while streaming; finishing without the final edit",
                    message_id=self.message.id,
                    reply_id=self.reply.id,
                )
                self.reply = None
                self._reply_deleted = True
                return
        previous = self.reply
        for chunk in follow_up_chunks:
            previous = await self.surface.follow_up(previous=previous, content=chunk)

    def _on_reasoning_delta(self, delta: str) -> None:
        """Accumulates one reasoning-summary delta, logging the first one's latency."""
        if not self.reasoning_content:
            # Gemini may prepend newlines to the first reasoning delta too.
            delta = delta.lstrip("\n")
            if not delta:
                return
            if not self._reasoning_logged:
                self._reasoning_logged = True
                logfire.info(
                    "gen_reply first reasoning delta",
                    elapsed_seconds=time.monotonic() - self.created_at,
                    model=self.model_name,
                    message_id=self.message.id,
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
            if not self.content_ever_started:
                self.content_ever_started = True
                logfire.info(
                    "gen_reply first content delta",
                    elapsed_seconds=time.monotonic() - self.created_at,
                    model=self.model_name,
                    message_id=self.message.id,
                )
        self.stored_content += delta
        self._ensure_editor_started()

    def reset_for_retry(self) -> None:
        """Resets per-attempt text state and re-arms the preview editor for the next attempt.

        Preserves `reply`, `created_at`, `content_ever_started`, `_reasoning_logged`, and token
        counters across attempts, and clears `_editor_stop` so the next delta can restart live
        preview edits.
        """
        self.attempts += 1
        self.stored_content = ""
        self.reasoning_content = ""
        self.content_started = False
        self._editor_stop.clear()

    async def announce_retry(self) -> None:
        """Marks the source message with `RETRY_HINT_EMOJI` and edits an existing reply to a retry notice.

        Does not create a new reply message when none exists yet, and suppresses Discord HTTP
        errors so announcing a retry never aborts the retry itself.
        """
        await self.surface.mark(emoji=RETRY_HINT_EMOJI)
        if self.reply is None:
            return
        notice = self._retry_notice()
        with contextlib.suppress(HTTPException):
            await self.reply.edit(content=notice)
            self.displayed_content = notice

    def _retry_notice(self) -> str:
        """The text a reply carries while the next attempt is in flight."""
        return f"-# {RETRY_HINT_EMOJI} Retrying... ({self.attempts}/{ANSWER_STREAM_MAX_ATTEMPTS})"

    async def withdraw_retry_notice(self) -> None:
        """Deletes `reply` if it is still showing only the retry notice after all attempts fail."""
        if self.reply is None or self.displayed_content != self._retry_notice():
            return
        with contextlib.suppress(HTTPException):
            await self.reply.delete()
        self.reply = None

    async def land_failure(self, embed: Embed) -> bool:
        """Returns whether `embed` was edited onto the in-progress `reply` message.

        Preserves any partial answer text while clearing a reasoning-only preview with `content=""`
        (never `None`, which nextcord omits from multipart edits carrying the embed spacer file).
        Returns False when no `reply` exists or Discord rejects the edit so the caller can send a
        fresh failure message.

        Args:
            embed (Embed): Failure embed to attach to the reply.
        """
        if self.reply is None:
            return False
        try:
            await self.reply.edit(
                content=self._render_preview() if self.content_started else "",
                embed=embed,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=self.reply),
            )
        except NotFound:
            # Someone removed the half-written reply while the answer was failing: a routine
            # end rather than a defect, and the caller's fresh message is the whole repair.
            logfire.info(
                "The streamed reply is gone; posting the failure fresh",
                message_id=self.message.id,
                reply_id=self.reply.id,
            )
            return False
        except HTTPException as exc:
            # Anything else is Discord refusing the edit (a rejected spacer upload, an edit
            # rate limit), which costs the user the single-message shape this method exists for.
            logfire.warn(
                "Could not land the failure on the streamed reply; posting it fresh",
                message_id=self.message.id,
                reply_id=self.reply.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return False
        return True

    async def _consume(self, responses: AsyncIterator[ResponseStreamEvent]) -> None:
        """Streams the reply, accumulating text and usage onto the instance.

        Only accumulates state; the snapshot editor task renders it to Discord, so this
        loop never blocks on a Discord edit between deltas.
        """
        # Discriminate on the `type` literal, never isinstance against the SDK event classes:
        # the YouTube path's `adapt_interactions_stream` yields duck-typed namespaces that carry
        # only the field names read here, so an isinstance check would silently drop its events.
        async for response in responses:
            if response.type == "response.created":
                # Capture the model on `created` too so the usage footer never falls back
                # to an empty model name (and no cost) when a stream ends without a
                # clean `completed` event.
                self.model_name = response.response.model
            elif response.type == "response.completed":
                self.model_name = response.response.model
                if response.response.usage:
                    self.input_tokens += response.response.usage.input_tokens
                    self.output_tokens += response.response.usage.output_tokens
                # `output` is None on the Interactions path, which reports grounding in a shape
                # this event cannot carry; leaving the counter None there keeps "not reported"
                # distinct from "reported as zero".
                if response.response.output is not None:
                    self._url_citations = (self._url_citations or 0) + _count_url_citations(
                        output=response.response.output
                    )
            elif response.type == "response.reasoning_summary_text.delta":
                self._on_reasoning_delta(delta=response.delta)
            elif response.type == "response.output_text.delta":
                self._on_content_delta(delta=response.delta)

    async def finalize_reply(self) -> str:
        """Returns the finalized reply text after extracting inline markers, writing the footer, and attaching media."""
        # Priced as the model the stream reports having answered from, which is not always the
        # one that was dispatched: a LiteLLM fallback shows up here and nowhere else.
        usage_line, cost = render_usage_footer(
            model_name=self.model_name,
            label=f"{self.model_name} ({self.model_effort})"
            if self.model_effort
            else self.model_name,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

        self.stored_content = CODED_MENTION_RE.sub(r"\1", self.stored_content)
        self.markers = extract_inline_markers(text=self.stored_content)
        self.stored_content = self.markers.cleaned_text
        # The spoken clip must not narrate raw Discord markup (a `<@id>` mention reads as a bare
        # snowflake), so the voice input is normalised while the visible reply keeps its markup.
        self.voice_text = (
            speechify_discord_markup(
                text=self.markers.voice_text, resolve_name=self._resolve_mention_name
            )
            if self.markers.voice_requested
            else ""
        )
        # Credit looked-up memory owners on a second -# subtext line. Dedupe while
        # preserving lookup order; past two names collapse to "等 N 人" so a busy
        # lookup stays short. USAGE_FOOTER_RE matches this optional second line too.
        # A user the reply read but cannot name is folded into that same count rather
        # than printed, which is the only shape that reports them at all (see
        # `MemoryCredits`), so the collapse also fires whenever there is one.
        memory_line = ""
        names = list(dict.fromkeys(self.memory_lookups.named))
        read_count = len(names) + self.memory_lookups.unnamed
        if not names and read_count:
            memory_line = f"\n-# {MEMORY_READ_EMOJI} 讀了 {read_count} 人的記憶"
        elif read_count > len(names) or read_count > 2:
            memory_line = (
                f"\n-# {MEMORY_READ_EMOJI} 讀了 {', '.join(names[:2])} 等 {read_count} 人的記憶"
            )
        elif names:
            memory_line = f"\n-# {MEMORY_READ_EMOJI} 讀了 {', '.join(names)} 的記憶"
        # `utils/llm_transcript.py::USAGE_FOOTER_RE` anchors on the blank line opening the footer.
        usage_footer = f"\n\n{usage_line}{memory_line}"

        reply_chars = len(self.stored_content)
        chunked = reply_chars + len(usage_footer) > DISCORD_MESSAGE_LIMIT
        if self._wants_pending_memory_note(footer_chars=len(usage_footer)):
            self._memory_note = MEMORY_PENDING_NOTE
            self.stored_content = f"{self.stored_content}\n{self._memory_note}"
        await self._write_final_message(content=self.stored_content, footer=usage_footer)
        self.stored_content += usage_footer
        self._usage_footer = usage_footer
        # The answer is on screen in full, so the turn's failure path lets go of it here rather
        # than when the caller returns: the media attach below is unprotected, and an error
        # landing on a finished reply would take its attachments with it, the spacer payload
        # retaining none of them.
        if self.carries_turn_notices:
            current_answer_streamer.set(None)

        await self._attach_generated_media()
        # Every best-effort hint this turn produced is in by now, so the surface that could not
        # react gets its one line here rather than an edit per dropped item.
        await self._write_hint_line()
        logfire.info(
            "gen_reply reply finalized",
            message_id=self.message.id,
            model=self.model_name,
            backend=self.backend,
            effort=self.model_effort,
            answer_seconds=self._answer_seconds,
            attempts=self.attempts,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost=cost,
            url_citations=self._url_citations,
            reasoning_chars=len(self.reasoning_content),
            reply_chars=reply_chars,
            voice_requested=self.markers.voice_requested,
            image_count=self.markers.image_requests,
            music_requested=bool(self.markers.music_prompt),
            video_requested=bool(self.markers.video_prompt),
            memory_lookups=self.memory_lookups.total,
            chunked=chunked,
        )
        return self._without_added_lines(text=self.stored_content)

    def _without_memory_note(self, text: str) -> str:
        """Returns `text` with the memory note currently on the reply removed, if any."""
        if not self._memory_note:
            return text
        return text.replace(f"\n{self._memory_note}", "", 1)

    def _without_added_lines(self, text: str) -> str:
        """Returns `text` without the memory note and the dropped-media hint line.

        Both are chrome rather than something the answer said, and both have to go before the
        text becomes `full_reply`: that string is the transcript the memory reviewer reads back
        and the `/ask` turn the store replays. `USAGE_FOOTER_RE` cannot be relied on for either:
        it takes the note only while the note sits directly above the footer, which a hosted-URL
        line breaks, and it never takes the hint.
        """
        stripped = self._without_memory_note(text=text)
        if not self._hint_line:
            return stripped
        return stripped.replace(f"\n{self._hint_line}", "", 1)

    def _wants_pending_memory_note(self, footer_chars: int) -> bool:
        """Returns whether `reply` should display `MEMORY_PENDING_NOTE` before background review.

        Requires `carries_turn_notices`, at least one personal write or forget marker, and enough
        room to fit the note and usage footer within `DISCORD_MESSAGE_LIMIT`.
        """
        if not self.carries_turn_notices:
            return False
        if not (self.markers.memory_notes or self.markers.forget_notes):
            return False
        spliced = len(self.stored_content) + 1 + len(MEMORY_PENDING_NOTE) + footer_chars
        return spliced <= DISCORD_MESSAGE_LIMIT

    async def set_memory_note(self, line: str) -> None:
        """Replaces the memory status line on `reply` before the usage footer.

        Splices `line` above `_usage_footer` so `USAGE_FOOTER_RE` still strips the footer from
        later history, and removes any previous note by value since hosted-media links may sit
        between the note and the footer. Does nothing when `reply` is absent or when adding a
        first note would exceed `DISCORD_MESSAGE_LIMIT`; when a replacement overflows an existing
        pending note, withdraws the pending note instead of leaving it stuck.

        Args:
            line (str): Formatted memory subtext line to display.
        """
        if self.reply is None or not self._usage_footer:
            return
        body = self._without_memory_note(text=self.stored_content.removesuffix(self._usage_footer))
        updated = f"{body}\n{line}{self._usage_footer}"
        if len(updated) > DISCORD_MESSAGE_LIMIT:
            if not self._memory_note:
                return
            updated, line = f"{body}{self._usage_footer}", ""
        if not await self._edit_reply_text(
            reply=self.reply,
            content=updated,
            failure="Failed to write the memory note onto the reply",
        ):
            # The note already on the reply stays recorded, so a later attempt still replaces it
            # rather than stacking under it.
            return
        self._memory_note = line

    async def _edit_reply_text(self, reply: Message, content: str, failure: str) -> bool:
        """Rewrites the reply's text to `content`, which becomes `stored_content` once it lands.

        Returns False, logged as `failure`, when the edit did not land.
        """
        try:
            await reply.edit(content=content, allowed_mentions=AllowedMentions.none())
        except Exception as exc:
            # Broad on purpose: the reply may have been deleted, and a footnote is never worth
            # surfacing a failure for.
            logfire.warn(
                failure, message_id=self.message.id, error_type=type(exc).__name__, _exc_info=exc
            )
            return False
        self.stored_content = content
        return True

    def _resolve_mention_name(self, target_id: int) -> str | None:
        """Looks up a member/role/channel display name for the spoken-clip mention rewrite."""
        guild = self.message.guild
        if guild is None:
            return None
        member = guild.get_member(target_id)
        if member is not None:
            return member.display_name
        role = guild.get_role(target_id)
        if role is not None:
            return role.name
        channel = guild.get_channel(target_id)
        return getattr(channel, "name", None) if channel is not None else None

    async def _write_hint_line(self) -> None:
        """Writes the hints a surface without reactions collected, as one line on the reply.

        Placed and tracked exactly like the memory note — before the usage footer, kept by value
        so `set_memory_note` can rebuild the reply around it and `_without_added_lines` can keep
        it out of `full_reply`. Best-effort throughout: a hint about a failure must never become
        one, and there is nothing to write when the reply never landed or the surface reacts.
        """
        hints = self.surface.take_hints()
        if not hints or self.reply is None or not self._usage_footer:
            return
        body = self._without_memory_note(text=self.stored_content.removesuffix(self._usage_footer))
        line = f"-# {''.join(hints)}"
        note = f"\n{self._memory_note}" if self._memory_note else ""
        updated = f"{body}\n{line}{note}{self._usage_footer}"
        if len(updated) > DISCORD_MESSAGE_LIMIT:
            # Nowhere left to say it. Recorded rather than dropped in silence, since the point of
            # the hint is that a dropped clip never goes unmentioned.
            logfire.warn(
                "No room left on the reply for the dropped-media hints",
                message_id=self.message.id,
                hints="".join(hints),
            )
            return
        if await self._edit_reply_text(
            reply=self.reply,
            content=updated,
            failure="Failed to write the dropped-media hints onto the reply",
        ):
            self._hint_line = line

    async def _build_voice_candidate(self) -> MediaItem | None:
        """Synthesizes the <generate-voice> segment to a WAV candidate, or None when not delivered.

        Best-effort: a skip (not requested / disabled / empty) is silent, while a
        requested-but-failed clip (timeout / refusal) hints the source message and returns None.
        The upload-limit decision (attach vs host vs drop) is made by `_attach_generated_media`,
        so an oversized clip is not dropped here (there is deliberately no spoken-length cap).
        """
        if not self.markers.voice_requested:
            logfire.debug("Voice not requested by the answer model", message_id=self.message.id)
            return None
        if self.voice_generator is None:
            # Voice is intentionally off this turn (kill-switch), not a failure: no hint.
            logfire.info(
                "Voice requested but disabled for this turn; replying without audio",
                message_id=self.message.id,
            )
            return None
        await self.surface.mark(emoji=VOICE_EMOJI)
        logfire.info(
            "Synthesizing voice reply", message_id=self.message.id, text_chars=len(self.voice_text)
        )
        clip = await self.voice_generator.generate(
            text=self.voice_text, end_user_id=self.message.author.name
        )
        if clip.outcome == "empty":
            # Nothing to say (the segment was empty after stripping): no hint.
            return None
        if clip.outcome == "timeout":
            # generate() logged the timeout; cue the user that the clip ran out of time.
            await self.surface.hint(emoji=TIMEOUT_HINT_EMOJI)
            return None
        if clip.audio is None:
            # Any other synthesis failure (most often a policy refusal); generate() logged it.
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
            return None
        return MediaItem(source=clip.audio, filename=VOICE_REPLY_FILENAME)

    async def _load_marker_source_images(self) -> list[LoadedMedia]:
        """Best-effort source images from the current and replied-to message.

        Each source keeps its mime beside the bytes, since omni rejects an image content block
        with an empty mime. Best-effort: no builder or a load failure simply yields no sources, so
        the marker falls back to fresh generation.
        """
        if self.input_builder is None:
            return []
        try:
            return await self.input_builder.get_turn_image_sources(
                message=self.message, replied_to=replied_to_message(message=self.message)
            )
        except Exception as exc:  # broad: best-effort source load, see docstring
            logfire.warn(
                "Inline image source load failed; generating without source pixels",
                message_id=self.message.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return []

    async def _build_image_candidates(
        self, source_images_task: asyncio.Task[list[LoadedMedia]] | None
    ) -> list[MediaItem]:
        """Renders the <generate-image> requests to PNG candidates, in order; [] when none delivered.

        Best-effort like voice: no request or a disabled generator is silent. The capped prompts
        render concurrently; a generation failure drops that image and a single ⚠️ hint rides on
        the source message. The upload-limit decision (attach vs host) is left to
        `_attach_generated_media`, so a large image is not dropped here for size. The uploaded
        source images (for editing) are awaited from the shared `source_images_task` so an inline
        `<generate-image>` and `<generate-video>` in the same reply load them only once; only the raw
        bytes are used here (the edit path needs no mime).
        """
        if not self.markers.image_prompts:
            return []
        if self.image_generator is None:
            # Inline image is intentionally off this turn (kill-switch / non-QA route): no hint.
            logfire.info(
                "Inline image requested but disabled for this turn; replying without an image",
                message_id=self.message.id,
            )
            return []
        if self.markers.image_requests > MAX_INLINE_IMAGES:
            logfire.info(
                "Inline image requests exceed the per-reply cap; dropping the extras",
                message_id=self.message.id,
                requested=self.markers.image_requests,
                cap=MAX_INLINE_IMAGES,
            )
        await self.surface.mark(emoji=IMAGE_EMOJI)
        logfire.info(
            "Generating inline image reply",
            message_id=self.message.id,
            image_count=len(self.markers.image_prompts),
        )
        source_images = await source_images_task if source_images_task is not None else []
        source_bytes = [loaded.data for loaded in source_images]
        images = await asyncio.gather(
            *(
                self.image_generator.generate(
                    user_prompt=prompt,
                    end_user_id=self.message.author.name,
                    image_bytes_list=source_bytes or None,
                )
                for prompt in self.markers.image_prompts
            )
        )
        candidates: list[MediaItem] = []
        dropped = False
        for index, image in enumerate(images, start=1):
            if image is None:
                # generate() logged the failure/timeout; hint once after the loop.
                dropped = True
                continue
            # A single image keeps `generated.png` to mirror the IMAGE route; multiples need
            # distinct names since Discord collides on duplicate attachment filenames.
            filename = (
                INLINE_IMAGE_FILENAME
                if len(self.markers.image_prompts) == 1
                else f"generated_{index}.png"
            )
            candidates.append(MediaItem(source=image, filename=filename))
        if dropped:
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
        return candidates

    async def _build_music_candidate(self) -> MediaItem | None:
        """Generates the <generate-music> clip to an audio candidate, or None when not delivered.

        Best-effort like the inline image path: a skip (not requested / disabled) is silent, while
        a requested-but-failed clip hints the source message and returns None. The filename suffix
        follows the returned audio mime type so Discord (or the hosted link) renders a player; the
        upload-limit decision (attach vs host) is left to `_attach_generated_media`.
        """
        if self.markers.music_prompt is None:
            logfire.debug("Music not requested by the answer model", message_id=self.message.id)
            return None
        if self.music_generator is None:
            # Music is intentionally off this turn (kill-switch / missing key): no hint.
            logfire.info(
                "Inline music requested but disabled for this turn; replying without music",
                message_id=self.message.id,
            )
            return None
        await self.surface.mark(emoji=MUSIC_EMOJI)
        logfire.info("Generating inline music reply", message_id=self.message.id)
        clip = await self.music_generator.generate(user_prompt=self.markers.music_prompt)
        if clip is None:
            # generate() logged the failure/timeout; hint once.
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
            return None
        return MediaItem(source=clip.audio, filename=music_filename(mime_type=clip.mime_type))

    async def _build_video_candidate(
        self, source_images_task: asyncio.Task[list[LoadedMedia]] | None
    ) -> MediaItem | None:
        """Generates the <generate-video> clip to an MP4 candidate, or None when not delivered.

        Best-effort like the inline music path: a skip (not requested / disabled) is silent, while
        a requested-but-failed clip hints the source message and returns None. When the user
        uploaded image(s) they are awaited from the shared `source_images_task` and ride as
        `(bytes, mime)` reference pairs so omni infers the task; otherwise it is plain text-to-video.
        The upload-limit decision (attach vs host) is left to `_attach_generated_media`, so a large
        clip is hosted as a URL rather than dropped for size.
        """
        if self.markers.video_prompt is None:
            logfire.debug("Video not requested by the answer model", message_id=self.message.id)
            return None
        if self.video_generator is None:
            # Video is intentionally off this turn (kill-switch / missing key): no hint.
            logfire.info(
                "Inline video requested but disabled for this turn; replying without video",
                message_id=self.message.id,
            )
            return None
        await self.surface.mark(emoji=VIDEO_EMOJI)
        logfire.info("Generating inline video reply", message_id=self.message.id)
        source_images = await source_images_task if source_images_task is not None else []
        video_bytes = await self.video_generator.generate(
            user_prompt=self.markers.video_prompt, reference_image_sources=source_images or None
        )
        if video_bytes is None:
            # generate() logged the failure/timeout; hint once.
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
            return None
        return MediaItem(source=video_bytes, filename=INLINE_VIDEO_FILENAME)

    async def _attach_generated_media(self) -> None:
        """Attaches the voice, music, video, and image media in one edit, hosting overflow.

        The text reply is already on screen, so this adds no latency to it; the media are
        best-effort. Anything that fits the upload limit rides a single `reply.edit(files=...)`
        (one edit, because `edit` replaces the attachment list). Anything too big to upload (a
        long voice WAV in a DM, or almost any video clip) is hosted on the external static server
        and its URL appended to the reply instead of being dropped; if hosting is unavailable it
        degrades to the drop + ⚠️ hint. Voice/music/video are ordered first because the planner
        applies Discord's attachment-count cap to the tail, so the rare over-cap overflow drops a
        trailing image rather than a clip.
        """
        if self._reply_deleted:
            # There is no message left to attach to, and the user removed it themselves, so a
            # ⚠️ hint on their message would be noise about media they cannot see anyway.
            return
        if self.reply is None:
            if self.markers.media_requested:
                logfire.warn(
                    "Media requested but the reply was never sent; dropping it",
                    message_id=self.message.id,
                )
                await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
            return
        reply = self.reply
        # The uploaded source images (for editing an inline <generate-image> / grounding an inline <generate-video>)
        # are loaded at most once even when both markers fire, since load_image_bytes re-fetches
        # per call; both builders await this shared task. None when neither visual marker fired.
        source_images_task = (
            asyncio.ensure_future(self._load_marker_source_images())
            if (self.image_generator is not None and self.markers.image_prompts)
            or (self.video_generator is not None and self.markers.video_prompt is not None)
            else None
        )
        # Build every path concurrently so a slow one never blocks the others: a TTS clip, a music
        # render, or a video render that hangs to its timeout must not delay ready inline images
        # (and vice versa).
        voice_task = asyncio.ensure_future(self._build_voice_candidate())
        music_task = asyncio.ensure_future(self._build_music_candidate())
        video_task = asyncio.ensure_future(
            self._build_video_candidate(source_images_task=source_images_task)
        )
        image_task = asyncio.ensure_future(
            self._build_image_candidates(source_images_task=source_images_task)
        )
        # On `/ask` the attach edit and the hint line both go through the interaction token, so a
        # builder still running when the budget ends is dropped while the ⏱️ can still be written,
        # and the builders that finished in time attach as usual. The cut is per builder: one slow
        # inline image drops the images that finished beside it. Unlike `gather`, `wait` leaves
        # its tasks running when the turn itself is cancelled, hence the finally.
        builds = {voice_task, music_task, video_task, image_task}
        try:
            done, pending = await asyncio.wait(
                builds, timeout=self.surface.delivery_budget_seconds()
            )
        finally:
            for task in builds:
                task.cancel()
        if pending:
            await asyncio.wait(pending)
            logfire.warn(
                "gen_reply inline media outlived the /ask window; dropping it",
                message_id=self.message.id,
                pending=len(pending),
            )
            await self.surface.hint(emoji=TIMEOUT_HINT_EMOJI)
        voice_candidate = voice_task.result() if voice_task in done else None
        music_candidate = music_task.result() if music_task in done else None
        video_candidate = video_task.result() if video_task in done else None
        image_candidates = image_task.result() if image_task in done else []
        items = [
            item
            for item in (voice_candidate, music_candidate, video_candidate, *image_candidates)
            if item is not None
        ]
        if not items:
            return
        plan = await self.media_delivery.plan(
            items=items,
            upload_limit=upload_limit_for(
                guild=self.message.guild, interaction=self.surface.interaction
            ),
            envelope_margin=MEDIA_ENVELOPE_MARGIN,
        )
        files = [item.to_file() for item in plan.native]
        if not await self._finalize_media_edit(
            reply=reply, files=files, hosted_urls=plan.hosted_urls
        ):
            return
        if plan.dropped_items or plan.clamped_items:
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
        logfire.info(
            "Generated media attached",
            message_id=self.message.id,
            file_count=len(files),
            hosted_count=len(plan.hosted_urls),
        )

    async def _finalize_media_edit(
        self, reply: Message, files: list[File], hosted_urls: list[str]
    ) -> bool:
        """Edits `files` and any `hosted_urls` onto `reply`, returning whether delivery succeeded.

        Splices `hosted_urls` before the usage footer when they fit `DISCORD_MESSAGE_LIMIT`, or
        posts them as a follow-up message otherwise.
        """
        if not files and not hosted_urls:
            return True
        content: str | None = None
        follow_up: str | None = None
        if hosted_urls:
            link_line = "\n-# 媒體過大，改用連結\n" + "\n".join(hosted_urls)
            if len(self.stored_content) + len(link_line) <= DISCORD_MESSAGE_LIMIT:
                # Splice the link before the usage footer (`stored_content` ends with it):
                # appending after the footer would leave `USAGE_FOOTER_RE` unable to strip it from
                # later history.
                body = self.stored_content.removesuffix(self._usage_footer)
                self.stored_content = f"{body}{link_line}{self._usage_footer}"
                content = self.stored_content
            else:
                follow_up = link_line.lstrip("\n")
        try:
            if files and content is not None:
                await reply.edit(
                    content=content, files=files, allowed_mentions=AllowedMentions.none()
                )
            elif files:
                await reply.edit(files=files, allowed_mentions=AllowedMentions.none())
            elif content is not None:
                await reply.edit(content=content, allowed_mentions=AllowedMentions.none())
        except Exception as exc:
            # Broad on purpose: this single edit is the best-effort delivery of every generated
            # attachment and must never raise into the reply pipeline. error_type separates a
            # deleted reply (NotFound) from a rejected payload (HTTPException, i.e. the planner
            # mis-decided the upload limit).
            logfire.warn(
                "Failed to attach generated media onto the reply",
                message_id=self.message.id,
                file_count=len(files),
                hosted_count=len(hosted_urls),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
            return False
        if follow_up is not None:
            try:
                await self.surface.follow_up(
                    previous=reply, content=follow_up, allowed_mentions=AllowedMentions.none()
                )
            except Exception as exc:
                # Broad on purpose: a deleted parent or any Discord HTTP error must never raise
                # into the reply pipeline. The follow-up is the delivery of the hosted clip, so a
                # failure here means the media is gone.
                logfire.warn(
                    "Hosted media link follow-up failed; the media URL was never posted",
                    message_id=self.message.id,
                    hosted_url_count=len(hosted_urls),
                    file_count=len(files),
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
                await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
                return False
        return True

    async def consume(self, responses: AsyncIterator[ResponseStreamEvent]) -> None:
        """Consumes `responses` into this streamer and stops the live preview editor when done.

        Args:
            responses (AsyncIterator[ResponseStreamEvent]): Stream of Responses API events for one
                attempt.
        """
        if self.carries_turn_notices:
            current_answer_streamer.set(self)
        try:
            await self._consume(responses=responses)
        finally:
            # Stamped here rather than at finalize, which runs after the inline media is
            # generated: a `<generate-video>` render can take minutes and would otherwise be
            # reported as answer latency.
            self._answer_seconds = time.monotonic() - self.created_at
            await self._stop_editor()


# Opens one answer stream. A factory rather than the stream itself, because the request has to
# be re-issued per attempt, and async because the Responses backend awaits `responses.create`
# while the Interactions one returns its generator directly.
type AnswerStreamFactory = Callable[[], Awaitable[AsyncIterator[ResponseStreamEvent]]]


async def stream_answer_with_retry(
    streamer: ResponseStreamer, open_stream: AnswerStreamFactory, message_id: int
) -> str:
    """Returns the finalized reply text after streaming `open_stream` into `streamer` with retries.

    The streaming answer turn is the only LLM request the bot re-issues mid-turn on an error:
    non-streaming proxied calls rely on LiteLLM's router retries and fallbacks (or degrade
    locally), whereas a streaming answer turn can fail across all proxy fallback hops or run
    direct to Google on the Interactions API with no proxy behind it. Re-issuing is safe because
    an answer turn is a pure read with no side effects before the stream completes, and stays on
    the same client and model so uploaded Files API URIs remain valid.

    Between attempts, `reset_for_retry` clears partial text and revives the preview editor, and
    `announce_retry` updates the surface when `streamer.carries_turn_notices` is True. When
    `streamer.carries_turn_notices` is True, attempts run inside
    `streamer.surface.delivery_budget_seconds()` so `/ask` expiry leaves margin to deliver
    `WINDOW_EXPIRED_NOTICE`; `finalize_reply` runs outside that window because inline media
    generation manages its own budget.

    Args:
        streamer (ResponseStreamer): Streamer rendering this reply in place across attempts.
        open_stream (AnswerStreamFactory): Factory that issues the request and returns its event
            stream.
        message_id (int): Triggering Discord message id for log correlation.

    Raises:
        TimeoutError: The `/ask` delivery window expired, carrying `WINDOW_EXPIRED_NOTICE`.
        Exception: The last attempt's unwrapped provider failure after all attempts are spent.
    """

    async def _before_retry(retry_state: RetryCallState) -> None:
        failure = retry_state.outcome.exception() if retry_state.outcome else None
        streamer.reset_for_retry()
        if streamer.carries_turn_notices:
            await streamer.announce_retry()
        logfire.warn(
            "gen_reply answer stream retry",
            message_id=message_id,
            attempt=retry_state.attempt_number,
            error_type=type(failure).__name__ if failure else None,
            status_code=llm_status_code(exc=failure) if failure else None,
            _exc_info=failure,
        )

    async def _attempt() -> None:
        await streamer.consume(responses=await open_stream())

    retrying = AsyncRetrying(
        retry=retry_if_exception(is_retryable_llm_error),
        wait=wait_fixed(wait=ANSWER_RETRY_INTERVAL_SECONDS)
        + wait_random(min=0, max=ANSWER_RETRY_JITTER_SECONDS),
        stop=stop_after_attempt(ANSWER_STREAM_MAX_ATTEMPTS),
        before_sleep=_before_retry,
        reraise=True,
    )
    window = asyncio.timeout(
        delay=streamer.surface.delivery_budget_seconds() if streamer.carries_turn_notices else None
    )
    try:
        async with window:
            await retrying(_attempt)
    except Exception as exc:
        # The streamer stays published on the way out: the caller's error path is what lands
        # the failure on whatever this reply is still showing.
        if streamer.carries_turn_notices:
            await streamer.withdraw_retry_notice()
        if window.expired():
            raise TimeoutError(WINDOW_EXPIRED_NOTICE) from exc
        raise
    return await streamer.finalize_reply()
