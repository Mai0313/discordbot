"""Streams a Responses API reply onto a Discord message."""

import re
import time
from typing import Literal
import asyncio
import contextlib
import unicodedata
from urllib.parse import urlsplit
from collections.abc import Callable, Awaitable, AsyncIterator

import logfire
from nextcord import File, Embed, Message, NotFound, Forbidden, HTTPException, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt
from nextcord.ui import View
from tenacity.wait import wait_fixed, wait_random
from nextcord.utils import escape_mentions
from openai.types.responses import Response, ResponseOutputItem, ResponseStreamEvent

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
from discordbot.cogs.gen_reply.feedback_views import FEEDBACK_POLICY_NOTE

# Unwrap backticked Discord mentions (`<@id>`, `<@&id>`, `<#id>`) that Gemini sometimes emits.
CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")

# Fixed retry delay plus random jitter to outlast 5xx bursts without lockstep retries.
ANSWER_RETRY_INTERVAL_SECONDS = 5.0
ANSWER_RETRY_JITTER_SECONDS = 1.0

# Shown before background memory review finishes, then replaced in place.
MEMORY_PENDING_NOTE = f"-# {MEMORY_WRITE_EMOJI} 正在整理記憶⋯"

# Appended when surface.answer_capacity caps an /ask reply.
TRUNCATED_NOTICE = "\n-# 回覆太長，這裡放不下後面的內容了"

_CODE_FENCE = "```"
# Bounded language tag carried onto reopened code blocks so splitting always makes progress.
_FENCE_LANGUAGE_RE = re.compile(pattern=r"[A-Za-z0-9_+\-.#]{0,20}")
# Combining, format, and emoji categories not to split beside when hard-cutting.
_CLINGING_CATEGORIES = frozenset({"Mn", "Mc", "Me", "Cf", "Sk", "So"})

# Window caps for the live thinking preview; char budget bounds wrapped paragraphs.
REASONING_PREVIEW_MAX_LINES = 4
REASONING_PREVIEW_MAX_CHARS = 320


def _citation_domain(url: str, title: str) -> str | None:
    """Returns the cited domain from `url`, falling back to `title` for Vertex redirects."""
    host = urlsplit(url=url).hostname
    if host != "vertexaisearch.cloud.google.com":
        return host
    if re.fullmatch(pattern=r"(?:[a-z0-9-]+\.)+[a-z]{2,}", string=title, flags=re.IGNORECASE):
        return title.lower()
    return None


def _url_citations(output: list[ResponseOutputItem]) -> list[tuple[str, str]]:
    """The `(url, title)` of every `url_citation` annotation across message output text parts."""
    return [
        (annotation.url, annotation.title)
        for item in output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
        for annotation in part.annotations
        if annotation.type == "url_citation"
    ]


class AnswerToolUse(BaseModel):
    """What the answer's server-side tools did, as far as its backend reports it.

    None on a field means the backend does not report it, never zero: a zero there would read as
    an answer that did not search or cite.
    """

    web_search_queries: int | None = Field(
        default=None, description="Distinct Google Search queries the answer ran."
    )
    url_fetches: int | None = Field(
        default=None,
        description="Distinct URLs URL context retrieved. The Responses bridge never reports it.",
    )
    url_citations: int | None = Field(
        default=None, description="`url_citation` annotations on the answer text."
    )
    cited_domains: list[str] | None = Field(
        default=None,
        description="Distinct domains those citations point at, in first-cited order.",
        examples=[["wikipedia.org", "nownews.com"]],
    )

    def record_citations(self, citations: list[tuple[str, str]]) -> None:
        """Records the answer's `(url, title)` citations as a count and distinct domains.

        Args:
            citations (list[tuple[str, str]]): Citation `(url, title)` pairs.
        """
        self.url_citations = len(citations)
        domains = (_citation_domain(url=url, title=title) for url, title in citations)
        self.cited_domains = list(dict.fromkeys(domain for domain in domains if domain))

    def record_response(self, response: Response) -> None:
        """Records citations and search-query counts reported on a completed Responses API answer.

        Args:
            response (Response): Completed Responses API response.
        """
        self.record_citations(citations=_url_citations(output=response.output))
        if response.usage:
            details = response.usage.input_tokens_details
            extra = (details.model_extra if details is not None else None) or {}
            self.web_search_queries = extra.get("web_search_requests", 0)


def _in_code_block(text: str, cut: int) -> bool:
    """Whether `cut` falls inside a code block of `text`, pairing fences the way Discord does."""
    return text.count(_CODE_FENCE, 0, cut) % 2 == 1 and text.find(_CODE_FENCE, cut) != -1


def _last_break(text: str, end: int, floor: int, outside_code: bool) -> int:
    """Returns the cleanest break index in `text[:end]` at or after `floor`."""
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
    """Splits `text` within `[earliest, budget]`, closing and reopening overrun code fences."""
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
    # Reserve room for the closing fence here and reopening fence on the remainder.
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


class IncompleteAnswerError(Exception):
    """An answer stream that ended without completing; its message is user-facing."""

    def __init__(self, reason: str) -> None:
        """Creates an incomplete-answer error with a user-facing notice for `reason`.

        Args:
            reason (str): Backend reason (`content_filter`, `max_output_tokens`, Interactions
                status, `incomplete`, or `no_terminal_event`).
        """
        if reason == "content_filter":
            notice = "The provider's content filter stopped this answer before it finished."
        elif reason == "max_output_tokens":
            notice = "This answer used up its whole output budget before it finished."
        else:
            notice = f"The answer stream ended before the answer finished ({reason})."
        super().__init__(notice)
        self.reason = reason
        # Retrying max_output_tokens would hit the same budget again.
        self.retryable = reason != "max_output_tokens"


class ResponseStreamer(BaseModel):
    """Renders one streaming Responses API reply onto a Discord message.

    `consume` reads one attempt while a background task edits Discord on a fixed cadence;
    `finalize_reply` writes the usage footer, optional memory credits, and inline generated media.
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
    last_reply: SkipValidation[Message | None] = Field(
        default=None,
        description="Final Discord message written for this reply (last chunk when chunked).",
    )
    feedback_view: SkipValidation[View | None] = Field(
        default=None,
        description="Optional persistent feedback view attached to the final reply message.",
    )
    on_final_message: Callable[["ResponseStreamer"], Awaitable[None]] | None = Field(
        default=None,
        description=(
            "Optional async hook called immediately after the final message lands (with feedback "
            "buttons attached) and before inline media generation starts."
        ),
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
    model_display_name: str = Field(
        default="",
        description=(
            "Optional display label shown in the Discord usage footer in place of `model_name` "
            "(also hiding `model_effort`); pricing, logs, and `reply_turn` still read the real "
            "`model_name` and `model_effort`."
        ),
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
    tool_use: AnswerToolUse = Field(
        default_factory=AnswerToolUse,
        description=(
            "What the answer's search and URL tools did, filled from the completed response on "
            "the Responses backend and by the stream adapter on the Interactions one."
        ),
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
    # Appended footer, kept so media and memory edits splice above it for USAGE_FOOTER_RE.
    _usage_footer: str = PrivateAttr(default="")
    # Active memory footnote on reply, replaced in place and stripped from returned text.
    _memory_note: str = PrivateAttr(default="")
    # Active dropped-media hint line, stripped from returned text like _memory_note.
    _hint_line: str = PrivateAttr(default="")
    # True if reply was deleted mid-stream, suppressing dropped-media hints.
    _reply_deleted: bool = PrivateAttr(default=False)
    # Suppresses repeated preview-edit warns across ticks and retries.
    _preview_error_logged: bool = PrivateAttr(default=False)
    # Ensures first-reasoning latency is logged once across retries.
    _reasoning_logged: bool = PrivateAttr(default=False)
    _answer_seconds: float = PrivateAttr(default=0.0)

    @staticmethod
    def _split_reply_for_discord(
        content: str, footer: str, max_messages: int | None = None
    ) -> tuple[str, list[str]]:
        """Splits `content` and `footer` into parent and follow-up chunks within `max_messages`."""
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
                # Last allowed message: close an overrun code block instead of breaking before it.
                head, _ = _cut_cleanly(text=remaining, budget=budget, earliest=budget // 2)
                remaining = f"{head}{TRUNCATED_NOTICE}"
                break
            if not messages and len(remaining) <= DISCORD_MESSAGE_LIMIT:
                # An answer that fits one message is not split for its footer, which follows alone.
                messages.append(remaining)
                remaining = ""
                break
            # Keep text ahead of the footer on follow-ups so Discord does not strip
            # USAGE_FOOTER_RE's leading blank line.
            budget = (
                DISCORD_MESSAGE_LIMIT
                if len(remaining) > DISCORD_MESSAGE_LIMIT
                else DISCORD_MESSAGE_LIMIT - len(footer)
            )
            # Earliest cut that lets remaining text + footer fit in the remaining message slots.
            earliest = 0
            if left is not None:
                needed = len(remaining) + len(footer) - (left - 1) * DISCORD_MESSAGE_LIMIT
                earliest = needed if needed <= budget else 0
            head, remaining = _cut_cleanly(text=remaining, budget=budget, earliest=earliest)
            messages.append(head)
        messages.append(f"{remaining}{footer}")
        return messages[0], messages[1:]

    def _render_preview(self) -> str:
        """Returns the preview text: scrubbed content once started, else a reasoning tail."""
        if self.content_started:
            return scrub_markers_for_preview(text=self.stored_content)[:DISCORD_MESSAGE_LIMIT]
        if not self.reasoning_content:
            return ""
        # Escape mentions so transient thinking text never pings.
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

    async def _reply_or_send(self, content: str, view: View | None = None) -> Message:
        """Replies with `content`, falling back to unparented send if the source was deleted."""
        try:
            return await self.surface.send(content=content, view=view)
        except HTTPException as exc:
            if not is_reply_target_gone(error=exc):
                raise
            logfire.info(
                "Source message deleted before reply; sending unparented",
                message_id=self.message.id,
            )
            return await self.surface.send_unparented(content=content, view=view)

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
        """Edits the reply on a fixed cadence until `_editor_stop` is set.

        Stopped via event rather than cancellation so an in-flight first reply finishes first.
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
                    # Reply deleted mid-stream; handled again in _write_final_message.
                    logfire.info(
                        "Reply deleted while streaming; stopping preview edits",
                        message_id=self.message.id,
                    )
                    return
                except Forbidden:
                    logfire.warn(
                        "Channel refused a preview write; stopping preview edits",
                        channel_id=self.message.channel.id,
                        message_id=self.message.id,
                    )
                    return
                except Exception as exc:
                    # Broad: live preview is best-effort; log once since displayed_content stays
                    # unchanged across ticks.
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
            # Broad: editor failure must not sink the finished reply; cancellation propagates.
            logfire.warn(
                "Preview editor task crashed; the reply still finalizes without live preview",
                message_id=self.message.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        self._editor_task = None

    async def _write_final_message(self, content: str, footer: str) -> None:
        """Writes the final reply and overflow chunks via `surface`; no-ops if deleted."""
        parent_content, follow_up_chunks = self._split_reply_for_discord(
            content=content,
            footer=footer,
            max_messages=self.surface.answer_capacity(has_landed_reply=self.reply is not None),
        )
        parent_view = self.feedback_view if not follow_up_chunks else None
        # Record reply here when no preview ran so later media edits target it.
        if self.reply is None:
            self.reply = await self._reply_or_send(content=parent_content, view=parent_view)
        else:
            try:
                if parent_view is not None:
                    await self.reply.edit(content=parent_content, view=parent_view)
                else:
                    await self.reply.edit(content=parent_content)
            except NotFound:
                logfire.info(
                    "Reply deleted while streaming; finishing without the final edit",
                    message_id=self.message.id,
                    reply_id=self.reply.id,
                )
                self.reply = None
                self.last_reply = None
                self._reply_deleted = True
                return
        previous = self.reply
        for index, chunk in enumerate(follow_up_chunks):
            chunk_view = self.feedback_view if index == len(follow_up_chunks) - 1 else None
            previous = await self.surface.follow_up(
                previous=previous, content=chunk, view=chunk_view
            )
        self.last_reply = previous

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

        Does not create a reply when none exists, and suppresses Discord HTTP errors.
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
        """Returns whether `embed` was landed under the partial answer on the in-progress `reply`.

        Preserves any partial answer text (overflowing via follow-ups with `embed` on the last),
        clears a reasoning-only preview with `content=""`, and returns False when no `reply` exists
        or Discord rejects the edit.

        Args:
            embed (Embed): Failure embed to attach to the reply.
        """
        if self.reply is None:
            return False
        # Clear reasoning preview with "" (not None): nextcord multipart edit omits content=None.
        content, chunks = self._split_reply_for_discord(
            content=scrub_markers_for_preview(text=self.stored_content)
            if self.content_started
            else "",
            footer="",
            max_messages=self.surface.answer_capacity(has_landed_reply=True),
        )
        try:
            target = self.reply
            if chunks:
                await target.edit(content=content)
                for chunk in chunks:
                    target = await self.surface.follow_up(previous=target, content=chunk)
                content = chunks[-1]
            await target.edit(
                content=content,
                embed=embed,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=target),
            )
        except NotFound:
            # Half-written reply was deleted; caller posts a fresh failure message.
            logfire.info(
                "The streamed reply is gone; posting the failure fresh",
                message_id=self.message.id,
                reply_id=self.reply.id,
            )
            return False
        except HTTPException as exc:
            logfire.warn(
                "Could not land the failure on the streamed reply; posting it fresh",
                message_id=self.message.id,
                reply_id=self.reply.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return False
        return True

    async def _consume(self, responses: AsyncIterator[ResponseStreamEvent]) -> str | None:
        """Consumes `responses` into state, returning the incomplete reason or None."""
        incomplete: str | None = "no_terminal_event"
        # Match response.type, not isinstance: adapt_interactions_stream yields SimpleNamespaces.
        async for response in responses:
            if response.type == "response.created":
                # Capture model on created too so incomplete streams still report a model name.
                self.model_name = response.response.model
            elif response.type == "response.completed":
                incomplete = None
                self.model_name = response.response.model
                if response.response.usage:
                    self.input_tokens += response.response.usage.input_tokens
                    self.output_tokens += response.response.usage.output_tokens
                # output is None on the Interactions path, whose adapter populates tool_use.
                if response.response.output is not None:
                    self.tool_use.record_response(response=response.response)
            elif response.type == "response.incomplete":
                self.model_name = response.response.model
                # Billed like any other attempt, so count it toward the final footer.
                if response.response.usage:
                    self.input_tokens += response.response.usage.input_tokens
                    self.output_tokens += response.response.usage.output_tokens
                details = response.response.incomplete_details
                incomplete = (
                    details.reason if details is not None and details.reason else "incomplete"
                )
            elif response.type == "response.reasoning_summary_text.delta":
                self._on_reasoning_delta(delta=response.delta)
            elif response.type == "response.output_text.delta":
                self._on_content_delta(delta=response.delta)
        return incomplete

    async def finalize_reply(self) -> str:
        """Returns the finalized reply text after extracting inline markers, writing the footer, and attaching media."""
        # Price using the stream's reported model so LiteLLM fallbacks reflect what answered;
        # `model_display_name` masks the visible model and effort label (cost stays visible by
        # design).
        if self.model_display_name:
            label = self.model_display_name
        elif self.model_effort:
            label = f"{self.model_name} ({self.model_effort})"
        else:
            label = self.model_name
        usage_line, cost = render_usage_footer(
            model_name=self.model_name,
            label=label,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

        self.stored_content = CODED_MENTION_RE.sub(r"\1", self.stored_content)
        self.markers = extract_inline_markers(text=self.stored_content)
        self.stored_content = self.markers.cleaned_text
        self.voice_text = (
            speechify_discord_markup(
                text=self.markers.voice_text, resolve_name=self._resolve_mention_name
            )
            if self.markers.voice_requested
            else ""
        )
        # Credit recalled memory owners; past two names or any unnamed, collapse to "等 N 人".
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
        policy_line = f"\n{FEEDBACK_POLICY_NOTE}" if self.feedback_view is not None else ""
        # `utils/llm_transcript.py::USAGE_FOOTER_RE` anchors on the blank line opening the footer.
        usage_footer = f"\n\n{usage_line}{memory_line}{policy_line}"

        reply_chars = len(self.stored_content)
        chunked = reply_chars + len(usage_footer) > DISCORD_MESSAGE_LIMIT
        if self._wants_pending_memory_note(footer_chars=len(usage_footer)):
            self._memory_note = MEMORY_PENDING_NOTE
            self.stored_content = f"{self.stored_content}\n{self._memory_note}"
        await self._write_final_message(content=self.stored_content, footer=usage_footer)
        self.stored_content += usage_footer
        self._usage_footer = usage_footer
        # Persist feedback turn rows before slow inline media / research runs so live buttons
        # already have a backing row as soon as the message appears in Discord.
        if self.on_final_message is not None and self.last_reply is not None:
            await self.on_final_message(self)
        # Release failure-landing target before media attach: an error edit with the spacer would
        # strip attachments.
        if self.carries_turn_notices:
            current_answer_streamer.set(None)

        await self._attach_generated_media()
        await self._write_hint_line()
        domains = self.tool_use.cited_domains
        shown_domains = 10
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
            web_search_queries=self.tool_use.web_search_queries,
            url_fetches=self.tool_use.url_fetches,
            url_citations=self.tool_use.url_citations,
            # Joined rather than a list, which the console exporter prints one element per line.
            cited_domains=None
            if domains is None
            else ",".join(
                domains[:shown_domains]
                + ([f"+{len(domains) - shown_domains}"] if len(domains) > shown_domains else [])
            ),
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

    def final_answer_text(self) -> str:
        """Returns the finalized reply text with memory notes and dropped-media hints stripped."""
        return self._without_added_lines(text=self.stored_content)

    def _without_added_lines(self, text: str) -> str:
        """Returns `text` with `_memory_note` and `_hint_line` stripped."""
        stripped = self._without_memory_note(text=text)
        if not self._hint_line:
            return stripped
        return stripped.replace(f"\n{self._hint_line}", "", 1)

    def _wants_pending_memory_note(self, footer_chars: int) -> bool:
        """Returns whether `reply` should show `MEMORY_PENDING_NOTE` within Discord's cap."""
        if not self.carries_turn_notices:
            return False
        if not (self.markers.memory_notes or self.markers.forget_notes):
            return False
        spliced = len(self.stored_content) + 1 + len(MEMORY_PENDING_NOTE) + footer_chars
        return spliced <= DISCORD_MESSAGE_LIMIT

    async def set_memory_note(self, line: str) -> None:
        """Replaces the memory status line on `reply` above `_usage_footer`.

        No-ops when `reply` is absent or adding a first note exceeds `DISCORD_MESSAGE_LIMIT`;
        withdraws an existing pending note if its replacement would overflow.

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
            # Keep prior _memory_note recorded on failed edit so a later edit still replaces it.
            return
        self._memory_note = line

    async def _edit_reply_text(self, reply: Message, content: str, failure: str) -> bool:
        """Edits `reply` to `content` best-effort, updating `stored_content` if it lands."""
        try:
            await reply.edit(content=content, allowed_mentions=AllowedMentions.none())
        except Exception as exc:
            # Broad: reply may be deleted; footnote edits never surface failures.
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
        """Writes queued `/ask` dropped-media hints onto `reply` above the footer."""
        hints = self.surface.take_hints()
        if not hints or self.reply is None or not self._usage_footer:
            return
        body = self._without_memory_note(text=self.stored_content.removesuffix(self._usage_footer))
        line = f"-# {''.join(hints)}"
        note = f"\n{self._memory_note}" if self._memory_note else ""
        updated = f"{body}\n{line}{note}{self._usage_footer}"
        if len(updated) > DISCORD_MESSAGE_LIMIT:
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
        """Synthesizes `<generate-voice>` to a WAV `MediaItem`, hinting on failure."""
        if not self.markers.voice_requested:
            logfire.debug("Voice not requested by the answer model", message_id=self.message.id)
            return None
        if self.voice_generator is None:
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
            return None
        if clip.outcome == "timeout":
            await self.surface.hint(emoji=TIMEOUT_HINT_EMOJI)
            return None
        if clip.audio is None:
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
            return None
        return MediaItem(source=clip.audio, filename=VOICE_REPLY_FILENAME)

    async def _load_marker_source_images(self) -> list[LoadedMedia]:
        """Loads source images from the current and replied-to messages best-effort."""
        if self.input_builder is None:
            return []
        try:
            return await self.input_builder.get_turn_image_sources(
                message=self.message, replied_to=replied_to_message(message=self.message)
            )
        except Exception as exc:  # Broad: best-effort source load; degrade to fresh generation.
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
        """Renders `<generate-image>` prompts concurrently to PNG `MediaItem`s."""
        if not self.markers.image_prompts:
            return []
        if self.image_generator is None:
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
                dropped = True
                continue
            # Use distinct names for multiple images; Discord collides on duplicate filenames.
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
        """Renders `<generate-music>` to an audio `MediaItem`, hinting on failure."""
        if self.markers.music_prompt is None:
            logfire.debug("Music not requested by the answer model", message_id=self.message.id)
            return None
        if self.music_generator is None:
            logfire.info(
                "Inline music requested but disabled for this turn; replying without music",
                message_id=self.message.id,
            )
            return None
        await self.surface.mark(emoji=MUSIC_EMOJI)
        logfire.info("Generating inline music reply", message_id=self.message.id)
        clip = await self.music_generator.generate(user_prompt=self.markers.music_prompt)
        if clip is None:
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
            return None
        return MediaItem(source=clip.audio, filename=music_filename(mime_type=clip.mime_type))

    async def _build_video_candidate(
        self, source_images_task: asyncio.Task[list[LoadedMedia]] | None
    ) -> MediaItem | None:
        """Renders `<generate-video>` to an MP4 `MediaItem`, hinting on failure."""
        if self.markers.video_prompt is None:
            logfire.debug("Video not requested by the answer model", message_id=self.message.id)
            return None
        if self.video_generator is None:
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
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
            return None
        return MediaItem(source=video_bytes, filename=INLINE_VIDEO_FILENAME)

    async def _attach_generated_media(self) -> None:
        """Builds inline media within the surface budget and attaches or hosts them in one edit."""
        if self._reply_deleted:
            # User deleted the reply; skip dropped-media hints.
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
        # Share one source-image fetch across inline image and video builders.
        source_images_task = (
            asyncio.ensure_future(self._load_marker_source_images())
            if (self.image_generator is not None and self.markers.image_prompts)
            or (self.video_generator is not None and self.markers.video_prompt is not None)
            else None
        )
        voice_task = asyncio.ensure_future(self._build_voice_candidate())
        music_task = asyncio.ensure_future(self._build_music_candidate())
        video_task = asyncio.ensure_future(
            self._build_video_candidate(source_images_task=source_images_task)
        )
        image_task = asyncio.ensure_future(
            self._build_image_candidates(source_images_task=source_images_task)
        )
        # Bound builders to /ask's budget so timed-out media still leaves room for a ⏱️ hint;
        # cancel pending in finally because asyncio.wait does not cancel on outer cancellation.
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
        """Edits `files` and `hosted_urls` onto `reply` (or follow-up), returning success."""
        if not files and not hosted_urls:
            return True
        content: str | None = None
        follow_up: str | None = None
        if hosted_urls:
            link_line = "\n-# 媒體過大，改用連結\n" + "\n".join(hosted_urls)
            if len(self.stored_content) + len(link_line) <= DISCORD_MESSAGE_LIMIT:
                # Splice before _usage_footer so USAGE_FOOTER_RE still strips it from history.
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
            # Broad: best-effort media attach must not raise into the reply pipeline.
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
                # Broad: best-effort hosted-link follow-up must not raise into the reply pipeline.
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

        Raises:
            IncompleteAnswerError: The stream ended without `response.completed`.
        """
        if self.carries_turn_notices:
            current_answer_streamer.set(self)
        try:
            incomplete = await self._consume(responses=responses)
        finally:
            # Stamp before finalize_reply so inline media renders do not inflate answer_seconds.
            self._answer_seconds = time.monotonic() - self.created_at
            await self._stop_editor()
        if incomplete is not None:
            logfire.warn(
                "gen_reply answer stream incomplete",
                message_id=self.message.id,
                model=self.model_name,
                backend=self.backend,
                reason=incomplete,
                attempt=self.attempts,
                reply_chars=len(self.stored_content),
                reasoning_chars=len(self.reasoning_content),
            )
            raise IncompleteAnswerError(reason=incomplete)


# Async factory so each retry re-issues the request across Responses and Interactions backends.
type AnswerStreamFactory = Callable[[], Awaitable[AsyncIterator[ResponseStreamEvent]]]


async def stream_answer_with_retry(
    streamer: ResponseStreamer, open_stream: AnswerStreamFactory, message_id: int
) -> str:
    """Returns the finalized reply text after streaming `open_stream` into `streamer` with retries.

    Retries on retryable LLM errors and on incomplete streams other than `max_output_tokens`, up
    to `ANSWER_STREAM_MAX_ATTEMPTS`. Calls `reset_for_retry` between attempts and `announce_retry`
    when `streamer.carries_turn_notices` is True. Runs attempts inside
    `streamer.surface.delivery_budget_seconds()` when `carries_turn_notices` is True;
    `finalize_reply` runs outside that window.

    Args:
        streamer (ResponseStreamer): Streamer rendering this reply in place across attempts.
        open_stream (AnswerStreamFactory): Factory that issues the request and returns its event
            stream.
        message_id (int): Triggering Discord message id for log correlation.

    Raises:
        TimeoutError: The `/ask` delivery window expired, carrying `WINDOW_EXPIRED_NOTICE`.
        IncompleteAnswerError: The last attempt ended without completing, or ran out of output
            budget.
        Exception: The last attempt's unwrapped provider failure after all attempts are spent.
    """

    def _worth_another_attempt(failure: BaseException) -> bool:
        if isinstance(failure, IncompleteAnswerError):
            return failure.retryable
        return is_retryable_llm_error(exc=failure)

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
        retry=retry_if_exception(_worth_another_attempt),
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
        # Keep streamer published so the caller's error path can land the failure on the reply.
        if streamer.carries_turn_notices:
            await streamer.withdraw_retry_notice()
        if window.expired():
            raise TimeoutError(WINDOW_EXPIRED_NOTICE) from exc
        raise
    return await streamer.finalize_reply()
