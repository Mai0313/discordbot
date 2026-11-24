"""Streams a Responses API reply onto a Discord message."""

import re
from http import HTTPStatus
import time
from typing import Literal, NoReturn, TypedDict
import asyncio
from collections.abc import Callable, Awaitable, AsyncIterator

import logfire
from nextcord import Embed, Message, NotFound, HTTPException, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.ui import View
from nextcord.utils import escape_mentions
from openai.types.responses import Response, ResponseUsage, ResponseStreamEvent

from discordbot.typings.memory import MemoryCredits
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT, embed_spacer_payload
from discordbot.utils.discord_errors import report_discord_failure
from discordbot.utils.llm_transcript import render_usage_footer
from discordbot.utils.media_delivery import MediaDeliveryPlanner
from discordbot.utils.preview_editor import CadencePreviewEditor
from discordbot.cogs.gen_reply.markers import (
    InlineMarkers,
    extract_inline_markers,
    scrub_markers_for_preview,
)
from discordbot.cogs.gen_reply.surface import WINDOW_EXPIRED_NOTICE, TurnSurface
from discordbot.cogs.gen_reply.chunking import split_reply_for_discord
from discordbot.cogs.gen_reply.tool_use import AnswerToolUse
from discordbot.cogs.gen_reply.generation import (
    ImageGenerator,
    MusicGenerator,
    VideoGenerator,
    VoiceGenerator,
    speechify_discord_markup,
)
from discordbot.cogs.gen_reply.turn_state import current_answer_streamer
from discordbot.cogs.gen_reply.inline_media import InlineMediaAttacher
from discordbot.cogs.gen_reply.status_marks import (
    ANSWER_EMOJI,
    MEMORY_READ_EMOJI,
    MEMORY_WRITE_EMOJI,
)
from discordbot.cogs.gen_reply.feedback_views import FEEDBACK_POLICY_NOTE

# Unwrap backticked Discord mentions (`<@id>`, `<@&id>`, `<#id>`) that Gemini sometimes emits.
CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")

# Shown before background memory review finishes, then replaced in place.
MEMORY_PENDING_NOTE = f"-# {MEMORY_WRITE_EMOJI} 正在整理記憶⋯"

# Window caps for the live thinking preview; char budget bounds wrapped paragraphs.
REASONING_PREVIEW_MAX_LINES = 4
REASONING_PREVIEW_MAX_CHARS = 320


class _EditKwargs(TypedDict, total=False):
    """Typed optional kwargs forwarded to `Message.edit` without passing None sentinels."""

    content: str
    view: View


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


class AnswerStreamError(Exception):
    """A stream-level error emitted via `response.failed` or `error` SSE events."""

    def __init__(self, message: str, code: str | None = None) -> None:
        """Creates an answer stream error carrying the upstream message and mapped status.

        Args:
            message (str): Upstream error message from the failed response or error event.
            code (str | None): OpenAI Responses error code or decimal HTTP status string.
        """
        super().__init__(message)
        status_codes = {
            "server_error": HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "rate_limit_exceeded": HTTPStatus.TOO_MANY_REQUESTS.value,
            "request_timeout": HTTPStatus.REQUEST_TIMEOUT.value,
            "insufficient_quota": HTTPStatus.PAYMENT_REQUIRED.value,
            "authentication_error": HTTPStatus.UNAUTHORIZED.value,
            "permission_error": HTTPStatus.FORBIDDEN.value,
            "not_found_error": HTTPStatus.NOT_FOUND.value,
            "invalid_request_error": HTTPStatus.BAD_REQUEST.value,
            "invalid_prompt": HTTPStatus.BAD_REQUEST.value,
        }
        if code is None:
            status_code: int | None = None
        elif code in status_codes:
            status_code = status_codes[code]
        elif code.isascii() and code.isdecimal():
            status_code = int(code)
        else:
            status_code = None
        self.code = code
        self.status_code = status_code


class ResponseStreamer(BaseModel):
    """Renders one streaming Responses API reply onto a Discord message.

    `consume` reads the stream while a background task edits Discord on a fixed cadence;
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
    reply_messages: SkipValidation[list[Message]] = Field(
        default_factory=list,
        description="All Discord messages written for this reply in send order (parent plus follow-ups).",
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
            "Whether `reply` is the turn's own surface, so the turn's `/ask` delivery window, "
            "pending memory note, and error embed belong on it. False for a media persona reply, "
            "which renders onto the DELIVERED image or video -- the deliverable itself must never "
            "be overwritten by an error."
        ),
    )
    displayed_content: str = Field(
        default="", description="The text last written to the Discord reply."
    )
    content_started: bool = Field(
        default=False, description="Whether the stream has seen its first non-newline text delta."
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
    streaming: bool = Field(
        default=True,
        description=(
            "Whether this reply streamed live preview edits; logged on finalization so streaming "
            "and non-streaming turns are distinguishable in telemetry."
        ),
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
    _editor: CadencePreviewEditor | None = PrivateAttr(default=None)
    # Final cleaned reply body before any hosted-link, hint, memory, or footer lines.
    _body_text: str = PrivateAttr(default="")
    # Hosted media URL block spliced above hint/memory/footer when it fits DISCORD_MESSAGE_LIMIT.
    _hosted_block: str = PrivateAttr(default="")
    # Appended footer, kept so media and memory edits splice above it for USAGE_FOOTER_RE.
    _usage_footer: str = PrivateAttr(default="")
    # Active memory footnote on reply, replaced in place and stripped from returned text.
    _memory_note: str = PrivateAttr(default="")
    # Active dropped-media hint line, stripped from returned text like _memory_note.
    _hint_line: str = PrivateAttr(default="")
    # True if reply was deleted mid-stream, suppressing dropped-media hints.
    _reply_deleted: bool = PrivateAttr(default=False)
    _answer_seconds: float = PrivateAttr(default=0.0)

    @property
    def answer_seconds(self) -> float:
        """Returns elapsed seconds from streamer creation through stream completion."""
        return self._answer_seconds

    def populate_answer(
        self,
        text: str,
        reasoning: str = "",
        model_name: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Populates completed non-streaming answer state and stamps elapsed answer latency.

        Args:
            text (str): Completed reply text before marker stripping.
            reasoning (str): Optional reasoning-summary text for telemetry.
            model_name (str): Model name reported by the completed response.
            input_tokens (int): Billed input tokens reported by the response.
            output_tokens (int): Billed output tokens reported by the response.
        """
        if model_name:
            self.model_name = model_name
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        cleaned_reasoning = reasoning.lstrip("\n")
        if cleaned_reasoning:
            self.reasoning_content = cleaned_reasoning
        cleaned_text = text.lstrip("\n")
        if cleaned_text:
            self.content_started = True
            self.stored_content = cleaned_text
        self._answer_seconds = time.monotonic() - self.created_at
        if self.carries_turn_notices:
            current_answer_streamer.set(self)

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

    async def _write_preview_snapshot(self) -> None:
        """Writes the latest preview snapshot to the Discord reply, skipping no-ops."""
        preview = self._render_preview()
        if not preview or preview == self.displayed_content:
            return
        if self.reply is None:
            self.reply = await self.surface.send_or_unparented(
                content=preview,
                unparented_log="Source message deleted before reply; sending unparented",
            )
        else:
            await self.reply.edit(content=preview)
        self.displayed_content = preview

    def _preview_editor(self) -> CadencePreviewEditor:
        """Returns the cadence editor driving live Discord preview edits for this streamer."""
        if self._editor is None:
            self._editor = CadencePreviewEditor(
                write_snapshot=self._write_preview_snapshot,
                interval_seconds=self.preview_interval_seconds,
                message_id=self.message.id,
                channel_id=self.message.channel.id,
                deleted_log="Reply deleted while streaming; stopping preview edits",
                forbidden_log="Channel refused a preview write; stopping preview edits",
                error_log="Preview snapshot edit failed; continuing to stream",
                crash_log=(
                    "Preview editor task crashed; the reply still finalizes without live preview"
                ),
            )
        return self._editor

    async def _write_final_message(self, content: str, footer: str) -> None:
        """Writes the final reply and overflow chunks via `surface`; no-ops if deleted."""
        parent_content, follow_up_chunks = split_reply_for_discord(
            content=content,
            footer=footer,
            max_messages=self.surface.answer_capacity(has_landed_reply=self.reply is not None),
        )
        parent_view = self.feedback_view if not follow_up_chunks else None
        # Record reply here when no preview ran so later media edits target it.
        if self.reply is None:
            self.reply = await self.surface.send_or_unparented(
                content=parent_content,
                view=parent_view,
                unparented_log="Source message deleted before reply; sending unparented",
            )
        else:
            edit_kwargs: _EditKwargs = {"content": parent_content}
            if parent_view is not None:
                edit_kwargs["view"] = parent_view
            try:
                await self.reply.edit(**edit_kwargs)
            except NotFound:
                logfire.info(
                    "Reply deleted while streaming; finishing without the final edit",
                    message_id=self.message.id,
                    reply_id=self.reply.id,
                )
                self.reply = None
                self.last_reply = None
                self.reply_messages = []
                self._reply_deleted = True
                return
        self.reply_messages = [self.reply]
        previous = self.reply
        for index, chunk in enumerate(follow_up_chunks):
            chunk_view = self.feedback_view if index == len(follow_up_chunks) - 1 else None
            previous = await self.surface.follow_up(
                previous=previous, content=chunk, view=chunk_view
            )
            self.reply_messages.append(previous)
        self.last_reply = previous

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
                message_id=self.message.id,
            )
        self.reasoning_content += delta
        self._preview_editor().start()

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
                message_id=self.message.id,
            )
        self.stored_content += delta
        self._preview_editor().start()

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
        content, chunks = split_reply_for_discord(
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
        except HTTPException as exc:
            report_discord_failure(
                error=exc,
                not_found_message="The streamed reply is gone; posting the failure fresh",
                forbidden_message=(
                    "Could not land the failure on the streamed reply; posting it fresh"
                ),
                other_message="Could not land the failure on the streamed reply; posting it fresh",
                message_id=self.message.id,
                reply_id=self.reply.id,
            )
            return False
        return True

    def _record_usage(self, usage: ResponseUsage | None) -> None:
        """Accumulates billed token counts from a terminal response event."""
        if usage is not None:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens

    def _raise_failed_response(self, response: Response) -> NoReturn:
        """Records model and usage from a failed response event and raises `AnswerStreamError`."""
        if response.model:
            self.model_name = response.model
        self._record_usage(usage=response.usage)
        error = response.error
        code = error.code if error is not None else None
        message = (
            error.message
            if error is not None and error.message
            else "The model failed to generate a response."
        )
        raise AnswerStreamError(message=message, code=code)

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
                self._record_usage(usage=response.response.usage)
                # output is None on the Interactions path, whose adapter populates tool_use.
                if response.response.output is not None:
                    self.tool_use.record_response(response=response.response)
            elif response.type == "response.incomplete":
                self.model_name = response.response.model
                self._record_usage(usage=response.response.usage)
                details = response.response.incomplete_details
                incomplete = (
                    details.reason if details is not None and details.reason else "incomplete"
                )
            elif response.type == "response.failed":
                self._raise_failed_response(response=response.response)
            elif response.type == "error":
                raise AnswerStreamError(
                    message=response.message or "The model failed to generate a response.",
                    code=response.code,
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
        self._body_text = self.markers.cleaned_text
        self.stored_content = self._body_text
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

        reply_chars = len(self._body_text)
        chunked = reply_chars + len(usage_footer) > DISCORD_MESSAGE_LIMIT
        self._usage_footer = usage_footer
        if self._wants_pending_memory_note(footer_chars=len(usage_footer)):
            self._memory_note = MEMORY_PENDING_NOTE
        await self._write_final_message(
            content=self._render_reply_content(include_footer=False), footer=usage_footer
        )
        self.stored_content = self._render_reply_content()
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
            streaming=self.streaming,
            effort=self.model_effort,
            answer_seconds=self._answer_seconds,
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
        return self.final_answer_text()

    def _render_reply_content(
        self,
        hosted_block: str | None = None,
        hint_line: str | None = None,
        memory_note: str | None = None,
        include_added_lines: bool = True,
        include_footer: bool = True,
    ) -> str:
        """Renders reply text from `_body_text`, hosted links, hint, memory note, and footer."""
        hosted = self._hosted_block if hosted_block is None else hosted_block
        hint = self._hint_line if hint_line is None else hint_line
        memory = self._memory_note if memory_note is None else memory_note
        text = self._body_text
        if hosted:
            text = f"{text}\n{hosted}"
        if include_added_lines:
            if hint:
                text = f"{text}\n{hint}"
            if memory:
                text = f"{text}\n{memory}"
        if include_footer:
            text = f"{text}{self._usage_footer}"
        return text

    def final_answer_text(self) -> str:
        """Returns the finalized reply text with memory notes and dropped-media hints stripped."""
        return self._render_reply_content(include_added_lines=False)

    def _wants_pending_memory_note(self, footer_chars: int) -> bool:
        """Returns whether `reply` should show `MEMORY_PENDING_NOTE` within Discord's cap."""
        if not self.carries_turn_notices:
            return False
        if not (self.markers.memory_notes or self.markers.forget_notes):
            return False
        spliced = len(self._body_text) + 1 + len(MEMORY_PENDING_NOTE) + footer_chars
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
        updated = self._render_reply_content(memory_note=line)
        if len(updated) > DISCORD_MESSAGE_LIMIT:
            if not self._memory_note:
                return
            line = ""
            updated = self._render_reply_content(memory_note="")
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
            report_discord_failure(
                error=exc,
                not_found_message=failure,
                not_found_level="warn",
                not_found_exc_info=True,
                forbidden_message=failure,
                other_message=failure,
                message_id=self.message.id,
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
        line = f"-# {''.join(hints)}"
        updated = self._render_reply_content(hint_line=line)
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

    async def _attach_generated_media(self) -> None:
        """Builds inline media within the surface budget and attaches or hosts them in one edit."""
        result = await InlineMediaAttacher(
            message=self.message,
            surface=self.surface,
            markers=self.markers,
            media_delivery=self.media_delivery,
            voice_text=self.voice_text,
            voice_generator=self.voice_generator,
            image_generator=self.image_generator,
            input_builder=self.input_builder,
            music_generator=self.music_generator,
            video_generator=self.video_generator,
        ).attach(
            reply=self.reply,
            reply_deleted=self._reply_deleted,
            render_with_hosted=lambda hosted_block: self._render_reply_content(
                hosted_block=hosted_block
            ),
        )
        if result.hosted_block is not None and result.updated_content is not None:
            self._hosted_block = result.hosted_block
            self.stored_content = result.updated_content
        if result.follow_up_message is not None:
            self.reply_messages.append(result.follow_up_message)

    async def consume(self, responses: AsyncIterator[ResponseStreamEvent]) -> None:
        """Consumes `responses` into this streamer and stops the live preview editor when done.

        Args:
            responses (AsyncIterator[ResponseStreamEvent]): Stream of Responses API events.

        Raises:
            AnswerStreamError: The stream emitted a `response.failed` or `error` event.
            IncompleteAnswerError: The stream ended without `response.completed`.
        """
        if self.carries_turn_notices:
            current_answer_streamer.set(self)
        try:
            incomplete = await self._consume(responses=responses)
        finally:
            # Stamp before finalize_reply so inline media renders do not inflate answer_seconds.
            self._answer_seconds = time.monotonic() - self.created_at
            if self._editor is not None:
                await self._editor.stop()
        if incomplete is not None:
            logfire.warn(
                "gen_reply answer stream incomplete",
                message_id=self.message.id,
                model=self.model_name,
                backend=self.backend,
                reason=incomplete,
                reply_chars=len(self.stored_content),
                reasoning_chars=len(self.reasoning_content),
            )
            raise IncompleteAnswerError(reason=incomplete)


# Async factory so opening the stream runs inside the `/ask` delivery window.
type AnswerStreamFactory = Callable[[], Awaitable[AsyncIterator[ResponseStreamEvent]]]


async def stream_answer(streamer: ResponseStreamer, open_stream: AnswerStreamFactory) -> str:
    """Returns the finalized reply text after streaming `open_stream` into `streamer`.

    Runs `open_stream` and `streamer.consume` inside
    `streamer.surface.delivery_budget_seconds()` when `streamer.carries_turn_notices` is True;
    `finalize_reply` runs outside that window.

    Args:
        streamer (ResponseStreamer): Streamer rendering this reply onto Discord.
        open_stream (AnswerStreamFactory): Factory that issues the request and returns its event
            stream.

    Raises:
        TimeoutError: The `/ask` delivery window expired, carrying `WINDOW_EXPIRED_NOTICE`.
        AnswerStreamError: The stream emitted a `response.failed` or `error` event.
        IncompleteAnswerError: The stream ended without completing, or ran out of output budget.
        Exception: Any unwrapped provider failure raised while opening or consuming the stream.
    """
    window = asyncio.timeout(
        delay=streamer.surface.delivery_budget_seconds() if streamer.carries_turn_notices else None
    )
    try:
        async with window:
            await streamer.consume(responses=await open_stream())
    except Exception as exc:
        # Keep streamer published so the caller's error path can land the failure on the reply.
        if window.expired():
            raise TimeoutError(WINDOW_EXPIRED_NOTICE) from exc
        raise
    return await streamer.finalize_reply()
