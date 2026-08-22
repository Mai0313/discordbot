"""Streams a Responses API reply onto a Discord message."""

import re
import time
import asyncio
import contextlib
from contextvars import ContextVar
from collections.abc import Callable, Awaitable, AsyncIterator

import logfire
from nextcord import File, Embed, Message, NotFound, HTTPException, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_fixed, wait_random
from nextcord.utils import escape_mentions
from openai.types.responses import ResponseOutputItem, ResponseStreamEvent

from discordbot.typings.models import strip_key_suffix
from discordbot.utils.reactions import update_reaction
from discordbot.typings.timeouts import ANSWER_STREAM_MAX_ATTEMPTS
from discordbot.utils.llm_errors import llm_status_code, is_retryable_llm_error
from discordbot.utils.model_pricing import get_token_rates
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaHostingConfig,
    MediaHostingService,
    MediaDeliveryPlanner,
    upload_limit_for,
)
from discordbot.cogs.gen_reply.markers import (
    MAX_INLINE_IMAGES,
    extract_inline_markers,
    scrub_markers_for_preview,
)
from discordbot.cogs.gen_reply.generation import (
    VOICE_REPLY_FILENAME,
    VoiceOutcome,
    ImageGenerator,
    MusicGenerator,
    VideoGenerator,
    VoiceGenerator,
    music_filename,
    speechify_discord_markup,
)

# Filename of a single inline-generated image attached onto a QA reply; mirrors the router IMAGE
# route's `generated.png` so the bot's own generated images render the same in history. Multiple
# images need distinct names, so they fall back to `generated_<n>.png` (Discord collides on dupes).
INLINE_IMAGE_FILENAME = "generated.png"

# Filename of the single inline-generated video attached onto a QA reply (one clip per reply, so
# no numbering); MP4 is what the omni renderer returns and what Discord inline-plays.
INLINE_VIDEO_FILENAME = "generated.mp4"

# Gemini occasionally wraps Discord mention syntax in backticks (inline code),
# which stops Discord from rendering the actual mention. Strip those wrappers
# before sending; matches user (<@id>, <@!id>), role (<@&id>) and channel (<#id>) mentions.
CODED_MENTION_RE = re.compile(r"`(<(?:@[!&]?|#)\d+>)`")
DISCORD_MESSAGE_LIMIT = 2000

# Spacing between answer-stream attempts. A cadence rather than a bound, which is why it stays
# here while `ANSWER_STREAM_MAX_ATTEMPTS` lives in `typings/timeouts.py`. Flat rather than
# doubling, and wide enough that the wait is the point: a provider 5xx burst is what this retry
# exists for, and re-opening the stream a second or two after one started reaches the same
# unhealthy deployment, so the attempts are spent rather than spaced. Every attempt costs the
# same, so the worst case for one reply is (`ANSWER_STREAM_MAX_ATTEMPTS` - 1) of these. The jitter
# is a whole `wait_random` added beside the interval rather than a parameter of it, so dropping
# that half removes jitter outright rather than falling back to a library default: it is what
# stops a busy channel's replies retrying in lockstep, and being ADDITIVE and independent of the
# interval is why a test zeroing the interval alone still sleeps.
ANSWER_RETRY_INTERVAL_SECONDS = 5.0
ANSWER_RETRY_JITTER_SECONDS = 1.0

# Shown on both retry surfaces, so the reaction on the source message and the notice on the
# reply read as the same event rather than two unrelated hints.
RETRY_HINT_EMOJI = "🔁"

# The thinking preview is a live glance, not a transcript: keep only the newest few subtext
# lines so a long think never grows into a wall of text above the reply. The char budget is
# the load-bearing half, since one thought line is often a whole paragraph that Discord wraps
# into several rendered lines; the line cap only stops many short lines from stacking up.
REASONING_PREVIEW_MAX_LINES = 4
REASONING_PREVIEW_MAX_CHARS = 320


def _count_url_citations(*, output: list[ResponseOutputItem]) -> int:
    """Counts the grounding citations a completed response carried.

    The Responses bridge reports a grounded answer ONLY as `url_citation` annotations hanging off
    the output text; it carries no `groundingMetadata` anywhere, which is what has repeatedly made
    a grounded reply read as ungrounded. The walk is guarded on both discriminants because an
    output item can also be a reasoning item and a content part can also be a refusal.
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
    backend: str = Field(
        default="responses",
        description="Which answer surface produced this stream, logged so a metric only one of "
        "them reports is not read as an absence on the other.",
        examples=["responses", "interactions"],
    )
    input_tokens: int = Field(default=0, description="Input tokens reported by the stream.")
    output_tokens: int = Field(default=0, description="Output tokens reported by the stream.")
    memory_lookups: list[str] = Field(
        default_factory=list,
        description="Labels of users whose stored memory was injected, for the footer.",
    )
    voice_generator: SkipValidation[VoiceGenerator | None] = Field(
        default=None,
        description="TTS engine for spoken replies; None disables voice for this reply.",
    )
    voice_requested: bool = Field(
        default=False,
        description="Whether the answer model wrapped a segment in <generate-voice> for this reply.",
    )
    voice_text: str = Field(
        default="",
        description="Speechified text of the <generate-voice> segment used as the spoken-clip input.",
    )
    image_generator: SkipValidation[ImageGenerator | None] = Field(
        default=None,
        description="Inline-image renderer; None disables inline <generate-image> for this reply.",
    )
    image_prompts: list[str] = Field(
        default_factory=list,
        description="The <generate-image> descriptions the answer model asked to illustrate, in order.",
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
    music_prompt: str | None = Field(
        default=None,
        description="The <generate-music> description the answer model asked to score, if any.",
    )
    video_generator: SkipValidation[VideoGenerator | None] = Field(
        default=None,
        description="Inline-video renderer; None disables inline <generate-video> for this reply.",
    )
    video_prompt: str | None = Field(
        default=None,
        description="The <generate-video> description the answer model asked to animate, if any.",
    )
    research_brief: str | None = Field(
        default=None,
        description="The <deep-research> brief the answer model asked to launch, if any.",
    )
    media_delivery: MediaDeliveryPlanner = Field(
        default_factory=lambda: MediaDeliveryPlanner(
            media_hosting=MediaHostingService(
                # model_validate: the alias kwarg form is invisible to type
                # checkers without a pydantic plugin (ty), and env merging is
                # irrelevant for an all-disabled config.
                config=MediaHostingConfig.model_validate({"MEDIA_HOSTING_ENABLED": False})
            )
        ),
        description=(
            "Decides attach-vs-host-vs-drop for generated media; defaults to a disabled planner "
            "so a streamer built without one drops oversize media exactly as the host-free path."
        ),
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
    # line BEFORE it (USAGE_FOOTER_RE strips only a footer at end-of-message).
    _usage_footer: str = PrivateAttr(default="")
    # Set when the reply message was deleted while streaming, so the media step knows the
    # difference between "never sent" (a real problem, worth a hint) and "sent then deleted".
    _reply_deleted: bool = PrivateAttr(default=False)
    # Set once the preview editor has reported a failed snapshot write, so the 1s cadence does
    # not repeat the same warn for the rest of the stream. Deliberately survives a retry reset:
    # a Discord edit that failed on one attempt fails on the next for the same reason.
    _preview_error_logged: bool = PrivateAttr(default=False)
    # Set once the first-reasoning-delta latency has been logged. Separate from
    # `reasoning_content`, which a retry clears: the record is per TURN (#568), so a retried
    # answer must not emit a second one carrying the failed attempt's wait as its elapsed.
    _reasoning_logged: bool = PrivateAttr(default=False)
    # How long the answer call itself took, stamped when the stream is consumed.
    _answer_seconds: float = PrivateAttr(default=0.0)
    # Grounding citations the completed response carried, or None when this backend never
    # reported any. The distinction is the point: only the Responses bridge reports grounding,
    # and only as `url_citation` annotations, so a zero written for the Interactions path (or for
    # a stream that ended without `response.completed`) would read as a genuinely ungrounded
    # answer. See the grounding note in CLAUDE.md's Responses API Gotchas.
    _url_citations: int | None = PrivateAttr(default=None)

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
        header = "-# <:message:1517560873000898860> Thinking..."
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

        Deleting the source before the reply lands makes Discord 400 with code 50035
        (unknown message_reference); we log it and send into the same channel instead of
        wasting the whole pipeline. Other HTTP errors still propagate to the caller.
        """
        try:
            return await self.message.reply(content=content)
        except HTTPException as exc:
            if exc.code != 50035 and not isinstance(exc, NotFound):
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
        """Writes the final reply, continuing overflow as follow-up replies in the same channel.

        A reply deleted while it streamed (author delete, moderator purge) makes the final edit
        404 with code 10008. That is a normal end, not a failure: the answer is complete and the
        message it belonged to is gone on purpose, so the handle is dropped and nothing is
        re-sent, since re-sending would resurrect exactly what someone just removed.
        """
        parent_content, follow_up_chunks = self._split_reply_for_discord(
            content=content, footer=footer
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
            previous = await previous.reply(content=chunk)

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
        """Drops what one failed attempt accumulated so the next one starts clean.

        Only the text of the attempt goes. What stays, stays for a reason:

        - `input_tokens` / `output_tokens`, because usage arrives on `response.completed`
          alone, which a failed attempt never reached, and because they are seeded with the
          memory-selection call's usage. The consequence is deliberate but real -- upstream
          billed the failed attempt and the footer cannot see it, so `attempts` rides out on
          `gen_reply reply finalized` to make an under-reported cost visible in the log.
        - `reply`, so the retry edits the message already on screen instead of posting a
          second one beside the first attempt's half-written text.
        - `created_at`, so the footer and the latency logs report the wait the user actually
          had, retries and backoff included.
        - `content_ever_started` and `_reasoning_logged`, which are per-turn rather than
          per-attempt (see their own notes).

        The editor is the one piece that needs work rather than none: `stream`'s finally
        stopped it by SETTING `_editor_stop`, and `_preview_editor` reads that before its
        first tick, so a task started by the retry's first delta would exit immediately and
        the whole retry would stream with no live preview. Clearing it is what keeps the
        stale preview being replaced as the new text arrives rather than sitting frozen until
        the final write.
        """
        self.attempts += 1
        self.stored_content = ""
        self.reasoning_content = ""
        self.content_started = False
        self._editor_stop.clear()

    async def announce_retry(self) -> None:
        """Tells the user the answer stalled and is being tried again.

        Two surfaces because neither covers both cases. The reaction always lands, and is the
        only signal when the stall came before anything was written; it rides the source
        message independently of the pipeline's status chain, the same way a dropped-media
        hint does. The notice takes over the reply's text because that is where someone
        waiting is actually looking, and because what sits there otherwise is the dead
        attempt's half-sentence, which reads as an answer that got cut off rather than as a
        stall. It is written only onto a reply that already exists: creating one here would
        leave a bare "Retrying" message behind on the turns that go on to fail anyway.

        Both writes are best-effort. This is a hint about a failure and must never become one:
        `update_reaction` suppresses its own, and a reply deleted mid-answer (or a rate-limited
        edit) must not cost the retry it is announcing.
        """
        await update_reaction(message=self.message, bot_user=None, emoji=RETRY_HINT_EMOJI)
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
        """Removes a reply left showing nothing but the notice, once the attempts are spent.

        The notice promises another attempt, so with none left it has to go: otherwise the turn
        ends with one message saying work is still in flight sitting beside the error the
        pipeline posts to say it is not. Only a message whose WHOLE content is still the notice
        is removed -- once the last attempt put real text on screen, that text is the better
        residue and the error is what says it is incomplete.
        """
        if self.reply is None or self.displayed_content != self._retry_notice():
            return
        with contextlib.suppress(HTTPException):
            await self.reply.delete()
        self.reply = None

    async def land_failure(self, *, embed: Embed) -> bool:
        """Puts the turn's error onto the reply this answer was streaming into.

        The half-written reply is what the user is already looking at, so the error belongs on
        it: left beside it, that reply carries no usage footer and reads as a complete answer
        that got cut off, with an unrelated-looking embed under it. What the message keeps is
        what it was showing. A partial answer stays, and the embed under it is what says the
        answer is incomplete; a thinking preview is cleared instead, being a live glance at a
        model that has stopped thinking rather than anything the user was reading.

        Clearing it takes an explicit empty string, never None: the embed brings a spacer file,
        which sends the edit as multipart, and `http.py::get_message_payload` drops a None
        content out of that body altogether instead of clearing it (it clears on the JSON path,
        which is exactly what makes the difference easy to miss).

        Returns False when the caller must post a fresh message instead: no reply was ever
        created (every pre-answer failure, and every attempt that died before its first delta),
        the retry notice was withdrawn just above, or Discord turned the edit down.
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

    async def _consume(self, *, responses: AsyncIterator[ResponseStreamEvent]) -> None:
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
                # to an empty model name (and $0.00000000) when a stream ends without a
                # clean `completed` event.
                self.model_name = response.response.model
            elif response.type == "response.completed":
                self.model_name = response.response.model
                # Usage only arrives on `completed`.
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

    async def _finalize_reply(self) -> str:
        """Writes the usage footer and final reply once the stream is consumed."""
        # The stream reports the deployment it answered from, so both the price lookup and
        # the label a human reads take the model out of it. `self.model_name` itself stays
        # whole, because the logfire record below is where a LiteLLM fallback is spotted.
        reported_model = strip_key_suffix(deployment_name=self.model_name)
        input_rate, output_rate = get_token_rates(model_name=reported_model)
        cost = input_rate * self.input_tokens + output_rate * self.output_tokens

        self.stored_content = CODED_MENTION_RE.sub(r"\1", self.stored_content)
        # The answer model may wrap <generate-voice> segments (spoken aloud, kept in the reply) plus
        # <generate-image> / <generate-music> / <generate-video> / <deep-research> blocks (requests,
        # removed from the reply). Extract them all before the footer is built or anything is
        # written. The <generate-voice> segments stay in the visible text; only they (not the whole
        # reply) feed the spoken clip so the audio matches what is read.
        markers = extract_inline_markers(text=self.stored_content)
        self.stored_content = markers.cleaned_text
        self.voice_requested = markers.voice_requested
        self.image_prompts = markers.image_prompts
        self.music_prompt = markers.music_prompt
        self.video_prompt = markers.video_prompt
        # The streamer only surfaces the brief; the cog (not the streamer) launches the research
        # after the single media edit so it never touches the reply's one attachment edit.
        self.research_brief = markers.research_brief
        # The spoken clip must not narrate raw Discord markup (a `<@id>` mention reads as a bare
        # snowflake), so the voice input is normalised while the visible reply keeps its markup.
        self.voice_text = (
            speechify_discord_markup(
                text=markers.voice_text, resolve_name=self._resolve_mention_name
            )
            if self.voice_requested
            else ""
        )
        # Credit looked-up memory owners on a second -# subtext line. Dedupe while
        # preserving lookup order; past two names collapse to "等 N 人" so a busy
        # lookup stays short. USAGE_FOOTER_RE matches this optional second line too.
        memory_line = ""
        if self.memory_lookups:
            names = list(dict.fromkeys(self.memory_lookups))
            if len(names) > 2:
                memory_line = f"\n-# <:tag:1517563887573143595> {', '.join(names[:2])} 等 {len(names)} 人的記憶"
            else:
                memory_line = f"\n-# <:tag:1517563887573143595> {', '.join(names)} 的記憶"
        # Footer format must stay matchable by `utils/llm_transcript.py::USAGE_FOOTER_RE`; the
        # ⬆/⬇ icons are its anchor.
        model_label = (
            f"{reported_model} ({self.model_effort})" if self.model_effort else reported_model
        )
        usage_footer = f"\n\n-# {model_label} · ⬆ {self.input_tokens:,} ⬇ {self.output_tokens:,} · ${cost:.8f}{memory_line}"

        # Final update to ensure complete message is displayed.
        await self._write_final_message(content=self.stored_content, footer=usage_footer)
        reply_chars = len(self.stored_content)
        chunked = reply_chars + len(usage_footer) > DISCORD_MESSAGE_LIMIT
        self.stored_content += usage_footer
        self._usage_footer = usage_footer
        # The answer is on screen in full, so the turn's failure path lets go of it here rather
        # than when the caller returns: the media attach below is unprotected, and an error
        # landing on a finished reply would take its attachments with it, the spacer payload
        # retaining none of them.
        if self.carries_turn_notices:
            current_answer_streamer.set(None)

        await self._attach_generated_media()
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
            voice_requested=self.voice_requested,
            image_count=len(self.image_prompts),
            music_requested=bool(self.music_prompt),
            video_requested=bool(self.video_prompt),
            memory_lookups=len(self.memory_lookups),
            chunked=chunked,
        )
        return self.stored_content

    def _resolve_mention_name(self, *, target_id: int) -> str | None:
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

    async def _hint_media_unavailable(self, *, emoji: str) -> None:
        """Marks the source message so dropped media (voice clip or inline image) is not silent.

        The reply stays without the attachment and the user gets no message; this best-effort
        reaction is the only signal. It rides on the source message as an independent reaction
        (no `previous`), so the pipeline's status chain never removes it. Failures are suppressed
        inside `update_reaction`.
        """
        await update_reaction(message=self.message, bot_user=None, emoji=emoji)

    def _upload_limit(self) -> int:
        """The destination's real upload ceiling, falling back to Discord's 20MB base in a DM.

        A boosted guild's 50/100MB is honored via nextcord's `filesize_limit`; a DM has no guild
        to query, so it falls back to Discord's non-Nitro base of 20MB (shared helper).
        """
        return upload_limit_for(guild=self.message.guild)

    async def _build_voice_candidate(self) -> MediaItem | None:
        """Synthesizes the <generate-voice> segment to a WAV candidate, or None when not delivered.

        Best-effort: a skip (not requested / disabled / empty) is silent, while a
        requested-but-failed clip (timeout / refusal) hints the source message and returns None.
        The upload-limit decision (attach vs host vs drop) is made by `_attach_generated_media`,
        so an oversized clip is no longer dropped here (there is deliberately no spoken-length cap).
        """
        if not self.voice_requested:
            # The expected common path: the answer model wrapped no <generate-voice> segment.
            logfire.debug("Voice not requested by the answer model", message_id=self.message.id)
            return None
        if self.voice_generator is None:
            # Voice is intentionally off this turn (kill-switch), not a failure: no hint.
            logfire.info(
                "Voice requested but disabled for this turn; replying without audio",
                message_id=self.message.id,
            )
            return None
        # Mark the source message with the bot's `voice` app emoji while the clip synthesizes.
        await update_reaction(
            message=self.message, bot_user=None, emoji="<:voice:1517558121092878376>"
        )
        logfire.info(
            "Synthesizing voice reply", message_id=self.message.id, text_chars=len(self.voice_text)
        )
        clip = await self.voice_generator.generate(
            text=self.voice_text, end_user_id=self.message.author.name
        )
        if clip.outcome is VoiceOutcome.EMPTY:
            # Nothing to say (the segment was empty after stripping): no hint.
            return None
        if clip.outcome is VoiceOutcome.TIMEOUT:
            # generate() logged the timeout; cue the user that the clip ran out of time.
            await self._hint_media_unavailable(emoji="⏱️")
            return None
        if clip.audio is None:
            # Any other synthesis failure (most often a policy refusal); generate() logged it.
            await self._hint_media_unavailable(emoji="⚠️")
            return None
        return MediaItem(source=clip.audio, filename=VOICE_REPLY_FILENAME)

    async def _load_marker_source_images(self) -> list[tuple[bytes, str]]:
        """Best-effort source images (current + replied-to message) as `(bytes, mime)` pairs.

        Shared by the inline `<generate-image>` edit (which needs only the bytes) and the inline
        `<generate-video>` reference path (which needs the mime, since omni rejects an image content
        block with an empty mime). Best-effort: no builder or a load failure simply yields no
        sources, so the marker falls back to fresh generation.
        """
        if self.input_builder is None:
            return []
        try:
            if self.message.reference and isinstance(self.message.reference.resolved, Message):
                own_images, ref_images = await asyncio.gather(
                    self.input_builder.get_image_sources_with_mime(message=self.message),
                    self.input_builder.get_image_sources_with_mime(
                        message=self.message.reference.resolved
                    ),
                )
                return own_images + ref_images
            return await self.input_builder.get_image_sources_with_mime(message=self.message)
        except Exception as exc:  # broad: best-effort source load, see docstring
            logfire.warn(
                "Inline image source load failed; generating without source pixels",
                message_id=self.message.id,
                error_type=type(exc).__name__,
                _exc_info=True,
            )
            return []

    async def _build_image_candidates(
        self, *, source_images_task: asyncio.Task[list[tuple[bytes, str]]] | None
    ) -> list[MediaItem]:
        """Renders the <generate-image> requests to PNG candidates, in order; [] when none delivered.

        Best-effort like voice: no request or a disabled generator is silent. The capped prompts
        render concurrently; a generation failure drops that image and a single ⚠️ hint rides on
        the source message. The upload-limit decision (attach vs host) is left to
        `_attach_generated_media`, so a large image is no longer dropped here for size. The uploaded
        source images (for editing) are awaited from the shared `source_images_task` so an inline
        `<generate-image>` and `<generate-video>` in the same reply load them only once; only the raw
        bytes are used here (the edit path needs no mime).
        """
        prompts = self.image_prompts[:MAX_INLINE_IMAGES]
        if not prompts:
            return []
        if self.image_generator is None:
            # Inline image is intentionally off this turn (kill-switch / non-QA route): no hint.
            logfire.info(
                "Inline image requested but disabled for this turn; replying without an image",
                message_id=self.message.id,
            )
            return []
        if len(self.image_prompts) > MAX_INLINE_IMAGES:
            logfire.info(
                "Inline image requests exceed the per-reply cap; dropping the extras",
                message_id=self.message.id,
                requested=len(self.image_prompts),
                cap=MAX_INLINE_IMAGES,
            )
        # Mark the source message with the bot's `image` app emoji while the images render.
        await update_reaction(
            message=self.message, bot_user=None, emoji="<:image:1517559727880667226>"
        )
        logfire.info(
            "Generating inline image reply", message_id=self.message.id, image_count=len(prompts)
        )
        # When the user uploaded image(s), feed them so an inline <generate-image> edits them instead of
        # generating a fresh picture (mirrors the IMAGE route); best-effort, [] when none / failure.
        source_images = await source_images_task if source_images_task is not None else []
        source_bytes = [raw for raw, _ in source_images]
        # Render every requested image concurrently so a slow one never delays the others.
        images = await asyncio.gather(
            *(
                self.image_generator.generate(
                    user_prompt=prompt,
                    end_user_id=self.message.author.name,
                    image_bytes_list=source_bytes or None,
                )
                for prompt in prompts
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
            filename = INLINE_IMAGE_FILENAME if len(prompts) == 1 else f"generated_{index}.png"
            candidates.append(MediaItem(source=image, filename=filename))
        if dropped:
            await self._hint_media_unavailable(emoji="⚠️")
        return candidates

    async def _build_music_candidate(self) -> MediaItem | None:
        """Generates the <generate-music> clip to an audio candidate, or None when not delivered.

        Best-effort like the inline image path: a skip (not requested / disabled) is silent, while
        a requested-but-failed clip hints the source message and returns None. The filename suffix
        follows the returned audio mime type so Discord (or the hosted link) renders a player; the
        upload-limit decision (attach vs host) is left to `_attach_generated_media`.
        """
        if self.music_prompt is None:
            # The expected common path: the answer model wrapped no <generate-music> block.
            logfire.debug("Music not requested by the answer model", message_id=self.message.id)
            return None
        if self.music_generator is None:
            # Music is intentionally off this turn (kill-switch / missing key): no hint.
            logfire.info(
                "Inline music requested but disabled for this turn; replying without music",
                message_id=self.message.id,
            )
            return None
        # Mark the source message while the clip renders (no custom app emoji for music yet).
        await update_reaction(message=self.message, bot_user=None, emoji="🎵")
        logfire.info("Generating inline music reply", message_id=self.message.id)
        clip = await self.music_generator.generate(user_prompt=self.music_prompt)
        if clip is None:
            # generate() logged the failure/timeout; hint once.
            await self._hint_media_unavailable(emoji="⚠️")
            return None
        return MediaItem(source=clip.audio, filename=music_filename(mime_type=clip.mime_type))

    async def _build_video_candidate(
        self, *, source_images_task: asyncio.Task[list[tuple[bytes, str]]] | None
    ) -> MediaItem | None:
        """Generates the <generate-video> clip to an MP4 candidate, or None when not delivered.

        Best-effort like the inline music path: a skip (not requested / disabled) is silent, while
        a requested-but-failed clip hints the source message and returns None. When the user
        uploaded image(s) they are awaited from the shared `source_images_task` and ride as
        `(bytes, mime)` reference pairs so omni infers the task; otherwise it is plain text-to-video.
        The upload-limit decision (attach vs host) is left to `_attach_generated_media`, so a large
        clip is hosted as a URL rather than dropped for size.
        """
        if self.video_prompt is None:
            # The expected common path: the answer model wrapped no <generate-video> block.
            logfire.debug("Video not requested by the answer model", message_id=self.message.id)
            return None
        if self.video_generator is None:
            # Video is intentionally off this turn (kill-switch / missing key): no hint.
            logfire.info(
                "Inline video requested but disabled for this turn; replying without video",
                message_id=self.message.id,
            )
            return None
        # Mark the source message with the bot's `video` app emoji while the clip renders.
        await update_reaction(
            message=self.message, bot_user=None, emoji="<:video:1517560671913377842>"
        )
        logfire.info("Generating inline video reply", message_id=self.message.id)
        source_images = await source_images_task if source_images_task is not None else []
        video_bytes = await self.video_generator.generate(
            user_prompt=self.video_prompt, reference_image_sources=source_images or None
        )
        if video_bytes is None:
            # generate() logged the failure/timeout; hint once.
            await self._hint_media_unavailable(emoji="⚠️")
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
            if (
                self.voice_requested
                or self.image_prompts
                or self.music_prompt
                or self.video_prompt
            ):
                logfire.warn(
                    "Media requested but the reply was never sent; dropping it",
                    message_id=self.message.id,
                )
                await self._hint_media_unavailable(emoji="⚠️")
            return
        reply = self.reply
        # The uploaded source images (for editing an inline <generate-image> / grounding an inline <generate-video>)
        # are loaded at most once even when both markers fire, since load_image_bytes re-fetches
        # per call; both builders await this shared task. None when neither visual marker fired.
        source_images_task = (
            asyncio.ensure_future(self._load_marker_source_images())
            if (self.image_generator is not None and self.image_prompts)
            or (self.video_generator is not None and self.video_prompt is not None)
            else None
        )
        # Build every path concurrently so a slow one never blocks the others: a TTS clip, a music
        # render, or a video render that hangs to its timeout must not delay ready inline images
        # (and vice versa).
        voice_candidate, music_candidate, video_candidate, image_candidates = await asyncio.gather(
            self._build_voice_candidate(),
            self._build_music_candidate(),
            self._build_video_candidate(source_images_task=source_images_task),
            self._build_image_candidates(source_images_task=source_images_task),
        )
        items = [
            item
            for item in (voice_candidate, music_candidate, video_candidate, *image_candidates)
            if item is not None
        ]
        if not items:
            return
        plan = await self.media_delivery.plan(
            items=items, upload_limit=self._upload_limit(), envelope_margin=MEDIA_ENVELOPE_MARGIN
        )
        files = [item.to_file() for item in plan.native]
        await self._finalize_media_edit(reply=reply, files=files, hosted_urls=plan.hosted_urls)
        if plan.dropped_items:
            await self._hint_media_unavailable(emoji="⚠️")
        logfire.info(
            "Generated media attached",
            message_id=self.message.id,
            file_count=len(files),
            hosted_count=len(plan.hosted_urls),
        )

    async def _finalize_media_edit(
        self, *, reply: Message, files: list[File], hosted_urls: list[str]
    ) -> None:
        """Runs the single media edit: native files plus any hosted-URL line on the reply.

        Hosted URLs are appended to the reply content when they fit Discord's 2000-char limit,
        else posted as a follow-up reply so a long answer never overflows. There is nothing to do
        when neither files nor URLs were produced.
        """
        if not files and not hosted_urls:
            return
        content: str | None = None
        follow_up: str | None = None
        if hosted_urls:
            link_line = "\n-# 媒體過大，改用連結\n" + "\n".join(hosted_urls)
            if len(self.stored_content) + len(link_line) <= DISCORD_MESSAGE_LIMIT:
                # Splice the link BEFORE the usage footer (stored_content ends with it): appending
                # after the footer would leave USAGE_FOOTER_RE unable to strip it, so later history
                # rendering would keep the model/token/cost footer inside the bot's answer.
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
            await self._hint_media_unavailable(emoji="⚠️")
            return
        if follow_up is not None:
            try:
                await reply.reply(content=follow_up, allowed_mentions=AllowedMentions.none())
            except Exception as exc:
                # Broad on purpose: a deleted parent or any Discord HTTP error must never raise
                # into the reply pipeline. The follow-up IS the delivery of the hosted clip, so a
                # failure here means the media is gone.
                logfire.warn(
                    "Hosted media link follow-up failed; the media URL was never posted",
                    message_id=self.message.id,
                    hosted_url_count=len(hosted_urls),
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
                await self._hint_media_unavailable(emoji="⚠️")

    async def stream(self, *, responses: AsyncIterator[ResponseStreamEvent]) -> str:
        """Streams the reply onto the message and writes the usage footer; returns the full text."""
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
        return await self._finalize_reply()


# The turn's UNFINISHED answer, so the pipeline's failure path can land its error on the reply
# already on screen. A ContextVar for the same reason `_dispatched_model` is one: the failure
# surfaces in `on_message`, several frames above the streamer that owns the reply handle, and
# nextcord dispatches each `on_message` as its own task, so a context copy can never be read by
# another user's turn. It holds the streamer rather than the message because what the error path
# needs is decided at failure time -- whether a reply exists at all (`withdraw_retry_notice` and a
# mid-stream delete each drop it) and what it is showing. Published by `stream` for a
# `carries_turn_notices` streamer, and taken back the moment its answer is written in full.
current_answer_streamer: ContextVar[ResponseStreamer | None] = ContextVar(
    "gen_reply_answer_streamer", default=None
)


# Opens one answer stream. A factory rather than the stream itself, because the request has to
# be re-issued per attempt, and async because the Responses backend awaits `responses.create`
# while the Interactions one returns its generator directly.
type AnswerStreamFactory = Callable[[], Awaitable[AsyncIterator[ResponseStreamEvent]]]


async def stream_answer_with_retry(
    *, streamer: ResponseStreamer, open_stream: AnswerStreamFactory, message_id: int
) -> str:
    """Streams one answer turn, re-opening the stream on a transient upstream failure.

    This is the only LLM call in the process with no retry anywhere beneath it. LiteLLM's
    router applies `num_retries` and its configured fallbacks to the non-streaming paths,
    which is why the triage and fast one-shots degrade instead of failing, but a provider 5xx
    that arrives as an SSE error frame mid-stream reaches the client untouched -- and that is
    the one turn whose failure a user watches happen.

    Re-issuing the request is safe because an answer turn is a pure read: nothing is written
    before the stream completes, and the retry stays on the same client, the same model and
    the same leased Gemini key, so a Files API uri named in the input keeps resolving. Between
    attempts `reset_for_retry` drops the dead attempt's text and revives the preview editor,
    and `announce_retry` says so on screen, since a silent retry is indistinguishable from a
    model that is simply thinking slowly.

    The provider error reaches the caller only once every attempt is spent (`reraise=True`),
    so the pipeline's error embed and its ❌ stay a single end-of-turn event rather than one
    per attempt. Whether the user is told anything at all is the streamer's own
    `carries_turn_notices`, not a parameter here: a media persona reply renders onto the
    delivered image or video, which is nobody's status surface, and that route is silent to the
    user by design -- the deliverable already landed and nobody is waiting on the words about it.

    Args:
        streamer: The streamer rendering this reply; retried in place so every attempt writes
            to the same Discord message.
        open_stream: Issues the request and returns its event stream.
        message_id: The turn's triggering message, so a retry is greppable with the rest of it.

    Returns:
        The finished reply text.

    Raises:
        Exception: Whatever the last attempt raised, unwrapped -- a caller's error path sees
            the provider failure rather than a tenacity wrapper.
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

    async def _attempt() -> str:
        return await streamer.stream(responses=await open_stream())

    retrying = AsyncRetrying(
        retry=retry_if_exception(is_retryable_llm_error),
        wait=wait_fixed(wait=ANSWER_RETRY_INTERVAL_SECONDS)
        + wait_random(min=0, max=ANSWER_RETRY_JITTER_SECONDS),
        stop=stop_after_attempt(ANSWER_STREAM_MAX_ATTEMPTS),
        before_sleep=_before_retry,
        reraise=True,
    )
    try:
        return await retrying(_attempt)
    except Exception:
        # The streamer stays published on the way out: the caller's error path is what lands
        # the failure on whatever this reply is still showing.
        if streamer.carries_turn_notices:
            await streamer.withdraw_retry_notice()
        raise
