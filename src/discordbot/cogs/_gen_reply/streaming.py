"""Streams a Responses API reply onto a Discord message."""

from io import BytesIO
import re
import time
import base64
from typing import cast
import asyncio
import contextlib

from openai import AsyncStream
import logfire
import nextcord
from nextcord import File, Message
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.utils import escape_mentions
from openai.types.responses import (
    ResponseStreamEvent,
    ResponseCreatedEvent,
    ResponseCompletedEvent,
)

from discordbot.utils.reactions import update_reaction
from discordbot.utils.model_pricing import get_token_rates
from discordbot.cogs._gen_reply.voice import (
    VOICE_REPLY_FILENAME,
    VoiceOutcome,
    VoiceSynthesizer,
    speechify_discord_markup,
)
from discordbot.cogs._gen_reply.markup import extract_reply_segments, strip_tags_for_preview
from discordbot.cogs._gen_reply.generation import (
    GENERATED_IMAGE_FILENAME,
    ImageGenerator,
    ImageGenerationOutcome,
)

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
    voice_synthesizer: SkipValidation[VoiceSynthesizer | None] = Field(
        default=None,
        description="TTS engine for spoken replies; None disables voice for this reply.",
    )
    voice_requested: bool = Field(
        default=False, description="Whether the answer model wrapped a <voice> span in this reply."
    )
    voice_text: str = Field(
        default="",
        description="The marked spoken span (markup-normalised) used as the voice-clip input.",
    )
    image_generator: SkipValidation[ImageGenerator | None] = Field(
        default=None,
        description="Inline image generator; None disables <image> generation for this reply.",
    )
    image_requested: bool = Field(
        default=False,
        description="Whether the answer model wrapped an <image> span in this reply.",
    )
    image_prompt: str = Field(
        default="", description="The extracted <image> span used as the rough generation request."
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
            return strip_tags_for_preview(text=self.stored_content)[:DISCORD_MESSAGE_LIMIT]
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
        # Track the parent reply so a later voice attach edits the right message even when
        # the reply was created here (no preview snapshot ran before finalize).
        self.reply = reply
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
            logfire.info(
                "gen_reply first content delta",
                elapsed_seconds=time.monotonic() - self.created_at,
                model=self.model_name,
                message_id=self.message.id,
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
        """Writes the usage footer and final reply once the stream is consumed."""
        input_rate, output_rate = get_token_rates(model_name=self.model_name)
        cost = input_rate * self.input_tokens + output_rate * self.output_tokens

        self.stored_content = CODED_MENTION_RE.sub(r"\1", self.stored_content)
        # The answer model marks parts of its reply with inline control tags: <voice>...</voice>
        # wraps the span to speak (kept in the text, only the tags removed) and <image>...</image>
        # wraps a draw request (removed from the text entirely). Split them out before the footer
        # is built or anything is written so the visible reply never shows a tag or a raw prompt.
        segments = extract_reply_segments(text=self.stored_content)
        self.stored_content = segments.display_text
        self.voice_requested = segments.voice_requested
        self.image_requested = segments.image_requested
        self.image_prompt = segments.image_prompt
        # Only the wrapped span is spoken, and the spoken clip must not narrate raw Discord
        # markup (a `<@id>` mention reads as a bare snowflake), so the voice input is normalised.
        self.voice_text = (
            speechify_discord_markup(
                text=segments.voice_text, resolve_name=self._resolve_mention_name
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
                memory_line = f"\n-# 🧠 已讀取 {', '.join(names[:2])} 等 {len(names)} 人的記憶"
            else:
                memory_line = f"\n-# 🧠 已讀取 {', '.join(names)} 的記憶"
        # Footer format must stay matchable by `input.USAGE_FOOTER_RE`; the ⬆/⬇ icons are its anchor.
        model_label = (
            f"{self.model_name} ({self.model_effort})" if self.model_effort else self.model_name
        )
        usage_footer = f"\n\n-# {model_label} · ⬆ {self.input_tokens:,} ⬇ {self.output_tokens:,} · ${cost:.8f}{memory_line}"

        # Final update to ensure complete message is displayed.
        await self._write_final_message(
            reply=self.reply, content=self.stored_content, footer=usage_footer
        )
        reply_chars = len(self.stored_content)
        chunked = reply_chars + len(usage_footer) > DISCORD_MESSAGE_LIMIT
        self.stored_content += usage_footer

        await self._maybe_attach_media()
        logfire.info(
            "gen_reply reply finalized",
            message_id=self.message.id,
            model=self.model_name,
            effort=self.model_effort,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost=cost,
            reply_chars=reply_chars,
            voice_requested=self.voice_requested,
            image_requested=self.image_requested,
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
        """Marks the source message so a dropped voice/image is not fully silent to the user.

        The reply stays text-only and the user gets no message; this best-effort reaction is
        the only signal. It rides on the source message as an independent reaction (no
        `previous`), so the pipeline's status chain never removes it. Failures are suppressed
        inside `update_reaction`.
        """
        await update_reaction(message=self.message, bot_user=None, emoji=emoji)

    def _upload_limit(self) -> int:
        """The destination's real attachment-size ceiling.

        A DM has no guild to query, so it falls back to Discord's non-Nitro base of 10MB; a
        guild trusts nextcord's `filesize_limit` (correct for boosted 50/100MB tiers).
        """
        return self.message.guild.filesize_limit if self.message.guild else 10 * 1024 * 1024

    async def _render_voice_file(self) -> File | None:
        """Synthesizes the marked spoken span to a File, or None (with a hint) on failure."""
        if not self.voice_requested or self.voice_synthesizer is None:
            return None
        logfire.info(
            "Synthesizing voice reply", message_id=self.message.id, text_chars=len(self.voice_text)
        )
        clip = await self.voice_synthesizer.synthesize(
            text=self.voice_text, end_user_id=self.message.author.name
        )
        if clip.outcome is VoiceOutcome.EMPTY:
            # Nothing to say (the span was empty after stripping): no hint.
            return None
        if clip.outcome is VoiceOutcome.TIMEOUT:
            # synthesize() logged the timeout; cue the user that the clip ran out of time.
            await self._hint_media_unavailable(emoji="⏱️")
            return None
        if clip.audio is None:
            # Any other synthesis failure (most often a policy refusal); synthesize() logged it.
            await self._hint_media_unavailable(emoji="⚠️")
            return None
        if len(clip.audio) > self._upload_limit():
            logfire.warn(
                "Synthesized voice exceeds the guild upload limit; dropping audio",
                message_id=self.message.id,
                audio_bytes=len(clip.audio),
                upload_limit=self._upload_limit(),
            )
            await self._hint_media_unavailable(emoji="⚠️")
            return None
        return File(fp=BytesIO(clip.audio), filename=VOICE_REPLY_FILENAME)

    async def _render_image_file(self) -> File | None:
        """Generates the marked inline image to a File, or None (with a hint) on failure."""
        if not self.image_requested or self.image_generator is None:
            return None
        logfire.info(
            "Generating inline image",
            message_id=self.message.id,
            prompt_chars=len(self.image_prompt),
        )
        image = await self.image_generator.generate(
            rough_prompt=self.image_prompt, end_user_id=self.message.author.name
        )
        if image.outcome is ImageGenerationOutcome.EMPTY:
            # Nothing to draw (the span was empty): no hint.
            return None
        if image.outcome is ImageGenerationOutcome.TIMEOUT:
            await self._hint_media_unavailable(emoji="⏱️")
            return None
        if image.image_b64 is None:
            await self._hint_media_unavailable(emoji="⚠️")
            return None
        try:
            image_bytes = base64.b64decode(image.image_b64)
        except (ValueError, TypeError):
            # A malformed/incorrectly-padded payload reported as OK must stay best-effort: the
            # text reply is already on screen, so degrade with a hint instead of letting the
            # decode raise out of _finalize_reply into the pipeline's generic error path.
            logfire.warn(
                "Inline image returned malformed base64; replying without it",
                message_id=self.message.id,
                _exc_info=True,
            )
            await self._hint_media_unavailable(emoji="⚠️")
            return None
        if len(image_bytes) > self._upload_limit():
            logfire.warn(
                "Generated image exceeds the guild upload limit; dropping image",
                message_id=self.message.id,
                image_bytes=len(image_bytes),
                upload_limit=self._upload_limit(),
            )
            await self._hint_media_unavailable(emoji="⚠️")
            return None
        return File(fp=BytesIO(image_bytes), filename=GENERATED_IMAGE_FILENAME)

    async def _maybe_attach_media(self) -> None:
        """Attaches any requested voice clip and inline image onto the reply, best-effort.

        Runs only for spans the answer model marked and for media enabled this turn. The text
        reply is already on screen, so this adds no latency to it; voice and image render
        concurrently and are attached together in one edit (a reply rarely has both, but a
        second `edit(file=...)` would clobber the first, so both must ride one `files=` edit).
        Each is best-effort: a failure, an over-limit clip, or an unsent reply leaves the text
        reply and adds a small emoji hint (⏱️ for a timeout, ⚠️ for any other failure).
        """
        want_voice = self.voice_requested and self.voice_synthesizer is not None
        want_image = self.image_requested and self.image_generator is not None
        if self.voice_requested and self.voice_synthesizer is None:
            # Voice is intentionally off this turn (kill-switch), not a failure: no hint.
            logfire.info(
                "Voice requested but disabled for this turn; replying without audio",
                message_id=self.message.id,
            )
        if self.image_requested and self.image_generator is None:
            logfire.info(
                "Image requested but disabled for this turn; replying without image",
                message_id=self.message.id,
            )
        if not want_voice and not want_image:
            # The expected common path: the answer model marked neither span.
            return
        if self.reply is None:
            logfire.warn(
                "Media requested but the reply was never sent; dropping it",
                message_id=self.message.id,
            )
            await self._hint_media_unavailable(emoji="⚠️")
            return
        reply = self.reply
        voice_file, image_file = await asyncio.gather(
            self._render_voice_file(), self._render_image_file()
        )
        # Image first so it shows above the audio attachment when both ride along.
        files = [media for media in (image_file, voice_file) if media is not None]
        if not files:
            return
        try:
            if len(files) == 1:
                await reply.edit(file=files[0])
            else:
                await reply.edit(files=files)
        except Exception:
            logfire.warn(
                "Failed to attach generated media onto the reply",
                message_id=self.message.id,
                _exc_info=True,
            )
            await self._hint_media_unavailable(emoji="⚠️")
            return
        logfire.info(
            "Media attached onto the reply",
            message_id=self.message.id,
            attachments=len(files),
            voice=voice_file is not None,
            image=image_file is not None,
        )

    async def stream(self, *, responses: AsyncStream[ResponseStreamEvent]) -> str:
        """Streams the reply onto the message and writes the usage footer; returns the full text."""
        try:
            await self._consume(responses=responses)
        finally:
            await self._stop_editor()
        return await self._finalize_reply()
