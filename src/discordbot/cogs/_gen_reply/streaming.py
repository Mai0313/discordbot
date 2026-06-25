"""Streams a Responses API reply onto a Discord message."""

import re
import time
from typing import cast
import asyncio
import contextlib
from collections.abc import AsyncIterator

import logfire
from nextcord import File, Message, NotFound, HTTPException, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, PrivateAttr, SkipValidation
from nextcord.utils import escape_mentions
from openai.types.responses import (
    ResponseStreamEvent,
    ResponseCreatedEvent,
    ResponseCompletedEvent,
)

from discordbot.utils.reactions import update_reaction
from discordbot.utils.model_pricing import get_token_rates
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaHostingConfig,
    MediaHostingService,
    MediaDeliveryPlanner,
    upload_limit_for,
)
from discordbot.cogs._gen_reply.voice import (
    VOICE_REPLY_FILENAME,
    VoiceOutcome,
    VoiceSynthesizer,
    speechify_discord_markup,
)
from discordbot.cogs._gen_reply.markers import (
    MAX_INLINE_IMAGES,
    extract_inline_markers,
    scrub_markers_for_preview,
)
from discordbot.cogs._gen_reply.generation import ImageGenerator, MusicGenerator, music_filename

# Filename of a single inline-generated image attached onto a QA reply; mirrors the router IMAGE
# route's `generated.png` so the bot's own generated images render the same in history. Multiple
# images need distinct names, so they fall back to `generated_<n>.png` (Discord collides on dupes).
INLINE_IMAGE_FILENAME = "generated.png"

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
        default=False,
        description="Whether the answer model wrapped a segment in <voice> for this reply.",
    )
    voice_text: str = Field(
        default="",
        description="Speechified text of the <voice> segment used as the spoken-clip input.",
    )
    image_generator: SkipValidation[ImageGenerator | None] = Field(
        default=None,
        description="Inline-image renderer; None disables inline <image> for this reply.",
    )
    image_prompts: list[str] = Field(
        default_factory=list,
        description="The <image> descriptions the answer model asked to illustrate, in order.",
    )
    music_generator: SkipValidation[MusicGenerator | None] = Field(
        default=None,
        description="Inline-music renderer; None disables inline <music> for this reply.",
    )
    music_prompt: str | None = Field(
        default=None,
        description="The <music> description the answer model asked to score, if any.",
    )
    research_brief: str | None = Field(
        default=None,
        description="The <deep-research> brief the answer model asked to launch, if any.",
    )
    media_delivery: MediaDeliveryPlanner = Field(
        default_factory=lambda: MediaDeliveryPlanner(
            media_hosting=MediaHostingService(
                config=MediaHostingConfig(MEDIA_HOSTING_ENABLED=False)
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
        the first real content delta replaces it. The window keeps the newest lines that
        fit one Discord message, so a long think never hits the 2000-char limit.
        """
        if self.content_started:
            return scrub_markers_for_preview(text=self.stored_content)[:DISCORD_MESSAGE_LIMIT]
        if not self.reasoning_content:
            return ""
        # Mentions are escaped because this transient text is never meant to ping;
        # the real reply may mention people, the thought process must not.
        tail = escape_mentions(self.reasoning_content[-1500:])
        lines = [f"-# {line}" for line in tail.splitlines() if line.strip()]
        header = "-# <:message:1517560873000898860> Thinking..."
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

    async def _write_final_message(self, content: str, footer: str) -> None:
        """Writes the final reply, continuing overflow as follow-up replies in the same channel."""
        parent_content, follow_up_chunks = self._split_reply_for_discord(
            content=content, footer=footer
        )
        # Track the parent reply so a later voice attach edits the right message even when
        # the reply is created here (no preview snapshot ran before finalize).
        if self.reply is None:
            self.reply = await self._reply_or_send(content=parent_content)
        else:
            await self.reply.edit(content=parent_content)
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

    async def _consume(self, *, responses: AsyncIterator[ResponseStreamEvent]) -> None:
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
        # The answer model may wrap a <voice> segment (spoken aloud, kept in the reply) and an
        # <image> block (a generation request, removed from the reply). Extract both before the
        # footer is built or anything is written. The <voice> segment stays in the visible text;
        # only it (not the whole reply) feeds the spoken clip so the audio matches what is read.
        markers = extract_inline_markers(text=self.stored_content)
        self.stored_content = markers.cleaned_text
        self.voice_requested = markers.voice_requested
        self.image_prompts = markers.image_prompts
        self.music_prompt = markers.music_prompt
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
        # Footer format must stay matchable by `input.USAGE_FOOTER_RE`; the ⬆/⬇ icons are its anchor.
        model_label = (
            f"{self.model_name} ({self.model_effort})" if self.model_effort else self.model_name
        )
        usage_footer = f"\n\n-# {model_label} · ⬆ {self.input_tokens:,} ⬇ {self.output_tokens:,} · ${cost:.8f}{memory_line}"

        # Final update to ensure complete message is displayed.
        await self._write_final_message(content=self.stored_content, footer=usage_footer)
        reply_chars = len(self.stored_content)
        chunked = reply_chars + len(usage_footer) > DISCORD_MESSAGE_LIMIT
        self.stored_content += usage_footer

        await self._attach_generated_media()
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
            image_count=len(self.image_prompts),
            music_requested=bool(self.music_prompt),
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
        """The destination's real upload ceiling, falling back to Discord's 10MB base in a DM.

        A boosted guild's 50/100MB is honored via nextcord's `filesize_limit`; a DM has no guild
        to query, so it falls back to Discord's non-Nitro base of 10MB (shared helper).
        """
        return upload_limit_for(guild=self.message.guild)

    async def _build_voice_candidate(self) -> MediaItem | None:
        """Synthesizes the <voice> segment to a WAV candidate, or None when not delivered.

        Best-effort: a skip (not requested / disabled / empty) is silent, while a
        requested-but-failed clip (timeout / refusal) hints the source message and returns None.
        The upload-limit decision (attach vs host vs drop) is made by `_attach_generated_media`,
        so an oversized clip is no longer dropped here (there is deliberately no spoken-length cap).
        """
        if not self.voice_requested:
            # The expected common path: the answer model wrapped no <voice> segment.
            logfire.debug("Voice not requested by the answer model", message_id=self.message.id)
            return None
        if self.voice_synthesizer is None:
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
        clip = await self.voice_synthesizer.synthesize(
            text=self.voice_text, end_user_id=self.message.author.name
        )
        if clip.outcome is VoiceOutcome.EMPTY:
            # Nothing to say (the segment was empty after stripping): no hint.
            return None
        if clip.outcome is VoiceOutcome.TIMEOUT:
            # synthesize() logged the timeout; cue the user that the clip ran out of time.
            await self._hint_media_unavailable(emoji="⏱️")
            return None
        if clip.audio is None:
            # Any other synthesis failure (most often a policy refusal); synthesize() logged it.
            await self._hint_media_unavailable(emoji="⚠️")
            return None
        return MediaItem(source=clip.audio, filename=VOICE_REPLY_FILENAME)

    async def _build_image_candidates(self) -> list[MediaItem]:
        """Renders the <image> requests to PNG candidates, in order; [] when none delivered.

        Best-effort like voice: no request or a disabled generator is silent. The capped prompts
        render concurrently; a generation failure drops that image and a single ⚠️ hint rides on
        the source message. The upload-limit decision (attach vs host) is left to
        `_attach_generated_media`, so a large image is no longer dropped here for size.
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
        # Render every requested image concurrently so a slow one never delays the others.
        images = await asyncio.gather(
            *(
                self.image_generator.generate(
                    user_prompt=prompt, end_user_id=self.message.author.name
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
        """Generates the <music> clip to an audio candidate, or None when not delivered.

        Best-effort like the inline image path: a skip (not requested / disabled) is silent, while
        a requested-but-failed clip hints the source message and returns None. The filename suffix
        follows the returned audio mime type so Discord (or the hosted link) renders a player; the
        upload-limit decision (attach vs host) is left to `_attach_generated_media`.
        """
        if self.music_prompt is None:
            # The expected common path: the answer model wrapped no <music> block.
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

    async def _attach_generated_media(self) -> None:
        """Attaches the spoken clip, music clip, and inline images in one edit, hosting overflow.

        The text reply is already on screen, so this adds no latency to it; the media are
        best-effort. Anything that fits the upload limit rides a single `reply.edit(files=...)`
        (one edit, because `edit` replaces the attachment list). Anything too big to upload (a
        long voice WAV in a DM is the common case) is hosted on the external static server and its
        URL appended to the reply instead of being dropped; if hosting is unavailable it degrades
        to today's drop + ⚠️ hint. Voice/music are ordered first so the rare 11-attachment
        overflow peels a trailing image, not the clip.
        """
        if self.reply is None:
            if self.voice_requested or self.image_prompts or self.music_prompt:
                logfire.warn(
                    "Media requested but the reply was never sent; dropping it",
                    message_id=self.message.id,
                )
                await self._hint_media_unavailable(emoji="⚠️")
            return
        reply = self.reply
        # Build every path concurrently so a slow one never blocks the others: a TTS clip or a
        # music render that hangs to its timeout must not delay ready inline images (and vice versa).
        voice_candidate, music_candidate, image_candidates = await asyncio.gather(
            self._build_voice_candidate(),
            self._build_music_candidate(),
            self._build_image_candidates(),
        )
        items = [
            item
            for item in (voice_candidate, music_candidate, *image_candidates)
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
                self.stored_content += link_line
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
        except Exception:
            logfire.warn(
                "Failed to attach generated media onto the reply",
                message_id=self.message.id,
                file_count=len(files),
                _exc_info=True,
            )
            await self._hint_media_unavailable(emoji="⚠️")
            return
        if follow_up is not None:
            with contextlib.suppress(Exception):
                await reply.reply(content=follow_up, allowed_mentions=AllowedMentions.none())

    async def stream(self, *, responses: AsyncIterator[ResponseStreamEvent]) -> str:
        """Streams the reply onto the message and writes the usage footer; returns the full text."""
        try:
            await self._consume(responses=responses)
        finally:
            await self._stop_editor()
        return await self._finalize_reply()
