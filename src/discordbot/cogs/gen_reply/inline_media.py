"""Inline-marker media synthesis and attachment onto streamed Discord replies."""

from typing import TypedDict
import asyncio
from collections.abc import Callable

import logfire
from nextcord import File, Message, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, SkipValidation

from discordbot.typings.media import (
    VOICE_REPLY_FILENAME,
    INLINE_IMAGE_FILENAME,
    INLINE_VIDEO_FILENAME,
    LoadedMedia,
    music_filename,
)
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.discord_embeds import DISCORD_MESSAGE_LIMIT
from discordbot.utils.discord_errors import report_discord_failure
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaDeliveryPlanner,
    upload_limit_for,
)
from discordbot.cogs.gen_reply.markers import MAX_INLINE_IMAGES, InlineMarkers
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.generation import (
    ImageGenerator,
    MusicGenerator,
    VideoGenerator,
    VoiceGenerator,
)
from discordbot.cogs.gen_reply.references import replied_to_message
from discordbot.cogs.gen_reply.status_marks import (
    IMAGE_EMOJI,
    MUSIC_EMOJI,
    VIDEO_EMOJI,
    VOICE_EMOJI,
    DROPPED_HINT_EMOJI,
    TIMEOUT_HINT_EMOJI,
)


class _MediaEditKwargs(TypedDict, total=False):
    """Typed optional kwargs forwarded to `Message.edit` for media attachments."""

    content: str
    files: list[File]
    allowed_mentions: AllowedMentions


class InlineMediaResult(BaseModel):
    """State changes produced when attaching or hosting inline generated media onto a reply."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    landed: bool = Field(
        default=True,
        description="Whether every requested edit or hosted-link follow-up succeeded.",
    )
    hosted_block: str | None = Field(
        default=None,
        description="Hosted media URL block spliced into the reply, or None when not spliced.",
    )
    updated_content: str | None = Field(
        default=None,
        description="Full reply text after splicing `hosted_block`, or None when unchanged.",
    )
    follow_up_message: SkipValidation[Message | None] = Field(
        default=None,
        description="Follow-up message carrying hosted URLs when they did not fit the main reply.",
    )


class InlineMediaAttacher(BaseModel):
    """Synthesizes and attaches inline-marker media (voice, image, music, video) onto a reply.

    Runs requested generators concurrently under the surface's delivery budget, plans native
    attachments vs URL hosting through `media_delivery`, and lands files or hosted links
    best-effort without raising into the reply pipeline.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message] = Field(
        ..., description="Triggering Discord message providing author, guild, and log correlation."
    )
    surface: TurnSurface = Field(
        ..., description="Turn surface carrying status reactions, hints, budget, and follow-ups."
    )
    markers: InlineMarkers = Field(
        ..., description="Extracted inline markers deciding which media types to generate."
    )
    media_delivery: MediaDeliveryPlanner = Field(
        ..., description="Decides attach-vs-host-vs-drop for generated media."
    )
    voice_text: str = Field(
        default="", description="Speechified text of `<generate-voice>` for TTS synthesis."
    )
    voice_generator: SkipValidation[VoiceGenerator | None] = Field(
        default=None, description="TTS engine for spoken replies; None disables voice."
    )
    image_generator: SkipValidation[ImageGenerator | None] = Field(
        default=None, description="Inline-image renderer; None disables `<generate-image>`."
    )
    input_builder: SkipValidation[MessageInputBuilder | None] = Field(
        default=None,
        description="Loads turn source images for inline image editing and video references.",
    )
    music_generator: SkipValidation[MusicGenerator | None] = Field(
        default=None, description="Inline-music renderer; None disables `<generate-music>`."
    )
    video_generator: SkipValidation[VideoGenerator | None] = Field(
        default=None, description="Inline-video renderer; None disables `<generate-video>`."
    )

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
        tasks = []
        for prompt in self.markers.image_prompts:
            tasks.append(
                self.image_generator.generate(
                    user_prompt=prompt,
                    end_user_id=self.message.author.name,
                    image_bytes_list=source_bytes or None,
                )
            )
        images = await asyncio.gather(*tasks)
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

    def _take_media_result[ResultT](
        self, task: asyncio.Task[ResultT], completed: bool, default: ResultT, media: str
    ) -> tuple[ResultT, bool]:
        """Extracts one finished media task's result, logging unexpected exceptions as dropped."""
        if not completed:
            return default, False
        try:
            return task.result(), False
        except Exception as exc:
            # Broad: best-effort inline media builders must not escape `finalize_reply`.
            logfire.warn(
                "Inline media builder raised unexpectedly; dropping it",
                message_id=self.message.id,
                media=media,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return default, True

    async def _collect_generated_media(self) -> list[MediaItem]:
        """Runs inline media builders under the surface budget, cancelling stragglers on exit."""
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
        # cancel and drain pending builds and `source_images_task` in finally because
        # `asyncio.wait` does not cancel on outer cancellation.
        builds: set[asyncio.Task[MediaItem | None] | asyncio.Task[list[MediaItem]]] = {
            voice_task,
            music_task,
            video_task,
            image_task,
        }
        try:
            done, pending = await asyncio.wait(
                builds, timeout=self.surface.delivery_budget_seconds()
            )
        finally:
            to_drain: list[asyncio.Task[object]] = []
            for task in builds:
                if not task.done():
                    to_drain.append(task)
            if source_images_task is not None:
                to_drain.append(source_images_task)
            for task in to_drain:
                if not task.done():
                    task.cancel()
            if to_drain:
                await asyncio.gather(*to_drain, return_exceptions=True)
        if pending:
            logfire.warn(
                "gen_reply inline media outlived the /ask window; dropping it",
                message_id=self.message.id,
                pending=len(pending),
            )
            await self.surface.hint(emoji=TIMEOUT_HINT_EMOJI)
        voice_candidate, voice_failed = self._take_media_result(
            task=voice_task, completed=voice_task in done, default=None, media="voice"
        )
        music_candidate, music_failed = self._take_media_result(
            task=music_task, completed=music_task in done, default=None, media="music"
        )
        video_candidate, video_failed = self._take_media_result(
            task=video_task, completed=video_task in done, default=None, media="video"
        )
        image_candidates, image_failed = self._take_media_result(
            task=image_task, completed=image_task in done, default=[], media="image"
        )
        if voice_failed or music_failed or video_failed or image_failed:
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
        return [
            item
            for item in (voice_candidate, music_candidate, video_candidate, *image_candidates)
            if item is not None
        ]

    async def _send_hosted_follow_up(
        self, reply: Message, follow_up: str, hosted_count: int, file_count: int
    ) -> Message | None:
        """Posts hosted media links as a follow-up message when appending them overflows `reply`."""
        try:
            return await self.surface.follow_up(
                previous=reply, content=follow_up, allowed_mentions=AllowedMentions.none()
            )
        except Exception as exc:
            # Broad: best-effort hosted-link follow-up must not raise into the reply pipeline.
            report_discord_failure(
                error=exc,
                not_found_message="Hosted media link follow-up failed; the media URL was never posted",
                not_found_level="warn",
                not_found_exc_info=True,
                forbidden_message="Hosted media link follow-up failed; the media URL was never posted",
                other_message="Hosted media link follow-up failed; the media URL was never posted",
                message_id=self.message.id,
                hosted_url_count=hosted_count,
                file_count=file_count,
            )
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
            return None

    async def apply_media_edit(
        self,
        reply: Message,
        files: list[File],
        hosted_urls: list[str],
        render_with_hosted: Callable[[str], str],
    ) -> InlineMediaResult:
        """Edits `files` and `hosted_urls` onto `reply` (or follow-up), returning state updates.

        Args:
            reply (Message): Primary Discord reply message to attach media onto.
            files (list[File]): Native Discord file attachments to upload.
            hosted_urls (list[str]): Hosted media URLs for oversized items.
            render_with_hosted (Callable[[str], str]): Renders the full reply text with a candidate
                hosted-URL block spliced above hint, memory note, and usage footer.
        """
        if not files and not hosted_urls:
            return InlineMediaResult()
        hosted_block = ""
        content: str | None = None
        follow_up: str | None = None
        if hosted_urls:
            hosted_block = "-# 媒體過大，改用連結\n" + "\n".join(hosted_urls)
            candidate = render_with_hosted(hosted_block)
            if len(candidate) <= DISCORD_MESSAGE_LIMIT:
                content = candidate
            else:
                follow_up = hosted_block
        if files or content is not None:
            edit_kwargs: _MediaEditKwargs = {"allowed_mentions": AllowedMentions.none()}
            if content is not None:
                edit_kwargs["content"] = content
            if files:
                edit_kwargs["files"] = files
            try:
                await reply.edit(**edit_kwargs)
            except Exception as exc:
                # Broad: best-effort media attach must not raise into the reply pipeline.
                report_discord_failure(
                    error=exc,
                    not_found_message="Failed to attach generated media onto the reply",
                    not_found_level="warn",
                    not_found_exc_info=True,
                    forbidden_message="Failed to attach generated media onto the reply",
                    other_message="Failed to attach generated media onto the reply",
                    message_id=self.message.id,
                    file_count=len(files),
                    hosted_count=len(hosted_urls),
                )
                await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
                return InlineMediaResult(landed=False)
        follow_up_message: Message | None = None
        if follow_up is not None:
            follow_up_message = await self._send_hosted_follow_up(
                reply=reply,
                follow_up=follow_up,
                hosted_count=len(hosted_urls),
                file_count=len(files),
            )
            if follow_up_message is None:
                return InlineMediaResult(landed=False)
        return InlineMediaResult(
            landed=True,
            hosted_block=hosted_block if content is not None else None,
            updated_content=content,
            follow_up_message=follow_up_message,
        )

    async def attach(
        self, reply: Message | None, reply_deleted: bool, render_with_hosted: Callable[[str], str]
    ) -> InlineMediaResult:
        """Builds inline media within the surface budget and attaches or hosts them in one edit.

        Args:
            reply (Message | None): Primary reply message, or None when no reply landed.
            reply_deleted (bool): Whether the user deleted `reply` mid-stream.
            render_with_hosted (Callable[[str], str]): Renders reply content with a hosted-URL block.
        """
        if reply_deleted:
            # User deleted the reply; skip dropped-media hints.
            return InlineMediaResult()
        if reply is None:
            if self.markers.media_requested:
                logfire.warn(
                    "Media requested but the reply was never sent; dropping it",
                    message_id=self.message.id,
                )
                await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
            return InlineMediaResult()
        items = await self._collect_generated_media()
        if not items:
            return InlineMediaResult()
        plan = await self.media_delivery.plan(
            items=items,
            upload_limit=upload_limit_for(
                guild=self.message.guild, interaction=self.surface.interaction
            ),
            envelope_margin=MEDIA_ENVELOPE_MARGIN,
        )
        files = [item.to_file() for item in plan.native]
        result = await self.apply_media_edit(
            reply=reply,
            files=files,
            hosted_urls=plan.hosted_urls,
            render_with_hosted=render_with_hosted,
        )
        if not result.landed:
            return result
        if plan.dropped_items or plan.clamped_items:
            await self.surface.hint(emoji=DROPPED_HINT_EMOJI)
        logfire.info(
            "Generated media attached",
            message_id=self.message.id,
            file_count=len(files),
            hosted_count=len(plan.hosted_urls),
        )
        return result
