"""The IMAGE and VIDEO routes: generate the media, deliver it, then talk about it.

Both routes keep media generation on the hard error path; once the picture or clip is delivered,
the streamed persona reply (`answer.py`) is best-effort and leaves the delivered media on screen
whatever happens.

Each generation runs inside `TurnSurface.delivery_budget_seconds` (None on the gateway path) so an
`/ask` turn times out while its interaction token can still carry both the deliverable and the
failure notice.
"""

import time
from typing import TYPE_CHECKING
import asyncio
import contextlib
from collections.abc import AsyncGenerator

import logfire
from nextcord import Message
from pydantic import Field, BaseModel
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.utils.images import to_data_uri
from discordbot.typings.timeouts import GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS
from discordbot.utils.media_delivery import MediaItem, upload_limit_for
from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.prompts import (
    IMAGE_PROMPT,
    VIDEO_PROMPT,
    IMAGE_REPLY_PROMPT,
    VIDEO_REPLY_PROMPT,
)
from discordbot.cogs.gen_reply.surface import WINDOW_EXPIRED_NOTICE
from discordbot.typings.context_budgets import MAX_VIDEO_REFERENCE_IMAGES
from discordbot.cogs.gen_reply.files_api import upload_as_input_file
from discordbot.cogs.gen_reply.generation import INLINE_IMAGE_FILENAME, INLINE_VIDEO_FILENAME
from discordbot.cogs.gen_reply.references import replied_to_message
from discordbot.cogs.gen_reply.turn_state import dispatched_model
from discordbot.cogs.gen_reply.speculation import discard_task

if TYPE_CHECKING:
    from discordbot.typings.media import LoadedMedia


class MediaReplyRoutes(BaseModel):
    """Runs the IMAGE and VIDEO routes for one message."""

    answer: AnswerTurn = Field(
        ...,
        description=(
            "Streams the best-effort persona reply about the delivered media, and carries the "
            "config, toolkit, surface and media planner this route renders and delivers through."
        ),
    )

    @property
    def message(self) -> Message:
        """Returns the message that triggered the media route."""
        return self.answer.message

    async def _deliver(self, data: bytes, filename: str) -> Message | None:
        """Delivers generated media bytes, returning the sent message or None when hosted as a URL.

        When hosting is unavailable, attempts a native upload so an oversize failure stays on the
        route's outer error path.
        """
        item = MediaItem(source=data, filename=filename)
        plan = await self.answer.media_delivery.plan(
            items=[item],
            upload_limit=upload_limit_for(
                guild=self.message.guild, interaction=self.answer.surface.interaction
            ),
        )
        if plan.native:
            return await self.answer.reply_or_send(
                content=self.message.author.mention, media=plan.native[0]
            )
        if not plan.hosted_urls:
            # Hosting is off or failed: attempt the native upload so oversize raises on the outer
            # error path.
            return await self.answer.reply_or_send(content=self.message.author.mention, media=item)
        # Too big to attach: the hosted URL is the deliverable (pings the author once). The persona
        # reply, if it runs, streams onto its own fresh message so it never clobbers this link.
        await self.answer.reply_or_send(
            content=f"{self.message.author.mention}\n{plan.hosted_urls[0]}"
        )
        return None

    @contextlib.asynccontextmanager
    async def _delivery_window(
        self, context_task: asyncio.Task[ReplyContext]
    ) -> AsyncGenerator[asyncio.Timeout]:
        """Yields the surface delivery timeout for media generation and drains `context_task` on error.

        Callers enter the yielded timeout around generation only, keeping upload delivery outside
        it. An expired window re-raises `TimeoutError(WINDOW_EXPIRED_NOTICE)` to distinguish
        surface expiry from a renderer's own timeout.
        """
        window = asyncio.timeout(delay=self.answer.surface.delivery_budget_seconds())
        try:
            yield window
        except Exception as exc:
            await discard_task(task=context_task, label="prep", message_id=self.message.id)
            if window.expired():
                raise TimeoutError(WINDOW_EXPIRED_NOTICE) from exc
            raise

    async def handle_image(
        self, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Generates or edits an image, delivers it, and streams a best-effort persona reply.

        `context_task` is awaited only after the image lands on screen so context preparation
        overlaps generation; on failure before delivery, `_delivery_window` drains `context_task`.

        Args:
            user_prompt (str): Mention-stripped request text from the user.
            context_task (asyncio.Task[ReplyContext]): Speculative reply context task for the
                persona reply.
        """
        message = self.message
        toolkit = self.answer.toolkit
        started = time.monotonic()
        async with self._delivery_window(context_task=context_task) as window:
            async with window:
                image_bytes_list = [
                    loaded.data
                    for loaded in await toolkit.input_builder.get_turn_image_sources(
                        message=message, replied_to=replied_to_message(message=message)
                    )
                ]
                # Logged once the sources are in, since only they say whether this is an edit.
                logfire.info(
                    "gen_reply image generation start",
                    message_id=message.id,
                    model=toolkit.runtime_models.image_model.name,
                    has_source_images=bool(image_bytes_list),
                )

                refined_prompt = await toolkit.prompt_generator.refine(
                    user_prompt=user_prompt,
                    instructions=IMAGE_PROMPT,
                    end_user_id=message.author.name,
                    enabled=self.answer.config.image_refine_prompt_enabled,
                    image_bytes_list=image_bytes_list or None,
                )
                # The director above is best-effort and swallows its own failures, so from here
                # the image model is the only one a failure can be reported against.
                dispatched_model.set(toolkit.runtime_models.image_model.name)
                image_bytes = await toolkit.image_generator.render(
                    prompt=refined_prompt,
                    end_user_id=message.author.name,
                    image_bytes_list=image_bytes_list or None,
                )
            reply = await self._deliver(data=image_bytes, filename=INLINE_IMAGE_FILENAME)
            logfire.info(
                "gen_reply image delivered",
                message_id=message.id,
                model=toolkit.runtime_models.image_model.name,
                elapsed_seconds=time.monotonic() - started,
            )

        # Inline base64 keeps the image persona reply provider-agnostic, unlike video's Files API
        # upload.
        await self.answer.stream_media_persona_reply(
            reply=reply,
            context_task=context_task,
            system_prompt=IMAGE_REPLY_PROMPT,
            focus_part=ResponseInputImageParam(
                image_url=to_data_uri(data=image_bytes), detail="auto", type="input_image"
            ),
            media_noun="image",
        )

    async def handle_video(
        self, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Generates or edits a video via the Gemini Interactions API, delivers it, and replies.

        When the current or replied-to message carries a video, edits that clip in place without
        prompt refinement; otherwise refines `user_prompt` and passes up to
        `MAX_VIDEO_REFERENCE_IMAGES` source images as references. `context_task` is consumed after
        delivery (or drained on error).

        Args:
            user_prompt (str): Mention-stripped request text from the user.
            context_task (asyncio.Task[ReplyContext]): Speculative reply context task for the
                persona reply.
        """
        message = self.message
        toolkit = self.answer.toolkit
        started = time.monotonic()
        logfire.info(
            "gen_reply video generation start",
            message_id=message.id,
            model=toolkit.runtime_models.video_model.name,
        )
        async with self._delivery_window(context_task=context_task) as window:
            async with window:
                replied_to = replied_to_message(message=message)
                source_messages = [message, *([replied_to] if replied_to is not None else [])]
                # Find the source video first, by priority (current message, then replied-to);
                # each message reads at most its first clip. Only when there is no source video do
                # we download reference images, so an edit is never delayed by media it discards.
                source_video: LoadedMedia | None = None
                for source_message in source_messages:
                    videos = await toolkit.input_builder.get_video_sources(message=source_message)
                    if videos:
                        source_video = videos[0]
                        break
                # Both branches end in the same omni render, and the director the else branch runs
                # first is best-effort, so this is the model a failure past here belongs to.
                dispatched_model.set(toolkit.runtime_models.video_model.name)
                if source_video is not None:
                    # A source video is edited in place (task=edit): omni ingests the actual clip,
                    # so the prompt is the literal edit instruction. The director is skipped here —
                    # it only grounds on image parts (a video-only edit would run it blind) and it
                    # sits serially on the time-to-video path; the user's edit request is already
                    # specific. omni takes a single input here, so any accompanying reference
                    # images are dropped.
                    video_bytes = await toolkit.video_generator.render(
                        prompt=user_prompt, reference_image_sources=[], source_video=source_video
                    )
                else:
                    # No source video: gather the message + replied-to images as subject
                    # references, capped to the same set render sends (omni takes a few), so the
                    # director grounds on exactly those frames and no unused bytes ride the path.
                    images = (
                        await toolkit.input_builder.get_turn_image_sources(
                            message=message, replied_to=replied_to
                        )
                    )[:MAX_VIDEO_REFERENCE_IMAGES]
                    refined_prompt = await toolkit.prompt_generator.refine(
                        user_prompt=user_prompt,
                        instructions=VIDEO_PROMPT,
                        end_user_id=message.author.name,
                        enabled=self.answer.config.video_refine_prompt_enabled,
                        image_bytes_list=[loaded.data for loaded in images] or None,
                    )
                    video_bytes = await toolkit.video_generator.render(
                        prompt=refined_prompt, reference_image_sources=images
                    )
            reply = await self._deliver(data=video_bytes, filename=INLINE_VIDEO_FILENAME)
            logfire.info(
                "gen_reply video delivered",
                message_id=message.id,
                model=toolkit.runtime_models.video_model.name,
                total_elapsed_seconds=time.monotonic() - started,
                bytes=len(video_bytes),
            )

        await self._reply_about_video(
            reply=reply, video_bytes=video_bytes, context_task=context_task
        )

    async def _reply_about_video(
        self, reply: Message | None, video_bytes: bytes, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Uploads `video_bytes` to the Gemini Files API and streams a best-effort persona reply.

        Drains `context_task` when the upload fails so no orphaned reply message is created.
        """
        video_part = await upload_as_input_file(
            client=self.answer.toolkit.gemini_client,
            source=video_bytes,
            mime_type="video/mp4",
            filename=INLINE_VIDEO_FILENAME,
            timeout_seconds=GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS,
        )
        if video_part is None:
            await discard_task(task=context_task, label="prep", message_id=self.message.id)
            return
        await self.answer.stream_media_persona_reply(
            reply=reply,
            context_task=context_task,
            system_prompt=VIDEO_REPLY_PROMPT,
            focus_part=video_part,
            media_noun="video",
        )
