"""The IMAGE and VIDEO routes: generate media, deliver it, then stream a best-effort persona reply.

Media generation runs inside `TurnSurface.delivery_budget_seconds` on the hard error path; once
delivered, the persona reply (`answer.py`) is best-effort and leaves the media on screen.
"""

import time
import asyncio
import contextlib
from collections.abc import AsyncGenerator

import logfire
from nextcord import Message
from pydantic import Field, BaseModel
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.media import INLINE_IMAGE_FILENAME, INLINE_VIDEO_FILENAME
from discordbot.typings.timeouts import GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.utils.discord_errors import report_discord_failure
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE, defuse_separator_fences
from discordbot.utils.media_delivery import MediaItem, upload_limit_for
from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.cogs.gen_reply.images import to_data_uri
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.prompts import (
    IMAGE_PROMPT,
    VIDEO_PROMPT,
    IMAGE_REPLY_PROMPT,
    VIDEO_REPLY_PROMPT,
)
from discordbot.cogs.gen_reply.surface import WINDOW_EXPIRED_NOTICE
from discordbot.typings.context_budgets import (
    MAX_VIDEO_REFERENCE_IMAGES,
    MEDIA_REFINE_HISTORY_LIMIT,
    MEDIA_REFINE_CONTEXT_MAX_CHARS,
)
from discordbot.cogs.gen_reply.files_api import upload_as_input_file
from discordbot.cogs.gen_reply.references import replied_to_message
from discordbot.cogs.gen_reply.turn_state import dispatched_model
from discordbot.cogs.gen_reply.speculation import discard_task


def _extract_refine_message_line(
    input_builder: MessageInputBuilder, msg: Message, prefix: str = ""
) -> str | None:
    """Renders one Discord message as a single text line for prompt refinement, or None if empty."""
    text = input_builder.get_cleaned_content(message=msg)
    if text:
        # Strip usage footers even when bot.user is unset or author id differs in synthetic tests.
        text = USAGE_FOOTER_RE.sub("", f"\n\n{text}").strip()
    if not text:
        return None
    safe_author = " ".join(defuse_separator_fences(text=msg.author.display_name).split()) or "User"
    safe_text = defuse_separator_fences(text=text)
    return f"{prefix}{safe_author}: {safe_text}"


def _format_refine_context(lines: list[str], max_chars: int) -> str | None:
    """Joins the newest `lines` that fit within `max_chars`, or None when `lines` is empty."""
    if not lines:
        return None
    kept: list[str] = []
    spent = 0
    for line in reversed(lines):
        separator = 1 if kept else 0
        if spent + separator + len(line) <= max_chars:
            kept.append(line)
            spent += separator + len(line)
            continue
        if not kept:
            # Keep the newest message truncated at the tail so long numbered lists stay usable.
            kept.append(
                f"{line[: max_chars - 1].rstrip()}…" if max_chars > 1 else line[:max_chars]
            )
        break
    kept.reverse()
    return "\n".join(kept)


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
        """Delivers generated media, returning the sent message or None when hosted as a URL."""
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
            # Hosting unavailable: try native upload so oversize raises on the outer error path.
            return await self.answer.reply_or_send(content=self.message.author.mention, media=item)
        # Hosted URL is the deliverable; return None so the persona reply uses a fresh message.
        await self.answer.reply_or_send(
            content=f"{self.message.author.mention}\n{plan.hosted_urls[0]}"
        )
        return None

    @contextlib.asynccontextmanager
    async def _delivery_window(
        self, context_task: asyncio.Task[ReplyContext]
    ) -> AsyncGenerator[asyncio.Timeout]:
        """Yields the surface delivery timeout and drains `context_task` on error.

        Caller must enter the timeout around generation only, leaving upload delivery outside.
        """
        window = asyncio.timeout(delay=self.answer.surface.delivery_budget_seconds())
        try:
            yield window
        except Exception as exc:
            await discard_task(task=context_task, label="prep", message_id=self.message.id)
            if window.expired():
                raise TimeoutError(WINDOW_EXPIRED_NOTICE) from exc
            raise

    async def _load_refine_context(self) -> str | None:
        """Fetches recent text-only history and replied-to text for prompt refinement.

        Best-effort: logs and degrades to whatever replied-to text is already on `message` if
        channel history cannot be read, and never awaits `parts_task` or attachment uploads.
        """
        message = self.message
        input_builder = self.answer.toolkit.input_builder
        replied_to = replied_to_message(message=message)
        try:
            history = await self.answer.surface.fetch_history(limit=MEDIA_REFINE_HISTORY_LIMIT)
        except Exception as exc:
            # Broad: history read is best-effort enrichment; never fail media generation over it.
            report_discord_failure(
                error=exc,
                not_found_message="Channel missing during media prompt refinement history read",
                forbidden_message=(
                    "Cannot read channel history for media prompt refinement; continuing without history"
                ),
                other_message=(
                    "Failed to read channel history for media prompt refinement; continuing without history"
                ),
                message_id=message.id,
                channel_id=message.channel.id,
            )
            history = []

        replied_line = (
            _extract_refine_message_line(
                input_builder=input_builder, msg=replied_to, prefix="[Replied-to] "
            )
            if replied_to is not None
            else None
        )
        replied_id = replied_to.id if replied_to is not None and replied_line is not None else None
        lines: list[str] = []
        for hist_msg in history:
            if replied_id is not None and hist_msg.id == replied_id:
                continue
            line = _extract_refine_message_line(input_builder=input_builder, msg=hist_msg)
            if line is not None:
                lines.append(line)
        if replied_line is not None:
            lines.append(replied_line)
        return _format_refine_context(lines=lines, max_chars=MEDIA_REFINE_CONTEXT_MAX_CHARS)

    async def handle_image(
        self, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Generates or edits an image, delivers it, and streams a best-effort persona reply.

        Awaits `context_task` only after delivery; `_delivery_window` drains it on error.

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
                # Logged after loading sources so has_source_images reflects edit vs fresh render.
                logfire.info(
                    "gen_reply image generation start",
                    message_id=message.id,
                    model=toolkit.runtime_models.image_model.name,
                    has_source_images=bool(image_bytes_list),
                )
                conversation_context = (
                    await self._load_refine_context()
                    if self.answer.config.image_refine_prompt_enabled
                    else None
                )
                refined_prompt = await toolkit.prompt_generator.refine(
                    user_prompt=user_prompt,
                    instructions=IMAGE_PROMPT,
                    end_user_id=message.author.name,
                    enabled=self.answer.config.image_refine_prompt_enabled,
                    image_bytes_list=image_bytes_list or None,
                    conversation_context=conversation_context,
                )
                # Prompt refinement is best-effort; failures past here belong to image_model.
                dispatched_model.set(toolkit.runtime_models.image_model.name)
                image_bytes = await toolkit.image_generator.render(
                    prompt=refined_prompt,
                    end_user_id=message.author.name,
                    image_bytes_list=image_bytes_list or None,
                )
            reply = await self._deliver(data=image_bytes, filename=INLINE_IMAGE_FILENAME)
            elapsed_seconds = time.monotonic() - started
            logfire.info(
                "gen_reply image delivered",
                message_id=message.id,
                model=toolkit.runtime_models.image_model.name,
                elapsed_seconds=elapsed_seconds,
            )

        # Inline base64 keeps the image persona reply provider-agnostic.
        await self.answer.stream_media_persona_reply(
            reply=reply,
            context_task=context_task,
            system_prompt=IMAGE_REPLY_PROMPT,
            focus_part=ResponseInputImageParam(
                image_url=to_data_uri(data=image_bytes), detail="auto", type="input_image"
            ),
            media_noun="image",
            media_model=toolkit.runtime_models.image_model.name,
            question=user_prompt,
            answer_seconds=elapsed_seconds,
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
                # Check for a source video first (current, then replied-to) before fetching images.
                source_video = await toolkit.input_builder.get_turn_video_source(
                    message=message, replied_to=replied_to
                )
                # Prompt refinement is best-effort; failures past here belong to video_model.
                dispatched_model.set(toolkit.runtime_models.video_model.name)
                if source_video is not None:
                    # task="edit" skips refinement (director sees images only) and drops reference
                    # images (omni takes one video input).
                    video_bytes = await toolkit.video_generator.render(
                        prompt=user_prompt, reference_image_sources=[], source_video=source_video
                    )
                else:
                    # Cap reference images before refinement so the director sees the exact frames
                    # passed to render.
                    images = (
                        await toolkit.input_builder.get_turn_image_sources(
                            message=message, replied_to=replied_to
                        )
                    )[:MAX_VIDEO_REFERENCE_IMAGES]
                    conversation_context = (
                        await self._load_refine_context()
                        if self.answer.config.video_refine_prompt_enabled
                        else None
                    )
                    refined_prompt = await toolkit.prompt_generator.refine(
                        user_prompt=user_prompt,
                        instructions=VIDEO_PROMPT,
                        end_user_id=message.author.name,
                        enabled=self.answer.config.video_refine_prompt_enabled,
                        image_bytes_list=[loaded.data for loaded in images] or None,
                        conversation_context=conversation_context,
                    )
                    video_bytes = await toolkit.video_generator.render(
                        prompt=refined_prompt, reference_image_sources=images
                    )
            reply = await self._deliver(data=video_bytes, filename=INLINE_VIDEO_FILENAME)
            elapsed_seconds = time.monotonic() - started
            logfire.info(
                "gen_reply video delivered",
                message_id=message.id,
                model=toolkit.runtime_models.video_model.name,
                total_elapsed_seconds=elapsed_seconds,
                bytes=len(video_bytes),
            )

        await self._reply_about_video(
            reply=reply,
            video_bytes=video_bytes,
            context_task=context_task,
            media_model=toolkit.runtime_models.video_model.name,
            question=user_prompt,
            answer_seconds=elapsed_seconds,
        )

    async def _reply_about_video(  # noqa: PLR0913 -- video upload + persona reply inputs
        self,
        reply: Message | None,
        video_bytes: bytes,
        context_task: asyncio.Task[ReplyContext],
        media_model: str = "",
        question: str = "",
        answer_seconds: float | None = None,
    ) -> None:
        """Uploads `video_bytes` for a persona reply, draining `context_task` on upload failure."""
        video_part = await upload_as_input_file(
            client=self.answer.toolkit.gemini_client,
            source=video_bytes,
            mime_type="video/mp4",
            filename=INLINE_VIDEO_FILENAME,
            timeout_seconds=GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS,
            file_api_enabled=self.answer.config.file_api_enabled,
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
            media_model=media_model,
            question=question,
            answer_seconds=answer_seconds,
        )
