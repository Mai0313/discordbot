"""The IMAGE and VIDEO routes: generate the media, deliver it, then talk about it.

Both routes share one shape. The media is the deliverable, so its generation stays on the hard
error path; everything after delivery is best-effort and must leave the picture or clip on
screen whatever happens. The persona reply itself lives in `answer.py`, since it is the same act
as any other streamed reply.
"""

import time
import base64
import asyncio

import logfire
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.utils.images import convert_base64_to_data_uri
from discordbot.typings.timeouts import GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS
from discordbot.utils.media_delivery import MediaItem, MediaDeliveryPlanner, upload_limit_for
from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.prompts import (
    IMAGE_PROMPT,
    VIDEO_PROMPT,
    IMAGE_REPLY_PROMPT,
    VIDEO_REPLY_PROMPT,
)
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import GeminiKeyToolkit
from discordbot.typings.context_budgets import MAX_VIDEO_REFERENCE_IMAGES
from discordbot.cogs.gen_reply.files_api import upload_to_files_api
from discordbot.cogs.gen_reply.references import replied_to_message
from discordbot.cogs.gen_reply.turn_state import dispatched_model
from discordbot.cogs.gen_reply.speculation import discard_task


class MediaReplyRoutes(BaseModel):
    """Runs the IMAGE and VIDEO routes for one message.

    Attributes:
        config: Runtime LLM config, read for the two prompt-refine kill-switches.
        media_delivery: Decides whether the generated media attaches or is hosted as a URL.
        toolkit: The leased Gemini key's generators, clients and model catalog.
        message: The message that asked for the media.
        surface: Where the delivered media goes.
        answer: The streamer used for the best-effort persona reply once media is delivered.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SkipValidation[LLMConfig] = Field(
        ..., description="Runtime LLM config, read for the prompt-refine kill-switches."
    )
    media_delivery: MediaDeliveryPlanner = Field(
        ..., description="Decides whether generated media attaches or is hosted as a URL."
    )
    toolkit: GeminiKeyToolkit = Field(
        ..., description="The leased Gemini key's generators, clients and model catalog."
    )
    message: SkipValidation[Message] = Field(
        ..., description="The message that asked for the media."
    )
    surface: TurnSurface = Field(..., description="Where the delivered media goes.")
    answer: AnswerTurn = Field(
        ..., description="Streams the best-effort persona reply about the delivered media."
    )

    async def _deliver(self, *, data: bytes, filename: str) -> Message | None:
        """Delivers generated image/video bytes, hosting a URL when too big to upload natively.

        Returns the delivered media message the persona reply should stream onto, or None when the
        bytes were too big and hosted as a standalone URL reply instead. On None the caller posts
        the persona reply on a fresh non-pinging message (via `AnswerTurn.persona_base_reply`) only
        if it proceeds, so the hosted-URL message is never clobbered and no stray persona-base is
        left if the persona reply bails. If hosting is unavailable the native attach is attempted
        anyway, raising on oversize so the route stays on its existing hard-fail error path.
        """
        item = MediaItem(source=data, filename=filename)
        plan = await self.media_delivery.plan(
            items=[item], upload_limit=upload_limit_for(guild=self.message.guild)
        )
        if plan.native:
            return await self.surface.send(
                content=self.message.author.mention, file=plan.native[0].to_file()
            )
        if not plan.hosted_urls:
            # Hosting off/failed: attempt the native attach, which raises on oversize and keeps
            # the route on the outer error path exactly as before.
            return await self.surface.send(
                content=self.message.author.mention, file=item.to_file()
            )
        # Too big to attach: the hosted URL is the deliverable (pings the author once). The persona
        # reply, if it runs, streams onto its own fresh message so it never clobbers this link.
        await self.surface.send(content=f"{self.message.author.mention}\n{plan.hosted_urls[0]}")
        return None

    async def handle_image(
        self, *, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Generates or edits an image, then replies about it in persona.

        The image is delivered first so the user sees it without waiting; the conversational
        reply then streams onto that same message, so the bot answers while holding the image
        it just made rather than coldly describing it. The reply brings in conversation
        history and the selected user memory (never server memory), using the speculative
        `context_task` that built in parallel with the route, awaited only after
        the image is on screen so the context build overlaps generation. Once the image is
        delivered the reply is best-effort: any failure leaves the delivered image untouched.
        """
        message = self.message
        toolkit = self.toolkit
        started = time.monotonic()
        replied_to = replied_to_message(message=message)
        logfire.info(
            "gen_reply image generation start",
            message_id=message.id,
            model=toolkit.runtime_models.image_model.name,
            has_source_images=replied_to is not None,
        )
        try:
            if replied_to is not None:
                own_bytes, ref_bytes = await asyncio.gather(
                    toolkit.input_builder.get_image_source_bytes(message=message),
                    toolkit.input_builder.get_image_source_bytes(message=replied_to),
                )
                image_bytes_list = own_bytes + ref_bytes
            else:
                image_bytes_list = await toolkit.input_builder.get_image_source_bytes(
                    message=message
                )

            # Refine the raw request into a full generation/edit prompt first (best-effort, raw
            # prompt on disable / failure); the source bytes ride along so an edit prompt is
            # grounded in the actual image without a re-download.
            refined_prompt = await toolkit.prompt_generator.refine(
                user_prompt=user_prompt,
                instructions=IMAGE_PROMPT,
                end_user_id=message.author.name,
                enabled=self.config.image_refine_prompt_enabled,
                image_bytes_list=image_bytes_list or None,
            )
            # The director above is best-effort and swallows its own failures, so from here the
            # image model is the only one a failure can be reported against.
            dispatched_model.set(toolkit.runtime_models.image_model.name)
            image_bytes = await toolkit.image_generator.render(
                prompt=refined_prompt,
                end_user_id=message.author.name,
                image_bytes_list=image_bytes_list or None,
            )
            # Send the generated image immediately so the user sees it without waiting on the
            # conversational reply; the reply text streams onto this same message right after.
            reply = await self._deliver(data=image_bytes, filename="generated.png")
            logfire.info(
                "gen_reply image delivered",
                message_id=message.id,
                model=toolkit.runtime_models.image_model.name,
                elapsed_seconds=time.monotonic() - started,
            )
        except Exception:
            # Generation failing IS a real error and stays on the outer error path, but the
            # speculative context must not leak when we bail before consuming it.
            await discard_task(task=context_task, label="prep", message_id=message.id)
            raise

        # The image is already delivered, so from here a failure must never surface as an
        # error: the conversational reply is best-effort and leaves the image untouched. The
        # image rides as inline base64 (provider-agnostic), unlike the video's Files API handle.
        await self.answer.stream_media_persona_reply(
            reply=reply,
            context_task=context_task,
            model=toolkit.runtime_models.fast_model,
            system_prompt=IMAGE_REPLY_PROMPT,
            focus_part=ResponseInputImageParam(
                image_url=convert_base64_to_data_uri(
                    base64_image=base64.b64encode(image_bytes).decode()
                ),
                detail="auto",
                type="input_image",
            ),
            media_noun="image",
            span_name="gen_reply image reply",
        )

    async def handle_video(
        self, *, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Generates a video via the native Gemini (omni) Interactions API, delivers it, then replies.

        Runs direct to Google via `interactions.create`. If the message (or the replied-to message,
        mirroring the IMAGE route) carries a video, omni edits that actual clip in place
        (`task="edit"`, the literal request as the edit instruction, no prompt director); otherwise
        the request is expanded by the prompt director and any images ride as subject reference
        frames (up to `MAX_VIDEO_REFERENCE_IMAGES`). The clip is delivered first; then, best-effort,
        the bot watches the video it just made (uploaded to the Gemini Files API) and streams a
        persona reply onto the same message, mirroring `handle_image` and consuming the
        speculative `ReplyContext` (history + the requester's memory) only after the video is on
        screen so its build overlaps generation.
        """
        message = self.message
        toolkit = self.toolkit
        started = time.monotonic()
        logfire.info(
            "gen_reply video generation start",
            message_id=message.id,
            model=toolkit.runtime_models.video_model.name,
        )
        try:
            replied_to = replied_to_message(message=message)
            source_messages = [message, *([replied_to] if replied_to is not None else [])]
            # Find the source video first, by priority (current message, then replied-to); each
            # message reads at most its first clip. Only when there is no source video do we
            # download reference images, so an edit is never delayed by media it discards.
            source_video: tuple[bytes, str] | None = None
            for source_message in source_messages:
                videos = await toolkit.input_builder.get_video_sources(message=source_message)
                if videos:
                    source_video = videos[0]
                    break
            # Both branches end in the same omni render, and the director the else branch runs
            # first is best-effort, so this is the model a failure past here belongs to.
            dispatched_model.set(toolkit.runtime_models.video_model.name)
            if source_video is not None:
                # A source video is edited in place (task=edit): omni ingests the actual clip, so
                # the prompt is the literal edit instruction. The director is skipped here — it
                # only grounds on image parts (a video-only edit would run it blind) and it sits
                # serially on the time-to-video path; the user's edit request is already specific.
                # omni takes a single input here, so any accompanying reference images are dropped.
                video_bytes = await toolkit.video_generator.render(
                    prompt=user_prompt, reference_image_sources=[], source_video=source_video
                )
            else:
                # No source video: gather the message + replied-to images as subject references,
                # capped to the same set render sends (omni takes a few), so the director grounds
                # on exactly those frames and no unused bytes ride the path.
                image_groups = await asyncio.gather(
                    *(
                        toolkit.input_builder.get_image_sources_with_mime(message=m)
                        for m in source_messages
                    )
                )
                images = [pair for group in image_groups for pair in group][
                    :MAX_VIDEO_REFERENCE_IMAGES
                ]
                # Refine the raw request into a full motion/camera prompt first (best-effort, raw
                # prompt on disable / failure); the reference frames ride along as grounding.
                refined_prompt = await toolkit.prompt_generator.refine(
                    user_prompt=user_prompt,
                    instructions=VIDEO_PROMPT,
                    end_user_id=message.author.name,
                    enabled=self.config.video_refine_prompt_enabled,
                    image_bytes_list=[raw for raw, _ in images] or None,
                )
                video_bytes = await toolkit.video_generator.render(
                    prompt=refined_prompt, reference_image_sources=images
                )
            reply = await self._deliver(data=video_bytes, filename="generated.mp4")
            logfire.info(
                "gen_reply video delivered",
                message_id=message.id,
                model=toolkit.runtime_models.video_model.name,
                total_elapsed_seconds=time.monotonic() - started,
                bytes=len(video_bytes),
            )
        except Exception:
            # Generation failing IS a real error and stays on the outer error path, but the
            # speculative context must not leak when we bail before consuming it.
            await discard_task(task=context_task, label="prep", message_id=message.id)
            raise

        # The video is already delivered, so from here a failure must never surface as an error:
        # the conversational reply is best-effort and leaves the delivered video untouched.
        await self._reply_about_video(
            reply=reply, video_bytes=video_bytes, context_task=context_task
        )

    async def _reply_about_video(
        self,
        *,
        reply: Message | None,
        video_bytes: bytes,
        context_task: asyncio.Task[ReplyContext],
    ) -> None:
        """Best-effort: watches the just-made video and streams a persona reply onto its message.

        Feeds the generated video as an uploaded Files API `input_file` (video cannot be
        inlined), then delegates to the shared media-persona-reply streamer. `reply` is None when
        the clip was hosted as a URL; the persona-base message is only created once the Files API
        upload succeeds, so a failed upload leaves no orphaned message. Any failure leaves the
        delivered video untouched.

        The activation bound is generous because video processing is slower than an image's. The
        reply then references the full `uri` through the proxy; see `files_api` for why a uri and
        not the clip's own URL.
        """
        file_uri = await upload_to_files_api(
            client=self.toolkit.gemini_client,
            source=video_bytes,
            mime_type="video/mp4",
            display_name="generated.mp4",
            timeout_seconds=GENERATED_VIDEO_ACTIVATION_TIMEOUT_SECONDS,
        )
        if file_uri is None:
            await discard_task(task=context_task, label="prep", message_id=self.message.id)
            return
        await self.answer.stream_media_persona_reply(
            reply=reply,
            context_task=context_task,
            model=self.toolkit.runtime_models.fast_model,
            system_prompt=VIDEO_REPLY_PROMPT,
            focus_part=ResponseInputFileParam(type="input_file", file_id=file_uri),
            media_noun="video",
            span_name="gen_reply video reply",
        )
