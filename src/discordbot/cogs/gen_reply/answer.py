"""The answer turn: the request one reply streams from, and the memory it schedules after.

`stream_answer` streams the QA reply, carrying the capability reference, inline markers, and
memory write scheduling. `stream_media_persona_reply` streams the short persona follow-up onto
media delivered by the `IMAGE` and `VIDEO` routes, without scheduling memory or owning a status
surface.
"""

from typing import Literal, cast
import asyncio
import contextlib
from collections.abc import Callable, Awaitable, AsyncIterator

import logfire
from nextcord import Message, HTTPException, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.memory import MemoryWriteSummary
from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.utils.discord_errors import is_reply_target_gone
from discordbot.utils.llm_transcript import render_author_identity, render_server_identity
from discordbot.utils.media_delivery import MediaItem, MediaDeliveryPlanner
from discordbot.services.memory.store import user_scope, server_scope
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.prompts import (
    MUSIC_INSTRUCTION,
    VIDEO_INSTRUCTION,
    VOICE_INSTRUCTION,
    INLINE_IMAGE_INSTRUCTION,
    DEEP_RESEARCH_INSTRUCTION,
    REQUEST_TIME_CONTEXT_PROMPT,
    REQUEST_LOCATION_CONTEXT_PROMPT,
)
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.services.memory.writer import (
    user_subject,
    server_subject,
    target_centered_memory_messages,
)
from discordbot.cogs.gen_reply.streaming import ResponseStreamer, stream_answer_with_retry
from discordbot.services.memory.pipeline import schedule_memory_update
from discordbot.utils.channel_visibility import channel_is_public
from discordbot.cogs.gen_reply.turn_state import dispatched_model
from discordbot.cogs.gen_reply.interactions import (
    to_interactions_input,
    create_interactions_answer_stream,
)
from discordbot.cogs.gen_reply.status_marks import (
    YOUTUBE_EMOJI,
    MEMORY_WRITE_EMOJI,
    MEMORY_FORGET_EMOJI,
)
from discordbot.cogs.gen_reply.research_bridge import maybe_launch_research


def build_runtime_instructions(system_prompt: str, message: Message, guild_id: int | None) -> str:
    """Returns `system_prompt` with per-request conversation-location and time context appended.

    Both blocks sit at the end (time last) so the static prefix remains cacheable, and
    `guild_id` comes from `TurnSurface` rather than `message.guild` (which is None on `/ask`).

    Args:
        system_prompt (str): Base developer instructions for the turn.
        message (Message): Source message whose creation timestamp anchors the request time.
        guild_id (int | None): Guild id from the turn surface, or None in a direct message.
    """
    message_created_at_asia_taipei = message.created_at.astimezone(tz=TAIWAN_TIMEZONE)
    request_time_context = REQUEST_TIME_CONTEXT_PROMPT.format(
        message_created_at_asia_taipei=message_created_at_asia_taipei.isoformat(timespec="seconds")
    ).strip()
    if guild_id is not None:
        # Deliberately id-only: the guild name is owner-controlled text and this block
        # rides the developer-authority `instructions` parameter, so embedding it would
        # hand a server owner an instruction-injection surface.
        conversation_location = f"a Discord server (guild id {guild_id})"
    else:
        conversation_location = "a Discord direct message (DM)"
    request_location_context = REQUEST_LOCATION_CONTEXT_PROMPT.format(
        conversation_location=conversation_location
    ).strip()
    return f"{system_prompt}\n\n{request_location_context}\n\n{request_time_context}"


def count_media_parts(answer_input: ResponseInputParam) -> int:
    """Returns the number of `input_file` and `input_image` parts in `answer_input`.

    Args:
        answer_input (ResponseInputParam): Assembled input items for the answer request.
    """
    total = 0
    for item in answer_input:
        content = cast("EasyInputMessageParam", item).get("content", "")
        if isinstance(content, str):
            continue
        total += sum(1 for part in content if part["type"] in ("input_file", "input_image"))
    return total


def memory_report_for(
    streamer: ResponseStreamer,
) -> Callable[[MemoryWriteSummary], Awaitable[None]]:
    """Returns the callback that updates `streamer`'s memory note after review completes.

    `source_only` observations are counted rather than quoted in the channel, and an empty
    summary still replaces the pending `正在整理記憶⋯` placeholder.

    Args:
        streamer (ResponseStreamer): Streamer that owns the reply message and footer.
    """

    async def report(summary: MemoryWriteSummary) -> None:
        """Replaces the reply's memory note with what this turn actually recorded."""
        lines: list[str] = []
        parts: list[str] = []
        if summary.remembered:
            parts.append("、".join(summary.remembered))
        if summary.private:
            # Counted, never quoted, and phrased as its own clause: "N 則私下的" read as a
            # fragment of whatever preceded it, and had no subject at all on a turn that
            # recorded nothing else.
            parts.append(f"另外私下記了 {summary.private} 則")
        if parts:
            lines.append(f"-# {MEMORY_WRITE_EMOJI} 記下了 {'；'.join(parts)}")
        if summary.forgotten:
            lines.append(f"-# {MEMORY_FORGET_EMOJI} 不再記得 {'、'.join(summary.forgotten)}")
        if not lines:
            lines.append(f"-# {MEMORY_WRITE_EMOJI} 這次沒有記下什麼")
        await streamer.set_memory_note(line="\n".join(lines))

    return report


class AnswerTurn(BaseModel):
    """Streams one reply's prose, on the QA path or onto delivered media."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: LLMConfig = Field(
        ..., description="Runtime LLM config, read for the inline-marker kill-switches."
    )
    media_delivery: MediaDeliveryPlanner = Field(
        ..., description="Attach-vs-host-vs-drop planner handed to the streamer."
    )
    toolkit: ReplyToolkit = Field(
        ..., description="The reply toolkit: the bot, its clients, generators and model catalog."
    )
    surface: TurnSurface = Field(
        ..., description="Where this turn's replies go, and which guild it is really happening in."
    )

    @property
    def message(self) -> Message:
        """Returns the message being answered, read off `self.surface`."""
        return self.surface.message

    async def stream_media_persona_reply(
        self,
        reply: Message | None,
        context_task: asyncio.Task[ReplyContext],
        system_prompt: str,
        focus_part: ResponseInputFileParam | ResponseInputImageParam,
        media_noun: str,
    ) -> None:
        """Streams a best-effort persona follow-up onto an already-delivered image or video.

        Awaits `context_task`, appends `focus_part` as the focus, and edits `reply` (or a fresh
        reply when the media was hosted as a separate URL). Swallows any error and deletes an
        unstarted hosted-case base message so no empty ping is orphaned.

        Args:
            reply (Message | None): Delivered attachment message, or None when hosted as a URL.
            context_task (asyncio.Task[ReplyContext]): Speculative context build task.
            system_prompt (str): Persona instructions for the media route.
            focus_part (ResponseInputFileParam | ResponseInputImageParam): Uploaded or inline
                part representing the generated media.
            media_noun (str): Media label (`"image"` or `"video"`) for the prompt and span.
        """
        model = self.toolkit.runtime_models.fast_model
        base: Message | None = None
        streamer: ResponseStreamer | None = None
        try:
            context = await context_task
            base = await self.persona_base_reply(reply=reply)
            response_input = context.persona_input()
            # The generated media is the focus, appended last right after the request it answers.
            response_input.append(
                EasyInputMessageParam(
                    role="user",
                    content=[
                        ResponseInputTextParam(
                            text=(
                                f"This is the {media_noun} you just made for them in response "
                                "to the request above. Reply to them about it."
                            ),
                            type="input_text",
                        ),
                        focus_part,
                    ],
                )
            )
            streamer = ResponseStreamer(
                message=self.message,
                surface=self.surface,
                reply=base,
                # This streamer renders onto the delivered media message itself, so no notice
                # belonging to the turn -- a retry, the failure embed -- may touch it.
                carries_turn_notices=False,
                memory_lookups=context.memory_credits,
                model_effort=model.effort or "",
                media_delivery=self.media_delivery,
            )
            with logfire.span(
                f"gen_reply {media_noun} reply", model=model.name, message_id=self.message.id
            ):

                async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
                    """Issues the persona-reply request; called again per retry attempt."""
                    return await self.toolkit.openai_client.responses.create(
                        model=model.name,
                        instructions=build_runtime_instructions(
                            system_prompt=system_prompt,
                            message=self.message,
                            guild_id=self.surface.guild_id,
                        ),
                        input=response_input,
                        reasoning=model.reasoning,
                        stream=True,
                        service_tier="auto",
                        extra_headers={"x-litellm-end-user-id": self.message.author.name},
                    )

                persona_reply = await stream_answer_with_retry(
                    streamer=streamer, open_stream=open_stream, message_id=self.message.id
                )
            # The media routes' persona reply is the only text this turn produced, so it is what
            # a `/ask` conversation has to carry forward; without it the next turn would see the
            # request for a picture and no sign that one was ever made.
            await self.surface.record_turn(answer=persona_reply)
        except Exception as exc:
            logfire.warn(
                "Media persona reply failed; leaving the delivered media without a reply",
                media=media_noun,
                message_id=self.message.id,
                model=model.name,
                error_type=type(exc).__name__,
                _exc_info=True,
            )
            # A fresh hosted-case base (reply was None) that never received content is a bare ping;
            # delete it so a failed persona reply leaves no orphan. A native media message
            # (reply is not None) is the deliverable itself and is always kept. Reads
            # `content_ever_started`, not `content_started`: a retry clears the latter, so text an
            # earlier attempt already wrote here would otherwise read as a bare ping and be deleted.
            if (
                reply is None
                and base is not None
                and (streamer is None or not streamer.content_ever_started)
            ):
                with contextlib.suppress(Exception):
                    await base.delete()

    async def persona_base_reply(self, reply: Message | None) -> Message:
        """Returns `reply` when present, or sends a fresh non-pinging reply for a hosted media URL.

        Args:
            reply (Message | None): Delivered media attachment message, or None when hosted.
        """
        if reply is not None:
            return reply
        return await self.reply_or_send(
            content=self.message.author.mention, allowed_mentions=AllowedMentions.none()
        )

    async def reply_or_send(
        self,
        content: str,
        media: MediaItem | None = None,
        allowed_mentions: AllowedMentions | None = None,
    ) -> Message:
        """Returns a reply to the source message, or an unparented message if it was deleted.

        Args:
            content (str): Message text to send.
            media (MediaItem | None): Optional attachment item, converted to a fresh `File` per attempt.
            allowed_mentions (AllowedMentions | None): Mention policy for the sent message.

        Raises:
            HTTPException: Sending failed for a reason other than a deleted reply target.
        """
        try:
            return await self.surface.send(
                content=content,
                file=media.to_file() if media else None,
                allowed_mentions=allowed_mentions,
            )
        except HTTPException as exc:
            if not is_reply_target_gone(error=exc):
                raise
            logfire.info(
                "Source message deleted before the media reply; sending unparented",
                message_id=self.message.id,
            )
            return await self.surface.send_unparented(
                content=content,
                file=media.to_file() if media else None,
                allowed_mentions=allowed_mentions,
            )

    async def stream_answer(
        self,
        system_prompt: str,
        context: ReplyContext,
        effort: Literal["low", "high"] = "high",
        allow_research: bool = False,
        yt_url: str | None = None,
    ) -> None:
        """Streams the QA answer from `context`, launches any research brief, and schedules memory updates.

        Offers each inline media or `<deep-research>` marker only when its generator and config
        switch are active, and swaps the answer request onto the Gemini Interactions API when
        `yt_url` is set and the Gemini video gates pass.

        Args:
            system_prompt (str): Base QA system prompt before marker instructions are applied.
            context (ReplyContext): Pre-built conversation, link, and memory context.
            effort (Literal["low", "high"]): Reasoning effort selected by the router.
            allow_research (bool): Whether the channel supports opening a deep-research thread.
            yt_url (str | None): YouTube URL to watch via the Interactions API, or None.
        """
        toolkit = self.toolkit
        voice_generator = toolkit.voice_generator if self.config.inline_voice_enabled else None
        image_generator = toolkit.image_generator if self.config.inline_image_enabled else None
        music_generator = toolkit.music_generator if self.config.music_available else None
        video_generator = toolkit.video_generator if self.config.video_available else None
        # Only advertise an inline marker when its renderer is actually active; with it disabled
        # the streamer would strip the block and produce nothing, silently dropping the request
        # from the reply, so a disabled deployment must not be told about it. The voice bullet
        # sits inside the prompt rather than after it, so it is cut out instead of appended.
        if voice_generator is None:
            system_prompt = system_prompt.replace(VOICE_INSTRUCTION, "")
        for instruction, offered in (
            (INLINE_IMAGE_INSTRUCTION, image_generator is not None),
            (MUSIC_INSTRUCTION, music_generator is not None),
            (VIDEO_INSTRUCTION, video_generator is not None),
        ):
            if offered:
                system_prompt = f"{system_prompt}\n{instruction}"
        research_offered = allow_research and self.config.deep_research_available
        if research_offered:
            system_prompt = f"{system_prompt}\n{DEEP_RESEARCH_INSTRUCTION}"
        slow_model = toolkit.runtime_models.slow_model.model_copy(update={"effort": effort})
        dispatched_model.set(slow_model.name)
        answer_input = context.answer_input(config=self.config)

        # A linked YouTube video the router asked to watch swaps the answer turn onto the Gemini
        # Interactions API: the Responses bridge cannot make Gemini watch the video, so this is
        # the one backend swap. It is Gemini-only and kill-switchable; otherwise (no video, a
        # non-Gemini answer model, the switch off, or no direct key to swap with) the turn falls
        # back to the Responses path, which never errors. Both feed the same streamer so footer /
        # memory / preview are shared.
        use_interactions = (
            yt_url is not None
            and slow_model.is_gemini
            and self.config.youtube_video_enabled
            and self.config.gemini_key_configured
        )
        backend = "interactions" if use_interactions else "responses"
        if yt_url is not None and not use_interactions:
            # The swap is silent to the user, so without this the log shows a Responses answer to
            # a message the router explicitly asked to watch, with nothing saying which gate said
            # no. The fallback itself is correct; only the reason was unrecoverable.
            logfire.info(
                "gen_reply youtube watch declined; answering on the responses backend",
                message_id=self.message.id,
                reason="model"
                if not slow_model.is_gemini
                else "kill-switch"
                if not self.config.youtube_video_enabled
                else "no-gemini-key",
                model=slow_model.name,
            )
        if use_interactions:
            # Persistent marker (added directly, not via the status chain) so it stays after the
            # chain's final reaction to show the reply was grounded in the watched video. The bot's
            # own application emoji `youtube`, usable as a reaction in any guild the bot is in.
            # Added before the streamer is built: its `created_at` is what the answer latency is
            # measured from, so leaving this REST round trip inside that window would bias the
            # figure against the one backend that pays for it.
            await self.surface.mark(emoji=YOUTUBE_EMOJI)
        streamer = ResponseStreamer(
            message=self.message,
            surface=self.surface,
            memory_lookups=context.memory_credits,
            model_effort=effort,
            backend=backend,
            voice_generator=voice_generator,
            image_generator=image_generator,
            music_generator=music_generator,
            video_generator=video_generator,
            media_delivery=self.media_delivery,
            input_builder=toolkit.input_builder,
        )
        # The one record of what the answer model was actually handed. Everything here is a count
        # or a flag: a reply that behaves as if it never saw an attachment, a memory or a linked
        # post is otherwise indistinguishable in the log from one that had them.
        logfire.info(
            "gen_reply answer dispatch",
            message_id=self.message.id,
            model=slow_model.name,
            backend=backend,
            effort=effort,
            input_blocks=len(answer_input),
            history=len(context.hist_messages),
            reference=len(context.reference_messages),
            link_blocks=len(context.link_blocks),
            media_parts=count_media_parts(answer_input=answer_input),
            server_memory=context.server_memory_block is not None,
            user_memory=context.memory_block is not None,
            tone=context.tone_block is not None,
            # Joined rather than a list: the console exporter renders a list one element per
            # line, and this record fires on every reply.
            markers=",".join(
                name
                for name, offered in (
                    ("voice", voice_generator is not None),
                    ("image", image_generator is not None),
                    ("music", music_generator is not None),
                    ("video", video_generator is not None),
                    ("research", research_offered),
                )
                if offered
            ),
        )
        instructions = build_runtime_instructions(
            system_prompt=system_prompt, message=self.message, guild_id=self.surface.guild_id
        )
        with logfire.span(
            "gen_reply answer", model=slow_model.name, backend=backend, message_id=self.message.id
        ):

            async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
                """Issues the answer request; called again per retry attempt."""
                if use_interactions and yt_url is not None:
                    return create_interactions_answer_stream(
                        client=toolkit.gemini_client,
                        model=slow_model.name,
                        system_instruction=instructions,
                        steps=to_interactions_input(answer_input=answer_input, youtube_url=yt_url),
                        effort=slow_model.effort,
                    )
                return await self.toolkit.openai_client.responses.create(
                    model=slow_model.name,
                    instructions=instructions,
                    input=answer_input,
                    reasoning=slow_model.reasoning,
                    tools=list(slow_model.tools),
                    stream=True,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": self.message.author.name},
                )

            full_reply = await stream_answer_with_retry(
                streamer=streamer, open_stream=open_stream, message_id=self.message.id
            )
        # A <deep-research> brief the answer model emitted launches a research thread. Done after
        # the stream (and its single media edit) so it never touches the reply's attachment edit;
        # best-effort, gated, and a no-op when the feature is off or no brief was emitted.
        if research_offered and streamer.markers.research_brief:
            await maybe_launch_research(
                bot=self.toolkit.bot,
                message=self.message,
                anchor=streamer.reply,
                brief=streamer.markers.research_brief,
            )
        # Recorded before the memory review is scheduled, so a conversation the store is meant to
        # carry survives even if the fire-and-forget review below never lands.
        await self.surface.record_turn(answer=full_reply)
        self._schedule_memory_updates(context=context, full_reply=full_reply, streamer=streamer)

    def _schedule_memory_updates(
        self, context: ReplyContext, full_reply: str, streamer: ResponseStreamer
    ) -> None:
        """Schedules the per-author and per-server memory reviews from this turn's markers.

        Target scopes come from the message and surface rather than marker text, keeping the
        compartment boundary structural.
        """
        message = self.message
        memory_message_list = target_centered_memory_messages(
            hist_messages=context.hist_messages,
            reference_messages=context.reference_messages,
            current_message=context.current_message,
            target_user_id=message.author.id,
        )
        # The second subject line names where this conversation happened (guild id
        # or DM); it survives the memory_job round-trip so the pipeline can stamp
        # each observation's source deterministically. Off the surface rather than
        # `message.guild`, which is None on the `/ask` route even in a server: stamping
        # `dm` there would file a server conversation's `source_only` observations —
        # roughly half of them — in the user's private DM compartment, where the read
        # side of that same conversation would never look for them again.
        schedule_memory_update(
            scope=user_scope(user_id=message.author.id),
            subject=user_subject(user_id=message.author.id, guild_id=self.surface.guild_id),
            message_list=memory_message_list,
            full_reply=full_reply,
            writer=self.toolkit.memory_writer,
            identity=render_author_identity(
                display_name=message.author.display_name,
                username=message.author.name,
                user_id=message.author.id,
            ),
            remember_notes=tuple(streamer.markers.memory_notes),
            forget_notes=tuple(streamer.markers.forget_notes),
            report=memory_report_for(streamer=streamer),
        )
        # Server memory learns community-level signal from the whole conversation (no
        # target-centering, since every message is server context). Skipped for DMs and for
        # channels not visible to `@everyone`, so private / restricted-channel content never
        # enters the server-wide memory any member can read. Those two gates are invisible to
        # the answer model, which is why a `<write-server-memory>` note written in a DM or a
        # restricted channel is dropped here without a word: the note arrives exactly as any
        # other would, and the channel decides, not the model. A `/ask` turn always takes the
        # first branch, since its message carries no guild, and that is the decision rather than
        # an accident: the bot is not in the channel, so it cannot tell whether `@everyone` can
        # read it, and a community memory it may write but never read back goes stale unseen.
        if message.guild is None:
            return
        if not channel_is_public(guild=message.guild, channel=message.channel):
            return
        schedule_memory_update(
            scope=server_scope(server_id=message.guild.id),
            subject=server_subject(server_id=message.guild.id),
            message_list=context.message_list,
            full_reply=full_reply,
            writer=self.toolkit.memory_writer,
            identity=render_server_identity(
                server_name=message.guild.name, server_id=message.guild.id
            ),
            remember_notes=tuple(streamer.markers.server_memory_notes),
        )
