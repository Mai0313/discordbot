"""The answer turn: the request one reply streams from, and the memory it schedules after.

`stream_answer` streams the QA reply, carrying the capability reference, inline markers, and
memory write scheduling. `stream_media_persona_reply` streams the short persona follow-up onto
media delivered by the `IMAGE` and `VIDEO` routes, without scheduling memory or owning a status
surface.
"""

from typing import Literal, cast
import asyncio
import secrets
import contextlib
from collections.abc import Callable, Sequence, Awaitable, AsyncIterator

import logfire
from nextcord import Message, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict
from openai.types.responses import Response, ResponseStreamEvent
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.memory import MemoryWriteSummary
from discordbot.typings.models import ModelSettings, SlowModelSettings
from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.utils.discord_errors import report_discord_failure
from discordbot.utils.llm_transcript import (
    USAGE_FOOTER_RE,
    render_author_identity,
    render_server_identity,
)
from discordbot.utils.media_delivery import MediaItem, MediaDeliveryPlanner
from discordbot.services.memory.store import user_scope, server_scope
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.prompts import (
    MUSIC_INSTRUCTION,
    VIDEO_INSTRUCTION,
    INLINE_IMAGE_INSTRUCTION,
    DEEP_RESEARCH_INSTRUCTION,
    REQUEST_TIME_CONTEXT_PROMPT,
    REQUEST_LOCATION_CONTEXT_PROMPT,
    format_identity_model,
)
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.services.memory.writer import (
    user_subject,
    server_subject,
    target_centered_memory_messages,
)
from discordbot.cogs.gen_reply.streaming import ResponseStreamer, stream_answer
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
from discordbot.cogs.gen_reply.non_streaming import (
    complete_responses_answer,
    create_interactions_answer,
    complete_interactions_answer,
)
from discordbot.cogs.gen_reply.feedback_store import (
    ReplyRoute,
    record_reply_turn,
    update_reply_message_ids,
)
from discordbot.cogs.gen_reply.feedback_views import ReplyFeedbackView
from discordbot.cogs.gen_reply.research_bridge import maybe_launch_research


class RegenOptions(BaseModel):
    """Parameters carried into `ReplyPipeline` and `AnswerTurn` when regenerating an earlier QA reply."""

    regen_of_reply_message_id: int = Field(
        ..., description="Discord message id of the reply being replaced."
    )
    regen_count: int = Field(
        ..., description="1-based regeneration depth of the new reply in its chain."
    )
    excluded_models: frozenset[str] = Field(
        default_factory=frozenset,
        description="Models already tried earlier in this regen chain, excluded when others remain.",
    )
    forced_effort: Literal["low", "medium", "high"] | None = Field(
        default=None, description="Reasoning effort preserved from the replaced turn, when valid."
    )


def select_slow_model(
    candidates: Sequence[SlowModelSettings], excluded_models: frozenset[str] = frozenset()
) -> SlowModelSettings:
    """Draws one QA candidate uniformly, excluding `excluded_models` while untried models remain.

    Args:
        candidates (Sequence[SlowModelSettings]): Non-empty pool of configured QA candidates.
        excluded_models (frozenset[str]): Model names already tried earlier in a regen chain.
    """
    remaining = [candidate for candidate in candidates if candidate.name not in excluded_models]
    pool = remaining or list(candidates)
    if len(pool) == 1:
        return pool[0]
    return secrets.choice(seq=pool)


def max_regens_for(candidates: Sequence[ModelSettings]) -> int:
    """Returns the maximum number of regenerations allowed for one QA reply chain."""
    return max(1, len(candidates) - 1)


def build_runtime_instructions(system_prompt: str, message: Message, guild_id: int | None) -> str:
    """Returns `system_prompt` with per-request conversation-location and time context appended.

    Appends location then time at the end to keep the static prefix cacheable.

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
        # Id-only: guild names are owner-controlled and `instructions` has developer authority.
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
            # Counted, never quoted, and phrased as a standalone clause when nothing else recorded.
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

    async def _record_feedback_turn(  # noqa: PLR0913 -- maps directly to reply_turn columns
        self,
        streamer: ResponseStreamer,
        route: ReplyRoute,
        model: str,
        effort: str,
        backend: str,
        question: str,
        regen_of_reply_message_id: int | None = None,
        answer_seconds: float | None = None,
    ) -> None:
        """Persists a best-effort `reply_turn` snapshot for feedback buttons when enabled."""
        if not self.config.feedback_enabled or streamer.last_reply is None:
            return
        recorded_question = question or self.message.content
        if not recorded_question and (self.message.attachments or self.message.stickers):
            recorded_question = "[attachment]"
        recorded_answer = USAGE_FOOTER_RE.sub(repl="", string=streamer.final_answer_text()).strip()
        chunk_ids = [msg.id for msg in streamer.reply_messages] or [streamer.last_reply.id]
        try:
            await record_reply_turn(
                reply_message_id=streamer.last_reply.id,
                source_message_id=self.message.id,
                guild_id=self.surface.guild_id,
                channel_id=self.message.channel.id,
                author_id=self.message.author.id,
                author_name=self.message.author.name,
                author_display_name=self.message.author.display_name,
                surface="ask" if self.surface.interaction is not None else "gateway",
                route=route,
                model=model,
                effort=effort,
                backend=backend,
                question=recorded_question,
                answer=recorded_answer,
                reply_message_ids=chunk_ids,
                regen_of_reply_message_id=regen_of_reply_message_id,
                answer_seconds=answer_seconds,
            )
        except Exception as exc:
            # Broad on purpose: best-effort feedback snapshot must never fail a delivered reply.
            logfire.warn(
                "Failed to record reply_turn snapshot for feedback",
                message_id=self.message.id,
                reply_message_id=streamer.last_reply.id,
                route=route,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    def _feedback_hook(  # noqa: PLR0913 -- carries per-turn metadata into the streamer callback
        self,
        route: ReplyRoute,
        default_model: str,
        effort: str,
        backend: str,
        question: str,
        model_override: str | None = None,
        regen_of_reply_message_id: int | None = None,
        answer_seconds: float | None = None,
    ) -> Callable[[ResponseStreamer], Awaitable[None]] | None:
        """Returns the `on_final_message` hook when feedback is enabled, else None."""
        if not self.config.feedback_enabled:
            return None

        async def record(active: ResponseStreamer) -> None:
            resolved_seconds = (
                answer_seconds
                if answer_seconds is not None
                else (active.answer_seconds if route == "QA" else None)
            )
            await self._record_feedback_turn(
                streamer=active,
                route=route,
                model=model_override or active.model_name or default_model,
                effort=effort,
                backend=backend,
                question=question,
                regen_of_reply_message_id=regen_of_reply_message_id,
                answer_seconds=resolved_seconds,
            )

        return record

    async def stream_media_persona_reply(  # noqa: PLR0913 -- media route passes prompt, pixels, noun, model, question, duration
        self,
        reply: Message | None,
        context_task: asyncio.Task[ReplyContext],
        system_prompt: str,
        focus_part: ResponseInputFileParam | ResponseInputImageParam,
        media_noun: str,
        media_model: str = "",
        question: str = "",
        answer_seconds: float | None = None,
    ) -> None:
        """Streams a best-effort persona follow-up onto an already-delivered image or video.

        Awaits `context_task`, appends `focus_part` as the focus, and edits `reply` (or a fresh
        reply when the media was hosted as a separate URL). Swallows errors and deletes an
        unstarted hosted-case base message.

        Args:
            reply (Message | None): Delivered attachment message, or None when hosted as a URL.
            context_task (asyncio.Task[ReplyContext]): Speculative context build task.
            system_prompt (str): Persona instructions for the media route.
            focus_part (ResponseInputFileParam | ResponseInputImageParam): Uploaded or inline
                part representing the generated media.
            media_noun (str): Media label (`"image"` or `"video"`) for the prompt and span.
            media_model (str): Generator model name recorded on the feedback snapshot.
            question (str): Mention-stripped user prompt recorded on the feedback snapshot.
            answer_seconds (float | None): Media generation and delivery duration in seconds.
        """
        model = self.toolkit.runtime_models.fast_model
        base: Message | None = None
        streamer: ResponseStreamer | None = None
        try:
            context = await context_task
            base = await self.persona_base_reply(reply=reply)
            response_input = context.persona_input()
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

            streaming = self.config.answer_streaming_enabled
            streamer = ResponseStreamer(
                message=self.message,
                surface=self.surface,
                reply=base,
                # Renders onto the delivered media message; turn notices must not edit it.
                carries_turn_notices=False,
                memory_lookups=context.memory_credits,
                model_effort=model.effort or "",
                streaming=streaming,
                media_delivery=self.media_delivery,
                feedback_view=(
                    ReplyFeedbackView(allow_regen=False) if self.config.feedback_enabled else None
                ),
                on_final_message=self._feedback_hook(
                    route="IMAGE" if media_noun == "image" else "VIDEO",
                    default_model=model.name,
                    effort="",
                    backend="interactions" if media_noun == "video" else "responses",
                    question=question,
                    model_override=media_model or None,
                    answer_seconds=answer_seconds,
                ),
            )
            instructions = build_runtime_instructions(
                system_prompt=format_identity_model(
                    system_prompt=system_prompt, model_label=model.name
                ),
                message=self.message,
                guild_id=self.surface.guild_id,
            )
            with logfire.span(
                f"gen_reply {media_noun} reply", model=model.name, message_id=self.message.id
            ):
                if streaming:

                    async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
                        """Issues the persona-reply request inside the stream helper."""
                        return await self.toolkit.openai_client.responses.create(
                            model=model.name,
                            instructions=instructions,
                            input=response_input,
                            reasoning=model.reasoning,
                            stream=True,
                            service_tier="auto",
                            extra_headers={"x-litellm-end-user-id": self.message.author.name},
                        )

                    persona_reply = await stream_answer(streamer=streamer, open_stream=open_stream)
                else:

                    async def fetch_response() -> Response:
                        """Issues the non-streaming persona-reply request."""
                        return await self.toolkit.openai_client.responses.create(
                            model=model.name,
                            instructions=instructions,
                            input=response_input,
                            reasoning=model.reasoning,
                            stream=False,
                            service_tier="auto",
                            extra_headers={"x-litellm-end-user-id": self.message.author.name},
                        )

                    persona_reply = await complete_responses_answer(
                        streamer=streamer, fetch=fetch_response
                    )
            # Record the persona reply so `/ask` history retains evidence the media was delivered.
            await self.surface.record_turn(answer=persona_reply)
        except Exception as exc:
            report_discord_failure(
                error=exc,
                not_found_message=(
                    "Media persona reply failed; leaving the delivered media without a reply"
                ),
                not_found_level="warn",
                not_found_exc_info=True,
                forbidden_message=(
                    "Media persona reply failed; leaving the delivered media without a reply"
                ),
                other_message=(
                    "Media persona reply failed; leaving the delivered media without a reply"
                ),
                media=media_noun,
                message_id=self.message.id,
                model=model.name,
            )
            # Delete an unstarted hosted-case base ping so it never lingers without reply text.
            if (
                reply is None
                and base is not None
                and (streamer is None or not streamer.content_started)
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
        return await self.surface.send_or_unparented(
            content=content,
            file=media.to_file if media else None,
            allowed_mentions=allowed_mentions,
            unparented_log="Source message deleted before the media reply; sending unparented",
        )

    def _resolve_qa_model(
        self,
        effort: Literal["low", "medium", "high"] | None,
        system_prompt: str,
        slow_model: SlowModelSettings | None = None,
        excluded_models: frozenset[str] = frozenset(),
    ) -> tuple[SlowModelSettings, str, str]:
        """Returns `(slow_model, system_prompt, model_display_name)` for one QA turn.

        Uses `slow_model` when already selected by the pipeline, otherwise samples one from
        `slow_candidates` (excluding `excluded_models` while untried models remain). Substitutes
        the visible identity label (`slow_display_name` when configured, else the real model name)
        into `system_prompt` while leaving pricing, logs, and `reply_turn.model` on the real model.
        """
        runtime_models = self.toolkit.runtime_models
        chosen = (
            slow_model
            if slow_model is not None
            else select_slow_model(
                candidates=runtime_models.slow_candidates, excluded_models=excluded_models
            )
        )
        resolved_effort: Literal["low", "medium", "high"] = (
            effort if effort is not None else chosen.effort
        )
        display_name = runtime_models.slow_display_name
        formatted_prompt = format_identity_model(
            system_prompt=system_prompt, model_label=display_name or chosen.name
        )
        return (
            chosen.model_copy(update={"effort": resolved_effort}),
            formatted_prompt,
            display_name,
        )

    def _interactions_decline_reason(self, slow_model: ModelSettings) -> str | None:
        """Returns why the Interactions YouTube backend cannot run for `slow_model`, or None."""
        if not slow_model.is_gemini:
            return "model"
        if not self.config.gemini_key_configured:
            return "no-gemini-key"
        return None

    async def _deliver_qa_answer(
        self,
        streamer: ResponseStreamer,
        slow_model: ModelSettings,
        instructions: str,
        answer_input: ResponseInputParam,
        yt_url: str | None,
    ) -> str:
        """Runs the QA answer request in streaming or non-streaming mode and finalizes `streamer`."""
        if self.config.answer_streaming_enabled:

            async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
                """Issues the answer request inside the `/ask` delivery window."""
                if yt_url is not None:
                    return create_interactions_answer_stream(
                        client=self.toolkit.gemini_client,
                        model=slow_model.name,
                        system_instruction=instructions,
                        steps=to_interactions_input(answer_input=answer_input, youtube_url=yt_url),
                        effort=slow_model.effort,
                        tool_use=streamer.tool_use,
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

            return await stream_answer(streamer=streamer, open_stream=open_stream)
        if yt_url is not None:

            async def fetch_interaction() -> object:
                """Issues the non-streaming Interactions answer inside the delivery window."""
                return await create_interactions_answer(
                    client=self.toolkit.gemini_client,
                    model=slow_model.name,
                    system_instruction=instructions,
                    steps=to_interactions_input(answer_input=answer_input, youtube_url=yt_url),
                    effort=slow_model.effort,
                )

            return await complete_interactions_answer(streamer=streamer, fetch=fetch_interaction)

        async def fetch_response() -> Response:
            """Issues the non-streaming Responses answer inside the delivery window."""
            return await self.toolkit.openai_client.responses.create(
                model=slow_model.name,
                instructions=instructions,
                input=answer_input,
                reasoning=slow_model.reasoning,
                tools=list(slow_model.tools),
                stream=False,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": self.message.author.name},
            )

        return await complete_responses_answer(streamer=streamer, fetch=fetch_response)

    async def stream_answer(  # noqa: PLR0913 -- QA turn parameters from pipeline
        self,
        system_prompt: str,
        context: ReplyContext,
        effort: Literal["low", "medium", "high"] | None = None,
        allow_research: bool = False,
        yt_url: str | None = None,
        question: str = "",
        regen: RegenOptions | None = None,
        slow_model: SlowModelSettings | None = None,
    ) -> int | None:
        """Streams the QA answer from `context`, launches any research brief, and schedules memory updates.

        Offers direct-to-Google inline markers (`<generate-music>`, `<generate-video>`, and
        `<deep-research>`) only when `GEMINI_API_KEY` is configured, and swaps the answer request
        onto the Gemini Interactions API when `yt_url` is set and the Gemini video gates pass.

        Args:
            system_prompt (str): Base QA system prompt before marker instructions are applied.
            context (ReplyContext): Pre-built conversation, link, and memory context.
            effort (Literal["low", "medium", "high"] | None): Reasoning effort selected by the
                router, or None to fall back to the selected candidate's default effort.
            allow_research (bool): Whether the channel supports opening a deep-research thread.
            yt_url (str | None): YouTube URL to watch via the Interactions API, or None.
            question (str): Mention-stripped user prompt recorded on the feedback snapshot.
            regen (RegenOptions | None): Regeneration depth, parent reply id, and excluded models.
            slow_model (SlowModelSettings | None): Preselected QA candidate for this turn, or
                None to sample from `runtime_models.slow_candidates`.

        Returns:
            int | None: Discord message id of the final landed reply chunk, or None if not sent.
        """
        toolkit = self.toolkit
        voice_generator = toolkit.voice_generator
        image_generator = toolkit.image_generator
        music_generator = toolkit.music_generator if self.config.gemini_key_configured else None
        video_generator = toolkit.video_generator if self.config.gemini_key_configured else None
        research_offered = allow_research and self.config.gemini_key_configured
        offered_markers = (
            ("voice", None, True),
            ("image", INLINE_IMAGE_INSTRUCTION, True),
            ("music", MUSIC_INSTRUCTION, music_generator is not None),
            ("video", VIDEO_INSTRUCTION, video_generator is not None),
            ("research", DEEP_RESEARCH_INSTRUCTION, research_offered),
        )
        # Disabled renderers still strip tags silently; optional media instructions are appended
        # only when their backend is configured.
        for _name, instruction, offered in offered_markers:
            if offered and instruction is not None:
                system_prompt = f"{system_prompt}\n{instruction}"
        regen_count = regen.regen_count if regen is not None else 0
        regen_of_reply_message_id = regen.regen_of_reply_message_id if regen is not None else None
        excluded_models = regen.excluded_models if regen is not None else frozenset()
        resolved_slow_model, system_prompt, model_display_name = self._resolve_qa_model(
            effort=effort,
            system_prompt=system_prompt,
            slow_model=slow_model,
            excluded_models=excluded_models,
        )
        resolved_effort = resolved_slow_model.effort
        dispatched_model.set(resolved_slow_model.name)
        answer_input = context.answer_input(config=self.config)

        decline_reason = (
            self._interactions_decline_reason(slow_model=resolved_slow_model)
            if yt_url is not None
            else None
        )
        use_interactions = yt_url is not None and decline_reason is None
        backend: Literal["responses", "interactions"] = (
            "interactions" if use_interactions else "responses"
        )
        if decline_reason is not None:
            logfire.info(
                "gen_reply youtube watch declined; answering on the responses backend",
                message_id=self.message.id,
                reason=decline_reason,
                model=resolved_slow_model.name,
            )
        if use_interactions:
            # Persistent reaction outside the status chain; added before `ResponseStreamer` so
            # this REST round trip does not inflate answer latency measured from `created_at`.
            await self.surface.mark(emoji=YOUTUBE_EMOJI)

        allow_regen = self.surface.interaction is None and regen_count < max_regens_for(
            candidates=toolkit.runtime_models.slow_candidates
        )
        streaming = self.config.answer_streaming_enabled
        streamer = ResponseStreamer(
            message=self.message,
            surface=self.surface,
            memory_lookups=context.memory_credits,
            model_display_name=model_display_name,
            model_effort=resolved_effort,
            backend=backend,
            streaming=streaming,
            voice_generator=voice_generator,
            image_generator=image_generator,
            music_generator=music_generator,
            video_generator=video_generator,
            media_delivery=self.media_delivery,
            input_builder=toolkit.input_builder_for(model=resolved_slow_model),
            feedback_view=(
                ReplyFeedbackView(allow_regen=allow_regen)
                if self.config.feedback_enabled
                else None
            ),
            on_final_message=self._feedback_hook(
                route="QA",
                default_model=resolved_slow_model.name,
                effort=resolved_effort,
                backend=backend,
                question=question,
                regen_of_reply_message_id=regen_of_reply_message_id,
            ),
        )
        logfire.info(
            "gen_reply answer dispatch",
            message_id=self.message.id,
            model=resolved_slow_model.name,
            backend=backend,
            streaming=streaming,
            effort=resolved_effort,
            input_blocks=len(answer_input),
            history=len(context.hist_messages),
            reference=len(context.reference_messages),
            link_blocks=len(context.link_blocks),
            media_parts=count_media_parts(answer_input=answer_input),
            server_memory=context.server_memory_block is not None,
            user_memory=context.memory_block is not None,
            tone=context.tone_block is not None,
            # Joined string avoids multi-line list formatting in the console exporter.
            markers=",".join(name for name, _instruction, offered in offered_markers if offered),
        )
        instructions = build_runtime_instructions(
            system_prompt=system_prompt, message=self.message, guild_id=self.surface.guild_id
        )
        with logfire.span(
            "gen_reply answer",
            model=resolved_slow_model.name,
            backend=backend,
            message_id=self.message.id,
        ):
            full_reply = await self._deliver_qa_answer(
                streamer=streamer,
                slow_model=resolved_slow_model,
                instructions=instructions,
                answer_input=answer_input,
                yt_url=yt_url if use_interactions else None,
            )
        await self._sync_extra_reply_chunk_ids(streamer=streamer)
        # Launch after streaming finishes so thread creation never races the reply's media edit.
        if research_offered and streamer.markers.research_brief:
            await maybe_launch_research(
                bot=self.toolkit.bot,
                message=self.message,
                anchor=streamer.reply,
                brief=streamer.markers.research_brief,
            )
        # Persist the `/ask` turn before scheduling fire-and-forget memory review.
        await self.surface.record_turn(answer=full_reply)
        self._schedule_memory_updates(context=context, full_reply=full_reply, streamer=streamer)
        return streamer.last_reply.id if streamer.last_reply is not None else None

    async def _sync_extra_reply_chunk_ids(self, streamer: ResponseStreamer) -> None:
        """Persists all chunk ids if hosted-link media appended a follow-up after `on_final_message`."""
        if (
            not self.config.feedback_enabled
            or streamer.last_reply is None
            or not streamer.reply_messages
            or streamer.reply_messages[-1].id == streamer.last_reply.id
        ):
            return
        try:
            await update_reply_message_ids(
                reply_message_id=streamer.last_reply.id,
                reply_message_ids=[msg.id for msg in streamer.reply_messages],
            )
        except Exception as exc:
            # Broad on purpose: best-effort chunk-id update must not fail a delivered reply.
            logfire.warn(
                "Failed to update reply_message_ids after hosted-link follow-up",
                message_id=self.message.id,
                reply_message_id=streamer.last_reply.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    def _schedule_memory_updates(
        self, context: ReplyContext, full_reply: str, streamer: ResponseStreamer
    ) -> None:
        """Schedules per-author and per-server memory reviews using message and surface scopes."""
        message = self.message
        memory_message_list = target_centered_memory_messages(
            hist_messages=context.hist_messages,
            reference_messages=context.reference_messages,
            current_message=context.current_message,
            target_user_id=message.author.id,
        )
        # Use `surface.guild_id`, not `message.guild` (None on `/ask`), so source compartments
        # match what reads look up.
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
        # Server memory requires a real guild and `@everyone` visibility so private channels
        # (and `/ask`, where channel visibility cannot be checked) never leak into shared memory.
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
