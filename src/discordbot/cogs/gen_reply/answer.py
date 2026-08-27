"""The answer turn: the request one reply streams from, and the memory it schedules after.

Two surfaces share this module because they are the same act — a model streaming prose onto a
Discord message through `ResponseStreamer`. `stream_answer` is the QA reply, the one turn that
carries the capability reference, the inline markers and the memory write path; the media
persona reply is the short line the IMAGE / VIDEO routes stream onto the picture they just
handed over, which schedules no memory and owns no status surface of its own.
"""

from typing import Literal, cast
import asyncio
import contextlib
from collections.abc import Callable, Awaitable, AsyncIterator

from openai import AsyncOpenAI
import logfire
from nextcord import Message, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.ext import commands
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.memory import MemoryWriteSummary
from discordbot.typings.models import ModelSettings
from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.utils.llm_transcript import render_author_identity, render_server_identity
from discordbot.utils.media_delivery import MediaDeliveryPlanner
from discordbot.services.memory.store import user_scope, server_scope
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.prompts import (
    MUSIC_INSTRUCTION,
    VIDEO_INSTRUCTION,
    INLINE_IMAGE_INSTRUCTION,
    DEEP_RESEARCH_INSTRUCTION,
    REQUEST_TIME_CONTEXT_PROMPT,
    REQUEST_LOCATION_CONTEXT_PROMPT,
)
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import GeminiKeyToolkit
from discordbot.services.memory.writer import subject_source_line, target_centered_memory_messages
from discordbot.cogs.gen_reply.streaming import (
    MEMORY_WRITE_EMOJI,
    MEMORY_FORGET_EMOJI,
    ResponseStreamer,
    stream_answer_with_retry,
)
from discordbot.services.memory.pipeline import schedule_memory_update
from discordbot.cogs.gen_reply.references import source_channel_is_public
from discordbot.cogs.gen_reply.turn_state import dispatched_model
from discordbot.cogs.gen_reply.capabilities import render_capabilities_block
from discordbot.cogs.gen_reply.interactions import (
    to_interactions_input,
    create_interactions_answer_stream,
)
from discordbot.cogs.gen_reply.research_bridge import maybe_launch_research


def build_runtime_instructions(
    *, system_prompt: str, message: Message, guild_id: int | None
) -> str:
    """Prepends per-request time and conversation-location context to the model instructions.

    The location line names the current guild (or DM) with developer authority so the
    model can reason about where it is speaking; the memory rules lean on it as the
    anchor for never attributing a remembered fact to another server.

    `guild_id` is handed in rather than read off the message because `Message.guild` resolves
    out of the client's own cache: on the `/ask` route that misses for a server the bot was
    never added to, and a guild conversation would tell the model at developer authority that
    it is in a DM. `TurnSurface` is what knows better.
    """
    message_created_at_asia_taipei = message.created_at.astimezone(tz=TAIWAN_TIMEZONE)
    request_time_context = REQUEST_TIME_CONTEXT_PROMPT.format(
        message_created_at_asia_taipei=message_created_at_asia_taipei.isoformat(timespec="seconds")
    ).strip()
    if guild_id is not None:
        # Deliberately id-only: the guild NAME is owner-controlled text and this block
        # rides the developer-authority `instructions` parameter, so embedding it would
        # hand a server owner an instruction-injection surface. The id anchors the
        # location just as well and cannot carry instructions.
        conversation_location = f"a Discord server (guild id {guild_id})"
    else:
        conversation_location = "a Discord direct message (DM)"
    request_location_context = REQUEST_LOCATION_CONTEXT_PROMPT.format(
        conversation_location=conversation_location
    ).strip()
    return f"{request_time_context}\n\n{request_location_context}\n\n{system_prompt}"


def count_media_parts(*, answer_input: ResponseInputParam) -> int:
    """Counts the media parts riding in an assembled answer request.

    An `input_file` (a Files API handle) or an `input_image` (inline base64) is the only shape a
    picture, clip or document reaches the model in, so this one number answers whether an
    attachment survived collection, the modality gate, the upload and the render.
    """
    total = 0
    for item in answer_input:
        content = cast("EasyInputMessageParam", item).get("content", "")
        if isinstance(content, str):
            continue
        total += sum(1 for part in content if part["type"] in ("input_file", "input_image"))
    return total


def memory_report_for(
    *, streamer: ResponseStreamer
) -> Callable[[MemoryWriteSummary], Awaitable[None]]:
    """Builds the callback that tells the user what this turn's memory work recorded.

    Closes over the streamer rather than reaching for it later: by the time memory lands the
    reply path has returned, and the streamer is the only thing that knows where the answer
    ended up and where its usage footer sits.

    The wording is deliberately "took this down" rather than "remembered": the observations
    are staged evidence at this point, and the merge that turns them into stored facts can
    still fold or drop any of them. Showing the content is the point of the line — it is what
    lets someone correct a memory the bot got wrong, on the spot — so a `source_only`
    observation, which is exactly the kind that should not be repeated in a channel long after
    the exchange scrolls past, is counted instead of quoted.

    Every branch produces a line, the empty summary included. The streamer put `正在整理記憶⋯`
    on the reply the moment it landed, so saying nothing here would leave that promise standing
    over a turn that has finished and recorded nothing.
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
    """Streams one reply's prose, on the QA path or onto delivered media.

    Attributes:
        client: The shared LiteLLM-proxy client every answer request dispatches on.
        bot: The Discord bot instance, for its own user (reactions) and the research hop.
        config: Runtime LLM config, read for the inline-marker kill-switches.
        media_delivery: The attach-vs-host-vs-drop planner handed to the streamer.
        toolkit: The leased Gemini key's clients, generators and model catalog.
        message: The message being answered.
        surface: Where this turn's replies go, and which guild it is really happening in.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client every answer request dispatches on."
    )
    bot: SkipValidation[commands.Bot] = Field(
        ..., description="The Discord bot instance, for its own user and the research hop."
    )
    config: SkipValidation[LLMConfig] = Field(
        ..., description="Runtime LLM config, read for the inline-marker kill-switches."
    )
    media_delivery: MediaDeliveryPlanner = Field(
        ..., description="Attach-vs-host-vs-drop planner handed to the streamer."
    )
    toolkit: GeminiKeyToolkit = Field(
        ..., description="The leased Gemini key's clients, generators and model catalog."
    )
    message: SkipValidation[Message] = Field(..., description="The message being answered.")
    surface: TurnSurface = Field(
        ..., description="Where this turn's replies go, and which guild it is happening in."
    )

    async def stream_media_persona_reply(  # noqa: PLR0913 -- shared by IMAGE/VIDEO; the prompt / focus part / noun / span differ per route
        self,
        *,
        reply: Message | None,
        context_task: asyncio.Task[ReplyContext],
        model: ModelSettings,
        system_prompt: str,
        focus_part: ResponseInputFileParam | ResponseInputImageParam,
        media_noun: str,
        span_name: str,
    ) -> None:
        """Best-effort: streams a persona reply onto an already-delivered generated image/video.

        Shared by the IMAGE and VIDEO routes' post-delivery reply. `reply` is the delivered media
        message (native attachment) or None when the media was hosted as a separate URL; the
        persona-base message is built from it INSIDE the protected flow (`_persona_base_reply`), so a
        base-creation or streaming failure is swallowed here instead of surfacing to the outer error
        path, and a fresh hosted-case base that never received content is deleted (never an orphan).
        Builds the answer-path input (history, selected user memory, tone note, reference, current),
        appends the just-made media as the focus, and streams onto the base (its content edits keep an
        attached media). Injects only the selected user memory (already read through the
        compartments this conversation may open) plus the author's tone note, never the server
        memory block, and seeds the
        selection-call usage / memory labels so the footer matches the QA path. Consumes the
        speculative `context_task` (awaited here so its build overlaps generation); any failure
        leaves the delivered media untouched.
        """
        base: Message | None = None
        streamer: ResponseStreamer | None = None
        try:
            context = await context_task
            base = await self.persona_base_reply(reply=reply)
            # Mirror the answer path's order (history, memory, tone, reference, current),
            # injecting only the selected user memory (already compartment-scoped by
            # `recall_user_memories`) and the author's tone note, never the server memory block.
            response_input: ResponseInputParam = [*context.hist_messages]
            response_input.extend(
                block for block in (context.memory_block, context.tone_block) if block is not None
            )
            response_input.extend(context.reference_messages)
            response_input.extend(context.current_message)
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
                input_tokens=context.selection_input_tokens,
                output_tokens=context.selection_output_tokens,
                model_effort=model.effort or "",
            )
            with logfire.span(span_name, model=model.name, message_id=self.message.id):

                async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
                    """Issues the persona-reply request; called again per retry attempt."""
                    return await self.client.responses.create(
                        model=model.deployment_name,
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

    async def persona_base_reply(self, *, reply: Message | None) -> Message:
        """The message the persona stream edits: the delivered media message, or a fresh reply.

        When the media rode as a native attachment, that same message is reused (its content edits
        keep the attachment). When the media was hosted as a separate URL (`reply is None`), a fresh
        non-pinging reply is created here, lazily, only when the persona reply actually proceeds —
        so the hosted-URL message keeps the sole author ping and no empty message is ever orphaned.
        """
        if reply is not None:
            return reply
        return await self.surface.send(
            content=self.message.author.mention, allowed_mentions=AllowedMentions.none()
        )

    async def stream_answer(  # noqa: PLR0913 -- per-call reply inputs plus the route's memory/effort/voice gates
        self,
        *,
        system_prompt: str,
        context: ReplyContext,
        effort: Literal["low", "high"] = "high",
        allow_voice: bool = False,
        allow_image: bool = False,
        allow_music: bool = False,
        allow_video: bool = False,
        allow_research: bool = False,
        describe_capabilities: bool = False,
        yt_url: str | None = None,
    ) -> None:
        """Streams the answer from a pre-built reply context, then schedules memory updates.

        Both the per-user and the per-server update are scheduled here; the per-server one
        carries its own guild / public-channel guards. `allow_voice` enables a
        spoken clip, `allow_image` an inline generated image, `allow_music` an inline generated
        music clip, and `allow_video` an inline generated video clip when the answer model marks
        the reply for it (image / music / video are QA only; the media persona replies get voice
        alone). `describe_capabilities` injects the feature reference that replaced
        `/help`, carried by QA alone since a persona reply riding generated media is not fielding
        a question about the bot. `yt_url`, set only when the router asked
        to watch a linked YouTube video, swaps the answer turn onto the Gemini Interactions API
        (which can ingest the video) while reusing the same streamer / footer / memory path.
        """
        toolkit = self.toolkit
        voice_generator = (
            toolkit.voice_generator if allow_voice and self.config.inline_voice_enabled else None
        )
        image_generator = (
            toolkit.image_generator if allow_image and self.config.inline_image_enabled else None
        )
        music_generator = (
            toolkit.music_generator if allow_music and self.config.music_available else None
        )
        video_generator = (
            toolkit.video_generator if allow_video and self.config.video_available else None
        )
        # Only advertise an inline marker when its renderer is actually active; with it disabled
        # the streamer would strip the block and produce nothing, silently dropping the request
        # from the reply, so a disabled deployment must not be told about it.
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
        # Keep the current user message LAST so the model answers it. Memory rides earliest as
        # low-authority background; the reference message then sits just above the current
        # message so the reply pair (reference -> current) stays adjacent and reads as the
        # primary context rather than getting buried up near history. The feature reference
        # leads: it is the one block that is byte-identical on every reply, so the front is
        # where it costs the least against a prefix cache.
        answer_input: ResponseInputParam = (
            [render_capabilities_block()] if describe_capabilities else []
        )
        answer_input.extend(context.hist_messages)
        answer_input.extend(
            block
            for block in (context.server_memory_block, context.memory_block, context.tone_block)
            if block is not None
        )
        answer_input.extend(context.reference_messages)
        # The linked post(s) the user pointed at ride just before the current message, each
        # block led by its own separator; empty unless a registered source found a link to read
        # (in this message, or for Threads the one it replies to). The order inside is
        # LINK_CONTEXT_SOURCES order.
        answer_input.extend(context.link_blocks)
        answer_input.extend(context.current_message)

        # A linked YouTube video the router asked to watch swaps the answer turn onto the Gemini
        # Interactions API: the Responses bridge cannot make Gemini watch the video, so this is
        # the one backend swap. It is Gemini-only and kill-switchable; otherwise (no video, a
        # non-Gemini answer model, the switch off, or no direct key to swap with) the turn falls
        # back to the Responses path, which never errors. Both feed the same streamer so footer /
        # memory / preview are shared.
        use_interactions = (
            yt_url is not None
            and "gemini" in slow_model.name
            and self.config.youtube_video_enabled
            and bool(self.config.gemini_api_key.strip())
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
                if "gemini" not in slow_model.name
                else "kill-switch"
                if not self.config.youtube_video_enabled
                else "no-gemini-key",
                model=slow_model.name,
            )
        if use_interactions:
            # Persistent marker (added directly, not via the status chain) so it stays after the
            # chain's final reaction to show the reply was grounded in the watched video. The bot's
            # own application emoji `youtube`, usable as a reaction in any guild the bot is in.
            # Added BEFORE the streamer is built: its `created_at` is what the answer latency is
            # measured from, so leaving this REST round trip inside that window would bias the
            # figure against the one backend that pays for it.
            await self.surface.mark(emoji="<:youtube:1517546722535018596>", bot_user=self.bot.user)
        # Seed the streamer with the selection request's usage so the footer and chat reward
        # reflect both LLM calls; the answer stream sums its own usage on top.
        streamer = ResponseStreamer(
            message=self.message,
            surface=self.surface,
            memory_lookups=context.memory_credits,
            input_tokens=context.selection_input_tokens,
            output_tokens=context.selection_output_tokens,
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
            capabilities=describe_capabilities,
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
        with logfire.span(
            "gen_reply answer", model=slow_model.name, backend=backend, message_id=self.message.id
        ):

            async def open_stream() -> AsyncIterator[ResponseStreamEvent]:
                """Issues the answer request; called again per retry attempt."""
                if use_interactions and yt_url is not None:
                    return create_interactions_answer_stream(
                        client=toolkit.gemini_client,
                        model=slow_model.name,
                        system_instruction=build_runtime_instructions(
                            system_prompt=system_prompt,
                            message=self.message,
                            guild_id=self.surface.guild_id,
                        ),
                        steps=to_interactions_input(answer_input=answer_input, youtube_url=yt_url),
                        effort=slow_model.effort,
                    )
                return await self.client.responses.create(
                    model=slow_model.deployment_name,
                    instructions=build_runtime_instructions(
                        system_prompt=system_prompt,
                        message=self.message,
                        guild_id=self.surface.guild_id,
                    ),
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
        if research_offered and streamer.research_brief:
            await maybe_launch_research(
                bot=self.bot,
                message=self.message,
                anchor=streamer.reply,
                brief=streamer.research_brief,
            )
        # Recorded before the memory review is scheduled, so a conversation the store is meant to
        # carry survives even if the fire-and-forget review below never lands.
        await self.surface.record_turn(answer=full_reply)
        self._schedule_memory_updates(context=context, full_reply=full_reply, streamer=streamer)

    def _schedule_memory_updates(
        self, *, context: ReplyContext, full_reply: str, streamer: ResponseStreamer
    ) -> None:
        """Schedules the per-author and per-server memory reviews this turn's markers asked for.

        Whose memory a note lands in is decided HERE, from the message, and the marker body
        never gets a say: the author for the two personal tags and the guild for the server
        one. That is what keeps the compartment boundary structural now that the model, not a
        separate extraction pass, is the one proposing what to write.

        Both calls are unconditional; `schedule_memory_update` itself is what turns a turn that
        marked nothing into a no-op, so the gating that matters stays in one place.
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
        source_line = subject_source_line(guild_id=self.surface.guild_id)
        schedule_memory_update(
            scope=user_scope(user_id=message.author.id),
            subject=f"target_user_id: {message.author.id}\n{source_line}",
            message_list=memory_message_list,
            full_reply=full_reply,
            writer=self.toolkit.memory_writer,
            identity=render_author_identity(
                display_name=message.author.display_name,
                username=message.author.name,
                user_id=message.author.id,
            ),
            remember_notes=tuple(streamer.memory_notes),
            forget_notes=tuple(streamer.forget_notes),
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
        if not source_channel_is_public(message=message):
            return
        schedule_memory_update(
            scope=server_scope(server_id=message.guild.id),
            subject=f"target_server_id: {message.guild.id}",
            message_list=context.message_list,
            full_reply=full_reply,
            writer=self.toolkit.server_memory_writer,
            identity=render_server_identity(
                server_name=message.guild.name, server_id=message.guild.id
            ),
            remember_notes=tuple(streamer.server_memory_notes),
        )
