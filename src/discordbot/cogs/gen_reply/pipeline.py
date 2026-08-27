"""One reply turn, from the route call to the answer that lands.

`ReplyPipeline` owns the order the phases run in and nothing else: it starts the speculative
builds, waits on the one call every dispatch depends on, and hands the turn to whichever handler
the route named. Each phase lives in its own module (`routing`, `context`, `answer`,
`media_reply`), so what is left here is the sequencing, the shared post-route deadline the link
builders run against, and the teardown that guarantees no speculative task outlives the turn.
"""

import time
from typing import TYPE_CHECKING, Literal
import asyncio
from collections.abc import Callable

from openai import AsyncOpenAI
import logfire
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.ext import commands
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.utils.reactions import ReactionStatusChain
from discordbot.utils.usage_log import UsageRecorder
from discordbot.typings.timeouts import LINK_CONTEXT_GRACE_SECONDS
from discordbot.utils.media_delivery import MediaDeliveryPlanner
from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.cogs.gen_reply.context import MessageParts, ReplyContext, ReplyContextBuilder
from discordbot.cogs.gen_reply.prompts import REPLY_PROMPT
from discordbot.cogs.gen_reply.routing import RouteClassifier
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import GeminiKeyToolkit
from discordbot.typings.context_budgets import HISTORY_MESSAGE_LIMIT
from discordbot.cogs.gen_reply.references import find_youtube_url, link_url_for_source
from discordbot.cogs.gen_reply.media_reply import MediaReplyRoutes
from discordbot.cogs.gen_reply.speculation import (
    discard_task,
    run_until_deadline,
    await_deadline_bound_task,
    drain_deadline_bound_task,
)
from discordbot.cogs.gen_reply.research_bridge import can_launch_research
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES

if TYPE_CHECKING:
    from discordbot.typings.models import EffortGrade, RouteClassification

# Recorded as a reply's route when the pipeline failed before the router returned one.
UNROUTED_REPLY = "unrouted"

type LinkTask = asyncio.Task[list[EasyInputMessageParam]]


async def discard_link_tasks(
    *, link_tasks: dict[str, LinkTask], deadline: float | None, message_id: int
) -> None:
    """Drains link builds without stealing cancellation from their shared deadline."""
    if link_tasks and deadline is None:
        raise RuntimeError("Selected link tasks have no route deadline")
    if deadline is None:
        return
    for name, task in link_tasks.items():
        await drain_deadline_bound_task(
            task=task, deadline=deadline, label=name, message_id=message_id
        )
    link_tasks.clear()


class ReplyPipeline(BaseModel):
    """Routes one message and dispatches the handler the route named.

    Attributes:
        client: The shared LiteLLM-proxy client every phase dispatches on.
        bot: The Discord bot instance, for its own user and the cross-cog hops.
        config: Runtime LLM config, read for the per-feature kill-switches.
        media_delivery: The process-wide attach-vs-host-vs-drop planner.
        usage_recorder: Writes the one usage record this turn produces.
        toolkit: The Gemini key leased for this turn; every Gemini call inherits it.
        message: The message being answered.
        surface: Where this turn happens: its replies, its history and its guild.
        user_prompt: The mention-stripped request text the media routes render from.
        reactions: The ordered status-reaction chain on the source message.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client every phase dispatches on."
    )
    bot: SkipValidation[commands.Bot] = Field(
        ..., description="The Discord bot instance, for its own user and the cross-cog hops."
    )
    config: SkipValidation[LLMConfig] = Field(
        ..., description="Runtime LLM config, read for the per-feature kill-switches."
    )
    media_delivery: MediaDeliveryPlanner = Field(
        ..., description="The process-wide attach-vs-host-vs-drop planner."
    )
    usage_recorder: UsageRecorder = Field(
        ..., description="Writes the one usage record this turn produces."
    )
    toolkit: GeminiKeyToolkit = Field(
        ..., description="The Gemini key leased for this turn; every Gemini call inherits it."
    )
    message: SkipValidation[Message] = Field(..., description="The message being answered.")
    surface: TurnSurface = Field(
        ..., description="Where this turn happens: its replies, its history and its guild."
    )
    user_prompt: str = Field(
        ..., description="Mention-stripped request text the media routes render from."
    )
    reactions: ReactionStatusChain = Field(
        ..., description="Ordered status-reaction chain on the source message."
    )

    def _answer_turn(self) -> AnswerTurn:
        """The streamer both the QA answer and the media persona replies run through."""
        return AnswerTurn(
            client=self.client,
            bot=self.bot,
            config=self.config,
            media_delivery=self.media_delivery,
            toolkit=self.toolkit,
            message=self.message,
            surface=self.surface,
        )

    async def _resolve_link_block(
        self,
        *,
        source: str,
        link_task: LinkTask,
        deadline: float,
        on_timeout: Callable[[], list[EasyInputMessageParam]],
    ) -> list[EasyInputMessageParam]:
        """Resolves an intent-selected linked-post build before the shared route deadline.

        Each selected builder owns the deadline fixed when routing finishes. This resolver only
        retrieves that task, so the builder's cancellation cleanup cannot be interrupted by a
        second timeout. On expiry it injects a short "could not read it in time" notice instead
        of nothing; on any other unexpected error it returns [] (cancellation propagates). The
        builders themselves never raise (they degrade to their own notices).
        """
        started = time.monotonic()
        try:
            blocks = await await_deadline_bound_task(
                task=link_task, deadline=deadline, label=source
            )
        except TimeoutError as exc:
            logfire.warn(
                "Linked-post context exceeded the post-route grace; injecting timeout notice",
                source=source,
                grace_seconds=LINK_CONTEXT_GRACE_SECONDS,
                message_id=self.message.id,
                _exc_info=exc,
            )
            return on_timeout()
        except Exception as exc:
            # Broad on purpose: the builders are documented never to raise, so anything landing
            # here is unexpected (a builder bug, or a fetch/WAF failure that escaped its own
            # notice); error_type is what tells those apart in logs. CancelledError is a
            # BaseException and deliberately propagates instead of being swallowed as "no link".
            logfire.warn(
                "Linked-post context failed; answering without it",
                source=source,
                message_id=self.message.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return []
        logfire.info(
            "gen_reply link context done",
            source=source,
            elapsed_seconds=time.monotonic() - started,
            blocks=len(blocks),
            message_id=self.message.id,
        )
        return blocks

    def _start_link_builds(self, *, selected: set[str], deadline: float) -> dict[str, LinkTask]:
        """Starts a build per selected source that has a URL it is allowed to read.

        The router selects only source names; URL ownership stays local and the registry still
        applies every URL filter and replied-to rule. Each builder starts only now, after intent
        is known, so an incidental link never begins a metadata fetch, media download, or Files
        API upload.
        """
        link_tasks: dict[str, LinkTask] = {}
        for link_source in LINK_CONTEXT_SOURCES:
            if link_source.name not in selected:
                continue
            link_url = link_url_for_source(source=link_source, message=self.message)
            if link_url is None:
                # The router named this source, but its URL is not where the source is
                # allowed to look (Threads alone walks the reply chain), so the answer
                # silently goes without the post the user was pointing at.
                logfire.info(
                    "gen_reply selected link source has no readable URL; skipping it",
                    source=link_source.name,
                    message_id=self.message.id,
                )
                continue
            link_tasks[link_source.name] = asyncio.create_task(
                coro=run_until_deadline(
                    awaitable=link_source.build(
                        url=link_url,
                        answer_model_is_gemini=(
                            "gemini" in self.toolkit.runtime_models.slow_model.name
                        ),
                        gemini_client=self.toolkit.gemini_client_if_configured,
                        allow_media_ingest=link_source.media_ingest_allowed(config=self.config),
                    ),
                    deadline=deadline,
                )
            )
        return link_tasks

    async def _collect_link_blocks(
        self, *, link_tasks: dict[str, LinkTask], deadline: float
    ) -> list[EasyInputMessageParam]:
        """Resolves every started build under the shared grace, in registry splice order."""
        link_blocks: list[EasyInputMessageParam] = []
        for link_source in LINK_CONTEXT_SOURCES:
            link_task = link_tasks.pop(link_source.name, None)
            if link_task is None:
                continue
            link_blocks.extend(
                await self._resolve_link_block(
                    source=link_source.name,
                    link_task=link_task,
                    deadline=deadline,
                    on_timeout=link_source.on_timeout,
                )
            )
        return link_blocks

    async def _dispatch_media(
        self, *, decision: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Runs the IMAGE or VIDEO route, which consumes the speculative context.

        The handler awaits `context_task` only after the media is on screen, so the context
        build overlaps generation instead of delaying it.
        """
        routes = MediaReplyRoutes(
            config=self.config,
            media_delivery=self.media_delivery,
            toolkit=self.toolkit,
            message=self.message,
            surface=self.surface,
            answer=self._answer_turn(),
        )
        handler = routes.handle_image if decision == "IMAGE" else routes.handle_video
        await handler(user_prompt=self.user_prompt, context_task=context_task)

    async def _answer_qa(
        self,
        *,
        route: "RouteClassification",
        context: ReplyContext,
        effort: Literal["low", "high"],
        pipeline_started: float,
    ) -> None:
        """Streams the QA answer, watching a linked YouTube video when the router asked for one."""
        message = self.message
        # Watch a linked YouTube video only when the router judged the user is asking
        # about it; the URL itself is taken from the message text or the replied-to
        # message (never the model) so the answer turn ingests the exact link posted.
        yt_url = find_youtube_url(message=message) if route.watch_video else None
        if route.watch_video and yt_url is None:
            # The router judged the user is asking about a video, but the URL scan
            # found none where it is allowed to look, so the answer is written
            # without watching anything and nothing else records that.
            logfire.info(
                "gen_reply watch_video requested but no YouTube URL was found",
                message_id=message.id,
            )
        # Total time from pipeline start to answer dispatch (the user's 'router stage').
        logfire.info(
            "gen_reply pre-answer latency",
            elapsed_seconds=time.monotonic() - pipeline_started,
            decision=route.decision,
            message_id=message.id,
        )
        await self._answer_turn().stream_answer(
            system_prompt=REPLY_PROMPT,
            context=context,
            effort=effort,
            allow_voice=True,
            allow_image=True,
            allow_music=True,
            allow_video=True,
            allow_research=can_launch_research(message=message),
            describe_capabilities=True,
            yt_url=yt_url,
        )

    async def run(self) -> None:  # noqa: PLR0915 -- the turn's sequence, and the task handles its `finally` drains
        """Routes the message and dispatches the matching handler with speculative QA context."""
        message = self.message
        # Named in the usage record below, which is written from this method's `finally`
        # because this is the one scope that has both outcomes and the route in hand.
        route_decision: str | None = None
        prep_task: asyncio.Task[ReplyContext] | None = None
        parts_task: asyncio.Task[MessageParts] | None = None
        effort_task: asyncio.Task[EffortGrade] | None = None
        link_tasks: dict[str, LinkTask] = {}
        link_context_deadline: float | None = None
        context_builder = ReplyContextBuilder(
            client=self.client,
            bot=self.bot,
            toolkit=self.toolkit,
            message=message,
            surface=self.surface,
        )
        classifier = RouteClassifier(client=self.client, toolkit=self.toolkit, message=message)
        try:
            with logfire.span("gen_reply pipeline", message_id=message.id) as pipeline_span:
                pipeline_started = time.monotonic()
                self.reactions.advance(emoji="<:flowchart:1517561877973045349>")
                # The reference + current attachment uploads (and their activation polls)
                # run in the background and only the answer awaits them. The route and the
                # optional memory selection use the text-only renders, so neither waits on the Files
                # API. The QA context builds speculatively in parallel with the route call
                # since QA is the dominant route — non-QA routes discard it.
                parts_task = asyncio.create_task(coro=context_builder.render_parts())
                text_reference, text_current = await context_builder.render_parts(text_only=True)
                # Signals optional memory selection that the route has returned: selection runs
                # unbounded while this is clear and gets only a short grace once it is set.
                route_done = asyncio.Event()
                prep_task = asyncio.create_task(
                    coro=context_builder.build(
                        history_limit=HISTORY_MESSAGE_LIMIT,
                        parts_task=parts_task,
                        text_parts=(text_reference, text_current),
                        route_done=route_done,
                    )
                )
                # Effort grading rides the same route_done gate as memory selection: it runs
                # in parallel with the route and only the QA answer model consumes it, so
                # IMAGE/VIDEO cancel it below.
                effort_task = asyncio.create_task(
                    coro=classifier.grade_effort(
                        reference_messages=text_reference, current_message=text_current
                    )
                )
                route = await classifier.classify(
                    reference_messages=text_reference, current_message=text_current
                )
                reads_links = route.decision == "QA" and bool(route.link_context_sources)
                if reads_links:
                    link_context_deadline = (
                        asyncio.get_running_loop().time() + LINK_CONTEXT_GRACE_SECONDS
                    )
                route_done.set()
                route_decision = route.decision
                pipeline_span.set_attribute(key="route", value=route.decision)
                if reads_links:
                    if link_context_deadline is None:
                        raise RuntimeError("Selected link sources have no route deadline")
                    link_tasks = self._start_link_builds(
                        selected=set(route.link_context_sources), deadline=link_context_deadline
                    )
                    if "threads" in link_tasks:
                        # Persistent marker (added directly, not via the status chain) saying a
                        # Threads post was read, the same one `parse_threads` adds when it expands
                        # a link instead. Added once every builder is started so the REST call
                        # never sits between two of them.
                        await self.surface.mark(
                            emoji="<:threads:1535657820668559380>", bot_user=self.bot.user
                        )
                if route.decision in ("IMAGE", "VIDEO"):
                    # IMAGE and VIDEO share identical speculative-task teardown; they differ only
                    # in the status emoji and which media handler runs. Effort is answer-only,
                    # while intent-gated link builders never start for these routes.
                    await discard_task(task=effort_task, label="effort", message_id=message.id)
                    effort_task = None
                    self.reactions.advance(
                        emoji="<:image:1517559727880667226>"
                        if route.decision == "IMAGE"
                        else "<:video:1517560671913377842>"
                    )
                    # `parts_task` is left for the finally backstop — prep awaits it via
                    # asyncio.shield, so if the handler discards prep on a generation failure the
                    # shielded upload keeps running and the finally must drain it.
                    media_context_task = prep_task
                    prep_task = None
                    await self._dispatch_media(
                        decision=route.decision, context_task=media_context_task
                    )
                else:
                    self.reactions.advance(emoji="<:message:1517560873000898860>")
                    # Selection still gates the answer here; if this wait ever needs to go,
                    # the answer could speculatively start without memory and refire when
                    # selection picks some.
                    context = await prep_task
                    prep_task = None
                    parts_task = None
                    effort = await classifier.resolve_effort(
                        effort_task=effort_task, route_done=route_done
                    )
                    effort_task = None
                    # The selected builds overlapped the remaining reply preparation. Resolve
                    # each under the same grace and fold the post blocks into the answer context
                    # in registry order so the splice stays deterministic.
                    if link_tasks:
                        if link_context_deadline is None:
                            raise RuntimeError("Selected link tasks have no route deadline")
                        link_blocks = await self._collect_link_blocks(
                            link_tasks=link_tasks, deadline=link_context_deadline
                        )
                        context = context.model_copy(update={"link_blocks": link_blocks})
                    pipeline_span.set_attribute(key="effort", value=effort)
                    await self._answer_qa(
                        route=route,
                        context=context,
                        effort=effort,
                        pipeline_started=pipeline_started,
                    )
                self.reactions.advance(emoji="<:greencheck:1517565102424068226>")
                # End of the turn on the success path; the failure path is `gen_reply failed`,
                # which carries the traceback. The console exporter prints no span-end line, so
                # without this the file holds no total for the turn at all.
                logfire.info(
                    "gen_reply pipeline done",
                    elapsed_seconds=time.monotonic() - pipeline_started,
                    decision=route.decision,
                    message_id=message.id,
                )
        finally:
            for task, label in (
                (prep_task, "prep"),
                (effort_task, "effort"),
                (parts_task, "parts"),
            ):
                if task is not None:
                    await discard_task(task=task, label=label, message_id=message.id)
            await discard_link_tasks(
                link_tasks=link_tasks, deadline=link_context_deadline, message_id=message.id
            )
            # One record per triggering message, not per delivered artifact: the inline
            # clips and the media persona reply are all parts of this same turn. A failure
            # is recorded too — someone still talked to the bot — under the route it had
            # reached, or `unrouted` when it did not get that far.
            await self.usage_recorder.record(
                kind="reply",
                name=route_decision or UNROUTED_REPLY,
                user_id=message.author.id,
                user_name=message.author.name,
                guild_id=self.surface.guild_id,
                channel_id=message.channel.id,
            )
