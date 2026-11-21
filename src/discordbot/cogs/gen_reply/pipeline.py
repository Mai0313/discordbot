"""Reply turn orchestration from routing to dispatch and speculative-task teardown.

`ReplyPipeline` sequences speculative context/attachment builds, triage routing, shared-deadline
linked-post builds, route dispatch, and `finally` cleanup so no background task outlives a turn.
"""

import time
from typing import TYPE_CHECKING, Literal
import asyncio
from collections.abc import Callable

import logfire
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.emojis import LINK_SOURCE_EMOJIS, LinkSourceName
from discordbot.typings.models import RecallRouteClassification
from discordbot.utils.reactions import ReactionStatusChain
from discordbot.utils.usage_log import UsageRecorder
from discordbot.typings.timeouts import LINK_CONTEXT_GRACE_SECONDS
from discordbot.utils.media_delivery import MediaDeliveryPlanner
from discordbot.cogs.gen_reply.answer import AnswerTurn, RegenOptions
from discordbot.cogs.gen_reply.context import MessageParts, ReplyContext, ReplyContextBuilder
from discordbot.cogs.gen_reply.prompts import REPLY_PROMPT
from discordbot.cogs.gen_reply.routing import RouteClassifier
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.typings.context_budgets import HISTORY_MESSAGE_LIMIT
from discordbot.cogs.gen_reply.references import find_youtube_url, link_url_for_source
from discordbot.cogs.gen_reply.media_reply import MediaReplyRoutes
from discordbot.cogs.gen_reply.speculation import (
    discard_task,
    discard_link_tasks,
    run_until_deadline,
    await_deadline_bound_task,
)
from discordbot.cogs.gen_reply.status_marks import (
    DONE_EMOJI,
    IMAGE_EMOJI,
    VIDEO_EMOJI,
    ANSWER_EMOJI,
    ROUTING_EMOJI,
)
from discordbot.cogs.gen_reply.research_bridge import can_launch_research
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES

if TYPE_CHECKING:
    from discordbot.typings.models import RouteClassification

# Recorded as a reply's route when the pipeline failed before the router returned one.
UNROUTED_REPLY = "unrouted"

type LinkTask = asyncio.Task[list[EasyInputMessageParam]]
type LinkBuilds = tuple[dict[str, LinkTask], float]


class ReplyPipeline(BaseModel):
    """Routes one message and dispatches the handler the route named."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: LLMConfig = Field(
        ..., description="Runtime LLM config, read for the per-feature kill-switches."
    )
    media_delivery: MediaDeliveryPlanner = Field(
        ..., description="The process-wide attach-vs-host-vs-drop planner."
    )
    usage_recorder: UsageRecorder = Field(
        ..., description="Writes the one usage record this turn produces."
    )
    toolkit: ReplyToolkit = Field(
        ..., description="The clients, generators and caches every phase of this turn uses."
    )
    surface: TurnSurface = Field(
        ..., description="Where this turn happens: its replies, its history and its guild."
    )
    user_prompt: str = Field(
        ..., description="Mention-stripped request text the media routes render from."
    )
    reactions: ReactionStatusChain = Field(
        ..., description="Ordered status-reaction chain on the source message."
    )
    regen: RegenOptions | None = Field(
        default=None,
        description="Optional regeneration context when re-answering an earlier QA reply.",
    )

    @property
    def message(self) -> Message:
        """Returns the message being answered."""
        return self.surface.message

    def _answer_turn(self) -> AnswerTurn:
        """The streamer both the QA answer and the media persona replies run through."""
        return AnswerTurn(
            config=self.config,
            media_delivery=self.media_delivery,
            toolkit=self.toolkit,
            surface=self.surface,
        )

    async def _resolve_link_block(
        self,
        source: LinkSourceName,
        link_task: LinkTask,
        deadline: float,
        on_timeout: Callable[[], list[EasyInputMessageParam]],
    ) -> list[EasyInputMessageParam]:
        """Resolves a deadline-bound link task, returning `on_timeout()` on timeout or `[]`."""
        started = time.monotonic()
        try:
            blocks = await await_deadline_bound_task(
                task=link_task, deadline=deadline, label=source, message_id=self.message.id
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
            # Broad: link builders degrade to notices rather than raising; answer without the link
            # on unexpected errors while letting CancelledError propagate.
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

    def _start_link_builds(self, selected: set[str]) -> LinkBuilds | None:
        """Starts deadline-bound link builds for each selected source with a readable URL."""
        if not selected:
            return None
        deadline = asyncio.get_running_loop().time() + LINK_CONTEXT_GRACE_SECONDS
        link_tasks: dict[str, LinkTask] = {}
        for link_source in LINK_CONTEXT_SOURCES:
            if link_source.name not in selected:
                continue
            link_url = link_url_for_source(source=link_source, message=self.message)
            if link_url is None:
                # Router named a source whose URL sits outside where the source may scan.
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
                        answer_model_is_gemini=self.toolkit.runtime_models.slow_model.is_gemini,
                        gemini_client=self.toolkit.gemini_client_if_configured,
                        allow_media_ingest=link_source.media_ingest_allowed(config=self.config),
                        deadline=deadline,
                    ),
                    deadline=deadline,
                )
            )
        return (link_tasks, deadline) if link_tasks else None

    async def _mark_link_sources(self, link_builds: LinkBuilds | None) -> None:
        """Adds persistent provenance reactions for every started link build."""
        if link_builds is None:
            return
        for source_name in link_builds[0]:
            await self.surface.mark(emoji=LINK_SOURCE_EMOJIS[source_name])

    async def _collect_link_blocks(
        self, link_builds: LinkBuilds | None
    ) -> list[EasyInputMessageParam]:
        """Resolves every started build under the shared grace, in registry splice order."""
        if link_builds is None:
            return []
        link_tasks, deadline = link_builds
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
                    on_timeout=link_source.timeout_blocks,
                )
            )
        return link_blocks

    async def _dispatch_media(
        self, decision: Literal["IMAGE", "VIDEO"], context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Dispatches IMAGE or VIDEO, transferring `context_task` ownership to the handler."""
        routes = MediaReplyRoutes(answer=self._answer_turn())
        handler = routes.handle_image if decision == "IMAGE" else routes.handle_video
        await handler(user_prompt=self.user_prompt, context_task=context_task)

    async def _answer_qa(
        self, route: "RouteClassification", context: ReplyContext, pipeline_started: float
    ) -> int | None:
        """Streams the QA answer, watching a linked YouTube video when the router asked for one."""
        message = self.message
        # Take the URL from the message/replied-to (never the model) so the exact link is watched.
        yt_url = find_youtube_url(message=message) if route.watch_video else None
        if route.watch_video and yt_url is None:
            logfire.info(
                "gen_reply watch_video requested but no YouTube URL was found",
                message_id=message.id,
            )
        logfire.info(
            "gen_reply pre-answer latency",
            elapsed_seconds=time.monotonic() - pipeline_started,
            decision=route.decision,
            message_id=message.id,
        )
        return await self._answer_turn().stream_answer(
            system_prompt=REPLY_PROMPT,
            context=context,
            effort=route.effort,
            allow_research=can_launch_research(message=message),
            yt_url=yt_url,
            question=self.user_prompt,
            regen=self.regen,
        )

    def _apply_route_overrides(self, route: "RouteClassification") -> "RouteClassification":
        """Applies regeneration overrides (forced QA + preserved effort) and keyless VIDEO downgrade."""
        if self.regen is not None:
            overrides: dict[str, object] = {"decision": "QA"}
            if self.regen.forced_effort is not None:
                overrides["effort"] = self.regen.forced_effort
            return route.model_copy(update=overrides)
        if route.decision == "VIDEO" and not self.config.gemini_key_configured:
            # VIDEO renders direct to Google; without a Gemini key, downgrade to QA.
            logfire.info(
                "gen_reply VIDEO route needs a Gemini key; answering as QA",
                message_id=self.message.id,
            )
            return route.model_copy(update={"decision": "QA"})
        return route

    async def run(self) -> int | None:
        """Routes the message and dispatches the matching handler with speculative QA context."""
        message = self.message
        # Tracked here so finally records the reached route (or UNROUTED_REPLY) in usage.
        route_decision: Literal["IMAGE", "VIDEO", "QA"] | None = None
        reply_message_id: int | None = None
        prep_task: asyncio.Task[ReplyContext] | None = None
        parts_task: asyncio.Task[MessageParts] | None = None
        link_builds: LinkBuilds | None = None
        context_builder = ReplyContextBuilder(toolkit=self.toolkit, surface=self.surface)
        classifier = RouteClassifier(
            toolkit=self.toolkit,
            message=message,
            inline_image_enabled=self.config.inline_image_enabled,
        )
        try:
            with logfire.span("gen_reply pipeline", message_id=message.id) as pipeline_span:
                pipeline_started = time.monotonic()
                self.reactions.advance(emoji=ROUTING_EMOJI)
                # Upload attachments and build context speculatively in parallel with routing.
                parts_task = asyncio.create_task(coro=context_builder.render_parts())
                text_reference, text_current = await context_builder.render_parts(text_only=True)
                recall = context_builder.plan_recall()
                recall_picks: asyncio.Future[list[str]] = (
                    asyncio.get_running_loop().create_future()
                )
                prep_task = asyncio.create_task(
                    coro=context_builder.build(
                        history_limit=HISTORY_MESSAGE_LIMIT,
                        parts_task=parts_task,
                        recall=recall,
                        recall_picks=recall_picks,
                    )
                )
                route = await classifier.classify(
                    reference_messages=text_reference,
                    current_message=text_current,
                    recall_candidates=recall.optional_candidates,
                    server_memory_block=recall.server_memory_block,
                )
                # Resolve before handing prep_task to a media route (finally stops cancelling it).
                recall_picks.set_result(
                    route.recall_user_ids if isinstance(route, RecallRouteClassification) else []
                )
                route = self._apply_route_overrides(route=route)
                route_decision = route.decision
                pipeline_span.set_attribute(key="route", value=route.decision)
                if route.decision in ("IMAGE", "VIDEO"):
                    self.reactions.advance(
                        emoji=IMAGE_EMOJI if route.decision == "IMAGE" else VIDEO_EMOJI
                    )
                    # Keep parts_task for finally: a discarded media prep may not have awaited it.
                    media_context_task = prep_task
                    prep_task = None
                    await self._dispatch_media(
                        decision=route.decision, context_task=media_context_task
                    )
                else:
                    # Assign before awaiting marks so finally drains tasks if marking raises/cancels.
                    link_builds = self._start_link_builds(selected=set(route.link_context_sources))
                    await self._mark_link_sources(link_builds=link_builds)
                    self.reactions.advance(emoji=ANSWER_EMOJI)
                    context = await prep_task
                    prep_task = None
                    parts_task = None
                    link_blocks = await self._collect_link_blocks(link_builds=link_builds)
                    context = (
                        context.model_copy(update={"link_blocks": link_blocks})
                        if link_blocks
                        else context
                    )
                    pipeline_span.set_attribute(key="effort", value=route.effort)
                    reply_message_id = await self._answer_qa(
                        route=route, context=context, pipeline_started=pipeline_started
                    )
                self.reactions.advance(emoji=DONE_EMOJI)
                # Log turn total explicitly because logfire's console exporter prints no span end.
                logfire.info(
                    "gen_reply pipeline done",
                    elapsed_seconds=time.monotonic() - pipeline_started,
                    decision=route.decision,
                    message_id=message.id,
                )
                return reply_message_id
        finally:
            for task, label in ((prep_task, "prep"), (parts_task, "parts")):
                if task is not None:
                    await discard_task(task=task, label=label, message_id=message.id)
            if link_builds is not None:
                await discard_link_tasks(
                    link_tasks=link_builds[0], deadline=link_builds[1], message_id=message.id
                )
            # One usage record per triggering message (including failures under route or unrouted).
            await self.usage_recorder.record(
                kind="reply",
                name=route_decision or UNROUTED_REPLY,
                user_id=message.author.id,
                user_name=message.author.name,
                guild_id=self.surface.guild_id,
                channel_id=message.channel.id,
            )
