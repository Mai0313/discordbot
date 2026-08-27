"""The two triage calls that decide how one message is answered.

`classify` is the only one on the critical path — every dispatch waits on it — so it stays short
and reads the text-only renders, never an upload. `grade_effort` runs beside it under the same
`route_done` gate and is consumed by the QA answer alone; IMAGE and VIDEO cancel it.
"""

import time
from typing import Literal, cast
import asyncio

from openai import AsyncOpenAI
import logfire
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.models import EffortGrade, RouteClassification
from discordbot.typings.timeouts import EFFORT_GRACE_SECONDS
from discordbot.cogs.gen_reply.prompts import ROUTE_PROMPT, EFFORT_PROMPT
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.turn_state import dispatched_model
from discordbot.cogs.gen_reply.speculation import await_gated


class RouteClassifier(BaseModel):
    """Runs the route and effort triage calls for one message.

    Attributes:
        client: The shared LiteLLM-proxy client both calls dispatch on.
        toolkit: The reply toolkit, which owns the triage model tier.
        message: The message being classified.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client both triage calls dispatch on."
    )
    toolkit: ReplyToolkit = Field(
        ..., description="The reply toolkit's clients and model catalog."
    )
    message: SkipValidation[Message] = Field(..., description="The message being classified.")

    async def classify(
        self,
        *,
        reference_messages: list[EasyInputMessageParam],
        current_message: list[EasyInputMessageParam],
    ) -> RouteClassification:
        """Classifies the message into a reply mode using pre-built context parts.

        The handler choice and the two content-read decisions that ride with it
        (`watch_video`, `link_context_sources`) all come from this one call; the answer
        effort is graded by `grade_effort` in a parallel call, so this stays short on the
        critical path. The reference + current parts arrive already text-only (attachment
        markers, no file ids), so the route classifies on the text without reading or
        waiting on uploads.
        """
        message_list = [*reference_messages, *current_message]

        triage_model = self.toolkit.runtime_models.triage_model
        dispatched_model.set(triage_model.name)
        started = time.monotonic()
        try:
            with logfire.span("gen_reply route", message_id=self.message.id):
                responses = await self.client.responses.parse(
                    model=triage_model.name,
                    instructions=ROUTE_PROMPT,
                    input=cast("ResponseInputParam", message_list),
                    text_format=RouteClassification,
                    reasoning=triage_model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": self.message.author.name},
                )
            parsed = responses.output_parsed
            route = parsed if parsed is not None else RouteClassification(decision="QA")
        except ValidationError as exc:
            # `responses.parse` validates before `output_parsed` is reachable, so an empty /
            # safety-filtered response and a genuine schema mismatch both land here; the
            # attached exception is the only way to tell them apart.
            logfire.warn(
                "RouteClassification parse failed; defaulting to QA",
                message_id=self.message.id,
                model=triage_model.name,
                _exc_info=exc,
            )
            route = RouteClassification(decision="QA")
        # Route-call latency is logged on every path: this is the prime suspect for slow
        # replies, so the log file must show its duration directly, not just a span start.
        logfire.info(
            "gen_reply route done",
            elapsed_seconds=time.monotonic() - started,
            model=triage_model.name,
            decision=route.decision,
            link_context_sources=route.link_context_sources,
            watch_video=route.watch_video,
            message_id=self.message.id,
        )
        return route

    async def grade_effort(
        self,
        *,
        reference_messages: list[EasyInputMessageParam],
        current_message: list[EasyInputMessageParam],
    ) -> EffortGrade:
        """Grades how much reasoning effort the answer model should spend on this message.

        Runs in parallel with the route under the shared `route_done` gate (`await_gated`);
        the grade is consumed only on the QA path, while IMAGE and VIDEO cancel
        this task. The parts arrive already text-only, so grading never waits on uploads.
        Raises on any provider/parse failure so the caller (`resolve_effort`) can fall back.

        Every message is graded by the model, including one carrying an attachment or a link:
        #491's code-decided "high" for those was measured against the live grader in #493 and
        bought nothing the prompt does not already deliver (19-20 of 20 on a screenshot, a thin
        caption over an image, a bare URL and a casual line beside a link), while costing the
        one case it reads wrong, a sticker-only reaction. What that case actually needed was for
        the marker to name a sticker instead of calling it an image (`render_text_only`).
        """
        message_list = [*reference_messages, *current_message]

        triage_model = self.toolkit.runtime_models.triage_model
        started = time.monotonic()
        with logfire.span("gen_reply effort", message_id=self.message.id):
            responses = await self.client.responses.parse(
                model=triage_model.name,
                instructions=EFFORT_PROMPT,
                input=cast("ResponseInputParam", message_list),
                text_format=EffortGrade,
                reasoning=triage_model.reasoning,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": self.message.author.name},
            )
        parsed = responses.output_parsed
        grade = parsed if parsed is not None else EffortGrade(effort="high")
        logfire.info(
            "gen_reply effort done",
            elapsed_seconds=time.monotonic() - started,
            model=triage_model.name,
            effort=grade.effort,
            message_id=self.message.id,
        )
        return grade

    async def resolve_effort(
        self, *, effort_task: asyncio.Task[EffortGrade], route_done: asyncio.Event
    ) -> Literal["low", "high"]:
        """Resolves the parallel effort grade, bounded by the route like memory selection.

        Falls back to "high" on the post-route grace timeout or any grading error, so a slow
        or failed effort call never stalls or silently degrades the reply.
        """
        try:
            grade = await await_gated(
                task=effort_task,
                label="effort",
                route_done=route_done,
                grace_seconds=EFFORT_GRACE_SECONDS,
            )
        except TimeoutError as exc:
            logfire.warn(
                "Effort grading exceeded the post-route grace; defaulting to high effort",
                grace_seconds=EFFORT_GRACE_SECONDS,
                message_id=self.message.id,
                model=self.toolkit.runtime_models.triage_model.name,
                _exc_info=exc,
            )
            return "high"
        except Exception as e:
            logfire.warn(
                "Effort grading failed; defaulting to high effort",
                message_id=self.message.id,
                model=self.toolkit.runtime_models.triage_model.name,
                error_type=type(e).__name__,
                _exc_info=True,
            )
            return "high"
        return grade.effort
