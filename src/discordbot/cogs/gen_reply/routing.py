"""The one triage call that decides how one message is answered.

`classify` is on the critical path — every dispatch waits on it — so it reads the text-only
renders, never an upload, and never the channel history. Besides the route it grades the answer
effort and, on a turn that offers optional recall candidates, names the ones the latest message
refers to; one call does all three because the separate calls cost as much and the separate
selector picked someone on most turns where nobody was named (#725).
"""

import time
from typing import cast

from openai import AsyncOpenAI
import logfire
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.models import RouteClassification, RecallRouteClassification
from discordbot.cogs.gen_reply.recall import RecallCandidate, render_callable_users_block
from discordbot.cogs.gen_reply.prompts import ROUTE_PROMPT, ROUTE_RECALL_SECTION
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.turn_state import dispatched_model


class RouteClassifier(BaseModel):
    """Runs the triage call for one message."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Shared LiteLLM-proxy client the triage call dispatches on."
    )
    toolkit: ReplyToolkit = Field(
        ..., description="The reply toolkit's model catalog, which owns the triage model tier."
    )
    message: SkipValidation[Message] = Field(..., description="The message being classified.")

    async def classify(
        self,
        *,
        reference_messages: list[EasyInputMessageParam],
        current_message: list[EasyInputMessageParam],
        recall_candidates: dict[int, RecallCandidate],
        server_memory_block: EasyInputMessageParam | None,
    ) -> RouteClassification:
        """Classifies the message into a reply mode using pre-built context parts.

        The handler choice, the two content-read decisions that ride with it (`watch_video`,
        `link_context_sources`) and the answer effort all come from this one call. The reference
        + current parts arrive already text-only (attachment markers, no file ids), so the route
        classifies on the text without reading or waiting on uploads.

        With `recall_candidates` the same call also names which of them the latest message
        refers to, and only then does it read the server memory (whose nickname table maps an
        alias to an id) and the candidate block, kept last so it is read right before deciding.
        The picks are ids only; `ReplyContextBuilder.build` resolves them against the same
        allowlist.
        """
        message_list = [*reference_messages, *current_message]
        instructions = ROUTE_PROMPT
        text_format: type[RouteClassification] = RouteClassification
        if recall_candidates:
            message_list = [
                *([server_memory_block] if server_memory_block is not None else []),
                *message_list,
                render_callable_users_block(allowed=recall_candidates),
            ]
            instructions = ROUTE_PROMPT + ROUTE_RECALL_SECTION
            text_format = RecallRouteClassification

        triage_model = self.toolkit.runtime_models.triage_model
        dispatched_model.set(triage_model.name)
        started = time.monotonic()
        try:
            with logfire.span("gen_reply route", message_id=self.message.id):
                responses = await self.client.responses.parse(
                    model=triage_model.name,
                    instructions=instructions,
                    input=cast("ResponseInputParam", message_list),
                    text_format=text_format,
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
            effort=route.effort,
            link_context_sources=route.link_context_sources,
            watch_video=route.watch_video,
            recall_candidates=len(recall_candidates),
            message_id=self.message.id,
        )
        return route
