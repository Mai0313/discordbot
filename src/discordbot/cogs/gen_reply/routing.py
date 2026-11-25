"""Single-call triage for route selection, reasoning effort, and optional memory recall.

Runs on the critical path from text-only message renders without attachment uploads or history.
"""

import time
from typing import Literal, cast

from openai import APIError
import logfire
from nextcord import Message
from pydantic import Field, BaseModel, ConfigDict, SkipValidation, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.models import RouteClassification, RecallRouteClassification
from discordbot.utils.llm_errors import llm_status_code, is_retryable_llm_error
from discordbot.cogs.gen_reply.recall import RecallCandidate, render_callable_users_block
from discordbot.cogs.gen_reply.prompts import ROUTE_RECALL_SECTION, route_prompt
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.turn_state import dispatched_model


class RouteClassifier(BaseModel):
    """Runs the triage call for one message."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    toolkit: ReplyToolkit = Field(
        ...,
        description=(
            "The reply toolkit: the proxy client the triage call dispatches on, and the model "
            "catalog that owns the triage tier."
        ),
    )
    message: SkipValidation[Message] = Field(..., description="The message being classified.")

    async def classify(
        self,
        reference_messages: list[EasyInputMessageParam],
        current_message: list[EasyInputMessageParam],
        recall_candidates: dict[int, RecallCandidate],
        server_memory_block: EasyInputMessageParam | None,
        fallback_effort: Literal["low", "medium", "high"] | None = None,
    ) -> RouteClassification:
        """Returns the route, effort, link-read flags, and optional recall picks for one message.

        Reads text-only message parts without waiting on attachment uploads. When
        `recall_candidates` is non-empty, widens the schema to `RecallRouteClassification` and
        includes `server_memory_block` plus the candidate block to also pick obliquely referenced
        user ids.

        Args:
            reference_messages (list[EasyInputMessageParam]): Text-only render of the replied-to
                message, if any.
            current_message (list[EasyInputMessageParam]): Text-only render of the triggering
                message.
            recall_candidates (dict[int, RecallCandidate]): Optional oblique-reference candidates
                keyed by Discord user id.
            server_memory_block (EasyInputMessageParam | None): Server memory context block
                carrying the `## 成員稱呼` table, or None.
            fallback_effort (Literal["low", "medium", "high"] | None): Reasoning effort used only
                when the triage call fails; defaults to the primary slow model's configured effort.
        """
        default_effort: Literal["low", "medium", "high"] = (
            fallback_effort
            if fallback_effort is not None
            else self.toolkit.runtime_models.slow_model.effort
        )
        message_list = [*reference_messages, *current_message]
        instructions = route_prompt()
        text_format: type[RouteClassification] = RouteClassification
        if recall_candidates:
            message_list = [
                *([server_memory_block] if server_memory_block is not None else []),
                *message_list,
                render_callable_users_block(allowed=recall_candidates),
            ]
            instructions += ROUTE_RECALL_SECTION
            text_format = RecallRouteClassification

        triage_model = self.toolkit.runtime_models.triage_model
        dispatched_model.set(triage_model.name)
        started = time.monotonic()
        try:
            with logfire.span("gen_reply route", message_id=self.message.id):
                responses = await self.toolkit.openai_client.responses.parse(
                    model=triage_model.name,
                    instructions=instructions,
                    input=cast("ResponseInputParam", message_list),
                    text_format=text_format,
                    reasoning=triage_model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": self.message.author.name},
                )
            route = responses.output_parsed
            if route is None:
                # Refusals or empty outputs parse to None without raising.
                logfire.warn(
                    "RouteClassification returned no parsed output; defaulting to QA",
                    message_id=self.message.id,
                    model=triage_model.name,
                )
                route = RouteClassification(decision="QA", effort=default_effort)
        except ValidationError as exc:
            # ValidationError covers both empty output text and schema mismatches.
            logfire.warn(
                "RouteClassification parse failed; defaulting to QA",
                message_id=self.message.id,
                model=triage_model.name,
                _exc_info=exc,
            )
            route = RouteClassification(decision="QA", effort=default_effort)
        except APIError as exc:
            # Retryable provider errors degrade to QA; non-retryable errors propagate.
            if not is_retryable_llm_error(exc=exc):
                raise
            logfire.warn(
                "RouteClassification call failed; defaulting to QA",
                message_id=self.message.id,
                model=triage_model.name,
                status_code=llm_status_code(exc=exc),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            route = RouteClassification(decision="QA", effort=default_effort)
        # Log elapsed_seconds explicitly; logfire's console exporter prints no span duration.
        logfire.info(
            "gen_reply route done",
            elapsed_seconds=time.monotonic() - started,
            model=triage_model.name,
            decision=route.decision,
            effort=route.effort,
            link_context_sources=route.link_context_sources,
            watch_video=route.watch_video,
            recall_candidates=len(recall_candidates),
            recall_user_ids=route.recall_user_ids
            if isinstance(route, RecallRouteClassification)
            else [],
            message_id=self.message.id,
        )
        return route
