"""Non-streaming answer path for `ANSWER_STREAMING_ENABLED=false`.

Isolated from `streaming.py` so either transport can be removed cleanly without unwinding
`ResponseStreamer.finalize_reply`, inline media rendering, feedback buttons, or memory updates.
"""

from typing import Protocol, cast
import asyncio
from collections.abc import Callable, Sequence, Awaitable

from google import genai
import logfire
from openai.types.responses import Response
from google.genai.interactions import Step, StepParam
from openai.types.shared.reasoning_effort import ReasoningEffort

from discordbot.utils.llm import output_text_or_empty
from discordbot.typings.timeouts import INTERACTIONS_ANSWER_TIMEOUT_SECONDS
from discordbot.cogs.gen_reply.surface import WINDOW_EXPIRED_NOTICE
from discordbot.cogs.gen_reply.streaming import (
    ResponseStreamer,
    AnswerStreamError,
    IncompleteAnswerError,
)
from discordbot.cogs.gen_reply.interactions import (
    interactions_api_error,
    interactions_answer_options,
    record_interactions_tool_use,
)


class _InteractionError(Protocol):
    """One diagnostic error entry on a non-streaming Interactions result."""

    @property
    def code(self) -> str | None: ...

    @property
    def message(self) -> str | None: ...


class _InteractionUsage(Protocol):
    """Token counters on a non-streaming Interactions result."""

    @property
    def total_input_tokens(self) -> int | None: ...

    @property
    def total_output_tokens(self) -> int | None: ...


class _AnswerInteractionResult(Protocol):
    """Structural view of a non-streaming `interactions.create` QA answer result."""

    @property
    def status(self) -> str: ...

    @property
    def model(self) -> str | None: ...

    @property
    def output_text(self) -> str | None: ...

    @property
    def steps(self) -> Sequence[Step] | None: ...

    @property
    def usage(self) -> _InteractionUsage | None: ...

    @property
    def errors(self) -> Sequence[_InteractionError] | None: ...


type ResponsesAnswerFetcher = Callable[[], Awaitable[Response]]
type InteractionsAnswerFetcher = Callable[[], Awaitable[object]]


async def _run_within_delivery_window[T](
    streamer: ResponseStreamer, fetch: Callable[[], Awaitable[T]]
) -> T:
    """Awaits `fetch` inside the `/ask` delivery window when `streamer.carries_turn_notices`."""
    window = asyncio.timeout(
        delay=streamer.surface.delivery_budget_seconds() if streamer.carries_turn_notices else None
    )
    try:
        async with window:
            return await fetch()
    except Exception as exc:
        if window.expired():
            raise TimeoutError(WINDOW_EXPIRED_NOTICE) from exc
        raise


def _responses_reasoning_summary(responses: Response) -> str:
    """Returns concatenated reasoning-summary text across reasoning items in `responses.output`."""
    return "".join(
        summary.text
        for item in responses.output
        if item.type == "reasoning"
        for summary in (item.summary or [])
        if summary.text
    )


def _interactions_text(interaction: _AnswerInteractionResult) -> str:
    """Returns completed answer text from `interaction.output_text`, falling back to `steps`."""
    if interaction.output_text is not None:
        return interaction.output_text
    return "".join(
        item.text
        for step in interaction.steps or []
        if step.type == "model_output"
        for item in step.content or []
        if item.type == "text" and item.text
    )


def _interactions_reasoning_summary(steps: Sequence[Step]) -> str:
    """Returns concatenated thought-summary text across `thought` steps."""
    return "".join(
        item.text
        for step in steps
        if step.type == "thought"
        for item in step.summary or []
        if item.type == "text" and item.text
    )


async def complete_responses_answer(
    streamer: ResponseStreamer, fetch: ResponsesAnswerFetcher
) -> str:
    """Returns the finalized reply text after fetching one non-streaming Responses API answer.

    Runs `fetch` inside `streamer.surface.delivery_budget_seconds()` when
    `streamer.carries_turn_notices` is True; `finalize_reply` runs outside that window without
    starting any live preview editor.

    Args:
        streamer (ResponseStreamer): Streamer that finalizes the reply onto Discord.
        fetch (ResponsesAnswerFetcher): Factory issuing `responses.create(..., stream=False)`.

    Raises:
        TimeoutError: The `/ask` delivery window expired, carrying `WINDOW_EXPIRED_NOTICE`.
        AnswerStreamError: The response reported `status == "failed"`.
        IncompleteAnswerError: The response ended with a non-`"completed"` status.
    """
    responses = await _run_within_delivery_window(streamer=streamer, fetch=fetch)
    input_tokens = responses.usage.input_tokens if responses.usage is not None else 0
    output_tokens = responses.usage.output_tokens if responses.usage is not None else 0
    if responses.status == "failed":
        if responses.model:
            streamer.model_name = responses.model
        streamer.input_tokens += input_tokens
        streamer.output_tokens += output_tokens
        error = responses.error
        code = error.code if error is not None else None
        message = (
            error.message
            if error is not None and error.message
            else "The model failed to generate a response."
        )
        raise AnswerStreamError(message=message, code=code)
    text = output_text_or_empty(responses=responses)
    reasoning = _responses_reasoning_summary(responses=responses)
    if responses.status != "completed":
        if responses.model:
            streamer.model_name = responses.model
        streamer.input_tokens += input_tokens
        streamer.output_tokens += output_tokens
        details = responses.incomplete_details
        reason = (
            details.reason
            if details is not None and details.reason
            else (responses.status or "incomplete")
        )
        logfire.warn(
            "gen_reply answer incomplete",
            message_id=streamer.message.id,
            model=streamer.model_name,
            backend=streamer.backend,
            reason=reason,
            reply_chars=len(text.lstrip("\n")),
            reasoning_chars=len(reasoning.lstrip("\n")),
        )
        raise IncompleteAnswerError(reason=reason)
    streamer.tool_use.record_response(response=responses)
    streamer.populate_answer(
        text=text,
        reasoning=reasoning,
        model_name=responses.model or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return await streamer.finalize_reply()


async def create_interactions_answer(
    client: genai.Client,
    model: str,
    system_instruction: str,
    steps: list[StepParam],
    effort: ReasoningEffort,
) -> object:
    """Issues one non-streaming YouTube QA request on the Gemini Interactions API.

    Args:
        client (genai.Client): Direct-to-Google Gemini client.
        model (str): Gemini model name to dispatch.
        system_instruction (str): Runtime developer instructions for the turn.
        steps (list[StepParam]): Translated conversation and YouTube input steps.
        effort (ReasoningEffort): Reasoning effort grade (`"low"`, `"medium"`, or `"high"`).
    """
    environment, generation_config, tools = interactions_answer_options(effort=effort)
    async with asyncio.timeout(delay=INTERACTIONS_ANSWER_TIMEOUT_SECONDS):
        return await client.aio.interactions.create(
            model=model,
            system_instruction=system_instruction,
            input=steps,
            environment=environment,
            generation_config=generation_config,
            tools=tools,
            stream=False,
        )


async def complete_interactions_answer(
    streamer: ResponseStreamer, fetch: InteractionsAnswerFetcher
) -> str:
    """Returns the finalized reply text after fetching one non-streaming Interactions API answer.

    Runs `fetch` inside `streamer.surface.delivery_budget_seconds()` when
    `streamer.carries_turn_notices` is True; `finalize_reply` runs outside that window without
    starting any live preview editor.

    Args:
        streamer (ResponseStreamer): Streamer that finalizes the reply onto Discord.
        fetch (InteractionsAnswerFetcher): Factory issuing `interactions.create(..., stream=False)`.

    Raises:
        TimeoutError: The `/ask` delivery window expired, carrying `WINDOW_EXPIRED_NOTICE`.
        APIError: A non-completed interaction carried an error entry in `errors`.
        IncompleteAnswerError: The interaction ended with a non-`"completed"` status and no errors.
    """
    raw = await _run_within_delivery_window(streamer=streamer, fetch=fetch)
    interaction = cast("_AnswerInteractionResult", raw)
    usage = interaction.usage
    input_tokens = (usage.total_input_tokens or 0) if usage is not None else 0
    output_tokens = (usage.total_output_tokens or 0) if usage is not None else 0
    steps = interaction.steps or []
    text = _interactions_text(interaction=interaction)
    reasoning = _interactions_reasoning_summary(steps=steps)
    if interaction.status != "completed":
        if interaction.model:
            streamer.model_name = interaction.model
        streamer.input_tokens += input_tokens
        streamer.output_tokens += output_tokens
        if interaction.errors:
            raise interactions_api_error(
                failure=interaction.errors[0],
                fallback_message=f"Gemini interactions error: {interaction!r}",
            )
        reason = interaction.status or "incomplete"
        logfire.warn(
            "gen_reply answer incomplete",
            message_id=streamer.message.id,
            model=streamer.model_name,
            backend=streamer.backend,
            reason=reason,
            reply_chars=len(text.lstrip("\n")),
            reasoning_chars=len(reasoning.lstrip("\n")),
        )
        raise IncompleteAnswerError(reason=reason)
    record_interactions_tool_use(steps=steps, tool_use=streamer.tool_use)
    streamer.populate_answer(
        text=text,
        reasoning=reasoning,
        model_name=interaction.model or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return await streamer.finalize_reply()
