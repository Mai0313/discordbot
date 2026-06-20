"""Direct Gemini Interactions call layer for the deep-research cog.

All three agents (`antigravity-preview-05-2026` default, `deep-research-preview-04-2026`
and `deep-research-max-preview-04-2026` escalation) run through one injected
`genai.Client` that talks DIRECT to Google (`gemini_api_key`, no proxy). The proxy is
bypassed on purpose: it drops `agent_config` in its interactions->responses transform, so
`collaborative_planning` only works direct (verified 2026-06-20). Every call uses
`background=True` + `store=True` and is polled to a terminal status via `interactions.get`.

Call shapes share the poll + extract helpers:
- `start_antigravity`: starts the default one-shot agent in a remote sandbox environment.
- `start_plan` / `refine_plan`: Deep Research collaborative planning (returns a plan, not a report).
- `start_deep_research`: approves the planned interaction and starts the full research.
- `resume_research`: polls an already-started interaction to its terminal result.

These functions can raise (network errors, `TimeoutError` on the poll bound); the cog wraps
each call and maps failure to a friendly thread message.
"""

import time
import base64
from typing import TYPE_CHECKING
import asyncio

from google import genai
import logfire
from pydantic import Field, BaseModel
from google.genai._interactions.types import EnvironmentParam, DeepResearchAgentConfigParam
from google.genai._interactions.types.environment_param import (
    NetworkAllowlist,
    NetworkAllowlistAllowlist,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

# Polls the running interaction; research is minutes-long so a coarse interval is plenty.
RESEARCH_POLL_INTERVAL_SECONDS = 15.0
# The collaborative-planning call returns a short plan fast (~16s observed), so poll tighter.
PLAN_POLL_INTERVAL_SECONDS = 5.0
# A transient get() error mid-research (e.g. a server 504 gateway timeout) is retried, not fatal;
# only this many CONSECUTIVE failures give up. There is no wall-clock timeout: the Gemini SDK
# bounds each request and the agent settles server-side (Deep Research caps at 60 min).
MAX_CONSECUTIVE_POLL_ERRORS = 30

# Reports progress to the thread: (latest thought summary or None, elapsed seconds).
type ProgressCallback = Callable[[str | None, float], Awaitable[None]]


class ResearchPlan(BaseModel):
    """A proposed research plan returned by Deep Research collaborative planning."""

    interaction_id: str = Field(
        ..., description="The plan interaction's id, used to refine or approve it."
    )
    plan_text: str = Field(
        ..., description="The proposed research plan text, in the user's language."
    )
    status: str = Field(..., description="Terminal interaction status the plan was read from.")


class ResearchResult(BaseModel):
    """The terminal outcome of a research run."""

    interaction_id: str = Field(..., description="The research interaction's id.")
    status: str = Field(
        ..., description="Terminal interaction status (completed / failed / cancelled / ...)."
    )
    report_text: str = Field(
        default="", description="The final report markdown; empty on a non-completed status."
    )
    image_bytes: bytes | None = Field(
        default=None, description="First generated chart/visualization image, if any."
    )
    input_tokens: int = Field(default=0, description="Reported input tokens for the interaction.")
    output_tokens: int = Field(
        default=0, description="Reported output tokens for the interaction."
    )

    @property
    def ok(self) -> bool:
        """Whether the research finished cleanly."""
        return self.status == "completed"


def _deep_research_agent_config(*, collaborative_planning: bool) -> DeepResearchAgentConfigParam:
    """Builds the Deep Research `agent_config` for a plan or a full run."""
    return DeepResearchAgentConfigParam(
        type="deep-research",
        thinking_summaries="auto",
        visualization="auto",
        collaborative_planning=collaborative_planning,
    )


def _created_id(created: object) -> str:
    """Returns a freshly created interaction's id, narrowing past the create() union return."""
    return str(getattr(created, "id", ""))


def _latest_thought(*, interaction: object) -> str | None:
    """Returns the most recent thought-summary text from an interaction's steps, if any."""
    latest: str | None = None
    for step in getattr(interaction, "steps", None) or []:
        for item in getattr(step, "content", None) or []:
            if getattr(item, "type", None) in ("thought_summary", "thought"):
                text = getattr(item, "text", None)
                if text:
                    latest = text
    return latest


def _extract_image(*, interaction: object) -> bytes | None:
    """Returns the first generated image (decoded) from an interaction's model_output steps."""
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for item in getattr(step, "content", None) or []:
            if getattr(item, "type", None) == "image" and getattr(item, "data", None):
                try:
                    return base64.b64decode(item.data)
                except Exception:
                    return None
    return None


def _extract_usage(*, interaction: object) -> tuple[int, int]:
    """Returns `(input_tokens, output_tokens)` from an interaction, defaulting to zero."""
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return 0, 0
    inp = getattr(usage, "total_input_tokens", None) or getattr(usage, "input_tokens", None) or 0
    out = getattr(usage, "total_output_tokens", None) or getattr(usage, "output_tokens", None) or 0
    return int(inp), int(out)


def _to_result(*, interaction: object) -> ResearchResult:
    """Maps a terminal interaction to a `ResearchResult`."""
    input_tokens, output_tokens = _extract_usage(interaction=interaction)
    return ResearchResult(
        interaction_id=str(getattr(interaction, "id", "")),
        status=str(getattr(interaction, "status", "failed")),
        report_text=(getattr(interaction, "output_text", "") or ""),
        image_bytes=_extract_image(interaction=interaction),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def _poll_until_terminal(
    *,
    client: genai.Client,
    interaction_id: str,
    on_progress: "ProgressCallback | None",
    poll_interval_seconds: float,
) -> object:
    """Polls `interactions.get` until the status leaves `in_progress`.

    No wall-clock timeout (the SDK bounds each request; the agent settles server-side). A
    transient get() error mid-research is retried so one 504 does not kill a long run; it gives
    up only after `MAX_CONSECUTIVE_POLL_ERRORS` consecutive failures (re-raising the last error).
    """
    started = time.monotonic()
    consecutive_errors = 0
    while True:
        try:
            interaction = await client.aio.interactions.get(id=interaction_id)
        except Exception:
            consecutive_errors += 1
            logfire.warn(
                "research poll error; retrying",
                interaction_id=interaction_id,
                consecutive_errors=consecutive_errors,
            )
            if consecutive_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                raise
            await asyncio.sleep(poll_interval_seconds)
            continue
        consecutive_errors = 0
        if getattr(interaction, "status", None) != "in_progress":
            return interaction
        if on_progress is not None:
            await on_progress(_latest_thought(interaction=interaction), time.monotonic() - started)
        await asyncio.sleep(poll_interval_seconds)


async def start_antigravity(
    *, client: genai.Client, agent: str, brief: str, system_instruction: str
) -> str:
    """Creates a background Antigravity research interaction and returns its id.

    Split from the poll so the cog can persist the interaction id (for restart resume)
    before the minutes-long poll begins.
    """
    environment = EnvironmentParam(
        type="remote", network=NetworkAllowlist(allowlist=[NetworkAllowlistAllowlist(domain="*")])
    )
    interaction = await client.aio.interactions.create(
        agent=agent,
        input=brief,
        system_instruction=system_instruction,
        environment=environment,
        background=True,
        store=True,
    )
    interaction_id = _created_id(created=interaction)
    logfire.info("research antigravity started", interaction_id=interaction_id, agent=agent)
    return interaction_id


async def start_plan(
    *, client: genai.Client, agent: str, brief: str, system_instruction: str
) -> ResearchPlan:
    """Asks Deep Research for a proposed research plan (collaborative planning)."""
    interaction = await client.aio.interactions.create(
        agent=agent,
        input=brief,
        system_instruction=system_instruction,
        agent_config=_deep_research_agent_config(collaborative_planning=True),
        background=True,
        store=True,
    )
    interaction_id = _created_id(created=interaction)
    final = await _poll_until_terminal(
        client=client,
        interaction_id=interaction_id,
        on_progress=None,
        poll_interval_seconds=PLAN_POLL_INTERVAL_SECONDS,
    )
    return ResearchPlan(
        interaction_id=str(getattr(final, "id", interaction_id)),
        plan_text=(getattr(final, "output_text", "") or ""),
        status=str(getattr(final, "status", "failed")),
    )


async def refine_plan(
    *,
    client: genai.Client,
    agent: str,
    previous_interaction_id: str,
    feedback: str,
    system_instruction: str,
) -> ResearchPlan:
    """Refines a prior plan with the owner's feedback, staying in planning mode."""
    interaction = await client.aio.interactions.create(
        agent=agent,
        input=feedback,
        previous_interaction_id=previous_interaction_id,
        system_instruction=system_instruction,
        agent_config=_deep_research_agent_config(collaborative_planning=True),
        background=True,
        store=True,
    )
    interaction_id = _created_id(created=interaction)
    final = await _poll_until_terminal(
        client=client,
        interaction_id=interaction_id,
        on_progress=None,
        poll_interval_seconds=PLAN_POLL_INTERVAL_SECONDS,
    )
    return ResearchPlan(
        interaction_id=str(getattr(final, "id", interaction_id)),
        plan_text=(getattr(final, "output_text", "") or ""),
        status=str(getattr(final, "status", "failed")),
    )


async def start_deep_research(
    *, client: genai.Client, agent: str, previous_interaction_id: str, system_instruction: str
) -> str:
    """Approves a planned interaction and starts the full Deep Research run; returns its id."""
    interaction = await client.aio.interactions.create(
        agent=agent,
        input="The plan looks good, please proceed with the research.",
        previous_interaction_id=previous_interaction_id,
        system_instruction=system_instruction,
        agent_config=_deep_research_agent_config(collaborative_planning=False),
        background=True,
        store=True,
    )
    interaction_id = _created_id(created=interaction)
    logfire.info("research deep-research started", interaction_id=interaction_id, agent=agent)
    return interaction_id


async def resume_research(
    *, client: genai.Client, interaction_id: str, on_progress: "ProgressCallback | None"
) -> ResearchResult:
    """Re-enters the poll loop for an already-running interaction (restart resume)."""
    final = await _poll_until_terminal(
        client=client,
        interaction_id=interaction_id,
        on_progress=on_progress,
        poll_interval_seconds=RESEARCH_POLL_INTERVAL_SECONDS,
    )
    return _to_result(interaction=final)
