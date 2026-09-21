"""Local OpenAI Agents smoke test for the Discord reply prompt."""

from typing import TYPE_CHECKING, cast

from agents import Agent, Runner, set_tracing_disabled
from google import genai
from openai import AsyncOpenAI
import orjson
from rich.console import Console
from agents.result import RunResult
from google.genai.interactions import (
    AllowlistParam,
    EnvironmentParam,
    AllowlistEntryParam,
    InteractionSSEEvent,
)
from agents.models.openai_responses import OpenAIResponsesModel

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs.gen_reply.prompts import REPLY_PROMPT

if TYPE_CHECKING:
    from collections.abc import Iterator

console = Console()
config = LLMConfig()

# The OpenAI-compatible path takes the proxy's own model aliases, the same ones
# cogs/gen_reply/cog.py sends.
AGENT_MODEL = ModelSettings(name="gemini-3.8-flash", effort="minimal")


def gen_reply_oai(user_prompt: str) -> RunResult:
    """Runs a dev reply through OpenAI Agents against the proxy's Responses API.

    Args:
        user_prompt (str): User message to send as the single prompt input.

    Returns:
        RunResult: Final agent run result.
    """
    set_tracing_disabled(disabled=True)
    agent = Agent(
        name="Assistant",
        instructions=REPLY_PROMPT,
        model=OpenAIResponsesModel(
            model=AGENT_MODEL.name,
            openai_client=AsyncOpenAI(base_url=config.base_url, api_key=config.api_key),
        ),
    )

    result = Runner.run_sync(starting_agent=agent, input=user_prompt)
    console.print(result.final_output)
    return result


def gen_reply_gemini(user_prompt: str) -> None:
    """Streams a dev reply using the Antigravity agent on the Gemini Interactions API.

    Args:
        user_prompt (str): User message to send as the single prompt input.
    """
    client = genai.Client()
    responses = client.interactions.create(
        agent="antigravity-preview-09-2026",
        system_instruction=REPLY_PROMPT,
        input=user_prompt,
        environment=EnvironmentParam(
            type="remote", network=AllowlistParam(allowlist=[AllowlistEntryParam(domain="*")])
        ),
        stream=True,
        tools=[{"type": "google_search"}, {"type": "url_context"}],
        agent_config={"type": "dynamic"},
    )
    # The SDK's `AgentOption` literal list lags the live API, so an agent it has not been
    # regenerated for falls to the overload returning `Interaction | Stream`, exactly as the
    # production path's `str` argument does; cast as `research/agent.py` does.
    responses_list = []
    for response in cast("Iterator[InteractionSSEEvent]", responses):
        if response.event_type == "step.delta":
            delta = response.delta
            if delta.type == "thought_summary":
                text = getattr(delta.content, "text", "") if delta.content is not None else ""
                console.print(f"[dim]{text}[/dim]", end="")
            else:
                console.print(getattr(delta, "text", ""), end="")
        responses_list.append(response.model_dump())
    with open("./data/agent_response.json", "wb") as f:
        f.write(orjson.dumps(responses_list, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    # gen_reply_oai(user_prompt="為何 37 是質數?")
    gen_reply_gemini(user_prompt="為何 37 是質數?")
