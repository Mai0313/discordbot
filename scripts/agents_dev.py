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
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.cogs.research.agent import RESEARCH_TOOLS, RESEARCH_AGENT_CONFIG
from discordbot.cogs.gen_reply.prompts import REPLY_PROMPT

if TYPE_CHECKING:
    from collections.abc import Iterator

console = Console()
config = LLMConfig()

# The OpenAI-compatible path takes the proxy's own model aliases.
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
    client = genai.Client(api_key=config.gemini_api_key)
    responses = client.interactions.create(
        agent=RuntimeModelCatalog().antigravity_model.name,
        system_instruction=REPLY_PROMPT,
        input=user_prompt,
        environment=EnvironmentParam(
            type="remote", network=AllowlistParam(allowlist=[AllowlistEntryParam(domain="*")])
        ),
        stream=True,
        tools=RESEARCH_TOOLS,
        agent_config=RESEARCH_AGENT_CONFIG,
    )
    # A `str` agent misses the SDK's `AgentOption` literal overloads, so the call types as
    # `Interaction | Stream`; `stream=True` makes it the stream.
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
