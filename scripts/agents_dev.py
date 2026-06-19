"""Local OpenAI Agents smoke test for the Discord reply prompt."""

from agents import Agent, Runner, set_tracing_disabled
from google import genai, antigravity
import orjson
from rich.console import Console
from agents.result import RunResult
from agents.extensions.models.litellm_model import LitellmModel
from google.genai._interactions.types import (
    EnvironmentParam,
    NetworkAllowlist,
    NetworkAllowlistAllowlist,
)

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs._gen_reply.prompts import REPLY_PROMPT

console = Console()
config = LLMConfig()

# LitellmModel expects LiteLLM provider-prefixed names instead of the model
# aliases used by the OpenAI-compatible request path in cogs/gen_reply.py.
AGENT_MODEL = ModelSettings(name="gemini/gemini-flash-latest", effort="none")


def gen_reply_oai(user_prompt: str) -> RunResult:
    """Runs a dev reply through OpenAI Agents with the LiteLLM model adapter.

    Mirrors the local dev scripts by keeping the model setting near the top and
    reusing `REPLY_PROMPT` from the Discord reply flow. Prints the final agent
    output to the console.

    Args:
        user_prompt: User message to send as the single prompt input.
    """
    set_tracing_disabled(disabled=True)
    agent = Agent(
        name="Assistant",
        instructions=REPLY_PROMPT,
        model=LitellmModel(
            model=AGENT_MODEL.name,
            base_url=config.base_url,
            api_key=config.api_key,
            should_replay_reasoning_content=True,
        ),
    )

    result = Runner.run_sync(starting_agent=agent, input=user_prompt)
    console.print(result.final_output)
    return result


def gen_reply_gemini(user_prompt: str) -> RunResult:
    client = genai.Client()
    responses = client.interactions.create(
        agent="antigravity-preview-05-2026",
        system_instruction=REPLY_PROMPT,
        input=user_prompt,
        environment=EnvironmentParam(
            type="remote",
            network=NetworkAllowlist(allowlist=[NetworkAllowlistAllowlist(domain="*")]),
        ),
        stream=True,
        tools=[{"type": "google_search"}, {"type": "url_context"}],
        agent_config={"type": "dynamic"},
    )
    responses_list = []
    for response in responses:
        if response.event_type == "step.delta":
            if response.delta.type == "thought_summary":
                console.print(f"[dim]{response.delta.content.text}[/dim]", end="")
            else:
                console.print(response.delta.text, end="")
        responses_list.append(response.model_dump())
    with open("./data/agent_response.json", "wb") as f:
        f.write(orjson.dumps(responses_list, option=orjson.OPT_INDENT_2))


async def gen_reply_agy(user_prompt: str) -> RunResult:
    agent_config = antigravity.LocalAgentConfig(
        system_instructions=REPLY_PROMPT, api_key=config.gemini_api_key
    )
    async with antigravity.Agent(config=agent_config) as agent:
        response = await agent.chat(prompt=user_prompt)
        async for thought in response.thoughts:
            console.print(f"[dim]{thought}[/dim]", end="")
        async for delta in response:
            console.print(delta, end="")
        # response_content = await response.text()
        # console.print(response_content)


if __name__ == "__main__":
    # import asyncio
    # gen_reply_oai(user_prompt="為何 37 是質數?")
    gen_reply_gemini(user_prompt="為何 37 是質數?")
    # asyncio.run(gen_reply_agy(user_prompt="為何 37 是質數?"))
