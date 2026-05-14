"""Local OpenAI Agents smoke test for the Discord reply prompt."""

from agents import Agent, Runner, set_tracing_disabled
from rich.console import Console
from agents.result import RunResult
from agents.extensions.models.litellm_model import LitellmModel

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs._gen_reply.prompts import REPLY_PROMPT

console = Console()
config = LLMConfig()

# LitellmModel expects LiteLLM provider-prefixed names instead of the model
# aliases used by the OpenAI-compatible request path in cogs/gen_reply.py.
AGENT_MODEL = ModelSettings(name="gemini/gemini-3-flash-preview", effort="none")


def gen_reply(user_prompt: str) -> RunResult:
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


if __name__ == "__main__":
    result = gen_reply(user_prompt="為何 37 是質數?")
