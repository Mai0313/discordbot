"""Local prompt development helpers for the Antigravity agent SDK.

Unlike `prompt_dev.py`, nothing here goes through the LiteLLM proxy: `Agent` spawns the
`localharness` binary shipped inside the `google-antigravity` wheel and that harness talks to
the Gemini Developer API itself, so a run here is billed and traced nowhere but Google.
"""

import time
import asyncio
from datetime import datetime

from pydantic import Field, BaseModel
from rich.console import Console
from google.antigravity import (
    Agent,
    ModelTarget,
    BuiltinTools,
    ThinkingLevel,
    LocalAgentConfig,
    GeminiAPIEndpoint,
    CapabilitiesConfig,
    GeminiModelOptions,
    SystemInstructionSection,
    TemplatedSystemInstructions,
)
from google.antigravity.types import Text, Thought, ToolCall

from discordbot.typings.llm import LLMConfig
from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.cogs.gen_reply.prompts import (
    COMMON_PROMPT,
    PERSONA_CHOICES,
    REQUEST_TIME_CONTEXT_PROMPT,
)

console = Console()
config = LLMConfig()

# Every builtin, which is also what an unset `enabled_tools` means; naming it keeps the knob
# visible. The other useful preset is `read_only()`, which carries neither `search_web` nor
# `start_subagent` — the harness ANDs `enable_subagents` with this list, so a preset that omits
# that tool switches the flag off with it. What `all_tools()` adds and a dev run should know
# about: `create_file` / `edit_file`, allowed anywhere under `workspaces`. `run_command` is the
# one exception, denied by the default `policy.confirm_run_command()` until it is given a
# handler or replaced with `policy.allow_all()`, and an `ask_question` nobody hooked comes back
# unanswered rather than blocking.
AGENT_TOOLS = BuiltinTools.all_tools()

# The harness defaults to `gemini-3.5-flash` and takes a `thinking_level` where the Responses
# path takes a `reasoning_effort`; naming the endpoint is what pins both to one key.
AGENT_MODEL = ModelTarget(
    name="gemini-3.5-flash",
    endpoint=GeminiAPIEndpoint(
        api_key=config.gemini_api_key,
        options=GeminiModelOptions(thinking_level=ThinkingLevel.HIGH),
    ),
)


class PrimeAnswer(BaseModel):
    """Whether one number is prime, and why.

    Both this docstring and the field descriptions are serialized into the JSON schema the
    harness hands the model, so they are prompt text: they steer the answer, not just the type.
    """

    number: int = Field(..., description="The number the question was about.", examples=[37])
    is_prime: bool = Field(..., description="Whether that number is prime.")
    reason: str = Field(..., description="One sentence saying why it is or is not.")


def build_agent_config(response_schema: type[BaseModel] | None = None) -> LocalAgentConfig:
    """Builds one agent config, with the reply prompt split the way the harness wants it.

    Args:
        response_schema (type[BaseModel] | None): Pydantic model to pin the answer to, or None
            for a free-text turn.

    Returns:
        LocalAgentConfig: Config for a single local harness agent.
    """
    request_time = REQUEST_TIME_CONTEXT_PROMPT.format(
        message_created_at_asia_taipei=datetime.now(tz=TAIWAN_TIMEZONE).isoformat(
            timespec="seconds"
        )
    ).strip()
    # `TemplatedSystemInstructions` APPENDS. The harness keeps its own system prompt (tool
    # protocols, safety mandates) and `identity` replaces only the line saying who the agent
    # is, so the persona and the reply rules ride on top of it as named sections. A bare
    # `system_instructions="..."` string is silently this same shape with one untitled section
    # and no identity override, while `CustomSystemInstructions(text=...)` is the other end and
    # throws the harness prompt away — which is what handing it REPLY_PROMPT whole would mean.
    system_instructions = TemplatedSystemInstructions(
        identity=PERSONA_CHOICES.strip(),
        sections=[
            SystemInstructionSection(title="request_time", content=request_time),
            SystemInstructionSection(title="common_rules", content=COMMON_PROMPT.strip()),
        ],
    )
    # `workspaces` defaults to the current working directory and scopes every file tool to it,
    # so a dev run from the repo root reaches this repo and nothing outside it.
    return LocalAgentConfig(
        model=AGENT_MODEL,
        api_key=config.gemini_api_key,
        system_instructions=system_instructions,
        response_schema=response_schema,
        capabilities=CapabilitiesConfig(enable_subagents=True, enabled_tools=AGENT_TOOLS),
    )


async def gen_reply_agent(user_prompt: str) -> str:
    """Streams one Antigravity turn, printing thoughts, tool calls and text as they arrive.

    In plain terms this is the STREAMING shape; `gen_reply_agent_buffered` is the same turn
    read after the fact.

    Args:
        user_prompt (str): User message to send as the single turn input.

    Returns:
        str: The turn's aggregated response text.
    """
    start = time.time()
    text_parts: list[str] = []
    async with Agent(config=build_agent_config()) as agent:
        response = await agent.chat(prompt=user_prompt)
        # One cursor over `.chunks` is what makes this live. `.thoughts`, `.tool_calls` and
        # `async for delta in response` are each an INDEPENDENT cursor over the same buffer, so
        # consuming them one after another drains the whole turn on the first one and replays
        # the rest from memory; only `asyncio.gather` over them streams in parallel. The SDK
        # docs' own example is the sequential form, and it is not live: timing every delta on
        # one prompt, this loop spread its six text deltas over 0.744s while the documented
        # `.thoughts`-then-`response` pair emitted all six at the same instant, 1.7s after its
        # thought cursor had already hit the end of the stream.
        async for chunk in response.chunks:
            if isinstance(chunk, Thought):
                console.print(f"[dim]{chunk.text}[/dim]", end="")
            elif isinstance(chunk, Text):
                text_parts.append(chunk.text)
                console.print(chunk.text, end="")
            elif isinstance(chunk, ToolCall):
                # The stream carries the CALL only. A `ToolResult` is built solely for a
                # Python-side custom tool and handed straight back to the harness, so it never
                # reaches a cursor even though the chunk union names it.
                console.print(f"\n[cyan]-> {chunk.name}[/cyan] {chunk.args}")
        usage = response.usage_metadata
    end = time.time()
    console.print(f"\n{usage}")
    console.print(f"\n{AGENT_MODEL.name} on Antigravity SDK takes {end - start:.2f} seconds")
    return "".join(text_parts)


async def gen_reply_agent_buffered(user_prompt: str) -> str:
    """Runs one Antigravity turn and prints only its final text.

    In plain terms this is the NON-STREAMING shape: nothing reaches the screen until the turn
    is over. Note there is no stream on/off switch to reach for here, unlike the `stream=True`
    of `prompt_dev.py`'s Responses call — the difference is only how the caller consumes.

    Args:
        user_prompt (str): User message to send as the single turn input.

    Returns:
        str: The turn's aggregated response text.
    """
    async with Agent(config=build_agent_config()) as agent:
        response = await agent.chat(prompt=user_prompt)
        # `agent.chat()` handed back the same streamed `ChatResponse` either way; `.text()`
        # drains it and joins the text chunks. So the thoughts and tool calls this one never
        # prints are still buffered, and a cursor opened after this line replays them at once.
        response_content = await response.text()
        console.print(response_content)
        console.print(response.usage_metadata)
        return response_content


async def gen_reply_agent_structured(user_prompt: str) -> PrimeAnswer:
    """Runs one Antigravity turn whose answer is pinned to a Pydantic schema.

    Args:
        user_prompt (str): User message to send as the single turn input.

    Returns:
        PrimeAnswer: The structured answer, validated back into its model.

    Raises:
        RuntimeError: The turn ended without a finish step, so it carried no structured output.
    """
    async with Agent(config=build_agent_config(response_schema=PrimeAnswer)) as agent:
        response = await agent.chat(prompt=user_prompt)
        # `Agent.__init__` stringifies the schema into `capabilities.finish_tool_schema_json`,
        # so the answer arrives as the `finish` tool's payload and `BuiltinTools.FINISH` has to
        # be in `enabled_tools` or there is nothing to read back. What comes back is parsed
        # JSON rather than an instance of the model, hence the validate.
        structured_output = await response.structured_output()
        if structured_output is None:
            raise RuntimeError("The turn produced no finish step, so it carried no output")
        answer = PrimeAnswer.model_validate(structured_output)
        console.print(answer)
        return answer


if __name__ == "__main__":
    asyncio.run(gen_reply_agent(user_prompt="幫我上網搜尋一下今天台灣新聞"))
    # asyncio.run(gen_reply_agent_buffered(user_prompt="幫我上網搜尋一下今天台灣新聞"))
    # asyncio.run(gen_reply_agent_structured(user_prompt="為何 37 是質數?"))
