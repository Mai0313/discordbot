"""Local prompt development helpers for LiteLLM and provider-native SDKs."""

import time
from typing import TYPE_CHECKING, cast
from collections.abc import Iterator

from google import genai
from openai import OpenAI
from anthropic import Anthropic
from rich.console import Console
from google.genai.types import HttpOptions
from openai.types.responses import (
    ResponseCreatedEvent,
    ResponseCompletedEvent,
    ResponseTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningSummaryTextDeltaEvent,
)
from google.genai.interactions import (
    StepDelta,
    TextDelta,
    URLContext,
    TextContent,
    GoogleSearch,
    AllowlistParam,
    EnvironmentParam,
    TextContentParam,
    # VideoContentParam,
    AllowlistEntryParam,
    ThoughtSummaryDelta,
    GenerationConfigParam,
    InteractionCreatedEvent,
)

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs.gen_reply.prompts import REPLY_PROMPT

if TYPE_CHECKING:
    from anthropic.types.tool_param import ToolParam as AnthropicToolParam
    from openai.types.responses.tool_param import ToolParam
    from openai.types.responses.response_input_param import ResponseInputParam
    from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

console = Console()
config = LLMConfig()

# Mirror the @property value in cogs/gen_reply/cog.py. slow_model has a time-of-day
# dispatch in production (peak hours swap to gemini-flash-latest); for
# dev we pin to the off-peak default. Swap manually when testing peak behaviour.
SLOW_MODEL = ModelSettings(name="gemini-flash-latest", effort="high")


def gen_reply(user_prompt: str) -> None:
    """Streams a dev reply through the LiteLLM Responses API.

    Args:
        user_prompt (str): User message to send as the single prompt input.
    """
    message_list = [{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}]
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    start = time.time()
    responses = client.responses.create(
        model=SLOW_MODEL.name,
        instructions=REPLY_PROMPT,
        input=cast("ResponseInputParam", message_list),
        reasoning=SLOW_MODEL.reasoning,
        tools=SLOW_MODEL.tools,
        stream=True,
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "prompt_dev"},
        extra_body={
            "cache": {
                "no-cache": True  # Skip cache check, get fresh response
            }
        },
    )
    model_name = ""
    for response in responses:
        if isinstance(response, (ResponseCreatedEvent, ResponseCompletedEvent)):
            model_name = response.response.model
        elif isinstance(
            response, (ResponseReasoningSummaryTextDeltaEvent, ResponseReasoningTextDeltaEvent)
        ):
            console.print(f"[dim]{response.delta}[/dim]", end="")
        elif isinstance(response, ResponseTextDeltaEvent):
            console.print(response.delta, end="")
    end = time.time()
    console.print(f"\n{responses.response.headers}")
    console.print(f"\n{model_name} on Litellm (Responses API) takes {end - start:.2f} seconds")


def gen_reply_chat(user_prompt: str) -> None:
    """Streams a dev reply through LiteLLM Chat Completions.

    Args:
        user_prompt (str): User message to send as the single prompt input.
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    tools: list[ToolParam] = SLOW_MODEL.tools
    start = time.time()
    responses = client.chat.completions.create(
        model=SLOW_MODEL.name,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": REPLY_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
        reasoning_effort=SLOW_MODEL.effort,
        stream=True,
        stream_options={"include_usage": True},
        # Responses-API tool shape (SLOW_MODEL.tools) sent through the Chat Completions
        # endpoint; LiteLLM translates it, but the two SDKs' tool TypedDicts differ statically.
        tools=cast("list[ChatCompletionToolUnionParam]", tools),
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "prompt_dev"},
        extra_body={
            "cache": {
                "no-cache": True  # Skip cache check, get fresh response
            }
        },
    )
    model_name = ""
    for response in responses:
        model_name = response.model
        if response.choices and response.choices[0].delta.content:
            console.print(response.choices[0].delta.content, end="")
    end = time.time()
    console.print(f"\n{responses.response.headers}")
    console.print(f"\n{model_name} on Litellm (Chat Completions) takes {end - start:.2f} seconds")


def gen_reply_gemini(user_prompt: str) -> None:
    """Streams a dev reply through the native Gemini SDK.

    Args:
        user_prompt (str): User message to send as the comparison prompt.

    Raises:
        RuntimeError: The SDK returned an interaction instead of the requested event stream.
    """
    client = genai.Client(
        api_key=config.api_key,
        http_options=HttpOptions(
            base_url=config.base_url,
            # NOTICE: extra_body properties are not supported in `.interactions` yet
            # But this is fine for leaving it here.
            extra_body={
                "cache": {
                    "no-cache": True  # Skip cache check, get fresh response
                }
            },
        ),
    )
    thinking_level = SLOW_MODEL.effort
    if thinking_level not in {"minimal", "low", "medium", "high"}:
        raise RuntimeError(f"Unsupported Gemini interactions thinking level: {thinking_level}")
    start = time.time()
    responses = client.interactions.create(
        model=SLOW_MODEL.name,
        system_instruction=REPLY_PROMPT,
        service_tier="auto",
        input=[
            TextContentParam(text=user_prompt, type="text")
            # Check this docs for more info: https://ai.google.dev/gemini-api/docs/video-understanding.md.txt
            # We can send a YouTube video URL to the model by this way:
            # VideoContentParam(uri="https://www.youtube.com/watch?v=jNQXAC9IVRw", type="video"),
        ],
        environment=EnvironmentParam(
            type="remote", network=AllowlistParam(allowlist=[AllowlistEntryParam(domain="*")])
        ),
        generation_config=GenerationConfigParam(
            thinking_level=thinking_level, thinking_summaries="auto"
        ),
        tools=[
            URLContext(type="url_context"),
            GoogleSearch(search_types=["web_search"], type="google_search"),
        ],
        stream=True,
    )
    # `stream=True` returns the event stream, but a plain `str` model name misses the SDK's
    # `Model` literal overloads, so the call types as `Interaction | Stream[...]`. Narrow by
    # excluding the interaction (it is iterable but not an iterator). Pydantic's `Interaction`
    # also implements `__iter__` (field iteration), so ty keeps a residual `tuple[str, Any]`
    # branch through the isinstance guard; cast the stream to the events we actually read.
    if not isinstance(responses, Iterator):
        raise RuntimeError("Gemini interactions.create returned an interaction, not a stream")
    stream = cast("Iterator[InteractionCreatedEvent | StepDelta]", responses)
    model_name = ""
    for response in stream:
        if isinstance(response, InteractionCreatedEvent):
            model_name = response.interaction.model or ""
        elif isinstance(response, StepDelta):
            if isinstance(response.delta, ThoughtSummaryDelta) and isinstance(
                response.delta.content, TextContent
            ):
                console.print(f"[dim]{response.delta.content.text}[/dim]", end="")
            elif isinstance(response.delta, TextDelta):
                console.print(response.delta.text, end="")
    end = time.time()
    console.print(f"\n{model_name} on Gemini SDK takes {end - start:.2f} seconds")


def gen_reply_anthropic(user_prompt: str) -> None:
    """Streams a dev reply through the native Anthropic SDK.

    Args:
        user_prompt (str): User message to send as the comparison prompt.
    """
    client = Anthropic(base_url=config.base_url, api_key=config.api_key)
    start = time.time()
    with client.messages.stream(
        model=SLOW_MODEL.name,
        system=REPLY_PROMPT,
        service_tier="auto",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_prompt}],
        # Same cross-SDK tool shape note as `gen_reply_chat`.
        tools=cast("list[AnthropicToolParam]", SLOW_MODEL.tools),
    ) as responses:
        model_name = ""
        for response in responses:
            if response.type == "message_start":
                model_name = response.message.model
            elif response.type != "content_block_delta":
                continue
            elif response.delta.type == "thinking_delta":
                console.print(f"[dim]{response.delta.thinking}[/dim]", end="")
            elif response.delta.type == "text_delta":
                console.print(response.delta.text, end="")
    end = time.time()
    console.print(f"\n{responses.response.headers}")
    console.print(f"\n{model_name} on Anthropic SDK takes {end - start:.2f} seconds")


if __name__ == "__main__":
    gen_reply(user_prompt="為何 37 是質數?")
    # gen_reply_chat(user_prompt="為何 37 是質數?")
    # gen_reply_gemini(user_prompt="為何 37 是質數?")
    # gen_reply_anthropic(user_prompt="為何 37 是質數?")
