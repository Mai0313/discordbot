"""Local prompt development helpers for LiteLLM and provider-native SDKs."""

import time
from typing import TYPE_CHECKING

from google import genai
from openai import OpenAI
from anthropic import Anthropic
from rich.console import Console
from google.genai.types import (
    Tool,
    UrlContext,
    HttpOptions,
    GoogleSearch,
    ThinkingConfig,
    GenerateContentConfig,
)

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs._gen_reply.prompts import REPLY_PROMPT

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

console = Console()
config = LLMConfig()

# Mirror the @property value in cogs/gen_reply.py. slow_model has a time-of-day
# dispatch in production (peak hours swap to gemini-3-flash-preview); for
# dev we pin to the off-peak default. Swap manually when testing peak behaviour.
SLOW_MODEL = ModelSettings(name="gemini-pro-latest", effort="high")


def gen_reply(user_prompt: str) -> None:
    """Streams a dev reply through the LiteLLM Responses API.

    Mirrors `_handle_message_reply` in `cogs/gen_reply.py` by sending
    `REPLY_PROMPT`, the configured slow model, reasoning settings, and model
    tools through `client.responses.create`. Prints reasoning deltas dimmed,
    output text deltas as they stream, and elapsed time to the console.

    Args:
        user_prompt: User message to send as the single prompt input.
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    start = time.time()
    responses = client.responses.create(
        model=SLOW_MODEL.name,
        instructions=REPLY_PROMPT,
        input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
        reasoning=SLOW_MODEL.reasoning,
        tools=SLOW_MODEL.tools,
        stream=True,
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "prompt_dev"},
        extra_body={"mock_testing_fallbacks": False},
    )
    model_name = ""
    for response in responses:
        if response.type in {"response.created", "response.completed"}:
            model_name = response.response.model
        elif response.type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            console.print(f"[dim]{response.delta}[/dim]", end="")
        elif response.type == "response.output_text.delta":
            console.print(response.delta, end="")
    end = time.time()
    console.print(f"\n{model_name} on Litellm (Responses API) takes {end - start:.2f} seconds")


def gen_reply_chat(user_prompt: str) -> None:
    """Streams a dev reply through LiteLLM Chat Completions.

    Uses the same `REPLY_PROMPT`, configured slow model, reasoning effort, and
    tools as the deployed reply flow, but sends them through
    `client.chat.completions.create` for comparison. Prints streamed text and
    elapsed time to the console.

    Args:
        user_prompt: User message to send as the single prompt input.
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    tools: list[ChatCompletionToolUnionParam] = SLOW_MODEL.tools
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
        tools=tools,
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "prompt_dev"},
        extra_body={"mock_testing_fallbacks": False},
    )
    model_name = ""
    for response in responses:
        model_name = response.model
        if response.choices and response.choices[0].delta.content:
            console.print(response.choices[0].delta.content, end="")
    end = time.time()
    console.print(f"\n{model_name} on Litellm (Chat Completions) takes {end - start:.2f} seconds")


def gen_reply_gemini(user_prompt: str) -> None:
    """Streams a dev reply through the native Gemini SDK.

    Sends `REPLY_PROMPT` and the user prompt through
    `client.models.generate_content_stream` using the configured slow model,
    thinking config, Google Search, and URL context tools. Prints thought parts
    dimmed, answer text as it streams, and elapsed time to the console.

    Args:
        user_prompt: User message to send as the comparison prompt.
    """
    client = genai.Client(
        api_key=config.api_key,
        http_options=HttpOptions(
            base_url=config.base_url, extra_body={"mock_testing_fallbacks": False}
        ),
    )
    start = time.time()
    responses = client.models.generate_content_stream(
        model=SLOW_MODEL.name,
        contents=[
            {"role": "user", "parts": [{"text": REPLY_PROMPT}]},
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
        config=GenerateContentConfig(
            thinking_config=ThinkingConfig(
                include_thoughts=True, thinking_level=SLOW_MODEL.effort.upper()
            ),
            tools=[Tool(googleSearch=GoogleSearch(), url_context=UrlContext())],
        ),
    )
    model_name = ""
    for response in responses:
        model_name = response.model_version or model_name
        if not response.candidates or not response.candidates[0].content.parts:
            continue
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                console.print(f"[dim]{part.text}[/dim]", end="")
            else:
                console.print(part.text, end="")
    end = time.time()
    console.print(f"\n{model_name} on Gemini SDK takes {end - start:.2f} seconds")


def gen_reply_anthropic(user_prompt: str) -> None:
    """Streams a dev reply through the native Anthropic SDK.

    Uses `REPLY_PROMPT` with a pinned Claude model instead of `SLOW_MODEL`, then
    streams responses from `client.messages.stream` with adaptive thinking and
    the model's tool configuration. Prints thinking deltas dimmed, answer text
    as it streams, and elapsed time to the console.

    Args:
        user_prompt: User message to send as the comparison prompt.
    """
    model = ModelSettings(name="claude-haiku-4-5", effort="medium")
    client = Anthropic(base_url=config.base_url, api_key=config.api_key)
    start = time.time()
    with client.messages.stream(
        model=model.name,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=REPLY_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        tools=model.tools,
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
    console.print(f"\n{model_name} on Anthropic SDK takes {end - start:.2f} seconds")


if __name__ == "__main__":
    # gen_reply_chat(user_prompt="為何 37 是質數?")
    gen_reply(user_prompt="為何 37 是質數?")
    # gen_reply_gemini(user_prompt="為何 37 是質數?")
    # gen_reply_anthropic(user_prompt="為何 37 是質數?")
