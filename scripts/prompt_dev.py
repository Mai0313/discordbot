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
from discordbot.cogs._gen_reply.prompts import REPLY_PROMPT

if TYPE_CHECKING:
    from openai.types.responses.tool_param import ToolParam
    from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

console = Console()
config = LLMConfig()


def use_oai() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    tools: list[ChatCompletionToolUnionParam] = [{"googleSearch": {}}, {"urlContext": {}}]
    start = time.time()
    responses = client.chat.completions.create(
        model="gemini-3.1-pro-preview",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": REPLY_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": "為何 37 是質數?"}]},
        ],
        reasoning_effort="high",
        stream=True,
        stream_options={"include_usage": True},
        tools=tools,
        service_tier="auto",
        # extra_body={
        #     "mock_testing_fallbacks": False
        #     "request_timeout": 300,
        # },
    )
    model_name = ""
    console.print(dict(responses.response.headers))
    for response in responses:
        model_name = response.model
        if response.choices[0].delta.content:
            console.print(response.choices[0].delta.content, end="")
    end = time.time()
    console.print(f"\n{model_name} on Litellm takes {end - start:.2f} seconds")


def use_oai_responses() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    tools: list[ToolParam] = [{"googleSearch": {}}, {"urlContext": {}}]
    start = time.time()
    responses = client.responses.create(
        model="gemini-3.1-pro-preview",
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": REPLY_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": "為何 37 是質數?"}]},
        ],
        reasoning={"effort": "high", "generate_summary": "auto", "summary": "auto"},
        stream=True,
        tools=tools,
    )
    for response in responses:
        if response.type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            console.print(f"[dim]{response.delta}[/dim]", end="")
        elif response.type == "response.output_text.delta":
            console.print(response.delta, end="")
    end = time.time()
    console.print(f"\nLitellm (Responses API) takes {end - start:.2f} seconds")


def use_gemini() -> None:
    client = genai.Client(
        api_key=config.api_key, http_options=HttpOptions(base_url=config.base_url)
    )
    start = time.time()
    responses = client.models.generate_content_stream(
        model="gemini-3.1-pro-preview",
        contents=[
            {"role": "user", "parts": [{"text": REPLY_PROMPT}]},
            {"role": "user", "parts": [{"text": "為何 37 是質數?"}]},
        ],
        config=GenerateContentConfig(
            thinking_config=ThinkingConfig(include_thoughts=True, thinking_level="HIGH"),
            tools=[Tool(googleSearch=GoogleSearch(), url_context=UrlContext())],
        ),
    )
    for response in responses:
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
    console.print(f"\nGemini SDK takes {end - start:.2f} seconds")


def use_anthropic() -> None:
    client = Anthropic(base_url=config.base_url, api_key=config.api_key)
    start = time.time()
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=REPLY_PROMPT,
        messages=[
            {"role": "user", "content": REPLY_PROMPT},
            {"role": "user", "content": "為何 37 是質數?"},
        ],
        tools=[
            {"type": "web_search_20260209", "name": "web_search"},
            {"type": "web_fetch_20260209", "name": "web_fetch"},
        ],
    ) as responses:
        for response in responses:
            if response.type != "content_block_delta":
                continue
            if response.delta.type == "thinking_delta":
                console.print(f"[dim]{response.delta.thinking}[/dim]", end="")
            elif response.delta.type == "text_delta":
                console.print(response.delta.text, end="")
    end = time.time()
    console.print(f"\nAnthropic SDK takes {end - start:.2f} seconds")


if __name__ == "__main__":
    use_oai()
    # use_oai_responses()
    # use_gemini()
    # use_anthropic()
