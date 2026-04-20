import time
from typing import TYPE_CHECKING

from google import genai
from openai import OpenAI
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
    from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam


MODEL = "gemini-3.1-pro-preview"

console = Console()
config = LLMConfig()


def use_oai() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    tools: list[ChatCompletionToolUnionParam] = [{"googleSearch": {}}, {"urlContext": {}}]
    start = time.time()
    responses = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": REPLY_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": "為何 37 是質數?"}]},
        ],
        reasoning_effort="low",
        stream=True,
        stream_options={"include_usage": True},
        tools=tools,
        service_tier="auto",
    )
    console.print(dict(responses.response.headers))
    for response in responses:
        if response.choices[0].delta.content:
            console.print(response.choices[0].delta.content, end="")
    end = time.time()
    console.print(f"\n{MODEL} takes {end - start:.2f} seconds")


def use_gemini() -> None:
    client = genai.Client(
        api_key=config.api_key, http_options=HttpOptions(base_url=config.base_url)
    )

    start = time.time()
    responses = client.models.generate_content_stream(
        model=MODEL,
        contents=[
            {"role": "user", "parts": [{"text": REPLY_PROMPT}]},
            {"role": "user", "parts": [{"text": "為何 37 是質數?"}]},
        ],
        config=GenerateContentConfig(
            thinking_config=ThinkingConfig(include_thoughts=True, thinking_level="HIGH"),
            tools=[Tool(googleSearch=GoogleSearch(), url_context=UrlContext())],
        ),
    )
    for chunk in responses:
        if not chunk.candidates or not chunk.candidates[0].content.parts:
            continue
        for part in chunk.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                console.print(f"[dim]{part.text}[/dim]", end="")
            else:
                console.print(part.text, end="")
    end = time.time()
    console.print(f"\n{MODEL} takes {end - start:.2f} seconds")


if __name__ == "__main__":
    use_oai()
    use_gemini()
