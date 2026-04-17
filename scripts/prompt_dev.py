import os
import time
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from rich.console import Console
from google.genai.types import (
    Tool,
    UrlContext,
    GoogleSearch,
    ThinkingConfig,
    GenerateContentConfig,
)

from discordbot.typings.llm import LLMConfig

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

load_dotenv()

console = Console()

SYSTEM_PROMPT = """
You are a routing classifier for a Discord bot.
Decide whether the bot should answer normally, generate an image, edit an existing image, generate a video, or summarize recent chat history.

Reply with exactly one word:
- IMAGE
- EDIT
- VIDEO
- QA
- SUMMARY

Choose IMAGE only when the user explicitly wants the bot to create, draw, render, generate, or make a brand-new image from scratch.
Choose EDIT when the user has attached or referenced an image and explicitly wants to modify, edit, alter, transform, or retouch that image.
Choose VIDEO when the user explicitly wants the bot to create, generate, or make a video or animation.
Choose SUMMARY when the user explicitly asks the bot to summarize, recap, or give a summary of the recent chat/conversation/messages.
Choose QA for everything else, including normal questions, image analysis, captioning, or discussions about art that do not ask the bot to actually generate or edit an image.
If you are not sure, reply QA.
"""

MODEL = "gemini-flash-latest"

config = LLMConfig()


def use_oai() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    tools: list[ChatCompletionToolUnionParam] = [{"googleSearch": {}}, {"urlContext": {}}]
    start = time.time()
    responses = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": "幫我畫一隻狗"}]},
        ],
        reasoning_effort="low",
        stream=True,
        stream_options={"include_usage": True},
        tools=tools,
        service_tier="priority",
    )
    for response in responses:
        if response.choices[0].delta.content:
            console.print(response.choices[0].delta.content, end="")
    end = time.time()
    console.print(f"{MODEL} takes {end - start:.2f} seconds")


def use_gemini() -> None:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    start = time.time()
    responses = client.models.generate_content_stream(
        model=MODEL,
        contents=[
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
            {"role": "user", "parts": [{"text": "幫我畫一隻狗"}]},
        ],
        config=GenerateContentConfig(
            thinking_config=ThinkingConfig(thinking_level="LOW"),
            tools=[Tool(googleSearch=GoogleSearch(), url_context=UrlContext())],
        ),
    )
    for response in responses:
        console.print(response.text, end="")
    end = time.time()
    console.print(f"{MODEL} takes {end - start:.2f} seconds")


if __name__ == "__main__":
    use_oai()
    console.print()
    use_gemini()
