import time

from openai import OpenAI
from rich.console import Console
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from discordbot.typings.llm import LLMConfig

console = Console()

COMPLETION_MODEL = "gemini-3-flash-preview"
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
message_chain: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "text", "text": "幫我畫一隻狗"}]},
]

config = LLMConfig()


def main() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    start = time.time()
    responses = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=message_chain,
        reasoning_effort="none",
        stream=False,
        service_tier="auto",
    )
    end = time.time()
    console.print(f"{COMPLETION_MODEL} takes {end - start:.2f} seconds")
    console.print(responses.choices[0].message.content)


if __name__ == "__main__":
    main()
