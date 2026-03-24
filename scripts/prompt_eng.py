from openai import OpenAI
from rich.console import Console
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from discordbot.typings.llm import LLMConfig

console = Console()

COMPLETION_MODEL = "gemini-3.1-flash-lite-preview"
SYSTEM_PROMPT = """
You are a routing classifier for a Discord bot.
Decide whether the bot should answer normally, generate an image, or summarize recent chat history.

Reply with exactly one word:
- IMAGE
- QA
- SUMMARY

Choose IMAGE only when the user explicitly wants the bot to create, draw, render, generate, or make an image.
Choose SUMMARY when the user explicitly asks the bot to summarize, recap, or give a summary of the recent chat/conversation/messages.
Choose QA for everything else, including normal questions, image analysis, captioning, or discussions about art that do not ask the bot to actually generate an image.
If you are not sure, reply QA.
"""
message_chain: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "text", "text": "幫我畫一隻狗"}]},
]

config = LLMConfig()


def main() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    responses = client.chat.completions.create(
        model=COMPLETION_MODEL, messages=message_chain, reasoning_effort="none", stream=False
    )
    console.print(responses.choices[0].message.content)


if __name__ == "__main__":
    main()
