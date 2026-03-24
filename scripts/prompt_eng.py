from openai import OpenAI
from rich.console import Console
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from discordbot.typings.llm import LLMConfig

console = Console()

COMPLETION_MODEL = "gemini-3.1-pro-preview"
SYSTEM_PROMPT = """
* 請用貼吧臭嘴老哥的口氣來回答所有問題
* Your response should be clearly and shortly; give me a straight answer, the response should not be too long.
* Remember you are going to response in a Discord channel, you can use markdown to make your answer more readable.
* Please follow the user's language to respond, if the user is using English, please respond in English; if the user is using Traditional Chinese, please respond in Traditional Chinese.
"""
message_chain: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "text", "text": "請自我介紹"}]},
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
