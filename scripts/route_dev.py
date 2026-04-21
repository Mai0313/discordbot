from typing import Literal

from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.cogs._gen_reply.prompts import ROUTE_PROMPT

console = Console()
config = LLMConfig()

FAST_MODEL = "gemini-3-flash-preview"


class RouteDecision(BaseModel):
    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"]


def use_oai_responses_parse() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    responses = client.responses.parse(
        model=FAST_MODEL,
        instructions=ROUTE_PROMPT,
        input=[
            {"role": "user", "content": [{"type": "input_text", "text": "幫我畫一隻穿西裝的柴犬"}]}
        ],
        text_format=RouteDecision,
        reasoning={"effort": "none"},
        extra_body={"mock_testing_fallbacks": False},
    )
    console.print(responses.output_parsed)


if __name__ == "__main__":
    use_oai_responses_parse()
