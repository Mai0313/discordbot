"""Local structured-output smoke test for AI route decisions."""

import time
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs._gen_reply.prompts import ROUTE_PROMPT

console = Console()
config = LLMConfig()

FAST_MODEL = ModelSettings(name="gemini-flash-lite-latest", effort="none")


class RouteDecision(BaseModel):
    """Model for structured output of route decisions.

    Attributes:
        decision: The categorized intent of the user input.
    """

    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"]


def use_oai_responses_parse(user_prompt: str) -> None:
    """Tests structured output parsing for route decisions using OpenAI API."""
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    start = time.time()
    responses = client.responses.parse(
        model=FAST_MODEL.name,
        instructions=ROUTE_PROMPT,
        input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
        text_format=RouteDecision,
        reasoning=FAST_MODEL.reasoning,
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "route_dev"},
        extra_body={"mock_testing_fallbacks": False},
    )
    console.print(responses.output_parsed)
    end = time.time()
    console.print(f"\n{responses.model} on Litellm takes {end - start:.2f} seconds")


if __name__ == "__main__":
    use_oai_responses_parse(user_prompt="畫一隻柴犬")
