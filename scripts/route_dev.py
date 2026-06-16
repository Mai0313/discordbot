"""Local structured-output smoke test for AI route and effort decisions."""

import time

from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import EffortGrade, ModelSettings, RouteClassification
from discordbot.cogs._gen_reply.prompts import ROUTE_PROMPT, EFFORT_PROMPT

console = Console()
config = LLMConfig()

FAST_MODEL = ModelSettings(name="gemini-flash-lite-latest", effort="none")


def _smoke_parse(
    client: OpenAI, user_prompt: str, label: str, instructions: str, text_format: type[BaseModel]
) -> None:
    """Runs one structured-output parse call and prints the result and latency."""
    start = time.time()
    responses = client.responses.parse(
        model=FAST_MODEL.name,
        instructions=instructions,
        input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
        text_format=text_format,
        reasoning=FAST_MODEL.reasoning,
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "route_dev"},
        extra_body={"mock_testing_fallbacks": False},
    )
    console.print(f"[{label}] {responses.output_parsed}")
    console.print(f"{responses.model} on Litellm takes {time.time() - start:.2f} seconds")


def use_oai_responses_parse(user_prompt: str) -> None:
    """Smoke-tests the parallel route classification and effort grading calls."""
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    _smoke_parse(
        client=client,
        user_prompt=user_prompt,
        label="route",
        instructions=ROUTE_PROMPT,
        text_format=RouteClassification,
    )
    _smoke_parse(
        client=client,
        user_prompt=user_prompt,
        label="effort",
        instructions=EFFORT_PROMPT,
        text_format=EffortGrade,
    )


if __name__ == "__main__":
    use_oai_responses_parse(user_prompt="畫一隻柴犬")
