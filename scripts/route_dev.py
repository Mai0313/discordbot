"""Local structured-output smoke test for the triage call's route and effort decisions."""

import time

from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import RouteClassification, RuntimeModelCatalog
from discordbot.cogs.gen_reply.prompts import ROUTE_PROMPT, ROUTE_INLINE_IMAGE_SECTION

console = Console()
config = LLMConfig()

TRIAGE_MODEL = RuntimeModelCatalog().triage_model


def _smoke_parse(
    client: OpenAI, user_prompt: str, label: str, instructions: str, text_format: type[BaseModel]
) -> None:
    """Runs one structured-output parse call and prints the result and latency."""
    start = time.time()
    responses = client.responses.parse(
        model=TRIAGE_MODEL.name,
        instructions=instructions,
        input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
        text_format=text_format,
        reasoning=TRIAGE_MODEL.reasoning,
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "route_dev"},
    )
    console.print(f"[{label}] {responses.output_parsed}")
    console.print(f"{responses.model} on Litellm takes {time.time() - start:.2f} seconds")


def use_oai_responses_parse(user_prompt: str) -> None:
    """Smoke-tests the triage call, which classifies the route and grades the effort together.

    Args:
        user_prompt (str): User prompt to classify and grade.
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    _smoke_parse(
        client=client,
        user_prompt=user_prompt,
        label="route",
        instructions=ROUTE_PROMPT
        + (ROUTE_INLINE_IMAGE_SECTION if config.inline_image_enabled else ""),
        text_format=RouteClassification,
    )


if __name__ == "__main__":
    use_oai_responses_parse(user_prompt="畫一隻柴犬")
