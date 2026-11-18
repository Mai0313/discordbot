"""Local structured-output smoke test for the triage call's route and effort decisions."""

import time

from openai import OpenAI
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import RouteClassification, RuntimeModelCatalog
from discordbot.cogs.gen_reply.prompts import route_prompt

console = Console()
config = LLMConfig()

TRIAGE_MODEL = RuntimeModelCatalog().triage_model


def use_oai_responses_parse(user_prompt: str) -> None:
    """Smoke-tests the triage call, which classifies the route and grades the effort together.

    Prints the parsed result and the call's latency.

    Args:
        user_prompt (str): User prompt to classify and grade.
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    start = time.time()
    responses = client.responses.parse(
        model=TRIAGE_MODEL.name,
        instructions=route_prompt(inline_image_enabled=config.inline_image_enabled),
        input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
        text_format=RouteClassification,
        reasoning=TRIAGE_MODEL.reasoning,
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "route_dev"},
    )
    console.print(f"{responses.output_parsed}")
    console.print(f"{responses.model} on Litellm takes {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    use_oai_responses_parse(user_prompt="畫一隻柴犬")
