from typing import TYPE_CHECKING, cast

from openai import OpenAI
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs._gen_reply.prompts import REPLY_PROMPT

if TYPE_CHECKING:
    from openai.types.responses.response_input_param import ResponseInputParam

console = Console()
config = LLMConfig()

# Mirror the @property value in cogs/gen_reply.py. slow_model has a time-of-day
# dispatch in production (peak hours swap to gemini-flash-latest); for
# dev we pin to the off-peak default. Swap manually when testing peak behaviour.
SLOW_MODEL = ModelSettings(name="gemini-flash-latest", effort="low")


def gen_reply(user_prompt: str) -> None:
    """Streams a dev reply through the LiteLLM Responses API.

    Mirrors `_handle_message_reply` in `cogs/gen_reply.py` by sending
    `REPLY_PROMPT`, the configured slow model, reasoning settings, and model
    tools through `client.responses.create`. Prints reasoning deltas dimmed,
    output text deltas as they stream, and elapsed time to the console.

    Args:
        user_prompt: User message to send as the single prompt input.
    """
    message_list = [{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}]
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    responses = client.responses.create(
        model=SLOW_MODEL.name,
        instructions=REPLY_PROMPT,
        input=cast("ResponseInputParam", message_list),
        reasoning=SLOW_MODEL.reasoning,
        tools=SLOW_MODEL.tools,
        stream=True,
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "voice_dev"},
        extra_body={
            "mock_testing_fallbacks": False,
            "cache": {
                "no-cache": True  # Skip cache check, get fresh response
            },
        },
    )
    full_content = ""
    for response in responses:
        if response.type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            console.print(f"[dim]{response.delta}[/dim]", end="")
        elif response.type == "response.output_text.delta":
            console.print(response.delta, end="")
        elif response.type == "response.output_text.done":
            full_content = response.text
    audio_responses = client.audio.speech.create(
        input=full_content,
        model="gemini-3.1-flash-tts-preview",
        voice="Zephyr",
        instructions="",
        speed=1.3,
        extra_headers={"x-litellm-end-user-id": "voice_dev"},
    )
    audio_responses.write_to_file("./speech.mp3")


if __name__ == "__main__":
    gen_reply(user_prompt="為何 37 是質數?")
