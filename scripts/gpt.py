import time

from openai import OpenAI, AzureOpenAI
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.cogs._gen_reply.prompts import REPLY_PROMPT

console = Console()
config = LLMConfig()

FAST_MODEL = "gemini-3-flash-preview"


def use_chat() -> None:
    client = AzureOpenAI(
        azure_endpoint=config.base_url, api_version="2024-05-01-preview", api_key=config.api_key
    )
    start = time.time()
    responses = client.chat.completions.create(
        model="azure/gpt-5.4",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": REPLY_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": "為何 37 是質數?"}]},
        ],
        reasoning_effort="medium",
        stream=True,
        stream_options={"include_usage": True},
        service_tier="priority",
        extra_body={"mock_testing_fallbacks": False},
    )
    model_name = ""
    for response in responses:
        model_name = response.model
        if response.choices[0].delta.content:
            console.print(response.choices[0].delta.content, end="")
    end = time.time()
    console.print(f"\n{model_name} on Litellm takes {end - start:.2f} seconds")


def use_responses() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    start = time.time()
    responses = client.responses.create(
        model="azure/gpt-5.4",
        instructions=REPLY_PROMPT,
        input=[{"role": "user", "content": [{"type": "input_text", "text": "為何 37 是質數?"}]}],
        reasoning={"effort": "medium", "summary": "auto"},
        stream=True,
        extra_body={"mock_testing_fallbacks": False},
    )
    model_name = ""
    for response in responses:
        if response.type in {"response.created", "response.completed"}:
            model_name = response.response.model
        elif response.type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            console.print(f"[dim]{response.delta}[/dim]", end="")
        elif response.type == "response.output_text.delta":
            console.print(response.delta, end="")
    end = time.time()
    console.print(f"\n{model_name} on Litellm (Responses API) takes {end - start:.2f} seconds")


if __name__ == "__main__":
    use_chat()
    use_responses()
