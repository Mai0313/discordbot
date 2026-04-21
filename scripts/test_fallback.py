from openai import OpenAI
from rich.console import Console

from discordbot.typings.llm import LLMConfig

console = Console()
config = LLMConfig()


def use_non_streaming() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    responses = client.chat.completions.create(
        model="gemini-pro-latest",
        messages=[{"role": "user", "content": "ping"}],
        extra_body={"mock_testing_fallbacks": True},
        stream=False,
    )
    console.print(responses.choices[0].message.content)
    console.print(responses.model)


def use_streaming() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    responses = client.chat.completions.create(
        model="gemini-pro-latest",
        messages=[{"role": "user", "content": "ping"}],
        extra_body={"mock_testing_fallbacks": True},
        stream=True,
    )

    console.print(dict(responses.response.headers))
    for response in responses:
        model_name = response.model
        console.print(response.choices[0].delta.content, end="")
    console.print(model_name)


if __name__ == "__main__":
    use_streaming()
