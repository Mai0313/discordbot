import dotenv
from openai import OpenAI, Stream, AzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseStreamEvent
from openai.types.responses.tool_param import ImageGeneration
from openai.types.responses.web_search_tool_param import WebSearchToolParam

dotenv.load_dotenv()


def get_aoai_reply(
    model: str, question: str, stream: bool
) -> ChatCompletion | Stream[ChatCompletionChunk]:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": question}], stream=stream
    )
    return response


def get_oai_reply(
    model: str, question: str, stream: bool
) -> ChatCompletion | Stream[ChatCompletionChunk]:
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": question}], stream=stream
    )
    return response


def get_oai_response(
    model: str, question: str, stream: bool
) -> Response | Stream[ResponseStreamEvent]:
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    responses = client.responses.create(
        model=model,
        tools=[
            WebSearchToolParam(type="web_search_preview"),
            ImageGeneration(type="image_generation"),
        ],
        input=[{"role": "user", "content": question}],
        stream=stream,
    )
    return responses


if __name__ == "__main__":
    import os

    import dotenv
    from rich.console import Console

    console = Console()
    dotenv.load_dotenv()
    model = "chatgpt-4o-latest"
    question = "Hi"
    responses = get_oai_response(model=model, question=question, stream=True)
    if isinstance(responses, (Response, ChatCompletion)):
        console.print(responses)
    else:
        for response in responses:
            console.print(response)
