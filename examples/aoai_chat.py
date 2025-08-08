import dotenv
from openai import OpenAI, AzureOpenAI

dotenv.load_dotenv()


def get_aoai_reply(model: str, question: str) -> str:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message


def get_oai_reply(model: str, question: str) -> str:
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message


def get_oai_response(model: str, question: str) -> str:
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(model=model, input=[{"role": "user", "content": question}])
    return response.output_text


if __name__ == "__main__":
    import os

    import dotenv
    from rich.console import Console

    console = Console()
    dotenv.load_dotenv()
    model = "gpt-5"
    question = "What is the meaning of life?"
    response = get_oai_response(model=model, question=question)
    console.print(response)
