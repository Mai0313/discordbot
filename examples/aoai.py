from openai import AzureOpenAI
from rich.console import Console

console = Console()


def get_aoai_response(api_key: str, question: str) -> str:
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-04-01-preview",
        azure_deployment="aide-o3-mini",
        azure_endpoint="https://xxx.openai.azure.com",
    )
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. I am here to help you with your questions.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message


if __name__ == "__main__":
    api_key = "..."
    question = "What is the meaning of life?"
    response = get_aoai_response(api_key, question)
    console.print(response)
