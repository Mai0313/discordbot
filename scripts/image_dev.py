"""Local image generation and edit smoke test for the bot image models."""

import time
import base64
from pathlib import Path

from openai import OpenAI
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.utils.images import convert_base64_to_data_uri
from discordbot.typings.models import ModelSettings
from discordbot.cogs._gen_reply.prompts import IMAGE_REPLY_PROMPT

console = Console()
config = LLMConfig()

# Mirror the @property values in cogs/gen_reply.py. Update here when the bot's
# image_model / media_reply_model swap, otherwise this script tests stale models.
IMAGE_MODEL = ModelSettings(name="gemini-3.1-flash-image")
MEDIA_REPLY_MODEL = ModelSettings(name="gemini-flash-latest", effort="low")


def gen_image(user_prompt: str, image_path: str | Path | None = None) -> None:
    """Runs the dev image generation or edit flow and writes the PNG result.

    The raw user request is sent straight to the image model (no prompt director), then the
    reply stage answers about the image as the bot would (production also feeds it history
    and the user's memory).
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    start = time.time()
    if image_path is not None:
        path = Path(image_path)
        source_bytes = path.read_bytes()
        console.print(f"[bold]Editing {path} with {IMAGE_MODEL.name}...[/bold]")
        result = client.images.edit(
            image=[source_bytes],
            prompt=user_prompt,
            model=IMAGE_MODEL.name,
            n=1,
            response_format="b64_json",
            quality="auto",
            size="auto",
            extra_headers={"x-litellm-end-user-id": "image_dev"},
        )
    else:
        console.print(f"[bold]Generating image with {IMAGE_MODEL.name}...[/bold]")
        result = client.images.generate(
            prompt=user_prompt,
            model=IMAGE_MODEL.name,
            n=1,
            response_format="b64_json",
            quality="auto",
            size="auto",
            extra_headers={"x-litellm-end-user-id": "image_dev"},
        )

    if not result.data:
        raise ValueError("Image operation returned no results")
    image_b64 = result.data[0].b64_json
    if image_b64 is None:
        raise ValueError("Image operation returned no b64_json")

    image_url = convert_base64_to_data_uri(base64_image=image_b64)
    reply_responses = client.responses.create(
        model=MEDIA_REPLY_MODEL.name,
        instructions=IMAGE_REPLY_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"They asked: {user_prompt}\nThis is the image you just made for "
                            "them in response. Reply to them about it."
                        ),
                    },
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ],
            }
        ],
        reasoning=MEDIA_REPLY_MODEL.reasoning,
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "image_dev"},
        extra_body={"mock_testing_fallbacks": False},
    )
    image_reply = (reply_responses.output_text or "").strip()

    model_name = IMAGE_MODEL.name
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    output_path = Path(
        f"edited_{model_name}.png" if image_path is not None else f"{model_name}.png"
    )
    output_path.write_bytes(data=base64.b64decode(s=image_b64))

    end = time.time()
    console.print(f"[green]Saved to {output_path}[/green]")
    console.print(f"Reply: {image_reply}")
    console.print(f"\n{IMAGE_MODEL.name} on Litellm takes {end - start:.2f} seconds")


if __name__ == "__main__":
    gen_image(user_prompt="幫我畫山田小姐")
    # gen_image(user_prompt="幫我把這張圖改成像素風格", image_path="generated.png")
