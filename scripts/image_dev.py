"""Local image generation and edit smoke test for the bot image models."""

import time
import base64
from pathlib import Path

from openai import OpenAI
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.utils.images import convert_base64_to_data_uri
from discordbot.typings.models import ModelSettings
from discordbot.cogs._gen_reply.prompts import IMAGE_PROMPT, DESCRIPTION_PROMPT

console = Console()
config = LLMConfig()

# Mirror the @property values in cogs/gen_reply.py. Update here when the bot's
# image_model / fast_model swap, otherwise this script tests stale models.
IMAGE_MODEL = ModelSettings(name="gemini-3.1-flash-image")
FAST_MODEL = ModelSettings(name="gemini-flash-latest", effort="none")

# The director that refines the user's request into a rich image prompt before the image
# model draws it. Runs with grounding tools so a thin request can be looked up first.
PROMPT_MODEL = ModelSettings(name="gemini-flash-latest", effort="medium")


def _build_image_prompt(client: OpenAI, user_prompt: str, image_url: str | None = None) -> str:
    """Refines the user's request into a detailed, self-contained image prompt.

    Runs PROMPT_MODEL with its grounding tools so a thin user request (e.g. "draw the heroine
    of some anime") is looked up and resolved into a concrete prompt before the image model
    ever sees it. When a reference image is supplied it rides along so the draft is an edit
    instruction grounded in that image. Falls back to the raw request if the draft is empty.
    """
    director_content: list[dict[str, str]] = [
        {"type": "input_text", "text": f"User image request:\n{user_prompt}"}
    ]
    if image_url is not None:
        director_content.append({"type": "input_image", "image_url": image_url, "detail": "auto"})

    console.print(f"[bold]Drafting image prompt with {PROMPT_MODEL.name}...[/bold]")
    prompt_responses = client.responses.create(
        model=PROMPT_MODEL.name,
        instructions=IMAGE_PROMPT,
        input=[{"role": "user", "content": director_content}],
        reasoning=PROMPT_MODEL.reasoning,
        tools=list(PROMPT_MODEL.tools),
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "image_dev"},
        extra_body={"mock_testing_fallbacks": False},
    )
    refined_prompt = (prompt_responses.output_text or "").strip()
    return refined_prompt or user_prompt


def gen_image(user_prompt: str, image_path: str | Path | None = None) -> None:
    """Runs the dev image generation or edit flow and writes the PNG result.

    A director round (`_build_image_prompt`) first refines the user's request into a detailed
    prompt, then the image model generates or edits from it. The caption stage describes the
    result for the Discord reply.
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    start = time.time()
    if image_path is not None:
        path = Path(image_path)
        source_bytes = path.read_bytes()
        source_url = convert_base64_to_data_uri(
            base64_image=base64.b64encode(source_bytes).decode()
        )
        refined_prompt = _build_image_prompt(
            client=client, user_prompt=user_prompt, image_url=source_url
        )
        console.print(f"[cyan]Refined prompt:[/cyan]\n{refined_prompt}\n")
        console.print(f"[bold]Editing {path} with {IMAGE_MODEL.name}...[/bold]")
        result = client.images.edit(
            image=[source_bytes],
            prompt=refined_prompt,
            model=IMAGE_MODEL.name,
            n=1,
            response_format="b64_json",
            quality="auto",
            size="auto",
            extra_headers={"x-litellm-end-user-id": "image_dev"},
        )
    else:
        refined_prompt = _build_image_prompt(client=client, user_prompt=user_prompt)
        console.print(f"[cyan]Refined prompt:[/cyan]\n{refined_prompt}\n")
        console.print(f"[bold]Generating image with {IMAGE_MODEL.name}...[/bold]")
        result = client.images.generate(
            prompt=refined_prompt,
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
    image_responses = client.responses.create(
        model=FAST_MODEL.name,
        instructions=DESCRIPTION_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Describe this generated image briefly for the Discord reply.",
                    },
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ],
            }
        ],
        reasoning=FAST_MODEL.reasoning,
        service_tier="auto",
        extra_headers={"x-litellm-end-user-id": "image_dev"},
        extra_body={"mock_testing_fallbacks": False},
    )
    image_description = (image_responses.output_text or "").strip()

    model_name = IMAGE_MODEL.name
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    output_path = Path(
        f"edited_{model_name}.png" if image_path is not None else f"{model_name}.png"
    )
    output_path.write_bytes(data=base64.b64decode(s=image_b64))

    end = time.time()
    console.print(f"[green]Saved to {output_path}[/green]")
    console.print(f"Caption: {image_description}")
    console.print(f"\n{IMAGE_MODEL.name} on Litellm takes {end - start:.2f} seconds")


if __name__ == "__main__":
    gen_image(user_prompt="幫我畫山田小姐")
    # gen_image(user_prompt="幫我把這張圖改成像素風格", image_path="generated.png")
