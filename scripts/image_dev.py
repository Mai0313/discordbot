import time
import base64
from pathlib import Path

from openai import OpenAI
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.utils.images import convert_base64_to_data_uri
from discordbot.typings.models import ModelSettings
from discordbot.cogs._gen_reply.prompts import IMAGE_PROMPT

console = Console()
config = LLMConfig()

# Mirror the @property values in cogs/gen_reply.py. Update here when the bot's
# image_model / fast_model swap, otherwise this script tests stale models.
IMAGE_MODEL = ModelSettings(name="gemini-3.1-flash-image-preview", effort=None)
FAST_MODEL = ModelSettings(name="gemini-flash-latest", effort="none")


def gen_image(user_prompt: str, image_path: str | Path | None = None) -> None:
    """Mirrors `_handle_image_reply` in cogs/gen_reply.py.

    When `image_path` is provided the script calls `client.images.edit`,
    otherwise `client.images.generate`. The result then feeds into a captioning
    pass through `client.responses.create` + IMAGE_PROMPT, exactly like the
    deployed flow.
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    start = time.time()
    if image_path is not None:
        path = Path(image_path)
        image_bytes_list = [path.read_bytes()]
        console.print(f"[bold]Editing {path} with {IMAGE_MODEL.name}...[/bold]")
        result = client.images.edit(
            image=image_bytes_list,
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
    image_responses = client.responses.create(
        model=FAST_MODEL.name,
        instructions=IMAGE_PROMPT,
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
    gen_image(user_prompt="幫我畫一隻穿西裝的柴犬，背景是台北 101")
    # gen_image(user_prompt="幫我把這張圖改成像素風格", image_path="generated.png")
