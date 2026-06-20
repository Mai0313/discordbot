"""Local video generation smoke test for the bot video model."""

import time
from pathlib import Path

from google import genai
from rich.console import Console
from google.genai.types import (
    Image,
    GenerateVideosConfig,
    VideoGenerationReferenceType,
    VideoGenerationReferenceImage,
)

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings

console = Console()
config = LLMConfig()

# Mirror the @property value in cogs/gen_reply.py. Update here when the bot's
# video_model swaps, otherwise this script tests a stale model.
VIDEO_MODEL = ModelSettings(name="veo-3.1-generate-preview")
POLL_INTERVAL = 5


def gen_video(user_prompt: str, image_paths: list[str] | None = None) -> None:
    """Runs the dev video generation flow and saves the MP4 result.

    Mirrors `_handle_video_reply` in `cogs/gen_reply.py`: submits a native Gemini (Veo)
    `generate_videos` job, polls the operation until it completes, downloads the video, and
    writes it to `generated.mp4`. Any `image_paths` ride as asset reference images (up to three).

    Args:
        user_prompt: Prompt to send to the video generation endpoint.
        image_paths: Optional local image files used as asset reference images.

    Raises:
        RuntimeError: The video job ended without a generated video.
    """
    client = genai.Client(api_key=config.gemini_api_key)

    images = [
        (Path(path).read_bytes(), f"image/{Path(path).suffix.lstrip('.') or 'png'}")
        for path in image_paths or []
    ]
    reference_images = [
        VideoGenerationReferenceImage(
            image=Image(image_bytes=raw, mime_type=mime),
            reference_type=VideoGenerationReferenceType.ASSET,
        )
        for raw, mime in images[:3]
    ]
    video_config = GenerateVideosConfig(
        number_of_videos=1,
        aspect_ratio="16:9",
        resolution="1080p",
        duration_seconds=8,
        reference_images=reference_images or None,
    )

    start = time.time()
    console.print(f"[bold]Submitting video job to {VIDEO_MODEL.name}...[/bold]")
    operation = client.models.generate_videos(
        model=VIDEO_MODEL.name, prompt=user_prompt, config=video_config
    )
    console.print(f"Operation: {operation.name}, Done: {operation.done}")

    while not operation.done:
        time.sleep(POLL_INTERVAL)
        operation = client.operations.get(operation)
        console.print(f"Polling... Done: {operation.done}")

    if operation.error or not (operation.response and operation.response.generated_videos):
        raise RuntimeError(f"Video generation failed: {operation.error}")

    console.print("[bold]Downloading video...[/bold]")
    generated = operation.response.generated_videos[0]
    video_bytes = client.files.download(file=generated.video.uri)

    output_path = Path("generated.mp4")
    output_path.write_bytes(data=video_bytes)

    end = time.time()
    console.print(f"[green]Saved to {output_path}[/green]")
    console.print(f"\n{VIDEO_MODEL.name} native takes {end - start:.2f} seconds")


if __name__ == "__main__":
    gen_video(user_prompt="A cat dancing on a table")
