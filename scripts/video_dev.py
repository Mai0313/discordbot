"""Local video generation smoke test for the bot video model."""

import time
from pathlib import Path

from openai import OpenAI
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings

console = Console()
config = LLMConfig()

# Mirror the @property value in cogs/gen_reply.py. Update here when the bot's
# video_model swaps, otherwise this script tests a stale model.
VIDEO_MODEL = ModelSettings(name="veo-3.1-fast-generate-preview")
POLL_INTERVAL = 5


def gen_video(user_prompt: str) -> None:
    """Runs the dev video generation flow and saves the MP4 result.

    Mirrors `_handle_video_reply` in `cogs/gen_reply.py` by submitting a
    video job, polling until it completes or fails, downloading the completed
    video, and writing it to `generated.mp4`. Prints job status, polling
    progress, the saved path, and elapsed time to the console.

    Args:
        user_prompt: Prompt to send to the video generation endpoint.

    Raises:
        RuntimeError: The video job ended in a non-completed status.
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    start = time.time()
    console.print(f"[bold]Submitting video job to {VIDEO_MODEL.name}...[/bold]")
    video = client.videos.create(
        model=VIDEO_MODEL.name,
        prompt=user_prompt,
        extra_headers={"x-litellm-end-user-id": "video_dev"},
    )
    console.print(f"ID: {video.id}, Status: {video.status}")

    while video.status not in ("completed", "failed"):
        time.sleep(POLL_INTERVAL)
        video = client.videos.retrieve(
            video_id=video.id, extra_headers={"x-litellm-end-user-id": "video_dev"}
        )
        console.print(f"Polling... Status: {video.status}, Progress: {video.progress}%")

    if video.status != "completed":
        raise RuntimeError(f"Video generation failed: {video.error}")

    console.print("[bold]Downloading video...[/bold]")
    video_content = client.videos.download_content(
        video_id=video.id, extra_headers={"x-litellm-end-user-id": "video_dev"}
    )

    output_path = Path("generated.mp4")
    output_path.write_bytes(data=video_content.content)

    end = time.time()
    console.print(f"[green]Saved to {output_path}[/green]")
    console.print(f"\n{VIDEO_MODEL.name} on Litellm takes {end - start:.2f} seconds")


if __name__ == "__main__":
    gen_video(user_prompt="A cat dancing on a table")
