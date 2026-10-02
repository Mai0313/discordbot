"""Local video generation smoke test: drives the bot's own `VideoGenerator.render`.

A source video is edited in place; otherwise any images ride as references and omni infers the
task; otherwise the prompt alone is rendered. The clip is saved to ./data/generated.mp4.
"""

import time
import asyncio
from pathlib import Path
from mimetypes import guess_type

from google import genai
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.typings.media import LoadedMedia
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.typings.context_budgets import MAX_VIDEO_REFERENCE_IMAGES
from discordbot.cogs.gen_reply.generation import VideoGenerator

console = Console()
config = LLMConfig()


def _load_media(path: str, fallback_mime: str) -> LoadedMedia:
    """Reads a local file with the MIME type its extension implies."""
    return LoadedMedia(
        data=Path(path).read_bytes(), mime_type=guess_type(url=path)[0] or fallback_mime
    )


def gen_video(
    user_prompt: str, image_paths: list[str] | None = None, source_video_path: str | None = None
) -> None:
    """Renders one clip through `VideoGenerator.render` and saves it to ./data/generated.mp4.

    Args:
        user_prompt (str): Prompt (or, in edit mode, the literal edit instruction).
        image_paths (list[str] | None): Optional local reference images; an edit ignores them,
            since it takes the source clip alone.
        source_video_path (str | None): Optional local video file to edit in place.
    """
    generator = VideoGenerator(
        client=genai.Client(api_key=config.gemini_api_key),
        video_model=RuntimeModelCatalog().video_model,
    )
    source_video = (
        None
        if source_video_path is None
        else _load_media(path=source_video_path, fallback_mime="video/mp4")
    )
    images = [
        _load_media(path=path, fallback_mime="image/png")
        for path in (image_paths or [])[:MAX_VIDEO_REFERENCE_IMAGES]
    ]

    start = time.time()
    console.print(f"[bold]Submitting omni video job to {generator.video_model.name}...[/bold]")
    video_bytes = asyncio.run(
        main=generator.render(
            prompt=user_prompt, reference_image_sources=images, source_video=source_video
        )
    )
    output_path = Path("./data/generated.mp4")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_bytes(data=video_bytes)

    console.print(f"[green]Saved {len(video_bytes)} bytes to {output_path}[/green]")
    console.print(f"\n{generator.video_model.name} took {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    # Text-to-video by default. To exercise the other modes, edit the call:
    #   gen_video(user_prompt="A cat dancing", image_paths=["cat.png"])      # images
    #   gen_video(user_prompt="make it snowy", source_video_path="clip.mp4")  # edit
    gen_video(user_prompt="A cat dancing on a table")
