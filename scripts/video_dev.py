"""Local video generation smoke test for the bot video model (omni Interactions API).

Calls `client.interactions.create` directly (no `VideoGenerator` import) so this stays a clean,
self-contained reference for the omni call shape. Mirrors `VideoGenerator.render` in
`cogs/_gen_reply/generation.py`: three input modes (source-video edit / image references / plain
text), fixed 16:9, `delivery="uri"`, single `files.download` (the interaction only reports
`completed` once the file is ready).
"""

import time
import base64
from pathlib import Path

from google import genai
from rich.console import Console
from google.genai.types import FileState
from google.genai.interactions import (
    TextContentParam,
    VideoConfigParam,
    ImageContentParam,
    VideoContentParam,
    GenerationConfigParam,
    VideoResponseFormatParam,
)

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings

console = Console()
config = LLMConfig()

# Mirror the @property value in typings/models.py. Update here when the bot's video_model swaps,
# otherwise this script tests a stale model.
VIDEO_MODEL = ModelSettings(name="gemini-omni-flash-preview")
MAX_REFERENCE_IMAGES = 3


def _upload_source_video(client: genai.Client, path: str) -> str:
    """Uploads a source clip to the Files API and returns its ACTIVE uri."""
    mime = f"video/{Path(path).suffix.lstrip('.') or 'mp4'}"
    uploaded = client.files.upload(file=path, config={"mime_type": mime})
    while uploaded.state == FileState.PROCESSING:
        console.print("Uploading source video... (PROCESSING)")
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state != FileState.ACTIVE or uploaded.uri is None:
        raise RuntimeError(f"Source video upload failed: state={uploaded.state}")
    return uploaded.uri


def gen_video(
    user_prompt: str, *, image_paths: list[str] | None = None, source_video_path: str | None = None
) -> None:
    """Runs the dev omni video flow and saves the MP4 result to generated.mp4.

    A `source_video_path` is uploaded and edited in place (task=edit); otherwise any
    `image_paths` ride as subject reference images (task=reference_to_video, up to three);
    otherwise plain text (task=text_to_video).

    Args:
        user_prompt: Prompt (or, in edit mode, the literal edit instruction).
        image_paths: Optional local image files used as subject reference images.
        source_video_path: Optional local video file to edit in place.

    Raises:
        RuntimeError: The interaction did not complete with a video.
    """
    client = genai.Client(api_key=config.gemini_api_key)

    content: list[object] = [TextContentParam(type="text", text=user_prompt)]
    if source_video_path is not None:
        video_uri = _upload_source_video(client=client, path=source_video_path)
        content = [
            VideoContentParam(type="video", uri=video_uri),
            TextContentParam(type="text", text=user_prompt),
        ]
        # image_paths ride alongside the video only to probe whether omni edit accepts both.
        for path in (image_paths or [])[:MAX_REFERENCE_IMAGES]:
            content.append(
                ImageContentParam(
                    type="image", data=base64.b64encode(Path(path).read_bytes()).decode()
                )
            )
        task = "edit"
    elif image_paths:
        for path in image_paths[:MAX_REFERENCE_IMAGES]:
            content.append(
                ImageContentParam(
                    type="image", data=base64.b64encode(Path(path).read_bytes()).decode()
                )
            )
        task = "reference_to_video"
    else:
        task = "text_to_video"

    # omni 400s an aspect_ratio on an edit ("cannot be set in response format for edit task"),
    # so only text / reference generation pins 16:9; an edit keeps the source clip's ratio.
    response_format = VideoResponseFormatParam(type="video", delivery="uri")
    if task != "edit":
        response_format["aspect_ratio"] = "16:9"

    start = time.time()
    console.print(f"[bold]Submitting omni video job ({task}) to {VIDEO_MODEL.name}...[/bold]")
    interaction = client.interactions.create(
        model=VIDEO_MODEL.name,
        input=content,
        response_format=response_format,
        generation_config=GenerationConfigParam(video_config=VideoConfigParam(task=task)),
    )
    console.print(f"status={interaction.status}")

    video = interaction.output_video
    if interaction.status != "completed" or video is None or video.uri is None:
        raise RuntimeError(
            f"Video generation failed: status={interaction.status} note={interaction.output_text!r}"
        )

    console.print("[bold]Downloading video...[/bold]")
    video_bytes = client.files.download(file=video.uri)
    output_path = Path("generated.mp4")
    output_path.write_bytes(data=video_bytes)

    console.print(f"[green]Saved {len(video_bytes)} bytes to {output_path}[/green]")
    console.print(f"\n{VIDEO_MODEL.name} ({task}) took {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    # Text-to-video by default. To exercise the other modes, edit the call:
    #   gen_video("A cat dancing", image_paths=["cat.png"])            # reference_to_video
    #   gen_video("make it snowy", source_video_path="clip.mp4")       # edit
    #   gen_video("add a hat", source_video_path="clip.mp4", image_paths=["hat.png"])  # edit + image probe
    gen_video(user_prompt="A cat dancing on a table")
