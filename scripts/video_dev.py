import time

from openai import OpenAI
from rich.console import Console

from discordbot.typings.llm import LLMConfig

console = Console()

VIDEO_MODEL = "veo-3.1-fast-generate-preview"
PROMPT = "A cat dancing on a table"
POLL_INTERVAL = 5

config = LLMConfig()


def main() -> None:
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    console.print("[bold]Submitting video generation job...[/bold]")
    video = client.videos.create(model=VIDEO_MODEL, prompt=PROMPT)
    console.print(f"ID: {video.id}, Status: {video.status}")

    while video.status not in ("completed", "failed"):
        time.sleep(POLL_INTERVAL)
        video = client.videos.retrieve(video.id)
        console.print(f"Polling... Status: {video.status}, Progress: {video.progress}%")

    if video.status != "completed":
        console.print(f"[red]Video generation failed: {video.error}[/red]")
        return

    console.print("[bold]Downloading video...[/bold]")
    content = client.videos.download_content(video.id)

    output_path = "generated.mp4"
    with open(output_path, "wb") as f:
        f.write(content.content)

    console.print(f"[green]Saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
