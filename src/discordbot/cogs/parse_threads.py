import re
from pathlib import Path
import contextlib

import logfire
from nextcord import File, Color, Embed, Message
from nextcord.ext import commands

from discordbot.utils.threads import ThreadsOutput, ThreadsDownloader

URL_REGEX = re.compile(r"https?://(?:www\.)?threads\.(?:net|com)/@[^/]+/post/[^\s]+")


class ThreadsCogs(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.output_folder = Path("./data/threads")
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.downloader = ThreadsDownloader(output_folder=str(self.output_folder))

    def _build_embeds(self, result: ThreadsOutput) -> list[Embed]:
        embeds = []
        main_embed = Embed(description=result.text, url=result.url, color=Color.default())
        if result.author_name:
            main_embed.set_author(
                name=f"@{result.author_name}", icon_url=result.author_icon_url, url=result.url
            )

        if result.image_urls:
            main_embed.set_image(url=result.image_urls[0])
            embeds.append(main_embed)
            # Add subsequent images as their own embeds with the same URL to visually group them
            for img_url in result.image_urls[1:10]:
                embed = Embed(url=result.url)
                embed.set_image(url=img_url)
                embeds.append(embed)
        else:
            embeds.append(main_embed)
        return embeds

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        if message.author.bot:
            return

        match = URL_REGEX.search(message.content)
        if not match:
            return

        url = match.group(0)

        # Start typing indicator
        async with message.channel.typing():
            try:
                # Run parsing logic
                with self.downloader.parse(url) as result:
                    if not result.text and not result.video_paths and not result.image_urls:
                        return

                    # Compute total size of all downloaded media
                    total_size = sum(f.stat().st_size for f in result.video_paths if f.exists())
                    max_size = 25 * 1024 * 1024  # 25 MB limit

                    # If limits are exceeded, just ignore the message
                    if (
                        total_size > max_size
                        or len(result.video_paths) + len(result.image_urls) > 10
                        or len(result.text) > 4096
                    ):
                        return

                    files = [
                        File(str(path), filename=path.name)
                        for path in result.video_paths
                        if path.exists()
                    ]

                    embeds = self._build_embeds(result)

                    with contextlib.suppress(Exception):
                        await message.edit(suppress=True)

                    await message.reply(embeds=embeds, files=files, mention_author=False)
            except Exception as e:
                logfire.error(f"Failed to send Threads message: {e}")


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(ThreadsCogs(bot), override=True)
