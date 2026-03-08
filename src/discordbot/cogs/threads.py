import re
from pathlib import Path
import contextlib

import logfire
import nextcord
from nextcord.ext import commands

from discordbot.utils.threads import ThreadsDownloader

URL_REGEX = re.compile(r"https?://(?:www\.)?threads\.(?:net|com)/@[^/]+/post/[^\s]+")


class ThreadsCogs(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.output_folder = Path("./data/threads")
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.downloader = ThreadsDownloader(output_folder=str(self.output_folder))

    @commands.Cog.listener()
    async def on_message(self, message: nextcord.Message) -> None:
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
                result = self.downloader.parse(url)
            except Exception as e:
                logfire.error(f"Failed to parse Threads URL: {e}")
                return

            if not result.text and not result.media_paths and not result.media_urls:
                return

            try:
                # Compute total size of all downloaded media
                total_size = sum(f.stat().st_size for f in result.media_paths if f.exists())
                max_size = 25 * 1024 * 1024  # 25 MB limit

                # If limits are exceeded, just ignore the message
                if (
                    total_size > max_size
                    or len(result.media_paths) > 10
                    or len(result.text) > 2000
                ):
                    return

                files = [
                    nextcord.File(str(path), filename=path.name)
                    for path in result.media_paths
                    if path.exists()
                ]

                # Optionally suppress the original embed to keep the chat clean
                with contextlib.suppress(Exception):
                    await message.edit(suppress=True)

                await message.reply(content=result.text, files=files, mention_author=False)
            except Exception as e:
                logfire.error(f"Failed to send Threads message: {e}")
            finally:
                # Clean up downloaded files
                for path in result.media_paths:
                    with contextlib.suppress(Exception):
                        if path.exists():
                            path.unlink()


async def setup(bot: commands.Bot) -> None:
    bot.add_cog(ThreadsCogs(bot), override=True)
