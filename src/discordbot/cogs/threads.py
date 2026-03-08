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

            # Compute total size of all downloaded media
            total_size = sum(f.stat().st_size for f in result.media_paths if f.exists())
            max_size = 25 * 1024 * 1024  # 25 MB limit

            files: list[nextcord.File] = []
            use_urls = False

            if total_size > max_size or len(result.media_paths) > 10:
                use_urls = True
            else:
                for path in result.media_paths:
                    if path.exists():
                        files.append(nextcord.File(str(path), filename=path.name))

            if use_urls and result.media_urls:
                urls_text = "\n\n" + "\n".join(result.media_urls)
                max_text_len = 2000 - len(urls_text) - 3
                if len(result.text) > max_text_len:
                    result.text = result.text[:max_text_len] + "..."
                result.text = f"{result.text}{urls_text}"
            elif len(result.text) > 2000:
                result.text = result.text[:1997] + "..."

            try:
                # Optionally suppress the original embed to keep the chat clean
                with contextlib.suppress(Exception):
                    await message.edit(suppress=True)

                await message.reply(content=result.text, files=files)
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
