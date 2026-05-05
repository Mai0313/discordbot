import re
from pathlib import Path
import contextlib

import logfire
from nextcord import File, Color, Embed, Message
from nextcord.ext import commands

from discordbot.utils.threads import ThreadsOutput, ThreadsDownloader

URL_REGEX = re.compile(r"https?://(?:www\.)?threads\.(?:net|com)/@[^/]+/post/[^\s\"'<>)]+")


class ThreadsCogs(commands.Cog):
    """Expands Threads links into Discord embeds and media attachments.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        output_folder: Directory where downloaded Threads media is stored.
        downloader: Downloader used to parse Threads posts and fetch media.
    """

    def __init__(self, bot: commands.Bot):
        """Initializes the ThreadsCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.output_folder = Path("./data/threads")
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.downloader = ThreadsDownloader(output_folder=str(self.output_folder))

    async def _handle_reaction(
        self, message: Message, emoji: str, previous_emoji: str | None = None
    ) -> None:
        """Handles adding and removing reactions on a message."""
        if previous_emoji and self.bot.user:
            with contextlib.suppress(Exception):
                await message.remove_reaction(emoji=previous_emoji, member=self.bot.user)
        with contextlib.suppress(Exception):
            await message.add_reaction(emoji=emoji)

    @staticmethod
    def _gradient_color(*, index: int, total: int) -> Color:
        """Greyscale gradient — lightest at index=0 (root), darkest at index=total-1 (leaf).

        Both ends stay inside [0x40, 0xC0] so every layer renders a visible stripe; pure black
        (#000000) is reserved for "no stripe" on solo posts.
        """
        if total <= 1:
            return Color.default()
        light = 0xC0
        dark = 0x40
        shade = round(light + (dark - light) * index / (total - 1))
        return Color.from_rgb(r=shade, g=shade, b=shade)

    @staticmethod
    def _build_post_embed(output: ThreadsOutput, *, color: Color) -> Embed:
        """Builds an embed for a single Threads post."""
        embed = Embed(
            description=output.text, url=output.url, color=color, timestamp=output.taken_at
        )
        if output.author_name:
            embed.set_author(
                name=output.author_name, url=output.url, icon_url=output.author_icon_url
            )
        footer_parts = [
            f"❤️ {output.like_count:,}",
            f"💬 {output.reply_count:,}",
            f"🔁 {output.repost_count:,}",
            f"🔗 {output.quote_count:,}",
            f"↗️ {output.reshare_count:,}",
        ]
        embed.set_footer(text=" | ".join(footer_parts))
        return embed

    def _build_embeds(self, result: ThreadsOutput) -> list[Embed]:
        """Builds a list of embeds for a Threads post and its parents."""
        # Discord caps a single message at 10 embeds. We always keep the current post's main
        # embed; the rest of the budget is split between ancestor context (oldest → newest)
        # and extra image embeds for the current post. Context wins over extra images when
        # the chain is deep — that's the whole point of fetching the chain.
        max_embeds = 10
        embeds: list[Embed] = []
        chain_depth = len(result.parents) + 1

        for i, parent in enumerate(result.parents):
            if len(embeds) >= max_embeds - 1:  # leave one slot for the current post's main embed
                break
            parent_embed = self._build_post_embed(
                output=parent, color=self._gradient_color(index=i, total=chain_depth)
            )
            if parent.image_urls:
                parent_embed.set_image(url=parent.image_urls[0])
            if parent.video_urls and parent.url:
                # Parent videos aren't downloaded (see ThreadsDownloader.parse), so surface a
                # link hint — otherwise a video-only parent shows as an empty embed.
                hint = f"\n\n🎬 [點此觀看影片]({parent.url})"
                parent_embed.description = (parent_embed.description or "") + hint
            embeds.append(parent_embed)

        main_embed = self._build_post_embed(
            output=result, color=self._gradient_color(index=chain_depth - 1, total=chain_depth)
        )
        if result.image_urls:
            main_embed.set_image(url=result.image_urls[0])
        embeds.append(main_embed)

        # Subsequent images of the current post share the same URL so Discord visually groups them.
        remaining = max_embeds - len(embeds)
        for img_url in result.image_urls[1 : 1 + remaining]:
            extra = Embed(url=result.url)
            extra.set_image(url=img_url)
            embeds.append(extra)

        return embeds

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and parses Threads links.

        Args:
            message: The message that was sent.
        """
        if message.author.bot:
            return

        match = URL_REGEX.search(message.content)
        if not match:
            return

        url = match.group(0)
        current_emoji = "🔗"
        await self._handle_reaction(message=message, emoji=current_emoji)

        try:
            with self.downloader.parse(url) as result:
                if not result.text and not result.video_paths and not result.image_urls:
                    await self._handle_reaction(
                        message=message, emoji="⚠️", previous_emoji=current_emoji
                    )
                    return

                total_size = sum(f.stat().st_size for f in result.video_paths if f.exists())
                # Discord measures the full multipart request body, not just file bytes,
                # so reserve 1 MiB for the multipart envelope + embeds JSON. Pull the
                # actual per-guild limit from nextcord (boost tier 2/3 raises it to 50/100 MiB);
                # message.guild is None in DMs, so fall back to the unboosted 25 MiB.
                guild_limit = message.guild.filesize_limit if message.guild else 25 * 1024 * 1024
                max_size = guild_limit - 1024 * 1024

                if (
                    total_size > max_size
                    or len(result.video_paths) + len(result.image_urls) > 10
                    or len(result.text) > 4096
                ):
                    await self._handle_reaction(
                        message=message, emoji="⚠️", previous_emoji=current_emoji
                    )
                    return

                files = [
                    File(fp=str(path), filename=path.name)
                    for path in result.video_paths
                    if path.exists()
                ]

                embeds = self._build_embeds(result=result)

                with contextlib.suppress(Exception):
                    await message.edit(suppress=True)

                await message.reply(embeds=embeds, files=files, mention_author=False)
                await self._handle_reaction(
                    message=message, emoji="🆗", previous_emoji=current_emoji
                )
        except Exception:
            logfire.error("Failed to send Threads message", _exc_info=True)
            with contextlib.suppress(Exception):
                await self._handle_reaction(
                    message=message, emoji="❌", previous_emoji=current_emoji
                )


async def setup(bot: commands.Bot) -> None:
    """Adds the ThreadsCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ThreadsCogs(bot), override=True)
