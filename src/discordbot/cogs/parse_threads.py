"""Cog that expands Threads post URLs into Discord embeds and media files."""

import asyncio
from pathlib import Path
import tempfile
import contextlib

import logfire
from nextcord import File, Color, Embed, Message
from nextcord.ext import commands

from discordbot.utils.threads import (
    THREADS_URL_RE,
    ThreadsOutput,
    ThreadsDownloader,
    threads_expansion_relay,
)
from discordbot.utils.reactions import update_reaction
from discordbot.utils.discord_embeds import embed_spacer_payload

URL_REGEX = THREADS_URL_RE


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
        self.output_folder = Path(tempfile.gettempdir())
        self.downloader = ThreadsDownloader(output_folder=str(self.output_folder))

    @staticmethod
    def _gradient_color(index: int, total: int) -> Color:
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
    def _build_post_embed(output: ThreadsOutput, color: Color) -> Embed:
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

    def _build_post_embeds(
        self, output: ThreadsOutput, color: Color, image_count: int, is_target: bool
    ) -> list[Embed]:
        """Builds the embeds for one post, showing `image_count` of its images.

        The main embed carries the post text plus its first shown image; further shown
        images become bare image embeds reusing the post URL so Discord merges them into
        one gallery. `image_count == 0` yields a single text-only context embed.
        """
        main_embed = self._build_post_embed(output=output, color=color)
        embeds = [main_embed]
        if image_count > 0:
            main_embed.set_image(url=output.image_urls[0])
            for img_url in output.image_urls[1:image_count]:
                extra = Embed(url=output.url)
                extra.set_image(url=img_url)
                embeds.append(extra)
        # Target videos are downloaded and attached as files; ancestor videos are not,
        # so surface a link hint — otherwise a video-only parent shows as an empty embed.
        if not is_target and output.video_urls and output.url:
            hint = f"\n\n🎬 [點此觀看影片]({output.url})"
            main_embed.description = (main_embed.description or "") + hint
        return embeds

    def _build_embeds(self, results: list[ThreadsOutput]) -> list[Embed]:
        """Builds a list of embeds for a Threads reply chain.

        Args:
            results: Ordered chain `[root, ..., direct_parent, target]`.
        """
        # Discord caps a single message at 10 embeds, one image each. The posted URL is
        # the target (last item) and owns the message, so its images claim slots first,
        # then the direct parent's, on up the chain; a post that loses the image race
        # still earns a text-only context embed, but only from slots no image needed.
        max_embeds = 10
        # A chain deeper than the embed cap can't show every post; keep the target and its
        # nearest ancestors, which are the most relevant context.
        if len(results) > max_embeds:
            results = results[-max_embeds:]
        chain_depth = len(results)

        priority = list(reversed(range(chain_depth)))  # target, direct parent, ..., root
        image_count = [0] * chain_depth
        budget = max_embeds
        for index in priority:
            take = min(len(results[index].image_urls), budget)
            image_count[index] = take
            budget -= take

        keep_text = [False] * chain_depth
        for index in priority:
            if budget <= 0:
                break
            if image_count[index] == 0:
                keep_text[index] = True
                budget -= 1

        embeds: list[Embed] = []
        for index, output in enumerate(results):
            if image_count[index] == 0 and not keep_text[index]:
                continue
            embeds.extend(
                self._build_post_embeds(
                    output=output,
                    color=self._gradient_color(index=index, total=chain_depth),
                    image_count=image_count[index],
                    is_target=index == chain_depth - 1,
                )
            )
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
        # Register the pending slot up front so the reply pipeline (which awaits the same key)
        # is never left waiting on a future we would otherwise resolve before it asked. The
        # posted expansion is published on every exit path via the outer finally below.
        threads_expansion_relay.get_or_create(message_id=message.id)
        current_emoji = await update_reaction(message=message, bot_user=self.bot.user, emoji="🔗")

        posted_reply: Message | None = None
        try:
            # parse() blocks on HTTP fetch + media downloads, so run its enter
            # off the event loop; the reply runs while the temp files still exist
            # and the matching exit cleans them up afterwards.
            parse_cm = self.downloader.parse(url)
            results = await asyncio.to_thread(parse_cm.__enter__)
            try:
                if not results:
                    await update_reaction(
                        message=message, bot_user=self.bot.user, emoji="⚠️", previous=current_emoji
                    )
                    return

                target = results[-1]
                total_size = sum(f.stat().st_size for f in target.video_paths if f.exists())
                # Discord measures the full multipart request body, not just file bytes,
                # so reserve 1 MiB for the multipart envelope + embeds JSON. Pull the
                # actual per-guild limit from nextcord (boost tier 2/3 raises it to 50/100 MiB);
                # message.guild is None in DMs, so fall back to Discord's non-Nitro base of 10 MiB.
                guild_limit = message.guild.filesize_limit if message.guild else 10 * 1024 * 1024
                max_size = guild_limit - 1024 * 1024

                # Image count is no longer guarded here: _build_embeds caps the message at
                # 10 embeds and shows as many images as fit, so an oversized carousel
                # degrades to its first images instead of refusing the whole post.
                if total_size > max_size or len(target.text) > 4096:
                    await update_reaction(
                        message=message, bot_user=self.bot.user, emoji="⚠️", previous=current_emoji
                    )
                    return

                files = [
                    File(fp=str(path), filename=path.name)
                    for path in target.video_paths
                    if path.exists()
                ]

                embeds = self._build_embeds(results=results)

                with contextlib.suppress(Exception):
                    await message.edit(suppress=True)

                posted_reply = await message.reply(
                    embeds=embeds,
                    mention_author=False,
                    **embed_spacer_payload(
                        embeds=embeds, is_edit=False, target=message, extra_files=files
                    ),
                )
                await update_reaction(
                    message=message, bot_user=self.bot.user, emoji="🆗", previous=current_emoji
                )
            finally:
                await asyncio.to_thread(parse_cm.__exit__, None, None, None)
        except Exception:
            logfire.error("Failed to send Threads message", _exc_info=True)
            with contextlib.suppress(Exception):
                await update_reaction(
                    message=message, bot_user=self.bot.user, emoji="❌", previous=current_emoji
                )
        finally:
            # Always publish: a success hands over the expanded reply, every other path (empty,
            # oversized, fetch error) hands over None so a waiting reply pipeline stops waiting.
            threads_expansion_relay.resolve(message_id=message.id, message=posted_reply)


def setup(bot: commands.Bot) -> None:
    """Adds the ThreadsCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ThreadsCogs(bot), override=True)
