"""Expands public Facebook post URLs into Discord embeds via `ConversationExpansionCog`."""

import asyncio
from pathlib import Path
from collections.abc import Callable

import logfire
from nextcord.ext import commands

from discordbot.typings.timeouts import FACEBOOK_EXPAND_TIMEOUT_SECONDS
from discordbot.utils.expansion_cog import ConversationExpansionCog
from discordbot.utils.discord_embeds import DISCORD_ATTACHMENT_LIMIT
from discordbot.utils.file_downloads import DownloadTooLargeError
from discordbot.services.platforms.ytdlp import VideoDownloader, download_with_stop_signal
from discordbot.services.platforms.facebook import (
    FACEBOOK_URL_RE,
    FacebookOutput,
    FacebookDownloader,
    FacebookConversation,
    is_facebook_post_url,
)


class FacebookCogs(ConversationExpansionCog[FacebookOutput, FacebookConversation]):
    """Expands Facebook post links into Discord embeds."""

    SOURCE = "facebook"
    PLATFORM = "Facebook"
    URL_PATTERN = FACEBOOK_URL_RE
    PLACEHOLDER_TEXT = "-# 正在讀取 Facebook 貼文⋯"
    READ_TIMEOUT_SECONDS = FACEBOOK_EXPAND_TIMEOUT_SECONDS
    EMBED_COLOR = 0x1877F2
    DOWNLOAD_VIDEOS = True
    downloader_factory = FacebookDownloader
    video_downloader_factory: Callable[..., VideoDownloader] = VideoDownloader

    @staticmethod
    def url_is_expandable(url: str) -> bool:
        """Returns whether `url` names a Facebook post.

        `FACEBOOK_URL_RE` matches the host rather than the path, so profile and group home pages
        are filtered here before the read.

        Args:
            url (str): The URL matched by `URL_PATTERN`.
        """
        return is_facebook_post_url(url=url)

    async def _download_one_clip(  # noqa: PLR0913 -- one argument per clip download context field
        self, index: int, video_url: str, base_dir: Path, max_bytes: int, url: str, message_id: int
    ) -> Path | None:
        """Downloads one Facebook clip via yt-dlp into `base_dir/video_{index}.mp4`."""
        clip_dir = base_dir / f"clip_{index}"
        clip_dir.mkdir(exist_ok=True)
        try:
            result = await download_with_stop_signal(
                downloader=self.video_downloader_factory(output_folder=str(clip_dir)),
                url=video_url,
                quality="best",
                max_items=1,
                max_bytes=max_bytes,
            )
            target_path = base_dir / f"video_{index}.mp4"
            result.filenames[0].replace(target=target_path)
            return target_path
        except DownloadTooLargeError as error:
            logfire.info(
                f"{self.PLATFORM} video exceeds the byte cap; falling back to link",
                url=url,
                video_url=video_url,
                message_id=message_id,
                max_bytes=max_bytes,
                reason=str(error),
            )
        except TimeoutError:
            raise
        except Exception as error:
            # Broad on purpose: yt-dlp raises site-specific and extractor errors outside
            # requests/OSError; a clip failure degrades to its link without refusing the post.
            logfire.warn(
                f"Could not download a {self.PLATFORM} video; falling back to link",
                url=url,
                video_url=video_url,
                message_id=message_id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
        return None

    async def _download_videos(
        self, post: FacebookOutput, download_dir: str, max_bytes: int, url: str, message_id: int
    ) -> list[Path]:
        """Downloads up to `DISCORD_ATTACHMENT_LIMIT` target videos via yt-dlp in parallel."""
        base_dir = Path(download_dir)
        tasks: list[asyncio.Task[Path | None]] = []
        for index, video_url in enumerate(post.video_urls[:DISCORD_ATTACHMENT_LIMIT]):
            tasks.append(
                asyncio.create_task(
                    self._download_one_clip(
                        index=index,
                        video_url=video_url,
                        base_dir=base_dir,
                        max_bytes=max_bytes,
                        url=url,
                        message_id=message_id,
                    )
                )
            )
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # Cancel untouched siblings without double-cancelling ones `gather` already cancelled,
            # so every `download_with_stop_signal` finishes its bounded join before scratch cleanup.
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [path for path in results if path is not None]

    def _footer_text(self, post: FacebookOutput, shown_images: int) -> str:
        """Returns the footer line with the group name, positive engagement counts, and omitted-media notes."""
        parts = [post.group_name] if post.group_name else []
        if post.like_count > 0:
            parts.append(f"👍 {post.like_count:,}")
        if post.comment_count > 0:
            parts.append(f"💬 {post.comment_count:,}")
        if post.share_count > 0:
            parts.append(f"↗️ {post.share_count:,}")
        parts.extend(self._omitted_media_notes(post=post, shown_images=shown_images))
        return " · ".join(parts)

    def _comment_url(self, post_url: str, comment: FacebookOutput) -> str:
        """Returns `post_url` with `comment_id` appended (`permalink.php` links already carry a query)."""
        joiner = "&" if "?" in post_url else "?"
        return f"{post_url}{joiner}comment_id={comment.comment_id}"


def setup(bot: commands.Bot) -> None:
    """Adds the FacebookCogs to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(FacebookCogs(bot), override=True)
