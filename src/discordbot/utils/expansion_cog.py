"""Shared base cogs and card-building helpers for link auto-expansion.

`ExpansionCog` owns the placeholder lifecycle, reactions, image upload, and delivery;
`ConversationExpansionCog` adds blocking conversation reads and the default embed card.
`utils/expansion_placeholder.py` owns the persisted placeholder and restart-resume half.

Failures leave no message in the channel; the reaction on the source message is the sole report.
There is deliberately no kill-switch; disabling an expansion means removing its cog.
"""

import re
from typing import Any, ClassVar, Protocol
import asyncio
from pathlib import Path
from datetime import datetime
from mimetypes import guess_extension
import contextlib
from collections.abc import Callable, Coroutine

import logfire
from nextcord import Color, Embed, Message, NotFound, Forbidden
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
import requests
from nextcord.ext import commands

from discordbot.typings.media import LoadedMedia
from discordbot.typings.emojis import LINK_SOURCE_EMOJIS, LinkSourceName
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.timezone import database_now
from discordbot.utils.reactions import update_reaction
from discordbot.typings.timeouts import (
    CARD_IMAGE_FETCH_TIMEOUT_SECONDS,
    EXPANSION_VIDEO_READ_TIMEOUT_SECONDS,
    EXPANSION_VIDEO_DOWNLOAD_TIMEOUT_SECONDS,
)
from discordbot.utils.link_errors import (
    LinkReadError,
    LinkRetryableError,
    LinkUnavailableError,
    is_retryable_fetch_failure,
)
from discordbot.utils.scratch_dir import scratch_directory
from discordbot.utils.discord_embeds import (
    DISCORD_ATTACHMENT_LIMIT,
    DISCORD_EMBED_TOTAL_LIMIT,
    DISCORD_EMBED_DESCRIPTION_LIMIT,
    utf16_length,
    embed_text_length,
    clip_to_utf16_limit,
    target_allows_file_uploads,
)
from discordbot.utils.file_downloads import DownloadTooLargeError, stream_to_file
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaHostingConfig,
    MediaHostingService,
    MediaDeliveryPlanner,
    upload_limit_for,
    build_media_delivery_planner,
)
from discordbot.utils.expansion_placeholder import (
    EXPANSION_DONE_EMOJI,
    EXPANSION_FAILED_EMOJI,
    EXPANSION_WORKING_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    EXPANSION_RETRY_LATER_EMOJI,
    ExpansionPlaceholder,
    send_expansion_placeholder,
    resume_expansion_placeholders,
)

# Cap images per card to avoid channel flood; not a model context budget.
POST_CARD_MAX_IMAGES = 4

# Neutral rail color for secondary cards (quoted/replied posts, comments), distinct from platform
# brand colors.
CONTEXT_CARD_COLOR = 0x65686C

TRUNCATION_NOTICE = "\n\n⋯（全文請看原貼文）"
VIDEO_HINT = "\n\n🎬 [點此觀看影片]({url})"

# Unmeasured UTF-16 overhead on secondary cards (author line, header, spacing).
_CONTEXT_CARD_SLACK = 400

# UTF-16 budget reserved from the post card so a singled-out comment always fits.
_COMMENT_RESERVE = 2000
_COMMENT_HEADER = "💬 **指定的留言**"

# Matches spoiler bars (`||...||`) and angle-bracketed links (`<...>`) shortest-first; does not
# parse code spans or escapes.
_HIDDEN_LINK_RE = re.compile(pattern=r"\|\|.+?\|\||<[^\s<>]+>", flags=re.DOTALL)

_DEFAULT_VIDEO_HEADERS = {"User-Agent": "Mozilla/5.0"}
_VIDEO_FILE_INDEX_RE = re.compile(pattern=r"^video_(\d+)\.mp4$")


class CardPost(Protocol):
    """Structural protocol for one post or comment rendered on an expansion card."""

    @property
    def text(self) -> str:
        """The post or comment body."""
        ...

    @property
    def url(self) -> str:
        """Where it can be read."""
        ...

    @property
    def author_name(self) -> str:
        """The author's handle, empty when the page served none."""
        ...

    @property
    def author_icon_url(self) -> str:
        """The author's profile picture, empty when the page served none."""
        ...

    @property
    def image_urls(self) -> list[str]:
        """Still images, as URLs the card downloads and uploads."""
        ...

    @property
    def video_urls(self) -> list[str]:
        """Videos it carries."""
        ...

    @property
    def taken_at(self) -> datetime | None:
        """When it was published."""
        ...

    @property
    def is_readable(self) -> bool:
        """Whether enough came back to be worth showing."""
        ...


class CardConversation[PostT: CardPost](Protocol):
    """What the shared read and card read off a parsed conversation."""

    @property
    def target(self) -> PostT | None:
        """The linked post, None when it could not be read."""
        ...

    @property
    def selected_comment(self) -> PostT | None:
        """The comment the URL singled out, None when it named none."""
        ...


class ConversationReader[ConversationT](Protocol):
    """A platform reader that fetches and parses a post in one blocking call."""

    def parse_metadata(self, url: str) -> ConversationT:
        """Returns the parsed conversation at `url`.

        Args:
            url (str): Post URL to fetch and parse.
        """
        ...


def with_gallery(card: Embed, images: list[str]) -> list[Embed]:
    """Returns `card` with its first image attached, followed by a bare embed per extra image.

    Extra embeds reuse `card.url` to merge into one Discord gallery under `card`.

    Args:
        card (Embed): The post's own embed.
        images (list[str]): Image URLs to show, empty for none.
    """
    if images:
        card.set_image(url=images[0])
    embeds = [card]
    for image_url in images[1:]:
        extra = Embed(url=card.url)
        extra.set_image(url=image_url)
        embeds.append(extra)
    return embeds


def post_card_embeds(  # noqa: PLR0913 -- one argument per part a platform supplies
    post: CardPost,
    color: int,
    author: str,
    footer: str,
    images: list[str],
    suffix: str,
    reserve: int,
) -> list[Embed]:
    """Returns the linked post's embed followed by one bare embed per extra image.

    Clips long bodies with `TRUNCATION_NOTICE` rather than splitting across embeds.

    Args:
        post (CardPost): The linked post.
        color (int): The platform's own colour for the post's rail.
        author (str): The author line, empty for none.
        footer (str): The counter line.
        images (list[str]): The images to show, the first on the post's own embed.
        suffix (str): Text appended after the body, reserved before clipping so a body at the
            ceiling never overflows.
        reserve (int): UTF-16 units reserved from the message-wide ceiling for secondary cards,
            0 when there are none.
    """
    limit = min(
        DISCORD_EMBED_DESCRIPTION_LIMIT - utf16_length(value=suffix),
        DISCORD_EMBED_TOTAL_LIMIT - reserve,
    )
    description = (
        clip_to_utf16_limit(text=post.text, limit=limit, notice=TRUNCATION_NOTICE) + suffix
    )
    main = Embed(
        description=description or None,
        url=post.url,
        color=Color(value=color),
        timestamp=post.taken_at,
    )
    if author:
        main.set_author(name=author, url=post.url, icon_url=post.author_icon_url or None)
    main.set_footer(text=footer)
    return with_gallery(card=main, images=images)


def context_card_budget(card: Embed) -> int:
    """Returns remaining UTF-16 units under Discord's message-wide limit for secondary cards.

    Args:
        card (Embed): The linked post's own embed.
    """
    return DISCORD_EMBED_TOTAL_LIMIT - embed_text_length(embed=card) - _CONTEXT_CARD_SLACK


def fetch_card_image(url: str, max_bytes: int = 25 * 1024 * 1024) -> LoadedMedia:
    """Returns the image at `url` as served, for a card to upload rather than link.

    Args:
        url (str): The image URL the card would otherwise link.
        max_bytes (int): Maximum allowed size in bytes before refusing and keeping the link;
            defaults to 25 MiB so oversized `image/*` payloads stay linked instead of spiking RAM.

    Raises:
        requests.RequestException: The fetch failed or answered an error status.
        ValueError: What came back is not an image or exceeds `max_bytes`.
    """
    with requests.get(url=url, timeout=CARD_IMAGE_FETCH_TIMEOUT_SECONDS, stream=True) as response:
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "").partition(";")[0].strip()
        if not mime_type.startswith("image/"):
            raise ValueError(f"{url} served {mime_type or 'no type'}, not an image")
        declared = response.headers.get("Content-Length")
        if declared is not None and declared.isdigit() and int(declared) > max_bytes:
            raise ValueError(f"{url} declares {declared} bytes, over the {max_bytes} byte cap")
        buffer = bytearray()
        for chunk in response.iter_content(chunk_size=1 << 16):
            if not chunk:
                continue
            if len(buffer) + len(chunk) > max_bytes:
                raise ValueError(f"{url} exceeds {max_bytes} bytes")
            buffer.extend(chunk)
    return LoadedMedia(data=bytes(buffer), mime_type=mime_type)


async def _download_card_image(image_url: str, url: str, message_id: int) -> LoadedMedia | None:
    """Returns one card image's bytes, or None when the card keeps linking it."""
    try:
        # `requests` bounds each socket operation, not the whole transfer.
        async with asyncio.timeout(delay=CARD_IMAGE_FETCH_TIMEOUT_SECONDS):
            return await asyncio.to_thread(fetch_card_image, url=image_url)
    except requests.RequestException as error:
        logfire.warn(
            "Could not download a card image; it stays linked",
            url=url,
            message_id=message_id,
            image_url=image_url,
            retryable=is_retryable_fetch_failure(error=error),
            error_type=type(error).__name__,
            _exc_info=error,
        )
    except (TimeoutError, ValueError) as error:
        logfire.warn(
            "Could not download a card image; it stays linked",
            url=url,
            message_id=message_id,
            image_url=image_url,
            error_type=type(error).__name__,
            _exc_info=error,
        )
    return None


def omitted_images_note(count: int) -> str:
    """Returns the footer note counting a post's images the card leaves out.

    Args:
        count (int): Number of omitted images.
    """
    return f"🖼️ 另有 {count} 張"


def omitted_videos_note(count: int) -> str:
    """Returns the footer note counting a post's undelivered videos past the linked one.

    Args:
        count (int): Number of omitted videos.
    """
    return f"🎬 另有 {count} 部影片"


def omitted_posts_note(count: int) -> str:
    """Returns the footer note counting the posts above the linked one that the card leaves out.

    Args:
        count (int): Number of omitted posts.
    """
    return f"📝 另有 {count} 篇未展開"


def expansion_failure_emoji(error: Exception) -> str:
    """Returns the outcome reaction emoji for a failed read.

    Selected from `error`'s class rather than its message: a retryable fetch failure or timeout
    earns `EXPANSION_RETRY_LATER_EMOJI`, any other `LinkReadError` earns
    `EXPANSION_UNREADABLE_EMOJI`, and anything outside that hierarchy earns
    `EXPANSION_FAILED_EMOJI`.

    Args:
        error (Exception): Exception raised by the read.
    """
    if isinstance(error, LinkRetryableError | TimeoutError):
        return EXPANSION_RETRY_LATER_EMOJI
    if isinstance(error, LinkReadError):
        return EXPANSION_UNREADABLE_EMOJI
    return EXPANSION_FAILED_EMOJI


def report_expansion_read_failure(
    error: Exception, platform: str, url: str, message_id: int
) -> None:
    """Logs a failed read at the severity matching its outcome.

    `LinkUnavailableError` logs at `info` with `reason` and no traceback; other `LinkReadError`s
    and `TimeoutError` log at `warn`; unexpected exceptions log at `error`.

    Args:
        error (Exception): Exception raised by the read.
        platform (str): The platform's display name, for the log message.
        url (str): The post being expanded.
        message_id (int): The source message carrying the link.
    """
    if isinstance(error, LinkUnavailableError):
        logfire.info(
            f"{platform} post is gone or private",
            url=url,
            message_id=message_id,
            error_type=type(error).__name__,
            reason=str(error),
        )
        return
    report = logfire.warn if isinstance(error, LinkReadError | TimeoutError) else logfire.error
    report(
        f"{platform} read failed",
        url=url,
        message_id=message_id,
        error_type=type(error).__name__,
        _exc_info=error,
    )


def report_expansion_delivery_failure(
    error: Exception, platform: str, url: str, message_id: int, channel_id: int
) -> None:
    """Logs a failed delivery at the severity matching its outcome.

    Logs `NotFound` at `info`, `Forbidden` at `warn` (code only), and other errors at `error`;
    on edits, HTTP 50035 is a rejected body rather than a missing target.

    Args:
        error (Exception): Exception raised by the delivery.
        platform (str): The platform's display name, for the log message.
        url (str): The post being expanded.
        message_id (int): The source message carrying the link.
        channel_id (int): The channel it was posted in.
    """
    if isinstance(error, NotFound):
        logfire.info(
            f"The {platform} expansion placeholder is gone",
            url=url,
            message_id=message_id,
            channel_id=channel_id,
        )
    elif isinstance(error, Forbidden):
        # Omit traceback: stack is identical on every refusal; error.code identifies which refusal.
        logfire.warn(
            f"Missing permission to post the {platform} expansion",
            url=url,
            message_id=message_id,
            channel_id=channel_id,
            error_type=type(error).__name__,
            code=error.code,
        )
    else:
        logfire.error(
            f"Failed to send {platform} expansion",
            url=url,
            message_id=message_id,
            channel_id=channel_id,
            error_type=type(error).__name__,
            _exc_info=error,
        )


class ExpansionDelivery(BaseModel):
    """What a readable post becomes on screen."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str | None = Field(
        default=None,
        description="Text under the card, or None when the card is embeds and attachments alone.",
    )
    embeds: list[SkipValidation[Embed]] = Field(..., description="The finished card.")
    media: list[MediaItem] = Field(
        default_factory=list,
        description=(
            "Media attaching natively beside the card, sized so its images can be fitted after it."
        ),
    )


class ExpansionCog[ParsedT](commands.Cog):
    """Base for a cog that expands one platform's links.

    A subclass declares the class attributes below, overrides `read` and `build_delivery`, and
    overrides `url_is_expandable` when its pattern matches more than posts.

    Attributes:
        bot (commands.Bot): The Discord bot instance that owns this cog.
    """

    SOURCE: ClassVar[LinkSourceName]
    """Pending-expansion table key and `LINK_SOURCE_EMOJIS` key for this platform."""

    PLATFORM: ClassVar[str]
    """The platform's display name, for log messages."""

    URL_PATTERN: ClassVar[re.Pattern[str]]
    """The first match in the open that `url_is_expandable` accepts selects the link to expand."""

    PLACEHOLDER_TEXT: ClassVar[str]
    """The line shown under the link until the card replaces it."""

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the cog.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot
        # Host-free on purpose: an image left without room keeps the link it already has.
        self.card_image_planner = MediaDeliveryPlanner(
            media_hosting=MediaHostingService(
                config=MediaHostingConfig.model_validate(obj={"MEDIA_HOSTING_ENABLED": False})
            )
        )
        self._resume_started = False
        # Captured before on_ready so the restart sweep skips expansions from this process.
        self._started_at = database_now()

    @staticmethod
    def url_is_expandable(url: str) -> bool:
        """Returns whether a matched URL names a post worth reading.

        Overridden by a cog whose pattern anchors on the host, where a profile or home page
        matches but is not a post.

        Args:
            url (str): The URL the pattern matched.
        """
        del url
        return True

    async def read(self, message: Message, url: str, stack: contextlib.AsyncExitStack) -> ParsedT:
        """Returns the parsed post read from `url` under this platform's wall-clock bound.

        Raises rather than reporting: the caller turns the exception into the reaction and the
        log line its class earns. Anything that must outlive the read and be cleaned up after
        delivery goes on `stack`, which unwinds once the card is on screen.

        Args:
            message (Message): The message carrying the link, so a log line here can be joined to
                the expansion it came from.
            url (str): The post to read.
            stack (contextlib.AsyncExitStack): Where to register whatever the delivery still
                needs.

        Raises:
            NotImplementedError: Always; a cog overrides this.
        """
        raise NotImplementedError

    async def build_delivery(
        self, message: Message, url: str, parsed: ParsedT
    ) -> ExpansionDelivery | None:
        """Returns the finished card for `parsed`, or None when there is nothing showable.

        None means the platform answered and there is nothing showable — a deleted or private
        post, or one past a limit no retry gets under. It earns the unreadable mark rather than
        the cross, so a cog returning None logs its own reason first.

        Args:
            message (Message): The message carrying the link.
            url (str): The post that was read.
            parsed (ParsedT): What `read` returned.

        Raises:
            NotImplementedError: Always; a cog overrides this.
        """
        raise NotImplementedError

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Resumes expansions interrupted by a restart, guarded to run once across reconnects."""
        if self._resume_started:
            return
        self._resume_started = True
        await resume_expansion_placeholders(
            bot=self.bot, source=self.SOURCE, expand=self._expand, created_before=self._started_at
        )

    async def _mark_failed(self, message: Message, current_emoji: str | None) -> None:
        """Reacts with the failure cross, adding the platform mark if `current_emoji` is None."""
        if current_emoji is None:
            await update_reaction(
                message=message, bot_user=self.bot.user, emoji=LINK_SOURCE_EMOJIS[self.SOURCE]
            )
        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji=EXPANSION_FAILED_EMOJI,
            previous=current_emoji,
        )

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Expands the first non-hidden URL matched by `URL_PATTERN` and `url_is_expandable`.

        Skips bot messages, messages addressed to the bot, and links inside spoiler bars or angle
        brackets.

        Args:
            message (Message): Incoming Discord message.
        """
        if message.author.bot:
            return

        matches = list(self.URL_PATTERN.finditer(string=message.content))
        if not matches:
            return
        hidden_spans = [
            (hidden.start(), hidden.end())
            for hidden in _HIDDEN_LINK_RE.finditer(string=message.content)
        ]
        url = next(
            (
                match.group(0)
                for match in matches
                if not any(start < match.start() < end for start, end in hidden_spans)
                and self.url_is_expandable(url=match.group(0))
            ),
            None,
        )
        if url is None:
            return

        # Messages addressed to the bot belong to gen_reply; checked after regex to skip non-links.
        if is_addressed_to_bot(message=message, bot_user=self.bot.user):
            return

        # None until a working reaction is added so _mark_failed only removes an existing mark.
        current_emoji: str | None = None
        try:
            placeholder = await send_expansion_placeholder(
                message=message, text=self.PLACEHOLDER_TEXT, source=self.SOURCE, url=url
            )
            if placeholder is None:
                # A channel refusing the placeholder will also refuse the card; skip reading.
                await self._mark_failed(message=message, current_emoji=current_emoji)
                return
            # Reactions follow placeholder send: reactions share a separate per-channel rate limit.
            await update_reaction(
                message=message, bot_user=self.bot.user, emoji=LINK_SOURCE_EMOJIS[self.SOURCE]
            )
            current_emoji = await update_reaction(
                message=message, bot_user=self.bot.user, emoji=EXPANSION_WORKING_EMOJI
            )
            try:
                await self._expand(
                    message=message, url=url, current_emoji=current_emoji, placeholder=placeholder
                )
            finally:
                # No-op once delivered; cleans up the placeholder on any early return.
                await placeholder.discard()
        # Broad on purpose: the listener's last line of defence, so nothing escapes into the
        # dispatcher and every failure still reaches the user as a reaction.
        except Exception as error:
            logfire.error(
                f"{self.PLATFORM} expansion failed outside the read and the send",
                url=url,
                message_id=message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            await self._mark_failed(message=message, current_emoji=current_emoji)

    async def _expand(
        self, message: Message, url: str, current_emoji: str, placeholder: ExpansionPlaceholder
    ) -> None:
        """Reads `url`, uploads card images, delivers onto `placeholder`, and updates reactions.

        Signature matches `resume_expansion_placeholders`'s `ExpansionRetry` protocol.
        """
        async with contextlib.AsyncExitStack() as stack:
            try:
                parsed = await self.read(message=message, url=url, stack=stack)
            # Broad on purpose: a read failure must not escape into the listener. Which mark it
            # earns comes off the error's class rather than a check here, so every platform
            # answers a refusal the same way.
            except Exception as error:
                report_expansion_read_failure(
                    error=error, platform=self.PLATFORM, url=url, message_id=message.id
                )
                await update_reaction(
                    message=message,
                    bot_user=self.bot.user,
                    emoji=expansion_failure_emoji(error=error),
                    previous=current_emoji,
                )
                return

            delivery = await self.build_delivery(message=message, url=url, parsed=parsed)
            if delivery is None:
                await update_reaction(
                    message=message,
                    bot_user=self.bot.user,
                    emoji=EXPANSION_UNREADABLE_EMOJI,
                    previous=current_emoji,
                )
                return

            await self._attach_card_images(message=message, url=url, delivery=delivery)
            await self._deliver(
                message=message,
                url=url,
                delivery=delivery,
                current_emoji=current_emoji,
                placeholder=placeholder,
            )

    async def _attach_card_images(
        self, message: Message, url: str, delivery: ExpansionDelivery
    ) -> None:
        """Downloads card images and uploads them as attachments within message upload limits."""
        # Reverse order prioritizes the linked/quoted post over ancestor embeds when caps bind.
        image_urls = list(
            dict.fromkeys(
                embed.image.url
                for embed in reversed(delivery.embeds)
                if embed.image.url and embed.image.url.startswith(("https://", "http://"))
            )
        )
        if not image_urls or not target_allows_file_uploads(target=message):
            return
        remaining_slots = DISCORD_ATTACHMENT_LIMIT - len(delivery.media)
        remaining_upload_limit = upload_limit_for(guild=message.guild) - sum(
            item.size for item in delivery.media
        )
        if remaining_slots <= 0 or remaining_upload_limit <= MEDIA_ENVELOPE_MARGIN:
            return

        tasks: list[Coroutine[Any, Any, LoadedMedia | None]] = []
        for image_url in image_urls:
            tasks.append(_download_card_image(image_url=image_url, url=url, message_id=message.id))
        downloaded = await asyncio.gather(*tasks)
        items = {
            image_url: MediaItem(
                source=image.data,
                filename=f"card_image_{index}{guess_extension(type=image.mime_type) or '.jpg'}",
            )
            for index, (image_url, image) in enumerate(zip(image_urls, downloaded, strict=True))
            if image is not None
        }
        plan = await self.card_image_planner.plan(
            items=list(items.values()),
            upload_limit=remaining_upload_limit,
            envelope_margin=MEDIA_ENVELOPE_MARGIN,
            attachment_limit=remaining_slots,
        )
        attached = {id(item) for item in plan.native}
        uploaded = {
            image_url: f"attachment://{item.filename}"
            for image_url, item in items.items()
            if id(item) in attached
        }
        for embed in delivery.embeds:
            attachment = uploaded.get(embed.image.url or "")
            if attachment is not None:
                embed.set_image(url=attachment)
        delivery.media.extend(plan.native)
        if len(uploaded) < len(items):
            logfire.info(
                "Card images past the message's upload room stay linked",
                url=url,
                message_id=message.id,
                linked=len(items) - len(uploaded),
            )

    async def _set_source_embed_suppressed(self, message: Message, suppress: bool) -> bool:
        """Toggles `message`'s embed suppression, returning whether the edit succeeded.

        Propagates `NotFound` when suppressing so the caller can withdraw the expansion; ignores
        `NotFound` when restoring.
        """
        action = "suppress" if suppress else "restore"
        try:
            await message.edit(suppress=suppress)
        except NotFound:
            if suppress:
                raise
            # Source message deleted meanwhile; nothing left to restore.
            return False
        except Forbidden as error:
            # Missing Manage Messages is expected; omit identical traceback.
            logfire.warn(
                f"Could not {action} the source message embed",
                message_id=message.id,
                guild_id=message.guild.id if message.guild else None,
                error_type=type(error).__name__,
            )
            return False
        except Exception as error:
            # Broad on purpose: toggling Discord's preview is cosmetic and must not abort
            # delivery or keep the failure mark off.
            logfire.warn(
                f"Could not {action} the source message embed",
                message_id=message.id,
                guild_id=message.guild.id if message.guild else None,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            return False
        return True

    async def _deliver(
        self,
        message: Message,
        url: str,
        delivery: ExpansionDelivery,
        current_emoji: str,
        placeholder: ExpansionPlaceholder,
    ) -> None:
        """Suppresses source preview, delivers the card, and restores the preview on failure."""
        try:
            # Restore preview only if this call hid it; otherwise the guild would refuse restore too.
            preview_hidden = await self._set_source_embed_suppressed(
                message=message, suppress=True
            )
        # Source message deleted mid-read withdraws the expansion.
        except NotFound:
            logfire.info(
                f"{self.PLATFORM} expansion target is gone",
                url=url,
                message_id=message.id,
                channel_id=message.channel.id,
            )
            return

        # Broad on purpose: delivery failures must not escape into the listener, and severity is
        # classified by `report_expansion_delivery_failure`.
        try:
            await placeholder.deliver(
                content=delivery.content,
                embeds=delivery.embeds,
                files=[item.to_file() for item in delivery.media],
            )
        except Exception as error:
            report_expansion_delivery_failure(
                error=error,
                platform=self.PLATFORM,
                url=url,
                message_id=message.id,
                channel_id=message.channel.id,
            )
            if preview_hidden:
                await self._set_source_embed_suppressed(message=message, suppress=False)
            await self._mark_failed(message=message, current_emoji=current_emoji)
            return

        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji=EXPANSION_DONE_EMOJI,
            previous=current_emoji,
        )


class _ConversationRead[ConversationT](BaseModel):
    """The parsed conversation paired with target videos downloaded on the expansion's stack."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    conversation: SkipValidation[ConversationT] = Field(
        ..., description="The parsed conversation returned by the platform reader."
    )
    video_paths: list[Path] = Field(
        default_factory=list,
        description="Target video files downloaded into the expansion's scratch directory.",
    )


class ConversationExpansionCog[PostT: CardPost, ConversationT: CardConversation[Any]](
    ExpansionCog[_ConversationRead[ConversationT]]
):
    """Base for a cog whose reader parses a post and its discussion in one blocking call.

    The read, optional target-video download, refusal of an unreadable post, and default card
    are written here. A subclass declares `ExpansionCog`'s attributes plus the three below and
    `_footer_text`, and `_comment_url` as well when its conversations can carry a selected
    comment; the remaining hooks have defaults it overrides where its platform differs. The
    default card is the post, its gallery and the comment its URL singled out; a platform whose
    card shows other context replaces `_build_embeds` and builds it from `post_card_embeds` and
    `context_card_budget`.
    """

    READ_TIMEOUT_SECONDS: ClassVar[float]
    """The wall-clock bound on one read; `typings/timeouts.py` owns the number."""

    EMBED_COLOR: ClassVar[int]
    """The platform's own colour, down the post card's left edge."""

    DOWNLOAD_VIDEOS: ClassVar[bool] = False
    """Whether to download and deliver target videos instead of linking them."""

    VIDEO_DOWNLOAD_MAX_BYTES: ClassVar[int] = 100 * 1024 * 1024
    """Per-video byte cap when external media hosting is available (100 MiB)."""

    VIDEO_DOWNLOAD_TIMEOUT_SECONDS: ClassVar[float] = EXPANSION_VIDEO_DOWNLOAD_TIMEOUT_SECONDS
    """Whole-operation wall-clock bound on downloading a post's target videos."""

    VIDEO_READ_TIMEOUT_SECONDS: ClassVar[float] = EXPANSION_VIDEO_READ_TIMEOUT_SECONDS
    """Per-socket read timeout for streamed target video downloads."""

    downloader_factory: Callable[[], ConversationReader[ConversationT]]
    """Builds the reader; the seam a test replaces to keep an expansion off the network."""

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the cog and its external media-delivery planner.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        super().__init__(bot=bot)
        self.media_delivery = build_media_delivery_planner()

    def _video_download_headers(self) -> dict[str, str]:
        """Returns HTTP headers used when downloading target videos."""
        return dict(_DEFAULT_VIDEO_HEADERS)

    def _hosting_available(self) -> bool:
        """Returns whether external media hosting is configured and its serve dir exists."""
        config = self.media_delivery.media_hosting.config
        return config.available and Path(config.serve_dir).is_dir()

    def _video_download_cap(self, message: Message) -> int:
        """Returns the max bytes worth downloading per video for `message`, or 0 to skip."""
        if self._hosting_available():
            return self.VIDEO_DOWNLOAD_MAX_BYTES
        if not target_allows_file_uploads(target=message):
            return 0
        native_cap = max(upload_limit_for(guild=message.guild) - MEDIA_ENVELOPE_MARGIN, 0)
        return min(self.VIDEO_DOWNLOAD_MAX_BYTES, native_cap)

    async def _download_one_video(  # noqa: PLR0913 -- one argument per download context field
        self,
        video_url: str,
        filepath: Path,
        headers: dict[str, str],
        max_bytes: int,
        url: str,
        message_id: int,
    ) -> Path | None:
        """Downloads one target video into `filepath`, returning None on oversize or fetch failure."""
        try:
            return await asyncio.to_thread(
                stream_to_file,
                url=video_url,
                filepath=filepath,
                headers=headers,
                timeout=self.VIDEO_READ_TIMEOUT_SECONDS,
                max_bytes=max_bytes,
                allow_empty=False,
            )
        except DownloadTooLargeError as error:
            logfire.info(
                f"{self.PLATFORM} video exceeds the byte cap; falling back to link",
                url=url,
                message_id=message_id,
                max_bytes=max_bytes,
                reason=str(error),
            )
        except requests.RequestException as error:
            logfire.warn(
                f"Could not download a {self.PLATFORM} video; falling back to link",
                url=url,
                message_id=message_id,
                retryable=is_retryable_fetch_failure(error=error),
                error_type=type(error).__name__,
                _exc_info=error,
            )
        except (OSError, ValueError) as error:
            logfire.warn(
                f"Could not download a {self.PLATFORM} video; falling back to link",
                url=url,
                message_id=message_id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
        return None

    async def _download_videos(
        self, post: PostT, download_dir: str, max_bytes: int, url: str, message_id: int
    ) -> list[Path]:
        """Downloads up to `DISCORD_ATTACHMENT_LIMIT` target videos into `download_dir`.

        Subclasses whose videos are not direct progressive HTTP URLs override this hook and
        return whichever files succeeded.
        """
        headers = self._video_download_headers()
        tasks: list[Coroutine[Any, Any, Path | None]] = []
        for index, video_url in enumerate(post.video_urls[:DISCORD_ATTACHMENT_LIMIT]):
            tasks.append(
                self._download_one_video(
                    video_url=video_url,
                    filepath=Path(download_dir) / f"video_{index}.mp4",
                    headers=headers,
                    max_bytes=max_bytes,
                    url=url,
                    message_id=message_id,
                )
            )
        results = await asyncio.gather(*tasks)
        return [path for path in results if path is not None]

    async def read(
        self, message: Message, url: str, stack: contextlib.AsyncExitStack
    ) -> _ConversationRead[ConversationT]:
        """Returns the parsed conversation and any downloaded target videos held on `stack`.

        Args:
            message (Message): The message carrying the link, used for upload limits and logging.
            url (str): The post to read.
            stack (contextlib.AsyncExitStack): Keeps downloaded videos alive until delivery ends.
        """
        downloader = self.downloader_factory()
        async with asyncio.timeout(delay=self.READ_TIMEOUT_SECONDS):
            conversation = await asyncio.to_thread(downloader.parse_metadata, url=url)

        target = conversation.target
        if not (
            self.DOWNLOAD_VIDEOS
            and target is not None
            and target.is_readable
            and target.video_urls
        ):
            return _ConversationRead(conversation=conversation)

        max_bytes = self._video_download_cap(message=message)
        if max_bytes <= 0:
            return _ConversationRead(conversation=conversation)

        download_dir = stack.enter_context(scratch_directory(prefix=f"parse-{self.SOURCE}-"))
        try:
            async with asyncio.timeout(delay=self.VIDEO_DOWNLOAD_TIMEOUT_SECONDS):
                video_paths = await self._download_videos(
                    post=target,
                    download_dir=download_dir,
                    max_bytes=max_bytes,
                    url=url,
                    message_id=message.id,
                )
        except TimeoutError as error:
            logfire.warn(
                f"{self.PLATFORM} video download timed out; falling back to link",
                url=url,
                message_id=message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
            video_paths = []
        return _ConversationRead(conversation=conversation, video_paths=video_paths)

    async def _plan_video_delivery(
        self, message: Message, url: str, total_videos: int, video_paths: list[Path]
    ) -> tuple[list[MediaItem], list[str], list[int]]:
        """Plans video attachments and hosted links, returning `(native, content_lines, undelivered_indices)`."""
        all_indices = list(range(total_videos))
        if not self.DOWNLOAD_VIDEOS or total_videos == 0 or not video_paths:
            return [], [], all_indices

        items = [MediaItem(source=path, filename=path.name) for path in video_paths]
        allows_uploads = target_allows_file_uploads(target=message)
        plan = await self.media_delivery.plan(
            items=items,
            upload_limit=upload_limit_for(guild=message.guild) if allows_uploads else 0,
            envelope_margin=MEDIA_ENVELOPE_MARGIN,
        )
        if plan.dropped_items:
            logfire.info(
                f"{self.PLATFORM} videos could not be attached or hosted; falling back to link",
                url=url,
                message_id=message.id,
                dropped=len(plan.dropped_items),
            )
        if not plan.native and not plan.hosted_urls:
            return [], [], all_indices

        attempted = min(total_videos, DISCORD_ATTACHMENT_LIMIT)
        clamped_count = max(total_videos - DISCORD_ATTACHMENT_LIMIT, 0) + len(plan.clamped_items)
        content_lines: list[str] = []
        if clamped_count:
            content_lines.append(
                f"-# 已省略 {clamped_count} 部影片 (Discord 單則訊息最多 "
                f"{DISCORD_ATTACHMENT_LIMIT} 個附件)"
            )
        content_lines.extend(plan.hosted_urls)

        undelivered_ids = {id(item) for item in (*plan.dropped_items, *plan.clamped_items)}
        delivered_indices = {
            int(match.group(1))
            if (match := _VIDEO_FILE_INDEX_RE.fullmatch(string=item.filename))
            else pos
            for pos, item in enumerate(items)
            if id(item) not in undelivered_ids
        }
        undelivered_indices = [
            index for index in range(attempted) if index not in delivered_indices
        ]
        if plan.dropped_items and not undelivered_indices:
            undelivered_indices = [0]
        return plan.native, content_lines, undelivered_indices

    async def build_delivery(
        self, message: Message, url: str, parsed: _ConversationRead[ConversationT]
    ) -> ExpansionDelivery | None:
        """Returns the finished card and target videos, or None when target is unreadable.

        Args:
            message (Message): The message carrying the link.
            url (str): The post that was read.
            parsed (_ConversationRead[ConversationT]): The conversation and downloaded videos.
        """
        conversation = parsed.conversation
        target = conversation.target
        if target is None or not target.is_readable:
            logfire.info(
                f"{self.PLATFORM} post is not readable; nothing to expand",
                url=url,
                message_id=message.id,
            )
            return None

        native_media, content_lines, undelivered_indices = await self._plan_video_delivery(
            message=message,
            url=url,
            total_videos=len(target.video_urls),
            video_paths=parsed.video_paths,
        )
        embeds = self._build_embeds(
            conversation=conversation, undelivered_video_indices=undelivered_indices
        )
        return ExpansionDelivery(
            content="\n".join(content_lines) if content_lines else None,
            embeds=embeds,
            media=native_media,
        )

    def _build_embeds(
        self, conversation: ConversationT, undelivered_video_indices: list[int] | None = None
    ) -> list[Embed]:
        """Builds the post embed, gallery embeds, and optional selected-comment embed."""
        post: PostT | None = conversation.target
        if post is None:
            return []
        comment: PostT | None = conversation.selected_comment
        missing_videos = (
            list(range(len(post.video_urls)))
            if undelivered_video_indices is None
            else undelivered_video_indices
        )
        hint = (
            VIDEO_HINT.format(url=self._video_link(post=post, index=missing_videos[0]))
            if missing_videos
            else ""
        )
        shown = post.image_urls[:POST_CARD_MAX_IMAGES]
        footer_parts = [
            part
            for part in (
                self._footer_text(post=post, shown_images=len(shown)),
                omitted_videos_note(count=len(missing_videos) - 1)
                if len(missing_videos) > 1
                else "",
            )
            if part
        ]
        embeds = post_card_embeds(
            post=post,
            color=self.EMBED_COLOR,
            author=self._author_label(post=post),
            footer=" · ".join(footer_parts),
            images=shown,
            suffix=hint,
            reserve=_COMMENT_RESERVE if comment is not None else 0,
        )
        if comment is not None:
            embeds.append(
                self._comment_embed(
                    comment=comment, post_url=post.url, budget=context_card_budget(card=embeds[0])
                )
            )
        return embeds

    def _comment_embed(self, comment: PostT, post_url: str, budget: int) -> Embed:
        """Returns the neutral-rail embed for the comment singled out by the URL."""
        header = f"{_COMMENT_HEADER}\n\n"
        # Beside a short post the message-wide budget exceeds what one description may carry.
        body = clip_to_utf16_limit(
            text=comment.text,
            limit=min(budget, DISCORD_EMBED_DESCRIPTION_LIMIT) - utf16_length(value=header),
            notice=TRUNCATION_NOTICE,
        )
        embed = Embed(
            description=f"{header}{body}",
            # Distinct URL keeps Discord from merging the comment into the post's image gallery.
            url=self._comment_url(post_url=post_url, comment=comment),
            color=Color(value=CONTEXT_CARD_COLOR),
            timestamp=comment.taken_at,
        )
        author = self._comment_author_label(comment=comment)
        if author:
            embed.set_author(name=author, icon_url=comment.author_icon_url or None)
        return embed

    def _author_label(self, post: PostT) -> str:
        """The post card's author line, empty for none."""
        return post.author_name

    def _comment_author_label(self, comment: PostT) -> str:
        """The comment card's author line, empty for none."""
        return comment.author_name

    def _video_link(self, post: PostT, index: int = 0) -> str:
        """Where a video post's fallback link points; defaults to `post.video_urls[index]`."""
        return post.video_urls[index] if index < len(post.video_urls) else post.video_urls[0]

    @staticmethod
    def _omitted_media_notes(post: CardPost, shown_images: int) -> list[str]:
        """Returns the footer note for images past `shown_images`."""
        remaining_images = len(post.image_urls) - shown_images
        return [omitted_images_note(count=remaining_images)] if remaining_images > 0 else []

    def _footer_text(self, post: PostT, shown_images: int) -> str:
        """The post card's counter line, given how many of its images the card shows.

        Raises:
            NotImplementedError: Always; a cog overrides this.
        """
        raise NotImplementedError

    def _comment_url(self, post_url: str, comment: PostT) -> str:
        """The comment card's own link, which must differ from the post's.

        Raises:
            NotImplementedError: Always; a cog whose conversations can carry a selected comment
                overrides this.
        """
        raise NotImplementedError
