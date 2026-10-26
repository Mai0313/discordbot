"""The shell around an auto-expansion, so a platform module contributes only its card.

Expanding a pasted link is one feature: claim a reply slot under the link, mark the message,
read the post, upload the card's images, edit the card onto the slot, mark the outcome. None of
that differs per platform, and a difference that turns up there is a defect rather than a choice.
So it lives here once, and a cog supplies the platform: where its URLs are, how to read one, and
what the card looks like.

`utils/expansion_placeholder.py` owns the other half of this contract, the parts a resumed
expansion needs as much as a fresh one does. This module is the cog side.

`ConversationExpansionCog` goes one step further for a platform whose reader fetches a post and
its discussion in one blocking call and whose card is embeds alone: the read, the refusal of an
unreadable post and the card's shape are written once, and a cog supplies the parts that are its
platform's own.

A failure leaves nothing in the channel. The reaction is the whole report, which is what lets
every step below simply return.

There is deliberately no kill-switch anywhere in this feature: turning an expansion off means
deleting its cog.
"""

import re
from typing import Any, ClassVar, Protocol
import asyncio
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
from discordbot.typings.timeouts import CARD_IMAGE_FETCH_TIMEOUT_SECONDS
from discordbot.utils.link_errors import (
    LinkReadError,
    LinkRetryableError,
    LinkUnavailableError,
    is_retryable_fetch_failure,
)
from discordbot.utils.discord_embeds import (
    DISCORD_ATTACHMENT_LIMIT,
    DISCORD_EMBED_TOTAL_LIMIT,
    DISCORD_EMBED_DESCRIPTION_LIMIT,
    utf16_length,
    embed_text_length,
    clip_to_utf16_limit,
    target_allows_file_uploads,
)
from discordbot.utils.media_delivery import (
    MEDIA_ENVELOPE_MARGIN,
    MediaItem,
    MediaHostingConfig,
    MediaHostingService,
    MediaDeliveryPlanner,
    upload_limit_for,
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

# What a post card shows before it starts scrolling the channel. A gallery post can carry far
# more, so the footer says how many were left behind and the embed's own link leads to the rest.
# Not a `context_budgets` constant: nothing here reaches a model, this bounds a rendered message.
POST_CARD_MAX_IMAGES = 4

# The rail of every card that is not the linked post itself (a comment it singled out, a post it
# replies to or quotes), neutral so the one being linked is never confused with its context. The
# platform's own brand colour belongs to the cog wearing it, and neither is in
# `typings/colors.py`, which is Discord's own semantic palette.
CONTEXT_CARD_COLOR = 0x65686C

TRUNCATION_NOTICE = "\n\n⋯（全文請看原貼文）"
VIDEO_HINT = "\n\n🎬 [點此觀看影片]({url})"

# What a secondary card's budget leaves unmeasured: every card's author line, plus its header and
# the line break under it on a card that does not take them off its own budget.
# Every measurement against Discord's limits is in UTF-16 units (`utf16_length`), Discord's own.
_CONTEXT_CARD_SLACK = 400

# What the post gives up so the comment its URL singled out always fits beside it.
_COMMENT_RESERVE = 2000
_COMMENT_HEADER = "💬 **指定的留言**"

# Where a poster kept a link out of sight: behind spoiler bars, which may cover the text around it
# too, or in the angle brackets that turn Discord's preview off. Paired the way Discord pairs them,
# left to right and shortest first, so `||a|| link ||b||` leaves the link in the open. Code spans
# and escapes are not parsed, so a `||` inside one pairs with the next real bar.
_HIDDEN_LINK_RE = re.compile(pattern=r"\|\|.+?\|\||<[^\s<>]+>", flags=re.DOTALL)


class CardPost(Protocol):
    """What the shared card reads off one post or comment.

    Spelled structurally because the platforms' own models live under `services/`, which
    `utils/` may not import.
    """

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

    The further embeds reuse the card's URL, which makes Discord merge them into one gallery
    under the card rather than stacking separate cards.

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
    """Returns the linked post's own embed followed by one bare embed per further image.

    A long body is cut rather than split across a second embed: the whole card is one post, and
    a reader who wants the tail has the link.

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
    """Returns the UTF-16 units left under the message-wide ceiling for secondary cards.

    Budgeted across all secondary cards rather than clipping each card independently, so their
    sum never exceeds Discord's message-wide embed limit.

    Args:
        card (Embed): The linked post's own embed.
    """
    return DISCORD_EMBED_TOTAL_LIMIT - embed_text_length(embed=card) - _CONTEXT_CARD_SLACK


def fetch_card_image(url: str) -> LoadedMedia:
    """Returns the image at `url` as served, for a card to upload rather than link.

    Args:
        url (str): The image URL the card would otherwise link.

    Raises:
        requests.RequestException: The fetch failed or answered an error status.
        ValueError: What came back is not an image.
    """
    response = requests.get(url=url, timeout=CARD_IMAGE_FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    mime_type = response.headers.get("Content-Type", "").partition(";")[0].strip()
    if not mime_type.startswith("image/"):
        raise ValueError(f"{url} served {mime_type or 'no type'}, not an image")
    return LoadedMedia(data=response.content, mime_type=mime_type)


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

    Only the placeholder's own disappearance is routine. On an edit, error code 50035 is a
    rejected body rather than a missing message, so `is_reply_target_gone` is not used here.

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
        # No traceback, whose stack is the same on every refusal; the code says which refusal it
        # was, since not every one here is a missing permission.
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
    """Keys this cog's rows in the pending-expansion table and its marker in
    `LINK_SOURCE_EMOJIS`, so one platform is spelled one way wherever it is named."""

    PLATFORM: ClassVar[str]
    """The platform's display name, for log messages."""

    URL_PATTERN: ClassVar[re.Pattern[str]]
    """The first match in the open that `url_is_expandable` accepts selects the link to expand."""

    PLACEHOLDER_TEXT: ClassVar[str]
    """The line shown under the link until the card replaces it."""

    def __init__(self, bot: commands.Bot):
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
        # Taken at load, before the gateway connects, as `DiscordBot._started_at` is: the restart
        # sweep must not re-run an expansion this process starts ahead of `on_ready`.
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
        """Runs again the expansions a restart interrupted (once per process).

        `on_ready` fires on every gateway reconnect, so the flag guards it to one sweep.
        """
        if self._resume_started:
            return
        self._resume_started = True
        await resume_expansion_placeholders(
            bot=self.bot, source=self.SOURCE, expand=self._expand, created_before=self._started_at
        )

    async def _mark_failed(self, message: Message, current_emoji: str | None) -> None:
        """Paints the failure cross, naming the platform when nothing else on the message does.

        `current_emoji` is None only when claiming the reply slot failed, before either mark
        went on. The cross alone cannot say which link died, so the platform marker goes on
        first.
        """
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
        """Expands the first link in the open that this cog's pattern matches and accepts.

        A refused match is skipped rather than ending the scan, so a profile link ahead of a
        post never hides the post. A link behind spoiler bars or in angle brackets is skipped
        the same way: the poster asked Discord not to show it, and a card would show it to
        everyone.

        Args:
            message (Message): The message that was sent.
        """
        if message.author.bot:
            return

        url = next(
            (
                match.group(0)
                for match in self.URL_PATTERN.finditer(string=message.content)
                if not any(
                    hidden.start() < match.start() < hidden.end()
                    for hidden in _HIDDEN_LINK_RE.finditer(string=message.content)
                )
                and self.url_is_expandable(url=match.group(0))
            ),
            None,
        )
        if url is None:
            return

        # A link addressed to the bot is gen_reply's to answer about, not ours to expand.
        # Checked after the URL match so the common no-link message costs one regex, not two.
        if is_addressed_to_bot(message=message, bot_user=self.bot.user):
            return

        # Nothing is on the message yet, and the handler below is reachable before anything is:
        # claiming the reply slot is itself a step that can fail. Left unset rather than
        # pre-filled so a failure mark removes a reaction only when one is there.
        current_emoji: str | None = None
        try:
            placeholder = await send_expansion_placeholder(
                message=message, text=self.PLACEHOLDER_TEXT, source=self.SOURCE, url=url
            )
            if placeholder is None:
                # A channel that refused the placeholder will refuse the card too, so the read
                # is never started.
                await self._mark_failed(message=message, current_emoji=current_emoji)
                return
            # The platform marker, then the working ring under it. Both go on after the reply
            # slot is claimed: they share one per-channel rate-limit bucket that a message send
            # does not, so claiming first is what stops the card queueing behind them.
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
                # Every step below returns rather than raising, so this one line covers all of
                # them: an expansion that delivered nothing leaves nothing behind. Once
                # delivered it is a no-op.
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
        """Reads the post and puts it on screen, marking the outcome it earned.

        The signature is the one `resume_expansion_placeholders` calls with, so a resumed
        expansion runs exactly what the listener runs.
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
        """Swaps the card's linked images for uploaded copies, as far as the message has room.

        Discord fetches a linked embed image once, as the card is sent, and keeps a failed fetch
        as an empty image it never retries; on Instagram's CDN that emptied about a third of the
        cards (#1122). An image that fails to download, that the room left beside the card's own
        media cannot carry, or in a channel refusing attachments keeps its link.
        """
        # Walked from the last embed: a Threads card puts the linked post and the post it quotes
        # after the chain above them, so when the attachment cap binds, a far ancestor's image is
        # the one left linked. No other card carries enough images for the cap to bind.
        image_urls = list(
            dict.fromkeys(
                embed.image.url
                for embed in reversed(delivery.embeds)
                if embed.image.url and embed.image.url.startswith(("https://", "http://"))
            )
        )
        if not image_urls or not target_allows_file_uploads(target=message):
            return

        tasks: list[Coroutine[Any, Any, LoadedMedia | None]] = []
        for image_url in image_urls:
            tasks.append(_download_card_image(image_url=image_url, url=url, message_id=message.id))
        downloaded = await asyncio.gather(*tasks)
        items = {
            image_url: MediaItem(
                source=image.data,
                filename=f"card_image_{index}{guess_extension(image.mime_type) or '.jpg'}",
            )
            for index, (image_url, image) in enumerate(zip(image_urls, downloaded, strict=True))
            if image is not None
        }
        plan = await self.card_image_planner.plan(
            items=list(items.values()),
            upload_limit=upload_limit_for(guild=message.guild)
            - sum(item.size for item in delivery.media),
            envelope_margin=MEDIA_ENVELOPE_MARGIN,
            attachment_limit=DISCORD_ATTACHMENT_LIMIT - len(delivery.media),
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

    async def _deliver(
        self,
        message: Message,
        url: str,
        delivery: ExpansionDelivery,
        current_emoji: str,
        placeholder: ExpansionPlaceholder,
    ) -> None:
        """Edits the card onto the placeholder and marks the source message done.

        A card that never lands hands back the preview hidden for it, so the link is left with
        its own preview rather than with nothing.
        """
        # Only a preview this call did hide is shown again: a guild that refused the hide would
        # refuse showing it too.
        preview_hidden = False
        # Broad on purpose: the delivery step must never escape into the listener, and the
        # severity its failure earns is `report_expansion_delivery_failure`'s to pick.
        try:
            try:
                await message.edit(suppress=True)
                preview_hidden = True
            # Deleting the link while the post was being read withdraws the request.
            except NotFound:
                logfire.info(
                    f"{self.PLATFORM} expansion target is gone",
                    url=url,
                    message_id=message.id,
                    channel_id=message.channel.id,
                )
                return
            # A guild without Manage Messages refuses this on every expansion with an identical
            # stack, so the ids are the whole finding.
            except Forbidden as error:
                logfire.warn(
                    "Could not suppress the source message embed",
                    message_id=message.id,
                    guild_id=message.guild.id if message.guild else None,
                    error_type=type(error).__name__,
                )
            # Broad on purpose: hiding Discord's own preview is cosmetic and must not abort the
            # expansion.
            except Exception as error:
                logfire.warn(
                    "Could not suppress the source message embed",
                    message_id=message.id,
                    guild_id=message.guild.id if message.guild else None,
                    error_type=type(error).__name__,
                    _exc_info=error,
                )

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
                try:
                    await message.edit(suppress=False)
                # The link was deleted meanwhile, so there is no preview left to hand back.
                except NotFound:
                    pass
                # The guild took Manage Messages away since the hide; the ids are the whole
                # finding.
                except Forbidden as restore_error:
                    logfire.warn(
                        "Could not restore the source message embed",
                        message_id=message.id,
                        guild_id=message.guild.id if message.guild else None,
                        error_type=type(restore_error).__name__,
                    )
                # Broad on purpose: showing the preview again is as cosmetic as hiding it, and
                # must not keep the failure mark off.
                except Exception as restore_error:
                    logfire.warn(
                        "Could not restore the source message embed",
                        message_id=message.id,
                        guild_id=message.guild.id if message.guild else None,
                        error_type=type(restore_error).__name__,
                        _exc_info=restore_error,
                    )
            await self._mark_failed(message=message, current_emoji=current_emoji)
            return

        await update_reaction(
            message=message,
            bot_user=self.bot.user,
            emoji=EXPANSION_DONE_EMOJI,
            previous=current_emoji,
        )


class ConversationExpansionCog[PostT: CardPost, ConversationT: CardConversation[Any]](
    ExpansionCog[ConversationT]
):
    """Base for a cog whose reader parses a post and its discussion in one blocking call.

    The read, the refusal of an unreadable post and the card are written here. A subclass
    declares `ExpansionCog`'s attributes plus the three below and `_footer_text`, and
    `_comment_url` as well when its conversations can carry a selected comment; the remaining
    hooks have defaults it overrides where its platform differs. The default card is the post,
    its gallery and the comment its URL singled out; a platform whose card shows other context
    replaces `_build_embeds` and builds it from `post_card_embeds` and `context_card_budget`.
    """

    READ_TIMEOUT_SECONDS: ClassVar[float]
    """The wall-clock bound on one read; `typings/timeouts.py` owns the number."""

    EMBED_COLOR: ClassVar[int]
    """The platform's own colour, down the post card's left edge."""

    downloader_factory: Callable[[], ConversationReader[ConversationT]]
    """Builds the reader; the seam a test replaces to keep an expansion off the network."""

    async def read(
        self, message: Message, url: str, stack: contextlib.AsyncExitStack
    ) -> ConversationT:
        """Returns the parsed conversation read under `READ_TIMEOUT_SECONDS` off the event loop.

        Args:
            message (Message): Unused; nothing here logs.
            url (str): The post to read.
            stack (contextlib.AsyncExitStack): Unused; nothing here outlives the read.
        """
        del message, stack
        downloader = self.downloader_factory()
        async with asyncio.timeout(delay=self.READ_TIMEOUT_SECONDS):
            return await asyncio.to_thread(downloader.parse_metadata, url=url)

    async def build_delivery(
        self, message: Message, url: str, parsed: ConversationT
    ) -> ExpansionDelivery | None:
        """Returns the finished card for `parsed`, or None when the post is not readable.

        A post that comes back unreadable is a deleted, private, protected or login-walled one:
        an ordinary outcome for a link someone pasted rather than a defect.

        Args:
            message (Message): The message carrying the link.
            url (str): The post that was read.
            parsed (ConversationT): The parsed conversation.
        """
        target = parsed.target
        if target is None or not target.is_readable:
            logfire.info(
                f"{self.PLATFORM} post is not readable; nothing to expand",
                url=url,
                message_id=message.id,
            )
            return None
        return ExpansionDelivery(embeds=self._build_embeds(conversation=parsed))

    def _build_embeds(self, conversation: ConversationT) -> list[Embed]:
        """Builds the whole expansion: the post, its images, and the named comment if any.

        A video post carries a link to it, since nothing here attaches the clip and the card
        would otherwise have nothing in it.
        """
        post: PostT | None = conversation.target
        if post is None:
            return []
        comment: PostT | None = conversation.selected_comment
        hint = VIDEO_HINT.format(url=self._video_link(post=post)) if post.video_urls else ""
        shown = post.image_urls[:POST_CARD_MAX_IMAGES]
        embeds = post_card_embeds(
            post=post,
            color=self.EMBED_COLOR,
            author=self._author_label(post=post),
            footer=self._footer_text(post=post, shown_images=len(shown)),
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
        """Returns the card for the one comment the URL singled out.

        Grey rather than the post's colour, and headed by a line saying what it is. Its URL is
        the comment's own, which keeps it out of the image gallery since Discord merges embeds
        sharing a URL.
        """
        header = f"{_COMMENT_HEADER}\n\n"
        # Beside a short post the message-wide budget exceeds what one description may carry.
        body = clip_to_utf16_limit(
            text=comment.text,
            limit=min(budget, DISCORD_EMBED_DESCRIPTION_LIMIT) - utf16_length(value=header),
            notice=TRUNCATION_NOTICE,
        )
        embed = Embed(
            description=f"{header}{body}",
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

    def _video_link(self, post: PostT) -> str:
        """Where a video post's link points; only its first video gets one."""
        return post.video_urls[0]

    @staticmethod
    def _omitted_media_notes(post: CardPost, shown_images: int) -> list[str]:
        """Returns the footer notes for images past `shown_images` and videos past the first."""
        notes: list[str] = []
        remaining_images = len(post.image_urls) - shown_images
        if remaining_images > 0:
            notes.append(omitted_images_note(count=remaining_images))
        remaining_videos = len(post.video_urls) - 1
        if remaining_videos > 0:
            notes.append(f"🎬 另有 {remaining_videos} 部影片")
        return notes

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
