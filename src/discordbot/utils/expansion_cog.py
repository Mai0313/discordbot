"""The shell around an auto-expansion, so a platform module contributes only its card.

Expanding a pasted link is one feature: claim a reply slot under the link, mark the message,
read the post, edit the card onto the slot, mark the outcome. None of that differs per platform,
and a difference that turns up there is a defect rather than a choice. So it lives here once, and
a cog supplies the platform: where its URLs are, how to read one, and what the card looks like.

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
import contextlib
from collections.abc import Callable

import logfire
from nextcord import File, Color, Embed, Message, NotFound, Forbidden
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.ext import commands

from discordbot.typings.emojis import LINK_SOURCE_EMOJIS, LinkSourceName
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.reactions import update_reaction
from discordbot.utils.link_errors import LinkReadError, LinkRetryableError, LinkUnavailableError
from discordbot.utils.discord_embeds import (
    DISCORD_EMBED_TOTAL_LIMIT,
    DISCORD_EMBED_DESCRIPTION_LIMIT,
    utf16_length,
    embed_text_length,
    clip_to_utf16_limit,
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
        """Still images, as URLs Discord fetches itself."""
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
        """Reads the post at `url`."""
        ...


def with_gallery(card: Embed, images: list[str]) -> list[Embed]:
    """Shows the first image on `card` and each further one on a bare embed after it.

    The further embeds reuse the card's URL, which is what makes Discord merge them into one
    gallery under the card rather than stacking separate cards.

    Args:
        card: The post's own embed.
        images: The images to show, empty for none.

    Returns:
        The card followed by its gallery.
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
    """Builds the linked post's own embed plus one bare embed per further image.

    A long body is cut rather than split across a second embed: the whole card is one post, and
    a reader who wants the tail has the link.

    Args:
        post: The linked post.
        color: The platform's own colour for the post's rail.
        author: The author line, empty for none.
        footer: The counter line.
        images: The images to show, the first on the post's own embed.
        suffix: Text appended after the body. Its length is reserved BEFORE the body is cut
            rather than appended after, or a body already at the ceiling carries it past and
            Discord rejects the send.
        reserve: What the body gives up of the message-wide ceiling so the secondary cards
            under it always fit, 0 when there are none.

    Returns:
        The post's embed followed by its gallery.
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
    """What the message-wide ceiling leaves the secondary cards once the post's embed is spent.

    Budgeted rather than clipping each card on its own, since cards clipped independently can
    sum past the ceiling and Discord rejects the WHOLE send, losing the expansion rather than
    trimming it.

    Args:
        card: The linked post's own embed.

    Returns:
        The UTF-16 units left for every secondary card together.
    """
    return DISCORD_EMBED_TOTAL_LIMIT - embed_text_length(embed=card) - _CONTEXT_CARD_SLACK


def expansion_failure_emoji(error: Exception) -> str:
    """Picks the mark a failed read earns, the same way for every platform.

    Read off the exception's CLASS rather than its message: a platform refusing the request or a
    transport that never answered or dropped part-way through the body is the retryable mark,
    since the link is fine and works later; anything else `LinkReadError` covers means the
    platform answered and there is no post in it; and an error from outside that tree is the
    bot's own. `utils/link_errors.py` owns which fetch failures are classified at all and why a
    403 deliberately is not.

    Args:
        error: What the read raised.

    Returns:
        One of the `EXPANSION_*_EMOJI` outcome marks.
    """
    if isinstance(error, LinkRetryableError | TimeoutError):
        return EXPANSION_RETRY_LATER_EMOJI
    if isinstance(error, LinkReadError):
        return EXPANSION_UNREADABLE_EMOJI
    return EXPANSION_FAILED_EMOJI


def report_expansion_read_failure(
    error: Exception, platform: str, url: str, message_id: int
) -> None:
    """Logs a failed read at the severity `.github/CONTRIBUTING.md#logging` gives its outcome.

    The ladder is keyed on how tolerable the failure is, not on how deep it happened, so the
    levels line up with the marks `expansion_failure_emoji` picks rather than with any one
    platform's habits: a post the platform says is gone is a routine user-driven outcome, a read
    the platform explains any other way is degraded but handled, and an error from outside that
    tree broke a user-visible deliverable — a `TimeoutError` excepted, which rides `warn` for the
    same reason it rides the retryable mark: it says the read was too slow, never that the bot is
    wrong.

    The `info` branch deliberately carries no exception and does carry `reason`: a traceback for
    a deleted post is noise, while the platform's own words for WHY it refused exist in no other
    line, and a deleted post logged at `warn` with a traceback is what makes a real regression
    unfindable.

    Args:
        error: What the read raised.
        platform: The platform's display name, for the log message.
        url: The post being expanded.
        message_id: The source message carrying the link.
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
    """Logs a failed delivery at the severity it deserves, the same way for every platform.

    Only the placeholder's own disappearance is routine. On an edit 50035 is a rejected body
    rather than a message that went away, so it is a defect here and
    `utils/discord_errors.py::is_reply_target_gone` must not be used.

    Args:
        error: What the delivery raised.
        platform: The platform's display name, for the log message.
        url: The post being expanded.
        message_id: The source message carrying the link.
        channel_id: The channel it was posted in.
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
        # was, since not every one here is a missing permission (#719).
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
    files: list[SkipValidation[File]] = Field(
        default_factory=list, description="Media attaching natively beside the card."
    )


class ExpansionCog[ParsedT](commands.Cog):
    """Base for a cog that expands one platform's links.

    A subclass declares the class attributes below, overrides `read` and `build_delivery`, and
    overrides `url_is_expandable` when its pattern matches more than posts.

    Attributes:
        bot: The Discord bot instance that owns this cog.
    """

    SOURCE: ClassVar[LinkSourceName]
    """Keys this cog's rows in the pending-expansion table and its marker in
    `LINK_SOURCE_EMOJIS`, so one platform is spelled one way wherever it is named."""

    PLATFORM: ClassVar[str]
    """The platform's display name, for log messages."""

    URL_PATTERN: ClassVar[re.Pattern[str]]
    """The first match `url_is_expandable` accepts selects the link to expand."""

    PLACEHOLDER_TEXT: ClassVar[str]
    """The line shown under the link until the card replaces it."""

    def __init__(self, bot: commands.Bot):
        """Initializes the cog.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self._resume_started = False

    @staticmethod
    def url_is_expandable(url: str) -> bool:
        """Whether a matched URL is worth reading.

        Overridden by a cog whose pattern anchors on the host, where a profile or home page
        matches but is not a post: reading one spends a request to find out there is nothing
        there, and paints a failure on a link that was never wrong.

        Args:
            url: The URL the pattern matched.

        Returns:
            True when the URL names a post.
        """
        del url
        return True

    async def read(self, message: Message, url: str, stack: contextlib.AsyncExitStack) -> ParsedT:
        """Reads the post, under this platform's own wall-clock bound.

        Raises rather than reporting: the caller turns the exception into the reaction and the
        log line its class earns, so every platform answers a refusal the same way.

        Anything that must outlive the read and be cleaned up after delivery goes on `stack` —
        a scratch directory, a downloaded file, a walk's context manager. It unwinds once the
        card is on screen.

        Args:
            message: The message carrying the link, so a log line here can be joined to the
                expansion it came from.
            url: The post to read.
            stack: Where to register whatever the delivery still needs.

        Returns:
            The platform's own parsed model.

        Raises:
            NotImplementedError: Always; a cog overrides this.
        """
        raise NotImplementedError

    async def build_delivery(
        self, message: Message, url: str, parsed: ParsedT
    ) -> ExpansionDelivery | None:
        """Turns a parsed post into the card, or refuses it.

        None means the platform answered and there is nothing showable — a deleted or private
        post, or one past a limit no retry gets under. It earns the unreadable mark rather than
        the cross, so a cog returning None logs its own reason first.

        Args:
            message: The message carrying the link.
            url: The post that was read.
            parsed: What `read` returned.

        Returns:
            The card, or None when there is nothing to show.

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
        await resume_expansion_placeholders(bot=self.bot, source=self.SOURCE, expand=self._expand)

    async def _mark_failed(self, message: Message, current_emoji: str | None) -> None:
        """Paints the failure cross, naming the platform when nothing else on the message does.

        `current_emoji` is None only when claiming the reply slot failed, before either mark
        went on. The cross alone cannot say WHICH link died, and on a message carrying two that
        is the whole of what someone needs, so the platform marker goes on first.
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
        """Expands the first link this cog's pattern matches and `url_is_expandable` accepts.

        A refused match is skipped rather than ending the scan, so a profile link ahead of a
        post never hides the post.

        Args:
            message: The message that was sent.
        """
        if message.author.bot:
            return

        url = next(
            (
                match.group(0)
                for match in self.URL_PATTERN.finditer(string=message.content)
                if self.url_is_expandable(url=match.group(0))
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
            # The platform marker, then the working ring under it. Both go on AFTER the reply
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

        Args:
            message: The message carrying the link.
            url: The post to expand.
            current_emoji: The mark already on the message, which each outcome replaces.
            placeholder: The reply slot to deliver onto.
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

            await self._deliver(
                message=message,
                url=url,
                delivery=delivery,
                current_emoji=current_emoji,
                placeholder=placeholder,
            )

    async def _deliver(
        self,
        message: Message,
        url: str,
        delivery: ExpansionDelivery,
        current_emoji: str,
        placeholder: ExpansionPlaceholder,
    ) -> None:
        """Edits the card onto the placeholder and marks the source message done."""
        # Broad on purpose: the delivery step must never escape into the listener, and the
        # severity its failure earns is `report_expansion_delivery_failure`'s to pick.
        try:
            try:
                await message.edit(suppress=True)
            # Deleting the link while the post was being read is a withdrawal: before the
            # placeholder existed the late reply was simply refused, and this keeps that, since
            # a reply outlives the message it answers.
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
                content=delivery.content, embeds=delivery.embeds, files=delivery.files
            )
        except Exception as error:
            report_expansion_delivery_failure(
                error=error,
                platform=self.PLATFORM,
                url=url,
                message_id=message.id,
                channel_id=message.channel.id,
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
        """Reads the post under `READ_TIMEOUT_SECONDS`, off the event loop since it blocks.

        Args:
            message: Unused; nothing here logs.
            url: The post to read.
            stack: Unused; nothing here outlives the read.

        Returns:
            The parsed conversation.
        """
        del message, stack
        downloader = self.downloader_factory()
        async with asyncio.timeout(delay=self.READ_TIMEOUT_SECONDS):
            return await asyncio.to_thread(downloader.parse_metadata, url=url)

    async def build_delivery(
        self, message: Message, url: str, parsed: ConversationT
    ) -> ExpansionDelivery | None:
        """Builds the card, refusing a post with nothing showable in it.

        A post that comes back unreadable is a deleted, private, protected or login-walled one:
        an ordinary outcome for a link someone pasted rather than a defect.

        Args:
            message: The message carrying the link.
            url: The post that was read.
            parsed: The parsed conversation.

        Returns:
            The card, or None when there is nothing to show.
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
        """The card for the one comment the URL singled out.

        Grey rather than the post's colour, and headed by a line saying what it is: without both,
        a second card under the post reads as a second post rather than as a reply to this one.
        Its URL is the comment's own, which is what keeps it OUT of the image gallery, since
        Discord merges embeds sharing a URL.
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
        """The footer notes for what the card leaves out of a post's media.

        Images past the ones shown are counted, and so is every video but the first, since only
        the first gets a link.

        Args:
            post: The linked post.
            shown_images: How many of its images the card shows.

        Returns:
            One note per kind left out, empty when nothing was.
        """
        notes: list[str] = []
        remaining_images = len(post.image_urls) - shown_images
        if remaining_images > 0:
            notes.append(f"🖼️ 另有 {remaining_images} 張")
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
