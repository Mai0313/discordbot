"""The shell around an auto-expansion, so a platform module contributes only its card.

Expanding a pasted link is one feature: claim a reply slot under the link, mark the message,
read the post, edit the card onto the slot, mark the outcome. None of that differs per platform,
and a difference that turns up there is a defect rather than a choice. So it lives here once, and
a cog supplies the platform: where its URLs are, how to read one, and what the card looks like.

`utils/expansion_placeholder.py` owns the other half of this contract, the parts a resumed
expansion needs as much as a fresh one does. This module is the cog side.

A failure leaves nothing in the channel. The reaction is the whole report, which is what lets
every step below simply return.

There is deliberately no kill-switch anywhere in this feature: turning an expansion off means
deleting its cog.
"""

import re
from typing import ClassVar
import contextlib

import logfire
from nextcord import File, Embed, Message, NotFound
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.ext import commands

from discordbot.typings.emojis import LINK_SOURCE_EMOJIS
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.reactions import update_reaction
from discordbot.utils.expansion_placeholder import (
    EXPANSION_DONE_EMOJI,
    EXPANSION_FAILED_EMOJI,
    EXPANSION_WORKING_EMOJI,
    EXPANSION_UNREADABLE_EMOJI,
    ExpansionPlaceholder,
    expansion_failure_emoji,
    send_expansion_placeholder,
    report_expansion_read_failure,
    resume_expansion_placeholders,
    report_expansion_delivery_failure,
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

    SOURCE: ClassVar[str]
    """Keys this cog's rows in the pending-expansion table and its marker in
    `LINK_SOURCE_EMOJIS`, so one platform is spelled one way wherever it is named."""

    PLATFORM: ClassVar[str]
    """The platform's display name, for log messages."""

    URL_PATTERN: ClassVar[re.Pattern[str]]
    """The first match in a message selects the link to expand."""

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
    def url_is_expandable(*, url: str) -> bool:
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

    async def read(
        self, *, message: Message, url: str, stack: contextlib.AsyncExitStack
    ) -> ParsedT:
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
        self, *, message: Message, url: str, parsed: ParsedT
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

    async def _mark_failed(self, *, message: Message, current_emoji: str | None) -> None:
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
        """Expands the first link this cog's pattern matches.

        Args:
            message: The message that was sent.
        """
        if message.author.bot:
            return

        match = self.URL_PATTERN.search(string=message.content)
        if not match:
            return

        url = match.group(0)
        if not self.url_is_expandable(url=url):
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
        self, *, message: Message, url: str, current_emoji: str, placeholder: ExpansionPlaceholder
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
        *,
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
            # Broad on purpose: hiding Discord's own preview is cosmetic and must not abort the
            # expansion. A persistent Forbidden means the guild lacks Manage Messages.
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
