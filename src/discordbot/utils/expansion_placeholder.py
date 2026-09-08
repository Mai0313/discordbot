"""The reply slot an expansion cog claims before it has anything to put in it.

An expansion takes seconds: `parse_threads` walks a whole conversation, `parse_douyin`
downloads the clip. In a busy channel that is long enough for other people to talk, and the
finished card then lands several messages below the link it belongs to, which is worst for
exactly the reader it was posted for — someone tagged in it has to work out which link the
card came from. So the cog replies with one line the moment it starts and edits that same
message into the finished expansion: the card sits under the link whatever happened in
between.

A failure leaves NOTHING behind. `discard` removes the placeholder and the reaction on the
source message is the whole report, which is what every expansion cog already does.

The one failure that rule could not cover is a restart, which kills the expansion mid-read
with the placeholder already posted and nothing left running to withdraw it. So a posted
placeholder is a row in `pending_expansion` (`data/database/reply.db`) from the moment it
lands until it is either delivered onto or discarded, and `resume_expansion_placeholders` is
the entry point a cog calls on `on_ready` to run the interrupted expansion again. What it
runs is the cog's own `_expand`, unchanged and with the same arguments the listener passes,
so a resumed expansion succeeds or fails exactly as a fresh one does — and a failed one ends
in the same `discard`, which is what turns the stale placeholder back into nothing.

The table is new rather than a column on an existing one, which is what makes it safe on a
deployed bot: `SqliteBootstrap.ensure_schema` is one `create_all`, which creates but never
alters, so the repo's lack of a migration mechanism does not bite. Engine and bootstrap
follow `cogs/gen_reply/ask_store.py`: a module-level `AsyncEngine` singleton on the shared
`reply.db` with its own `Base`, distinct from the `research`, `ask_turn` and `memory_job`
tables in the same file.
"""

from typing import Final, Protocol, runtime_checkable
from datetime import datetime

import logfire
from nextcord import File, Embed, Message, NotFound, Forbidden, HTTPException, AllowedMentions
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from sqlalchemy import Text, Index, Integer, DateTime, delete, select
from nextcord.ext import commands
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.reactions import update_reaction
from discordbot.utils.link_errors import LinkReadError, LinkRetryableError, LinkUnavailableError
from discordbot.utils.sqlite_config import SqliteBootstrap
from discordbot.utils.discord_embeds import embed_spacer_payload

# What Discord answers when the message being replied to no longer exists. It is a generic
# invalid-form-body code, so it means this only on a send that carries a message reference.
_UNSENDABLE_REPLY = 50035

# The whole reaction vocabulary an auto-expansion answers with, shared so one symbol means one
# thing whichever platform was linked. Every cog reports through exactly these four and adds
# nothing to the channel, so the reaction IS the report and a reader who learns it once reads
# every expansion. What separates them is what the reader should do next:
#
#   WORKING     the expansion is under way. Also what the restart sweep hands `_expand`, since
#               it cannot ask the interrupted listener what it had scheduled.
#   DONE        the card is on screen.
#   RETRY_LATER the platform refused the request or took too long. The post is fine and the
#               same link works later, which is why this is never the unreadable mark: telling
#               someone their working link is dead is the worst answer this feature can give.
#   UNREADABLE  the platform served an answer and there is nothing showable in it — deleted,
#               private, or past a Discord limit no retry can get under.
#   FAILED      the bot broke. Nothing about the post explains it, and it is worth a log.
#
# `gen_reply` spells three of these inline for its own reply turn; that is a different surface
# with a different lifecycle, and this vocabulary is the auto-expansion contract alone.
EXPANSION_WORKING_EMOJI: Final[str] = "🔗"
EXPANSION_DONE_EMOJI: Final[str] = "<:greencheck:1517565102424068226>"
EXPANSION_RETRY_LATER_EMOJI: Final[str] = "⏱️"
EXPANSION_UNREADABLE_EMOJI: Final[str] = "⚠️"
EXPANSION_FAILED_EMOJI: Final[str] = "<:redcross:1517565100838355016>"

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/reply.db")


class Base(DeclarativeBase):
    """Base class for the pending-expansion model (its own metadata, not research's)."""


class PendingExpansionRow(Base):
    """A placeholder that is posted but not yet delivered onto or discarded.

    Attributes:
        message_id: The placeholder's own Discord id, which is also the row key: one
            placeholder is one expansion, and the cog reaches the row through it.
        channel_id: The channel both messages live in, so the resume can fetch them.
        source_message_id: The message carrying the link, which `_expand` reacts on.
        source: Which cog owns the row, keyed as in `typings/emojis.py::LINK_SOURCE_EMOJIS`,
            so each cog sweeps its own rows and never resumes another's.
        url: The URL the listener matched, stored rather than re-derived so an edited or
            unreadable source message cannot change what the resumed expansion reads.
        created_at: Write timestamp; the sweep runs oldest first.
    """

    __tablename__ = "pending_expansion"

    message_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source_message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)

    __table_args__ = (Index("ix_pending_expansion_source", "source", "created_at"),)


class PendingExpansion(BaseModel):
    """One interrupted expansion, read back for the restart sweep."""

    message_id: int = Field(..., description="The placeholder message to deliver onto or remove.")
    channel_id: int = Field(..., description="The channel holding both messages.")
    source_message_id: int = Field(..., description="The message carrying the link.")
    url: str = Field(..., description="The URL the interrupted expansion was reading.")


@runtime_checkable
class MessageFetcher(Protocol):
    """A channel the sweep can read both of its messages back out of.

    Named for the one capability rather than typed `nextcord.abc.Messageable`, which
    `bot.get_channel` can also answer a category or forum for: those carry no messages, so
    what the sweep needs is exactly this method. `runtime_checkable` is what makes the check
    an honest `isinstance` — `Messageable` is a plain ABC, so nothing that did not inherit
    from it passes, a test double included.
    """

    async def fetch_message(self, id: int, /) -> Message:  # noqa: A002 -- nextcord's own name
        """Reads one message back off the channel."""
        ...


class ExpansionRetry(Protocol):
    """A cog's own `_expand`, called with exactly what its listener passes it."""

    async def __call__(
        self,
        *,
        message: Message,
        url: str,
        current_emoji: str,
        placeholder: "ExpansionPlaceholder",
    ) -> None:
        """Runs the expansion and reports its own failures; never raises for a bad post."""
        ...


_database = SqliteBootstrap(metadata=Base.metadata)
_database.install_hooks(engine=_engine)


async def _record_pending(
    *, placeholder: Message, source_message: Message, source: str, url: str
) -> None:
    """Persists a posted placeholder so a restart can find it again.

    Best effort: the durable half is what makes a restart recoverable, and losing it costs
    exactly the stale placeholder this feature exists to remove — which is where the whole
    expansion stood before it, so a failing write must never take the expansion with it.
    """
    try:
        await _database.ensure_schema(engine=_engine)
        async with _database.open_session(engine=_engine) as session, session.begin():
            session.add(
                PendingExpansionRow(
                    message_id=placeholder.id,
                    channel_id=placeholder.channel.id,
                    source_message_id=source_message.id,
                    source=source,
                    url=url,
                )
            )
    # Broad on purpose: see above — the expansion is already under way and nothing here can
    # improve its outcome.
    except Exception as error:
        logfire.warn(
            "Could not record a pending expansion",
            source=source,
            message_id=placeholder.id,
            error_type=type(error).__name__,
            _exc_info=error,
        )


async def _forget_pending(*, message_id: int) -> None:
    """Drops the row for a placeholder that is now settled, best effort."""
    try:
        await _database.ensure_schema(engine=_engine)
        async with _database.open_session(engine=_engine) as session, session.begin():
            await session.execute(
                delete(PendingExpansionRow).where(PendingExpansionRow.message_id == message_id)
            )
    # Broad on purpose: a row that outlives its placeholder costs one resume that finds the
    # message already gone and drops the row then, which is a path this already has.
    except Exception as error:
        logfire.warn(
            "Could not clear a pending expansion",
            message_id=message_id,
            error_type=type(error).__name__,
            _exc_info=error,
        )


async def _load_pending(*, source: str) -> list[PendingExpansion]:
    """Reads one cog's interrupted expansions, oldest first."""
    await _database.ensure_schema(engine=_engine)
    async with _database.open_session(engine=_engine) as session:
        rows = await session.scalars(
            select(PendingExpansionRow)
            .where(PendingExpansionRow.source == source)
            .order_by(PendingExpansionRow.created_at, PendingExpansionRow.message_id)
        )
        return [
            PendingExpansion(
                message_id=row.message_id,
                channel_id=row.channel_id,
                source_message_id=row.source_message_id,
                url=row.url,
            )
            for row in rows
        ]


class ExpansionPlaceholder(BaseModel):
    """A posted placeholder and whether the expansion has landed on it yet.

    Attributes:
        message: The placeholder itself, the message every later edit lands on.
        delivered: Whether `deliver` succeeded, which is what stops `discard` from removing
            an expansion the reader can already see.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message] = Field(
        ..., description="The placeholder reply the finished expansion is edited into."
    )
    delivered: bool = Field(
        default=False, description="Whether the expansion landed, making `discard` a no-op."
    )

    async def deliver(
        self, *, content: str | None = None, embeds: list[Embed], files: list[File] | None = None
    ) -> None:
        """Edits the finished expansion onto the placeholder.

        Clearing the placeholder line takes an explicit empty string, never None: an
        expansion brings files (its media, or the embed spacer), which sends the edit as
        multipart, and `http.py::get_message_payload` drops a None content out of that body
        altogether instead of clearing it. It clears on the JSON path, which is exactly what
        makes the difference easy to miss, and `streaming.py::land_failure` carries the same
        note for the same reason.

        The spacer rides as an edit so its `attachments` key drops whatever the placeholder
        held, which is also what lets `files` be the whole of the new attachment list.

        Whatever the edit raises travels out, so the cog's own failure path reports it. The
        row is dropped only once the edit has landed, so an edit that raises leaves the
        restart sweep something to find.

        Args:
            content: The text under the expansion, or None for an expansion that is embeds
                and attachments alone.
            embeds: The finished expansion.
            files: Media attaching natively beside the embeds.
        """
        await self.message.edit(
            content=content or "",
            embeds=embeds,
            allowed_mentions=AllowedMentions.none(),
            **embed_spacer_payload(
                embeds=embeds, is_edit=True, target=self.message, extra_files=files
            ),
        )
        self.delivered = True
        await _forget_pending(message_id=self.message.id)

    async def discard(self) -> None:
        """Removes an undelivered placeholder, leaving the reaction as the whole report.

        Safe in a `finally`: it swallows its own failure rather than replacing whatever sent
        the expansion down the failure path, and a placeholder already deleted by hand is the
        outcome this wanted anyway.

        The row goes whether or not the delete did, because a resume would only find the same
        undeliverable placeholder and try the same delete again.
        """
        if self.delivered:
            return
        try:
            await self.message.delete()
        # Broad on purpose: the expansion has already failed and the reaction says so, so
        # nothing here can improve the outcome or is worth reaching the listener's handler.
        except Exception as error:
            logfire.debug(
                "Could not remove an expansion placeholder",
                message_id=self.message.id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
        await _forget_pending(message_id=self.message.id)


async def send_expansion_placeholder(
    *, message: Message, text: str, source: str, url: str
) -> ExpansionPlaceholder | None:
    """Claims the reply slot under `message` with a line saying the expansion is coming.

    A refusal here is the whole expansion's answer as well: a channel that will not take this
    one line will not take the card either, and learning it now costs no fetch. Both refusals
    are ordinary rather than defects, which is why they are classified here instead of
    reaching the listener's last-resort handler, where a misconfigured channel would log an
    error per pasted link.

    Args:
        message: The message carrying the link, which the placeholder replies to.
        text: The line to show until the expansion replaces it.
        source: The owning cog's key, as in `typings/emojis.py::LINK_SOURCE_EMOJIS`.
        url: The URL being expanded, kept so a restart can run the same expansion again.

    Returns:
        The placeholder to deliver onto or discard, or None when the channel refused it and
        the caller should mark the expansion failed without reading anything.
    """
    try:
        placeholder = await message.reply(
            content=text, mention_author=False, allowed_mentions=AllowedMentions.none()
        )
    except Forbidden as error:
        logfire.warn(
            "Missing permission to post an expansion placeholder",
            message_id=message.id,
            channel_id=message.channel.id,
            error_type=type(error).__name__,
            _exc_info=error,
        )
        return None
    except HTTPException as error:
        # A reply to a message that is already gone comes back as 50035, not only as NotFound.
        if not isinstance(error, NotFound) and error.code != _UNSENDABLE_REPLY:
            raise
        logfire.info(
            "The message to expand is gone", message_id=message.id, channel_id=message.channel.id
        )
        return None
    await _record_pending(placeholder=placeholder, source_message=message, source=source, url=url)
    return ExpansionPlaceholder(message=placeholder)


def expansion_failure_emoji(*, error: Exception) -> str:
    """Picks the mark a failed read earns, the same way for every platform.

    The three outcomes are the shared vocabulary's, read off the exception's CLASS rather than
    its message: a platform refusing the request or a transport that never answered is the
    retryable mark, since the link is fine and works later; anything else `LinkReadError`
    covers means the platform answered and there is no post in it; and an error from outside
    that tree is the bot's own. `utils/link_errors.py` owns which fetch failures are classified
    at all and why a 403 deliberately is not.

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
    *, error: Exception, platform: str, url: str, message_id: int
) -> None:
    """Logs a failed read at the severity `.github/CONTRIBUTING.md#logging` gives its outcome.

    The ladder is keyed on how tolerable the failure is, not on how deep it happened, so the
    three levels line up with the marks `expansion_failure_emoji` picks rather than with any
    one platform's habits: a post the platform says is gone is a routine user-driven outcome
    and its own example of `info`, a read the platform explains any other way is degraded but
    handled, and an error from outside that tree broke a user-visible deliverable.

    The `info` branch deliberately carries no exception and does carry `reason`: a traceback
    for a deleted post is noise, while the platform's own words for WHY it refused exist in no
    other line. That split was Douyin's alone before this; the other three logged a deleted
    post at `warn` with a traceback, which is what makes a real regression unfindable.

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
    *, error: Exception, platform: str, url: str, message_id: int, channel_id: int
) -> None:
    """Logs a failed delivery at the severity it deserves, the same way for every platform.

    Only the placeholder's own disappearance is routine. The 50035 an unsendable reply used to
    raise is not: on an edit that code is a rejected body, which is a defect rather than a
    message that went away.

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
        logfire.warn(
            f"Missing permission to post the {platform} expansion",
            url=url,
            message_id=message_id,
            channel_id=channel_id,
            error_type=type(error).__name__,
            _exc_info=error,
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


async def _resume_one(
    *, bot: commands.Bot, record: PendingExpansion, expand: ExpansionRetry
) -> None:
    """Re-runs one interrupted expansion, or removes what it left behind.

    Both messages are fetched rather than taken as partials: `embed_spacer_payload` reads the
    placeholder's own attachments to retain an already-uploaded spacer, and `_expand` reacts
    on the source and suppresses its preview.
    """
    # A guild channel is in the cache by `on_ready`; a DM channel need not be, so the fetch
    # is the fallback rather than the first move.
    try:
        channel = bot.get_channel(record.channel_id) or await bot.fetch_channel(record.channel_id)
    # The channel is gone, or the bot was removed from the guild while it was down: the
    # placeholder went with it, so there is nothing to withdraw and nothing to retry. Narrow
    # on purpose — anything else keeps the row for the next restart rather than dropping a
    # placeholder that is still sitting in a channel this could not reach right now.
    except (NotFound, Forbidden):
        await _forget_pending(message_id=record.message_id)
        return
    if not isinstance(channel, MessageFetcher):
        await _forget_pending(message_id=record.message_id)
        return

    try:
        placeholder_message = await channel.fetch_message(record.message_id)
    # Someone removed the stale line by hand, or the channel is no longer readable. Either
    # way there is nothing left to deliver onto and nothing to clean up.
    except (NotFound, Forbidden):
        await _forget_pending(message_id=record.message_id)
        return
    placeholder = ExpansionPlaceholder(message=placeholder_message)

    try:
        source_message = await channel.fetch_message(record.source_message_id)
    # The link was withdrawn while the bot was down, which is exactly the case a live
    # expansion answers by posting nothing.
    except (NotFound, Forbidden):
        await placeholder.discard()
        return

    try:
        await expand(
            message=source_message,
            url=record.url,
            current_emoji=EXPANSION_WORKING_EMOJI,
            placeholder=placeholder,
        )
    # `_expand` reports its own failures and returns, so this is the unexpected one — and on
    # the listener path the cog's outer handler paints the cross for exactly that. This sweep
    # IS that handler's counterpart, and without the mark the source keeps the working ring
    # with no outcome ever painted: the never-resolving state the sweep exists to clear,
    # moved off the placeholder and onto the reaction. Re-raised so the caller still logs it.
    except Exception:
        await update_reaction(
            message=source_message,
            bot_user=bot.user,
            emoji=EXPANSION_FAILED_EMOJI,
            previous=EXPANSION_WORKING_EMOJI,
        )
        raise
    finally:
        # The same line the listener runs, covering every failure `_expand` returns on, and a
        # no-op once delivered.
        await placeholder.discard()


async def resume_expansion_placeholders(
    *, bot: commands.Bot, source: str, expand: ExpansionRetry
) -> None:
    """Runs again every expansion of `source` that a restart interrupted.

    The entry point is the whole of what is new: what runs is the cog's own `_expand`, with
    the arguments its listener passes, so a resumed expansion delivers, refuses or fails
    exactly as a fresh one does. One row at a time, because a restart's backlog would
    otherwise hit a platform with every stale link at once — which for Douyin is the request
    volume its WAF bans on.

    Args:
        bot: The bot, used to resolve the channel both messages live in.
        source: The cog's key, so it sweeps its own rows and never another cog's.
        expand: The cog's `_expand`.
    """
    try:
        records = await _load_pending(source=source)
    # Broad on purpose: an unreadable table must not stop the cog from serving new links.
    except Exception as error:
        logfire.warn(
            "Could not read the interrupted expansions",
            source=source,
            error_type=type(error).__name__,
            _exc_info=error,
        )
        return

    if not records:
        return
    logfire.info("Resuming interrupted expansions", source=source, pending=len(records))
    for record in records:
        try:
            await _resume_one(bot=bot, record=record, expand=expand)
        # Broad on purpose: one unresolvable channel or message must not strand the rest of
        # the backlog, and the row survives for the next restart to try again.
        except Exception as error:
            logfire.warn(
                "Could not resume an interrupted expansion",
                source=source,
                message_id=record.message_id,
                error_type=type(error).__name__,
                _exc_info=error,
            )
