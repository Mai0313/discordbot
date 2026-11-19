"""Persisted reply placeholders and restart resumption for link auto-expansions.

Claims a reply slot under the source link immediately, persists it in `pending_expansion`
(`data/database/reply.db`), and edits the finished card onto it. Undelivered placeholders are
deleted on failure; rows left by a restart are resumed on `on_ready` via
`resume_expansion_placeholders`.
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

from discordbot.utils.timezone import as_taipei as _as_taipei
from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.reactions import update_reaction
from discordbot.utils.sqlite_config import SqliteBootstrap
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.discord_errors import is_reply_target_gone

# Shared auto-expansion reaction outcomes (independent of reply-turn marks):
# WORKING: in progress; DONE: card delivered; RETRY_LATER: transient fetch/timeout failure;
# UNREADABLE: gone, private, or over Discord limits; FAILED: unexpected bot error.
EXPANSION_WORKING_EMOJI: Final[str] = "🔗"
EXPANSION_DONE_EMOJI: Final[str] = "<:greencheck:1517565102424068226>"
EXPANSION_RETRY_LATER_EMOJI: Final[str] = "⏱️"
EXPANSION_UNREADABLE_EMOJI: Final[str] = "⚠️"
EXPANSION_FAILED_EMOJI: Final[str] = "<:redcross:1517565100838355016>"

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/reply.db")


class Base(DeclarativeBase):
    """Base class for the pending-expansion model, with metadata of its own."""


class PendingExpansionRow(Base):
    """A posted placeholder not yet delivered onto or discarded.

    Attributes:
        message_id (Mapped[int]): Placeholder Discord message id (primary key).
        channel_id (Mapped[int]): Channel holding both the source and placeholder messages.
        source_message_id (Mapped[int]): Source Discord message carrying the link.
        source (Mapped[str]): Owning cog key matching `typings/emojis.py::LINK_SOURCE_EMOJIS`.
        url (Mapped[str]): Matched post URL persisted for restart resumption.
        created_at (Mapped[datetime]): Write timestamp in Asia/Taipei; sweeps oldest first.
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
    """Channel protocol supporting `fetch_message`, excluding non-messageable channels."""

    async def fetch_message(self, id: int, /) -> Message:  # noqa: A002 -- nextcord's own name
        """Returns one message read back off the channel.

        Args:
            id (int): Discord message id to fetch.
        """
        ...


class ExpansionRetry(Protocol):
    """A cog's own `_expand`, called with exactly what its listener passes it."""

    async def __call__(
        self, message: Message, url: str, current_emoji: str, placeholder: "ExpansionPlaceholder"
    ) -> None:
        """Runs the expansion and reports its own failures; never raises for a bad post.

        Args:
            message (Message): The message carrying the link.
            url (str): The post to expand.
            current_emoji (str): The mark already on the message, which each outcome replaces.
            placeholder (ExpansionPlaceholder): The reply slot to deliver onto.
        """
        ...


_database = SqliteBootstrap()


async def _record_pending(
    placeholder: Message, source_message: Message, source: str, url: str
) -> None:
    """Persists a posted placeholder so a restart can find it again, best-effort."""
    try:
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
    # Broad on purpose: best-effort bookkeeping must not abort an in-flight expansion.
    except Exception as error:
        logfire.warn(
            "Could not record a pending expansion",
            source=source,
            message_id=placeholder.id,
            error_type=type(error).__name__,
            _exc_info=error,
        )


async def _forget_pending(message_id: int) -> None:
    """Drops the row for a placeholder that is now settled, best effort."""
    try:
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


async def _load_pending(source: str, created_before: datetime) -> list[PendingExpansion]:
    """Reads one cog's expansions recorded before `created_before`, oldest first."""
    async with _database.open_session(engine=_engine) as session:
        rows = await session.scalars(
            select(PendingExpansionRow)
            .where(
                PendingExpansionRow.source == source,
                # SQLite keeps the Taipei wall clock without its offset.
                PendingExpansionRow.created_at < _as_taipei(dt=created_before),
            )
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
    """A posted placeholder and whether the expansion has landed on it yet."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: SkipValidation[Message] = Field(
        ..., description="The placeholder reply the finished expansion is edited into."
    )
    delivered: bool = Field(
        default=False,
        description=(
            "Whether the expansion landed, which is what stops `discard` from removing an "
            "expansion the reader can already see."
        ),
    )

    async def deliver(
        self, embeds: list[Embed], content: str | None = None, files: list[File] | None = None
    ) -> None:
        """Edits the finished expansion onto the placeholder and drops its pending row on success.

        Propagates edit errors without clearing the pending row so a restart sweep can retry.

        Args:
            embeds (list[Embed]): Finished expansion embeds.
            content (str | None): Text under the embeds, or None to clear the placeholder line.
            files (list[File] | None): Media attachments sent beside the embeds.
        """
        await self.message.edit(
            # Multipart edits in nextcord drop content=None instead of clearing existing text.
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
        """Deletes an undelivered placeholder and clears its pending row, swallowing delete errors.

        No-op if `delivered` is True; safe to call in a `finally` block.
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
    message: Message, text: str, source: str, url: str
) -> ExpansionPlaceholder | None:
    """Returns a posted placeholder replying to `message`, or None if refused or target is gone.

    Persists the placeholder in `pending_expansion` before returning.

    Args:
        message (Message): Source message carrying the link.
        text (str): Temporary placeholder line.
        source (str): Owning cog key from `typings/emojis.py::LINK_SOURCE_EMOJIS`.
        url (str): Post URL being expanded, stored for restart resumption.
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
        )
        return None
    except HTTPException as error:
        if not is_reply_target_gone(error=error):
            raise
        logfire.info(
            "The message to expand is gone", message_id=message.id, channel_id=message.channel.id
        )
        return None
    await _record_pending(placeholder=placeholder, source_message=message, source=source, url=url)
    return ExpansionPlaceholder(message=placeholder)


async def _resume_one(bot: commands.Bot, record: PendingExpansion, expand: ExpansionRetry) -> None:
    """Re-runs one interrupted expansion or cleans up its placeholder if channel/source is gone."""
    # Guild channels are cached by on_ready; fall back to fetch for DMs.
    try:
        channel = bot.get_channel(record.channel_id) or await bot.fetch_channel(record.channel_id)
    # Drop row only on NotFound/Forbidden; transient errors keep it for the next restart.
    except (NotFound, Forbidden):
        await _forget_pending(message_id=record.message_id)
        return
    if not isinstance(channel, MessageFetcher):
        await _forget_pending(message_id=record.message_id)
        return

    try:
        placeholder_message = await channel.fetch_message(record.message_id)
    # Placeholder already gone or channel unreadable; nothing left to clean up.
    except (NotFound, Forbidden):
        await _forget_pending(message_id=record.message_id)
        return
    placeholder = ExpansionPlaceholder(message=placeholder_message)

    try:
        source_message = await channel.fetch_message(record.source_message_id)
    # Source link deleted while down; discard the placeholder.
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
    # Broad on purpose: mark unexpected _expand failures on the source message before re-raising
    # so the working reaction does not remain stuck.
    except Exception:
        await update_reaction(
            message=source_message,
            bot_user=bot.user,
            emoji=EXPANSION_FAILED_EMOJI,
            previous=EXPANSION_WORKING_EMOJI,
        )
        raise
    finally:
        # Cleans up the placeholder on early returns; no-op once delivered.
        await placeholder.discard()


async def resume_expansion_placeholders(
    bot: commands.Bot, source: str, expand: ExpansionRetry, created_before: datetime
) -> None:
    """Resumes pending expansions for `source` recorded before `created_before`, sequentially.

    Args:
        bot (commands.Bot): Bot used to resolve channels and messages.
        source (str): Cog key matching `PendingExpansionRow.source`.
        expand (ExpansionRetry): Cog `_expand` callback.
        created_before (datetime): Process start timestamp; skips rows created by this process.
    """
    try:
        records = await _load_pending(source=source, created_before=created_before)
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
    # Sequential to avoid bursting a platform WAF on restart backlogs.
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
