"""The conversation `/ask` keeps for itself (`data/database/reply.db`).

A user-installed app receives no gateway message events and cannot read channel history, so a
`/ask` turn has no conversation unless one is kept here. One row per turn, keyed
`(channel_id, user_id)` rather than by channel alone: the bot only ever sees the halves of the
channel that were addressed to it, so a channel-wide key would feed one person the text of
another's turns that they never saw.

The table is new rather than an extra column anywhere, which is what makes it safe on a deployed
bot: `SqliteBootstrap.ensure_schema` is one `create_all`, which creates but never alters, so this
touches no existing table and the repo's lack of a migration mechanism does not bite.

Engine and bootstrap follow `cogs/research/database.py` exactly: a module-level `AsyncEngine`
singleton on the shared `reply.db` (a `cached_property` one would leak the pool and the dialect
cache, and tests monkeypatch it by that name) with this module owning its own `Base`, distinct
from research's `research` table and the memory inbox's `memory_job` in the same file. No money
columns, so no `StoredInteger`. Like both of them it avoids `from __future__ import annotations`:
SQLAlchemy resolves the `Mapped[datetime]` column at class-definition time.
"""

from datetime import datetime

from pydantic import Field, BaseModel
from sqlalchemy import Text, Index, Integer, DateTime, delete, select
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.sqlite_config import SqliteBootstrap

# How many of one conversation's turns survive a write. A write-side clamp on the file, like
# `RAW_FILE_MAX_BYTES`, rather than a bound on what a request carries — which is why it is not in
# `typings/context_budgets.py`, and why it is deliberately larger than anything a read can ask
# for: the pipeline reads `HISTORY_MESSAGE_LIMIT` (500) messages, i.e. half that many turns, so
# the message limit and then `HISTORY_CHAR_BUDGET` always bind first and this number never
# decides what the model sees. Its only job is to stop the table growing without limit for
# someone who talks to the bot every day for a year.
ASK_TURN_RETENTION = 500

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/reply.db")


class Base(DeclarativeBase):
    """Base class for the `/ask` conversation model (its own metadata, not research's)."""

    pass


class AskTurnRow(Base):
    """One `/ask` exchange: what was asked, and what the bot answered.

    Attributes:
        id: Surrogate key; the insertion order this conversation is replayed in.
        channel_id: The channel the command was invoked in.
        user_id: The person who invoked it, so one channel's conversations stay separate.
        message_id: The interaction's own snowflake, reused as the rebuilt message's id so its
            `created_at` is the real time the question was asked.
        question: The `question` option, verbatim.
        answer: The reply as it was finalized, with the memory note and the dropped-media hint
            already stripped but the usage footer still on the end. That footer is what a real
            bot message in a channel carries too, and `get_cleaned_content` takes it off at
            render time for the bot's own messages, so it is stored verbatim rather than trimmed
            here twice.
        created_at: Write timestamp.
    """

    __tablename__ = "ask_turn"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)

    __table_args__ = (Index("ix_ask_turn_conversation", "channel_id", "user_id", "id"),)


class AskTurn(BaseModel):
    """One stored exchange, read back for rebuilding the conversation."""

    message_id: int = Field(
        ..., description="The interaction snowflake the question was asked on."
    )
    question: str = Field(..., description="What the user asked.")
    answer: str = Field(..., description="What the bot replied.")


_database = SqliteBootstrap(metadata=Base.metadata)
_database.install_hooks(engine=_engine)


async def _ensure_schema() -> None:
    """Bootstraps the `ask_turn` table once per engine (loop-local-locked)."""
    await _database.ensure_schema(engine=_engine)


def open_session() -> AsyncSession:
    """Creates an async session bound to the current reply.db engine."""
    return _database.open_session(engine=_engine)


async def load_ask_turns(*, channel_id: int, user_id: int, limit: int) -> list[AskTurn]:
    """Reads the newest `limit` turns of one conversation, returned oldest-first.

    Oldest-first because that is the order `ReplyContextBuilder.fetch_history` returns channel
    history in, and everything downstream of it — the budget trim, the renders, the role
    pairing — reads that order.

    Args:
        channel_id: The channel the command was invoked in.
        user_id: The person whose conversation this is.
        limit: How many turns to read at most.

    Returns:
        The stored turns, oldest first.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            select(AskTurnRow)
            .where(AskTurnRow.channel_id == channel_id, AskTurnRow.user_id == user_id)
            .order_by(AskTurnRow.id.desc())
            .limit(limit)
        )
        rows = list(result.scalars())
    rows.reverse()
    return [
        AskTurn(message_id=row.message_id, question=row.question, answer=row.answer)
        for row in rows
    ]


async def record_ask_turn(
    *, channel_id: int, user_id: int, message_id: int, question: str, answer: str
) -> None:
    """Appends one finished exchange and prunes that conversation to its retention bound.

    Args:
        channel_id: The channel the command was invoked in.
        user_id: The person whose conversation this is.
        message_id: The interaction snowflake the question was asked on.
        question: The `question` option, verbatim.
        answer: The finalized reply text.
    """
    await _ensure_schema()
    async with open_session() as session, session.begin():
        session.add(
            AskTurnRow(
                channel_id=channel_id,
                user_id=user_id,
                message_id=message_id,
                question=question,
                answer=answer,
            )
        )
        await session.flush()
        # Prune inside the same transaction as the insert, so a conversation is never left over
        # its bound by a crash between the two.
        keep = await session.execute(
            select(AskTurnRow.id)
            .where(AskTurnRow.channel_id == channel_id, AskTurnRow.user_id == user_id)
            .order_by(AskTurnRow.id.desc())
            .limit(ASK_TURN_RETENTION)
        )
        oldest_kept = min(keep.scalars())
        await session.execute(
            delete(AskTurnRow).where(
                AskTurnRow.channel_id == channel_id,
                AskTurnRow.user_id == user_id,
                AskTurnRow.id < oldest_kept,
            )
        )
