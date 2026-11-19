"""Per-user `/ask` conversation history stored in `data/database/reply.db`.

Rows are keyed by `(channel_id, user_id)` so one user's `/ask` turns never leak into another's in
the same channel. Owns its own `Base` metadata on `reply.db` and omits
`from __future__ import annotations` so SQLAlchemy resolves `Mapped[datetime]` at class-build time
without ruff's TC rules hiding `datetime` under `TYPE_CHECKING`.
"""

from datetime import datetime
from contextlib import AbstractAsyncContextManager

from pydantic import Field, BaseModel
from sqlalchemy import Text, Index, Integer, DateTime, delete, select
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.sqlite_config import SqliteBootstrap

# Write-side table retention cap (not a request budget in `typings/context_budgets.py`);
# deliberately larger than `HISTORY_MESSAGE_LIMIT` so read-side budgets always bind first.
ASK_TURN_RETENTION = 500

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/reply.db")


class Base(DeclarativeBase):
    """Base class for the `/ask` conversation model (its own metadata, not research's)."""


class AskTurnRow(Base):
    """One `/ask` exchange: what was asked, and what the bot answered.

    Attributes:
        id (Mapped[int]): Surrogate key; the insertion order this conversation is replayed in.
        channel_id (Mapped[int]): The channel the command was invoked in.
        user_id (Mapped[int]): The person who invoked it, keeping conversations in one channel separate.
        message_id (Mapped[int]): The interaction's own snowflake, reused as the rebuilt message's id
            so its `created_at` reflects when the question was asked.
        question (Mapped[str]): The `question` option, verbatim.
        answer (Mapped[str]): The finalized reply with the memory note and dropped-media hint
            stripped and the usage footer intact (`get_cleaned_content` strips the footer at
            render time).
        created_at (Mapped[datetime]): Write timestamp.
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


_database = SqliteBootstrap()


def open_session() -> AbstractAsyncContextManager[AsyncSession]:
    """Returns an async session context manager on `reply.db`."""
    return _database.open_session(engine=_engine)


async def load_ask_turns(channel_id: int, user_id: int, limit: int) -> list[AskTurn]:
    """Returns the newest `limit` turns of one conversation, ordered oldest-first.

    Args:
        channel_id (int): The channel the command was invoked in.
        user_id (int): The person whose conversation this is.
        limit (int): Maximum number of turns to read.
    """
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
    channel_id: int, user_id: int, message_id: int, question: str, answer: str
) -> None:
    """Appends one finished exchange and prunes that conversation to `ASK_TURN_RETENTION`.

    Args:
        channel_id (int): The channel the command was invoked in.
        user_id (int): The person whose conversation this is.
        message_id (int): The interaction snowflake the question was asked on.
        question (str): The `question` option, verbatim.
        answer (str): The finalized reply text.
    """
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
