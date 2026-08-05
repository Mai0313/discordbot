"""Local inbox for user reports (`data/database/feedback.db`).

One row per report. The row is written **before** the GitHub issue is opened, so a
report can never be lost by anything outside this process: the worst outcome is a row
whose `issue_number` is still empty, which the retry sweep picks up later.

What lives here is what GitHub does not carry (who filed it and from where) plus the
verbatim text and the write-up. Replies and status are deliberately absent: those live
on the issue, and a stored copy would only drift from it.

The engine is a module-level `AsyncEngine` singleton, like `services/economy/database.py`
and `cogs/research/database.py`: a per-instance `cached_property` engine would leak the
connection pool and dialect cache. Each call opens an `AsyncSession` bound to the current
`_engine`, so tests can monkeypatch `_engine` per test.

This module deliberately avoids `from __future__ import annotations`: SQLAlchemy resolves
the `Mapped[datetime]` column annotations at class-definition time, and postponed
evaluation breaks that.
"""

from typing import Any
from datetime import datetime, timedelta

from pydantic import Field, BaseModel
from sqlalchemy import String, Integer, DateTime, func, event, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.asyncio_locks import LoopLocalLock
from discordbot.utils.sqlite_config import ensure_sqlite_hooks, configure_sqlite_connection

# A Discord modal caps a paragraph input well below this; the column is sized for the
# cap plus room, so a longer client never truncates a report silently.
RAW_TEXT_MAX_CHARS = 4096

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/feedback.db")


def _configure_sqlite_connection(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Applies the project's standard PRAGMA setup to a new feedback.db connection."""
    configure_sqlite_connection(dbapi_connection=dbapi_connection)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Configures a newly opened SQLite connection."""
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _configure_sqlite_on_checkout(
    dbapi_connection: object, _connection_record: object, _connection_proxy: object
) -> None:
    """Configures pooled connections from test-swapped engines."""
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


class Base(DeclarativeBase):
    """Base class for feedback.db ORM models."""

    pass


class FeedbackTicketRow(Base):
    """One report filed through `/feedback`.

    Attributes:
        ticket_id: Local primary key. A report has an identity before its issue exists.
        issue_number: The issue this became; `None` until it is opened, which is what
            the retry sweep selects on.
        user_id: Discord user ID of the reporter; how the panel finds someone's reports.
        user_name: Discord username at filing time.
        display_name: Display name at filing time; names change, the id does not.
        guild_id: Guild the report came from, or `None` in a DM.
        guild_name: Guild name at filing time, for reading the row later without a lookup.
        channel_id: Channel the report came from.
        locale: The reporter's Discord locale, so a reply can be written in their language.
        raw_text: What the reporter typed, unchanged.
        label: One-line Traditional Chinese summary from the write-up; empty until it lands.
        category: Write-up classification (`bug` / `feature` / `question`).
        draft_title: English issue title from the write-up.
        draft_body: English issue body from the write-up.
        relayed_replies: How many of the issue's comments this bot posted for the
            reporter. Counting our own writes is what lets the panel tell "nobody has
            answered yet" from "the developer replied" off a single issue read.
        created_at: First-write timestamp.
        updated_at: Latest-write timestamp.
    """

    __tablename__ = "feedback_ticket"

    ticket_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    issue_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    display_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    guild_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    guild_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    channel_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    locale: Mapped[str] = mapped_column(String(length=16), default="", nullable=False)
    raw_text: Mapped[str] = mapped_column(
        String(length=RAW_TEXT_MAX_CHARS), default="", nullable=False
    )
    label: Mapped[str] = mapped_column(String(length=256), default="", nullable=False)
    category: Mapped[str] = mapped_column(String(length=32), default="", nullable=False)
    draft_title: Mapped[str] = mapped_column(String(length=256), default="", nullable=False)
    draft_body: Mapped[str] = mapped_column(String(length=32768), default="", nullable=False)
    relayed_replies: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, onupdate=_database_now
    )


class FeedbackTicket(BaseModel):
    """A report read back out of feedback.db."""

    ticket_id: int = Field(..., description="Local primary key of the report.")
    issue_number: int | None = Field(..., description="The issue number, or None if not opened.")
    user_id: int = Field(..., description="Discord user ID of the reporter.")
    user_name: str = Field(..., description="Discord username at filing time.")
    display_name: str = Field(..., description="Display name at filing time.")
    guild_id: int | None = Field(..., description="Guild the report came from, None in a DM.")
    guild_name: str = Field(..., description="Guild name at filing time.")
    channel_id: int | None = Field(..., description="Channel the report came from.")
    locale: str = Field(..., description="The reporter's Discord locale.")
    raw_text: str = Field(..., description="What the reporter typed, unchanged.")
    label: str = Field(..., description="One-line summary from the write-up, or empty.")
    category: str = Field(..., description="Write-up classification, or empty.")
    draft_title: str = Field(..., description="English issue title from the write-up.")
    draft_body: str = Field(..., description="English issue body from the write-up.")
    relayed_replies: int = Field(
        ..., description="Comments this bot posted on the reporter's behalf."
    )
    created_at: datetime = Field(..., description="When the report was filed.")

    @property
    def summary_line(self) -> str:
        """The one line naming this report, falling back to the reporter's own words.

        The write-up runs in the background, so a report is listed before its label
        exists; showing the first line of what they wrote is both immediate and the
        thing the reporter recognizes.
        """
        if self.label:
            return self.label
        first_line = next(
            (line.strip() for line in self.raw_text.splitlines() if line.strip()), ""
        )
        return first_line or "（沒有內容）"


_schema_ready_for: AsyncEngine | None = None
_schema_lock = LoopLocalLock()


async def _ensure_schema() -> None:
    """Bootstraps the `feedback_ticket` table once per engine (loop-local-locked)."""
    global _schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    if _schema_ready_for is _engine:
        return
    async with _schema_lock.get():
        if _schema_ready_for is _engine:
            return
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _schema_ready_for = _engine


def open_session() -> AsyncSession:
    """Creates an async session bound to the current feedback.db engine."""
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    return AsyncSession(bind=_engine, expire_on_commit=False)


def _row_to_model(row: FeedbackTicketRow) -> FeedbackTicket:
    """Maps an ORM row to its pydantic snapshot."""
    return FeedbackTicket(
        ticket_id=row.ticket_id,
        issue_number=row.issue_number,
        user_id=row.user_id,
        user_name=row.user_name,
        display_name=row.display_name,
        guild_id=row.guild_id,
        guild_name=row.guild_name,
        channel_id=row.channel_id,
        locale=row.locale,
        raw_text=row.raw_text,
        label=row.label,
        category=row.category,
        draft_title=row.draft_title,
        draft_body=row.draft_body,
        relayed_replies=row.relayed_replies,
        created_at=row.created_at,
    )


async def create_ticket(  # noqa: PLR0913 -- one row's columns are all per-call inputs
    *,
    user_id: int,
    user_name: str,
    display_name: str,
    guild_id: int | None,
    guild_name: str,
    channel_id: int | None,
    locale: str,
    raw_text: str,
) -> FeedbackTicket:
    """Stores a freshly submitted report and returns it.

    This is the first durable write of the submit path, before anything leaves the
    process, so a GitHub outage can only delay the issue rather than lose the report.
    """
    await _ensure_schema()
    async with open_session() as session:
        row = FeedbackTicketRow(
            user_id=user_id,
            user_name=user_name,
            display_name=display_name,
            guild_id=guild_id,
            guild_name=guild_name,
            channel_id=channel_id,
            locale=locale,
            raw_text=raw_text[:RAW_TEXT_MAX_CHARS],
        )
        session.add(row)
        await session.commit()
        return _row_to_model(row=row)


async def attach_issue_number(*, ticket_id: int, issue_number: int) -> None:
    """Records the issue a report became, taking it out of the retry sweep."""
    await _ensure_schema()
    async with open_session() as session:
        await session.execute(
            statement=update(FeedbackTicketRow)
            .where(FeedbackTicketRow.ticket_id == ticket_id)
            .values(issue_number=issue_number, updated_at=_database_now())
        )
        await session.commit()


async def store_write_up(
    *, ticket_id: int, label: str, category: str, draft_title: str, draft_body: str
) -> None:
    """Stores the background write-up next to the original text.

    Kept locally as well as on the issue so a failed edit knows what to retry, and so a
    later pass over the store can read the drafts without calling GitHub.
    """
    await _ensure_schema()
    async with open_session() as session:
        await session.execute(
            statement=update(FeedbackTicketRow)
            .where(FeedbackTicketRow.ticket_id == ticket_id)
            .values(
                label=label[:256],
                category=category[:32],
                draft_title=draft_title[:256],
                draft_body=draft_body[:32768],
                updated_at=_database_now(),
            )
        )
        await session.commit()


async def count_relayed_reply(*, ticket_id: int) -> None:
    """Records that one more comment was posted for the reporter on their own report.

    The panel subtracts this from the issue's comment count, so a reporter adding detail
    never reads back to them as the developer having answered.
    """
    await _ensure_schema()
    async with open_session() as session:
        await session.execute(
            statement=update(FeedbackTicketRow)
            .where(FeedbackTicketRow.ticket_id == ticket_id)
            .values(
                relayed_replies=FeedbackTicketRow.relayed_replies + 1, updated_at=_database_now()
            )
        )
        await session.commit()


async def list_user_tickets(*, user_id: int, limit: int) -> list[FeedbackTicket]:
    """Returns one person's reports, newest first.

    The panel answers "whose report is this" from here rather than from GitHub search,
    which is only eventually consistent: a report opened seconds ago has to be listed.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(FeedbackTicketRow)
            .where(FeedbackTicketRow.user_id == user_id)
            .order_by(FeedbackTicketRow.ticket_id.desc())
            .limit(limit)
        )
        return [_row_to_model(row=row) for row in result.scalars().all()]


async def get_ticket(*, ticket_id: int) -> FeedbackTicket | None:
    """Returns one report by its local id, or None when it is gone."""
    await _ensure_schema()
    async with open_session() as session:
        row = await session.get(FeedbackTicketRow, ticket_id)
        return _row_to_model(row=row) if row is not None else None


async def seconds_since_last_ticket(*, user_id: int) -> float | None:
    """Seconds since this person's most recent submission, or None if they have none."""
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(func.max(FeedbackTicketRow.created_at)).where(
                FeedbackTicketRow.user_id == user_id
            )
        )
        latest = result.scalar_one_or_none()
    if latest is None:
        return None
    reference = _database_now()
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=reference.tzinfo)
    return max((reference - latest) / timedelta(seconds=1), 0.0)


async def tickets_awaiting_issue(*, limit: int) -> list[FeedbackTicket]:
    """Returns reports whose issue was never opened, oldest first.

    These are the rows the submit path could not hand to GitHub. They are the whole
    reason the local write comes first, so the sweep exists to finish that hand-off.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(FeedbackTicketRow)
            .where(FeedbackTicketRow.issue_number.is_(None))
            .order_by(FeedbackTicketRow.ticket_id.asc())
            .limit(limit)
        )
        return [_row_to_model(row=row) for row in result.scalars().all()]
