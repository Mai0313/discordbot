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

from typing import Any, cast
from datetime import datetime, timedelta

from pydantic import Field, BaseModel
from sqlalchemy import String, Integer, DateTime, CursorResult, func, delete, select, update
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.sqlite_config import SqliteBootstrap

# A Discord modal caps a paragraph input well below this; the column is sized for the
# cap plus room, so a longer client never truncates a report silently.
RAW_TEXT_MAX_CHARS = 4096

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/feedback.db")


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


class FeedbackCloseNoticeRow(Base):
    """One report's progress towards telling its reporter that it was closed.

    A table rather than two more columns on `feedback_ticket`, because `_ensure_schema`
    builds the schema with `create_all`: it is `checkfirst`, so it creates a table that is
    missing and never alters one that is already there. A new column would simply be
    absent from a deployed database and every read of it would raise `no such column`,
    while a new table costs no migration at all.

    A row means the close has been seen. `notified_at` means the report is finished with
    and is never looked at again, which is what keeps a report from being announced twice
    over.

    It is at-least-once, not exactly-once, and deliberately so: the message is sent before
    this row records that it was, so a process that dies in between sends a second copy on
    the next pass. The other order trades that for losing the message outright, and a
    duplicate the reporter can see beats a silence nobody can.

    What is deliberately NOT here is why the issue was closed. The delivery pass reads
    the issue again rather than trusting what discovery saw, so a reason stored here would
    have no reader, and `create_all` never alters a table: a column that ships is a column
    the deployed database keeps forever.

    Attributes:
        ticket_id: The report this is about; one notice per report, so it is the key.
        observed_at: When the close was first seen. The delivery pass waits on this, and
            past `CLOSE_NOTICE_STALLED_AFTER_SECONDS` it is also what says a notice has
            been failing to go out for far too long.
        notified_at: When the report was finished with, whether that meant sending a
            message or deciding not to send one. `None` while it is still owed.
    """

    __tablename__ = "feedback_close_notice"

    ticket_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)
    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


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


_database = SqliteBootstrap(metadata=Base.metadata)
_database.install_hooks(engine=_engine)


async def _ensure_schema() -> None:
    """Bootstraps the `feedback_ticket` table once per engine (loop-local-locked)."""
    await _database.ensure_schema(engine=_engine)


def open_session() -> AsyncSession:
    """Creates an async session bound to the current feedback.db engine."""
    return _database.open_session(engine=_engine)


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


async def attach_issue_number(*, ticket_id: int, issue_number: int) -> bool:
    """Records the issue a report became, and says whether this call is the one that did.

    Conditional on the column still being empty, because two writers can reach the same
    row: the submit path and the retry sweep both select on `issue_number IS NULL`, and
    the window between opening an issue and recording it is a real one. Last-write-wins
    would quietly replace the number the reporter was already given.

    Returns:
        True when this call recorded the number, False when the row already had one.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=update(FeedbackTicketRow)
            .where(
                FeedbackTicketRow.ticket_id == ticket_id, FeedbackTicketRow.issue_number.is_(None)
            )
            .values(issue_number=issue_number, updated_at=_database_now())
        )
        await session.commit()
        # An UPDATE always yields a CursorResult; the async wrapper is typed as the wider
        # `Result`, which does not carry the row count this call is asking for.
        return cast("CursorResult[Any]", result).rowcount > 0


async def store_write_up(
    *, ticket_id: int, label: str, category: str, draft_title: str, draft_body: str
) -> None:
    """Stores the background write-up next to the original text.

    A report that already has an issue has that issue rewritten first, so the panel's
    summary line can never describe a report in words the issue does not use. A report
    still waiting for credentials has no issue to rewrite, and the draft stored here is
    what its issue is later opened from; the retry sweep reads it back to tell that case
    apart and skip a rewrite it no longer needs.
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


async def count_user_tickets(*, user_id: int) -> int:
    """How many reports one person has filed in total.

    The panel lists only the newest few, and it says how many it is not showing rather
    than letting an older report look like it was never filed.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(func.count())
            .select_from(FeedbackTicketRow)
            .where(FeedbackTicketRow.user_id == user_id)
        )
        return int(result.scalar_one())


async def get_ticket(*, ticket_id: int) -> FeedbackTicket | None:
    """Returns one report by its local id, or None when it is gone."""
    await _ensure_schema()
    async with open_session() as session:
        row = await session.get(entity=FeedbackTicketRow, ident=ticket_id)
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


class PendingCloseNotice(BaseModel):
    """A report whose close was seen and whose reporter has not been told yet."""

    ticket: FeedbackTicket = Field(..., description="The report the close belongs to.")
    observed_at: datetime = Field(..., description="When the close was first seen.")


async def tickets_awaiting_close_check() -> list[FeedbackTicket]:
    """Returns reports whose issue is filed and whose close has not been seen yet.

    These are the ones the discovery pass has to ask GitHub about, so the set is every
    report that is still open plus any that closed since the last pass. It is deliberately
    unbounded: a report that stays open never leaves the set, so a batch cap would starve
    the tail of the list rather than spread the work out. The repository holds a handful of
    reports against a 5000-per-hour credential, and the sweep runs every ten minutes.
    """
    await _ensure_schema()
    async with open_session() as session:
        seen = (
            select(FeedbackCloseNoticeRow.ticket_id)
            .where(FeedbackCloseNoticeRow.ticket_id == FeedbackTicketRow.ticket_id)
            .exists()
        )
        result = await session.execute(
            statement=select(FeedbackTicketRow)
            .where(FeedbackTicketRow.issue_number.is_not(None), ~seen)
            .order_by(FeedbackTicketRow.ticket_id.asc())
        )
        return [_row_to_model(row=row) for row in result.scalars().all()]


async def record_close_observed(*, ticket_id: int, notified: bool) -> None:
    """Records that a report's issue was seen closed, and whether that settles it.

    `notified=True` is for a close nobody is told about, which finishes the report here
    rather than leaving a row the delivery pass would keep picking up and re-deciding.

    Writing is conditional on there being no row yet, so a second discovery pass that
    overlaps the first cannot reset an `observed_at` the delivery pass is already waiting
    on, nor overwrite a `notified_at` that is already set.
    """
    await _ensure_schema()
    async with open_session() as session:
        existing = await session.get(entity=FeedbackCloseNoticeRow, ident=ticket_id)
        if existing is not None:
            return
        session.add(
            FeedbackCloseNoticeRow(
                ticket_id=ticket_id, notified_at=_database_now() if notified else None
            )
        )
        await session.commit()


async def close_notices_awaiting_delivery(*, min_age_seconds: float) -> list[PendingCloseNotice]:
    """Returns the closes that have waited long enough to be told to their reporter.

    The wait is what catches a closing comment written after the close: they are separate
    actions on GitHub, and the comment is the whole point of the message. Anchoring it on
    `observed_at` rather than on the tick that follows discovery keeps that true across a
    restart, which a tick counter would not survive.
    """
    await _ensure_schema()
    cutoff = _database_now() - timedelta(seconds=min_age_seconds)
    async with open_session() as session:
        result = await session.execute(
            statement=select(FeedbackCloseNoticeRow, FeedbackTicketRow)
            .join(
                FeedbackTicketRow, FeedbackTicketRow.ticket_id == FeedbackCloseNoticeRow.ticket_id
            )
            .where(
                FeedbackCloseNoticeRow.notified_at.is_(None),
                FeedbackCloseNoticeRow.observed_at <= cutoff,
            )
            .order_by(FeedbackCloseNoticeRow.ticket_id.asc())
        )
        return [
            PendingCloseNotice(ticket=_row_to_model(row=ticket), observed_at=notice.observed_at)
            for notice, ticket in result.all()
        ]


async def mark_close_notified(*, ticket_id: int) -> None:
    """Finishes a report off, so its close is never announced a second time."""
    await _ensure_schema()
    async with open_session() as session:
        await session.execute(
            statement=update(FeedbackCloseNoticeRow)
            .where(FeedbackCloseNoticeRow.ticket_id == ticket_id)
            .values(notified_at=_database_now())
        )
        await session.commit()


async def forget_close_notice(*, ticket_id: int) -> None:
    """Drops a recorded close, putting the report back to never having been closed.

    For a report reopened during the wait between discovery and delivery. Marking it
    finished would spend the one message it gets on a close that was taken back, so the row
    goes instead and a later close is discovered from scratch.

    Conditional on nothing having been sent yet, because that is the whole distinction:
    once a message has gone out, a reopen must not hand the report a second one.
    """
    await _ensure_schema()
    async with open_session() as session:
        await session.execute(
            statement=delete(FeedbackCloseNoticeRow).where(
                FeedbackCloseNoticeRow.ticket_id == ticket_id,
                FeedbackCloseNoticeRow.notified_at.is_(None),
            )
        )
        await session.commit()


async def tickets_awaiting_issue(
    *, limit: int, min_age_seconds: float = 0.0
) -> list[FeedbackTicket]:
    """Returns reports whose issue was never opened, oldest first.

    These are the rows the submit path could not hand to GitHub. They are the whole
    reason the local write comes first, so the sweep exists to finish that hand-off.

    `min_age_seconds` excludes rows a submit may still be working on: between opening an
    issue and recording its number the row looks exactly like one that failed, and taking
    it here would open a second issue for the same report.
    """
    await _ensure_schema()
    cutoff = _database_now() - timedelta(seconds=min_age_seconds)
    async with open_session() as session:
        result = await session.execute(
            statement=select(FeedbackTicketRow)
            .where(
                FeedbackTicketRow.issue_number.is_(None), FeedbackTicketRow.created_at <= cutoff
            )
            .order_by(FeedbackTicketRow.ticket_id.asc())
            .limit(limit)
        )
        return [_row_to_model(row=row) for row in result.scalars().all()]
