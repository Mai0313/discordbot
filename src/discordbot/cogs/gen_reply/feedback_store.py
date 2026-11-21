"""Reply turn snapshots and user feedback stored in `data/database/reply.db`.

Owns its own `Base` metadata on `reply.db` and omits `from __future__ import annotations` so
SQLAlchemy resolves `Mapped[datetime]` at class-build time without ruff's TC rules hiding
`datetime` under `TYPE_CHECKING`.
"""

from typing import Literal
from datetime import datetime
from contextlib import AbstractAsyncContextManager

from pydantic import Field, BaseModel
from sqlalchemy import Text, Index, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.sqlite_config import SqliteBootstrap

FeedbackVote = Literal["up", "down"]
DownvoteReasonTag = Literal["off_topic", "inaccurate", "tone", "verbose", "other"]
ReplyRoute = Literal["QA", "IMAGE", "VIDEO"]
ReplySurface = Literal["gateway", "ask"]
FeedbackAction = Literal["set", "change", "clear"]

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/reply.db")


class Base(DeclarativeBase):
    """Base class for the reply-turn snapshot and feedback models."""


class ReplyTurnRow(Base):
    """Snapshot of one finalized bot reply, keyed by the message holding its feedback buttons."""

    __tablename__ = "reply_turn"

    reply_message_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    guild_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channel_id: Mapped[int] = mapped_column(Integer, nullable=False)
    author_id: Mapped[int] = mapped_column(Integer, nullable=False)
    author_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    author_display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    surface: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    effort: Mapped[str] = mapped_column(Text, nullable=False)
    backend: Mapped[str] = mapped_column(Text, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)

    __table_args__ = (
        Index("ix_reply_turn_source_message_id", "source_message_id"),
        Index("ix_reply_turn_model_created_at", "model", "created_at"),
    )


class ReplyFeedbackRow(Base):
    """One user's current feedback vote (and optional downvote reason) on a finalized reply."""

    __tablename__ = "reply_feedback"

    reply_message_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("reply_turn.reply_message_id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_name: Mapped[str] = mapped_column(Text, nullable=False)
    user_display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    vote: Mapped[str] = mapped_column(Text, nullable=False)
    reason_tag: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)


class FeedbackVoteResult(BaseModel):
    """Outcome of pressing thumbs-up or thumbs-down on a reply."""

    action: FeedbackAction = Field(
        ..., description="Whether the press created, switched, or cleared the user's vote."
    )
    vote: FeedbackVote | None = Field(
        ..., description="The user's resulting vote, or None when cleared."
    )
    source_message_id: int = Field(
        ..., description="Triggering Discord message or interaction id for log correlation."
    )
    model: str = Field(..., description="Actual model recorded on the reply turn.")
    effort: str = Field(..., description="Reasoning effort recorded on the reply turn.")
    route: str = Field(..., description="Pipeline route recorded on the reply turn.")
    is_asker: bool = Field(
        ..., description="Whether the voter is the original author of the prompt."
    )


class FeedbackReasonResult(BaseModel):
    """Outcome of selecting a downvote reason tag or submitting free-text reason details."""

    updated: bool = Field(
        ..., description="False when the user no longer holds an active downvote on this reply."
    )
    reason_tag: DownvoteReasonTag | None = Field(
        default=None, description="Resulting reason tag on the downvote row, or None when cleared."
    )
    reason_text: str | None = Field(
        default=None, description="Resulting free-text detail for `other`, or None."
    )
    source_message_id: int | None = Field(
        default=None, description="Triggering Discord message id when the turn exists."
    )
    model: str | None = Field(
        default=None, description="Actual model recorded on the reply turn, when found."
    )
    effort: str | None = Field(
        default=None, description="Reasoning effort recorded on the reply turn, when found."
    )
    route: str | None = Field(
        default=None, description="Pipeline route recorded on the reply turn, when found."
    )
    is_asker: bool = Field(
        default=False, description="Whether the voter is the original author of the prompt."
    )


_database = SqliteBootstrap(enable_foreign_keys=True)


def open_session() -> AbstractAsyncContextManager[AsyncSession]:
    """Returns an async session context manager on `reply.db`."""
    return _database.open_session(engine=_engine)


async def record_reply_turn(  # noqa: PLR0913 -- one snapshot row's columns are all per-turn inputs
    reply_message_id: int,
    source_message_id: int,
    guild_id: int | None,
    channel_id: int,
    author_id: int,
    surface: ReplySurface,
    route: ReplyRoute,
    model: str,
    effort: str,
    backend: str,
    question: str,
    answer: str,
    author_name: str | None = None,
    author_display_name: str | None = None,
) -> None:
    """Persists one finalized reply snapshot, updating in place if the reply id already exists.

    Args:
        reply_message_id (int): Discord message id carrying the feedback buttons.
        source_message_id (int): Triggering message or `/ask` interaction id.
        guild_id (int | None): Guild id from the turn surface, or None in a DM.
        channel_id (int): Channel id where the turn ran.
        author_id (int): Discord user id who asked the prompt.
        surface (ReplySurface): `"gateway"` or `"ask"`.
        route (ReplyRoute): `"QA"`, `"IMAGE"`, or `"VIDEO"`.
        model (str): Actual model name (post-fallback for QA; media generator for IMAGE/VIDEO).
        effort (str): Reasoning effort string, or empty for non-reasoning media routes.
        backend (str): Backend label (`"responses"` or `"interactions"`).
        question (str): Snapshot of the user's request text.
        answer (str): Snapshot of the finalized reply text without the usage footer.
        author_name (str | None): Write-time username of the prompt author, if available.
        author_display_name (str | None): Write-time display name or guild nick of the author.
    """
    async with open_session() as session, session.begin():
        await session.merge(
            instance=ReplyTurnRow(
                reply_message_id=reply_message_id,
                source_message_id=source_message_id,
                guild_id=guild_id,
                channel_id=channel_id,
                author_id=author_id,
                author_name=author_name,
                author_display_name=author_display_name,
                surface=surface,
                route=route,
                model=model,
                effort=effort,
                backend=backend,
                question=question,
                answer=answer,
            )
        )


async def toggle_feedback_vote(
    reply_message_id: int,
    user_id: int,
    user_name: str,
    vote: FeedbackVote,
    user_display_name: str | None = None,
) -> FeedbackVoteResult | None:
    """Creates, switches, or clears `user_id`'s vote on `reply_message_id`.

    Pressing the same vote again removes the row; pressing the opposite vote overwrites it in
    place and clears any prior downvote reason. Returns None when no `reply_turn` row exists for
    `reply_message_id`.

    Args:
        reply_message_id (int): Discord message id carrying the feedback buttons.
        user_id (int): Discord user id pressing the button.
        user_name (str): Write-time username snapshot for readability in reports.
        vote (FeedbackVote): `"up"` or `"down"`.
        user_display_name (str | None): Write-time display name or guild nick of the voter.
    """
    async with open_session() as session, session.begin():
        turn = await session.get(entity=ReplyTurnRow, ident=reply_message_id)
        if turn is None:
            return None
        existing = await session.get(entity=ReplyFeedbackRow, ident=(reply_message_id, user_id))
        now = _database_now()
        action: FeedbackAction
        resulting_vote: FeedbackVote | None
        if existing is None:
            session.add(
                instance=ReplyFeedbackRow(
                    reply_message_id=reply_message_id,
                    user_id=user_id,
                    user_name=user_name,
                    user_display_name=user_display_name,
                    vote=vote,
                    created_at=now,
                    updated_at=now,
                )
            )
            action = "set"
            resulting_vote = vote
        elif existing.vote == vote:
            await session.delete(instance=existing)
            action = "clear"
            resulting_vote = None
        else:
            existing.vote = vote
            existing.user_name = user_name
            existing.user_display_name = user_display_name
            existing.reason_tag = None
            existing.reason_text = None
            existing.updated_at = now
            action = "change"
            resulting_vote = vote
        return FeedbackVoteResult(
            action=action,
            vote=resulting_vote,
            source_message_id=turn.source_message_id,
            model=turn.model,
            effort=turn.effort,
            route=turn.route,
            is_asker=user_id == turn.author_id,
        )


async def clear_feedback_vote(
    reply_message_id: int, user_id: int, expected_vote: FeedbackVote
) -> FeedbackVoteResult | None:
    """Deletes `user_id`'s vote on `reply_message_id` only when it still matches `expected_vote`.

    Returns `FeedbackVoteResult` with `action="clear"` and `vote=None` when cleared, or None when
    the turn is missing or the user already changed or cleared their vote.

    Args:
        reply_message_id (int): Discord message id carrying the feedback buttons.
        user_id (int): Discord user id cancelling their vote.
        expected_vote (FeedbackVote): Vote (`"up"` or `"down"`) that this confirmation belongs to.
    """
    async with open_session() as session, session.begin():
        turn = await session.get(entity=ReplyTurnRow, ident=reply_message_id)
        if turn is None:
            return None
        existing = await session.get(entity=ReplyFeedbackRow, ident=(reply_message_id, user_id))
        if existing is None or existing.vote != expected_vote:
            return None
        await session.delete(instance=existing)
        return FeedbackVoteResult(
            action="clear",
            vote=None,
            source_message_id=turn.source_message_id,
            model=turn.model,
            effort=turn.effort,
            route=turn.route,
            is_asker=user_id == turn.author_id,
        )


async def set_feedback_reason(
    reply_message_id: int,
    user_id: int,
    reason_tag: DownvoteReasonTag,
    reason_text: str | None = None,
) -> FeedbackReasonResult:
    """Sets, switches, or toggles off the downvote reason on an active downvote row.

    Re-clicking the already selected preset tag clears it; submitting `"other"` always stores
    `reason_text`. Returns `updated=False` when the user no longer holds a `"down"` vote.

    Args:
        reply_message_id (int): Discord message id carrying the feedback buttons.
        user_id (int): Discord user id supplying the reason.
        reason_tag (DownvoteReasonTag): Selected reason category.
        reason_text (str | None): Optional free-text detail when `reason_tag == "other"`.
    """
    async with open_session() as session, session.begin():
        turn = await session.get(entity=ReplyTurnRow, ident=reply_message_id)
        existing = await session.get(entity=ReplyFeedbackRow, ident=(reply_message_id, user_id))
        if turn is None or existing is None or existing.vote != "down":
            return FeedbackReasonResult(updated=False)
        cleaned_text = reason_text.strip() if reason_text else None
        resulting_tag: DownvoteReasonTag | None
        resulting_text: str | None
        if reason_tag != "other" and existing.reason_tag == reason_tag:
            resulting_tag = None
            resulting_text = None
        elif reason_tag == "other":
            resulting_tag = "other"
            resulting_text = cleaned_text
        else:
            resulting_tag = reason_tag
            resulting_text = None
        existing.reason_tag = resulting_tag
        existing.reason_text = resulting_text
        existing.updated_at = _database_now()
        return FeedbackReasonResult(
            updated=True,
            reason_tag=resulting_tag,
            reason_text=resulting_text,
            source_message_id=turn.source_message_id,
            model=turn.model,
            effort=turn.effort,
            route=turn.route,
            is_asker=user_id == turn.author_id,
        )
