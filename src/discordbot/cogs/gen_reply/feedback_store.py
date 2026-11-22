"""Reply turn snapshots and user feedback stored in `data/database/reply.db`.

Owns its own `Base` metadata on `reply.db` and omits `from __future__ import annotations` so
SQLAlchemy resolves `Mapped[datetime]` at class-build time without ruff's TC rules hiding
`datetime` under `TYPE_CHECKING`.
"""

import json
from typing import Literal
from datetime import datetime
from contextlib import AbstractAsyncContextManager
from collections.abc import Sequence

from pydantic import Field, BaseModel
from sqlalchemy import Text, Index, Integer, DateTime, ForeignKey, select
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncSession

from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.sqlite_config import open_reply_session

FeedbackVote = Literal["up", "down"]
DownvoteReasonTag = Literal["off_topic", "inaccurate", "tone", "verbose", "other"]
FeedbackReasonTag = DownvoteReasonTag | Literal["regen"]
ReplyRoute = Literal["QA", "IMAGE", "VIDEO"]
ReplySurface = Literal["gateway", "ask"]
FeedbackAction = Literal["set", "change", "clear"]


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
    regen_of_reply_message_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(
            "reply_turn.reply_message_id",
            name="fk_reply_turn_regen_of_reply_message_id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )
    reply_message_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_database_now)

    __table_args__ = (
        Index("ix_reply_turn_source_message_id", "source_message_id"),
        Index("ix_reply_turn_model_created_at", "model", "created_at"),
        Index("ix_reply_turn_regen_of_reply_message_id", "regen_of_reply_message_id", unique=True),
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


class RegenChainState(BaseModel):
    """Target turn metadata and ancestry state for one regeneration check."""

    reply_message_id: int = Field(
        ..., description="Discord message id of the reply targeted for regeneration."
    )
    source_message_id: int = Field(
        ..., description="Triggering Discord source message id to re-answer."
    )
    channel_id: int = Field(..., description="Discord channel id where the reply lives.")
    author_id: int = Field(..., description="Discord user id who originally asked the prompt.")
    surface: str = Field(..., description="Surface that ran the turn (`gateway` or `ask`).")
    route: str = Field(..., description="Pipeline route (`QA`, `IMAGE`, or `VIDEO`).")
    effort: str = Field(..., description="Reasoning effort recorded on the reply turn.")
    reply_message_ids: str | None = Field(
        default=None, description="JSON-encoded list of Discord chunk message ids for this reply."
    )
    already_regenerated: bool = Field(
        ..., description="Whether another `reply_turn` row already regenerates this reply."
    )
    chain_depth: int = Field(
        ..., description="Number of ancestor turns before this reply in its regen chain."
    )
    tried_models: tuple[str, ...] = Field(
        ..., description="Models already tried across this reply and its regen ancestors."
    )


def parse_reply_message_ids(raw: str | None, fallback_id: int) -> list[int]:
    """Returns ordered, deduplicated chunk message ids decoded from `raw`, falling back to `fallback_id`."""
    if not raw:
        return [fallback_id]
    try:
        parsed = json.loads(s=raw)
    except ValueError:
        return [fallback_id]
    if not isinstance(parsed, list):
        return [fallback_id]
    ids: list[int] = []
    for item in parsed:
        if isinstance(item, int) and not isinstance(item, bool) and item not in ids:
            ids.append(item)
    if not ids:
        return [fallback_id]
    if fallback_id not in ids:
        ids.append(fallback_id)
    return ids


def open_session() -> AbstractAsyncContextManager[AsyncSession]:
    """Returns an async session context manager on the shared `reply.db` engine."""
    return open_reply_session()


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
    reply_message_ids: Sequence[int] | None = None,
    regen_of_reply_message_id: int | None = None,
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
        reply_message_ids (Sequence[int] | None): Ordered Discord chunk message ids for this reply.
        regen_of_reply_message_id (int | None): Parent reply message id replaced by this regen turn.
    """
    serialized_ids = json.dumps(obj=list(reply_message_ids)) if reply_message_ids else None
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
                regen_of_reply_message_id=regen_of_reply_message_id,
                reply_message_ids=serialized_ids,
            )
        )


async def update_reply_message_ids(
    reply_message_id: int, reply_message_ids: Sequence[int]
) -> None:
    """Updates `reply_message_ids` on an existing `reply_turn` row (e.g. after hosted-link follow-up)."""
    if not reply_message_ids:
        return
    async with open_session() as session, session.begin():
        turn = await session.get(entity=ReplyTurnRow, ident=reply_message_id)
        if turn is None:
            return
        turn.reply_message_ids = json.dumps(obj=list(reply_message_ids))


async def load_reply_turn(reply_message_id: int) -> ReplyTurnRow | None:
    """Returns the single `ReplyTurnRow` for `reply_message_id`, or None when absent."""
    async with open_session() as session:
        return await session.get(entity=ReplyTurnRow, ident=reply_message_id)


async def load_regen_chain_state(reply_message_id: int) -> RegenChainState | None:
    """Loads regeneration target metadata, whether it was already regenerated, and chain ancestry."""
    async with open_session() as session:
        turn = await session.get(entity=ReplyTurnRow, ident=reply_message_id)
        if turn is None:
            return None
        child_id = (
            await session.execute(
                statement=select(ReplyTurnRow.reply_message_id)
                .where(ReplyTurnRow.regen_of_reply_message_id == reply_message_id)
                .limit(limit=1)
            )
        ).scalar_one_or_none()
        tried_models: list[str] = [turn.model]
        seen: set[int] = {turn.reply_message_id}
        parent_id = turn.regen_of_reply_message_id
        chain_depth = 0
        while parent_id is not None and parent_id not in seen:
            seen.add(parent_id)
            parent = await session.get(entity=ReplyTurnRow, ident=parent_id)
            if parent is None:
                break
            chain_depth += 1
            tried_models.append(parent.model)
            parent_id = parent.regen_of_reply_message_id
        return RegenChainState(
            reply_message_id=turn.reply_message_id,
            source_message_id=turn.source_message_id,
            channel_id=turn.channel_id,
            author_id=turn.author_id,
            surface=turn.surface,
            route=turn.route,
            effort=turn.effort,
            reply_message_ids=turn.reply_message_ids,
            already_regenerated=child_id is not None,
            chain_depth=chain_depth,
            tried_models=tuple(tried_models),
        )


async def record_regen_feedback(
    reply_message_id: int, user_id: int, user_name: str, user_display_name: str | None = None
) -> None:
    """Records implicit `reason_tag='regen'` downvote on a replaced reply without clobbering explicit downvotes.

    Inserts a new `vote='down', reason_tag='regen'` row when absent, overwrites an existing
    `vote='up'` row, and preserves any existing `vote='down'` row (and its explicit reason).
    """
    async with open_session() as session, session.begin():
        turn = await session.get(entity=ReplyTurnRow, ident=reply_message_id)
        if turn is None:
            return
        existing = await session.get(entity=ReplyFeedbackRow, ident=(reply_message_id, user_id))
        now = _database_now()
        if existing is None:
            session.add(
                instance=ReplyFeedbackRow(
                    reply_message_id=reply_message_id,
                    user_id=user_id,
                    user_name=user_name,
                    user_display_name=user_display_name,
                    vote="down",
                    reason_tag="regen",
                    reason_text=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif existing.vote == "up":
            existing.vote = "down"
            existing.user_name = user_name
            existing.user_display_name = user_display_name
            existing.reason_tag = "regen"
            existing.reason_text = None
            existing.updated_at = now


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
