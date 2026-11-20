"""Tests for the `reply_turn` and `reply_feedback` store in `reply.db`."""

from sqlalchemy import select

from discordbot.cogs.gen_reply.feedback_store import (
    ReplyTurnRow,
    ReplyFeedbackRow,
    open_session,
    record_reply_turn,
    set_feedback_reason,
    toggle_feedback_vote,
)


async def _seed_turn(reply_message_id: int = 9001, author_id: int = 100) -> None:
    """Inserts one baseline `reply_turn` row for feedback tests."""
    await record_reply_turn(
        reply_message_id=reply_message_id,
        source_message_id=8001,
        guild_id=500,
        channel_id=600,
        author_id=author_id,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="medium",
        backend="responses",
        question="What is SQLite WAL?",
        answer="Write-ahead logging lets readers proceed alongside a writer.",
    )


async def test_vote_toggle_switch_and_cancel_leave_a_single_clean_row() -> None:
    """Misclicks overwrite in place, re-clicking cancels cleanly, and switching clears reasons."""
    await _seed_turn()

    first = await toggle_feedback_vote(
        reply_message_id=9001, user_id=100, user_name="asker", vote="down"
    )
    assert first is not None
    assert first.action == "set"
    assert first.vote == "down"
    assert first.is_asker is True
    assert first.model == "gemini-4-neon"

    reason = await set_feedback_reason(reply_message_id=9001, user_id=100, reason_tag="inaccurate")
    assert reason.updated is True
    assert reason.reason_tag == "inaccurate"

    switched = await toggle_feedback_vote(
        reply_message_id=9001, user_id=100, user_name="asker", vote="up"
    )
    assert switched is not None
    assert switched.action == "change"
    assert switched.vote == "up"

    async with open_session() as session:
        rows = list((await session.execute(select(ReplyFeedbackRow))).scalars())
    assert len(rows) == 1
    assert rows[0].vote == "up"
    assert rows[0].reason_tag is None
    assert rows[0].reason_text is None

    cancelled = await toggle_feedback_vote(
        reply_message_id=9001, user_id=100, user_name="asker", vote="up"
    )
    assert cancelled is not None
    assert cancelled.action == "clear"
    assert cancelled.vote is None

    async with open_session() as session:
        rows = list((await session.execute(select(ReplyFeedbackRow))).scalars())
    assert rows == []


async def test_vote_returns_none_for_unknown_reply_message() -> None:
    """Pressing a button on a reply with no stored turn returns None instead of failing."""
    result = await toggle_feedback_vote(
        reply_message_id=999999, user_id=100, user_name="asker", vote="up"
    )
    assert result is None


async def test_reason_selection_switches_clears_and_rejects_non_downvotes() -> None:
    """Preset tags switch or toggle off; `other` stores text; upvotes ignore stale reason clicks."""
    await _seed_turn(reply_message_id=9002, author_id=100)
    await toggle_feedback_vote(reply_message_id=9002, user_id=200, user_name="viewer", vote="down")

    tagged = await set_feedback_reason(reply_message_id=9002, user_id=200, reason_tag="verbose")
    assert tagged.updated is True
    assert tagged.is_asker is False
    assert tagged.reason_tag == "verbose"

    cleared = await set_feedback_reason(reply_message_id=9002, user_id=200, reason_tag="verbose")
    assert cleared.updated is True
    assert cleared.reason_tag is None

    other = await set_feedback_reason(
        reply_message_id=9002, user_id=200, reason_tag="other", reason_text="  missed the image  "
    )
    assert other.updated is True
    assert other.reason_tag == "other"
    assert other.reason_text == "missed the image"

    switched = await set_feedback_reason(reply_message_id=9002, user_id=200, reason_tag="tone")
    assert switched.reason_tag == "tone"
    assert switched.reason_text is None

    await toggle_feedback_vote(reply_message_id=9002, user_id=200, user_name="viewer", vote="up")
    stale = await set_feedback_reason(reply_message_id=9002, user_id=200, reason_tag="off_topic")
    assert stale.updated is False


async def test_record_reply_turn_is_idempotent_on_reply_message_id() -> None:
    """Re-recording the same `reply_message_id` updates in place rather than failing."""
    await _seed_turn(reply_message_id=9003)
    await record_reply_turn(
        reply_message_id=9003,
        source_message_id=8001,
        guild_id=500,
        channel_id=600,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-hydrogen",
        effort="high",
        backend="interactions",
        question="Updated",
        answer="Updated answer",
    )
    async with open_session() as session:
        turn = await session.get(entity=ReplyTurnRow, ident=9003)
    assert turn is not None
    assert turn.model == "gemini-4-hydrogen"
    assert turn.answer == "Updated answer"


async def test_deleting_reply_turn_cascades_to_feedback_rows() -> None:
    """`PRAGMA foreign_keys` is enabled on `feedback_store`, so deleting a turn cascades votes."""
    await _seed_turn(reply_message_id=9004)
    await toggle_feedback_vote(reply_message_id=9004, user_id=100, user_name="asker", vote="up")

    async with open_session() as session, session.begin():
        turn = await session.get(entity=ReplyTurnRow, ident=9004)
        assert turn is not None
        await session.delete(instance=turn)

    async with open_session() as session:
        remaining = list((await session.execute(select(ReplyFeedbackRow))).scalars())
    assert remaining == []
