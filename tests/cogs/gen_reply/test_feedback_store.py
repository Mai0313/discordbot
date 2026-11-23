"""Tests for the `reply_turn` and `reply_feedback` store in `reply.db`."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from discordbot.cogs.gen_reply.feedback_store import (
    ReplyTurnRow,
    ReplyFeedbackRow,
    open_session,
    record_reply_turn,
    clear_feedback_vote,
    set_feedback_reason,
    toggle_feedback_vote,
    record_regen_feedback,
    load_regen_chain_state,
    parse_reply_message_ids,
    update_reply_message_ids,
)


async def _seed_turn(reply_message_id: int = 9001, author_id: int = 100) -> None:
    """Inserts one baseline `reply_turn` row for feedback tests."""
    await record_reply_turn(
        reply_message_id=reply_message_id,
        source_message_id=8001,
        guild_id=500,
        channel_id=600,
        author_id=author_id,
        author_name="asker",
        author_display_name="Asker Nick",
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
        reply_message_id=9001,
        user_id=100,
        user_name="asker",
        user_display_name="Old Nick",
        vote="down",
    )
    assert first is not None
    assert first.action == "set"
    assert first.vote == "down"
    assert first.is_asker is True
    assert first.model == "gemini-4-neon"

    async with open_session() as session:
        turn = await session.get(entity=ReplyTurnRow, ident=9001)
        initial_row = await session.get(entity=ReplyFeedbackRow, ident=(9001, 100))
    assert turn is not None
    assert turn.author_name == "asker"
    assert turn.author_display_name == "Asker Nick"
    assert initial_row is not None
    assert initial_row.user_display_name == "Old Nick"

    reason = await set_feedback_reason(reply_message_id=9001, user_id=100, reason_tag="inaccurate")
    assert reason.updated is True
    assert reason.reason_tag == "inaccurate"

    switched = await toggle_feedback_vote(
        reply_message_id=9001,
        user_id=100,
        user_name="asker_renamed",
        user_display_name="New Nick",
        vote="up",
    )
    assert switched is not None
    assert switched.action == "change"
    assert switched.vote == "up"

    async with open_session() as session:
        rows = list((await session.execute(select(ReplyFeedbackRow))).scalars())
    assert len(rows) == 1
    assert rows[0].vote == "up"
    assert rows[0].user_name == "asker_renamed"
    assert rows[0].user_display_name == "New Nick"
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
        effort="xhigh",
        backend="interactions",
        question="Updated",
        answer="Updated answer",
        answer_seconds=3.75,
    )
    async with open_session() as session:
        turn = await session.get(entity=ReplyTurnRow, ident=9003)
    assert turn is not None
    assert turn.model == "gemini-4-hydrogen"
    assert turn.effort == "xhigh"
    assert turn.answer == "Updated answer"
    assert turn.answer_seconds == pytest.approx(expected=3.75)


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


async def test_clear_feedback_vote_requires_matching_expected_vote() -> None:
    """`clear_feedback_vote` deletes only when the current vote matches `expected_vote`."""
    await _seed_turn(reply_message_id=9005, author_id=100)
    await toggle_feedback_vote(reply_message_id=9005, user_id=100, user_name="asker", vote="up")

    # Mismatched expected_vote leaves the row intact and signals not cleared.
    mismatch = await clear_feedback_vote(reply_message_id=9005, user_id=100, expected_vote="down")
    assert mismatch is None
    async with open_session() as session:
        still_present = await session.get(entity=ReplyFeedbackRow, ident=(9005, 100))
    assert still_present is not None
    assert still_present.vote == "up"

    # Matching expected_vote deletes the row and returns metadata with action="clear".
    cleared = await clear_feedback_vote(reply_message_id=9005, user_id=100, expected_vote="up")
    assert cleared is not None
    assert cleared.action == "clear"
    assert cleared.vote is None
    assert cleared.is_asker is True
    assert cleared.model == "gemini-4-neon"

    # Clearing again after removal returns None.
    again = await clear_feedback_vote(reply_message_id=9005, user_id=100, expected_vote="up")
    assert again is None


async def test_regen_chain_state_tracks_ancestry_chunk_ids_and_unique_parent() -> None:
    """`load_regen_chain_state` walks regen ancestors, reports child existence, and enforces unique parent."""
    await record_reply_turn(
        reply_message_id=9101,
        source_message_id=8100,
        guild_id=500,
        channel_id=600,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-carbon",
        effort="high",
        backend="responses",
        question="Compare A and B",
        answer="First try",
        reply_message_ids=[9100, 9101],
    )
    state1 = await load_regen_chain_state(reply_message_id=9101)
    assert state1 is not None
    assert state1.already_regenerated is False
    assert state1.chain_depth == 0
    assert state1.tried_models == ("gemini-4-carbon",)
    assert parse_reply_message_ids(raw=state1.reply_message_ids, fallback_id=9101) == [9100, 9101]

    await update_reply_message_ids(reply_message_id=9101, reply_message_ids=[9100, 9101, 9109])
    updated = await load_regen_chain_state(reply_message_id=9101)
    assert updated is not None
    assert parse_reply_message_ids(raw=updated.reply_message_ids, fallback_id=9101) == [
        9100,
        9101,
        9109,
    ]

    await record_reply_turn(
        reply_message_id=9102,
        source_message_id=8100,
        guild_id=500,
        channel_id=600,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-neon",
        effort="high",
        backend="responses",
        question="Compare A and B",
        answer="Second try",
        reply_message_ids=[9102],
        regen_of_reply_message_id=9101,
    )
    state1_after = await load_regen_chain_state(reply_message_id=9101)
    assert state1_after is not None
    assert state1_after.already_regenerated is True

    state2 = await load_regen_chain_state(reply_message_id=9102)
    assert state2 is not None
    assert state2.already_regenerated is False
    assert state2.chain_depth == 1
    assert state2.tried_models == ("gemini-4-neon", "gemini-4-carbon")

    await record_reply_turn(
        reply_message_id=9103,
        source_message_id=8100,
        guild_id=500,
        channel_id=600,
        author_id=100,
        surface="gateway",
        route="QA",
        model="gemini-4-hydrogen",
        effort="high",
        backend="responses",
        question="Compare A and B",
        answer="Third try",
        reply_message_ids=[9103],
        regen_of_reply_message_id=9102,
    )
    state3 = await load_regen_chain_state(reply_message_id=9103)
    assert state3 is not None
    assert state3.chain_depth == 2
    assert state3.tried_models == ("gemini-4-hydrogen", "gemini-4-neon", "gemini-4-carbon")

    # Unique index prevents a second regen row from pointing at the same parent reply.
    with pytest.raises(IntegrityError):
        await record_reply_turn(
            reply_message_id=9104,
            source_message_id=8100,
            guild_id=500,
            channel_id=600,
            author_id=100,
            surface="gateway",
            route="QA",
            model="gemini-4-hydrogen",
            effort="high",
            backend="responses",
            question="Compare A and B",
            answer="Duplicate regen",
            regen_of_reply_message_id=9101,
        )

    assert await load_regen_chain_state(reply_message_id=999999) is None


async def test_record_regen_feedback_inserts_overwrites_up_and_preserves_explicit_down() -> None:
    """Implicit regen downvote inserts or flips an upvote, while preserving any explicit downvote."""
    await _seed_turn(reply_message_id=9201, author_id=100)
    await record_regen_feedback(
        reply_message_id=9201, user_id=100, user_name="asker", user_display_name="Asker Nick"
    )
    async with open_session() as session:
        row1 = await session.get(entity=ReplyFeedbackRow, ident=(9201, 100))
        turn1 = await session.get(entity=ReplyTurnRow, ident=9201)
    assert turn1 is not None
    assert row1 is not None
    assert row1.vote == "down"
    assert row1.reason_tag == "regen"
    assert row1.reason_text is None

    # Prior upvote is overwritten to implicit regen downvote.
    await _seed_turn(reply_message_id=9202, author_id=100)
    await toggle_feedback_vote(reply_message_id=9202, user_id=100, user_name="asker", vote="up")
    await record_regen_feedback(
        reply_message_id=9202, user_id=100, user_name="asker", user_display_name="Updated Nick"
    )
    async with open_session() as session:
        row2 = await session.get(entity=ReplyFeedbackRow, ident=(9202, 100))
    assert row2 is not None
    assert row2.vote == "down"
    assert row2.reason_tag == "regen"
    assert row2.user_display_name == "Updated Nick"

    # Prior explicit downvote and its reason are preserved untouched.
    await _seed_turn(reply_message_id=9203, author_id=100)
    await toggle_feedback_vote(reply_message_id=9203, user_id=100, user_name="asker", vote="down")
    await set_feedback_reason(
        reply_message_id=9203, user_id=100, reason_tag="other", reason_text="wrong code"
    )
    await record_regen_feedback(
        reply_message_id=9203, user_id=100, user_name="asker", user_display_name="Asker Nick"
    )
    async with open_session() as session:
        row3 = await session.get(entity=ReplyFeedbackRow, ident=(9203, 100))
    assert row3 is not None
    assert row3.vote == "down"
    assert row3.reason_tag == "other"
    assert row3.reason_text == "wrong code"


def test_parse_reply_message_ids_handles_legacy_malformed_and_missing_fallback() -> None:
    """Pre-migration NULL, invalid JSON, non-lists, and partial lists degrade cleanly."""
    assert parse_reply_message_ids(raw=None, fallback_id=50) == [50]
    assert parse_reply_message_ids(raw="not-json", fallback_id=50) == [50]
    assert parse_reply_message_ids(raw='{"a": 1}', fallback_id=50) == [50]
    assert parse_reply_message_ids(raw="[]", fallback_id=50) == [50]
    assert parse_reply_message_ids(raw="[10, 10, 20]", fallback_id=20) == [10, 20]
    assert parse_reply_message_ids(raw="[10, 11]", fallback_id=12) == [10, 11, 12]
