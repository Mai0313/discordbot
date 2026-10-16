"""Tests for the `memory_job` rows that make a turn resumable after a restart."""

import pytest

from discordbot.utils.timezone import database_now
from discordbot.services.memory import database as memory_db

from tests.helpers.memory import get_job, assert_cleared_row
from tests.helpers.user_memory import USER_ID, IDENTITY, USER_SCOPE, _stage_row


async def test_db_upsert_pending_then_get() -> None:
    await _stage_row(transcript="逐字稿", subject=f"target_user_id: {USER_ID}", identity=IDENTITY)
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "pending"
    assert job.transcript == "逐字稿"
    assert job.flavor == "user"
    assert job.token == 1


async def test_db_upsert_newest_wins_and_older_token_noop() -> None:
    await _stage_row(token=10, transcript="新")
    # An older token must not clobber the newer row.
    await _stage_row(token=5, transcript="舊")
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.token == 10
    assert job.transcript == "新"


async def test_db_mark_done_clears_transcript_and_is_token_guarded() -> None:
    await _stage_row(token=7, transcript="逐字稿")
    # A stale token does not transition the row.
    await memory_db.mark_done(scope=USER_SCOPE, token=6)
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "pending"
    # The owning token marks it done and drops the consumed transcript.
    await memory_db.mark_done(scope=USER_SCOPE, token=7)
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "done"
    assert job.transcript is None


async def test_db_mark_failed_keeps_transcript() -> None:
    await _stage_row(token=3, transcript="逐字稿")
    await memory_db.mark_failed(scope=USER_SCOPE, token=3, error="boom")
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "failed"
    assert job.transcript == "逐字稿"
    assert job.last_error == "boom"


async def test_db_list_resumable_excludes_done() -> None:
    await _stage_row(transcript="a", scope="111")
    await _stage_row(transcript="b", scope="222")
    await memory_db.mark_done(scope="222", token=1)
    scopes = {job.scope for job in await memory_db.list_resumable(updated_before=database_now())}
    assert scopes == {"111"}


async def test_db_logical_tokens_follow_capture_order() -> None:
    older = memory_db.new_token()
    newer = memory_db.new_token()
    await _stage_row(token=older, transcript="older", scope="111")
    await _stage_row(token=newer, transcript="newer", scope="222")

    older_job = await get_job(scope="111")
    newer_job = await get_job(scope="222")
    assert older_job is not None
    assert newer_job is not None
    assert 0 < older_job.token < newer_job.token


async def test_db_new_process_reserves_a_newer_token_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _stage_row(token=memory_db.new_token(), transcript="first process", scope="111")
    first_job = await get_job(scope="111")
    assert first_job is not None

    # A process restart loses its local mapping and sequence, then reserves past
    # the durable high watermark rather than reusing the old range.
    monkeypatch.setattr(memory_db, "_token_block_bases", {})
    monkeypatch.setattr(memory_db, "_token_sequence", iter(range(1, 10)))
    await _stage_row(token=memory_db.new_token(), transcript="second process", scope="222")
    second_job = await get_job(scope="222")
    assert second_job is not None
    assert second_job.token > first_job.token


async def test_db_clear_job_scrubs_payload_and_is_not_resumable() -> None:
    """A durable clear marker must retain no extractable conversation content."""
    await _stage_row(
        token=7,
        transcript="要清除的逐字稿",
        subject="target_user_id: 123456789",
        identity=IDENTITY,
    )
    await memory_db.mark_failed(scope=USER_SCOPE, token=7, error="provider leaked this error")

    assert await memory_db.clear_job(scope=USER_SCOPE, flavor="user", token=8) is True

    assert (await assert_cleared_row(scope=USER_SCOPE)).token == 8
    assert USER_SCOPE not in {
        job.scope for job in await memory_db.list_resumable(updated_before=database_now())
    }


async def test_db_clear_job_rejects_stale_upsert_but_allows_a_newer_turn() -> None:
    await memory_db.clear_job(scope=USER_SCOPE, flavor="user", token=20)
    await _stage_row(
        token=19, transcript="stale transcript", subject="stale subject", identity="stale identity"
    )

    assert (await assert_cleared_row(scope=USER_SCOPE)).token == 20

    await _stage_row(
        token=21, transcript="new transcript", subject="new subject", identity="new identity"
    )
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "pending"
    assert job.token == 21
    assert job.transcript == "new transcript"
    # A clear older than the row cannot write its tombstone, and reporting that as an
    # ordinary empty scope let the caller delete the files with this transcript still
    # resumable. It refuses instead, leaving the row exactly as it found it.
    with pytest.raises(RuntimeError, match="newer than the clear"):
        await memory_db.clear_job(scope=USER_SCOPE, flavor="user", token=20)
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "pending"
    assert job.token == 21
    assert job.transcript == "new transcript"


async def test_db_clear_job_keeps_an_empty_tombstone_and_is_idempotent() -> None:
    await _stage_row(transcript="逐字稿")
    assert await memory_db.clear_job(scope=USER_SCOPE, flavor="user", token=2) is True
    await assert_cleared_row(scope=USER_SCOPE)
    # The durable marker remains, but a second clear reports no user data removed.
    assert await memory_db.clear_job(scope=USER_SCOPE, flavor="user", token=3) is False
