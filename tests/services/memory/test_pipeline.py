"""Tests for the memory pipeline: staging a turn's notes, reports, resumed rows and clears."""

import time
from types import SimpleNamespace
import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from discordbot.typings.memory import MemoryWriteSummary
from discordbot.utils.timezone import database_now
from discordbot.services.memory import database as memory_db
from discordbot.services.memory import inflight, pipeline, consolidation
from discordbot.services.memory.store import (
    read_tone,
    write_fact,
    mark_cleared,
    server_scope,
    cleared_since,
    raw_file_bytes,
    append_raw_entry,
    read_detail_tail,
    read_raw_entries,
    count_raw_entries,
    list_compartments,
)
from discordbot.services.memory.writer import (
    MemoryWriterAI,
    RawMemoryDraft,
    MemoryObservation,
    user_subject,
    server_subject,
    render_turn_payload,
)
from discordbot.services.memory.prompts import PHASE1_EVALUATOR_PROMPT
from discordbot.services.memory.raw_entries import render_memory_observations

from tests.helpers.memory import (
    get_job,
    assert_cleared_row,
    drain_memory_turns,
    wait_for_persisted_writes,
)
from tests.helpers.user_memory import (
    _NOTES,
    USER_ID,
    IDENTITY,
    _SUBJECT,
    USER_SCOPE,
    _draft,
    _writer,
    _answers,
    _schedule,
    _no_signal,
    _stage_row,
    _memory_text,
    _observation,
    _stored_fact,
    _report_recorder,
    _wait_for_inflight,
    _populate_every_tier,
    _fail_the_file_delete,
)
from tests.helpers.logfire_capture import capture_logs


async def test_pipeline_skips_a_turn_that_marked_nothing() -> None:
    """No marker, no work at all: no model call, no reply.db row, no background task.

    Most replies are this case, so it has to cost nothing.
    """
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _draft("喜歡簡短")
    _schedule(writer=writer, remember_notes=())
    await _wait_for_inflight()
    assert count_raw_entries(scope=USER_SCOPE) == 0
    assert fake_client.responses.parse_models == []
    assert await get_job(scope=USER_SCOPE) is None


async def test_pipeline_writes_a_forget_without_asking_a_model() -> None:
    """A forget needs no review: it stores nothing, it only names what should go.

    A stored fact has to exist first, because a forget is copied into the compartments the
    scope actually has: a scope with none has nothing to delete, and the request is dropped
    rather than kept around waiting for a compartment to appear.
    """
    write_fact(scope=USER_SCOPE, fact=_stored_fact(fact_id="a" * 16, text="使用者住在台中"))
    writer, fake_client = _writer()
    # The consolidation the forget forces fails, so `raw.md` still holds it for the reads below.
    fake_client.responses.raises = RuntimeError("consolidation is down")
    _schedule(writer=writer, remember_notes=(), forget_notes=("使用者已經不住台中了",))
    await _wait_for_inflight()
    raw_text = read_raw_entries(scope=USER_SCOPE)
    assert "### forget_request" in raw_text
    assert "使用者已經不住台中了" in raw_text
    assert "- source: guild 42" in raw_text


async def test_pipeline_reports_private_observations_as_a_count() -> None:
    """What gets named under the reply is what is safe to repeat in that channel later.

    Showing the content is the point of the report: it is what lets someone correct a memory
    the bot got wrong, on the spot. A `source_only` observation is by definition one that
    should not be repeated outside the conversation it came from, and the note under a reply
    outlives the exchange in the channel, so those are counted rather than quoted.
    """
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = RawMemoryDraft(
        has_signal=True,
        observations=(
            _observation(
                summary="偏好繁體中文", normalized_key="preference.lang", sharing="global"
            ),
            _observation(
                summary="正在跟人吵架", normalized_key="recent.fight", sharing="source_only"
            ),
        ),
    )
    reported, record = _report_recorder()
    _schedule(writer=writer, report=record)
    await _wait_for_inflight()
    assert len(reported) == 1
    assert reported[0].remembered == ("偏好繁體中文",)
    assert reported[0].private == 1
    assert "正在跟人吵架" not in str(reported[0])


@pytest.mark.usefixtures("memory_isolated_dir")
@pytest.mark.parametrize(
    "outcome", ["kept-nothing", "review-failed", "cleared-mid-flight", "raised"]
)
async def test_a_turn_that_records_nothing_still_answers_the_report(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """The reply is showing `正在整理記憶⋯`, so every way a turn can end has to take it back.

    Four of them record nothing, and a silent one leaves the promise standing over work that
    has finished. The guarantee is a `finally` in `_run_memory_update` rather than a report
    call per branch, because the branch that forgets to report is exactly the one nobody
    notices.
    """

    def _blow_up(**kwargs: object) -> None:
        """Stands in for a store write that fails after the review succeeded."""
        del kwargs
        raise RuntimeError("raw append blew up")

    writer, fake_client = _writer()
    if outcome == "review-failed":
        # The LLM call itself failing, which parks the row for the restart sweep. Not the same
        # as a review that returned nothing: that one is `kept-nothing`.
        fake_client.responses.raises = RuntimeError("the evaluator call blew up")
    elif outcome == "raised":
        fake_client.responses.output_parsed = _draft("喜歡簡短")
        monkeypatch.setattr(pipeline, "append_raw_entry", _blow_up)
    else:
        fake_client.responses.output_parsed = RawMemoryDraft(has_signal=False, observations=())
    reported, record = _report_recorder()
    _schedule(writer=writer, report=record)
    if outcome == "cleared-mid-flight":
        mark_cleared(scope=USER_SCOPE)
    # Tolerates the `raised` turn's exception, which the task keeps after its callback logs it.
    await drain_memory_turns(scopes=(USER_SCOPE,))

    assert len(reported) == 1
    assert reported[0] == MemoryWriteSummary()


@pytest.mark.usefixtures("memory_isolated_dir")
async def test_a_failed_review_still_reports_the_forget_it_already_wrote() -> None:
    """A forget is durable before the evaluator runs, so a failed review does not hide it.

    Its remembered half stays empty rather than being guessed at: the notes that would fill it
    are exactly what the call that failed was reviewing.

    The turn carries a remember note as well, and has to: `evaluate` short-circuits on an empty
    `notes` and never reaches the LLM, so a forget-only turn cannot fail this way at all.
    """
    writer, fake_client = _writer()
    fake_client.responses.raises = RuntimeError("the evaluator call blew up")
    reported, record = _report_recorder()
    _schedule(
        writer=writer,
        remember_notes=("他換了新桌機",),
        forget_notes=("別再提那台舊筆電",),
        report=record,
    )
    await _wait_for_inflight()

    assert len(reported) == 1
    assert reported[0].forgotten == ("別再提那台舊筆電",)
    assert reported[0].remembered == ()


async def test_a_failed_review_still_writes_the_forgets_of_later_rounds() -> None:
    """A merged turn whose first review fails still stages the forgets behind it.

    The row is kept for the restart retry, but a forget needs no model, and waiting for that
    retry would leave the bot repeating what the newer turn asked it to drop.
    """
    writer, fake_client = _writer()
    fake_client.responses.raises = RuntimeError("review is down")
    pipeline.resume_memory_update(
        scope=USER_SCOPE,
        subject=user_subject(user_id=USER_ID, guild_id=42),
        transcript=render_turn_payload(
            transcript="Alice (alice) [id: 123456789]: 哈囉",
            rounds=((("他住在台中",), ()), ((), ("他已經不住台中了",))),
        ),
        writer=writer,
        identity=IDENTITY,
        token=memory_db.new_token(),
        status="pending",
    )
    await _wait_for_inflight()

    staged = read_detail_tail(scope=USER_SCOPE, max_chars=100_000) + read_raw_entries(
        scope=USER_SCOPE
    )
    assert "他已經不住台中了" in staged


@pytest.mark.usefixtures("memory_isolated_dir")
async def test_a_correction_under_the_key_it_forgets_is_still_taken_down() -> None:
    """The new fact of a correction is staged behind its forget, not dropped as already known.

    Dropping it by its reused key left the forget to delete the old fact with nothing in its
    place, so the user lost both.
    """
    append_raw_entry(
        scope=USER_SCOPE,
        entry_text=render_memory_observations(
            observations=(_observation(summary="住在台中", normalized_key="fact.city"),),
            source="guild 42",
        ),
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(review=_draft("住在台南", normalized_key="fact.city"))
    reported, record = _report_recorder()
    _schedule(
        writer=writer,
        remember_notes=("使用者住在台南",),
        forget_notes=("使用者已經不住台中了",),
        report=record,
    )
    await _wait_for_inflight()

    assert [summary.remembered for summary in reported] == [("住在台南",)]
    # The forget forces a consolidation, which retires the batch to `detail.md` in order.
    staged = read_detail_tail(scope=USER_SCOPE, max_chars=100_000) + read_raw_entries(
        scope=USER_SCOPE
    )
    assert staged.index("### forget_request") < staged.index("住在台南")


async def test_pipeline_no_op_gate_writes_nothing() -> None:
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _no_signal()
    _schedule(writer=writer)
    await _wait_for_inflight()
    assert count_raw_entries(scope=USER_SCOPE) == 0
    assert raw_file_bytes(scope=USER_SCOPE) == 0


async def test_pipeline_aborts_write_after_clear() -> None:
    writer, fake_client = _writer()
    parse_started = asyncio.Event()
    release = asyncio.Event()

    async def slow_answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        del body, text_format
        parse_started.set()
        await release.wait()
        return _draft("不該被寫入")

    fake_client.responses.answer = slow_answer
    _schedule(writer=writer)
    await parse_started.wait()
    mark_cleared(scope=USER_SCOPE)
    release.set()
    await _wait_for_inflight()
    assert count_raw_entries(scope=USER_SCOPE) == 0


async def test_pipeline_background_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn that raises past the review frees its scope and still replays the turn behind it."""
    appends: list[str] = []

    def fail_the_first_append(scope: str, entry_text: str) -> None:
        """Fails the first turn after its review succeeded, and appends normally after that."""
        appends.append(scope)
        if len(appends) == 1:
            raise RuntimeError("raw append blew up")
        append_raw_entry(scope=scope, entry_text=entry_text)

    monkeypatch.setattr(pipeline, "append_raw_entry", fail_the_first_append)
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _draft("喜歡簡短")
    _schedule(writer=writer, full_reply="第一")
    # Deferred behind the first, which has not started yet.
    _schedule(writer=writer, full_reply="第二")
    # Returns only once the scope's slot is empty.
    await drain_memory_turns(scopes=(USER_SCOPE,))
    assert count_raw_entries(scope=USER_SCOPE) == 1


async def test_pipeline_success_marks_done_and_clears_transcript() -> None:
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _draft("喜歡簡短")
    _schedule(writer=writer)
    await _wait_for_inflight()
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "done"
    assert job.transcript is None


async def test_pipeline_review_failure_marks_failed_and_keeps_transcript() -> None:
    writer, fake_client = _writer()
    # `evaluate` returns None on an LLM error, which must park the row at failed.
    fake_client.responses.raises = RuntimeError("llm down")
    _schedule(writer=writer)
    await _wait_for_inflight()
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "failed"
    assert job.transcript is not None
    assert count_raw_entries(scope=USER_SCOPE) == 0


async def test_pipeline_no_signal_marks_done() -> None:
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _no_signal()
    _schedule(writer=writer)
    await _wait_for_inflight()
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "done"


async def test_resume_memory_update_reruns_failed_job() -> None:
    """A persisted failed row is re-run on restart and succeeds, notes included.

    The notes ride inside the stored `transcript` rather than in a column of their own, so a
    resumed row carries what the answer model marked without `memory_job` growing a field.
    """
    payload = render_turn_payload(
        transcript="Alice (alice) [id: 123456789]: 哈囉", rounds=((_NOTES, ()),)
    )
    await _stage_row(token=42, transcript=payload, subject=_SUBJECT, identity=IDENTITY)
    await memory_db.mark_failed(scope=USER_SCOPE, token=42, error="boom")
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _draft("喜歡簡短")
    pipeline.resume_memory_update(
        scope=USER_SCOPE,
        subject=_SUBJECT,
        transcript=payload,
        writer=writer,
        identity=IDENTITY,
        token=42,
        status="failed",
    )
    await _wait_for_inflight()
    assert count_raw_entries(scope=USER_SCOPE) == 1
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "done"


async def test_resume_of_a_row_predating_markers_writes_nothing() -> None:
    """A row staged by the old extraction pass carries a transcript and no notes.

    Nothing can be done with it: the pass that would have mined it is gone, and mining the
    transcript here is exactly the guessing this change removed. It closes quietly rather than
    parking forever as a failure the restart sweep keeps retrying.
    """
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _draft("喜歡簡短")
    pipeline.resume_memory_update(
        scope=USER_SCOPE,
        subject=f"target_user_id: {USER_ID}",
        transcript="Alice (alice) [id: 123456789]: 哈囉",
        writer=writer,
        identity=IDENTITY,
        token=42,
        status="pending",
    )
    await _wait_for_inflight()
    assert count_raw_entries(scope=USER_SCOPE) == 0
    assert fake_client.responses.parse_models == []
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "done"


def _member_alias(summary: str) -> MemoryObservation:
    """Builds one community-nickname observation for member 42."""
    return _observation(
        summary=summary,
        normalized_key="vocab.member_alias.42",
        category="stable_fact",
        evidence_kind="stable_fact",
        durability="permanent",
        evidence_quote="大家都叫他李董",
    )


async def test_a_members_new_nickname_is_staged_beside_the_one_already_waiting() -> None:
    """Every nickname of one member shares its key, so a new one must not read as already known.

    Dropping it left the member's `## 成員稱呼` row without the new nickname for good.
    """
    scope = server_scope(server_id=555)
    append_raw_entry(
        scope=scope,
        entry_text=render_memory_observations(
            observations=(_member_alias(summary="社群都叫 [id: 42] 李董"),), source=None
        ),
    )
    # Holds the consolidation back, so the staged batch stays readable in `raw.md`.
    consolidation._last_consolidation[scope] = time.monotonic()
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = RawMemoryDraft(
        has_signal=True,
        observations=(_member_alias(summary="社群都叫 [id: 42] 李董，最近也叫他老李"),),
    )
    _schedule(writer=writer, subject=server_subject(server_id=555), scope=scope)
    task = inflight._inflight_tasks.get(key=scope)
    assert task is not None
    await task
    assert "老李" in read_raw_entries(scope=scope)


async def test_pipeline_stamps_subject_source_into_raw_entries() -> None:
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _draft("喜歡簡短")
    _schedule(writer=writer, subject=user_subject(user_id=USER_ID, guild_id=123))
    await _wait_for_inflight()
    raw_text = read_raw_entries(scope=USER_SCOPE)
    assert "- source: guild 123" in raw_text
    assert "- sharing: global" in raw_text


async def test_pipeline_server_subject_renders_without_source_fields() -> None:
    """A server-flavor subject carries no source line, so its observations carry neither field."""
    scope = server_scope(server_id=555)
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _draft("喜歡簡短")
    _schedule(writer=writer, subject=server_subject(server_id=555), scope=scope)
    task = inflight._inflight_tasks.get(key=scope)
    assert task is not None
    await task
    raw_text = read_raw_entries(scope=scope)
    assert "喜歡簡短" in raw_text
    assert "- source:" not in raw_text
    assert "- sharing:" not in raw_text


async def test_clear_scope_memory_removes_every_tier(memory_isolated_dir: Path) -> None:
    """A clear has to take the files AND the staged turn, or the wipe partly returns."""
    _populate_every_tier()
    await _stage_row(subject=f"target_user_id: {USER_ID}", identity=IDENTITY)

    assert await pipeline.clear_scope_memory(scope=USER_SCOPE) is True

    assert _memory_text() == ""
    assert list_compartments(scope=USER_SCOPE) == []
    assert read_tone(scope=USER_SCOPE) == ""
    assert count_raw_entries(scope=USER_SCOPE) == 0
    assert read_detail_tail(scope=USER_SCOPE, max_chars=10_000) == ""
    assert not (memory_isolated_dir / str(USER_ID)).exists()
    # Nothing the restart sweep could resume, and reply.db retains no transcript.
    await assert_cleared_row(scope=USER_SCOPE)
    assert await memory_db.list_resumable(updated_before=database_now()) == []


async def test_clear_scope_memory_reports_nothing_to_clear() -> None:
    assert await pipeline.clear_scope_memory(scope=USER_SCOPE) is False
    await assert_cleared_row(scope=USER_SCOPE)
    # The permanent marker itself is not user memory, so a repeated clear stays empty.
    assert await pipeline.clear_scope_memory(scope=USER_SCOPE) is False


async def test_clear_scope_memory_removes_a_staged_turn_without_files() -> None:
    """A scope whose only trace is a staged transcript still has something to erase."""
    await _stage_row()
    assert await pipeline.clear_scope_memory(scope=USER_SCOPE) is True
    await assert_cleared_row(scope=USER_SCOPE)


async def test_clear_token_advances_past_the_largest_stored_token() -> None:
    stored_token = 4_000_000_000_000_000_000
    await _stage_row(token=stored_token)

    assert await pipeline.clear_scope_memory(scope=USER_SCOPE) is True

    assert (await assert_cleared_row(scope=USER_SCOPE)).token > stored_token


async def test_clear_scope_memory_drops_the_deferred_replay() -> None:
    """The deferred turn holds a pre-clear transcript in memory and in reply.db."""
    writer, fake_client = _writer()
    first_started = asyncio.Event()
    release = asyncio.Event()
    parse_calls = 0

    async def first_call_waits(body: str, text_format: type[BaseModel]) -> BaseModel:
        del body, text_format
        nonlocal parse_calls
        parse_calls += 1
        if parse_calls == 1:
            first_started.set()
            await release.wait()
        return _draft("不該被寫入")

    fake_client.responses.answer = first_call_waits
    for reply in ("一", "二"):
        _schedule(writer=writer, full_reply=reply)
        await first_started.wait()
    assert inflight._pending_updates.get(key=USER_SCOPE) is not None

    await pipeline.clear_scope_memory(scope=USER_SCOPE)

    assert inflight._pending_updates.get(key=USER_SCOPE) is None
    release.set()
    await _wait_for_inflight()
    await wait_for_persisted_writes()
    # Neither the in-flight turn nor the dropped replay may write anything back,
    # and the clear leaves only a scrubbed marker that restart cannot resume.
    # The unwrapped `get_job` is what makes that second claim mean something:
    # `safe_list_resumable` degrades a read failure to `[]`, so on its own it can
    # pass without having looked.
    assert count_raw_entries(scope=USER_SCOPE) == 0
    assert await pipeline.safe_list_resumable(updated_before=database_now()) == []
    await assert_cleared_row(scope=USER_SCOPE)


async def test_clear_completion_drops_a_turn_staged_during_its_db_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A during-clear turn cannot resume, while a turn after return belongs to the next lifetime."""
    clear_job_finished = asyncio.Event()
    release_clear = asyncio.Event()
    real_clear_job = memory_db.clear_job

    async def blocked_clear_job(scope: str, flavor: str, token: int) -> bool:
        removed = await real_clear_job(
            scope=scope, flavor=memory_db.cast_flavor(value=flavor), token=token
        )
        clear_job_finished.set()
        await release_clear.wait()
        return removed

    monkeypatch.setattr(memory_db, "clear_job", blocked_clear_job)
    clearing = asyncio.create_task(pipeline.clear_scope_memory(scope=USER_SCOPE))
    await clear_job_finished.wait()
    during_clear = asyncio.create_task(
        inflight.stage_turn(
            scope=USER_SCOPE,
            subject=f"target_user_id: {USER_ID}",
            transcript="清除尚未回傳",
            identity=IDENTITY,
            token=memory_db.new_token(),
            captured_at=time.monotonic(),
        )
    )
    await asyncio.sleep(0)
    assert during_clear.done() is False

    release_clear.set()
    assert await clearing is False
    await during_clear
    assert await memory_db.list_resumable(updated_before=database_now()) == []

    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _no_signal()
    _schedule(writer=writer, full_reply="清除已經回傳")
    await _wait_for_inflight()

    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "done"
    assert job.transcript is None


async def test_cancelled_clear_waiting_for_staging_lock_finishes_the_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller cancellation cannot leave a pre-clear transcript resumable after restart."""
    await _stage_row(
        transcript="secret transcript", subject="secret subject", identity="secret identity"
    )
    lock_held = asyncio.Event()
    release_lock = asyncio.Event()

    async def hold_staging_lock() -> None:
        async with inflight.staging_locks.hold(key=USER_SCOPE):
            lock_held.set()
            await release_lock.wait()

    holder = asyncio.create_task(hold_staging_lock())
    await lock_held.wait()
    clearing = asyncio.create_task(pipeline.clear_scope_memory(scope=USER_SCOPE))
    await asyncio.sleep(0)
    assert cleared_since(scope=USER_SCOPE, started_at=0.0) is True

    clearing.cancel()
    release_lock.set()
    await holder
    with pytest.raises(asyncio.CancelledError):
        await clearing

    monkeypatch.setattr("discordbot.services.memory.store._cleared_at", {})
    await assert_cleared_row(scope=USER_SCOPE)
    assert await memory_db.list_resumable(updated_before=database_now()) == []


def _hold_clear_job(monkeypatch: pytest.MonkeyPatch) -> tuple[asyncio.Event, asyncio.Event]:
    """Holds every `clear_job` call before it writes, until the returned release is set.

    Returns `(started, release)`: `started` is set once a call is waiting.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    real_clear_job = memory_db.clear_job

    async def blocked_clear_job(scope: str, flavor: str, token: int) -> bool:
        started.set()
        await release.wait()
        return await real_clear_job(
            scope=scope, flavor=memory_db.cast_flavor(value=flavor), token=token
        )

    monkeypatch.setattr(memory_db, "clear_job", blocked_clear_job)
    return started, release


async def test_cancelled_clear_waits_for_an_inflight_tombstone_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during `clear_job` still drains its durable privacy boundary."""
    await _stage_row(
        transcript="secret transcript", subject="secret subject", identity="secret identity"
    )
    clear_job_started, release_clear_job = _hold_clear_job(monkeypatch=monkeypatch)
    clearing = asyncio.create_task(pipeline.clear_scope_memory(scope=USER_SCOPE))
    await clear_job_started.wait()
    clearing.cancel()
    release_clear_job.set()
    with pytest.raises(asyncio.CancelledError):
        await clearing

    monkeypatch.setattr("discordbot.services.memory.store._cleared_at", {})
    await assert_cleared_row(scope=USER_SCOPE)
    assert await memory_db.list_resumable(updated_before=database_now()) == []


async def test_cancelled_clear_still_records_that_it_erased(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled clear erased as much as any other, so both its traces still land."""
    _populate_every_tier()
    commits: list[tuple[str, str]] = []

    def record_commit(scope: str, reason: str) -> None:
        commits.append((scope, reason))

    monkeypatch.setattr(pipeline, "memory_git", SimpleNamespace(enqueue=record_commit))
    infos = capture_logs(monkeypatch=monkeypatch, level="info")
    clear_job_started, release_clear_job = _hold_clear_job(monkeypatch=monkeypatch)
    clearing = asyncio.create_task(pipeline.clear_scope_memory(scope=USER_SCOPE))
    await clear_job_started.wait()
    clearing.cancel()
    release_clear_job.set()
    with pytest.raises(asyncio.CancelledError):
        await clearing

    assert not (memory_isolated_dir / str(USER_ID)).exists()
    assert [scope for scope, _ in commits] == [USER_SCOPE]
    recorded = [
        fields
        for _, fields in infos
        if fields.get("scope") == USER_SCOPE and "caller_cancelled" in fields
    ]
    assert len(recorded) == 1
    assert recorded[0]["removed_files"] is True
    assert recorded[0]["caller_cancelled"] is True


async def test_clear_keeps_the_files_when_the_tombstone_cannot_be_written() -> None:
    """A row newer than the clear stops the erase instead of outliving it.

    Reproduces the second writer the single-process token block rules out: the clear's
    range is already reserved when a higher token lands, so its own token comes out
    below the stored row and the guarded tombstone upsert would silently no-op.
    """
    _populate_every_tier()
    await _stage_row(
        token=memory_db.new_token(), transcript="這段逐字稿不可以比檔案活得久", identity=IDENTITY
    )
    await _stage_row(token=9_999_999, transcript="這段逐字稿不可以比檔案活得久", identity=IDENTITY)

    with pytest.raises(RuntimeError, match="newer than the clear"):
        await pipeline.clear_scope_memory(scope=USER_SCOPE)

    assert _memory_text() != ""
    assert read_tone(scope=USER_SCOPE) != ""
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "pending"
    assert job.transcript == "這段逐字稿不可以比檔案活得久"


async def test_cancelled_clear_propagates_a_critical_tombstone_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed critical write is observed rather than hidden behind cancellation."""
    clear_job_started = asyncio.Event()
    release_clear_job = asyncio.Event()

    async def failing_clear_job(scope: str, flavor: str, token: int) -> bool:
        del scope, flavor, token
        clear_job_started.set()
        await release_clear_job.wait()
        raise RuntimeError("reply.db unavailable")

    monkeypatch.setattr(memory_db, "clear_job", failing_clear_job)
    clearing = asyncio.create_task(pipeline.clear_scope_memory(scope=USER_SCOPE))
    await clear_job_started.wait()
    clearing.cancel()
    release_clear_job.set()

    with pytest.raises(RuntimeError, match="reply\\.db unavailable"):
        await clearing


async def test_a_row_write_starting_after_the_clear_never_lands() -> None:
    """A staging write that starts after the clear must not write the row at all.

    The clear stamps the scope before its first await, so a deferred turn's
    detached staging task always finds the stamp already set. Staging anyway
    would put the erased conversation back on disk just to retire it again, and
    leave its removal resting on the best-effort `mark_done`.
    """
    await pipeline.clear_scope_memory(scope=USER_SCOPE)
    await inflight.stage_turn(
        scope=USER_SCOPE,
        subject=f"target_user_id: {USER_ID}",
        transcript="清除前的對話",
        identity=IDENTITY,
        token=1,
        captured_at=time.monotonic() - 1,
    )

    await assert_cleared_row(scope=USER_SCOPE)


async def test_a_row_write_racing_a_committed_clear_keeps_the_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale write after clear must not depend on best-effort `mark_done`.

    This forces the clear's reply.db commit ahead of the delayed stale upsert.
    If staging tried to repair that race with `mark_done`, an outage there would
    leave the erased transcript resumable. The clear token itself must reject it.
    """
    captured_at = time.monotonic()
    write_started = asyncio.Event()
    release = asyncio.Event()
    real_upsert = memory_db.upsert_pending

    async def slow_upsert(  # noqa: PLR0913 -- mirrors the patched signature
        scope: str, flavor: str, subject: str, transcript: str, identity: str, token: int
    ) -> None:
        write_started.set()
        await release.wait()
        await real_upsert(
            scope=scope,
            flavor=memory_db.cast_flavor(value=flavor),
            subject=subject,
            transcript=transcript,
            identity=identity,
            token=token,
        )

    monkeypatch.setattr(memory_db, "upsert_pending", slow_upsert)
    staging = asyncio.create_task(
        inflight.stage_turn(
            scope=USER_SCOPE,
            subject=f"target_user_id: {USER_ID}",
            transcript="清除前的對話",
            identity=IDENTITY,
            token=1,
            captured_at=captured_at,
        )
    )
    await write_started.wait()
    clearing = asyncio.create_task(pipeline.clear_scope_memory(scope=USER_SCOPE))
    await asyncio.sleep(0)
    assert clearing.done() is False
    release.set()
    await staging
    assert await clearing is True

    # The delayed stale write cannot overwrite a durable clear tombstone.
    await assert_cleared_row(scope=USER_SCOPE)
    assert USER_SCOPE not in {
        row.scope for row in await memory_db.list_resumable(updated_before=database_now())
    }


async def test_clear_overwrites_a_staged_row_even_if_its_task_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clear commit survives process exit after a stale staging commit."""
    captured_at = time.monotonic()
    write_committed = asyncio.Event()
    never_release = asyncio.Event()
    real_upsert = memory_db.upsert_pending

    async def committed_upsert(  # noqa: PLR0913 -- mirrors the patched signature
        scope: str, flavor: str, subject: str, transcript: str, identity: str, token: int
    ) -> None:
        await real_upsert(
            scope=scope,
            flavor=memory_db.cast_flavor(value=flavor),
            subject=subject,
            transcript=transcript,
            identity=identity,
            token=token,
        )
        write_committed.set()
        await never_release.wait()

    monkeypatch.setattr(memory_db, "upsert_pending", committed_upsert)
    staging = asyncio.create_task(
        inflight.stage_turn(
            scope=USER_SCOPE,
            subject=f"target_user_id: {USER_ID}",
            transcript="清除前的對話",
            identity=IDENTITY,
            token=1,
            captured_at=captured_at,
        )
    )
    await write_committed.wait()

    clearing = asyncio.create_task(pipeline.clear_scope_memory(scope=USER_SCOPE))
    await asyncio.sleep(0)
    assert clearing.done() is False
    staging.cancel()
    with pytest.raises(asyncio.CancelledError):
        await staging
    assert await clearing is True

    # A new process has no monotonic clear stamp, so only reply.db can protect it.
    monkeypatch.setattr("discordbot.services.memory.store._cleared_at", {})
    await assert_cleared_row(scope=USER_SCOPE)
    assert await memory_db.list_resumable(updated_before=database_now()) == []


async def test_clear_file_failure_leaves_tombstone(monkeypatch: pytest.MonkeyPatch) -> None:
    _populate_every_tier()
    _fail_the_file_delete(monkeypatch=monkeypatch)
    with pytest.raises(PermissionError, match=r"tone\.md is read-only"):
        await pipeline.clear_scope_memory(scope=USER_SCOPE)

    await assert_cleared_row(scope=USER_SCOPE)
    assert "新記憶" in _memory_text()


async def test_memory_update_scheduled_before_a_clear_never_starts() -> None:
    """A turn captured just before the clear must abort, not race it by microseconds.

    The worker times itself from the enqueue, so a clear landing while the task is
    still queued is newer than the turn and wins.
    """
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _draft("不該被寫入")
    _schedule(writer=writer)
    # The task has not run a single step yet; the clear lands first.
    await pipeline.clear_scope_memory(scope=USER_SCOPE)
    await _wait_for_inflight()

    assert count_raw_entries(scope=USER_SCOPE) == 0
    # The aborted turn cannot replace the clear marker, and restart has nothing.
    await assert_cleared_row(scope=USER_SCOPE)


async def _restart(writer: MemoryWriterAI) -> None:
    """Resumes every persisted row the way the reply cog's restart sweep does, then drains."""
    for job in await memory_db.list_resumable(updated_before=database_now()):
        assert job.transcript is not None
        pipeline.resume_memory_update(
            scope=job.scope,
            subject=job.subject,
            transcript=job.transcript,
            writer=writer,
            identity=job.identity,
            token=job.token,
            status=job.status,
        )
    await drain_memory_turns(scopes=(USER_SCOPE,))


def _staged_forgets() -> int:
    """Counts the forget requests the test user's evidence holds, raw and detail together."""
    staged = read_detail_tail(scope=USER_SCOPE, max_chars=100_000) + read_raw_entries(
        scope=USER_SCOPE
    )
    return staged.count("### forget_request")


async def test_a_review_refused_on_its_retry_is_not_retried_again() -> None:
    """A review the provider refuses every time gets one restart retry, not one per start (#874).

    Its forget is filed once, by the attempt that failed: the retry files no second copy and
    forces no second forget pass, and after it no restart has anything left to resume.
    """
    writer, fake_client = _writer()
    answers = _answers()

    async def review_refused(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Refuses every note review, as a content filter does; every other call answers."""
        if text_format is RawMemoryDraft:
            return None
        return await answers(body=body, text_format=text_format)

    fake_client.responses.answer = review_refused
    # A stored fact gives the forget pass a compartment to call the model for.
    write_fact(scope=USER_SCOPE, fact=_stored_fact())
    _schedule(writer=writer, forget_notes=("別再提那台舊筆電",))
    await drain_memory_turns(scopes=(USER_SCOPE,))
    calls = len(fake_client.responses.parse_instructions)
    assert any("forget_request" in body for body in fake_client.responses.parse_bodies)

    await _restart(writer=writer)
    await _restart(writer=writer)

    assert fake_client.responses.parse_instructions[calls:] == [PHASE1_EVALUATOR_PROMPT]
    assert _staged_forgets() == 1
    assert await memory_db.list_resumable(updated_before=database_now()) == []


async def test_a_retry_files_a_later_rounds_forget_behind_the_notes_it_stages() -> None:
    """A retry that re-stages an older round's notes files the newer round's forget behind them.

    The attempt that failed filed both forgets ahead of anything the retry stages, so a newer
    turn's forget left at that copy could not reach what the older turn asked to remember.
    """
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(review=_draft("住在台中", normalized_key="fact.city"))
    pipeline.resume_memory_update(
        scope=USER_SCOPE,
        subject=_SUBJECT,
        transcript=render_turn_payload(
            transcript="Alice (alice) [id: 123456789]: 哈囉",
            rounds=((("他住在台中",), ()), ((), ("他已經不住台中了",))),
        ),
        writer=writer,
        identity=IDENTITY,
        token=memory_db.new_token(),
        status="failed",
    )
    await drain_memory_turns(scopes=(USER_SCOPE,))

    staged = read_detail_tail(scope=USER_SCOPE, max_chars=100_000) + read_raw_entries(
        scope=USER_SCOPE
    )
    assert staged.index("fact.city") < staged.index("### forget_request")


async def test_a_retry_refused_again_files_no_forget_a_second_time() -> None:
    """With nothing re-staged ahead of it, a later round's forget is left at its first copy.

    A second copy would reach nothing the first one does not, and would force a forget pass of
    its own on a retry that is closing anyway.
    """
    writer, fake_client = _writer()
    fake_client.responses.raises = RuntimeError("review refused")
    pipeline.resume_memory_update(
        scope=USER_SCOPE,
        subject=_SUBJECT,
        transcript=render_turn_payload(
            transcript="Alice (alice) [id: 123456789]: 哈囉",
            rounds=((("他住在台中",), ()), ((), ("他已經不住台中了",))),
        ),
        writer=writer,
        identity=IDENTITY,
        token=memory_db.new_token(),
        status="failed",
    )
    await drain_memory_turns(scopes=(USER_SCOPE,))

    assert fake_client.responses.parse_instructions == [PHASE1_EVALUATOR_PROMPT]
    assert _staged_forgets() == 0


async def test_a_retry_merged_into_a_waiting_turn_still_files_that_turns_forget() -> None:
    """A merged payload opens with the waiting turn's round, whose forget nothing has filed yet.

    Three turns, because that is what it takes to reach the merge: the first occupies the scope
    and the other two queue behind it under the same subject, the resumed row arriving last.
    """
    payload = render_turn_payload(
        transcript="Alice (alice) [id: 123456789]: 哈囉", rounds=((_NOTES, ()),)
    )
    # Left by the previous process, so its token is below every one this process mints.
    await _stage_row(token=42, transcript=payload, subject=_SUBJECT, identity=IDENTITY)
    await memory_db.mark_failed(scope=USER_SCOPE, token=42, error="evaluate failed")
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(review=_draft("喜歡簡短"))
    _schedule(writer=writer)
    _schedule(writer=writer, remember_notes=(), forget_notes=("別再提那台舊筆電",))
    pipeline.resume_memory_update(
        scope=USER_SCOPE,
        subject=_SUBJECT,
        transcript=payload,
        writer=writer,
        identity=IDENTITY,
        token=42,
        status="failed",
    )
    await drain_memory_turns(scopes=(USER_SCOPE,))

    assert _staged_forgets() == 1
