"""Tests for in-flight memory turns: deferral, merging, replay and the concurrency cap."""

import time
import asyncio
from pathlib import Path
from functools import partial
import contextlib

import pytest
from pydantic import BaseModel

from discordbot.typings.memory import MemoryWriteSummary
from discordbot.services.memory import database as memory_db
from discordbot.services.memory import inflight, pipeline
from discordbot.services.memory.store import (
    user_scope,
    mark_cleared,
    read_detail_tail,
    read_raw_entries,
    count_raw_entries,
)
from discordbot.services.memory.writer import (
    ToneForget,
    RawMemoryDraft,
    user_subject,
    parse_turn_payload,
    render_turn_payload,
)

from tests.helpers.memory import get_job, drain_memory_turns, wait_for_persisted_writes
from tests.helpers.user_memory import (
    USER_ID,
    IDENTITY,
    USER_SCOPE,
    _draft,
    _writer,
    _schedule,
    _no_change,
    _no_signal,
    _stage_row,
    _observation,
    _consolidate_at,
    _report_recorder,
    _wait_for_inflight,
)


@pytest.mark.usefixtures("memory_isolated_dir")
async def test_a_superseded_turn_is_still_told_what_became_of_its_notes() -> None:
    """Merging two deferred turns' notes must merge their reports, not overwrite the older one.

    `_merged_payload` carries the superseded turn's notes into the payload that replaces it, so
    that reply is still waiting on an answer. Dropping its report the way the transcript is
    dropped would leave it saying `正在整理記憶⋯` for good.

    Three turns, because that is what it takes to reach the merge: the first occupies the scope
    and the other two queue behind it under the same subject, which is where one payload
    absorbs the other.
    """
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = RawMemoryDraft(
        has_signal=True,
        observations=(_observation(summary="偏好繁體中文", normalized_key="preference.lang"),),
    )
    seen = {key: _report_recorder() for key in ("first", "superseded", "newest")}
    for key, note in (
        ("first", "他喜歡繁體中文"),
        ("superseded", "他在台北工作"),
        ("newest", "他養了一隻貓"),
    ):
        _schedule(writer=writer, remember_notes=(note,), report=seen[key][1])
    await drain_memory_turns(scopes=(USER_SCOPE,))

    assert [len(reports) for reports, _ in seen.values()] == [1, 1, 1]


def test_a_merge_keeps_each_turns_notes_in_their_own_round() -> None:
    """Merged turns stay in order, and a round with no forget folds into the one before it.

    Folding costs nothing in order, since a round is staged forget first and the folded notes
    came after that forget anyway, and it saves a review.
    """
    older = render_turn_payload(transcript="舊", rounds=((("他住在台中",), ()),))
    newer = render_turn_payload(transcript="新", rounds=(((), ("他已經不住台中了",)),))
    transcript, rounds = parse_turn_payload(
        payload=inflight._merged_payload(newer=newer, older=older)
    )
    assert transcript == "新"
    assert rounds == ((("他住在台中",), ()), ((), ("他已經不住台中了",)))

    older = render_turn_payload(transcript="舊", rounds=((("他養貓",), ("別提舊筆電",)),))
    newer = render_turn_payload(transcript="新", rounds=((("他養貓", "他養狗"), ()),))
    _, rounds = parse_turn_payload(payload=inflight._merged_payload(newer=newer, older=older))
    assert rounds == ((("他養貓", "他養狗"), ("別提舊筆電",)),)


def test_a_repeated_note_keeps_its_newest_place_in_a_merge() -> None:
    """A note written twice counts where it was written last, on either side of a forget.

    Kept at its first place, a remember restated after a forget would be staged ahead of it and
    deleted, and a forget repeated after a remember would be staged ahead of it and miss it.
    """
    older = render_turn_payload(
        transcript="舊", rounds=((("他住在台中",), ()), ((), ("他已經不住台中了",)))
    )
    newer = render_turn_payload(transcript="新", rounds=((("他住在台中",), ()),))
    _, rounds = parse_turn_payload(payload=inflight._merged_payload(newer=newer, older=older))
    # The restatement follows the forget, which a round stages first.
    assert rounds == ((("他住在台中",), ("他已經不住台中了",)),)

    older = render_turn_payload(transcript="舊", rounds=(((), ("別提舊筆電",)), (("他養貓",), ())))
    newer = render_turn_payload(transcript="新", rounds=(((), ("別提舊筆電",)),))
    _, rounds = parse_turn_payload(payload=inflight._merged_payload(newer=newer, older=older))
    assert rounds == ((("他養貓",), ()), ((), ("別提舊筆電",)))


async def test_a_merged_forget_reaches_what_an_older_waiting_turn_remembered(
    memory_isolated_dir: Path,
) -> None:
    """A remember from an older waiting turn must land ahead of a newer turn's forget (#736).

    Merged into one list of each kind, the forget was staged first and the observation after
    it, where the forget can no longer reach it. Three turns, because that is what it takes to
    reach the merge: the first occupies the scope and the other two queue behind it.
    """
    writer, fake_client = _writer()

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Reviews each note into one observation, and changes no fact."""
        if text_format is ToneForget:
            return ToneForget()
        if text_format is RawMemoryDraft:
            key = "fact.city" if "台中" in body else "preference.lang"
            return _draft("住在台中", normalized_key=key)
        return _no_change()

    fake_client.responses.answer = answer
    for remember, forget in (
        (("他喜歡繁體中文",), ()),
        (("他住在台中",), ()),
        ((), ("他已經不住台中了",)),
    ):
        _schedule(writer=writer, remember_notes=remember, forget_notes=forget)
    await drain_memory_turns(scopes=(USER_SCOPE,))

    staged = read_detail_tail(scope=USER_SCOPE, max_chars=100_000) + read_raw_entries(
        scope=USER_SCOPE
    )
    assert staged.index("fact.city") < staged.index("### forget_request")


async def test_a_partly_failed_merge_still_reports_what_it_staged(
    memory_isolated_dir: Path,
) -> None:
    """A later round's failed review must not hide what an earlier round already took down.

    A resumed retry carries no report, so this is the only chance the replies get to hear it.
    """
    writer, fake_client = _writer()
    reviews: list[int] = []

    async def second_review_fails(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Reviews the first round, fails the second, changes no fact."""
        del body
        if text_format is ToneForget:
            return ToneForget()
        if text_format is RawMemoryDraft:
            reviews.append(1)
            if len(reviews) > 1:
                raise RuntimeError("review is down")
            return _draft("養了一隻貓", normalized_key="fact.pet")
        return _no_change()

    fake_client.responses.answer = second_review_fails
    reported, record = _report_recorder()
    inflight.enqueue_memory_update(
        turn=inflight.MemoryTurn(
            scope=USER_SCOPE,
            subject=user_subject(user_id=USER_ID, guild_id=42),
            transcript=render_turn_payload(
                transcript="Alice (alice) [id: 123456789]: 哈囉",
                rounds=((("他養了一隻貓",), ()), (("他住在台中",), ("別提舊筆電",))),
            ),
            writer=writer,
            identity=IDENTITY,
            token=memory_db.new_token(),
            report=record,
        ),
        run=pipeline._run_memory_update,
    )
    await _wait_for_inflight()

    assert len(reported) == 1
    assert reported[0].remembered == ("養了一隻貓",)
    assert reported[0].forgotten == ("別提舊筆電",)


async def test_a_merged_report_answers_the_newer_reply_when_the_older_one_raises() -> None:
    """One dead reply must not take the other's report down with it.

    The older reply is both the one still standing there promising work and the likelier of
    the two to have been deleted under it, so awaiting the two callbacks in sequence inside
    one handler left the newer reply saying `正在整理記憶⋯` for good.
    """
    seen: list[MemoryWriteSummary] = []

    async def older(summary: MemoryWriteSummary) -> None:
        """Stands in for a reply that has since been deleted."""
        del summary
        raise RuntimeError("the older reply is gone")

    async def newer(summary: MemoryWriteSummary) -> None:
        """Records what the surviving reply was told."""
        seen.append(summary)

    merged = inflight._merged_report(newer=newer, older=older, scope=USER_SCOPE)
    assert merged is not None
    await merged(MemoryWriteSummary(remembered=("偏好繁體中文",)))

    assert seen == [MemoryWriteSummary(remembered=("偏好繁體中文",))]


async def test_pipeline_defers_and_replays_newest_update_in_flight(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep this test about in-flight de-dupe only: the eager default threshold
    # would otherwise trigger consolidation on the replayed second entry.
    _consolidate_at(monkeypatch=monkeypatch, entries=10)
    writer, fake_client = _writer()
    started = asyncio.Event()
    release = asyncio.Event()
    seen_replies: list[str] = []

    async def slow_answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        del text_format
        seen_replies.append(body)
        started.set()
        if not release.is_set():
            await release.wait()
        return _draft(
            f"訊號 {len(seen_replies)}", normalized_key=f"preference.replay.{len(seen_replies)}"
        )

    fake_client.responses.answer = slow_answer
    # A two-line subject: the source line must round-trip through the deferred replay.
    subject = user_subject(user_id=USER_ID, guild_id=99)
    _schedule(writer=writer, subject=subject, full_reply="第一")
    await started.wait()
    first_task = inflight._inflight_tasks.get(key=USER_SCOPE)
    assert first_task is not None
    _schedule(writer=writer, subject=subject, full_reply="第二")
    _schedule(writer=writer, subject=subject, full_reply="第三")
    assert inflight._inflight_tasks.get(key=USER_SCOPE) is first_task
    release.set()
    await first_task
    # Only the newest skipped turn is replayed; its history already covers the
    # earlier skipped one.
    replay_task = inflight._inflight_tasks.get(key=USER_SCOPE)
    assert replay_task is not None
    await replay_task
    assert count_raw_entries(scope=USER_SCOPE) == 2
    assert any("第三" in reply for reply in seen_replies)
    assert not any("第二" in reply for reply in seen_replies)
    # Both the direct run and the replayed turn stamped the subject's source.
    assert read_raw_entries(scope=USER_SCOPE).count("- source: guild 99") == 2


async def test_pipeline_carries_a_skipped_turns_notes_into_the_replay(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replaying only the newest skipped turn must not silently drop an older turn's notes.

    For a transcript, replaying the newest is enough: its history window already contains the
    earlier skipped turns. A marker note is not in that window. It exists only in the reply
    that emitted it, so a user who says "remember X" and then "remember Y" while the first
    review is still running would lose X entirely, with nothing in the logs to say so.
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=10)
    writer, fake_client = _writer()
    started = asyncio.Event()
    release = asyncio.Event()
    seen_notes: list[str] = []

    async def slow_answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        del text_format
        seen_notes.append(body)
        started.set()
        if not release.is_set():
            await release.wait()
        return _no_signal()

    fake_client.responses.answer = slow_answer
    for note in ("記住 X", "記住 Y", "記住 Z"):
        _schedule(writer=writer, remember_notes=(note,))
        if note == "記住 X":
            await started.wait()
    first_task = inflight._inflight_tasks.get(key=USER_SCOPE)
    assert first_task is not None
    release.set()
    await first_task
    replay_task = inflight._inflight_tasks.get(key=USER_SCOPE)
    assert replay_task is not None
    await replay_task
    # The replay carries the note of the turn it superseded as well as its own.
    assert "記住 Y" in seen_notes[-1]
    assert "記住 Z" in seen_notes[-1]


async def test_pipeline_never_merges_notes_across_conversation_sources(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A note written in one guild must never be replayed under another guild's source stamp.

    The scope is guild-independent, so holding one pending turn per scope and keeping only the
    newest subject would file a `source_only` observation derived from guild A's note into
    `g/<B>`, readable by a server the speaker never said it in. Pending turns are therefore
    held per source and replayed one after another, each keeping its own subject.
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=10)
    writer, fake_client = _writer()
    started = asyncio.Event()
    release = asyncio.Event()
    requests: list[str] = []

    async def slow_answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        del text_format
        requests.append(body)
        started.set()
        if not release.is_set():
            await release.wait()
        return _no_signal()

    fake_client.responses.answer = slow_answer
    for guild, note in ((99, "在 99 說的"), (77, "在 77 說的"), (99, "也在 99 說的")):
        _schedule(
            writer=writer,
            remember_notes=(note,),
            subject=user_subject(user_id=USER_ID, guild_id=guild),
        )
        if guild == 99 and note == "在 99 說的":
            await started.wait()
    release.set()
    while (task := inflight._inflight_tasks.get(key=USER_SCOPE)) is not None:
        await task

    by_note = {
        note: request
        for note in ("在 99 說的", "在 77 說的", "也在 99 說的")
        for request in requests
        if note in request
    }
    assert len(by_note) == 3
    assert "source: guild 77" in by_note["在 77 說的"]
    assert "source: guild 99" in by_note["也在 99 說的"]
    # The two sources never share a request, in either direction.
    assert "在 77 說的" not in by_note["也在 99 說的"]
    assert "也在 99 說的" not in by_note["在 77 說的"]


async def test_pipeline_cancelled_task_does_not_raise_or_replay(memory_isolated_dir: Path) -> None:
    writer, fake_client = _writer()
    started = asyncio.Event()

    async def hang(body: str, text_format: type[BaseModel]) -> None:
        del body, text_format
        started.set()
        await asyncio.sleep(100)

    fake_client.responses.answer = hang
    _schedule(writer=writer, full_reply="一")
    await started.wait()
    task = inflight._inflight_tasks.get(key=USER_SCOPE)
    assert task is not None
    _schedule(writer=writer, full_reply="二")
    assert inflight._pending_updates.get(key=USER_SCOPE) is not None
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    # The callback must not raise, must clear the slot, and must not replay.
    assert inflight._inflight_tasks.get(key=USER_SCOPE) is None
    assert inflight._pending_updates.get(key=USER_SCOPE) is not None


async def test_pipeline_drops_pending_replay_after_clear(memory_isolated_dir: Path) -> None:
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
    _schedule(writer=writer, full_reply="一")
    await first_started.wait()
    # Queue a pending replay, then clear before the in-flight task finishes.
    _schedule(writer=writer, full_reply="二")
    assert inflight._pending_updates.get(key=USER_SCOPE) is not None
    mark_cleared(scope=USER_SCOPE)
    release.set()
    first_task = inflight._inflight_tasks.get(key=USER_SCOPE)
    assert first_task is not None
    await first_task
    # The pre-clear pending turn must not be replayed back into storage.
    assert inflight._inflight_tasks.get(key=USER_SCOPE) is None
    assert count_raw_entries(scope=USER_SCOPE) == 0


@pytest.mark.parametrize(
    "dm_after_the_clear", [False, True], ids=["dm-during-the-clear", "dm-after-the-clear"]
)
async def test_a_cleared_waiting_turn_does_not_end_the_replay(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch, dm_after_the_clear: bool
) -> None:
    """Dropping one cleared waiting turn must still settle every other source's turn (#872).

    Each replay is started by the previous one's done-callback, so a dropped turn that ended the
    walk left the next source waiting with nothing in flight, its reply at `正在整理記憶⋯`. A turn
    captured after the clear belongs to the new memory and still runs (#401).
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=10)
    writer, fake_client = _writer()
    started = asyncio.Event()
    release = asyncio.Event()

    async def first_call_waits(body: str, text_format: type[BaseModel]) -> BaseModel:
        del body, text_format
        started.set()
        await release.wait()
        return _draft("私訊說的")

    fake_client.responses.answer = first_call_waits
    _schedule(writer=writer)
    await started.wait()
    guild_reports, guild_report = _report_recorder()
    dm_reports, dm_report = _report_recorder()
    schedule_dm = partial(
        _schedule,
        writer=writer,
        subject=user_subject(user_id=USER_ID, guild_id=None),
        report=dm_report,
    )
    _schedule(writer=writer, report=guild_report)
    if not dm_after_the_clear:
        schedule_dm()
    # Stands in for the clear's closing stamp, which every turn deferred during the clear predates.
    mark_cleared(scope=USER_SCOPE)
    if dm_after_the_clear:
        schedule_dm()
    release.set()
    await drain_memory_turns(scopes=(USER_SCOPE,))
    await wait_for_persisted_writes()

    assert inflight._pending_updates.get(key=USER_SCOPE) is None
    assert guild_reports == [MemoryWriteSummary()]
    assert len(dm_reports) == 1
    assert bool(dm_reports[0].remembered) is dm_after_the_clear
    assert count_raw_entries(scope=USER_SCOPE) == int(dm_after_the_clear)


async def test_a_turn_waiting_from_during_a_clear_is_not_merged_into_a_later_one(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A waiting turn the clear's closing stamp postdates is dropped even when superseded.

    The newer turn from the same source then waits alone: neither its staged row nor its review
    carries the older turn's note, and the older reply is told nothing was recorded.
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=10)
    writer, fake_client = _writer()
    started = asyncio.Event()
    release = asyncio.Event()
    requests: list[str] = []

    async def first_call_waits(body: str, text_format: type[BaseModel]) -> BaseModel:
        del text_format
        requests.append(body)
        started.set()
        await release.wait()
        return _draft("清除之後說的")

    fake_client.responses.answer = first_call_waits
    _schedule(writer=writer)
    await started.wait()
    during_reports, during_report = _report_recorder()
    after_reports, after_report = _report_recorder()
    _schedule(writer=writer, remember_notes=("清除期間寫的",), report=during_report)
    # Stands in for the clear's closing stamp, which the turn deferred during the clear predates.
    mark_cleared(scope=USER_SCOPE)
    _schedule(writer=writer, remember_notes=("清除之後寫的",), report=after_report)
    await wait_for_persisted_writes()
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.transcript is not None
    assert "清除期間寫的" not in job.transcript
    release.set()
    await drain_memory_turns(scopes=(USER_SCOPE,))
    await wait_for_persisted_writes()

    assert not any("清除期間寫的" in request for request in requests)
    assert "清除之後寫的" in requests[-1]
    assert during_reports == [MemoryWriteSummary()]
    assert len(after_reports) == 1
    assert after_reports[0].remembered
    assert count_raw_entries(scope=USER_SCOPE) == 1


async def test_memory_semaphore_is_stable_within_a_loop(memory_isolated_dir: Path) -> None:
    assert inflight.memory_semaphore() is inflight.memory_semaphore()


def test_the_in_flight_registries_do_not_survive_an_event_loop_change() -> None:
    """A task belongs to the loop that made it, so neither registry may outlive its loop.

    `inflight.enqueue_memory_update` defers a turn whenever the scope's slot holds a task
    that is not `done()`, and `_finish_memory_update` replays a deferred one only from that task's
    own done-callback. An entry carried across a loop change is therefore either a task this
    loop can never see finish, parking the scope for good, or a queue of turns whose replay
    was wired to a loop that is gone.

    Being loop-local is what rules both out. Two real `asyncio.run` loops rather than the
    per-test one, because the rebuild is exactly what happens BETWEEN loops and a single test
    only ever sees one.
    """
    scope = user_scope(user_id=987654321)

    async def park() -> None:
        """Leaves a task and a deferred turn in the scope's slots on a loop about to close."""

        async def never() -> None:
            """Never finishes, so the slot it occupies would defer every later turn."""
            await asyncio.Event().wait()

        inflight._inflight_tasks.set(key=scope, value=asyncio.ensure_future(never()))
        inflight._pending_updates.set(key=scope, value={})

    async def read() -> tuple[object, object]:
        """Reads the same two slots from a second, unrelated loop."""
        return (inflight._inflight_tasks.get(key=scope), inflight._pending_updates.get(key=scope))

    asyncio.run(park())
    assert asyncio.run(read()) == (None, None)


async def test_memory_semaphore_caps_concurrent_updates(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("discordbot.services.memory.inflight.MEMORY_GLOBAL_CONCURRENCY", 1)
    writer, fake_client = _writer()
    in_flight = 0
    max_in_flight = 0

    async def tracking_answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        del body, text_format
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _no_signal()

    fake_client.responses.answer = tracking_answer
    scopes = [user_scope(user_id=USER_ID + offset) for offset in range(3)]
    for offset, scope in enumerate(scopes):
        _schedule(
            writer=writer, subject=user_subject(user_id=USER_ID + offset, guild_id=42), scope=scope
        )
    tasks = [
        task for scope in scopes if (task := inflight._inflight_tasks.get(key=scope)) is not None
    ]
    assert len(tasks) == len(scopes), "each scope started its own turn"
    await asyncio.gather(*tasks)
    # Three users started concurrently but the patched semaphore allows one
    # LLM call at a time.
    assert max_in_flight == 1


async def test_pipeline_cleared_deferred_turn_marks_job_done(memory_isolated_dir: Path) -> None:
    # A deferred (stashed) turn whose scope is cleared before replay must mark its
    # persisted row done, so a restart does not resume the cleared conversation.
    await _stage_row(
        token=7,
        transcript="Alice (alice) [id: 123456789]: 哈囉",
        subject=f"target_user_id: {USER_ID}",
        identity=IDENTITY,
    )
    captured_at = time.monotonic()
    writer, _ = _writer()
    subject = f"target_user_id: {USER_ID}"
    # Pending turns are held per conversation source, so the map is scope -> subject -> turn.
    inflight._pending_updates.set(
        key=USER_SCOPE,
        value={
            subject: inflight.MemoryTurn(
                scope=USER_SCOPE,
                subject=subject,
                transcript="Alice (alice) [id: 123456789]: 哈囉",
                writer=writer,
                identity=IDENTITY,
                captured_at=captured_at,
                token=7,
            )
        },
    )
    mark_cleared(scope=USER_SCOPE)
    done_task = asyncio.create_task(asyncio.sleep(0))
    await done_task
    inflight._finish_memory_update(
        scope=USER_SCOPE, task=done_task, run=pipeline._run_memory_update
    )
    await wait_for_persisted_writes()
    job = await get_job(scope=USER_SCOPE)
    assert job is not None
    assert job.status == "done"
    assert job.transcript is None
    assert count_raw_entries(scope=USER_SCOPE) == 0
