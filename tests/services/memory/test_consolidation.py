"""Tests for consolidation: when it runs, how a batch fans out, and the forget passes."""

import re
import time
from typing import Any, Self
import asyncio
from pathlib import Path
from datetime import UTC, tzinfo, datetime, timedelta
from itertools import count
from collections import Counter

import pytest
from pydantic import BaseModel

from discordbot.services.memory import pipeline, consolidation
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    read_facts,
    write_fact,
    write_tone,
    mark_cleared,
    append_detail,
    append_raw_entry,
    read_detail_tail,
    read_raw_entries,
    count_raw_entries,
    list_compartments,
)
from discordbot.services.memory.deltas import DeltaOutcome
from discordbot.services.memory.writer import ToneForget, RawMemoryDraft, ConsolidatedMemory
from discordbot.services.memory.prompts import PHASE2_COMPACTION_BLOCK
from discordbot.services.memory.constants import (
    COMPACTION_TRIGGER_CHARS,
    MEMORY_CONSOLIDATION_COOLDOWN_SECONDS,
)
from discordbot.services.memory.raw_entries import (
    render_forget_requests,
    render_memory_observations,
)

from tests.helpers.memory import MemoryAnswer, make_delta
from tests.helpers.user_memory import (
    _PET,
    _SIX,
    _BIG5,
    _CITY,
    USER_ID,
    IDENTITY,
    USER_SCOPE,
    _GUILD_222,
    _TONE_NOTE,
    DETAIL_EVIDENCE,
    _GUILD_222_NOTE,
    TEST_MEMORY_MODEL,
    _OLDEST_DETAIL_HEADER,
    _draft,
    _entry,
    _writer,
    _answers,
    _schedule,
    _stage_raw,
    _memory_text,
    _observation,
    _stored_fact,
    _consolidated,
    _forget_entry,
    _consolidate_at,
    _forget_all_answer,
    _wait_for_inflight,
    _consolidate_forced,
    _consolidation_stage,
    _stage_raw_observation,
)


async def test_neither_a_forget_nor_an_observation_call_is_told_to_compact() -> None:
    """Compaction is a call of its own; the passes that carry evidence never get its block.

    A forget call told to compact produced rewrites `apply_deltas` throws away, and an
    observation call told to compact a large compartment beside the detail log spent its whole
    output budget reasoning and failed on every run.
    """
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(fact_id="a" * 16, text="住" * (COMPACTION_TRIGGER_CHARS + 1)),
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(
        review=_draft("希望被叫阿明", normalized_key="preference.name")
    )
    _schedule(
        writer=writer,
        remember_notes=("使用者希望被叫阿明",),
        forget_notes=("使用者已經不住台中了",),
    )
    await _wait_for_inflight()

    calls = zip(
        fake_client.responses.parse_instructions, fake_client.responses.parse_bodies, strict=True
    )
    consolidations = [(prompt, body) for prompt, body in calls if "<raw_entries>" in body]
    forget_prompts = [prompt for prompt, body in consolidations if "forget_request" in body]
    observation_prompts = [
        prompt for prompt, body in consolidations if "### stable_preference" in body
    ]
    assert forget_prompts, "the forget reached consolidation"
    assert observation_prompts, "the observation reached consolidation"
    assert all(PHASE2_COMPACTION_BLOCK not in prompt for prompt in forget_prompts)
    assert all(PHASE2_COMPACTION_BLOCK not in prompt for prompt in observation_prompts)


async def test_a_forget_never_shares_a_consolidation_call_with_an_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mixed turn is the one that would quietly turn the guarantee into a prompt rule.

    "Forget I live in Taichung, and call me 阿明" writes both a forget request and a
    `sharing="global"` observation into the same batch. Handed to one call, the only thing
    left stopping the model from filing the forget's own sentence as a `global` fact would be
    a line in the prompt, and that sentence was copied into `global` precisely because it
    could not reach the fact any other way. So the forget gets its own call, applied with
    `deletes_only`, and the observation gets a separate one.
    """
    write_fact(scope=USER_SCOPE, fact=_stored_fact(fact_id="a" * 16, text="使用者住在台中"))
    writer, fake_client = _writer()
    forget_pass_deltas: list[bool] = []
    real_apply = consolidation.apply_deltas

    def recording_apply(**kwargs: Any) -> DeltaOutcome:  # noqa: ANN401 -- a pass-through of the real signature
        """Records whether each applied batch was gated to deletions."""
        forget_pass_deltas.append(bool(kwargs.get("deletes_only", False)))
        return real_apply(**kwargs)

    monkeypatch.setattr("discordbot.services.memory.consolidation.apply_deltas", recording_apply)
    fake_client.responses.answer = _answers(
        review=_draft("希望被叫阿明", normalized_key="preference.name")
    )
    _schedule(
        writer=writer,
        remember_notes=("使用者希望被叫阿明",),
        forget_notes=("使用者已經不住台中了",),
    )
    await _wait_for_inflight()

    consolidations = [
        body for body in fake_client.responses.parse_bodies if "<raw_entries>" in body
    ]
    forget_calls = [body for body in consolidations if "forget_request" in body]
    observation_calls = [body for body in consolidations if "### stable_preference" in body]
    assert forget_calls, "the forget reached consolidation"
    assert observation_calls, "the observation reached consolidation"
    # No call mixes the two, in either direction.
    assert all("### stable_preference" not in body for body in forget_calls)
    assert all("forget_request" not in body for body in observation_calls)
    # The forget's call could only delete; the observation's could write.
    assert Counter(forget_pass_deltas) == Counter({
        True: len(forget_calls),
        False: len(observation_calls),
    })


async def test_a_forget_takes_the_evidence_of_the_fact_it_deleted() -> None:
    """Deleting the fact is not enough while its evidence stays readable.

    Every later consolidation reads `detail.md` back as `<recent_detail>` with the forget request
    stripped out, so the model would be handed the forgotten evidence with nothing saying it was
    forgotten; a restatement still pending in `raw.md` would be consolidated right back. The
    forget pass takes both out before anything reads them again, and the request itself still
    retires to `detail.md` for a rebuild to replay.
    """
    city = _observation(summary="住在台中", normalized_key="fact.city")
    food = _observation(summary="愛吃拉麵", normalized_key="fact.food")
    pet = _observation(summary="養了一隻貓", normalized_key="fact.pet")
    append_detail(scope=USER_SCOPE, text=_entry("2026-09-01T00:00:00+00:00", city, food))
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(fact_id="a" * 16, text="使用者住在台中", keys=("fact.city",)),
    )
    _stage_raw(
        _entry("2026-09-02T00:00:00+00:00", city),
        _forget_entry("2026-09-03T00:00:00+00:00"),
        _entry("2026-09-04T00:00:00+00:00", pet),
    )
    writer, fake_client = _writer()
    # The forget pass deletes the city fact; nothing else changes anywhere.
    fake_client.responses.answer = _answers(
        forget=ConsolidatedMemory(deltas=(make_delta(action="delete", fact_id="a" * 16),))
    )
    await _consolidate_forced(writer=writer)

    assert read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT) == []
    observation_calls = [body for body in fake_client.responses.parse_bodies if "fact.pet" in body]
    assert observation_calls, "the pending observation reached consolidation"
    assert all("fact.city" not in body for body in observation_calls)
    # The rest of the evidence is still handed over as before.
    assert all("fact.food" in body for body in observation_calls)
    detail = read_detail_tail(scope=USER_SCOPE, max_chars=100_000)
    assert "fact.city" not in detail
    assert "fact.food" in detail
    assert "fact.pet" in detail
    assert "### forget_request" in detail
    assert read_raw_entries(scope=USER_SCOPE) == ""


async def test_a_forget_reaches_what_was_staged_before_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A "remember X" then "forget X" inside one consolidation window must end with no X (#731).

    One staged entry does not consolidate on its own, so the forget's forced run is the first to
    see the observation. Run after the forget, it stored exactly what the user had just asked to
    drop, and on a scope with no compartment yet the forget was not even copied anywhere. What
    precedes the forget is consolidated first instead, and only once: the pet it also carried is
    kept, and no later pass reads it again. The two are staged back to back, as a deferred turn
    lands right behind the one before it, so whole-second stamps would tie them and leave the
    city's evidence behind.
    """
    ticks = count()

    class TickingClock(datetime):
        """Stamps each entry a microsecond after the last, whatever the host clock's resolution."""

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Self:
            return cls(2026, 9, 1, tzinfo=tz) + timedelta(microseconds=next(ticks))

    monkeypatch.setattr("discordbot.services.memory.raw_entries.datetime", TickingClock)
    append_raw_entry(
        scope=USER_SCOPE,
        entry_text=render_memory_observations(observations=(_CITY, _PET), source="guild 42"),
    )
    append_raw_entry(
        scope=USER_SCOPE,
        entry_text=render_forget_requests(notes=("使用者已經不住台中了",), source="guild 42"),
    )
    writer, fake_client = _writer()
    calls: list[str] = []
    fake_client.responses.answer = _consolidation_stage(calls=calls)
    await _consolidate_forced(writer=writer)

    # order-contract: what precedes a forget is consolidated before it, so the forget reaches it.
    assert calls == ["observe", "forget"]
    facts = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.summary for fact in facts] == [_PET.summary_zh]
    detail = read_detail_tail(scope=USER_SCOPE, max_chars=100_000)
    assert "fact.city" not in detail
    assert "fact.pet" in detail


async def test_a_restatement_after_the_forget_survives_it() -> None:
    """Something said again after the forget is the user changing their mind, not its target."""
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", _CITY),
        _forget_entry("2026-09-02T00:00:00+00:00"),
        _entry("2026-09-03T00:00:00+00:00", _CITY),
    )
    writer, fake_client = _writer()
    calls: list[str] = []
    fake_client.responses.answer = _consolidation_stage(calls=calls)
    await _consolidate_forced(writer=writer)

    # order-contract: the forget splits the batch, so a restatement after it is consolidated after it.
    assert calls == ["observe", "forget", "observe"]
    facts = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.summary for fact in facts] == [_CITY.summary_zh]
    detail = read_detail_tail(scope=USER_SCOPE, max_chars=100_000)
    # Only the restatement is left as evidence; what the forget was about went with the fact.
    assert "2026-09-01" not in detail
    assert "2026-09-03" in detail


async def test_each_forget_reaches_only_what_came_before_it() -> None:
    """A batch holding two forgets is taken one forget at a time.

    Only a forced run that failed leaves one behind for the next. Split at the newer forget
    alone, the restatement between the two was consolidated before the older one, which then
    deleted it and took its evidence along.
    """
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", _CITY),
        _forget_entry("2026-09-02T00:00:00+00:00"),
        _entry("2026-09-03T00:00:00+00:00", _CITY),
        _forget_entry("2026-09-04T00:00:00+00:00", note="使用者不想再提工作的事"),
    )
    writer, fake_client = _writer()
    calls: list[str] = []
    fake_client.responses.answer = _consolidation_stage(calls=calls)
    await _consolidate_forced(writer=writer)

    # order-contract: each forget splits the batch at its own stamp, one forget at a time.
    assert calls == ["observe", "forget", "observe", "forget"]
    facts = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.summary for fact in facts] == [_CITY.summary_zh]
    detail = read_detail_tail(scope=USER_SCOPE, max_chars=100_000)
    assert "2026-09-01" not in detail
    assert "2026-09-03" in detail


async def test_a_forget_naming_several_facts_is_applied_and_the_batch_moves_on() -> None:
    """A forget pass can only delete, so the mass-deletion ceiling does not bound it.

    Refused, the same forget was derived and refused again on every later run, holding back
    everything staged after it while the reply had already said it was forgotten.
    """
    for index, observation in enumerate(_SIX):
        write_fact(
            scope=USER_SCOPE,
            fact=_stored_fact(
                fact_id=f"{index:016x}", summary=observation.summary_zh, durability="permanent"
            ),
        )
    _stage_raw(
        _forget_entry("2026-09-02T00:00:00+00:00"), _entry("2026-09-03T00:00:00+00:00", _PET)
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _forget_all_answer(
        observe=_consolidated(text=_PET.summary_zh, summary=_PET.summary_zh, section="fact")
    )
    await _consolidate_forced(writer=writer)

    facts = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.summary for fact in facts] == [_PET.summary_zh]
    assert read_raw_entries(scope=USER_SCOPE) == ""


async def test_a_forget_still_runs_when_the_pass_before_it_fails() -> None:
    """A failed observation pass must not hold back any forget that follows it.

    The fact an earlier batch stored, or the tone note, would otherwise go on being injected
    until a retry that a stuck compartment may never let succeed, while the reply already said
    it was dropped. The observation passes after the failed one are skipped, and the batch is
    kept for the retry.
    """
    write_tone(scope=USER_SCOPE, content=_TONE_NOTE)
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(fact_id="a" * 16, summary=_CITY.summary_zh, keys=("fact.city",)),
    )
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", _PET),
        _forget_entry("2026-09-02T00:00:00+00:00", note="使用者不想再提工作的事"),
        _entry("2026-09-03T00:00:00+00:00", _PET),
        _forget_entry("2026-09-04T00:00:00+00:00"),
    )
    writer, fake_client = _writer()
    calls: list[str] = []
    stage = _consolidation_stage(calls=calls)

    async def failing_observation(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Fails every observation call and answers the forget passes as usual."""
        if text_format is ToneForget:
            calls.append("tone")
            return ToneForget()
        if "forget_request" in body:
            return await stage(body=body, text_format=text_format)
        calls.append("observe")
        return None

    fake_client.responses.answer = failing_observation
    await _consolidate_forced(writer=writer)

    # order-contract: the consolidation passes run one after another; the forgets following the failed pass are the behaviour under test.
    assert calls == ["observe", "forget", "tone", "forget", "tone"]
    assert read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT) == []
    assert count_raw_entries(scope=USER_SCOPE) == 4


def _recording(calls: list[str], observe: ConsolidatedMemory | None = None) -> MemoryAnswer:
    """Answers every memory call with no change, recording each as `observe`, `forget`,
    `compact` or `tone` in `calls`; compaction fails, and an observation pass answers `observe`,
    failing when it is None.
    """

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Records the call's kind and answers it."""
        if text_format is ToneForget:
            return ToneForget()
        if "<tone_evidence>" in body:
            calls.append("tone")
            return ConsolidatedMemory()
        if "forget_request" in body:
            calls.append("forget")
            return ConsolidatedMemory()
        if "net fact removals allowed" in body:
            calls.append("compact")
            return None
        calls.append("observe")
        return observe

    return answer


def _FixedDay(day: str) -> type[datetime]:  # noqa: N802 -- stands in for the `datetime` class
    """Returns a `datetime` whose `now` is midnight UTC on `day`."""

    class FixedDay(datetime):
        """Answers `now` with the fixed day."""

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Self:
            return cls.fromisoformat(f"{day}T00:00:00+00:00").astimezone(tz)

    return FixedDay


def _oversized_fact() -> None:
    """Stores one fact whose text alone is past the compaction trigger."""
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(fact_id="a" * 16, text="住" * (COMPACTION_TRIGGER_CHARS + 1)),
    )


async def test_compaction_waits_for_every_forget_in_the_batch() -> None:
    """Compaction merges facts, so it runs only once each forget the batch carries has.

    Run before a forget, it could fold the fact the forget names into others, leaving the
    forget to delete all of them or none.
    """
    _oversized_fact()
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", _PET),
        _forget_entry("2026-09-02T00:00:00+00:00", note="使用者不想再提工作的事"),
        _entry("2026-09-03T00:00:00+00:00", _CITY),
        _forget_entry("2026-09-04T00:00:00+00:00"),
    )
    writer, fake_client = _writer()
    calls: list[str] = []
    fake_client.responses.answer = _recording(calls=calls, observe=ConsolidatedMemory())
    await _consolidate_forced(writer=writer)

    # order-contract: compaction follows the last forget, and the tone call closes the run.
    assert calls == ["observe", "forget", "observe", "forget", "compact", "tone"]


async def test_a_failed_observation_pass_still_compacts_once_the_forgets_ran() -> None:
    """A compartment too large for its own consolidation must shrink before the retry.

    That pass failing kept the batch, the forget it carried kept compaction off, and the same
    call failed again on every later run while the memory stopped moving.
    """
    _oversized_fact()
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", _PET),
        _forget_entry("2026-09-02T00:00:00+00:00", note="使用者不想再提工作的事"),
    )
    writer, fake_client = _writer()
    calls: list[str] = []
    fake_client.responses.answer = _recording(calls=calls)
    await _consolidate_forced(writer=writer)

    # order-contract: the forget runs past the failed pass, then compaction.
    assert calls == ["observe", "forget", "compact"]
    assert count_raw_entries(scope=USER_SCOPE) == 2


async def test_compaction_is_a_call_of_its_own_carrying_no_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shown the detail log beside a large compartment, the call reasoned until it ran out.

    It gets the facts, today's date for grading by age, and the net removal ceiling
    `apply_deltas` holds it to, and its result replaces the oversized fact.
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    _oversized_fact()
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    writer, fake_client = _writer()
    review = _answers(review=_draft("訊號"))

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Rewrites the oversized fact on the compaction pass alone."""
        if "net fact removals allowed" in body:
            return ConsolidatedMemory(
                deltas=(make_delta(action="update", fact_id="a" * 16, text="壓縮後"),)
            )
        return await review(body=body, text_format=text_format)

    fake_client.responses.answer = answer
    _schedule(writer=writer)
    await _wait_for_inflight()

    calls = zip(
        fake_client.responses.parse_instructions, fake_client.responses.parse_bodies, strict=True
    )
    compactions = [(prompt, body) for prompt, body in calls if PHASE2_COMPACTION_BLOCK in prompt]
    assert len(compactions) == 1
    body = compactions[0][1]
    assert "住住住" in body
    assert "<raw_entries>\n(empty)" in body
    assert "<recent_detail>\n(empty)" in body
    assert re.search(r"today: \d{4}-\d{2}-\d{2}", body) is not None
    assert "net fact removals allowed: 3" in body
    assert "壓縮後" in _memory_text()


async def test_a_failed_compaction_is_tried_again_only_the_next_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compartment over the trigger must not pay for a doomed compaction on every run."""
    _oversized_fact()
    writer, fake_client = _writer()
    calls: list[str] = []
    fake_client.responses.answer = _recording(calls=calls)
    for day in ("2026-09-02", "2026-09-02", "2026-09-03"):
        monkeypatch.setattr("discordbot.services.memory.run.datetime", _FixedDay(day))
        _stage_raw(_forget_entry(f"{day}T00:00:00+00:00"))
        await _consolidate_forced(writer=writer)

    assert calls.count("compact") == 2, "once on the first day, once on the next"


async def test_a_failed_observation_call_is_retried_on_less_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same prompt failing again would hold the batch back forever.

    The retry reads only the newest detail entries; a success restores the full window.
    """
    old = _observation(summary="很久以前的事", normalized_key="fact.old")
    new = _entry(
        "2026-09-01T00:00:00+00:00", _observation(summary="最近的事", normalized_key="fact.new")
    )
    append_detail(scope=USER_SCOPE, text=_entry("2026-01-01T00:00:00+00:00", old))
    append_detail(scope=USER_SCOPE, text=new)
    monkeypatch.setattr(consolidation, "MEMORY_DETAIL_RETRY_MAX_CHARS", len(new) + 10)
    writer, fake_client = _writer()
    calls: list[str] = []
    outcomes = iter([None, ConsolidatedMemory(), ConsolidatedMemory()])

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Fails the first observation call and answers the rest."""
        if "<tone_evidence>" in body:
            return ConsolidatedMemory()
        calls.append(body.split("<recent_detail>")[1])
        return next(outcomes)

    fake_client.responses.answer = answer
    for _ in range(3):
        _stage_raw(_entry("2026-09-02T00:00:00+00:00", _PET))
        await _consolidate_forced(writer=writer)

    first, retry, after = calls
    assert "fact.old" in first
    assert "fact.old" not in retry
    assert "fact.new" in retry
    assert "fact.old" in after


async def test_a_batch_too_long_for_one_run_is_retired_as_far_as_it_got(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restarting the whole batch after a timeout never got further than the first time.

    The forget segments that applied are retired, so the next run picks up where this stopped,
    and the segment cut off mid-way stays whole, forget included, for that run to redo.
    """
    monkeypatch.setattr(consolidation, "MEMORY_CONSOLIDATE_TIMEOUT_SECONDS", 0.5)
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", _PET),
        _forget_entry("2026-09-02T00:00:00+00:00", note="使用者不想再提工作的事"),
        _entry("2026-09-03T00:00:00+00:00", _CITY),
        _forget_entry("2026-09-04T00:00:00+00:00"),
    )
    writer, fake_client = _writer()
    stage = _consolidation_stage(calls=[])

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Hangs the second segment's observation pass past the fan-out timeout."""
        if "fact.city" in body and "forget_request" not in body and "<raw_entries>" in body:
            await asyncio.sleep(5)
        return await stage(body=body, text_format=text_format)

    fake_client.responses.answer = answer
    await _consolidate_forced(writer=writer)

    detail = read_detail_tail(scope=USER_SCOPE, max_chars=100_000)
    raw = read_raw_entries(scope=USER_SCOPE)
    assert "fact.pet" in detail
    assert "使用者不想再提工作的事" in detail
    assert "fact.pet" not in raw
    assert "fact.city" in raw
    assert "不住台中" in raw
    assert [
        fact.summary for fact in read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    ] == [_PET.summary_zh]


async def test_pipeline_consolidates_at_threshold(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=2)
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _draft("第一筆", normalized_key="preference.first")
    _schedule(writer=writer, full_reply="回覆一")
    await _wait_for_inflight()
    assert count_raw_entries(scope=USER_SCOPE) == 1

    fake_client.responses.answer = _answers(
        review=_draft("第二筆", normalized_key="preference.second"), facts=_consolidated()
    )
    _schedule(writer=writer, full_reply="回覆二")
    await _wait_for_inflight()
    assert "合併後" in _memory_text()
    # The fact is stamped with the scheduling identity, not written by the model.
    stored = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [(fact.owner_id, fact.owner_name) for fact in stored] == [(USER_ID, "Alice (alice)")]
    assert count_raw_entries(scope=USER_SCOPE) == 0
    # The consumed raw batch lands in the detail file, without author identity.
    detail_text = (memory_isolated_dir / str(USER_ID) / "detail.md").read_text(encoding="utf-8")
    assert "第一筆" in detail_text
    assert "第二筆" in detail_text
    assert IDENTITY not in detail_text


async def test_pipeline_keeps_raw_when_consolidation_fails(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    writer, fake_client = _writer()

    async def consolidation_down(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Reviews the note, then fails every consolidation call."""
        del body
        if text_format is RawMemoryDraft:
            return _draft("訊號")
        raise RuntimeError("consolidation down")

    fake_client.responses.answer = consolidation_down
    _schedule(writer=writer)
    await _wait_for_inflight()
    assert count_raw_entries(scope=USER_SCOPE) == 1
    assert _memory_text() == ""
    # Failure paths keep raw for retry and must not retire it as consumed.
    assert not (memory_isolated_dir / str(USER_ID) / "detail.md").exists()


async def test_pipeline_empty_delta_batch_still_clears_raw(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch that implies no change is applied, so it is consumed rather than replayed."""
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="既有內容"))
    writer, fake_client = _writer()

    fake_client.responses.answer = _answers(review=_draft("已知資訊"))
    _schedule(writer=writer)
    await _wait_for_inflight()
    assert "既有內容" in _memory_text()
    assert count_raw_entries(scope=USER_SCOPE) == 0
    # A genuine no-op still consumes the batch, so it lands in the detail file too.
    detail_text = (memory_isolated_dir / str(USER_ID) / "detail.md").read_text(encoding="utf-8")
    assert "已知資訊" in detail_text


async def test_pipeline_small_compartment_skips_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="小檔案"))
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(review=_draft("訊號"), facts=_consolidated())
    _schedule(writer=writer)
    await _wait_for_inflight()
    # order-contract: the compartment's call follows the note review and precedes the tone call.
    assert PHASE2_COMPACTION_BLOCK not in fake_client.responses.parse_instructions[1]


async def test_headers_alone_never_trigger_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server's alias rows render long headers over short bodies; only the bodies count."""
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    monkeypatch.setattr("discordbot.services.memory.consolidation.COMPACTION_TRIGGER_CHARS", 100)
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(text="短", keys=tuple(f"preference.key_{n}" for n in range(20))),
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(review=_draft("訊號"), facts=_consolidated())
    _schedule(writer=writer)
    await _wait_for_inflight()
    # order-contract: the compartment's call follows the note review and precedes the tone call.
    assert len(fake_client.responses.parse_bodies[1]) > 100, "the rendered facts pass the trigger"
    assert PHASE2_COMPACTION_BLOCK not in fake_client.responses.parse_instructions[1]


async def test_a_fact_past_its_date_ttl_is_neither_shown_nor_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shown, a fact the sweep is about to delete could trigger compaction or be merged back."""
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    monkeypatch.setattr("discordbot.services.memory.consolidation.COMPACTION_TRIGGER_CHARS", 100)
    write_fact(
        scope=USER_SCOPE, fact=_stored_fact(section="recent", durability="recent", text="舊" * 200)
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(review=_draft("訊號"), facts=_consolidated())
    _schedule(writer=writer)
    await _wait_for_inflight()
    # order-contract: the compartment's call follows the note review and precedes the tone call.
    assert "舊" not in fake_client.responses.parse_bodies[1]
    assert PHASE2_COMPACTION_BLOCK not in fake_client.responses.parse_instructions[1]


async def test_a_forget_is_still_shown_a_fact_past_its_date_ttl() -> None:
    """Only the forget's delete releases a fact's evidence; the sweep that would remove it does not."""
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(
            fact_id="e" * 16, section="recent", durability="recent", text="使用者正在搬家"
        ),
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers()
    _schedule(writer=writer, forget_notes=("使用者已經搬完家了",))
    await _wait_for_inflight()
    forget_calls = [
        body for body in fake_client.responses.parse_bodies if "forget_request" in body
    ]
    assert forget_calls, "the forget reached consolidation"
    assert all("e" * 16 in body for body in forget_calls)


def _stage_mixed_raw_batch() -> None:
    """Stages one raw batch whose two observations must end up in two different compartments."""
    _stage_raw_observation(
        summary="全域偏好", key="preference.global", sharing="global", source="guild 222"
    )
    _stage_raw_observation(
        summary="本群祕密", key="fact.secret", sharing="source_only", source="guild 222"
    )


async def test_consolidation_fans_one_batch_out_over_its_compartments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One batch, two directories: routing is per observation and neither call sees the other."""
    _consolidate_at(monkeypatch=monkeypatch, entries=2)
    _stage_mixed_raw_batch()
    writer, fake_client = _writer()

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Writes one fact per compartment, named after the compartment that wrote it."""
        del text_format
        written = "本群事實" if _GUILD_222_NOTE in body else "全域事實"
        return _consolidated(summary=written, text=written)

    fake_client.responses.answer = answer
    await consolidation.consolidate_if_needed(scope=USER_SCOPE, writer=writer, identity=IDENTITY)

    assert list_compartments(scope=USER_SCOPE) == [GLOBAL_COMPARTMENT, _GUILD_222]
    global_texts = [
        fact.text for fact in read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    ]
    guild_texts = [fact.text for fact in read_facts(scope=USER_SCOPE, compartment=_GUILD_222)]
    assert global_texts == ["全域事實"]
    assert guild_texts == ["本群事實"]
    # The global call is never shown the source_only evidence, so it cannot publish what
    # the flag confined: the partition runs before the model, not after it.
    [global_call] = [
        body for body in fake_client.responses.parse_bodies if _GUILD_222_NOTE not in body
    ]
    assert "全域偏好" in global_call
    assert "本群祕密" not in global_call
    assert count_raw_entries(scope=USER_SCOPE) == 0


async def test_a_failed_compartment_keeps_the_whole_raw_batch(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retiring a batch one compartment never read would lose that bucket's evidence for good.

    Replaying the compartment that did apply is safe (a delta is an upsert keyed on an id
    the model echoes back, then on the evidence keys), so the whole batch is kept.
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=2)
    _stage_mixed_raw_batch()
    writer, fake_client = _writer()

    async def guild_down(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Fails the guild compartment's call and answers the global one."""
        del text_format
        if _GUILD_222_NOTE in body:
            raise RuntimeError("consolidation down")
        return _consolidated(summary="全域事實", text="全域事實")

    fake_client.responses.answer = guild_down
    await consolidation.consolidate_if_needed(scope=USER_SCOPE, writer=writer, identity=IDENTITY)

    assert count_raw_entries(scope=USER_SCOPE) == 2
    assert not (memory_isolated_dir / str(USER_ID) / "detail.md").exists()
    assert [
        fact.text for fact in read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    ] == ["全域事實"]
    assert read_facts(scope=USER_SCOPE, compartment=_GUILD_222) == []


async def test_a_clear_during_a_compartment_call_writes_nothing_back(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clear finishing while a compartment's call is in flight keeps that call's deltas out.

    The clear never waits for `scope_lock`, so the answer arrives after the store is gone;
    applying it would recreate a fact from the conversation the user just erased.
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    _stage_raw_observation(
        summary="全域偏好", key="preference.global", sharing="global", source="guild 222"
    )
    writer, fake_client = _writer()
    cleared: list[bool] = []

    async def clear_mid_call(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Runs the user's clear to completion, then answers the call as if nothing happened."""
        del body, text_format
        cleared.append(await pipeline.clear_scope_memory(scope=USER_SCOPE))
        return _consolidated(summary="清除前的事實", text="清除前的事實")

    fake_client.responses.answer = clear_mid_call
    await consolidation.consolidate_if_needed(scope=USER_SCOPE, writer=writer, identity=IDENTITY)

    assert cleared == [True]
    assert not (memory_isolated_dir / str(USER_ID)).exists()


async def test_pipeline_cooldown_defers_entry_count_consolidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    consolidation._last_consolidation[USER_SCOPE] = time.monotonic()
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _draft("訊號")
    _schedule(writer=writer)
    await _wait_for_inflight()
    # Threshold is met but the cooldown has not elapsed: only the note review
    # ran and raw stays queued.
    assert count_raw_entries(scope=USER_SCOPE) == 1
    assert _memory_text() == ""
    assert fake_client.responses.parse_models == [TEST_MEMORY_MODEL.name]


async def test_pipeline_cooldown_elapsed_allows_consolidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    consolidation._last_consolidation[USER_SCOPE] = (
        time.monotonic() - MEMORY_CONSOLIDATION_COOLDOWN_SECONDS - 1
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(review=_draft("訊號"), facts=_consolidated())
    before = time.monotonic()
    _schedule(writer=writer)
    await _wait_for_inflight()
    assert "合併後" in _memory_text()
    # The attempt refreshed the per-user cooldown timestamp.
    assert consolidation._last_consolidation[USER_SCOPE] >= before


async def test_pipeline_byte_trigger_bypasses_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=99)
    monkeypatch.setattr("discordbot.services.memory.consolidation.RAW_CONSOLIDATION_MAX_BYTES", 10)
    consolidation._last_consolidation[USER_SCOPE] = time.monotonic()
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(
        review=_draft("超過位元組門檻的長訊號"), facts=_consolidated(text="爆量合併")
    )
    _schedule(writer=writer)
    await _wait_for_inflight()
    # The raw byte burst escape hatch consolidates despite the active cooldown.
    assert "爆量合併" in _memory_text()


async def test_pipeline_passes_recent_detail_to_consolidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    append_detail(scope=USER_SCOPE, text="## 2026-01-01T00:00:00+00:00\n舊的詳細證據")
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(review=_draft("訊號"), facts=_consolidated())
    _schedule(writer=writer)
    await _wait_for_inflight()
    # order-contract: the compartment's call follows the note review and precedes the tone call.
    consolidation_input = fake_client.responses.parse_bodies[1]
    assert "<recent_detail>" in consolidation_input
    assert "舊的詳細證據" in consolidation_input
    # Identity header suffixes never reach the consolidation LLM.
    assert IDENTITY not in consolidation_input


async def test_pipeline_clear_resets_consolidation_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    consolidation._last_consolidation[USER_SCOPE] = time.monotonic()
    # The clear lands after the recorded attempt, so the cooldown belonged to
    # the wiped memory and must not delay the fresh state's first consolidation.
    mark_cleared(scope=USER_SCOPE)
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(
        review=_draft("清除後的新訊號"), facts=_consolidated(text="全新整理")
    )
    _schedule(writer=writer)
    await _wait_for_inflight()
    assert "全新整理" in _memory_text()
    assert count_raw_entries(scope=USER_SCOPE) == 0


async def test_consolidate_if_needed_skips_under_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=5)
    append_raw_entry(scope=USER_SCOPE, entry_text="- 只有一筆")
    writer, fake_client = _writer()
    # A valid answer, so a consolidation that did run would change the store rather than fail.
    fake_client.responses.output_parsed = _consolidated(text="不該整理")
    await consolidation.consolidate_if_needed(scope=USER_SCOPE, writer=writer, identity=IDENTITY)
    # Below threshold: no consolidation, raw untouched.
    assert fake_client.responses.parse_models == []
    assert _memory_text() == ""
    assert count_raw_entries(scope=USER_SCOPE) == 1


async def test_an_undecodable_detail_file_keeps_its_scope_out_of_consolidation(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `detail.md` saved in another encoding stops its scope before the model sees any of it."""
    _consolidate_at(monkeypatch=monkeypatch, entries=2)
    append_raw_entry(scope=USER_SCOPE, entry_text="- 第一筆")
    append_raw_entry(scope=USER_SCOPE, entry_text="- 第二筆")
    detail = memory_isolated_dir / str(USER_ID) / "detail.md"
    big5 = _OLDEST_DETAIL_HEADER + _BIG5 + b"\n"
    detail.write_bytes(big5)
    writer, fake_client = _writer()
    # A valid answer, so a consolidation that did run would change the store rather than fail.
    fake_client.responses.output_parsed = _consolidated(text="不該整理")
    await consolidation.consolidate_if_needed(scope=USER_SCOPE, writer=writer, identity=IDENTITY)
    assert fake_client.responses.parse_models == []
    assert count_raw_entries(scope=USER_SCOPE) == 2
    # Left exactly as it was saved, for the operator to repair.
    assert detail.read_bytes() == big5


def test_needs_consolidation_reflects_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=2)
    assert consolidation.needs_consolidation(scope=USER_SCOPE) is False
    append_raw_entry(scope=USER_SCOPE, entry_text="- 第一筆")
    append_raw_entry(scope=USER_SCOPE, entry_text="- 第二筆")
    assert consolidation.needs_consolidation(scope=USER_SCOPE) is True


@pytest.mark.usefixtures("memory_isolated_dir")
async def test_a_restated_fact_is_confirmed_again_when_consolidation_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fact the user keeps repeating must not age out behind a fresher one.

    The restatement comes from the same server as evidence already retired to `detail.md`, and
    consolidation emits nothing for a batch that adds nothing, so only code can renew the date.
    """
    now = datetime.now(tz=UTC).replace(microsecond=0)
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(
            fact_id="a" * 16, keys=("preference.test",), last_confirmed=now - timedelta(days=60)
        ),
    )
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(
            fact_id="b" * 16,
            summary="職業",
            text="是工程師",
            keys=("fact.job",),
            last_confirmed=now - timedelta(days=1),
        ),
    )
    observation = render_memory_observations(
        observations=(_observation(summary="喜歡簡短", normalized_key="preference.test"),),
        source="guild 42",
    )
    append_detail(scope=USER_SCOPE, text=f"## 2026-01-01T00:00:00.000000+00:00\n{observation}")
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(
        review=_draft("喜歡簡短", normalized_key="preference.test")
    )
    _schedule(writer=writer)
    await _wait_for_inflight()

    confirmed = {
        fact.fact_id: fact.last_confirmed >= now
        for fact in read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    }
    # The restated fact is renewed and kept; the one nobody mentioned keeps its date.
    assert confirmed == {"a" * 16: True, "b" * 16: False}


@pytest.mark.usefixtures("memory_isolated_dir")
async def test_a_consolidation_dates_what_it_writes_by_when_it_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unlike a rebuild's, a consolidation's batch is a real confirmation, so old evidence behind
    a fact does not date it.
    """
    now = datetime.now(tz=UTC).replace(microsecond=0)
    append_detail(
        scope=USER_SCOPE,
        text=_entry(
            "2026-01-01T00:00:00+00:00",
            _observation(summary="住在台中", normalized_key="fact.city"),
        ),
    )
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(
        review=_draft("住在台中", normalized_key="fact.city"),
        facts=ConsolidatedMemory(
            deltas=(
                make_delta(
                    section="fact",
                    summary="住在台中",
                    text="使用者住在台中",
                    from_keys=("fact.city",),
                ),
            )
        ),
    )
    _schedule(writer=writer)
    await _wait_for_inflight()

    [fact] = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert fact.created >= now
    assert fact.last_confirmed >= now
