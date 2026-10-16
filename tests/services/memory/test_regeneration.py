"""Tests for rebuilding a scope's memory from its evidence."""

import time
from typing import Any
import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from discordbot.services.memory import regeneration
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    read_tone,
    read_facts,
    write_fact,
    write_tone,
    mark_cleared,
    server_scope,
    append_detail,
    append_raw_entry,
    read_detail_tail,
    count_raw_entries,
    guild_compartment,
    list_compartments,
    delete_memory_files,
)
from discordbot.services.memory.writer import ToneForget, MemoryWriterAI, ConsolidatedMemory

from tests.helpers.memory import MemoryAnswer, make_delta
from tests.helpers.user_memory import (
    _PET,
    _SIX,
    _CITY,
    _ROAST,
    _TERSE,
    USER_ID,
    IDENTITY,
    USER_SCOPE,
    _GUILD_222,
    _TONE_NOTE,
    DETAIL_EVIDENCE,
    _entry,
    _writer,
    _no_change,
    _stage_raw,
    _memory_text,
    _observation,
    _stored_fact,
    _consolidated,
    _forget_entry,
    _forget_four_answer,
    _consolidation_stage,
    _clearing_tone_answer,
    _stage_raw_observation,
    _stage_tone_observation,
    _server_tone_observation,
)


async def _regenerate(writer: MemoryWriterAI) -> regeneration.RegenerationReport:
    """Rebuilds the test user's memory from its evidence."""
    return await regeneration.regenerate_scope_memory(
        scope=USER_SCOPE, writer=writer, identity=IDENTITY
    )


async def test_a_rebuild_does_not_put_back_what_its_replayed_forget_took_out(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay prunes both files, so the rest of the rebuild must not use its earlier copies.

    Regeneration read `raw.md` and the detail tail before rebuilding. Retiring that raw copy
    would restore the evidence the replayed forget just removed, and the tone rebuild would
    still be handed it. The replayed forget runs under the same deletion-only gate as the
    incremental one, so the fact it also tries to write lands nowhere.
    """
    monkeypatch.setattr(
        "discordbot.services.memory.regeneration.MEMORY_REGENERATION_COOLDOWN_SECONDS", 0.0
    )
    _stage_raw(
        _entry(
            "2026-09-02T00:00:00+00:00",
            _observation(summary="住在台中", normalized_key="fact.city"),
        ),
        _forget_entry("2026-09-03T00:00:00+00:00"),
    )
    writer, fake_client = _writer()

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Rebuilds the city fact from evidence, then deletes it on the replayed forget."""
        del text_format
        if "forget_request" in body:
            rebuilt = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
            return ConsolidatedMemory(
                deltas=(
                    *(make_delta(action="delete", fact_id=fact.fact_id) for fact in rebuilt),
                    make_delta(section="fact", summary="使用者要求忘記住處", text="不住台中了"),
                )
            )
        if "<tone_evidence>" in body:
            return _no_change()
        return ConsolidatedMemory(
            deltas=(
                make_delta(
                    section="fact",
                    summary="住在台中",
                    text="使用者住在台中",
                    from_keys=("fact.city",),
                ),
            )
        )

    fake_client.responses.answer = answer
    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    assert read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT) == []
    detail = read_detail_tail(scope=USER_SCOPE, max_chars=100_000)
    assert "fact.city" not in detail
    assert "### forget_request" in detail
    assert not any(
        "<tone_evidence>" in body and "住在台中" in body
        for body in fake_client.responses.parse_bodies
    )


async def test_a_rebuild_keeps_a_fact_restated_after_its_forget(memory_isolated_dir: Path) -> None:
    """A rebuild takes the evidence one forget at a time, as consolidation took it (#870).

    Rebuilt from all of it and only then handed the forget, the city came back from the
    restatement and the forget deleted it again, on every later rebuild too.
    """
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", _CITY),
        _forget_entry("2026-09-02T00:00:00+00:00"),
        _entry("2026-09-03T00:00:00+00:00", _CITY),
    )
    writer, fake_client = _writer()
    calls: list[str] = []
    fake_client.responses.answer = _consolidation_stage(calls=calls)

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    # order-contract: the forget splits the rebuild, so a restatement after it is merged in after it.
    assert calls == ["observe", "forget", "observe"]
    facts = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.summary for fact in facts] == [_CITY.summary_zh]
    detail = read_detail_tail(scope=USER_SCOPE, max_chars=100_000)
    assert "2026-09-01" not in detail
    assert "2026-09-03" in detail


async def test_a_rebuild_takes_each_forget_against_only_what_came_before_it(
    memory_isolated_dir: Path,
) -> None:
    """A later forget about something else must not let an older one reach a restatement.

    Replayed together, the two forgets deleted the restated city, and the newer one's stamp
    took the restatement's evidence with it, leaving nothing to rebuild the city from (#870).
    """
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", _CITY),
        _forget_entry("2026-09-02T00:00:00+00:00"),
        _entry("2026-09-03T00:00:00+00:00", _CITY),
        _forget_entry("2026-09-04T00:00:00+00:00", note="使用者不想再提工作的事"),
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _consolidation_stage(calls=[])

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    detail = read_detail_tail(scope=USER_SCOPE, max_chars=100_000)
    assert "2026-09-01" not in detail
    assert "2026-09-03" in detail
    facts = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.summary for fact in facts] == [_CITY.summary_zh]


async def test_a_rebuild_hands_no_compartment_an_empty_corpus(memory_isolated_dir: Path) -> None:
    """A compartment whose evidence all follows a forget is emptied first without a call.

    The call could only answer with an empty batch: one more paid request, and one more way
    to fail the rebuild.
    """
    guild = guild_compartment(guild_id=42)
    city = _observation(
        summary=_CITY.summary_zh, normalized_key="fact.city", sharing="source_only"
    )
    write_fact(scope=USER_SCOPE, fact=_stored_fact(summary=_CITY.summary_zh, compartment=guild))
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", _PET),
        _forget_entry("2026-09-02T00:00:00+00:00", note="使用者不想再提工作的事"),
        _entry("2026-09-03T00:00:00+00:00", city),
    )
    writer, fake_client = _writer()
    calls: list[str] = []
    fake_client.responses.answer = _consolidation_stage(calls=calls)

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    # order-contract: the pet before the forget, the forget, then the city after it.
    assert calls == ["observe", "forget", "observe"]
    facts = read_facts(scope=USER_SCOPE, compartment=guild)
    assert [fact.summary for fact in facts] == [_CITY.summary_zh]


async def test_a_refused_pass_after_a_forget_fails_the_rebuild_and_puts_back_what_it_replaced(
    memory_isolated_dir: Path,
) -> None:
    """The pass merging what came after a forget keeps the mass-delete guard.

    It is shown the rebuilt facts but not the evidence behind them, so it may not wipe them. A
    refusal stops the run before it stands: the compartment gets back what it held before the
    run, and `raw.md` is kept.
    """
    write_fact(scope=USER_SCOPE, fact=_stored_fact(fact_id="a" * 16, summary="舊事實"))
    rebuilt = [
        _observation(summary=f"事實{index}", normalized_key=f"fact.k{index}") for index in range(5)
    ]
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", *rebuilt),
        _forget_entry("2026-09-02T00:00:00+00:00", note="使用者不想再提工作的事"),
        _entry(
            "2026-09-03T00:00:00+00:00", _observation(summary="新事實", normalized_key="fact.new")
        ),
    )
    writer, fake_client = _writer()

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Rebuilds the five facts, then deletes all of them on the pass after the forget."""
        if text_format is ToneForget:
            return ToneForget()
        if "forget_request" in body or "<tone_evidence>" in body:
            return _no_change()
        if "fact.new" in body:
            return ConsolidatedMemory(
                deltas=tuple(
                    make_delta(action="delete", fact_id=fact.fact_id)
                    for fact in read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
                )
            )
        return ConsolidatedMemory(
            deltas=tuple(
                make_delta(
                    section="fact",
                    summary=observation.summary_zh,
                    text=observation.summary_zh,
                    from_keys=(observation.normalized_key,),
                )
                for observation in rebuilt
            )
        )

    fake_client.responses.answer = answer
    report = await _regenerate(writer=writer)

    assert report.result == "failed"
    facts = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    assert [fact.summary for fact in facts] == ["舊事實"]
    assert count_raw_entries(scope=USER_SCOPE) == 3


async def test_a_rebuild_replays_a_forget_naming_several_facts(memory_isolated_dir: Path) -> None:
    """The rebuild's forget replay is a forget pass too, so the same forget completes it."""
    _stage_raw(
        _entry("2026-09-01T00:00:00+00:00", *_SIX), _forget_entry("2026-09-02T00:00:00+00:00")
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _forget_four_answer(
        observe=ConsolidatedMemory(
            deltas=tuple(
                make_delta(
                    section="fact",
                    summary=observation.summary_zh,
                    text=observation.summary_zh,
                    from_keys=(observation.normalized_key,),
                )
                for observation in _SIX
            )
        )
    )

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    assert len(read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)) == 2


_ROAST_FORGET = _forget_entry("2026-09-02T00:00:00+00:00", note="使用者不想再被粗口互嗆")


def _roast_forget_answer(rebuilt: bool = True) -> MemoryAnswer:
    """Builds an answer whose tone forget drops the roast and whose tone rebuild echoes its input.

    The tone forget drops every offered line naming the roast; the tone rebuild writes one note
    line per evidence bullet it was shown, oldest first, or fails when `rebuilt` is False.
    """

    def named(body: str, tag: str) -> tuple[int, ...]:
        """Returns the numbers of the lines inside `tag` that name the roast."""
        lines = body.split(f"<{tag}>")[1].split(f"</{tag}>", maxsplit=1)[0].splitlines()
        return tuple(int(line.split("]")[0].lstrip("[")) for line in lines if "粗口互嗆" in line)

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Answers the tone forget and the tone rebuild by what they were shown."""
        if text_format is ToneForget:
            return ToneForget(
                drop_lines=named(body=body, tag="tone_note"),
                drop_evidence=named(body=body, tag="tone_evidence"),
            )
        if "<tone_evidence>" not in body:
            return _no_change()
        if not rebuilt:
            return None
        evidence = body.split("<tone_evidence>")[1].split("</tone_evidence>", maxsplit=1)[0]
        bullets = [
            f"- {line.split('] ', maxsplit=1)[1]}" for line in evidence.strip().splitlines()
        ]
        return _no_change(tone="\n".join(["## 語氣偏好", *bullets]))

    return answer


@pytest.mark.parametrize(
    ("note", "detail", "raw", "expected"),
    [
        pytest.param(
            _TONE_NOTE,
            (_entry("2026-09-01T00:00:00+00:00", _ROAST, _TERSE),),
            (_ROAST_FORGET,),
            "## 語氣偏好\n- 回答要簡潔",
            id="tone-pass-never-ran",
        ),
        pytest.param(
            "## 語氣偏好\n- 回答要簡潔",
            (_entry("2026-09-01T00:00:00+00:00", _ROAST, _TERSE), _ROAST_FORGET),
            (),
            "## 語氣偏好\n- 回答要簡潔",
            id="tone-pass-missed-the-evidence",
        ),
        pytest.param(
            _TONE_NOTE,
            (_entry("2026-09-01T00:00:00+00:00", _ROAST, _TERSE),),
            (_ROAST_FORGET, _entry("2026-09-03T00:00:00+00:00", _ROAST)),
            "## 語氣偏好\n- 回答要簡潔\n- 喜歡被高強度粗口互嗆",
            id="restated-after-the-forget",
        ),
    ],
)
async def test_a_rebuild_replays_each_forget_over_the_tone_evidence(
    memory_isolated_dir: Path,
    note: str,
    detail: tuple[str, ...],
    raw: tuple[str, ...],
    expected: str,
) -> None:
    """A tone preference is never a fact, so the fact replay cannot keep it out of the note (#971).

    The tone rebuild writes the note from whatever evidence is left, so a forget whose tone pass
    failed, or missed the evidence, came back the moment the user ran `/memory regenerate`, and
    the retired forget was never applied to the tone tier again. A preference restated after
    the forget is still not its to take.
    """
    write_tone(scope=USER_SCOPE, content=note)
    append_detail(scope=USER_SCOPE, text="\n\n".join(detail))
    if raw:
        _stage_raw(*raw)
    writer, fake_client = _writer()
    fake_client.responses.answer = _roast_forget_answer()

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    assert read_tone(scope=USER_SCOPE) == expected


async def test_a_rebuild_failing_after_its_tone_replay_leaves_the_note_alone(
    memory_isolated_dir: Path,
) -> None:
    """The replay reaches only the tone evidence; the note is the rebuild's to rewrite.

    Its lines carry no stamp, so the older forget would be offered the roast the user restated
    after it, and with the rebuild failing past the replay the note would stay without it.
    """
    write_tone(scope=USER_SCOPE, content=_TONE_NOTE)
    append_detail(scope=USER_SCOPE, text=_entry("2026-09-01T00:00:00+00:00", _ROAST, _TERSE))
    _stage_raw(_ROAST_FORGET, _entry("2026-09-03T00:00:00+00:00", _ROAST))
    writer, fake_client = _writer()
    fake_client.responses.answer = _roast_forget_answer(rebuilt=False)

    report = await _regenerate(writer=writer)

    assert report.result == "failed"
    assert read_tone(scope=USER_SCOPE) == _TONE_NOTE


async def test_regenerate_scope_memory_rebuilds_from_evidence_only(
    memory_isolated_dir: Path,
) -> None:
    """The rebuild distils the cold-tier evidence alone; the stored facts never reach the model."""
    writer, fake_client = _writer()
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="舊的整理"))
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    append_raw_entry(scope=USER_SCOPE, entry_text="偏好訊號:\n- 喜歡簡短回覆")
    fake_client.responses.output_parsed = _consolidated(text="重建後的記憶")

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    # A rebuild REPLACES the compartment: it says a fact is gone by not re-emitting it,
    # so the previous generation must not survive alongside the new one.
    assert "重建後的記憶" in _memory_text()
    assert "舊的整理" not in _memory_text()
    # Dropping a readable fact is that ordinary replacement, not content destroyed unread.
    assert report.unreadable_removed == 0
    # The consumed raw batch retires into the cold tier like a consolidation.
    assert count_raw_entries(scope=USER_SCOPE) == 0
    assert "喜歡簡短回覆" in read_detail_tail(scope=USER_SCOPE, max_chars=10_000)
    # Pure-evidence rebuild: no existing facts are shown, compaction always applied.
    assert "COMPACTION" in fake_client.responses.parse_instructions[-1]
    user_text = fake_client.responses.parse_bodies[-1]
    assert "<existing_facts>\n(empty)\n</existing_facts>" in user_text
    assert "舊的整理" not in user_text
    assert "喜歡條列式" in user_text
    assert "喜歡簡短回覆" in user_text


async def test_regenerate_scope_memory_replaces_the_directory_not_only_what_it_could_read(
    memory_isolated_dir: Path,
) -> None:
    """A file no reader can parse must not outlive a rebuild that reports the scope replaced.

    `read_facts` skips it, so the snapshot the replace pass used to take there never saw
    it. A rebuild drops perfectly good facts by not re-emitting them, which makes the
    broken one surviving the odd case out. A file the store never wrote is a different
    thing: it stays where it is and is reported instead.
    """
    writer, fake_client = _writer()
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="舊的整理"))
    directory = memory_isolated_dir / str(USER_ID) / GLOBAL_COMPARTMENT
    broken = directory / f"{'b' * 16}.md"
    broken.write_text("hand-edited into nonsense\n", encoding="utf-8")
    stray = directory / "notes.md"
    stray.write_text("操作者自己放的筆記", encoding="utf-8")
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    fake_client.responses.output_parsed = _consolidated(text="重建後的記憶")

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    assert not broken.exists()
    assert stray.exists()
    assert "重建後的記憶" in _memory_text()
    # The stored fact it dropped for not being re-emitted parsed fine, so the count is
    # the broken file alone: what a rebuild destroys unread is the loss nothing else
    # reports, and one that removed nothing unread must not claim it did.
    assert report.unreadable_removed == 1


async def test_regenerate_scope_memory_never_calls_the_model_for_an_empty_compartment(
    memory_isolated_dir: Path,
) -> None:
    """A leftover directory with no evidence and no fact is pruned, not consolidated.

    `sweep_stale_facts` and an ordinary delta batch both delete through `delete_fact`,
    which leaves the directory it emptied behind, so this state arises on its own. The
    model would be handed an empty corpus and could only answer with an empty batch, and
    the answer it does not get is one more way to fail the compartments that do have
    something.
    """
    writer, fake_client = _writer()
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    leftover = memory_isolated_dir / str(USER_ID) / _GUILD_222
    leftover.mkdir(parents=True)
    fake_client.responses.output_parsed = _consolidated(text="重建後的記憶")

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    # One call, for the one compartment the evidence reached.
    assert len(fake_client.responses.parse_models) == 1
    # Removed, so it does not cost the same call again on the next rebuild.
    assert not leftover.exists()
    assert list_compartments(scope=USER_SCOPE) == [GLOBAL_COMPARTMENT]
    assert "重建後的記憶" in _memory_text()


async def test_regenerate_scope_memory_prunes_a_compartment_it_never_handed_to_the_model(
    memory_isolated_dir: Path,
) -> None:
    """Skipping the call keeps the replace pass's own rules about what may be removed.

    A compartment holding nothing a reader can parse has nothing to keep, so it is
    skipped — but a file the store never wrote is still not the store's to delete, and
    naming it is what makes the difference visible instead of assumed.
    """
    writer, fake_client = _writer()
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    leftover = memory_isolated_dir / str(USER_ID) / _GUILD_222
    leftover.mkdir(parents=True)
    broken = leftover / f"{'c' * 16}.md"
    broken.write_text("hand-edited into nonsense\n", encoding="utf-8")
    stray = leftover / "notes.md"
    stray.write_text("操作者自己放的筆記", encoding="utf-8")
    fake_client.responses.output_parsed = _consolidated(text="重建後的記憶")

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    assert len(fake_client.responses.parse_models) == 1
    assert not broken.exists()
    assert stray.read_text(encoding="utf-8") == "操作者自己放的筆記"
    # The skip path removes nothing BUT unreadable files — a compartment reaches it
    # precisely when nothing in it could be read — so it is the one that most needs to
    # say what it took.
    assert report.unreadable_removed == 1


async def test_regenerate_scope_memory_without_evidence_skips_llm(
    memory_isolated_dir: Path,
) -> None:
    writer, fake_client = _writer()
    # Stored facts alone are not evidence: the rebuild never reads them back in.
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="舊的整理"))

    report = await _regenerate(writer=writer)

    assert report.result == "no_evidence"
    assert fake_client.responses.parse_models == []
    assert "舊的整理" in _memory_text()
    # No LLM attempt happened, so the cooldown must stay untouched.
    assert regeneration.regeneration_on_cooldown(scope=USER_SCOPE) is False


def test_regeneration_has_evidence_tracks_raw_and_detail(memory_isolated_dir: Path) -> None:
    # Stored facts alone are not evidence; only raw or detail counts.
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="舊的整理"))
    assert regeneration.regeneration_has_evidence(scope=USER_SCOPE) is False
    append_raw_entry(scope=USER_SCOPE, entry_text="偏好訊號:\n- 喜歡簡短回覆")
    assert regeneration.regeneration_has_evidence(scope=USER_SCOPE) is True


def test_regeneration_has_evidence_detects_detail_only(memory_isolated_dir: Path) -> None:
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    assert regeneration.regeneration_has_evidence(scope=USER_SCOPE) is True


async def test_regenerate_scope_memory_failure_keeps_existing_state(
    memory_isolated_dir: Path,
) -> None:
    writer, fake_client = _writer()
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="舊的整理"))
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    append_raw_entry(scope=USER_SCOPE, entry_text="偏好訊號:\n- 喜歡簡短回覆")
    fake_client.responses.raises = TimeoutError()

    report = await _regenerate(writer=writer)

    assert report.result == "failed"
    assert "舊的整理" in _memory_text()
    assert count_raw_entries(scope=USER_SCOPE) == 1
    # Attempt-time cooldown: repeated failures are rate-limited too.
    assert regeneration.regeneration_on_cooldown(scope=USER_SCOPE) is True


async def test_regenerate_scope_memory_reports_what_it_destroyed_before_it_failed(
    memory_isolated_dir: Path,
) -> None:
    """A rebuild that gives up part way still accounts for what its earlier passes took.

    The compartments are rebuilt one at a time, so the failure of a later one comes after the
    earlier ones removed what they could not read, and putting them back restores only what
    could be read. Reporting the count only on the way out through the success path would
    lose exactly the runs an operator most needs to hear about.
    """
    writer, fake_client = _writer()
    broken = memory_isolated_dir / str(USER_ID) / GLOBAL_COMPARTMENT / f"{'b' * 16}.md"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("hand-edited into nonsense\n", encoding="utf-8")
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    # A second compartment for the run to fail on, after `global` has been replaced.
    append_detail(
        scope=USER_SCOPE,
        text=_entry(
            "2026-06-02T00:00:00+00:00",
            _observation(summary="本群祕密", normalized_key="fact.secret", sharing="source_only"),
            source="guild 222",
        ),
    )
    calls = 0

    async def failing_second_answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        del body, text_format
        nonlocal calls
        calls += 1
        if calls > 1:
            raise TimeoutError
        return _consolidated(text="重建後的記憶")

    fake_client.responses.answer = failing_second_answer
    report = await _regenerate(writer=writer)

    assert report.result == "failed"
    assert not broken.exists()
    assert report.unreadable_removed == 1


_MOVE = _observation(summary="搬到台中", normalized_key="fact.move")
_JOB = _observation(summary="在工廠上班", normalized_key="fact.job", sharing="source_only")


def _stage_forgotten_city() -> None:
    """Leaves a forgotten fact's evidence behind, as a forget that cited only another key does.

    `global/` holds the cat and no city: the forget already took the city fact, but the
    `fact.move` observation it never cited is still in `detail.md` for a rebuild to re-derive
    it from. The guild-42 observation gives the run a second compartment to stop on.
    """
    append_detail(
        scope=USER_SCOPE,
        text="\n\n".join([
            _entry("2026-09-01T00:00:00+00:00", _MOVE, _JOB),
            _forget_entry("2026-09-03T00:00:00+00:00"),
        ]),
    )
    write_fact(scope=USER_SCOPE, fact=_stored_fact(fact_id="a" * 16, summary="養了一隻貓"))


def _stopping_rebuild(stop: str, reached: asyncio.Event) -> MemoryAnswer:
    """Re-derives the city into `global`, then stops the run where `stop` says.

    `reached` is set once the guild's call is in flight, for a test that stops it from outside.
    """

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        if text_format is ToneForget:
            return ToneForget()
        if "forget_request" in body:
            if stop == "replay fails":
                raise TimeoutError
            city = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
            return ConsolidatedMemory(
                deltas=tuple(make_delta(action="delete", fact_id=fact.fact_id) for fact in city)
            )
        if "<tone_evidence>" in body:
            return _no_change()
        if "fact.job" not in body:
            return _consolidated(summary="住在台中", text="使用者住在台中", section="fact")
        if stop == "call fails":
            raise TimeoutError
        if stop == "clear":
            mark_cleared(scope=USER_SCOPE)
            delete_memory_files(scope=USER_SCOPE)
        if stop in ("times out", "cancelled"):
            reached.set()
            await asyncio.Event().wait()
        return _consolidated(summary="在工廠上班", text="使用者在工廠上班", section="fact")

    return answer


@pytest.mark.parametrize(
    "stop", ["call fails", "times out", "cancelled", "raises", "replay fails"]
)
async def test_a_rebuild_that_stops_before_its_forget_replay_puts_back_what_it_replaced(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    """Until the replay has run everywhere, a replaced compartment can hold a forgotten fact.

    `global` is rebuilt first and re-derives the city a forget had removed; the run then stops
    on the guild's call, on writing its result, or in the replay itself. Recall would hand the
    city back in every server and DM, and nothing but a rebuild that completes would remove it
    again (#893).
    """
    if stop == "times out":
        monkeypatch.setattr(regeneration, "MEMORY_CONSOLIDATE_TIMEOUT_SECONDS", 0.05)
    if stop == "raises":
        real_replace = regeneration._replace_compartment

        def failing_replace(**kwargs: Any) -> int:  # noqa: ANN401 -- a pass-through of the real signature
            """Fails writing the guild's rebuild, as a full disk would."""
            if kwargs["compartment"] != GLOBAL_COMPARTMENT:
                raise OSError("disk full")
            return real_replace(**kwargs)

        monkeypatch.setattr(regeneration, "_replace_compartment", failing_replace)
    _stage_forgotten_city()
    writer, fake_client = _writer()
    reached = asyncio.Event()
    fake_client.responses.answer = _stopping_rebuild(stop=stop, reached=reached)

    task = asyncio.create_task(_regenerate(writer=writer))
    if stop == "cancelled":
        await reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif stop == "raises":
        with pytest.raises(OSError, match="disk full"):
            await task
    else:
        assert (await task).result == "failed"
    if stop == "times out":
        # The deadline covers the `global` call too; a timeout landing there proves nothing.
        assert reached.is_set()

    assert [
        fact.summary for fact in read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
    ] == ["養了一隻貓"]


async def test_a_rebuild_stopped_by_a_clear_puts_nothing_back(memory_isolated_dir: Path) -> None:
    """Putting a replaced compartment back is a write, so a clear that stopped the run stops it."""
    _stage_forgotten_city()
    writer, fake_client = _writer()
    fake_client.responses.answer = _stopping_rebuild(stop="clear", reached=asyncio.Event())

    report = await _regenerate(writer=writer)

    assert report.result == "failed"
    assert _memory_text() == ""


def test_regeneration_cooldown_resets_after_clear(memory_isolated_dir: Path) -> None:
    regeneration._last_regeneration[USER_SCOPE] = time.monotonic()
    assert regeneration.regeneration_on_cooldown(scope=USER_SCOPE) is True
    # A clear wipes the memory the cooldown belonged to; the fresh post-clear
    # state deserves a prompt rebuild, mirroring the consolidation cooldown.
    mark_cleared(scope=USER_SCOPE)
    assert regeneration.regeneration_on_cooldown(scope=USER_SCOPE) is False


async def test_regenerate_scope_memory_recheck_cooldown_under_lock(
    memory_isolated_dir: Path,
) -> None:
    writer, fake_client = _writer()
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    # An invocation queued behind a held lock passes the command-level check
    # before the in-flight one stamps the attempt; the locked re-check is what
    # keeps the per-user limit on the expensive rewrite.
    regeneration._last_regeneration[USER_SCOPE] = time.monotonic()

    report = await _regenerate(writer=writer)

    assert report.result == "cooldown"
    assert fake_client.responses.parse_models == []


async def test_regenerate_scope_memory_aborts_write_after_clear(memory_isolated_dir: Path) -> None:
    writer, fake_client = _writer()
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)

    async def clearing_answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        del body, text_format
        mark_cleared(scope=USER_SCOPE)
        return _consolidated(text="不該被寫入")

    fake_client.responses.answer = clearing_answer
    report = await _regenerate(writer=writer)

    assert report.result == "failed"
    assert _memory_text() == ""


async def test_regenerate_scope_memory_stops_when_a_clear_lands_during_the_tone_call(
    memory_isolated_dir: Path,
) -> None:
    """The rebuild's tone call is its last await before the raw batch retires into
    `detail.md`, so a clear landing there must stop that tail too (#714).
    """
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    _stage_tone_observation()
    writer, fake_client = _writer()
    fake_client.responses.answer = _clearing_tone_answer(fact_text="重建事實")

    report = await _regenerate(writer=writer)

    assert report.result == "failed"
    assert count_raw_entries(scope=USER_SCOPE) == 1
    assert "喜歡有禮貌的回覆" not in read_detail_tail(scope=USER_SCOPE, max_chars=10_000)


async def test_schedule_memory_regeneration_runs_in_background(memory_isolated_dir: Path) -> None:
    writer, fake_client = _writer()
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    fake_client.responses.output_parsed = _consolidated(text="背景重建後的記憶")

    scheduled = regeneration.schedule_memory_regeneration(
        scope=USER_SCOPE, writer=writer, identity=IDENTITY
    )

    assert scheduled is True
    # The actual rebuild runs as a background task; await it to observe the write.
    task = regeneration._regeneration_tasks.get(key=USER_SCOPE)
    assert task is not None
    await task
    assert "背景重建後的記憶" in _memory_text()


async def test_schedule_memory_regeneration_dedupes_in_flight(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, _ = _writer()
    release = asyncio.Event()

    async def blocking_regen(
        scope: str, writer: object, identity: str
    ) -> regeneration.RegenerationReport:
        await release.wait()
        return regeneration.RegenerationReport(result="regenerated")

    monkeypatch.setattr(regeneration, "regenerate_scope_memory", blocking_regen)

    first = regeneration.schedule_memory_regeneration(
        scope=USER_SCOPE, writer=writer, identity=IDENTITY
    )
    second = regeneration.schedule_memory_regeneration(
        scope=USER_SCOPE, writer=writer, identity=IDENTITY
    )

    assert first is True
    # A rebuild already in flight must not double-schedule the whole-scope rebuild.
    assert second is False
    release.set()
    task = regeneration._regeneration_tasks.get(key=USER_SCOPE)
    assert task is not None
    await task


async def test_regenerate_scope_memory_writes_tone_and_ignores_existing_tone(
    memory_isolated_dir: Path,
) -> None:
    writer, fake_client = _writer()
    write_tone(scope=USER_SCOPE, content="## 語氣偏好\n* 舊語氣")
    # Structured evidence, because the rebuild's tone note is distilled from the batch's
    # tone-bearing observations; free-form prose carries no category to select on.
    _stage_raw_observation(
        summary="喜歡有禮貌的回覆",
        key="preference.tone",
        sharing="global",
        source="dm",
        category="interaction_style",
        evidence_kind="repeated_behavior",
    )
    fake_client.responses.output_parsed = _consolidated(
        text="重建後的記憶", tone="## 語氣偏好\n* 新語氣"
    )

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    assert read_tone(scope=USER_SCOPE) == "## 語氣偏好\n* 新語氣"
    # A pure-evidence rebuild feeds no existing tone to the model; the note is rebuilt
    # from the evidence alone, exactly like the facts.
    user_text = fake_client.responses.parse_bodies[-1]
    assert "<existing_tone>\n(empty)\n</existing_tone>" in user_text
    assert "舊語氣" not in user_text


async def test_regenerate_scope_memory_clears_stale_tone_on_empty_output(
    memory_isolated_dir: Path,
) -> None:
    """A full-evidence rebuild with no tone signal removes the now-unsupported note.

    Unlike an incremental consolidation (empty tone = "no signal in this batch",
    note kept), the rebuild saw the whole corpus, so a surviving note would keep
    injecting a preference the evidence no longer backs.
    """
    writer, fake_client = _writer()
    write_tone(scope=USER_SCOPE, content="## 語氣偏好\n* 舊語氣")
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    fake_client.responses.output_parsed = _consolidated(text="重建後的記憶")

    report = await _regenerate(writer=writer)

    assert report.result == "regenerated"
    assert read_tone(scope=USER_SCOPE) == ""


@pytest.mark.parametrize(
    argnames=("tone_answer", "expected", "result", "raw_left"),
    argvalues=[
        pytest.param(None, "## 語氣偏好\n* 舊語氣", "failed", 1, id="failed-call-keeps"),
        pytest.param(_no_change(tone=""), "", "regenerated", 0, id="empty-answer-clears"),
    ],
)
async def test_regenerate_scope_memory_clears_the_tone_note_on_an_empty_answer_only(
    memory_isolated_dir: Path,
    tone_answer: ConsolidatedMemory | None,
    expected: str,
    result: str,
    raw_left: int,
) -> None:
    """A failed tone call says nothing about the evidence; only an empty answer reads as no signal.

    Nor does a failed call retire the raw batch whose tone evidence the note never absorbed:
    once in `detail.md` it is out of every incremental tone update's reach (#825).
    """
    writer, fake_client = _writer()
    write_tone(scope=USER_SCOPE, content="## 語氣偏好\n* 舊語氣")
    _stage_tone_observation()

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Gives the tone call `tone_answer` and lets every other call through."""
        del text_format
        return tone_answer if "<tone_evidence>" in body else _consolidated(text="重建後的記憶")

    fake_client.responses.answer = answer

    report = await _regenerate(writer=writer)

    assert report.result == result
    assert read_tone(scope=USER_SCOPE) == expected
    assert count_raw_entries(scope=USER_SCOPE) == raw_left


async def test_regenerate_scope_memory_retires_a_server_raw_batch_with_no_tone_note(
    memory_isolated_dir: Path,
) -> None:
    """A server scope has no tone tier, so its absent tone call must not hold the batch back."""
    scope = server_scope(server_id=555)
    append_raw_entry(scope=scope, entry_text=_server_tone_observation())
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _consolidated(
        section="culture", text="整理", tone="## 語氣偏好\n* 不該存在"
    )

    report = await regeneration.regenerate_scope_memory(scope=scope, writer=writer, identity="srv")

    assert report.result == "regenerated"
    assert "整理" in _memory_text(scope=scope, flavor="server")
    assert count_raw_entries(scope=scope) == 0
    assert not (memory_isolated_dir / scope / "tone.md").exists()
