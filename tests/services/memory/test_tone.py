"""Tests for the tone note: its evidence, its consolidation call and tone forgets."""

import time
from pathlib import Path

import pytest
from pydantic import BaseModel

from discordbot.services.memory import tone, consolidation
from discordbot.services.memory.run import start_run
from discordbot.services.memory.tone import tone_evidence_from_raw
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    read_tone,
    write_fact,
    write_tone,
    server_scope,
    append_detail,
    append_raw_entry,
    read_detail_tail,
    read_raw_entries,
    count_raw_entries,
    list_compartments,
)
from discordbot.services.memory.writer import ToneForget, ConsolidatedMemory
from discordbot.services.memory.raw_entries import render_forget_requests

from tests.helpers.memory import MemoryAnswer
from tests.helpers.user_memory import (
    _ROAST,
    _TERSE,
    USER_ID,
    IDENTITY,
    USER_SCOPE,
    _RAW_BATCH,
    _TONE_NOTE,
    _GUILD_222_NOTE,
    _draft,
    _entry,
    _writer,
    _answers,
    _schedule,
    _no_change,
    _stage_raw,
    _memory_text,
    _stored_fact,
    _consolidated,
    _forget_entry,
    _consolidate_at,
    _wait_for_inflight,
    _consolidate_forced,
    _clearing_tone_answer,
    _stage_raw_observation,
    _stage_tone_observation,
    _server_tone_observation,
)


def _tone_forget_answer(answer: ToneForget | None) -> MemoryAnswer:
    """Builds an answer giving the tone forget call `answer` and changing no fact."""

    async def respond(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Only the tone forget call gets `answer`."""
        del body
        return answer if text_format is ToneForget else _no_change()

    return respond


async def test_a_forget_takes_what_it_names_out_of_the_tone_note(
    memory_isolated_dir: Path,
) -> None:
    """A tone preference is never a fact, so only its own pass can forget it.

    The answer is numbers into what the call was shown, so it can only drop: a number it was
    never shown changes nothing, the forget's own sentence never lands in the note, and a
    restatement made after the forget is not even offered.
    """
    write_tone(scope=USER_SCOPE, content=_TONE_NOTE)
    append_detail(scope=USER_SCOPE, text=_entry("2026-09-01T00:00:00+00:00", _ROAST, _TERSE))
    _stage_raw(_entry("2026-09-03T00:00:00+00:00", _ROAST))
    writer, fake_client = _writer()
    fake_client.responses.answer = _tone_forget_answer(
        answer=ToneForget(drop_lines=(1, 9), drop_evidence=(1, 7))
    )
    forgets = _forget_entry("2026-09-02T00:00:00+00:00", note="使用者不想再被粗口互嗆")
    run = start_run(
        scope=USER_SCOPE, writer=writer, identity=IDENTITY, started_at=time.monotonic()
    )
    assert await tone.forget_tone(run=run, forgets=forgets)

    assert read_tone(scope=USER_SCOPE) == "## 語氣偏好\n- 回答要簡潔"
    # Two older entries offered, the restatement after the forget not among them.
    [body] = fake_client.responses.parse_bodies
    assert "[2] [explicit_preference] 回答要簡潔" in body
    assert "[3]" not in body.split("<tone_evidence>")[1]
    detail = read_detail_tail(scope=USER_SCOPE, max_chars=100_000)
    assert "interaction.roast" not in detail
    assert "preference.terse" in detail
    assert "interaction.roast" in read_raw_entries(scope=USER_SCOPE)


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        pytest.param(
            "## 語氣偏好：\n- 偏好高強度粗口互嗆\n- 回答要簡潔",
            "## 語氣偏好\n- 回答要簡潔",
            id="fullwidth-colon",
        ),
        pytest.param(
            "## 語氣偏好 \n- 偏好高強度粗口互嗆\n- 回答要簡潔",
            "## 語氣偏好\n- 回答要簡潔",
            id="trailing-space",
        ),
        pytest.param(
            "## 語氣偏好：偏好高強度粗口互嗆\n- 回答要簡潔",
            "## 語氣偏好\n- 回答要簡潔",
            id="content-on-the-header-line",
        ),
        pytest.param(
            "## 語氣偏好：回答要簡潔\n- 偏好高強度粗口互嗆",
            "## 語氣偏好\n回答要簡潔",
            id="header-line-content-survives",
        ),
    ],
)
async def test_a_forget_reaches_a_tone_note_under_any_header_the_write_accepts(
    memory_isolated_dir: Path, note: str, expected: str
) -> None:
    """The write keeps any note leading with the header, so the forget must reach all of it."""
    write_tone(scope=USER_SCOPE, content=note)
    writer, fake_client = _writer()

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Drops whichever offered note line holds the forgotten preference."""
        del text_format
        lines = body.split("<tone_note>")[1].split("</tone_note>", maxsplit=1)[0].splitlines()
        return ToneForget(
            drop_lines=tuple(
                int(line.split("]")[0].lstrip("[")) for line in lines if "粗口互嗆" in line
            )
        )

    fake_client.responses.answer = answer
    forgets = _forget_entry("2026-09-02T00:00:00+00:00", note="使用者不想再被粗口互嗆")
    run = start_run(
        scope=USER_SCOPE, writer=writer, identity=IDENTITY, started_at=time.monotonic()
    )
    assert await tone.forget_tone(run=run, forgets=forgets)

    assert read_tone(scope=USER_SCOPE) == expected


async def test_each_tone_forget_sees_only_what_came_before_it(memory_isolated_dir: Path) -> None:
    """A tone preference restated between two forgets is not offered to the first of them.

    The evidence lines carry no stamp, so the model could not tell a restatement from what
    was forgotten; each segment's tone forget is offered only what predates its own requests.
    """
    write_tone(scope=USER_SCOPE, content=_TONE_NOTE)
    _stage_raw(
        _forget_entry("2026-09-01T00:00:00+00:00", note="使用者不想再被粗口互嗆"),
        _entry("2026-09-02T00:00:00+00:00", _ROAST),
        _forget_entry("2026-09-03T00:00:00+00:00", note="使用者不想再提工作的事"),
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers()
    await _consolidate_forced(writer=writer)

    tone_calls = [body for body in fake_client.responses.parse_bodies if "<tone_note>" in body]
    assert len(tone_calls) == 2
    # order-contract: the batch's forgets run one at a time, oldest stamp first.
    assert _ROAST.summary_zh not in tone_calls[0]
    assert _ROAST.summary_zh in tone_calls[1]


async def test_a_failed_tone_forget_keeps_the_batch(memory_isolated_dir: Path) -> None:
    """The note is injected into every reply, so a forget it has not taken yet must be retried.

    Retiring the batch would leave the request in `detail.md`, where only a rebuild reads it.
    """
    write_tone(scope=USER_SCOPE, content=_TONE_NOTE)
    append_raw_entry(
        scope=USER_SCOPE,
        entry_text=render_forget_requests(notes=("使用者不想再被粗口互嗆",), source="guild 42"),
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _tone_forget_answer(answer=None)
    await _consolidate_forced(writer=writer)

    assert any("<tone_note>" in body for body in fake_client.responses.parse_bodies), (
        "the tone forget was asked"
    )
    assert read_tone(scope=USER_SCOPE) == _TONE_NOTE
    assert count_raw_entries(scope=USER_SCOPE) == 1


async def test_a_source_only_batch_still_updates_the_tone_note(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tone is written by its own call fed the WHOLE batch, so a `source_only`-only
    conversation still updates it.

    Roughly half of all observations are `source_only`; a tone note fed only the global
    bucket would simply stop updating for those conversations. The call that writes it
    is separate from every compartment call precisely so the unpartitioned evidence it
    needs can never reach one that writes facts.
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    _stage_raw_observation(
        summary="喜歡有禮貌的回覆",
        key="preference.tone",
        sharing="source_only",
        source="guild 222",
        category="stable_preference",
        evidence_kind="explicit_preference",
    )
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(
        facts=_consolidated(summary="本群事實", text="本群事實"),
        tone=_no_change(tone="## 語氣偏好\n* 偏好禮貌"),
    )
    await consolidation.consolidate_if_needed(scope=USER_SCOPE, writer=writer, identity=IDENTITY)

    assert read_tone(scope=USER_SCOPE) == "## 語氣偏好\n* 偏好禮貌"
    tone_calls = [text for text in fake_client.responses.parse_bodies if "<tone_evidence>" in text]
    fact_calls = [
        text for text in fake_client.responses.parse_bodies if "<tone_evidence>" not in text
    ]
    # Exactly one call saw the unpartitioned evidence, and it was handed no facts to
    # write and no raw bucket, so it structurally cannot store one anywhere.
    assert len(tone_calls) == 1
    assert "喜歡有禮貌的回覆" in tone_calls[0]
    assert "<raw_entries>\n(empty)\n</raw_entries>" in tone_calls[0]
    assert "<existing_facts>\n(empty)\n</existing_facts>" in tone_calls[0]
    # No compartment call was shown the `source_only` summary outside its own bucket.
    assert fact_calls
    for text in fact_calls:
        assert "<tone_evidence>" not in text
        if _GUILD_222_NOTE not in text:
            assert "喜歡有禮貌的回覆" not in text


async def test_a_clear_during_the_tone_call_keeps_the_batch_out_of_detail(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tone call is the last await before the batch retires into `detail.md`.

    A clear never takes `scope_lock`, so one landing while that call is in flight must still
    stop the tail: re-creating `detail.md` from the batch would bring back the evidence the
    user just erased, and the next consolidation would read it again (#714).
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    _stage_tone_observation()
    writer, fake_client = _writer()
    fake_client.responses.answer = _clearing_tone_answer(fact_text="全域事實")

    await consolidation.consolidate_if_needed(scope=USER_SCOPE, writer=writer, identity=IDENTITY)

    assert not (memory_isolated_dir / str(USER_ID) / "detail.md").exists()
    assert count_raw_entries(scope=USER_SCOPE) == 1
    assert read_tone(scope=USER_SCOPE) == ""


async def test_a_failed_tone_call_keeps_the_batch(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tone call that failed keeps the batch, so the next run still offers its tone evidence.

    Retired into `detail.md`, that evidence is out of every later update's reach (#839). An
    answer that is empty or lacks the header is not a failed call and still retires the batch
    (`test_pipeline_bad_tone_output_keeps_existing_note`).
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    write_tone(scope=USER_SCOPE, content="## 語氣偏好\n* 原有偏好")
    _stage_tone_observation()
    writer, fake_client = _writer()
    tone_answer: ConsolidatedMemory | None = None

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel | None:
        """Gives the tone call `tone_answer` and lets every other call through."""
        del text_format
        return tone_answer if "<tone_evidence>" in body else _no_change()

    fake_client.responses.answer = answer
    await consolidation.consolidate_if_needed(scope=USER_SCOPE, writer=writer, identity=IDENTITY)

    assert count_raw_entries(scope=USER_SCOPE) == 1
    assert not (memory_isolated_dir / str(USER_ID) / "detail.md").exists()
    assert read_tone(scope=USER_SCOPE) == "## 語氣偏好\n* 原有偏好"

    consolidation._last_consolidation.clear()
    tone_answer = _no_change(tone="## 語氣偏好\n* 偏好禮貌")
    await consolidation.consolidate_if_needed(scope=USER_SCOPE, writer=writer, identity=IDENTITY)

    assert read_tone(scope=USER_SCOPE) == "## 語氣偏好\n* 偏好禮貌"
    assert count_raw_entries(scope=USER_SCOPE) == 0


async def test_pipeline_consolidation_writes_tone_note(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    write_tone(scope=USER_SCOPE, content="## 語氣偏好\n* 舊語氣")
    writer, fake_client = _writer()
    # Three calls: the note review, the compartment's facts, then the tone note on its own.
    # Only the last is asked for `tone_markdown`.
    fake_client.responses.answer = _answers(
        review=_draft("訊號"),
        facts=_consolidated(),
        tone=_no_change(tone="## 語氣偏好\n* 偏好禮貌"),
    )
    _schedule(writer=writer)
    await _wait_for_inflight()
    assert "合併後" in _memory_text()
    assert read_tone(scope=USER_SCOPE) == "## 語氣偏好\n* 偏好禮貌"
    # The current note rode the TONE call, not the compartment's; the compartment call
    # is never shown it, because it has no business rewriting the note.
    # order-contract: the tone call runs after the scope's one fact-compartment call.
    assert (
        "<existing_tone>\n## 語氣偏好\n* 舊語氣\n</existing_tone>"
        in fake_client.responses.parse_bodies[2]
    )
    # order-contract: the preceding fact-compartment call must not receive the tone document.
    assert "<existing_tone>" not in fake_client.responses.parse_bodies[1]


async def test_pipeline_no_op_consolidation_still_writes_tone(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="既有內容"))
    writer, fake_client = _writer()
    # A batch that changes no fact can still carry fresh tone signal, and it consumes the raw
    # entries either way, so the tone must land now or be lost. The tone call runs after the
    # compartment's regardless of whether that one changed anything.
    fake_client.responses.answer = _answers(
        review=_draft("已知資訊"),
        facts=_no_change(),
        tone=_no_change(tone="## 語氣偏好\n* 偏好簡短"),
    )
    _schedule(writer=writer)
    await _wait_for_inflight()
    assert "既有內容" in _memory_text()
    assert read_tone(scope=USER_SCOPE) == "## 語氣偏好\n* 偏好簡短"
    assert count_raw_entries(scope=USER_SCOPE) == 0


@pytest.mark.parametrize(
    argnames="bad_tone",
    argvalues=["", "語氣:很兇但沒有標頭"],
    ids=["empty-tone", "malformed-tone"],
)
async def test_pipeline_bad_tone_output_keeps_existing_note(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch, bad_tone: str
) -> None:
    """An unusable tone note is dropped on its own; the facts it rode with still commit.

    A delta batch is per fact, so it never holds the facts hostage to the tone tier, which
    is best-effort and repaired by the next pass.
    """
    _consolidate_at(monkeypatch=monkeypatch, entries=1)
    write_tone(scope=USER_SCOPE, content="## 語氣偏好\n* 原有偏好")
    writer, fake_client = _writer()
    fake_client.responses.answer = _answers(
        review=_draft("訊號"), facts=_consolidated(), tone=_no_change(tone=bad_tone)
    )
    _schedule(writer=writer)
    await _wait_for_inflight()
    assert "合併後" in _memory_text()
    assert count_raw_entries(scope=USER_SCOPE) == 0
    # Neither shape ever deletes the existing note.
    assert read_tone(scope=USER_SCOPE) == "## 語氣偏好\n* 原有偏好"


async def test_consolidate_if_needed_server_scope_never_writes_tone(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _consolidate_at(monkeypatch=monkeypatch, entries=2)
    scope = server_scope(server_id=555)
    append_raw_entry(scope=scope, entry_text=_server_tone_observation())
    append_raw_entry(scope=scope, entry_text="- 第二筆")
    writer, fake_client = _writer()
    fake_client.responses.output_parsed = _consolidated(
        section="culture", text="整理", tone="## 語氣偏好\n* 不該存在"
    )
    await consolidation.consolidate_if_needed(scope=scope, writer=writer, identity="srv")
    assert "整理" in _memory_text(scope=scope, flavor="server")
    # A server scope has exactly one compartment, so its evidence never fans out.
    assert list_compartments(scope=scope) == [GLOBAL_COMPARTMENT]
    # The tone note is a per-user tier; a server consolidation never writes one.
    assert read_tone(scope=scope) == ""
    assert not (memory_isolated_dir / scope / "tone.md").exists()


def test_tone_evidence_ignores_compartments() -> None:
    """Tone must see the whole batch, or it stops updating for half of all conversations."""
    evidence = tone_evidence_from_raw(raw_text=_RAW_BATCH)
    assert "私訊祕密" in evidence
    assert "全域偏好" in evidence


def test_tone_evidence_carries_the_evidence_kind() -> None:
    """A stated preference has to reach the note distinguishable from an inferred one.

    Carrying only the summary made both read alike, so the note converged on whichever
    reading had the most bullets: one stated "address me respectfully" lost to five
    inferred banter observations and the note came out saying the opposite.
    """
    text = (
        "## 2026-07-01T00:00:00+00:00\n"
        "### interaction_style\n"
        "- normalized_key: style.stated\n"
        "- evidence_kind: explicit_preference\n"
        "- sharing: source_only\n"
        "- summary_zh: 要求對本人使用尊敬語氣\n"
        "\n"
        "### interaction_style\n"
        "- normalized_key: style.inferred\n"
        "- evidence_kind: repeated_behavior\n"
        "- sharing: global\n"
        "- summary_zh: 會主動嗆機器人\n"
    )
    evidence = tone_evidence_from_raw(raw_text=text)
    assert "* [explicit_preference] 要求對本人使用尊敬語氣" in evidence
    assert "* [repeated_behavior] 會主動嗆機器人" in evidence
