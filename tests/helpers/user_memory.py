"""A test user's memory scope, staged model answers, and runners for the memory services."""

import time
from typing import TYPE_CHECKING, cast
from functools import partial

import pytest
from pydantic import BaseModel
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.memory import (
    MemoryOwner,
    MemoryFlavor,
    MemorySection,
    MemorySharing,
    MemoryCategory,
    MemoryConfidence,
    MemoryDurability,
    MemoryEvidenceKind,
    MemoryWriteSummary,
)
from discordbot.typings.models import ModelSettings
from discordbot.services.memory import database as memory_db
from discordbot.services.memory import inflight, pipeline, consolidation
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    read_facts,
    user_scope,
    write_fact,
    write_tone,
    memory_root,
    mark_cleared,
    append_detail,
    append_raw_entry,
    guild_compartment,
    list_compartments,
    read_memory_document,
)
from discordbot.services.memory.writer import (
    ToneForget,
    MemoryWriterAI,
    RawMemoryDraft,
    MemoryObservation,
    ConsolidatedMemory,
    user_subject,
)
from discordbot.services.memory.raw_entries import (
    render_forget_requests,
    render_memory_observations,
)

from tests.helpers.memory import MemoryAnswer, FakeMemoryClient, make_fact, make_delta

if TYPE_CHECKING:
    from openai import AsyncOpenAI


USER_ID = 123456789

USER_SCOPE = user_scope(user_id=USER_ID)

IDENTITY = f"Alice (alice) [id: {USER_ID}]"

TEST_MEMORY_MODEL = ModelSettings(name="test-memories-model", effort="minimal")

# One `<write-memory>` note, standing in for whatever the answer model wrote inline. The content
# is irrelevant to these tests (the fake client decides what comes back); what matters is that the
# list is non-empty, since an empty one short-circuits before any model call.
_NOTES = ("使用者提到一件值得記住的事",)

# The subject a reply schedules for its author: the target, then where the turn happened.
_SUBJECT = user_subject(user_id=USER_ID, guild_id=42)


def _observation(  # noqa: PLR0913 -- test helper mirrors the structured schema
    summary: str,
    normalized_key: str = "preference.test",
    category: str = "stable_preference",
    evidence_kind: str = "explicit_preference",
    confidence: str = "high",
    durability: str = "stable",
    promotion_eligible: bool = True,
    subject_is_target_user: bool = True,
    sharing: str = "global",
    evidence_quote: str = "我偏好這樣",
    ttl_days: int | None = None,
) -> MemoryObservation:
    """Builds one accepted structured memory observation."""
    return MemoryObservation(
        category=cast("MemoryCategory", category),
        subject_is_target_user=subject_is_target_user,
        evidence_kind=cast("MemoryEvidenceKind", evidence_kind),
        confidence=cast("MemoryConfidence", confidence),
        durability=cast("MemoryDurability", durability),
        promotion_eligible=promotion_eligible,
        normalized_key=normalized_key,
        sharing=cast("MemorySharing", sharing),
        summary_zh=summary,
        evidence_quote=evidence_quote,
        ttl_days=ttl_days,
    )


def _draft(summary: str, normalized_key: str = "preference.test") -> RawMemoryDraft:
    """Builds one signalful structured memory draft."""
    return RawMemoryDraft(
        has_signal=True,
        observations=(_observation(summary=summary, normalized_key=normalized_key),),
    )


def _no_signal() -> RawMemoryDraft:
    """Builds an empty memory draft."""
    return RawMemoryDraft(has_signal=False, observations=())


def _writer() -> tuple[MemoryWriterAI, FakeMemoryClient]:
    """Builds a MemoryWriterAI bound to a fake client."""
    fake_client = FakeMemoryClient()
    writer = MemoryWriterAI(client=cast("AsyncOpenAI", fake_client), model=TEST_MEMORY_MODEL)
    return writer, fake_client


_stored_fact = partial(make_fact, owner=MemoryOwner(owner_id=USER_ID, owner_name="Alice (alice)"))


def _consolidated(
    text: str = "合併後",
    summary: str = "整理後的事實",
    section: MemorySection = "preference",
    tone: str = "",
) -> ConsolidatedMemory:
    """Builds a one-delta consolidation result."""
    return ConsolidatedMemory(
        deltas=(make_delta(summary=summary, text=text, section=section),), tone_markdown=tone
    )


def _no_change(tone: str = "") -> ConsolidatedMemory:
    """Builds a consolidation result that asks for nothing; an empty batch is a valid no-op."""
    return ConsolidatedMemory(deltas=(), tone_markdown=tone)


def _answers(
    review: RawMemoryDraft | None = None,
    facts: ConsolidatedMemory | None = None,
    forget: ConsolidatedMemory | None = None,
    tone: ConsolidatedMemory | None = None,
) -> MemoryAnswer:
    """Answers each memory call by what it asks for, never by the order the calls arrive in.

    The note review gets `review`, a forget pass `forget`, the tone-note call (the one shown
    `<tone_evidence>`) `tone`, any other consolidation `facts`, and a tone forget drops nothing.
    A role left unset answers with an empty but valid result, so a call the test did not plan
    for still succeeds rather than falling into the failure path unnoticed.
    """

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Picks the staged result for one call."""
        if text_format is RawMemoryDraft:
            return review if review is not None else _no_signal()
        if text_format is ToneForget:
            return ToneForget()
        if "forget_request" in body:
            staged = forget
        elif "<tone_evidence>" in body:
            staged = tone
        else:
            staged = facts
        return staged if staged is not None else _no_change()

    return answer


def _memory_text(scope: str = USER_SCOPE, flavor: MemoryFlavor = "user") -> str:
    """Renders every compartment a scope holds, the way the owner's own DM would read it."""
    return read_memory_document(
        scope=scope, compartments=list_compartments(scope=scope), flavor=flavor
    )


def _consolidate_at(monkeypatch: pytest.MonkeyPatch, entries: int) -> None:
    """Sets how many staged raw entries make a scope due for consolidation."""
    monkeypatch.setattr(consolidation, "RAW_CONSOLIDATION_THRESHOLD", entries)


async def _consolidate_forced(writer: MemoryWriterAI) -> None:
    """Runs the test user's consolidation as a forget forces it, starting now."""
    await consolidation.consolidate_after_turn(
        scope=USER_SCOPE,
        forced=True,
        started_at=time.monotonic(),
        writer=writer,
        identity=IDENTITY,
    )


async def _stage_row(
    token: int = 1,
    transcript: str = "清除前的對話",
    scope: str = USER_SCOPE,
    subject: str = "s",
    identity: str = "",
) -> None:
    """Writes one `pending` memory_job row for a user scope, as a staged turn leaves it."""
    await memory_db.upsert_pending(
        scope=scope,
        flavor="user",
        subject=subject,
        transcript=transcript,
        identity=identity,
        token=token,
    )


def _report_recorder() -> tuple[list[MemoryWriteSummary], inflight.MemoryWriteReport]:
    """Builds a report callback and the list it records each summary into."""
    reported: list[MemoryWriteSummary] = []

    async def record(summary: MemoryWriteSummary) -> None:
        """Captures what the pipeline decided to report."""
        reported.append(summary)

    return reported, record


def _user_message() -> list[EasyInputMessageParam]:
    """Builds a minimal message list for pipeline tests."""
    return [EasyInputMessageParam(role="user", content=f"Alice (alice) [id: {USER_ID}]: 哈囉")]


def _schedule(  # noqa: PLR0913 -- one knob per turn field a test varies
    writer: MemoryWriterAI,
    remember_notes: tuple[str, ...] = _NOTES,
    forget_notes: tuple[str, ...] = (),
    subject: str = _SUBJECT,
    full_reply: str = "回覆",
    report: inflight.MemoryWriteReport | None = None,
    scope: str = USER_SCOPE,
) -> None:
    """Schedules the memory update one reply by the test user asks for."""
    pipeline.schedule_memory_update(
        scope=scope,
        subject=subject,
        message_list=_user_message(),
        full_reply=full_reply,
        writer=writer,
        identity=IDENTITY,
        remember_notes=remember_notes,
        forget_notes=forget_notes,
        report=report,
    )


async def _wait_for_inflight() -> None:
    """Awaits the scheduled background memory task for the test user."""
    task = inflight._inflight_tasks.get(key=USER_SCOPE)
    if task is not None:
        await task


def _entry(timestamp: str, *observations: MemoryObservation, source: str = "guild 42") -> str:
    """Renders one timestamped raw/detail entry holding the given observations."""
    return (
        f"## {timestamp}\n{render_memory_observations(observations=observations, source=source)}"
    )


def _forget_entry(timestamp: str, note: str = "使用者已經不住台中了") -> str:
    """Renders one timestamped entry holding a forget request spoken in guild 42."""
    return f"## {timestamp}\n{render_forget_requests(notes=(note,), source='guild 42')}"


def _stage_raw(*entries: str) -> None:
    """Writes the test user's `raw.md` as exactly these stamped entries, oldest first."""
    scope_dir = memory_root() / USER_SCOPE
    scope_dir.mkdir(parents=True, exist_ok=True)
    (scope_dir / "raw.md").write_text("\n\n".join(entries) + "\n", encoding="utf-8")


_CITY = _observation(summary="住在台中", normalized_key="fact.city")
_PET = _observation(summary="養了一隻貓", normalized_key="fact.pet")
_ROAST = _observation(
    summary="喜歡被高強度粗口互嗆",
    normalized_key="interaction.roast",
    category="interaction_style",
)
_TERSE = _observation(
    summary="回答要簡潔", normalized_key="preference.terse", category="interaction_style"
)
_TONE_NOTE = "## 語氣偏好\n- 偏好高強度粗口互嗆\n- 回答要簡潔"


def _consolidation_stage(calls: list[str]) -> MemoryAnswer:
    """Builds an answer that stores what its raw entries evidence and forgets only the city.

    A forget pass deletes the city fact only when its request says the user moved away.

    Each call is recorded as `forget` or `observe` in `calls`, so a test can read the order.
    """

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Answers the forget pass, the tone call and the observation passes in turn."""
        if text_format is ToneForget:
            return ToneForget()
        if "forget_request" in body:
            calls.append("forget")
            doomed = [
                fact
                for fact in read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
                if fact.summary == _CITY.summary_zh and "不住台中" in body
            ]
            return ConsolidatedMemory(
                deltas=tuple(make_delta(action="delete", fact_id=fact.fact_id) for fact in doomed)
            )
        if "<tone_evidence>" in body:
            return _no_change()
        calls.append("observe")
        raw_entries = body.split("<raw_entries>")[1].split("</raw_entries>", maxsplit=1)[0]
        return ConsolidatedMemory(
            deltas=tuple(
                make_delta(
                    section="fact",
                    summary=observation.summary_zh,
                    text=observation.summary_zh,
                    from_keys=(observation.normalized_key,),
                )
                for observation in (_CITY, _PET)
                if observation.normalized_key in raw_entries
            )
        )

    return answer


_SIX = tuple(
    _observation(summary=f"事實{index}", normalized_key=f"fact.k{index}") for index in range(6)
)


def _forget_four_answer(observe: ConsolidatedMemory) -> MemoryAnswer:
    """Answers a forget pass by deleting four of the facts `global/` holds, more than the
    mass-deletion ceiling allows a six-fact compartment, and any observation pass with `observe`.
    """

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        """Deletes four stored facts on the forget pass; nothing else changes the tone note."""
        if text_format is ToneForget:
            return ToneForget()
        if "forget_request" in body:
            stored = read_facts(scope=USER_SCOPE, compartment=GLOBAL_COMPARTMENT)
            return ConsolidatedMemory(
                deltas=tuple(
                    make_delta(action="delete", fact_id=fact.fact_id) for fact in stored[:4]
                )
            )
        if "<tone_evidence>" in body:
            return _no_change()
        return observe

    return answer


_GUILD_222 = guild_compartment(guild_id=222)
# The `_compartment_note` a guild call carries, used to tell the fan-out's calls apart.
_GUILD_222_NOTE = consolidation._compartment_note(compartment=_GUILD_222, flavor="user")


def _stage_raw_observation(  # noqa: PLR0913 -- one observation's routing fields plus its category
    summary: str,
    key: str,
    sharing: str,
    source: str,
    category: str = "stable_fact",
    evidence_kind: str = "stable_fact",
) -> None:
    """Appends one already-stamped raw observation, exactly as phase-1 would have written it."""
    append_raw_entry(
        scope=USER_SCOPE,
        entry_text=render_memory_observations(
            observations=(
                _observation(
                    summary=summary,
                    normalized_key=key,
                    sharing=sharing,
                    category=category,
                    evidence_kind=evidence_kind,
                ),
            ),
            source=source,
        ),
    )


def _stage_tone_observation() -> None:
    """Stages one observation that carries tone evidence, so the tone call actually runs."""
    _stage_raw_observation(
        summary="喜歡有禮貌的回覆",
        key="preference.tone",
        sharing="global",
        source="guild 222",
        category="stable_preference",
        evidence_kind="explicit_preference",
    )


def _clearing_tone_answer(fact_text: str) -> MemoryAnswer:
    """Builds an answer that stamps a clear while the tone call is in flight."""

    async def answer(body: str, text_format: type[BaseModel]) -> BaseModel:
        del text_format
        if "<tone_evidence>" in body:
            mark_cleared(scope=USER_SCOPE)
            return _no_change(tone="## 語氣偏好\n* 偏好禮貌")
        return _consolidated(summary=fact_text, text=fact_text)

    return answer


DETAIL_EVIDENCE = "## 2026-06-01T00:00:00+00:00\n偏好訊號:\n- 喜歡條列式"


_OLDEST_DETAIL_HEADER = b"## 2026-01-01T00:00:00+00:00\n"


_BIG5 = "喜歡簡短回覆".encode("big5")


def _server_tone_observation() -> str:
    """Renders one server observation carrying tone evidence, so a tone call would have work."""
    return render_memory_observations(
        observations=(
            _observation(
                summary="社群愛互嗆",
                category="interaction_style",
                evidence_kind="repeated_behavior",
            ),
        ),
        source=None,
    )


def _populate_every_tier() -> None:
    """Writes one entry into every personal memory tier, in more than one compartment."""
    write_fact(scope=USER_SCOPE, fact=_stored_fact(text="新記憶"))
    write_fact(
        scope=USER_SCOPE,
        fact=_stored_fact(fact_id="1" * 16, compartment=_GUILD_222, text="本群記憶"),
    )
    append_raw_entry(scope=USER_SCOPE, entry_text="偏好訊號:\n- 喜歡簡短")
    append_detail(scope=USER_SCOPE, text=DETAIL_EVIDENCE)
    write_tone(scope=USER_SCOPE, content="## 語氣偏好\n* 輕鬆")


def _fail_the_file_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes the clear's file half raise, as a read-only `tone.md` would."""

    def exploding_clear(scope: str) -> bool:
        raise PermissionError("tone.md is read-only")

    monkeypatch.setattr(pipeline, "delete_memory_files", exploding_clear)


_OWNER = MemoryOwner(owner_id=111, owner_name="Alice (alice)")
_fact = partial(make_fact, owner=_OWNER)


_RAW_BATCH = """## 2026-07-01T00:00:00+00:00
### stable_preference
- normalized_key: preference.a
- source: guild 222
- sharing: global
- summary_zh: 全域偏好

### stable_fact
- normalized_key: fact.b
- source: guild 222
- sharing: source_only
- summary_zh: 本群祕密

### interaction_style
- normalized_key: style.c
- source: dm
- sharing: source_only
- summary_zh: 私訊祕密
"""
