"""The per-user tone note: the one memory tier that is not partitioned.

Tone is cross-server safe by construction, which is exactly why it must NOT be routed
through the compartments: nearly half of all observations are `source_only`, so a note fed
only the `global` bucket would simply stop updating for those conversations. Both writes
therefore run as their own consolidation call, whose deltas are discarded by code, and both
live here rather than beside the fan-out they hang off.

The two differ only in what they are allowed to conclude from silence. `update_tone_note`
sees one batch, so no tone signal means "nothing this time" and the existing note stands.
`rebuild_tone_note` saw the whole evidence corpus, so no signal anywhere means the note is
stale, and it is the only path allowed to delete it.
"""

from discordbot.services.memory.facts import MemoryFlavor
from discordbot.services.memory.store import (
    flavor_of,
    read_tone,
    clear_tone,
    write_tone,
    cleared_since,
)
from discordbot.services.memory.deltas import tone_evidence_from_raw
from discordbot.services.memory.writer import MemoryWriterAI, ConsolidationRequest

# The exact header a tone note must lead with; the tier is injected on every reply,
# so anything else is a rewrite that did not land and must not be written.
_TONE_HEADER = "## 語氣偏好"


async def update_tone_note(  # noqa: PLR0913 -- the scope's identity plus the batch, its stamp, and the LLM handle
    scope: str,
    flavor: MemoryFlavor,
    started_at: float,
    writer: MemoryWriterAI,
    raw_entries: str,
    today: str,
) -> None:
    """Rewrites the per-user tone note from the WHOLE batch, in its own call.

    Tone is the one tier that is cross-server safe by construction, so it is the one
    thing that must not be partitioned: nearly half of all observations are
    `source_only`, and a tone note fed only the `global` bucket would simply stop
    updating for those conversations.

    That is exactly why it gets its own call rather than riding on the `global`
    compartment's. A compartment call sees only the evidence routed to the compartment
    it writes, which is what makes "a guild-locked observation cannot reach `global/`"
    structural; handing that same call the unpartitioned tone evidence would have
    demoted the boundary back to a rule the prompt asks the model to follow. Here the
    deltas are discarded by CODE — this call cannot write a fact anywhere, whatever it
    returns — so the unpartitioned input is safe by the same structural argument.

    Best-effort throughout: the note is a small always-read tier and the next
    consolidation repairs a bad write, so a failure never touches the raw batch.
    """
    if flavor != "user":
        return
    tone_evidence = tone_evidence_from_raw(raw_text=raw_entries)
    if not tone_evidence:
        # No tone signal in this batch is the normal case, and an empty output must
        # never delete the note; only the evidence-complete rebuild may do that.
        return
    result = await writer.consolidate(
        request=_tone_request(
            existing_tone=read_tone(scope=scope), tone_evidence=tone_evidence, today=today
        )
    )
    if result is None or cleared_since(scope=scope, started_at=started_at):
        return
    _write_tone_result(scope=scope, tone_markdown=result.tone_markdown)


async def rebuild_tone_note(  # noqa: PLR0913 -- the scope's identity plus the corpus, its stamp, and the LLM handle
    scope: str,
    flavor: MemoryFlavor,
    started_at: float,
    writer: MemoryWriterAI,
    evidence: str,
    today: str,
) -> None:
    """Rebuilds the tone note from the whole evidence corpus, in its own call.

    Unlike an incremental consolidation — whose empty tone output only means "no tone
    signal in this batch" — this pass saw everything, so no signal anywhere means a
    surviving note is stale and would keep injecting a preference the evidence no longer
    supports. This is the only path allowed to delete the note.
    """
    if flavor != "user":
        return
    tone_evidence = tone_evidence_from_raw(raw_text=evidence)
    result = (
        None
        if not tone_evidence
        else await writer.consolidate(
            # No `existing_tone`: this pass saw the whole corpus, so it rewrites the note
            # from the evidence rather than merging into what is already there.
            request=_tone_request(existing_tone="", tone_evidence=tone_evidence, today=today)
        )
    )
    if cleared_since(scope=scope, started_at=started_at):
        return
    if result is None or not result.tone_markdown:
        clear_tone(scope=scope)
        return
    _write_tone_result(scope=scope, tone_markdown=result.tone_markdown)


def _tone_request(existing_tone: str, tone_evidence: str, today: str) -> ConsolidationRequest:
    """Builds the tone note's own request, the one consolidation call that writes no fact.

    Its two callers send the same shape and differ only in whether the current note is
    offered back: the incremental pass merges into it, while the evidence-complete rebuild
    deliberately ignores it. Sharing the builder is what stops the compartment note — the
    line telling the model which tier it is writing — drifting between the two.
    """
    return ConsolidationRequest(
        compartment_note="the user's persona-independent tone note, read in every conversation",
        allowed_sections=(),
        raw_entries="",
        existing_tone=existing_tone,
        tone_evidence=tone_evidence,
        today=today,
        compact=False,
        emit_tone=True,
    )


def _write_tone_result(scope: str, tone_markdown: str) -> None:
    """Persists a tone-note call's output when it is acceptable for this scope.

    User scopes only, and only a note starting with the exact `## 語氣偏好` header;
    an empty or malformed output never deletes the existing note — the tier is
    best-effort and the next consolidation repairs it. Only `rebuild_tone_note`,
    which saw the whole evidence corpus, may clear it.
    """
    if flavor_of(scope=scope) != "user":
        return
    if not _tone_is_well_formed(tone_markdown=tone_markdown):
        return
    write_tone(scope=scope, content=tone_markdown)


def _tone_is_well_formed(tone_markdown: str) -> bool:
    """Whether a tone note carries the exact header the injected tier is contracted to."""
    return tone_markdown.startswith(_TONE_HEADER)
