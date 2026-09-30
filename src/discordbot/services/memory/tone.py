"""The per-user tone note: the one memory tier that is not partitioned.

Tone is cross-server safe by construction, which is exactly why it must NOT be routed
through the compartments: nearly half of all observations are `source_only`, so a note fed
only the `global` bucket would simply stop updating for those conversations. Both writes
therefore run as their own consolidation call, whose deltas are discarded by code, and both
live here rather than beside the fan-out they hang off.

The two differ only in what they are allowed to conclude from silence. `update_tone_note`
sees one batch, so no tone signal means "nothing this time" and the existing note stands.
`rebuild_tone_note` saw the whole evidence corpus, so no signal anywhere means the note is
stale, and it is the only path allowed to delete it for want of signal. `forget_tone` is the
third writer: it can only take lines away, and a forget that takes every line takes the note.
"""

from functools import partial

import logfire

from discordbot.typings.memory import TONE_HEADER
from discordbot.services.memory.run import ConsolidationRun
from discordbot.services.memory.store import (
    read_tone,
    clear_tone,
    write_tone,
    cleared_since,
    read_evidence,
    rewrite_evidence,
)
from discordbot.services.memory.deltas import (
    newest_stamp,
    drop_observations,
    tone_observations,
    tone_evidence_from_raw,
)
from discordbot.services.memory.writer import ConsolidationRequest


async def update_tone_note(run: ConsolidationRun, raw_entries: str) -> None:
    """Rewrites the per-user tone note from the WHOLE batch, in its own call.

    It gets a call of its own rather than riding on the `global` compartment's, whose input
    is partitioned by construction. Here the deltas are discarded by CODE — this call cannot
    write a fact anywhere, whatever it returns — which is what makes unpartitioned input safe.

    Best-effort throughout: the note is a small always-read tier and the next
    consolidation repairs a bad write, so a failure never touches the raw batch.
    """
    if run.flavor != "user":
        return
    tone_evidence = tone_evidence_from_raw(raw_text=raw_entries)
    if not tone_evidence:
        # No tone signal in this batch is the normal case, and an empty output must
        # never delete the note; only the evidence-complete rebuild may do that.
        return
    result = await run.writer.consolidate(
        request=_tone_request(
            existing_tone=read_tone(scope=run.scope), tone_evidence=tone_evidence, today=run.today
        )
    )
    if result is None or cleared_since(scope=run.scope, started_at=run.started_at):
        return
    _write_tone_result(scope=run.scope, tone_markdown=result.tone_markdown)


async def rebuild_tone_note(run: ConsolidationRun, evidence: str) -> None:
    """Rebuilds the tone note from the whole evidence corpus, in its own call.

    This pass saw everything, so no signal anywhere means a surviving note is stale and
    would keep injecting a preference the evidence no longer supports. It is the only path
    allowed to delete the note for want of signal; `forget_tone` deletes it only by taking
    its last line.
    """
    if run.flavor != "user":
        return
    tone_evidence = tone_evidence_from_raw(raw_text=evidence)
    result = (
        None
        if not tone_evidence
        else await run.writer.consolidate(
            # No `existing_tone`: this pass saw the whole corpus, so it rewrites the note
            # from the evidence rather than merging into what is already there.
            request=_tone_request(existing_tone="", tone_evidence=tone_evidence, today=run.today)
        )
    )
    if tone_evidence and result is None:
        logfire.warn("Memory tone rebuild call failed; tone note left untouched", scope=run.scope)
        return
    if cleared_since(scope=run.scope, started_at=run.started_at):
        return
    if result is None or not result.tone_markdown:
        clear_tone(scope=run.scope)
        return
    _write_tone_result(scope=run.scope, tone_markdown=result.tone_markdown)


async def forget_tone(run: ConsolidationRun, forgets: str) -> bool:
    """Takes what forget requests name out of the tone note and out of the evidence behind it.

    A tone preference is never stored as a fact, so the fact pass has nothing to delete for one,
    and the note would go on carrying it into every reply while its evidence waited for the
    next rebuild to read it back. Only evidence stamped before the newest request is offered,
    since anything later restates what was forgotten. The call can only point at lines it was
    shown, so this drops and never writes: the forget's own sentence has nowhere to go.

    Returns False when the call failed, so a caller that must not lose the forget keeps its
    batch for a retry; True when it ran or had nothing to do.
    """
    if run.flavor != "user" or not forgets:
        return True
    cutoff = newest_stamp(text=forgets)
    note = read_tone(scope=run.scope).splitlines()
    note_lines = (
        tuple(line for line in note[1:] if line.strip()) if note[:1] == [TONE_HEADER] else ()
    )
    evidence = [
        observation
        for observation in tone_observations(text=read_evidence(scope=run.scope))
        if observation[0] < cutoff
    ]
    if not note_lines and not evidence:
        return True
    result = await run.writer.forget_tone(
        forgets=forgets, note_lines=note_lines, evidence=tuple(line for _, _, line in evidence)
    )
    if result is None:
        logfire.warn("Memory tone forget call failed; keeping raw batch", scope=run.scope)
        return False
    if cleared_since(scope=run.scope, started_at=run.started_at):
        return False
    kept = [
        line for number, line in enumerate(note_lines, start=1) if number not in result.drop_lines
    ]
    if len(kept) < len(note_lines):
        if kept:
            write_tone(scope=run.scope, content="\n".join([TONE_HEADER, *kept]))
        else:
            clear_tone(scope=run.scope)
    doomed = {
        (timestamp, block)
        for number, (timestamp, block, _) in enumerate(evidence, start=1)
        if number in result.drop_evidence
    }
    if doomed:
        rewrite_evidence(scope=run.scope, edit=partial(drop_observations, doomed=doomed))
    return True


def _tone_request(existing_tone: str, tone_evidence: str, today: str) -> ConsolidationRequest:
    """Builds the tone note's own request, the one consolidation call that writes no fact.

    Sharing the builder is what stops the compartment note — the line telling the model
    which tier it is writing — drifting between the calls that make it.
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
    """Persists a tone-note call's output when it leads with `TONE_HEADER`.

    An empty or malformed output never deletes the existing note: the tier is best-effort
    and the next consolidation repairs it. Only the evidence-complete rebuild may clear it for
    want of signal.
    """
    if tone_markdown.startswith(TONE_HEADER):
        write_tone(scope=scope, content=tone_markdown)
