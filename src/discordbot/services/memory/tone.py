"""The per-user tone note: the one memory tier that is not partitioned.

Tone is cross-server safe by construction, which is why it must not be routed
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
from discordbot.services.memory.writer import ConsolidationRequest
from discordbot.services.memory.raw_entries import (
    fields_of,
    newest_stamp,
    iter_observations,
    render_file_entries,
    observation_category,
)

# Observation categories that carry how the user wants the bot to sound. Everything
# else is a fact and has no business in the always-injected tone note.
_TONE_CATEGORIES = frozenset({"interaction_style", "stable_preference"})


async def update_tone_note(run: ConsolidationRun, raw_entries: str) -> bool:
    """Rewrites the per-user tone note from the whole batch in its own consolidation call.

    Deltas returned by this call are discarded by code so unpartitioned input cannot write a
    fact into any compartment. An empty or headerless output leaves the existing note in place.

    Args:
        run (ConsolidationRun): Active consolidation run context.
        raw_entries (str): Raw entries from the batch being consolidated.

    Returns:
        True when the call succeeded or had nothing to do; False when the call failed or a
        clear overtook it and the caller must keep the raw batch.
    """
    if run.flavor != "user":
        return True
    tone_evidence = tone_evidence_from_raw(raw_text=raw_entries)
    if not tone_evidence:
        # No tone signal in this batch is the normal case, and an empty output must
        # never delete the note; only the evidence-complete rebuild may do that.
        return True
    result = await run.writer.consolidate(
        flavor=run.flavor,
        request=_tone_request(
            existing_tone=read_tone(scope=run.scope), tone_evidence=tone_evidence, today=run.today
        ),
    )
    if result is None:
        logfire.warn("Memory tone update call failed; keeping raw batch", scope=run.scope)
        return False
    if cleared_since(scope=run.scope, started_at=run.started_at):
        return False
    _write_tone_result(scope=run.scope, tone_markdown=result.tone_markdown)
    return True


async def rebuild_tone_note(run: ConsolidationRun, evidence: str) -> bool:
    """Rebuilds the per-user tone note from the full evidence corpus, clearing it when empty.

    Because this pass sees the whole evidence corpus, a lack of tone signal clears any stale
    note on disk.

    Args:
        run (ConsolidationRun): Active rebuild run context.
        evidence (str): Full evidence corpus (`detail.md` tail plus `raw.md`).

    Returns:
        True when the rebuild succeeded or had no tone evidence; False when the call failed or
        a clear overtook it.
    """
    if run.flavor != "user":
        return True
    tone_evidence = tone_evidence_from_raw(raw_text=evidence)
    result = (
        None
        if not tone_evidence
        else await run.writer.consolidate(
            flavor=run.flavor,
            # No `existing_tone`: this pass saw the whole corpus, so it rewrites the note
            # from the evidence rather than merging into what is already there.
            request=_tone_request(existing_tone="", tone_evidence=tone_evidence, today=run.today),
        )
    )
    if tone_evidence and result is None:
        logfire.warn("Memory tone rebuild call failed; tone note left untouched", scope=run.scope)
        return False
    if cleared_since(scope=run.scope, started_at=run.started_at):
        return False
    if result is None or not result.tone_markdown:
        clear_tone(scope=run.scope)
        return True
    _write_tone_result(scope=run.scope, tone_markdown=result.tone_markdown)
    return True


async def forget_tone(run: ConsolidationRun, forgets: str, evidence_only: bool = False) -> bool:
    """Removes lines and evidence named by `forgets` from the per-user tone tier.

    Only evidence stamped before the newest forget request is offered so later restatements
    survive, and the model answers with line numbers only so the forget's own sentence can
    never be written into the note.

    Args:
        run (ConsolidationRun): Active consolidation or rebuild run context.
        forgets (str): Raw forget-request entries for this segment.
        evidence_only (bool): Whether to skip the note itself when a rebuild is about to
            rewrite it from the surviving evidence.

    Returns:
        True when the pass succeeded or had nothing to offer; False when the call failed or a
        clear overtook it.
    """
    if run.flavor != "user" or not forgets:
        return True
    cutoff = newest_stamp(text=forgets)
    note = "" if evidence_only else read_tone(scope=run.scope)
    # Anything after the header on its own line is content too: the rewrite below replaces
    # that line, so content left unoffered there would be deleted without being named.
    body = note.removeprefix(TONE_HEADER).lstrip(" ：:") if note.startswith(TONE_HEADER) else ""
    note_lines = tuple(line for line in body.splitlines() if line.strip())
    evidence = [
        observation
        for observation in _tone_observations(text=read_evidence(scope=run.scope))
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
        rewrite_evidence(scope=run.scope, edit=partial(_drop_observations, doomed=doomed))
    return True


def tone_evidence_from_raw(raw_text: str) -> str:
    """Returns `raw_text`'s tone-bearing observations as `[evidence_kind] summary` bullets.

    Prefixing `evidence_kind` oldest-first lets consolidation distinguish a stated preference
    from one inferred off the user's own behavior across batches.

    Args:
        raw_text (str): Raw or evidence text to extract tone observations from.
    """
    return "\n".join(f"* {line}" for _, _, line in _tone_observations(text=raw_text))


def _tone_observations(text: str) -> list[tuple[str, str, str]]:
    """Returns each tone-bearing observation of a raw or detail text as `(stamp, block, line)`.

    `line` is how it reads as tone evidence, `[evidence_kind] summary`; a block with no summary
    carries no signal and is left out.
    """
    observations: list[tuple[str, str, str]] = []
    for timestamp, block in iter_observations(text=text):
        # The category is the block's `### <category>` header, not one of its fields.
        if observation_category(block=block) not in _TONE_CATEGORIES:
            continue
        fields = fields_of(block=block)
        summary = fields.get("summary_zh", "")
        if summary:
            line = f"[{fields.get('evidence_kind', 'unknown')}] {summary}"
            observations.append((timestamp, block, line))
    return observations


def _drop_observations(text: str, doomed: set[tuple[str, str]]) -> str:
    """Removes the given `(stamp, block)` observations from a raw or detail text.

    Returns `text` itself when none of them is there, so the caller can skip the rewrite.
    """
    pairs = iter_observations(text=text)
    kept = [pair for pair in pairs if pair not in doomed]
    if len(kept) == len(pairs):
        return text
    return render_file_entries(pairs=kept)


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
