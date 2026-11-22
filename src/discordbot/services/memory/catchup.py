"""Server-memory catchup for channels where the bot was not addressed.

Uses `MemoryWriterAI.propose_server_notes` in place of inline reply markers, then runs the same
`evaluate` review and `raw.md` staging used by the reply path.

Bypasses `schedule_memory_update`: that queue replaces pending transcripts on the same scope,
persists one `memory_job` row per scope, and folds a failed review into an empty result. Catchup
must judge notes against the scanned channel's transcript and distinguish model failure from an
empty result, as its result message anchors the next catchup's read point.
"""

import time

from discordbot.services.memory.store import (
    scope_lock,
    trim_detail,
    cleared_since,
    append_raw_entry,
)
from discordbot.services.memory.writer import MemoryWriterAI, MemoryObservation
from discordbot.services.memory.inflight import memory_semaphore
from discordbot.services.memory.constants import MEMORY_CATCHUP_COOLDOWN_SECONDS
from discordbot.services.memory.raw_entries import render_memory_observations

# In-process per-scope cooldowns; a restart resets them.
_last_catchup: dict[str, float] = {}


def catchup_on_cooldown(scope: str) -> bool:
    """Returns whether a catchup in `scope` started too recently for another one.

    Args:
        scope (str): Server memory scope to check.
    """
    last = _last_catchup.get(scope)
    return last is not None and time.monotonic() - last < MEMORY_CATCHUP_COOLDOWN_SECONDS


def start_catchup_cooldown(scope: str) -> None:
    """Starts the scope's cooldown.

    Call immediately after `catchup_on_cooldown` with no await in between.

    Args:
        scope (str): Server memory scope starting a catchup.
    """
    _last_catchup[scope] = time.monotonic()


def release_catchup_cooldown(scope: str) -> None:
    """Clears the scope's cooldown when a catchup exits before starting work.

    Args:
        scope (str): Server memory scope releasing its cooldown.
    """
    _last_catchup.pop(scope, None)


async def review_catchup_notes(
    scope: str, subject: str, transcript: str, writer: MemoryWriterAI
) -> tuple[MemoryObservation, ...] | None:
    """Proposes notes from `transcript`, reviews them, and stages what survives.

    Does not trigger consolidation; the caller schedules it after reporting the result.

    Args:
        scope (str): Server memory scope to stage observations into.
        subject (str): Phase-1 target directive for the server.
        transcript (str): Rendered channel transcript to review.
        writer (MemoryWriterAI): Memory writing service running the proposal and review.

    Returns:
        Staged observations (empty when nothing was worth keeping), or None when either model
        call failed.
    """
    started_at = time.monotonic()
    notes = await writer.propose_server_notes(subject=subject, transcript=transcript)
    if notes is None:
        return None
    if not notes:
        return ()
    async with scope_lock(scope=scope), memory_semaphore():
        draft = await writer.evaluate(
            flavor="server", subject=subject, transcript=transcript, notes=notes
        )
        if draft is None:
            return None
        if not draft.observations or cleared_since(scope=scope, started_at=started_at):
            return ()
        append_raw_entry(
            scope=scope,
            entry_text=render_memory_observations(observations=draft.observations, source=None),
        )
        await trim_detail(scope=scope, started_at=started_at)
    return draft.observations
