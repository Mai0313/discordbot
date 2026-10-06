"""Catching a server's memory up on a channel the bot was not addressed in (#1080).

Every other server-memory write starts from a marker the answer model wrote while it was part of
the conversation. `/memory server catchup` has no such model, so `MemoryWriterAI.propose_server_notes`
reads the channel and proposes the notes; from there they take the reply path's own review
(`evaluate`) and staging, and consolidation picks them up like any other raw entry.

It deliberately does not ride `schedule_memory_update`. That queue merges a held turn into the
next one of the same subject by keeping only the newer transcript, persists one `memory_job` row
per scope, and settles a failed review as "kept nothing". A catchup needs its notes judged against
its own channel, and a failure told apart from an empty result, because its result message is
where the next catchup in that channel starts reading.
"""

import time

from discordbot.services.memory.store import scope_lock, cleared_since, append_raw_entry
from discordbot.services.memory.writer import MemoryWriterAI, MemoryObservation
from discordbot.services.memory.inflight import memory_semaphore
from discordbot.services.memory.constants import MEMORY_CATCHUP_COOLDOWN_SECONDS
from discordbot.services.memory.raw_entries import render_memory_observations

# Per-scope start times of the last catchup. In-process like `/memory regenerate`'s, so a
# restart resets it.
_last_catchup: dict[str, float] = {}


def catchup_on_cooldown(scope: str) -> bool:
    """Whether a catchup in this scope started too recently for another one."""
    last = _last_catchup.get(scope)
    return last is not None and time.monotonic() - last < MEMORY_CATCHUP_COOLDOWN_SECONDS


def start_catchup_cooldown(scope: str) -> None:
    """Starts the scope's cooldown; called right after the check, with no await in between."""
    _last_catchup[scope] = time.monotonic()


def release_catchup_cooldown(scope: str) -> None:
    """Gives the cooldown back when the catchup never started, such as an empty channel."""
    _last_catchup.pop(scope, None)


async def review_catchup_notes(
    scope: str, subject: str, transcript: str, writer: MemoryWriterAI
) -> tuple[MemoryObservation, ...] | None:
    """Proposes notes from `transcript`, reviews them, and stages what survives.

    Returns the staged observations, empty when nothing was worth keeping, or None when either
    model call failed. Consolidation is left to the caller, so the result can be reported
    before it runs.
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
    return draft.observations
