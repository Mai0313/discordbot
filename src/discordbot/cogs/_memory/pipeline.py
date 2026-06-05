"""Background orchestration for the two-phase per-user memory pipeline."""

import time
import asyncio

import logfire
from pydantic import BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.cogs._memory.store import (
    clear_raw,
    user_lock,
    cleared_since,
    raw_file_bytes,
    append_raw_entry,
    read_raw_entries,
    count_raw_entries,
    write_main_memory,
    read_main_memory_full,
)
from discordbot.cogs._memory.constants import (
    RAW_CONSOLIDATION_MAX_BYTES,
    RAW_CONSOLIDATION_THRESHOLD,
)
from discordbot.cogs._memory.extraction import MemoryExtractorAI, transcript_from_messages


class _PendingMemoryUpdate(BaseModel):
    """The newest skipped update request, replayed once the in-flight task ends.

    Attributes:
        message_list: Reply-pipeline input messages captured for the skipped turn.
        full_reply: The streamed reply text for the skipped turn.
        extractor: The extraction service to run the replayed update with.
        captured_at: `time.monotonic()` when the turn was captured, so a clear
            that lands before the replay can abort it via `cleared_since`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_list: SkipValidation[list[EasyInputMessageParam]]
    full_reply: str
    extractor: SkipValidation[MemoryExtractorAI]
    captured_at: float


# Process-level per-user in-flight de-dupe; while one extraction runs, only the
# NEWEST skipped turn is kept and replayed afterwards. Its history window
# already contains the earlier skipped turns, so one replay recovers the
# dropped signal without a real queue.
_inflight_tasks: dict[int, asyncio.Task[None]] = {}
_pending_updates: dict[int, _PendingMemoryUpdate] = {}
_inflight_loop: asyncio.AbstractEventLoop | None = None


def schedule_memory_update(
    user_id: int,
    message_list: list[EasyInputMessageParam],
    full_reply: str,
    extractor: MemoryExtractorAI,
) -> None:
    """Starts a background memory update without delaying the reply path."""
    global _inflight_loop  # noqa: PLW0603 -- process task de-dupe
    loop = asyncio.get_running_loop()
    if _inflight_loop is not loop:
        _inflight_tasks.clear()
        _pending_updates.clear()
        _inflight_loop = loop
    running = _inflight_tasks.get(user_id)
    if running is not None and not running.done():
        _pending_updates[user_id] = _PendingMemoryUpdate(
            message_list=message_list,
            full_reply=full_reply,
            extractor=extractor,
            captured_at=time.monotonic(),
        )
        return
    task = asyncio.create_task(
        _run_memory_update(
            user_id=user_id, message_list=message_list, full_reply=full_reply, extractor=extractor
        )
    )
    _inflight_tasks[user_id] = task
    task.add_done_callback(lambda finished: _finish_memory_update(user_id=user_id, task=finished))


def _finish_memory_update(user_id: int, task: asyncio.Task[None]) -> None:
    """Clears the in-flight slot, logs failures, and replays a pending update."""
    if _inflight_tasks.get(user_id) is task:
        _inflight_tasks.pop(user_id, None)
    if task.cancelled():
        # Cancelled (e.g. bot shutdown): reading result() would raise
        # CancelledError (a BaseException on 3.11+) out of this callback, and a
        # pre-shutdown turn is not worth replaying.
        return
    try:
        task.result()
    except Exception:
        logfire.warn("Background memory update failed", user_id=user_id, _exc_info=True)
    pending = _pending_updates.pop(user_id, None)
    if pending is None:
        return
    if cleared_since(user_id=user_id, started_at=pending.captured_at):
        # The user cleared their memory after this turn was captured; replaying
        # it would write the pre-clear conversation back into storage.
        return
    schedule_memory_update(
        user_id=user_id,
        message_list=pending.message_list,
        full_reply=pending.full_reply,
        extractor=pending.extractor,
    )


async def _run_memory_update(
    user_id: int,
    message_list: list[EasyInputMessageParam],
    full_reply: str,
    extractor: MemoryExtractorAI,
) -> None:
    """Runs phase-1 extraction and, past the raw threshold, phase-2 consolidation."""
    started_at = time.monotonic()
    transcript = transcript_from_messages(message_list=message_list, full_reply=full_reply)
    async with user_lock(user_id=user_id):
        draft = await extractor.extract(target_user_id=user_id, transcript=transcript)
        if draft is None or not draft.has_signal or not draft.memory_markdown:
            return
        if cleared_since(user_id=user_id, started_at=started_at):
            # The user cleared their memory while this update was in flight;
            # dropping the write beats resurrecting deleted memory.
            return
        append_raw_entry(user_id=user_id, entry_text=draft.memory_markdown)
        if (
            count_raw_entries(user_id=user_id) < RAW_CONSOLIDATION_THRESHOLD
            and raw_file_bytes(user_id=user_id) < RAW_CONSOLIDATION_MAX_BYTES
        ):
            return
        await _consolidate_locked(user_id=user_id, started_at=started_at, extractor=extractor)


async def _consolidate_locked(
    user_id: int, started_at: float, extractor: MemoryExtractorAI
) -> None:
    """Consolidates accumulated raw entries into the main memory file."""
    result = await extractor.consolidate(
        existing_main=read_main_memory_full(user_id=user_id),
        raw_entries=read_raw_entries(user_id=user_id),
    )
    if result is None:
        # LLM path failed; keep the raw entries so the next update retries.
        return
    if cleared_since(user_id=user_id, started_at=started_at):
        return
    is_well_formed = result.memory_markdown.startswith("v1\n")
    if result.changed and not is_well_formed:
        # Malformed rewrite (changed but missing the exact `v1` header line, so
        # near-misses like `v10...` or `v1: ...` are rejected too): keep the raw
        # batch so the next consolidation retries instead of losing the signal.
        return
    if is_well_formed:
        # Accept any well-formed `v1` rewrite, even one the model flagged
        # `changed=false`, so a single contradictory boolean cannot silently
        # discard the whole raw batch.
        write_main_memory(user_id=user_id, content=result.memory_markdown)
    # Written or a genuine empty no-op: the batch is consumed either way, since
    # an unchanged verdict on the same raw entries would just re-burn a
    # consolidation call on every following extraction.
    clear_raw(user_id=user_id)
