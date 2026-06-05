"""Background orchestration for the two-phase per-user memory pipeline."""

import time
import asyncio

import logfire
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.cogs._memory import store
from discordbot.cogs._memory.constants import (
    RAW_CONSOLIDATION_MAX_BYTES,
    RAW_CONSOLIDATION_THRESHOLD,
)
from discordbot.cogs._memory.extraction import MemoryExtractorAI, transcript_from_messages

# Process-level per-user in-flight de-dupe; a user's rapid-fire replies only
# get one extraction at a time, the rest are skipped (not queued).
_inflight_tasks: dict[int, asyncio.Task[None]] = {}
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
        _inflight_loop = loop
    running = _inflight_tasks.get(user_id)
    if running is not None and not running.done():
        return
    task = asyncio.create_task(
        _run_memory_update(
            user_id=user_id, message_list=message_list, full_reply=full_reply, extractor=extractor
        )
    )
    _inflight_tasks[user_id] = task
    task.add_done_callback(lambda finished: _finish_memory_update(user_id=user_id, task=finished))


def _finish_memory_update(user_id: int, task: asyncio.Task[None]) -> None:
    """Clears the in-flight slot and logs unexpected background failures."""
    if _inflight_tasks.get(user_id) is task:
        _inflight_tasks.pop(user_id, None)
    try:
        task.result()
    except Exception:
        logfire.warn("Background memory update failed", user_id=user_id, _exc_info=True)


async def _run_memory_update(
    user_id: int,
    message_list: list[EasyInputMessageParam],
    full_reply: str,
    extractor: MemoryExtractorAI,
) -> None:
    """Runs phase-1 extraction and, past the raw threshold, phase-2 consolidation."""
    started_at = time.monotonic()
    transcript = transcript_from_messages(message_list=message_list, full_reply=full_reply)
    async with store.user_lock(user_id=user_id):
        draft = await extractor.extract(target_user_id=user_id, transcript=transcript)
        if draft is None or not draft.has_signal or not draft.memory_markdown:
            return
        if store.cleared_since(user_id=user_id, started_at=started_at):
            # The user cleared their memory while this update was in flight;
            # dropping the write beats resurrecting deleted memory.
            return
        store.append_raw_entry(user_id=user_id, entry_text=draft.memory_markdown)
        if (
            store.count_raw_entries(user_id=user_id) < RAW_CONSOLIDATION_THRESHOLD
            and store.raw_file_bytes(user_id=user_id) < RAW_CONSOLIDATION_MAX_BYTES
        ):
            return
        await _consolidate_locked(user_id=user_id, started_at=started_at, extractor=extractor)


async def _consolidate_locked(
    user_id: int, started_at: float, extractor: MemoryExtractorAI
) -> None:
    """Consolidates accumulated raw entries into the main memory file."""
    result = await extractor.consolidate(
        existing_main=store.read_main_memory_full(user_id=user_id),
        raw_entries=store.read_raw_entries(user_id=user_id),
    )
    if result is None:
        # LLM path failed; keep the raw entries so the next update retries.
        return
    if store.cleared_since(user_id=user_id, started_at=started_at):
        return
    if result.changed and not result.memory_markdown.startswith("v1\n"):
        # Malformed rewrite (changed but missing the exact `v1` header line, so
        # near-misses like `v10...` or `v1: ...` are rejected too): keep the raw
        # batch so the next consolidation retries instead of losing the signal.
        return
    if result.changed:
        store.write_main_memory(user_id=user_id, content=result.memory_markdown)
    # Written or genuinely unchanged: the batch is consumed either way, since an
    # unchanged verdict on the same raw entries would just re-burn a
    # consolidation call on every following extraction.
    store.clear_raw(user_id=user_id)
