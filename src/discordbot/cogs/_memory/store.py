"""File-backed storage for per-user long-term memory.

Memory lives as plain markdown under ``data/memories/``: ``<user_id>.md`` is
the consolidated main memory injected into reply prompts, and
``<user_id>.raw.md`` accumulates phase-1 raw extraction entries until
consolidation rewrites the main file. Files stay small (a few KB), so IO is
synchronous; cross-task safety comes from the per-user asyncio locks.
"""

import os
import re
import time
import asyncio
from pathlib import Path
from datetime import UTC, datetime
import itertools

from discordbot.cogs._memory.constants import RAW_FILE_MAX_BYTES, MEMORY_INJECTION_MAX_CHARS

_MEMORY_DIR = Path("./data/memories")

# Raw entries start with a `## <ISO-8601 timestamp>` header line. Extraction
# output is bullet-style prose, so this shape doubles as the split marker.
_RAW_ENTRY_HEADER_RE = re.compile(r"^## \d{4}-\d{2}-\d{2}T", flags=re.MULTILINE)

# Process-local registries; tests reset them through the conftest fixture.
_user_locks: dict[int, asyncio.Lock] = {}
_user_locks_loop: asyncio.AbstractEventLoop | None = None
_cleared_at: dict[int, float] = {}


def _main_path(user_id: int) -> Path:
    """Returns the consolidated main memory path for a user."""
    return _MEMORY_DIR / f"{user_id}.md"


def _raw_path(user_id: int) -> Path:
    """Returns the raw extraction accumulation path for a user."""
    return _MEMORY_DIR / f"{user_id}.raw.md"


def _read_text(path: Path) -> str:
    """Reads a memory file, treating a missing file as empty."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def user_lock(user_id: int) -> asyncio.Lock:
    """Returns the per-user lock that serializes memory file writes."""
    global _user_locks_loop  # noqa: PLW0603 -- process-local lock registry keyed by loop
    loop = asyncio.get_running_loop()
    if _user_locks_loop is not loop:
        _user_locks.clear()
        _user_locks_loop = loop
    return _user_locks.setdefault(user_id, asyncio.Lock())


def mark_cleared(user_id: int) -> None:
    """Records a manual memory clear so older in-flight updates abort their writes."""
    _cleared_at[user_id] = time.monotonic()


def cleared_since(user_id: int, started_at: float) -> bool:
    """Whether the user's memory was cleared at or after `started_at` (time.monotonic)."""
    cleared = _cleared_at.get(user_id)
    return cleared is not None and cleared >= started_at


def read_main_memory(user_id: int) -> str:
    """Returns the consolidated memory for prompt injection, stripped and truncated."""
    return _read_text(path=_main_path(user_id=user_id)).strip()[:MEMORY_INJECTION_MAX_CHARS]


def read_main_memory_full(user_id: int) -> str:
    """Returns the consolidated memory without the injection truncation."""
    return _read_text(path=_main_path(user_id=user_id)).strip()


def write_main_memory(user_id: int, content: str) -> None:
    """Atomically replaces the consolidated main memory file."""
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    main_path = _main_path(user_id=user_id)
    tmp_path = main_path.with_suffix(".md.tmp")
    tmp_path.write_text(data=content.strip() + "\n", encoding="utf-8")
    os.replace(src=tmp_path, dst=main_path)


def append_raw_entry(user_id: int, entry_text: str) -> None:
    """Appends one timestamped raw entry, evicting the oldest entries on overflow."""
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    raw_path = _raw_path(user_id=user_id)
    combined = f"{_read_text(path=raw_path)}\n\n## {timestamp}\n{entry_text.strip()}"
    entries = _split_raw_entries(text=combined)
    while len(entries) > 1 and _entries_bytes(entries=entries) > RAW_FILE_MAX_BYTES:
        entries.pop(0)
    raw_path.write_text(data="\n\n".join(entries) + "\n", encoding="utf-8")


def count_raw_entries(user_id: int) -> int:
    """Returns how many raw entries are waiting for consolidation."""
    return len(_split_raw_entries(text=_read_text(path=_raw_path(user_id=user_id))))


def raw_file_bytes(user_id: int) -> int:
    """Returns the raw file size in bytes, with a missing file counting as zero."""
    return len(_read_text(path=_raw_path(user_id=user_id)).encode("utf-8"))


def read_raw_entries(user_id: int) -> str:
    """Returns the full raw file text for consolidation input."""
    return _read_text(path=_raw_path(user_id=user_id)).strip()


def clear_raw(user_id: int) -> None:
    """Deletes the raw file after a consolidation consumed it."""
    _raw_path(user_id=user_id).unlink(missing_ok=True)


def clear_user_memory(user_id: int) -> bool:
    """Deletes both memory files and flags in-flight updates to abort.

    Returns:
        True when at least one memory file existed and was removed.
    """
    mark_cleared(user_id=user_id)
    removed = False
    for path in (_main_path(user_id=user_id), _raw_path(user_id=user_id)):
        try:
            path.unlink()
            removed = True
        except FileNotFoundError:
            # Already gone (e.g. offline maintenance); deletion stays idempotent
            # without the exists()-then-unlink() race.
            continue
    return removed


def _split_raw_entries(text: str) -> list[str]:
    """Splits raw file text into stripped per-entry blocks including headers."""
    starts = [match.start() for match in _RAW_ENTRY_HEADER_RE.finditer(text)]
    if not starts:
        return []
    bounds = [*starts, len(text)]
    blocks = [text[begin:end].strip() for begin, end in itertools.pairwise(bounds)]
    return [block for block in blocks if block]


def _entries_bytes(entries: list[str]) -> int:
    """Returns the rendered raw-file size for a list of entry blocks."""
    return len("\n\n".join(entries).encode("utf-8"))
