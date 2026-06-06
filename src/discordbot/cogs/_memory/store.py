"""File-backed storage for per-user long-term memory.

Memory lives as plain markdown under ``data/memories/<user_id>/``: ``main.md``
is the consolidated hot tier injected into reply prompts, ``raw.md``
accumulates phase-1 raw extraction entries until consolidation rewrites the
main file, ``main.bak.md`` keeps the previous main generation as a manual
recovery point against a bad consolidation rewrite, and ``detail.md`` is the
readable cold tier retaining consumed and evicted raw entries verbatim:
consolidation reads its tail window as provenance and ``/memory show`` can
page through it, but it is never injected into reply prompts. The live files
stay tens of KB at most and the detail file is appended in O(1), tail-window
read, and trimmed back to a hard byte cap, so IO is synchronous; cross-task
safety comes from the per-user asyncio locks.
"""

import os
import re
import time
import asyncio
from pathlib import Path
from datetime import UTC, datetime
import itertools
import contextlib

from discordbot.cogs._memory.constants import (
    RAW_FILE_MAX_BYTES,
    DETAIL_FILE_MAX_BYTES,
    DETAIL_FILE_TRIM_TARGET_BYTES,
)

_MEMORY_DIR = Path("./data/memories")

# Raw entries start with a `## <ISO-8601 timestamp>` header line. Extraction
# output is bullet-style prose, so the date prefix doubles as the split marker.
_RAW_ENTRY_HEADER_RE = re.compile(r"^## \d{4}-\d{2}-\d{2}T", flags=re.MULTILINE)

# The identity metadata line `write_main_memory` inserts after the `v1` header
# (e.g. `v1\nAlice (alice) [id: 123]`). Read paths strip it so prompt
# injection, consolidation input, and `/memory show` never see it and the LLM
# can never echo it back; files without the line pass through unchanged.
_IDENTITY_LINE_RE = re.compile(r"^v1\n[^\n]*\[id: \d+\][^\n]*\n")

# The ` | <identity>` suffix that raw entry headers carried before identity
# was confined to the main file. New headers are timestamp-only; the strip
# remains so raw / detail files written before the removal never leak author
# identity to the consolidation LLM or `/memory show`.
_RAW_HEADER_IDENTITY_RE = re.compile(r"^(## \d{4}-\d{2}-\d{2}T\S+) \| [^\n]*$", flags=re.MULTILINE)

# Process-local registries; tests reset them through the conftest fixture.
_user_locks: dict[int, asyncio.Lock] = {}
_user_locks_loop: asyncio.AbstractEventLoop | None = None
_cleared_at: dict[int, float] = {}


def _user_dir(user_id: int) -> Path:
    """Returns the per-user memory directory."""
    return _MEMORY_DIR / str(user_id)


def _main_path(user_id: int) -> Path:
    """Returns the consolidated main memory path for a user."""
    return _user_dir(user_id=user_id) / "main.md"


def _raw_path(user_id: int) -> Path:
    """Returns the raw extraction accumulation path for a user."""
    return _user_dir(user_id=user_id) / "raw.md"


def _bak_path(user_id: int) -> Path:
    """Returns the one-generation backup path written before each main rewrite."""
    return _user_dir(user_id=user_id) / "main.bak.md"


def _detail_path(user_id: int) -> Path:
    """Returns the cold-tier detail path for consumed and evicted raw entries."""
    return _user_dir(user_id=user_id) / "detail.md"


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
    """Returns the consolidated memory, stripped of identity metadata."""
    return _strip_identity(text=_read_text(path=_main_path(user_id=user_id))).strip()


def write_main_memory(user_id: int, content: str, identity: str) -> None:
    """Atomically replaces the consolidated main memory file.

    There is no size clamp: growth is bounded by the consolidation compaction
    pass, never by code-side truncation. The previous main generation is
    copied to `main.bak.md` first as a manual recovery point, and `identity`
    is inserted after the `v1` header as human-inspection metadata that every
    read path strips back out.
    """
    _user_dir(user_id=user_id).mkdir(parents=True, exist_ok=True)
    main_path = _main_path(user_id=user_id)
    previous = _read_text(path=main_path)
    if previous:
        _bak_path(user_id=user_id).write_text(data=previous, encoding="utf-8")
    rendered = content.strip()
    if rendered.startswith("v1\n"):
        body = rendered.removeprefix("v1\n")
        rendered = f"v1\n{identity}\n{body}"
    tmp_path = main_path.with_suffix(".md.tmp")
    tmp_path.write_text(data=rendered + "\n", encoding="utf-8")
    os.replace(src=tmp_path, dst=main_path)


def append_raw_entry(user_id: int, entry_text: str) -> None:
    """Appends one timestamped raw entry, archiving the oldest entries on overflow.

    Headers carry only the timestamp: raw entries flow verbatim into the
    detail file, and author identity must stay confined to the main file.
    """
    _user_dir(user_id=user_id).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    raw_path = _raw_path(user_id=user_id)
    combined = f"{_read_text(path=raw_path)}\n\n## {timestamp}\n{entry_text.strip()}"
    entries = _split_raw_entries(text=combined)
    evicted: list[str] = []
    while len(entries) > 1 and _entries_bytes(entries=entries) > RAW_FILE_MAX_BYTES:
        evicted.append(entries.pop(0))
    rendered = "\n\n".join(entries)
    encoded = rendered.encode("utf-8")
    if len(encoded) > RAW_FILE_MAX_BYTES:
        # A single oversized entry cannot be evicted; truncate it so the raw
        # file still honors the advertised hard cap (memory is best-effort,
        # and the truncated tail is the only loss not kept in the detail file).
        rendered = encoded[:RAW_FILE_MAX_BYTES].decode(encoding="utf-8", errors="ignore")
    raw_path.write_text(data=rendered + "\n", encoding="utf-8")
    if evicted:
        # Move to the detail file only after the raw write succeeded so a
        # failed write cannot retire entries that still live in the raw file.
        append_detail(user_id=user_id, text="\n\n".join(evicted))


def append_detail(user_id: int, text: str) -> None:
    """Appends consumed or evicted raw evidence to the cold-tier detail file.

    The detail file preserves raw entry content verbatim; author identity
    stays confined to the main file, so legacy ` | <identity>` header
    suffixes from a pre-removal raw file are stripped before the write.
    Append-mode IO keeps the common write O(1) in the file size; once the
    file outgrows `DETAIL_FILE_MAX_BYTES` the oldest entries are trimmed
    away, which is safe because content past the consolidation read window
    is unreachable by every consumer anyway.
    """
    block = _RAW_HEADER_IDENTITY_RE.sub(r"\1", text).strip()
    if not block:
        return
    _user_dir(user_id=user_id).mkdir(parents=True, exist_ok=True)
    detail_path = _detail_path(user_id=user_id)
    with detail_path.open(mode="a", encoding="utf-8") as handle:
        if handle.tell() > 0:
            handle.write("\n")
        handle.write(block + "\n")
    if detail_path.stat().st_size > DETAIL_FILE_MAX_BYTES:
        _trim_detail(path=detail_path)


def _trim_detail(path: Path) -> None:
    """Drops the oldest detail entries until the file fits the trim target.

    The dropped entries are deleted permanently instead of cascading into yet
    another unbounded file: nothing can read past the consolidation window, so
    they carry no functional value. The headroom between the cap and the trim
    target amortizes this O(file) rewrite to roughly once per megabyte of
    appended evidence; the write goes through tmp + os.replace so a crash
    cannot leave a half-trimmed file.
    """
    entries = _split_raw_entries(text=_read_text(path=path))
    # Track the rendered size incrementally; recomputing the joined size per
    # dropped entry would be O(n^2) on a megabyte-scale file and stall the
    # event loop, since store IO is synchronous by design.
    sizes = [len(entry.encode("utf-8")) for entry in entries]
    total = sum(sizes) + 2 * max(len(entries) - 1, 0)
    start = 0
    while len(entries) - start > 1 and total > DETAIL_FILE_TRIM_TARGET_BYTES:
        # Dropping an entry also drops one "\n\n" separator (2 bytes).
        total -= sizes[start] + 2
        start += 1
    tmp_path = path.with_suffix(".md.tmp")
    tmp_path.write_text(data="\n\n".join(entries[start:]) + "\n", encoding="utf-8")
    os.replace(src=tmp_path, dst=path)


def read_detail_tail(user_id: int, max_chars: int) -> str:
    """Returns the newest detail-file window, minus legacy identity metadata.

    Only a bounded byte window is read from the end of the file so the call
    stays O(window) as the uncapped detail file grows. The window is aligned
    to the first raw-entry header inside the tail so a partial entry never
    leads the result; when no header lands inside the window (e.g. one giant
    entry) the raw tail is returned as a best effort.
    """
    try:
        with _detail_path(user_id=user_id).open(mode="rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            # UTF-8 spends at most 4 bytes per character, so this window can
            # never decode to fewer than max_chars characters.
            window_bytes = max_chars * 4
            handle.seek(max(0, size - window_bytes))
            data = handle.read()
    except FileNotFoundError:
        return ""
    # A window starting mid-file can cut into a multi-byte character; ignoring
    # the partial leading bytes keeps the decode safe.
    text = data.decode(encoding="utf-8", errors="ignore")
    if size > len(data) or len(text) > max_chars:
        tail = text[max(0, len(text) - max_chars) :]
        match = _RAW_ENTRY_HEADER_RE.search(tail)
        text = tail[match.start() :] if match else tail
    return _RAW_HEADER_IDENTITY_RE.sub(r"\1", text).strip()


def count_raw_entries(user_id: int) -> int:
    """Returns how many raw entries are waiting for consolidation."""
    return len(_split_raw_entries(text=_read_text(path=_raw_path(user_id=user_id))))


def raw_file_bytes(user_id: int) -> int:
    """Returns the raw file size in bytes, with a missing file counting as zero."""
    return len(_read_text(path=_raw_path(user_id=user_id)).encode("utf-8"))


def read_raw_entries(user_id: int) -> str:
    """Returns the raw file text for consolidation input, minus legacy identity metadata.

    New headers are timestamp-only; the ` | <identity>` strip remains so a raw
    file written before the suffix was removed cannot leak author identity
    into the consolidated memory content.
    """
    text = _read_text(path=_raw_path(user_id=user_id))
    return _RAW_HEADER_IDENTITY_RE.sub(r"\1", text).strip()


def clear_raw(user_id: int) -> None:
    """Deletes the raw file after a consolidation consumed it."""
    _raw_path(user_id=user_id).unlink(missing_ok=True)


def clear_user_memory(user_id: int) -> bool:
    """Deletes the user's memory files and flags in-flight updates to abort.

    Returns:
        True when at least one memory file existed and was removed.
    """
    mark_cleared(user_id=user_id)
    removed = False
    main_path = _main_path(user_id=user_id)
    for path in (
        main_path,
        _raw_path(user_id=user_id),
        _bak_path(user_id=user_id),
        _detail_path(user_id=user_id),
    ):
        try:
            path.unlink()
            removed = True
        except FileNotFoundError:
            # Already gone (e.g. offline maintenance); deletion stays idempotent
            # without the exists()-then-unlink() race.
            continue
    # A crash between a tmp write and os.replace (main rewrite or detail trim)
    # can leave a tmp file behind; drop them so the directory removal below
    # does not fail.
    main_path.with_suffix(".md.tmp").unlink(missing_ok=True)
    _detail_path(user_id=user_id).with_suffix(".md.tmp").unlink(missing_ok=True)
    # A missing or unexpectedly non-empty directory is left for offline
    # maintenance instead of failing the clear.
    with contextlib.suppress(OSError):
        _user_dir(user_id=user_id).rmdir()
    return removed


def _strip_identity(text: str) -> str:
    """Removes the store-managed identity metadata line after the `v1` header."""
    return _IDENTITY_LINE_RE.sub("v1\n", text, count=1)


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
