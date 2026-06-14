"""File-backed storage for long-term memory, keyed by an opaque scope.

A scope is a relative path under ``data/memories/`` that doubles as the
registry key. Per-user memory uses ``user_scope(user_id)`` (``<user_id>``);
the bot's own per-server memory uses ``server_scope(bot_id, server_id)``
(``<bot_id>/<server_id>``). Both share the exact same file layout: ``main.md``
is the consolidated hot tier injected into reply prompts, ``raw.md``
accumulates phase-1 raw extraction entries until consolidation rewrites the
main file, ``main.bak.md`` keeps the previous main generation as a manual
recovery point against a bad consolidation rewrite, and ``detail.md`` is the
readable cold tier retaining consumed and evicted raw entries verbatim:
consolidation reads its tail window as provenance and ``/memory show`` can
page through it, but it is never injected into reply prompts. The live files
stay tens of KB at most and the detail file is appended in O(1), tail-window
read, and trimmed back to a hard byte cap, so IO is synchronous; cross-task
safety comes from the per-scope asyncio locks.
"""

import os
import re
import time
import asyncio
from pathlib import Path
from datetime import UTC, datetime
import itertools
import contextlib

from discordbot.utils.asyncio_locks import LoopLocalRegistry
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

# Capturing variant used by `read_main_identity` to recover the stored line.
_IDENTITY_CAPTURE_RE = re.compile(r"^v1\n([^\n]*\[id: \d+\][^\n]*)\n")

# Per-scope file-write locks, rebuilt per event loop by the shared registry.
_scope_locks: LoopLocalRegistry[str, asyncio.Lock] = LoopLocalRegistry()
# Manual-clear timestamps; monotonic, so it is not loop-keyed and tests reset it.
_cleared_at: dict[str, float] = {}


def user_scope(user_id: int) -> str:
    """Returns the storage scope for one user's memory."""
    return str(user_id)


def server_scope(bot_id: int, server_id: int) -> str:
    """Returns the storage scope for the bot's memory of one server."""
    return f"{bot_id}/{server_id}"


def _scope_dir(scope: str) -> Path:
    """Returns the memory directory for a scope."""
    return _MEMORY_DIR / scope


def _main_path(scope: str) -> Path:
    """Returns the consolidated main memory path for a scope."""
    return _scope_dir(scope=scope) / "main.md"


def _raw_path(scope: str) -> Path:
    """Returns the raw extraction accumulation path for a scope."""
    return _scope_dir(scope=scope) / "raw.md"


def _bak_path(scope: str) -> Path:
    """Returns the one-generation backup path written before each main rewrite."""
    return _scope_dir(scope=scope) / "main.bak.md"


def _detail_path(scope: str) -> Path:
    """Returns the cold-tier detail path for consumed and evicted raw entries."""
    return _scope_dir(scope=scope) / "detail.md"


def _read_text(path: Path) -> str:
    """Reads a memory file, treating a missing file as empty."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def scope_lock(scope: str) -> asyncio.Lock:
    """Returns the per-scope lock that serializes memory file writes."""
    return _scope_locks.setdefault(key=scope, default=asyncio.Lock())


def mark_cleared(scope: str) -> None:
    """Records a manual memory clear so older in-flight updates abort their writes."""
    _cleared_at[scope] = time.monotonic()


def cleared_since(scope: str, started_at: float) -> bool:
    """Whether the scope's memory was cleared at or after `started_at` (time.monotonic)."""
    cleared = _cleared_at.get(scope)
    return cleared is not None and cleared >= started_at


def read_main_memory(scope: str) -> str:
    """Returns the consolidated memory, stripped of identity metadata."""
    return _strip_identity(text=_read_text(path=_main_path(scope=scope))).strip()


def read_main_identity(scope: str) -> str:
    """Returns the identity metadata line stored in the main file, or empty.

    Offline regeneration has no Discord context to rebuild the identity from,
    so it preserves the line the last online write stamped into the file.
    """
    match = _IDENTITY_CAPTURE_RE.match(_read_text(path=_main_path(scope=scope)))
    return match.group(1) if match else ""


def write_main_memory(scope: str, content: str, identity: str) -> None:
    """Atomically replaces the consolidated main memory file.

    There is no size clamp: growth is bounded by the consolidation compaction
    pass, never by code-side truncation. The previous main generation is
    copied to `main.bak.md` first as a manual recovery point, and `identity`
    is inserted after the `v1` header as human-inspection metadata that every
    read path strips back out.
    """
    _scope_dir(scope=scope).mkdir(parents=True, exist_ok=True)
    main_path = _main_path(scope=scope)
    previous = _read_text(path=main_path)
    if previous:
        _bak_path(scope=scope).write_text(data=previous, encoding="utf-8")
    rendered = content.strip()
    if rendered.startswith("v1\n"):
        body = rendered.removeprefix("v1\n")
        rendered = f"v1\n{identity}\n{body}"
    tmp_path = main_path.with_suffix(".md.tmp")
    tmp_path.write_text(data=rendered + "\n", encoding="utf-8")
    os.replace(src=tmp_path, dst=main_path)


def append_raw_entry(scope: str, entry_text: str) -> None:
    """Appends one timestamped raw entry, archiving the oldest entries on overflow.

    Headers carry only the timestamp: raw entries flow verbatim into the
    detail file, and author identity must stay confined to the main file.
    """
    _scope_dir(scope=scope).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    raw_path = _raw_path(scope=scope)
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
        append_detail(scope=scope, text="\n\n".join(evicted))


def append_detail(scope: str, text: str) -> None:
    """Appends consumed or evicted raw evidence to the cold-tier detail file.

    The detail file preserves raw entry content verbatim; author identity
    stays confined to the main file. Append-mode IO keeps the common write
    O(1) in the file size; once the file outgrows `DETAIL_FILE_MAX_BYTES`
    the oldest entries are trimmed away, which is safe because content past
    the consolidation read window is unreachable by every consumer anyway.
    """
    block = text.strip()
    if not block:
        return
    _scope_dir(scope=scope).mkdir(parents=True, exist_ok=True)
    detail_path = _detail_path(scope=scope)
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


def read_detail_tail(scope: str, max_chars: int) -> str:
    """Returns the newest detail-file window, aligned to a raw-entry header.

    Only a bounded byte window is read from the end of the file so the call
    stays O(window) as the uncapped detail file grows. The window is aligned
    to the first raw-entry header inside the tail so a partial entry never
    leads the result; when no header lands inside the window (e.g. one giant
    entry) the raw tail is returned as a best effort.
    """
    try:
        with _detail_path(scope=scope).open(mode="rb") as handle:
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
    return text.strip()


def count_raw_entries(scope: str) -> int:
    """Returns how many raw entries are waiting for consolidation."""
    return len(_split_raw_entries(text=_read_text(path=_raw_path(scope=scope))))


def raw_file_bytes(scope: str) -> int:
    """Returns the raw file size in bytes, with a missing file counting as zero."""
    return len(_read_text(path=_raw_path(scope=scope)).encode("utf-8"))


def detail_file_bytes(scope: str) -> int:
    """Returns the cold-tier detail file size in bytes, missing file counting as zero.

    Uses ``stat`` rather than reading the whole file so an evidence-presence
    check stays O(1) even when the detail file is near its multi-megabyte cap.
    """
    path = _detail_path(scope=scope)
    return path.stat().st_size if path.is_file() else 0


def read_raw_entries(scope: str) -> str:
    """Returns the raw file text for consolidation input."""
    return _read_text(path=_raw_path(scope=scope)).strip()


def clear_raw(scope: str) -> None:
    """Deletes the raw file after a consolidation consumed it."""
    _raw_path(scope=scope).unlink(missing_ok=True)


def clear_memory(scope: str) -> bool:
    """Deletes the scope's memory files and flags in-flight updates to abort.

    Returns:
        True when at least one memory file existed and was removed.
    """
    mark_cleared(scope=scope)
    removed = False
    main_path = _main_path(scope=scope)
    for path in (
        main_path,
        _raw_path(scope=scope),
        _bak_path(scope=scope),
        _detail_path(scope=scope),
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
    _detail_path(scope=scope).with_suffix(".md.tmp").unlink(missing_ok=True)
    # A missing or unexpectedly non-empty directory is left for offline
    # maintenance instead of failing the clear.
    with contextlib.suppress(OSError):
        _scope_dir(scope=scope).rmdir()
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
