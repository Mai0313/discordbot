"""File-backed storage for per-user long-term memory.

Memory lives as plain markdown under ``data/memories/<user_id>/``: ``main.md``
is the consolidated hot tier injected into reply prompts, ``raw.md``
accumulates phase-1 raw extraction entries until consolidation rewrites the
main file, ``main.bak.md`` keeps the previous main generation as a manual
recovery point against a bad consolidation rewrite, and ``detail.md`` is the
readable cold tier retaining consumed and evicted raw entries verbatim:
consolidation reads its tail window as provenance and ``/memory show`` can
page through it, but it is never injected into reply prompts. The live files
stay tens of KB at most and the uncapped detail file is only ever appended to
in O(1) (reads are tail-windowed), so IO is synchronous; cross-task safety
comes from the per-user asyncio locks.
"""

import os
import re
import time
import asyncio
from pathlib import Path
from datetime import UTC, datetime
import itertools
import contextlib

from discordbot.cogs._memory.constants import RAW_FILE_MAX_BYTES

_MEMORY_DIR = Path("./data/memories")

# Raw entries start with a `## <ISO-8601 timestamp> | <identity>` header line.
# Extraction output is bullet-style prose, so the date prefix doubles as the
# split marker.
_RAW_ENTRY_HEADER_RE = re.compile(r"^## \d{4}-\d{2}-\d{2}T", flags=re.MULTILINE)

# The identity metadata line `write_main_memory` inserts after the `v1` header
# (e.g. `v1\nAlice (alice) [id: 123]`). Read paths strip it so prompt
# injection, consolidation input, and `/memory show` never see it and the LLM
# can never echo it back; files without the line pass through unchanged.
_IDENTITY_LINE_RE = re.compile(r"^v1\n[^\n]*\[id: \d+\][^\n]*\n")

# The ` | <identity>` suffix on raw entry headers. `read_raw_entries` strips it
# so the consolidation LLM never sees author identity either; headers written
# before the suffix existed pass through unchanged.
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


def append_raw_entry(user_id: int, entry_text: str, identity: str) -> None:
    """Appends one timestamped raw entry, archiving the oldest entries on overflow."""
    _user_dir(user_id=user_id).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    raw_path = _raw_path(user_id=user_id)
    combined = f"{_read_text(path=raw_path)}\n\n## {timestamp} | {identity}\n{entry_text.strip()}"
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

    The detail file is append-only and uncapped by design (the operator
    accepts the growth); it preserves raw entries verbatim, including the
    identity header suffixes, which every read path strips back out.
    Append-mode IO keeps the write O(1) in the file size so an old, large
    detail file never slows consolidation down or pins the user lock.
    """
    block = text.strip()
    if not block:
        return
    _user_dir(user_id=user_id).mkdir(parents=True, exist_ok=True)
    with _detail_path(user_id=user_id).open(mode="a", encoding="utf-8") as handle:
        if handle.tell() > 0:
            handle.write("\n")
        handle.write(block + "\n")


def read_detail_tail(user_id: int, max_chars: int) -> str:
    """Returns the newest detail-file window, minus identity metadata.

    The window is aligned to the first raw-entry header inside the tail so a
    partial entry never leads the result; when no header lands inside the
    window (e.g. one giant entry) the raw tail is returned as a best effort.
    """
    text = _read_text(path=_detail_path(user_id=user_id))
    if len(text) > max_chars:
        tail = text[len(text) - max_chars :]
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
    """Returns the raw file text for consolidation input, minus identity metadata.

    The ` | <identity>` header suffix is disk-only inspection metadata; it is
    stripped here so the consolidation LLM cannot copy author identity into
    the consolidated memory content.
    """
    text = _read_text(path=_raw_path(user_id=user_id))
    return _RAW_HEADER_IDENTITY_RE.sub(r"\1", text).strip()


def read_raw_on_disk(user_id: int) -> str:
    """Returns the raw file's verbatim on-disk text for the detail file.

    Unlike `read_raw_entries`, the identity header suffixes are kept: they are
    disk-only human-inspection metadata, and every detail read path strips
    them before the text reaches an LLM or a command response.
    """
    return _read_text(path=_raw_path(user_id=user_id)).strip()


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
    # A crash between the tmp write and os.replace can leave the tmp file
    # behind; drop it so the directory removal below does not fail.
    main_path.with_suffix(".md.tmp").unlink(missing_ok=True)
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
