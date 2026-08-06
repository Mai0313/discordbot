"""File-backed storage for long-term memory, keyed by an opaque scope.

A scope is a relative path under ``data/memories/`` that doubles as the registry key.
Per-user memory uses ``user_scope(user_id)`` (``<user_id>``); the bot's own per-server
memory uses ``server_scope(server_id)`` (``bot_memories/<server_id>``).

Inside a scope the consolidated tier is **one file per fact**, filed under the
compartment that decides who may read it — ``global/`` (safe anywhere),
``g/<guild_id>/`` (that guild only) and ``dm/`` (the owner's own DMs). The path *is*
the privacy boundary: reading for guild G is ``global/`` plus ``g/<G>/``, two path joins
past one shape check, with no read-time content filter to get wrong. A server scope has
exactly one compartment (``global/``) because a server memory is per-guild by
construction and its evidence carries no source to route by.

The remaining tiers are per-scope and unchanged: ``raw.md`` accumulates phase-1 entries
until consolidation consumes them, ``detail.md`` is the append-only cold evidence log
(read as a tail window, trimmed to a hard byte cap), and ``tone.md`` is the short
always-read note of how the user wants the bot to sound.

Everything here is a bare filesystem operation: no LLM call, no policy about which
compartment an observation belongs in (``deltas.partition_raw_entries`` decides that),
and no lock taken. Serialization is the caller's job through the ``scope_lock`` handed
out here — ``pipeline.py`` holds it around a whole extraction or consolidation, because
a delta batch is N renames rather than one atomic replace, and the clear path is the one
deliberate exception (it would otherwise wait minutes behind a consolidation). Readers
take nothing: ``gen_reply``'s ``memory_tool``, the ``/memory`` cog and the offline
``scripts/`` call straight in, which is why every read here treats a file that vanished
mid-walk as absent rather than an error. Single writes are still atomic on their own —
a fact, the tone note and a detail trim each go to a sibling ``.md.tmp`` and then
``os.replace``.

IO is synchronous, which one fact per file would otherwise make untenable on the reply
path: ``read_memory_document`` is cached under a per-scope generation counter that
``write_fact`` / ``delete_fact`` / ``delete_memory_files`` bump, so a repeat read costs
no syscalls at all. The counter is exact because every write in this process goes
through here; editing the tree from outside while the bot runs is not supported (nor is
it today, for ``_cleared_at``).
"""

import os
import re
import time
import asyncio
from pathlib import Path
from datetime import UTC, datetime
import itertools
import contextlib

from discordbot.typings.memory import MemoryFact, MemoryOwner
from discordbot.utils.asyncio_locks import LoopLocalRegistry
from discordbot.services.memory.facts import (
    MemoryFlavor,
    parse_fact_file,
    render_fact_file,
    render_memory_document,
)
from discordbot.services.memory.constants import (
    RAW_FILE_MAX_BYTES,
    TONE_FILE_MAX_BYTES,
    DETAIL_FILE_MAX_BYTES,
    RENDER_CACHE_MAX_ENTRIES,
    MEMORY_INJECTION_MAX_CHARS,
    DETAIL_FILE_TRIM_TARGET_BYTES,
)

_MEMORY_DIR = Path("./data/memories")

# Fixed parent directory for the bot's own per-server memory. Deliberately not the
# bot's user id: a numeric directory is indistinguishable from a user scope on disk,
# and it strands the whole server memory the moment the bot account changes.
BOT_MEMORY_DIR_NAME = "bot_memories"

# The three compartment shapes. `g/<id>` carries a separator on purpose so it joins
# straight onto the scope directory; `_COMPARTMENT_RE` is what stops anything else
# (`..`, an absolute path, a stray name) ever becoming a directory.
GLOBAL_COMPARTMENT = "global"
DM_COMPARTMENT = "dm"
_COMPARTMENT_RE = re.compile(r"^(?:global|dm|g/\d{1,20})$")
_GUILD_DIR_NAME = "g"

# Raw entries start with a `## <ISO-8601 timestamp>` header line. Extraction
# output is bullet-style prose, so the date prefix doubles as the split marker.
_RAW_ENTRY_HEADER_RE = re.compile(r"^## \d{4}-\d{2}-\d{2}T", flags=re.MULTILINE)

# Per-scope file-write locks, rebuilt per event loop by the shared registry.
_scope_locks: LoopLocalRegistry[str, asyncio.Lock] = LoopLocalRegistry()
# Manual-clear timestamps; monotonic, so it is not loop-keyed and tests reset it.
_cleared_at: dict[str, float] = {}
# Per-scope write counter and the rendered-document cache keyed on it. Not loop-bound
# (plain dicts, no asyncio primitive), but reset by the test fixture like `_cleared_at`.
_write_generation: dict[str, int] = {}
_render_cache: dict[tuple[str, tuple[str, ...], MemoryFlavor, int], tuple[int, str]] = {}


def user_scope(user_id: int) -> str:
    """Returns the storage scope for one user's memory.

    Args:
        user_id (int): Discord id of the user.

    Returns:
        The scope key, which is also the scope's directory name under the store root.
    """
    return str(user_id)


def server_scope(server_id: int) -> str:
    """Returns the storage scope for the bot's memory of one server.

    Args:
        server_id (int): Discord id of the guild.

    Returns:
        The scope key, nested under `BOT_MEMORY_DIR_NAME` so it can never be mistaken for
        a user scope on disk.
    """
    return f"{BOT_MEMORY_DIR_NAME}/{server_id}"


def guild_compartment(guild_id: int) -> str:
    """Returns the compartment holding facts readable only inside one guild.

    Args:
        guild_id (int): Discord id of the guild.

    Returns:
        The `g/<guild_id>` key, which joins straight onto a scope directory.
    """
    return f"{_GUILD_DIR_NAME}/{guild_id}"


def scope_owner_id(scope: str) -> int:
    """Returns the Discord id a scope belongs to (a user id, or a server id).

    The inverse of `user_scope` / `server_scope`, for the offline and restart paths that
    hold a scope key but no Discord context.

    Args:
        scope (str): The scope key.

    Returns:
        The id in its last path segment.
    """
    return int(scope.rsplit("/", maxsplit=1)[-1])


def memory_root() -> Path:
    """Returns the store root, read through this accessor so tests can relocate it.

    Returns:
        The directory every scope lives under, which the git service also treats as the
        repository root.
    """
    return _MEMORY_DIR


def _scope_dir(scope: str) -> Path:
    """Returns the memory directory for a scope.

    Args:
        scope (str): The scope key.

    Returns:
        The scope's directory, which need not exist yet.
    """
    return _MEMORY_DIR / scope


def compartment_dir(scope: str, compartment: str) -> Path:
    """Returns the directory holding one compartment's fact files.

    Args:
        scope (str): The scope key.
        compartment (str): One of `global`, `dm` or `g/<guild_id>`.

    Returns:
        The compartment directory, which need not exist yet.

    Raises:
        ValueError: The compartment is not one of the three known shapes. Callers build
            compartments from ints, so this only fires on a programming error — but it
            is the single chokepoint between a compartment string and a filesystem
            path, so it refuses rather than joining whatever it was handed.
    """
    if not _COMPARTMENT_RE.match(compartment):
        raise ValueError(f"invalid memory compartment: {compartment!r}")
    return _scope_dir(scope=scope) / compartment


def list_compartments(scope: str) -> list[str]:
    """Returns every compartment that exists on disk for a scope, in a stable order.

    Ordered `global`, ascending guild id, then `dm` — deterministic rather than
    `iterdir` order, so the assembled document (and therefore the prompt prefix) does
    not reshuffle between reads of an unchanged scope.

    Args:
        scope (str): The scope key.

    Returns:
        The compartment keys that have a directory on disk, empty for an unwritten scope.
    """
    scope_dir = _scope_dir(scope=scope)
    found: list[str] = []
    if (scope_dir / GLOBAL_COMPARTMENT).is_dir():
        found.append(GLOBAL_COMPARTMENT)
    guild_root = scope_dir / _GUILD_DIR_NAME
    if guild_root.is_dir():
        guild_ids = sorted(
            int(child.name)
            for child in guild_root.iterdir()
            if child.is_dir() and child.name.isdigit()
        )
        found.extend(guild_compartment(guild_id=guild_id) for guild_id in guild_ids)
    if (scope_dir / DM_COMPARTMENT).is_dir():
        found.append(DM_COMPARTMENT)
    return found


def _scope_has_memory(scope: str) -> bool:
    """Whether a scope already has live memory on disk.

    Checks the single-file tiers first: they answer for most scopes without touching the
    compartment tree, which costs one `iterdir` per compartment. `detail.md` counts even
    though it is never injected — it is the evidence a rebuild reconstructs everything
    from (`regeneration_has_evidence` reads it), and a scope that has gone quiet since
    its last consolidation holds nothing else, which is the steady state for a server.

    Args:
        scope (str): The scope key.

    Returns:
        True when any tier holds a file for this scope.
    """
    scope_dir = _scope_dir(scope=scope)
    if any((scope_dir / name).is_file() for name in ("raw.md", "tone.md", "detail.md")):
        return True
    return any(
        _fact_paths(directory=compartment_dir(scope=scope, compartment=compartment))
        for compartment in list_compartments(scope=scope)
    )


def iter_scopes() -> list[str]:
    """Returns every scope with on-disk memory (user = flat, server = under the bot dir).

    Walks `data/memories/`: a top-level directory holding memory is a user scope
    (`<user_id>`), and the child directories of `bot_memories/` holding memory are the
    `bot_memories/<server_id>` server scopes. Used by the restart consolidation sweep
    to find scopes whose raw backlog still needs digesting even when no extraction job
    is pending for them.

    `bot_memories` is the only directory descended into, and dot directories are
    skipped outright (the store is itself a git work tree), so a stray directory — or a
    symlink to this one — can never hand the sweep the same memory twice under another
    name.

    Returns:
        Every scope key with memory on disk, in sorted directory order.
    """
    scopes: list[str] = []
    if not _MEMORY_DIR.is_dir():
        return scopes
    for top in sorted(_MEMORY_DIR.iterdir()):
        if not top.is_dir() or top.name.startswith("."):
            continue
        if top.name != BOT_MEMORY_DIR_NAME:
            if _scope_has_memory(scope=top.name):
                scopes.append(top.name)
            continue
        for nested in sorted(top.iterdir()):
            scope = f"{BOT_MEMORY_DIR_NAME}/{nested.name}"
            if nested.is_dir() and _scope_has_memory(scope=scope):
                scopes.append(scope)
    return scopes


def _raw_path(scope: str) -> Path:
    """Returns the raw extraction accumulation path for a scope.

    One file for the whole scope, deliberately not split per compartment: it is staging
    and is never injected, so partitioning it would only multiply the job rows and the
    consolidation cooldowns by the number of compartments.

    Args:
        scope (str): The scope key.

    Returns:
        The scope's `raw.md`.
    """
    return _scope_dir(scope=scope) / "raw.md"


def _detail_path(scope: str) -> Path:
    """Returns the cold-tier detail path for consumed and evicted raw entries.

    Args:
        scope (str): The scope key.

    Returns:
        The scope's `detail.md`.
    """
    return _scope_dir(scope=scope) / "detail.md"


def _tone_path(scope: str) -> Path:
    """Returns the per-user tone-preference note path for a scope.

    Args:
        scope (str): The scope key.

    Returns:
        The scope's `tone.md`, deliberately beside the compartment tree rather than
        inside it.
    """
    return _scope_dir(scope=scope) / "tone.md"


def _read_text(path: Path) -> str:
    """Reads a memory file, treating a missing file as empty.

    Args:
        path (Path): The file to read.

    Returns:
        The file's text, or "" when it does not exist.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _fact_paths(directory: Path) -> list[Path]:
    """Returns the fact files in one compartment directory, missing dir counting as none.

    The `.md` test also excludes the `.md.tmp` sibling a write is midway through, so a
    half-written fact is never listed, parsed or counted.

    Args:
        directory (Path): The compartment directory.

    Returns:
        Its fact files sorted by name, or an empty list when the directory is absent.
    """
    try:
        return sorted(path for path in directory.iterdir() if path.suffix == ".md")
    except FileNotFoundError:
        return []


def scope_lock(scope: str) -> asyncio.Lock:
    """Returns the per-scope lock that serializes memory file writes.

    Nothing in this module acquires it; a writer holds it around its whole batch, since
    a consolidation is N renames rather than one atomic replace and a reader (or the git
    committer) must not see a half-applied batch.

    Args:
        scope (str): The scope key.

    Returns:
        This scope's lock on the running event loop, rebuilt when the loop changes.
    """
    return _scope_locks.setdefault(key=scope, default=asyncio.Lock())


def mark_cleared(scope: str) -> None:
    """Records a manual memory clear so older in-flight updates abort their writes.

    Process-local and monotonic, so it is only comparable against a `time.monotonic()`
    reading taken in the same process; the restart-proof half of a clear is the reply.db
    tombstone, not this.

    Args:
        scope (str): The scope key.
    """
    _cleared_at[scope] = time.monotonic()


def cleared_since(scope: str, started_at: float) -> bool:
    """Whether the scope's memory was cleared at or after `started_at` (time.monotonic).

    Args:
        scope (str): The scope key.
        started_at (float): `time.monotonic()` reading from when the caller's work began.

    Returns:
        True when a clear landed at or after that reading, meaning the caller must drop
        its result instead of writing it back.
    """
    cleared = _cleared_at.get(scope)
    return cleared is not None and cleared >= started_at


def _bump_generation(scope: str) -> None:
    """Invalidates the scope's cached documents after a write.

    Args:
        scope (str): The scope key.
    """
    _write_generation[scope] = _write_generation.get(scope, 0) + 1


def read_facts(scope: str, compartment: str) -> list[MemoryFact]:
    """Returns one compartment's parseable facts; unreadable files are skipped.

    A file can vanish between the listing and the read (a concurrent delete, an offline
    edit), and a malformed one is reported by `parse_fact_file`; either way the rest of
    the compartment still reaches the reply.

    Args:
        scope (str): The scope key.
        compartment (str): The compartment to read. It is passed down to
            `parse_fact_file` as the authoritative one, so a file whose stored
            compartment disagrees with the directory it sits in is dropped.

    Returns:
        The facts that parsed, in filename order.
    """
    facts: list[MemoryFact] = []
    for path in _fact_paths(directory=compartment_dir(scope=scope, compartment=compartment)):
        text = _read_text(path=path)
        if not text:
            continue
        fact = parse_fact_file(text=text, compartment=compartment)
        if fact is not None:
            facts.append(fact)
    return facts


def read_memory_document(
    scope: str,
    compartments: list[str],
    flavor: MemoryFlavor,
    max_chars: int = MEMORY_INJECTION_MAX_CHARS,
) -> str:
    """Returns the injectable document for one scope read through `compartments`.

    The read path's single entry point. Facts from every requested compartment are merged
    and rendered as one document, competing for the size cap by recency inside each
    section rather than by which compartment they came from, so a large shared tier
    cannot silently starve a guild's own memory.

    Cached on the scope's write generation: a repeat read of an unchanged scope returns
    without touching the filesystem, which is what keeps eight per-reply lookups
    affordable now that one fact is one file.

    Args:
        scope (str): The scope key.
        compartments (list[str]): The compartments this conversation may open, chosen by
            the caller's own boundary rule; nothing outside them is ever read.
        flavor (MemoryFlavor): Which section vocabulary and headings to render under.
        max_chars (int): Ceiling on the rendered document, past which rendering stops
            with a truncation notice and nothing is deleted.

    Returns:
        The merged document, or "" when no requested compartment holds a fact.
    """
    key = (scope, tuple(compartments), flavor, max_chars)
    cached = _render_cache.get(key)
    generation = _write_generation.get(scope, 0)
    if cached is not None and cached[0] == generation:
        return cached[1]
    facts = [
        fact
        for compartment in compartments
        for fact in read_facts(scope=scope, compartment=compartment)
    ]
    document = render_memory_document(facts=facts, flavor=flavor, max_chars=max_chars)
    if len(_render_cache) >= RENDER_CACHE_MAX_ENTRIES:
        # Whole-cache reset rather than an LRU: entries are cheap to rebuild and the
        # working set is one entry per (scope, reading context), so the bound is only
        # here to stop a long-lived process accumulating stale keys forever.
        _render_cache.clear()
    _render_cache[key] = (generation, document)
    return document


def write_fact(scope: str, fact: MemoryFact) -> None:
    """Atomically writes one fact file into its compartment.

    Tmp file then `os.replace`, so a reader racing the write sees either the old fact or
    the new one, never a partial file. There is no separate update path: a delta that
    rewrites a fact carries the same id, so it overwrites that one file.

    Args:
        scope (str): The scope key.
        fact (MemoryFact): The fact to write; its `fact_id` is the filename stem and its
            own `compartment` picks (and creates) the directory.
    """
    directory = compartment_dir(scope=scope, compartment=fact.compartment)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fact.fact_id}.md"
    tmp_path = path.with_suffix(".md.tmp")
    tmp_path.write_text(data=render_fact_file(fact=fact), encoding="utf-8")
    os.replace(src=tmp_path, dst=path)
    _bump_generation(scope=scope)


def delete_fact(scope: str, compartment: str, fact_id: str) -> bool:
    """Deletes one fact file, returning whether it existed.

    Args:
        scope (str): The scope key.
        compartment (str): The compartment holding the file.
        fact_id (str): The fact's id, which is the filename stem.

    Returns:
        True when a file was removed. An already-absent file is False rather than an
        error, so a replayed delta and the freshness sweep are both idempotent.
    """
    path = compartment_dir(scope=scope, compartment=compartment) / f"{fact_id}.md"
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    _bump_generation(scope=scope)
    return True


def read_owner(scope: str) -> MemoryOwner:
    """Recovers the stored owner identity from any one of the scope's facts.

    Offline regeneration has no Discord context to rebuild the identity from, so it
    preserves what the last online write stamped. A scope with no facts yet falls back
    to the id in the scope key and an empty name, which the next online write replaces.

    Args:
        scope (str): The scope key.

    Returns:
        The identity carried by the first fact that has a name, or the scope's own id
        with an empty name.
    """
    owner_id = scope_owner_id(scope=scope)
    for compartment in list_compartments(scope=scope):
        for fact in read_facts(scope=scope, compartment=compartment):
            if fact.owner_name:
                return MemoryOwner(owner_id=fact.owner_id, owner_name=fact.owner_name)
    return MemoryOwner(owner_id=owner_id, owner_name="")


def read_tone(scope: str) -> str:
    """Returns the per-user tone-preference note, or "" when there is none.

    Read on every reply for the message author and injected as a low-authority
    context block, so it is a plain short markdown note with no header to strip.
    Cross-server safe by construction: consolidation writes only persona-independent
    delivery qualities here, never facts, which is why it is the one tier that sits
    outside the compartment tree.

    Args:
        scope (str): The scope key.

    Returns:
        The note, stripped, or "" when the scope has none.
    """
    return _read_text(path=_tone_path(scope=scope)).strip()


def write_tone(scope: str, content: str) -> None:
    """Atomically replaces the per-user tone-preference note.

    Shortness is enforced by the consolidation prompt, not a compaction pass; the
    `TONE_FILE_MAX_BYTES` clamp is only a store-level backstop so a misbehaving
    rewrite cannot grow the always-injected note unbounded. There is no backup
    generation: the note is best-effort and the next consolidation repairs a bad
    write.

    Args:
        scope (str): The scope key.
        content (str): The whole note, replacing whatever is stored; anything past
            `TONE_FILE_MAX_BYTES` is cut off.
    """
    _scope_dir(scope=scope).mkdir(parents=True, exist_ok=True)
    rendered = content.strip()
    encoded = rendered.encode("utf-8")
    if len(encoded) > TONE_FILE_MAX_BYTES:
        rendered = encoded[:TONE_FILE_MAX_BYTES].decode(encoding="utf-8", errors="ignore")
    tone_path = _tone_path(scope=scope)
    tmp_path = tone_path.with_suffix(".md.tmp")
    tmp_path.write_text(data=rendered + "\n", encoding="utf-8")
    os.replace(src=tmp_path, dst=tone_path)


def append_raw_entry(scope: str, entry_text: str) -> None:
    """Appends one timestamped raw entry, archiving the oldest entries on overflow.

    Headers carry only the timestamp. Author identity must stay confined to the fact
    files (raw entries flow verbatim into the detail file); an observation body may
    carry the code-stamped conversation source (`- source: guild <id>` / `dm`) — that is
    provenance of where a conversation happened, not identity.

    Args:
        scope (str): The scope key.
        entry_text (str): The rendered observations for one turn, without a header.
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

    The detail file preserves raw entry content verbatim; owner identity stays confined
    to the fact files. Append-mode IO keeps the common write O(1) in the file size; once
    the file outgrows `DETAIL_FILE_MAX_BYTES` the oldest entries are trimmed away, which
    is safe because content past the consolidation read window is unreachable by every
    consumer anyway.

    Args:
        scope (str): The scope key.
        text (str): One or more raw entry blocks, headers included; an empty one is a
            no-op and does not create the file.
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

    Args:
        path (Path): The detail file to rewrite in place.
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
    stays O(window) as the multi-megabyte detail file grows. The window is aligned
    to the first raw-entry header inside the tail so a partial entry never
    leads the result; when no header lands inside the window (e.g. one giant
    entry) the raw tail is returned as a best effort.

    Args:
        scope (str): The scope key.
        max_chars (int): Character ceiling on the returned window.

    Returns:
        The newest evidence, stripped, or "" when the scope has no detail file.
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
    """Returns how many raw entries are waiting for consolidation.

    One of the two consolidation triggers, and what `/memory show` reports as pending
    background observations.

    Args:
        scope (str): The scope key.

    Returns:
        The number of `## <timestamp>` blocks in the raw file, zero when there is none.
    """
    return len(_split_raw_entries(text=_read_text(path=_raw_path(scope=scope))))


def raw_file_bytes(scope: str) -> int:
    """Returns the raw file size in bytes, with a missing file counting as zero.

    The other consolidation trigger: a verbose batch consolidates early, bypassing the
    entry-count threshold and the cooldown.

    Args:
        scope (str): The scope key.

    Returns:
        The UTF-8 size of the raw file's text.
    """
    return len(_read_text(path=_raw_path(scope=scope)).encode("utf-8"))


def detail_file_bytes(scope: str) -> int:
    """Returns the cold-tier detail file size in bytes, missing file counting as zero.

    Uses ``stat`` rather than reading the whole file so an evidence-presence
    check stays O(1) even when the detail file is near its multi-megabyte cap.

    Args:
        scope (str): The scope key.

    Returns:
        The detail file's size on disk.
    """
    path = _detail_path(scope=scope)
    return path.stat().st_size if path.is_file() else 0


def read_raw_entries(scope: str) -> str:
    """Returns the raw file text for consolidation input.

    Args:
        scope (str): The scope key.

    Returns:
        The whole raw file, stripped, or "" when nothing is staged.
    """
    return _read_text(path=_raw_path(scope=scope)).strip()


def clear_raw(scope: str) -> None:
    """Deletes the raw file after a consolidation consumed it.

    Only ever called once the batch has been appended to the detail file, so the
    evidence outlives the staging tier.

    Args:
        scope (str): The scope key.
    """
    _raw_path(scope=scope).unlink(missing_ok=True)


def clear_tone(scope: str) -> None:
    """Deletes the tone note when a full-evidence rebuild found no tone signal.

    Only the evidence-complete regeneration path may call this: an incremental
    consolidation's empty tone output merely means "no tone signal in this batch"
    and must never remove the note.

    Args:
        scope (str): The scope key.
    """
    _tone_path(scope=scope).unlink(missing_ok=True)


def clear_memory(scope: str) -> bool:
    """Deletes the scope's memory files and flags older in-flight updates to abort.

    A test-only convenience over `mark_cleared` + `delete_memory_files`; nothing under
    `src/` calls it. The pipeline clear drives those two itself because it owns a wider
    boundary around its awaited reply.db tombstone, so do not route a production clear
    back through here.

    Args:
        scope (str): The scope key.

    Returns:
        True when at least one memory file existed and was removed.
    """
    mark_cleared(scope=scope)
    return delete_memory_files(scope=scope)


def delete_memory_files(scope: str) -> bool:
    """Deletes the scope's memory files without moving its clear boundary.

    Walks the compartment tree as well as the three single-file tiers, removing only
    `.md` files and the `.md.tmp` leftovers a crash between a tmp write and its
    `os.replace` can strand — never a foreign file that shares the directory. Empty
    directories are then removed bottom-up; a non-empty or missing one is left for
    offline maintenance instead of failing the clear.

    Its one production caller, the pipeline clear, deliberately holds no `scope_lock`
    (that could wait minutes behind a consolidation), so an in-flight writer is stopped
    by `cleared_since` rather than by exclusion here. Every deletion is idempotent, which
    is what lets a partial failure be retried by simply running it again.

    Args:
        scope (str): The scope key.

    Returns:
        True when at least one memory file existed and was removed.
    """
    scope_dir = _scope_dir(scope=scope)
    removed = False
    for name in ("raw.md", "detail.md", "tone.md"):
        try:
            (scope_dir / name).unlink()
            removed = True
        except FileNotFoundError:
            # Already gone (e.g. offline maintenance); deletion stays idempotent
            # without the exists()-then-unlink() race.
            continue
        finally:
            (scope_dir / f"{name}.tmp").unlink(missing_ok=True)
    for compartment in list_compartments(scope=scope):
        directory = compartment_dir(scope=scope, compartment=compartment)
        for path in _fact_paths(directory=directory):
            path.unlink(missing_ok=True)
            removed = True
        for leftover in directory.glob("*.md.tmp"):
            leftover.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            directory.rmdir()
    with contextlib.suppress(OSError):
        (scope_dir / _GUILD_DIR_NAME).rmdir()
    with contextlib.suppress(OSError):
        scope_dir.rmdir()
    _bump_generation(scope=scope)
    return removed


def _split_raw_entries(text: str) -> list[str]:
    """Splits raw file text into stripped per-entry blocks including headers.

    Args:
        text (str): Raw or detail file text; both use the same entry format.

    Returns:
        One block per `## <timestamp>` header, empty when the text has no header at all
        (so anything written before the first one is dropped rather than merged into the
        first entry).
    """
    starts = [match.start() for match in _RAW_ENTRY_HEADER_RE.finditer(text)]
    if not starts:
        return []
    bounds = [*starts, len(text)]
    blocks = [text[begin:end].strip() for begin, end in itertools.pairwise(bounds)]
    return [block for block in blocks if block]


def _entries_bytes(entries: list[str]) -> int:
    """Returns the rendered raw-file size for a list of entry blocks.

    Args:
        entries (list[str]): The blocks, in the order they would be written.

    Returns:
        The UTF-8 size they occupy once joined by the blank-line separator.
    """
    return len("\n\n".join(entries).encode("utf-8"))
