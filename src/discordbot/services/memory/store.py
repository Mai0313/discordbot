"""File-backed storage for long-term memory, keyed by an opaque scope.

A scope is a relative path under ``data/memories/`` that doubles as the registry key.
Per-user memory uses ``user_scope(user_id)`` (``<user_id>``); the bot's own per-server
memory uses ``server_scope(server_id)`` (``bot_memories/<server_id>``).

Inside a scope the consolidated tier is **one file per fact**, filed under the
compartment that decides who may read it — ``global/`` (safe anywhere),
``g/<guild_id>/`` (that guild only) and ``dm/`` (the owner's own DMs). The path *is*
the privacy boundary: reading for guild G is ``global/`` plus ``g/<G>/``, two joins and
a containment check, with no read-time content filter to get wrong. A server scope has
exactly one compartment (``global/``) because a server memory is per-guild by
construction and its evidence carries no source to route by.

The remaining tiers are per-scope and unchanged: ``raw.md`` accumulates phase-1 entries
until consolidation consumes them, ``detail.md`` is the append-only cold evidence log
(read as a tail window, trimmed to a hard byte cap), and ``tone.md`` is the short
always-read note of how the user wants the bot to sound.

IO is synchronous, which one fact per file would otherwise make untenable on the reply
path: ``render_memory_document`` is cached under a per-scope generation counter that
every write bumps, so a repeat read costs no syscalls at all. The counter is exact
because every write in this process goes through here under ``scope_lock``; editing the
tree from outside while the bot runs is not supported (nor is it today, for
``_cleared_at``).
"""

import os
import re
import time
import asyncio
from pathlib import Path
from datetime import UTC, datetime
import itertools
import contextlib

import logfire

from discordbot.typings.memory import MemoryFact, MemoryOwner
from discordbot.utils.asyncio_locks import LoopLocalRegistry
from discordbot.services.memory.facts import (
    FACT_ID_RE,
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
    """Returns the storage scope for one user's memory."""
    return str(user_id)


def server_scope(server_id: int) -> str:
    """Returns the storage scope for the bot's memory of one server."""
    return f"{BOT_MEMORY_DIR_NAME}/{server_id}"


def guild_compartment(guild_id: int) -> str:
    """Returns the compartment holding facts readable only inside one guild."""
    return f"{_GUILD_DIR_NAME}/{guild_id}"


def scope_owner_id(scope: str) -> int:
    """Returns the Discord id a scope belongs to (a user id, or a server id)."""
    return int(scope.rsplit("/", maxsplit=1)[-1])


def memory_root() -> Path:
    """Returns the store root, read through this accessor so tests can relocate it."""
    return _MEMORY_DIR


def _scope_dir(scope: str) -> Path:
    """Returns the memory directory for a scope."""
    return _MEMORY_DIR / scope


def compartment_dir(scope: str, compartment: str) -> Path:
    """Returns the directory holding one compartment's fact files.

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

    The compartment tier is parsed rather than listed, so a file no reader can parse does
    not answer for a scope on its own. Listing it kept such a scope on `iter_scopes`
    permanently, handing it to the restart consolidation sweep and to
    `scripts/regen_memories.py` on every run with nothing either could do about it. The
    parse costs nothing in practice: a scope that has ever consolidated has a `detail.md`
    and answers above, so what reaches the walk is a leftover directory with no fact in
    it (every one of them, in the live store).
    """
    scope_dir = _scope_dir(scope=scope)
    if any((scope_dir / name).is_file() for name in ("raw.md", "tone.md", "detail.md")):
        return True
    return any(
        read_facts(scope=scope, compartment=compartment)
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
    """Returns the raw extraction accumulation path for a scope."""
    return _scope_dir(scope=scope) / "raw.md"


def _detail_path(scope: str) -> Path:
    """Returns the cold-tier detail path for consumed and evicted raw entries."""
    return _scope_dir(scope=scope) / "detail.md"


def _tone_path(scope: str) -> Path:
    """Returns the per-user tone-preference note path for a scope."""
    return _scope_dir(scope=scope) / "tone.md"


def _read_text(path: Path) -> str:
    """Reads a memory file, treating a missing file as empty."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _fact_paths(directory: Path) -> list[Path]:
    """Returns the fact files in one compartment directory, missing dir counting as none.

    The `is_file` test is load-bearing: a *directory* whose name ends in `.md` otherwise
    reaches `_read_text`, which catches only `FileNotFoundError`, so one hand-made
    directory inside a compartment takes down every reader that walks the tree.
    """
    try:
        return sorted(
            path for path in directory.iterdir() if path.suffix == ".md" and path.is_file()
        )
    except FileNotFoundError:
        return []


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


def _bump_generation(scope: str) -> None:
    """Invalidates the scope's cached documents after a write."""
    _write_generation[scope] = _write_generation.get(scope, 0) + 1


def read_facts(scope: str, compartment: str) -> list[MemoryFact]:
    """Returns one compartment's parseable facts; unreadable files are skipped.

    A file can vanish between the listing and the read (a concurrent delete, an offline
    edit), and a malformed one is reported by `parse_fact_file`; either way the rest of
    the compartment still reaches the reply.
    """
    facts: list[MemoryFact] = []
    for path in _fact_paths(directory=compartment_dir(scope=scope, compartment=compartment)):
        try:
            text = _read_text(path=path)
        except (OSError, UnicodeDecodeError) as error:
            # A file that cannot even be decoded is skipped like one that cannot be
            # parsed, rather than raised: `_scope_has_memory` reads through here, so one
            # hand edit saved in the wrong encoding would otherwise take down the whole
            # restart sweep and stop `scripts/regen_memories.py` starting at all — the
            # tool an operator reaches for to repair exactly that store.
            logfire.warn(
                "Memory fact file could not be read; skipping",
                compartment=compartment,
                error_type=type(error).__name__,
            )
            continue
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

    This is the read path's single entry point and the direct replacement for the old
    whole-file `read_main_memory`. Facts from every requested compartment are merged and
    rendered as one document, competing for the size cap by recency inside each section
    rather than by which compartment they came from, so a large shared tier cannot
    silently starve a guild's own memory.

    Cached on the scope's write generation: a repeat read of an unchanged scope returns
    without touching the filesystem, which is what keeps eight per-reply lookups
    affordable now that one fact is one file.
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
    """Atomically writes one fact file into its compartment."""
    directory = compartment_dir(scope=scope, compartment=fact.compartment)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fact.fact_id}.md"
    tmp_path = path.with_suffix(".md.tmp")
    tmp_path.write_text(data=render_fact_file(fact=fact), encoding="utf-8")
    os.replace(src=tmp_path, dst=path)
    _bump_generation(scope=scope)


def delete_fact(scope: str, compartment: str, fact_id: str) -> bool:
    """Deletes one fact file, returning whether it existed."""
    path = compartment_dir(scope=scope, compartment=compartment) / f"{fact_id}.md"
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    _bump_generation(scope=scope)
    return True


def _is_store_file(path: Path) -> bool:
    r"""Whether one entry of a compartment directory is a file the store itself wrote.

    Matched by NAME — `<fact id>.md`, or the `.md.tmp` a crash between `write_fact`'s tmp
    write and its `os.replace` can strand — the way the media reaper matches its own
    files. Every name the store mints is a `mint_fact_id` digest, so a `notes.md` an
    operator dropped in beside the facts fails the test and a prune cannot take it.
    `fullmatch`, not `match`, for the reason the reaper uses it too: `$` also matches
    before a trailing newline, so `<fact id>\\n.md` would otherwise pass for one of ours.
    """
    stem, _, suffix = path.name.partition(".")
    return path.is_file() and suffix in {"md", "md.tmp"} and FACT_ID_RE.fullmatch(stem) is not None


def unaccounted_files(scope: str, compartment: str) -> list[str]:
    """Returns the names in a compartment directory that the store never wrote.

    Anything not named the way the store names a fact file arrived from outside — a
    hand-dropped note, a backup copy, an editor's swap file — and `_fact_paths` globs
    `*.md`, so no code path can see most of them today. That is what lets a rebuild
    report a compartment replaced while such a file sits in it.

    Nothing here removes them, because a rebuild REPLACES a compartment and may only
    take what the store itself put there; naming them is what makes the difference
    visible instead of assumed. `delete_memory_files` is the opposite contract and takes
    every `.md` in the tree, this one included — a clear is a wipe its owner asked for,
    so sparing a file that might carry their memory would be the wrong answer there.
    """
    try:
        # Materialized inside the guard: `iterdir` is a generator, so a missing
        # directory would otherwise raise out of the comprehension below instead.
        children = list(compartment_dir(scope=scope, compartment=compartment).iterdir())
    except FileNotFoundError:
        return []
    return sorted(path.name for path in children if not _is_store_file(path=path))


def prune_compartment(scope: str, compartment: str, keep: set[str]) -> list[str]:
    """Reduces a compartment to `keep`, returning what it could not account for.

    The rebuild's replace pass runs through here rather than over the facts it read
    back. `read_facts` skips a file `parse_fact_file` rejects — a hand edit, an
    interrupted write, a stored `compartment` disagreeing with the directory holding it
    — so a snapshot taken there leaves exactly those files standing through a rebuild
    that reports the compartment replaced. Working off the listing makes "replaced" mean
    the directory instead of its readable part, which is the only reading that survives
    a path that already drops perfectly good facts by not re-emitting them.

    The `.md.tmp` leftovers go with them: they are the store's own, and nothing in THIS
    process can be mid-write, since every writer holds the same scope lock. Out of
    process is the offline rebuild's standing caveat rather than a new one — it is why
    `scripts/regen_memories.py` opens by telling the operator to stop the bot. An emptied
    directory
    is then removed, so a compartment a rebuild emptied stops being one
    `list_compartments` reports, and stops costing a consolidation call on every later
    rebuild.
    """
    directory = compartment_dir(scope=scope, compartment=compartment)
    removed = False
    try:
        children = list(directory.iterdir())
    except FileNotFoundError:
        return []
    for path in children:
        if not _is_store_file(path=path):
            continue
        if path.suffix == ".md" and path.stem in keep:
            continue
        path.unlink(missing_ok=True)
        removed = True
    if removed:
        _bump_generation(scope=scope)
    with contextlib.suppress(OSError):
        directory.rmdir()
        if compartment.startswith(f"{_GUILD_DIR_NAME}/"):
            # A `g/<id>` leaves the `g/` parent behind, which `delete_memory_files`
            # removes for the same reason. Inside the same suppression on purpose: a
            # compartment that survived is what stops the parent being tried at all.
            directory.parent.rmdir()
    return unaccounted_files(scope=scope, compartment=compartment)


def read_owner(scope: str) -> MemoryOwner:
    """Recovers the stored owner identity from any one of the scope's facts.

    Offline regeneration has no Discord context to rebuild the identity from, so it
    preserves what the last online write stamped. A scope with no facts yet falls back
    to the id in the scope key and an empty name, which the next online write replaces.
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
    """
    return _read_text(path=_tone_path(scope=scope)).strip()


def write_tone(scope: str, content: str) -> None:
    """Atomically replaces the per-user tone-preference note.

    Shortness is enforced by the consolidation prompt, not a compaction pass; the
    `TONE_FILE_MAX_BYTES` clamp is only a store-level backstop so a misbehaving
    rewrite cannot grow the always-injected note unbounded. There is no backup
    generation: the note is best-effort and the next consolidation repairs a bad
    write.
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


def clear_tone(scope: str) -> None:
    """Deletes the tone note when a full-evidence rebuild found no tone signal.

    Only the evidence-complete regeneration path may call this: an incremental
    consolidation's empty tone output merely means "no tone signal in this batch"
    and must never remove the note.
    """
    _tone_path(scope=scope).unlink(missing_ok=True)


def clear_memory(scope: str) -> bool:
    """Deletes the scope's memory files and flags older in-flight updates to abort.

    A test-only convenience over `mark_cleared` + `delete_memory_files`; nothing under
    `src/` calls it. The pipeline clear drives those two itself because it owns a wider
    boundary around its awaited reply.db tombstone, so do not route a production clear
    back through here.

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
