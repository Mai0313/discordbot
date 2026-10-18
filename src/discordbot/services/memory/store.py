"""File-backed storage for long-term memory, keyed by an opaque scope.

A scope is a relative path under `data/memories/` that doubles as the registry key.
Per-user memory uses `user_scope(user_id)` (`<user_id>`); the bot's own per-server
memory uses `server_scope(server_id)` (`bot_memories/<server_id>`).

Inside a scope the consolidated tier is **one file per fact**, filed under the
compartment that decides who may read it — `global/` (safe anywhere),
`g/<guild_id>/` (that guild only) and `dm/` (the owner's own DMs). The path *is*
the privacy boundary: reading for guild G is `global/` plus `g/<G>/`, two joins each
guarded by `_COMPARTMENT_RE`, with no read-time content filter to get wrong. A server
scope has one compartment (`global/`) because a server memory is per-guild by
construction and its evidence carries no source to route by.

The remaining tiers are per-scope: `raw.md` accumulates phase-1 entries
until consolidation consumes them, `detail.md` is the append-only cold evidence log
(read as a tail window, trimmed to a hard byte cap, and losing only what a forget released),
and `tone.md` is the short always-read note of how the user wants the bot to sound.

IO is synchronous, which one fact per file would otherwise make untenable on the reply
path: `read_memory_document` is cached under a per-scope generation counter that
every fact write bumps, so a repeat read costs no syscalls at all. The counter is exact
because every write in this process goes through here; editing the tree from outside
while the bot runs is not supported.
"""

import os
import re
import time
import asyncio
from pathlib import Path
import contextlib
from collections.abc import Callable

import logfire
from pydantic import Field, BaseModel, ConfigDict

from discordbot.typings.memory import MemoryFact, MemoryOwner, MemoryFlavor
from discordbot.utils.asyncio_locks import LoopLocalRegistry
from discordbot.services.memory.facts import (
    FACT_ID_RE,
    parse_fact_file,
    render_fact_file,
    render_memory_document,
)
from discordbot.typings.context_budgets import (
    MEMORY_INJECTION_MAX_CHARS,
    MEMORY_DETAIL_CONTEXT_MAX_CHARS,
)
from discordbot.services.memory.constants import (
    RAW_FILE_MAX_BYTES,
    TONE_FILE_MAX_BYTES,
    DETAIL_FILE_MAX_BYTES,
    RENDER_CACHE_MAX_ENTRIES,
    DETAIL_FILE_TRIM_TARGET_BYTES,
)
from discordbot.services.memory.raw_entries import (
    RAW_ENTRY_HEADER_RE,
    stamp_entry,
    split_raw_entries,
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

# Per-scope file-write locks, rebuilt per event loop by the shared registry.
_scope_locks: LoopLocalRegistry[str, asyncio.Lock] = LoopLocalRegistry()
# Manual-clear timestamps; monotonic, so it is not loop-keyed.
_cleared_at: dict[str, float] = {}
# Per-scope write counter and the rendered-document cache keyed on it. Not loop-bound
# (plain dicts, no asyncio primitive). All three are process-global and describe one store
# root, so relocating that root has to reset them.
_write_generation: dict[str, int] = {}
_render_cache: dict[tuple[str, tuple[str, ...], MemoryFlavor, int], tuple[int, str]] = {}


def user_scope(user_id: int) -> str:
    """Returns the storage scope for one user's memory.

    Args:
        user_id (int): Discord user id.
    """
    return str(user_id)


def server_scope(server_id: int) -> str:
    """Returns the storage scope for the bot's memory of one server.

    Args:
        server_id (int): Discord guild id.
    """
    return f"{BOT_MEMORY_DIR_NAME}/{server_id}"


def guild_compartment(guild_id: int) -> str:
    """Returns the compartment holding facts readable only inside one guild.

    Args:
        guild_id (int): Discord guild id.
    """
    return f"{_GUILD_DIR_NAME}/{guild_id}"


def compartment_guild_id(compartment: str) -> int | None:
    """Returns the guild id a `guild_compartment` is readable in, or None for any other shape.

    Args:
        compartment (str): Compartment name.
    """
    parent, _, guild_id = compartment.partition("/")
    return int(guild_id) if parent == _GUILD_DIR_NAME and guild_id.isdecimal() else None


def scope_owner_id(scope: str) -> int:
    """Returns the Discord user or guild id `scope` belongs to.

    Args:
        scope (str): Memory scope key.
    """
    return int(scope.rsplit("/", maxsplit=1)[-1])


def flavor_of(scope: str) -> MemoryFlavor:
    """Returns the memory flavor of `scope` (`'server'` when it contains `'/'`, else `'user'`).

    Args:
        scope (str): Memory scope key.
    """
    return "server" if "/" in scope else "user"


def memory_root() -> Path:
    """Returns the store root, read through this accessor so it can be relocated."""
    return _MEMORY_DIR


def _scope_dir(scope: str) -> Path:
    """Returns the memory directory for a scope."""
    return _MEMORY_DIR / scope


def compartment_dir(scope: str, compartment: str) -> Path:
    """Returns the directory holding one compartment's fact files.

    Args:
        scope (str): Memory scope key.
        compartment (str): Compartment name (`'global'`, `'dm'`, or `'g/<guild_id>'`).

    Raises:
        ValueError: `compartment` does not match `_COMPARTMENT_RE`.
    """
    if not _COMPARTMENT_RE.match(compartment):
        raise ValueError(f"invalid memory compartment: {compartment!r}")
    return _scope_dir(scope=scope) / compartment


def list_compartments(scope: str) -> list[str]:
    """Returns every compartment that exists on disk for `scope`, in a stable order.

    Ordered `global`, ascending guild id, then `dm` so the assembled document does not
    reshuffle between reads of an unchanged scope.

    Args:
        scope (str): Memory scope key.
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
    from, and a scope that has gone quiet since its last consolidation holds nothing else,
    which is the steady state for a server.

    The compartment tier is parsed rather than listed, so a file no reader can parse does
    not answer for a scope on its own: listing would keep such a scope on `iter_scopes`
    forever, handed to every sweep with nothing any of them could do about it. The parse
    costs nothing in practice: a scope that has ever consolidated has a `detail.md` and
    answers above, so what reaches the walk is a leftover directory with no fact in it.
    """
    scope_dir = _scope_dir(scope=scope)
    if any((scope_dir / name).is_file() for name in ("raw.md", "tone.md", "detail.md")):
        return True
    return any(
        read_facts(scope=scope, compartment=compartment)
        for compartment in list_compartments(scope=scope)
    )


def _is_scope_key(key: str) -> bool:
    """Whether a directory key under the store names a scope, logging one that does not.

    The store names a scope directory only after the id `scope_owner_id` parses back out of
    the key, so any other name is a hand-made copy such as a backup. Handed out as a scope,
    it would raise in the first caller that reads its owner.
    """
    if key.rsplit("/", maxsplit=1)[-1].isdecimal():
        return True
    logfire.info("Memory directory is not named after an id; skipping", directory=key)
    return False


def iter_scopes() -> list[str]:
    """Returns every scope with on-disk memory (user = flat, server = under the bot dir).

    Walks `data/memories/`: a top-level directory holding memory is a user scope
    (`<user_id>`), and the child directories of `bot_memories/` holding memory are the
    `bot_memories/<server_id>` server scopes. A directory at either level not named after
    an id is skipped (`_is_scope_key`).

    `bot_memories` is the only directory descended into, and dot directories are
    skipped outright (the store is itself a git work tree), so nested memory anywhere
    else — a stray directory, or a symlink to `bot_memories` — can never hand the sweep
    the same memory under a second name.
    """
    scopes: list[str] = []
    if not _MEMORY_DIR.is_dir():
        return scopes
    for top in sorted(_MEMORY_DIR.iterdir()):
        if not top.is_dir() or top.name.startswith("."):
            continue
        if top.name != BOT_MEMORY_DIR_NAME:
            if _is_scope_key(key=top.name) and _scope_has_memory(scope=top.name):
                scopes.append(top.name)
            continue
        for nested in sorted(top.iterdir()):
            scope = f"{BOT_MEMORY_DIR_NAME}/{nested.name}"
            if nested.is_dir() and _is_scope_key(key=scope) and _scope_has_memory(scope=scope):
                scopes.append(scope)
    return scopes


def _raw_path(scope: str) -> Path:
    """Returns the path where a scope's raw entries accumulate."""
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


def _replace_file(path: Path, text: str) -> None:
    """Replaces a memory file atomically, writing a `.md.tmp` sibling and moving it over `path`.

    The sibling's name is what `delete_memory_files` and `_is_store_file` recognise a stranded
    one by.
    """
    tmp_path = path.with_suffix(".md.tmp")
    tmp_path.write_text(data=text, encoding="utf-8")
    os.replace(src=tmp_path, dst=path)


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
    """Returns the per-scope lock that serializes memory file writes.

    Args:
        scope (str): Memory scope key.
    """
    return _scope_locks.setdefault(key=scope, default=asyncio.Lock())


def mark_cleared(scope: str) -> None:
    """Records a manual memory clear so older in-flight updates abort their writes.

    Args:
        scope (str): Memory scope key being cleared.
    """
    _cleared_at[scope] = time.monotonic()


def cleared_since(scope: str, started_at: float) -> bool:
    """Returns whether `scope`'s memory was cleared at or after `started_at`.

    Args:
        scope (str): Memory scope key.
        started_at (float): `time.monotonic()` timestamp to compare against.
    """
    cleared = _cleared_at.get(scope)
    return cleared is not None and cleared >= started_at


def _bump_generation(scope: str) -> None:
    """Invalidates the scope's cached documents after a write."""
    _write_generation[scope] = _write_generation.get(scope, 0) + 1


def read_facts(scope: str, compartment: str) -> list[MemoryFact]:
    """Returns one compartment's parseable facts, skipping unreadable files.

    Args:
        scope (str): Memory scope key.
        compartment (str): Compartment name to read.
    """
    facts: list[MemoryFact] = []
    for path in _fact_paths(directory=compartment_dir(scope=scope, compartment=compartment)):
        try:
            text = _read_text(path=path)
        except (OSError, UnicodeDecodeError) as error:
            # A file that cannot even be decoded is skipped like one that cannot be
            # parsed, rather than raised: every scope walk reads through here, so one hand
            # edit saved in the wrong encoding would otherwise take them all down — the
            # repair path an operator reaches for included.
            logfire.warn(
                "Memory fact file could not be read; skipping",
                scope=scope,
                compartment=compartment,
                filename=path.name,
                error_type=type(error).__name__,
            )
            continue
        if not text:
            continue
        fact = parse_fact_file(text=text, compartment=compartment, scope=scope, filename=path.name)
        if fact is not None:
            facts.append(fact)
    return facts


def read_memory_document(
    scope: str,
    compartments: list[str],
    flavor: MemoryFlavor,
    max_chars: int = MEMORY_INJECTION_MAX_CHARS,
) -> str:
    """Returns the injectable document for `scope` merged across `compartments`.

    Facts from every requested compartment compete for `max_chars` by recency inside each
    section rather than by compartment, and the rendered result is cached on `scope`'s write
    generation.

    Args:
        scope (str): Memory scope key.
        compartments (list[str]): Compartments readable in the current context.
        flavor (MemoryFlavor): Scope flavor (`'user'` or `'server'`).
        max_chars (int): Character budget before truncation.
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
    """Atomically writes `fact` into its compartment directory and bumps the generation.

    Args:
        scope (str): Memory scope key.
        fact (MemoryFact): Fact to write.
    """
    directory = compartment_dir(scope=scope, compartment=fact.compartment)
    directory.mkdir(parents=True, exist_ok=True)
    _replace_file(path=directory / f"{fact.fact_id}.md", text=render_fact_file(fact=fact))
    _bump_generation(scope=scope)


def delete_fact(scope: str, compartment: str, fact_id: str) -> bool:
    """Deletes one fact file and bumps the scope's generation when it existed.

    Args:
        scope (str): Memory scope key.
        compartment (str): Compartment containing the fact.
        fact_id (str): Fact id (filename stem) to delete.

    Returns:
        True when the fact file existed and was unlinked.
    """
    path = compartment_dir(scope=scope, compartment=compartment) / f"{fact_id}.md"
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    _bump_generation(scope=scope)
    return True


def _is_store_file(path: Path) -> bool:
    r"""Whether one entry of a compartment directory is a file the store itself wrote.

    Matched by name — `<fact id>.md`, or the `.md.tmp` a crash between `write_fact`'s tmp
    write and its `os.replace` can strand. Every name the store mints is a `mint_fact_id`
    digest, so a `notes.md` an operator dropped in beside the facts fails the test and a
    prune cannot take it. `fullmatch`, not `match`: `$` also matches before a trailing
    newline, so `<fact id>\\n.md` would otherwise pass for one of ours.
    """
    stem, _, suffix = path.name.partition(".")
    return path.is_file() and suffix in {"md", "md.tmp"} and FACT_ID_RE.fullmatch(stem) is not None


def unaccounted_files(scope: str, compartment: str) -> list[str]:
    """Returns the entry names in `compartment` that the store itself never wrote.

    A rebuild replaces only files matching `_is_store_file` and leaves foreign entries in
    place (`delete_memory_files` removes every `.md` on a clear instead).

    Args:
        scope (str): Memory scope key.
        compartment (str): Compartment name to inspect.
    """
    try:
        # Materialized inside the guard: `iterdir` is a generator, so a missing
        # directory would otherwise raise out of the comprehension below instead.
        children = list(compartment_dir(scope=scope, compartment=compartment).iterdir())
    except FileNotFoundError:
        return []
    return sorted(path.name for path in children if not _is_store_file(path=path))


class PrunedCompartment(BaseModel):
    """What one prune pass left standing, and what it destroyed unread."""

    model_config = ConfigDict(frozen=True)

    unaccounted: tuple[str, ...] = Field(
        default=(), description="Names the store never wrote, left where they are."
    )
    unreadable: tuple[str, ...] = Field(
        default=(), description="Fact files removed that no reader could have parsed."
    )


def prune_compartment(scope: str, compartment: str, keep: set[str]) -> PrunedCompartment:
    """Reduces `compartment` to the fact ids in `keep` and removes empty directories.

    Walks the directory listing rather than `read_facts` so unreadable `.md` fact files and
    stranded `.md.tmp` files are removed alongside unkept facts, while foreign entries stay
    untouched and are reported in `unaccounted`.

    Args:
        scope (str): Memory scope key.
        compartment (str): Compartment name to prune.
        keep (set[str]): Fact ids to retain.

    Returns:
        The `PrunedCompartment` listing unaccounted entries left behind and unreadable fact
        files removed.
    """
    directory = compartment_dir(scope=scope, compartment=compartment)
    removed = False
    try:
        children = list(directory.iterdir())
    except FileNotFoundError:
        return PrunedCompartment()
    readable = {fact.fact_id for fact in read_facts(scope=scope, compartment=compartment)}
    unreadable: list[str] = []
    for path in children:
        if not _is_store_file(path=path):
            continue
        if path.suffix == ".md":
            if path.stem in keep:
                continue
            if path.stem not in readable:
                unreadable.append(path.name)
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
    return PrunedCompartment(
        unaccounted=tuple(unaccounted_files(scope=scope, compartment=compartment)),
        unreadable=tuple(sorted(unreadable)),
    )


def read_owner(scope: str) -> MemoryOwner:
    """Returns the stored `MemoryOwner` recovered from `scope`'s facts, or a fallback from its id.

    Args:
        scope (str): Memory scope key.
    """
    owner_id = scope_owner_id(scope=scope)
    for compartment in list_compartments(scope=scope):
        for fact in read_facts(scope=scope, compartment=compartment):
            if fact.owner_name:
                return MemoryOwner(owner_id=fact.owner_id, owner_name=fact.owner_name)
    return MemoryOwner(owner_id=owner_id, owner_name="")


def read_tone(scope: str) -> str:
    """Returns the per-user tone-preference note for `scope`, or `""` when absent.

    Args:
        scope (str): Memory scope key.
    """
    return _read_text(path=_tone_path(scope=scope)).strip()


def write_tone(scope: str, content: str) -> None:
    """Atomically replaces `scope`'s tone-preference note, clamped to `TONE_FILE_MAX_BYTES`.

    Args:
        scope (str): Memory scope key.
        content (str): Full markdown tone note to write.
    """
    _scope_dir(scope=scope).mkdir(parents=True, exist_ok=True)
    rendered = content.strip()
    encoded = rendered.encode("utf-8")
    if len(encoded) > TONE_FILE_MAX_BYTES:
        rendered = encoded[:TONE_FILE_MAX_BYTES].decode(encoding="utf-8", errors="ignore")
    _replace_file(path=_tone_path(scope=scope), text=rendered + "\n")


def append_raw_entry(scope: str, entry_text: str) -> None:
    """Appends one timestamped raw entry to `scope`, archiving oldest entries on overflow.

    Args:
        scope (str): Memory scope key.
        entry_text (str): Rendered observation or forget-request body (without timestamp
            header).
    """
    _scope_dir(scope=scope).mkdir(parents=True, exist_ok=True)
    entry = stamp_entry(body=entry_text)
    raw_path = _raw_path(scope=scope)
    combined = f"{_read_text(path=raw_path)}\n\n{entry}"
    entries = split_raw_entries(text=combined)
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
    _replace_file(path=raw_path, text=rendered + "\n")
    if evicted:
        # Move to the detail file only after the raw write succeeded so a
        # failed write cannot retire entries that still live in the raw file.
        append_detail(scope=scope, text="\n\n".join(evicted))


def append_detail(scope: str, text: str) -> None:
    """Appends consumed or evicted raw evidence to `scope`'s cold-tier detail file.

    Trims the oldest entries once the file exceeds `DETAIL_FILE_MAX_BYTES`.

    Args:
        scope (str): Memory scope key.
        text (str): Timestamped entry blocks to append.
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
    target is what amortizes this O(file) rewrite; the write goes through
    tmp + os.replace so a crash cannot leave a half-trimmed file.
    """
    entries = split_raw_entries(text=_read_text(path=path))
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
    _replace_file(path=path, text="\n\n".join(entries[start:]) + "\n")


def rewrite_evidence(scope: str, edit: Callable[[str], str]) -> None:
    """Passes `raw.md` and `detail.md` through `edit`, atomically replacing each file it changes.

    Args:
        scope (str): Memory scope key.
        edit (Callable[[str], str]): Pure transformation over raw or detail file text.
    """
    for path in (_raw_path(scope=scope), _detail_path(scope=scope)):
        text = _read_text(path=path)
        edited = edit(text)
        if edited == text:
            continue
        _replace_file(path=path, text=edited + "\n")


def read_detail_tail(scope: str, max_chars: int) -> str:
    """Returns the newest `detail.md` window of at most `max_chars`, aligned to an entry header.

    Args:
        scope (str): Memory scope key.
        max_chars (int): Maximum characters to return from the tail of `detail.md`.

    Raises:
        UnicodeDecodeError: `detail.md` contains invalid UTF-8 bytes after the continuation
            byte skip.
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
    # A window starting mid-file can open inside a multi-byte character, on at most
    # three of its continuation bytes (0b10xxxxxx); only those are skipped.
    skip = 0
    if size > len(data):
        while skip < min(3, len(data)) and data[skip] & 0xC0 == 0x80:
            skip += 1
    text = data[skip:].decode(encoding="utf-8")
    if size > len(data) or len(text) > max_chars:
        tail = text[max(0, len(text) - max_chars) :]
        match = RAW_ENTRY_HEADER_RE.search(tail)
        text = tail[match.start() :] if match else tail
    return text.strip()


def read_evidence(scope: str) -> str:
    """Returns `scope`'s evidence corpus oldest first (`detail.md` tail followed by `raw.md`).

    Args:
        scope (str): Memory scope key.
    """
    parts = (
        read_detail_tail(scope=scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS),
        read_raw_entries(scope=scope),
    )
    return "\n\n".join(part for part in parts if part)


def count_raw_entries(scope: str) -> int:
    """Returns how many raw entries are waiting for consolidation in `scope`.

    Args:
        scope (str): Memory scope key.
    """
    return len(split_raw_entries(text=_read_text(path=_raw_path(scope=scope))))


def raw_file_bytes(scope: str) -> int:
    """Returns `scope`'s `raw.md` size in bytes, or `0` when missing.

    Args:
        scope (str): Memory scope key.
    """
    return len(_read_text(path=_raw_path(scope=scope)).encode("utf-8"))


def detail_file_bytes(scope: str) -> int:
    """Returns `scope`'s `detail.md` size in bytes via `stat`, or `0` when missing.

    Args:
        scope (str): Memory scope key.
    """
    path = _detail_path(scope=scope)
    return path.stat().st_size if path.is_file() else 0


def read_raw_entries(scope: str) -> str:
    """Returns `scope`'s `raw.md` text for consolidation input.

    Args:
        scope (str): Memory scope key.
    """
    return _read_text(path=_raw_path(scope=scope)).strip()


def clear_raw(scope: str) -> None:
    """Deletes `scope`'s `raw.md` after consolidation consumed it.

    Args:
        scope (str): Memory scope key.
    """
    _raw_path(scope=scope).unlink(missing_ok=True)


def clear_tone(scope: str) -> None:
    """Deletes `scope`'s tone note after a full rebuild found no tone signal or a forget emptied it.

    Args:
        scope (str): Memory scope key.
    """
    _tone_path(scope=scope).unlink(missing_ok=True)


def delete_memory_files(scope: str) -> bool:
    """Deletes `scope`'s memory files without moving its clear boundary.

    Removes `raw.md`, `detail.md`, `tone.md`, every `.md` and `.md.tmp` in the compartment
    tree, and any empty directories left behind.

    Args:
        scope (str): Memory scope key.

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


def _entries_bytes(entries: list[str]) -> int:
    """Returns the rendered raw-file size for a list of entry blocks."""
    return len("\n\n".join(entries).encode("utf-8"))
