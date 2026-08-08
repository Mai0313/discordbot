"""Offline rebuild of file-backed memory from cold-tier evidence.

Rebuilds one scope, or a batch of them, through `regenerate_main_memory` — the same
pure-evidence path `/memory regenerate` runs, which distills the detail tail window plus
any unconsumed raw entries and never reads the existing facts. That function is already
scope-agnostic (it resolves the flavor itself), so this script only decides which scopes
it is pointed at.

The target is a scope key or one of three collective words::

    all                       every scope on disk (default)
    users                     every user scope
    servers                   every server scope
    <user_id>                 one user
    bot_memories/<server_id>  one server

Stop the bot before `--apply`, whatever the target. This runs in a second process and
`scope_lock` is an in-process `asyncio.Lock`, so nothing serializes the two: the rebuild
ends by unlinking `raw.md`, which drops any observation the bot appended while it ran, and
the bot keeps serving its cached pre-rebuild document until its own next write to that
scope. `/memory regenerate` is the live-safe way to rebuild one scope, precisely because it
runs inside the bot under that lock. What the target grades is blast radius, not safety, so
a collective one also asks for the store to be committed first.

Two expected losses, reported rather than hidden:

* a scope with no evidence at all rebuilds empty — its facts were distilled from evidence
  that has since been trimmed away, and there is nothing to rebuild them from;
* a scope whose every observation is `source_only` gets an empty `global/`, so it will
  read as having no memory in a server it has never spoken in, until its next
  cross-server-safe observation.

The report also names any file in a compartment directory the store did not write. A
rebuild removes every fact file it did not re-emit, an unreadable one included, but
never a foreign file — so those are what a rebuilt scope still holds unaccounted for.
It counts the unreadable ones it removed as well: renaming a section or a durability
value makes every fact carrying the old one unparsable, and the next rebuild then drops
all of them in one pass, which nothing else here would say out loud.

Run from the repo root::

    uv run python -m scripts.regen_memories                            # dry run, all
    uv run python -m scripts.regen_memories 1234567890 --apply
    uv run python -m scripts.regen_memories bot_memories/9876543210 --apply
    uv run python -m scripts.regen_memories users --apply
"""

from typing import TYPE_CHECKING, cast
import asyncio
import argparse
from collections.abc import Sequence

from openai import AsyncOpenAI
from rich.table import Table
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.services.memory.facts import render_owner_identity
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    read_facts,
    read_owner,
    iter_scopes,
    read_detail_tail,
    read_raw_entries,
    list_compartments,
    unaccounted_files,
)
from discordbot.services.memory.deltas import partition_raw_entries
from discordbot.services.memory.pipeline import flavor_of, regenerate_main_memory
from discordbot.services.memory.constants import MEMORY_DETAIL_CONTEXT_MAX_CHARS
from discordbot.services.memory.extraction import MemoryExtractorAI

if TYPE_CHECKING:
    from openai.types.shared.reasoning_effort import ReasoningEffort

console = Console()

# The offline fan-out's own bound, deliberately not `MEMORY_GLOBAL_CONCURRENCY`: that one
# is sized for background work sharing the proxy with the latency-critical reply path, on
# the assumption that path exists. A batch run here has no reply latency to protect and no
# reason to push the proxy as hard. No flag on purpose — edit this if a store needs more.
_CONCURRENCY = 8

# Targets naming more than one scope, which is what makes a run store-scale.
_BATCH_TARGETS = ("all", "users", "servers")


def _scopes_for_target(target: str) -> list[str]:
    """Returns the scopes a target names, in store order.

    Args:
        target: One of `all` / `users` / `servers`, or a single scope key.

    Raises:
        SystemExit: The target names a single scope with nothing on disk to rebuild,
            which is what a mistyped id looks like. Reported here rather than left to
            come back as a `no_evidence` row, which reads like data loss.
    """
    scopes = iter_scopes()
    if target == "all":
        return scopes
    if target in _BATCH_TARGETS:
        wanted = "server" if target == "servers" else "user"
        return [scope for scope in scopes if flavor_of(scope=scope) == wanted]
    if target not in scopes:
        raise SystemExit(f"no memory on disk for scope {target!r}")
    return [target]


def _preview(scope: str) -> dict[str, int]:
    """Returns the per-compartment observation counts a rebuild of this scope would see."""
    evidence = "\n\n".join(
        part
        for part in (
            read_detail_tail(scope=scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS),
            read_raw_entries(scope=scope),
        )
        if part
    )
    buckets = partition_raw_entries(raw_text=evidence, flavor=flavor_of(scope=scope))
    return {compartment: text.count("### ") for compartment, text in buckets.items()}


def _written(scope: str) -> dict[str, int]:
    """Returns the fact counts actually written per compartment."""
    return {
        compartment: len(read_facts(scope=scope, compartment=compartment))
        for compartment in list_compartments(scope=scope)
    }


def _loss_note(result: str, buckets: dict[str, int]) -> str:
    """Returns the warning for a scope that rebuilds empty or loses its global compartment.

    The empty-bucket test is what makes the first note reachable BEFORE `--apply`: a dry
    run's result is always the literal `dry-run`, so keying on `no_evidence` alone flagged
    a scope with nothing to rebuild from only once the destructive run had happened, and
    until then mislabelled it `EMPTY GLOBAL` (one live scope, measured). It also retires
    that second note for server scopes, whose evidence all routes to `global` by
    construction, so its user-flavored wording can no longer land on one.
    """
    if result == "no_evidence" or not any(buckets.values()):
        return "REBUILDS EMPTY: no evidence left to rebuild from"
    if not buckets.get(GLOBAL_COMPARTMENT):
        return "EMPTY GLOBAL: unknown in a server this user has not spoken in"
    return ""


def _unaccounted_note(scope: str) -> str:
    """Returns the note for files a rebuild of this scope leaves behind untouched.

    A rebuild removes every fact file it did not re-emit, but never a file the store did
    not write, so those survive it and nothing else will ever remove them either. The
    rebuild logs them through logfire, which is unconfigured in a script run and reaches
    nobody here; this is the operator's copy, and it lands on the dry run too, while
    there is still time to act on it.
    """
    stray = [
        f"{compartment}/{name}"
        for compartment in list_compartments(scope=scope)
        for name in unaccounted_files(scope=scope, compartment=compartment)
    ]
    return f"UNACCOUNTED: {', '.join(stray)}" if stray else ""


def _unreadable_note(removed: int) -> str:
    """Returns the note for fact files a rebuild of this scope destroyed unread.

    The one loss nothing else here can report after the fact: the files are gone, and
    `prune_compartment` is the only thing that saw them, so the count travels back on the
    rebuild's own report. Names would not help — they are `mint_fact_id` digests of a
    file that no longer exists — but the count says a rebuild took content with it, which
    an operator running without `MEMORY_GIT_ENABLED` had no way to learn at all.
    """
    return f"UNREADABLE: {removed} fact file(s) removed unread" if removed else ""


def _report(rows: list[tuple[str, str, dict[str, int], int]]) -> None:
    """Prints one row per scope, flagging what a rebuild of it loses, leaves and destroys."""
    table = Table(title="memory regeneration")
    table.add_column("scope")
    table.add_column("result")
    table.add_column("compartments")
    table.add_column("note", style="yellow")
    for scope, result, buckets, removed in rows:
        summary = ", ".join(f"{name}={count}" for name, count in sorted(buckets.items()))
        notes = (
            _loss_note(result=result, buckets=buckets),
            _unaccounted_note(scope=scope),
            _unreadable_note(removed=removed),
        )
        table.add_row(scope, result, summary or "-", "; ".join(note for note in notes if note))
    console.print(table)


async def _regen_one(
    extractor: MemoryExtractorAI, scope: str, semaphore: asyncio.Semaphore
) -> tuple[str, str, dict[str, int], int]:
    """Rebuilds one scope, prints its outcome, and returns its report row."""
    removed = 0
    async with semaphore:
        # The script calls the rebuild directly rather than through the reply pipeline,
        # so it needs its own bound: `_memory_semaphore` is entered inside
        # `regenerate_main_memory`, but nothing else here throttles the fan-out.
        try:
            # Inside the handler because it is not safe either: `read_owner` parses the
            # id out of the scope key, so one non-numeric directory under the store (a
            # backup copy, which this tool's own advice invites) used to raise past the
            # gather and throw away every row that had already rebuilt.
            identity = render_owner_identity(owner=read_owner(scope=scope))
            report = await regenerate_main_memory(
                scope=scope, extractor=extractor, identity=identity
            )
            result, removed = report.result, report.unreadable_removed
            counts = _written(scope=scope)
        except Exception as error:
            # Broad on purpose: one scope failing must not abandon the rest of the batch.
            result, counts = f"error: {type(error).__name__}: {error}", _preview(scope=scope)
    # A 145-scope run is several minutes of LLM work, so each scope reports as it lands
    # rather than leaving the closing table as the only output.
    console.print(f"{scope}: {result}")
    return scope, result, counts, removed


async def _regen_all(model: ModelSettings, target: str, apply: bool) -> None:
    """Rebuilds every scope the target names, bounded by this script's own semaphore."""
    scopes = _scopes_for_target(target=target)
    console.print(f"{len(scopes)} scope(s) found; apply={apply}")
    # Printed on the dry run too, which is when there is still time to act on it, and on
    # every target: an out-of-process write races the bot's own `scope_lock` whether it
    # touches one scope or all of them (`/memory regenerate` is the live-safe single-scope
    # path). Only the blast radius is graded.
    console.print(
        "[yellow]Stop the bot before --apply: this writes from a second process, so a "
        "rebuilt scope loses whatever raw entries the bot appended meanwhile.[/yellow]"
    )
    if target in _BATCH_TARGETS:
        console.print(
            "[yellow]This one covers the whole store; commit data/memories first.[/yellow]"
        )
    if not apply:
        _report(rows=[(scope, "dry-run", _preview(scope=scope), 0) for scope in scopes])
        return
    config = LLMConfig()
    extractor = MemoryExtractorAI(
        client=AsyncOpenAI(base_url=config.base_url, api_key=config.api_key),
        extract_model=model,
        evaluate_model=model,
        consolidate_model=model,
    )
    console.print(f"Rebuilding with [bold]{model.name}[/bold] (effort: {model.effort})")
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    rows = await asyncio.gather(
        *(_regen_one(extractor=extractor, scope=scope, semaphore=semaphore) for scope in scopes)
    )
    _report(rows=list(rows))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the offline rebuild CLI arguments."""
    # `--help` carries the whole module docstring, so the target table and the
    # stop-the-bot framing reach an operator who never opens the file.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="all (default), users, servers, a user id, or bot_memories/<server_id>.",
    )
    writer = RuntimeModelCatalog().memory_writer_model
    parser.add_argument("--model", default=writer.name)
    parser.add_argument("--effort", default=writer.effort)
    parser.add_argument(
        "--apply", action="store_true", help="Rewrite the store; omit for a dry run."
    )
    return parser.parse_args(argv)


def main() -> None:
    """Parses arguments and runs the rebuild."""
    args = _parse_args()
    model = ModelSettings(name=args.model, effort=cast("ReasoningEffort", args.effort))
    asyncio.run(main=_regen_all(model=model, target=args.target, apply=args.apply))


if __name__ == "__main__":
    main()
