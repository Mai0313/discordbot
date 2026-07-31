"""Offline migration of the memory store onto compartment directories (issue #408).

Runs against a stopped bot and leaves no compatibility code in the tree. There is no
migration-specific writer: every scope is rebuilt through the same pure-evidence path
`/memory regenerate` uses, and the routing comes from the `- source:` / `- sharing:`
fields code already stamped onto every stored observation.

Order matters and is the one thing to get right. `regenerate_main_memory` reads
`raw.md` **and** the `detail.md` tail, and retires the raw batch into detail itself. So
the rebuild runs FIRST and the old files are deleted only after it succeeds; deleting
`raw.md` up front, as the issue body originally described, would zero every scope whose
only evidence is an unconsumed raw batch.

Two expected losses, reported rather than hidden:

* a scope with no evidence at all rebuilds empty — its old `main.md` was distilled from
  evidence that has since been trimmed away, and there is nothing to rebuild it from;
* a scope whose every observation is `source_only` gets an empty `global/`, so it will
  read as having no memory in a server it has never spoken in, until its next
  cross-server-safe observation.

Run from the repo root:

    uv run python -m scripts.migrate_memories                  # dry run
    uv run python -m scripts.migrate_memories --apply
"""

import asyncio
import argparse

from openai import AsyncOpenAI
from rich.table import Table
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.cogs._memory.facts import render_owner_identity
from discordbot.cogs._memory.store import (
    read_facts,
    read_owner,
    iter_scopes,
    memory_root,
    read_detail_tail,
    read_raw_entries,
    list_compartments,
)
from discordbot.cogs._memory.deltas import partition_raw_entries
from discordbot.cogs._memory.pipeline import flavor_of, regenerate_main_memory
from discordbot.cogs._memory.constants import (
    MEMORY_GLOBAL_CONCURRENCY,
    MEMORY_DETAIL_CONTEXT_MAX_CHARS,
)
from discordbot.cogs._memory.extraction import MemoryExtractorAI

console = Console()

# The legacy files the rebuild replaces. Deleted only after a scope rebuilds, and
# `detail.md` is never in this list — it is the evidence everything is rebuilt from.
_LEGACY_FILES = ("main.md", "main.bak.md", "tone.md")


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


def _report(rows: list[tuple[str, str, dict[str, int]]]) -> None:
    """Prints one row per scope, flagging the two expected kinds of loss."""
    table = Table(title="memory migration")
    table.add_column("scope")
    table.add_column("result")
    table.add_column("compartments")
    table.add_column("note", style="yellow")
    for scope, result, buckets in rows:
        if result == "no_evidence":
            note = "REBUILDS EMPTY: no evidence left to rebuild from"
        elif not buckets.get("global"):
            note = "EMPTY GLOBAL: unknown in a server this user has not spoken in"
        else:
            note = ""
        summary = ", ".join(f"{name}={count}" for name, count in sorted(buckets.items()))
        table.add_row(scope, result, summary or "-", note)
    console.print(table)


async def _migrate_one(
    extractor: MemoryExtractorAI, scope: str, semaphore: asyncio.Semaphore, apply: bool
) -> tuple[str, str, dict[str, int]]:
    """Rebuilds one scope and removes its legacy files, returning a report row."""
    buckets = _preview(scope=scope)
    if not apply:
        return scope, "dry-run", buckets
    identity = render_owner_identity(owner=read_owner(scope=scope))
    async with semaphore:
        # The script calls the rebuild directly rather than through the reply pipeline,
        # so it needs its own bound: `_memory_semaphore` is entered inside
        # `regenerate_main_memory`, but nothing else here throttles the fan-out.
        try:
            result = await regenerate_main_memory(
                scope=scope, extractor=extractor, identity=identity
            )
        except Exception as error:
            # Broad on purpose: one scope failing must not abandon the other 129.
            return scope, f"error: {type(error).__name__}: {error}", buckets
    if result == "regenerated":
        for name in _LEGACY_FILES:
            (memory_root() / scope / name).unlink(missing_ok=True)
    return scope, result, _preview_written(scope=scope)


def _preview_written(scope: str) -> dict[str, int]:
    """Returns the fact counts actually written per compartment."""
    return {
        compartment: len(read_facts(scope=scope, compartment=compartment))
        for compartment in list_compartments(scope=scope)
    }


async def _migrate_all(model: ModelSettings, apply: bool) -> None:
    """Migrates every scope on disk, bounded by its own semaphore."""
    scopes = iter_scopes()
    console.print(f"{len(scopes)} scope(s) found; apply={apply}")
    config = LLMConfig()
    extractor = MemoryExtractorAI(
        client=AsyncOpenAI(base_url=config.base_url, api_key=config.api_key),
        extract_model=model,
        evaluate_model=model,
        consolidate_model=model,
    )
    semaphore = asyncio.Semaphore(MEMORY_GLOBAL_CONCURRENCY)
    rows = await asyncio.gather(
        *(
            _migrate_one(extractor=extractor, scope=scope, semaphore=semaphore, apply=apply)
            for scope in scopes
        )
    )
    _report(rows=list(rows))


def main() -> None:
    """Parses arguments and runs the migration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Rewrite the store; omit for a dry run."
    )
    arguments = parser.parse_args()
    console.print(
        "[yellow]Stop the bot and commit data/memories (tag it `pre-408`) before "
        "running with --apply.[/yellow]"
    )
    asyncio.run(
        _migrate_all(model=RuntimeModelCatalog().memory_consolidator_model, apply=arguments.apply)
    )


if __name__ == "__main__":
    main()
