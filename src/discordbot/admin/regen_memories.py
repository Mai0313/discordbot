"""Offline rebuild of file-backed memory from cold-tier evidence.

Rebuilds one scope, or a batch of them, from the detail tail window plus any unconsumed
raw entries without reading existing facts (the same pure-evidence rebuild as
`/memory regenerate`).

The target is a scope key or one of three collective words::

    all                       every scope on disk (default)
    users                     every user scope
    servers                   every server scope
    <user_id>                 one user
    bot_memories/<server_id>  one server

Stop the bot before a real run, whatever the target. This runs in a second process with
no cross-process locking: the rebuild ends by unlinking `raw.md`, which drops any
observation the bot appended while it ran, and the bot keeps serving its cached
pre-rebuild document until its own next write to that scope. Use `/memory regenerate`
for a live-safe single-scope rebuild. A collective target also requires committing
`data/memories` first. Rebuilding is the default, so nothing is written until you confirm
with `y`.

Two expected losses, reported rather than hidden:

* a scope whose evidence yields no fact rebuilds empty, since the rebuild never reads the
  existing facts back in;
* a scope whose every observation is `source_only` gets an empty `global/`, so it will
  read as having no memory in a server it has never spoken in, until its next
  cross-server-safe observation.

A scope with no evidence at all is skipped with its facts and tone note untouched.

The report also lists foreign files in compartment directories (which a rebuild never
removes) and counts unreadable fact files removed during the rebuild.

Run from the repo root::

    uv run regen-memories                  # rebuild all
    uv run regen-memories --dry-run        # preview all
    uv run regen-memories 1234567890
    uv run regen-memories bot_memories/9876543210
    uv run regen-memories users
"""

from typing import TYPE_CHECKING, cast
import asyncio
import argparse
from collections.abc import Sequence

from openai import AsyncOpenAI
from pydantic import Field, BaseModel, ConfigDict
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.services.memory.facts import render_owner_identity
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    flavor_of,
    read_facts,
    read_owner,
    iter_scopes,
    read_evidence,
    list_compartments,
    unaccounted_files,
)
from discordbot.services.memory.deltas import partition_raw_entries
from discordbot.services.memory.writer import MemoryWriterAI
from discordbot.services.memory.regeneration import regenerate_scope_memory

if TYPE_CHECKING:
    from openai.types.shared.reasoning_effort import ReasoningEffort

console = Console()

# Offline batch concurrency; `regenerate_scope_memory` also acquires `memory_semaphore` (24).
_CONCURRENCY = 20

_BATCH_TARGETS = ("all", "users", "servers")


class _ScopeRow(BaseModel):
    """One scope's line in the closing report."""

    model_config = ConfigDict(frozen=True)

    scope: str = Field(..., description="The scope key this row is about.")
    result: str = Field(
        ...,
        description="How the rebuild ended, or the exception text when it raised.",
        examples=["regenerated"],
    )
    counts: dict[str, int] = Field(
        ...,
        description=(
            "Facts written per compartment; the previewed observation counts instead on a "
            "dry run and on a scope whose rebuild raised."
        ),
    )
    unreadable_removed: int = Field(
        ..., description="Fact files this run removed that no reader could parse.", examples=[3]
    )


def _scopes_for_target(target: str) -> list[str]:
    """Returns matching scopes in store order; exits if a single target is absent on disk."""
    scopes = iter_scopes()
    if target == "all":
        return scopes
    if target in _BATCH_TARGETS:
        wanted = "server" if target == "servers" else "user"
        return [scope for scope in scopes if flavor_of(scope=scope) == wanted]
    if target not in scopes:
        raise SystemExit(f"no memory on disk for scope {target!r}")
    return [target]


def _preview_row(scope: str) -> _ScopeRow:
    """Returns the dry-run observation counts for `scope`, capturing read errors in the row."""
    try:
        evidence = read_evidence(scope=scope)
    except (OSError, UnicodeDecodeError) as error:
        return _ScopeRow(
            scope=scope,
            result=f"error: {type(error).__name__}: {error}",
            counts={},
            unreadable_removed=0,
        )
    buckets = partition_raw_entries(raw_text=evidence, flavor=flavor_of(scope=scope))
    return _ScopeRow(
        scope=scope,
        # Matches `regenerate_scope_memory`'s empty-evidence skip condition.
        result="dry-run" if evidence else "no_evidence",
        counts={compartment: text.count("### ") for compartment, text in buckets.items()},
        unreadable_removed=0,
    )


def _written(scope: str) -> dict[str, int]:
    """Returns the fact counts actually written per compartment."""
    return {
        compartment: len(read_facts(scope=scope, compartment=compartment))
        for compartment in list_compartments(scope=scope)
    }


def _loss_note(result: str, buckets: dict[str, int]) -> str:
    """Returns a warning note when a scope is skipped, rebuilds empty, or lacks `global`."""
    if result == "no_evidence":
        return "SKIPPED: no evidence to rebuild from, so its memory is kept as it is"
    if not any(buckets.values()):
        if result in ("regenerated", "dry-run"):
            return "REBUILDS EMPTY: its evidence yields no fact"
        return ""
    if not buckets.get(GLOBAL_COMPARTMENT):
        return "EMPTY GLOBAL: unknown in a server this user has not spoken in"
    return ""


def _unaccounted_note(scope: str) -> str:
    """Returns a warning note listing foreign files left in `scope`'s compartments."""
    stray = [
        f"{compartment}/{name}"
        for compartment in list_compartments(scope=scope)
        for name in unaccounted_files(scope=scope, compartment=compartment)
    ]
    return f"UNACCOUNTED: {', '.join(stray)}" if stray else ""


def _unreadable_note(removed: int) -> str:
    """Returns a warning note when unreadable fact files were removed during rebuild."""
    return f"UNREADABLE: {removed} fact file(s) removed unread" if removed else ""


def _report(rows: list[_ScopeRow]) -> None:
    """Prints the per-scope regeneration summary and warning notes."""
    # Plain lines instead of `rich.Table` so long warning notes wrap rather than ellipsize.
    console.print("[bold]memory regeneration[/bold]")
    # Pad only scope width; failed results contain long exception strings.
    width = max((len(row.scope) for row in rows), default=0)
    for row in rows:
        summary = ", ".join(f"{name}={count}" for name, count in sorted(row.counts.items()))
        console.print(f"{row.scope:<{width}}  {row.result}  {summary or '-'}")
        notes = (
            _loss_note(result=row.result, buckets=row.counts),
            _unaccounted_note(scope=row.scope),
            _unreadable_note(removed=row.unreadable_removed),
        )
        for note in notes:
            if note:
                console.print(f"    [yellow]{note}[/yellow]")


async def _regen_one(
    writer: MemoryWriterAI, scope: str, semaphore: asyncio.Semaphore
) -> _ScopeRow:
    """Rebuilds one scope, prints its outcome, and returns its report row."""
    removed = 0
    async with semaphore:
        try:
            # Keep `read_owner` inside try so a store read error does not abort the batch.
            identity = render_owner_identity(owner=read_owner(scope=scope))
            report = await regenerate_scope_memory(scope=scope, writer=writer, identity=identity)
            result, removed = report.result, report.unreadable_removed
            counts = _written(scope=scope)
        except Exception as error:
            # Broad on purpose: one scope failing must not abandon the rest of the batch.
            result = f"error: {type(error).__name__}: {error}"
            counts = _preview_row(scope=scope).counts
    console.print(f"{scope}: {result}")
    return _ScopeRow(scope=scope, result=result, counts=counts, unreadable_removed=removed)


async def _rebuild_batch(writer: MemoryWriterAI, scopes: list[str]) -> list[_ScopeRow]:
    """Rebuilds `scopes` concurrently with a progress bar, returning rows in input order."""
    semaphore = asyncio.Semaphore(_CONCURRENCY)
    rows: dict[str, _ScopeRow] = {}
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]rebuilding", total=len(scopes))
        pending = [_regen_one(writer=writer, scope=scope, semaphore=semaphore) for scope in scopes]
        for landing in asyncio.as_completed(pending):
            row = await landing
            rows[row.scope] = row
            progress.advance(task)
    # Restore input store order after out-of-order async completion.
    return [rows[scope] for scope in scopes]


def _confirmed() -> bool:
    """Prompts for a typed `y` confirmation, returning False on EOF or any other input."""
    try:
        answer = console.input("[bold yellow]Type y to rebuild: [/bold yellow]")
    except EOFError:
        console.print("[red]Nothing on stdin; nothing was written.[/red]")
        return False
    if answer.strip().lower() == "y":
        return True
    console.print("[red]Not confirmed; nothing was written.[/red]")
    return False


async def _regen_all(model: ModelSettings, target: str, dry_run: bool) -> None:
    """Previews or rebuilds every scope the target names, warning about the race first."""
    scopes = _scopes_for_target(target=target)
    console.print(f"{len(scopes)} scope(s) found; dry_run={dry_run}")
    console.print(
        "[yellow]Stop the bot before rebuilding: this writes from a second process, so a "
        "rebuilt scope loses whatever raw entries the bot appended meanwhile.[/yellow]"
    )
    if target in _BATCH_TARGETS:
        console.print(
            "[yellow]This one covers the whole store; commit data/memories first.[/yellow]"
        )
    if dry_run:
        _report(rows=[_preview_row(scope=scope) for scope in scopes])
        return
    if not _confirmed():
        return
    config = LLMConfig()
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    writer = MemoryWriterAI(client=client, model=model)
    console.print(f"Rebuilding with [bold]{model.name}[/bold] (effort: {model.effort})")
    _report(rows=await _rebuild_batch(writer=writer, scopes=scopes))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the offline rebuild CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="all (default), users, servers, a user id, or bot_memories/<server_id>.",
    )
    writer = RuntimeModelCatalog().memory_model
    parser.add_argument("--model", default=writer.name)
    parser.add_argument("--effort", default=writer.effort)
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview only; omit to rewrite the store."
    )
    return parser.parse_args(argv)


def main() -> None:
    """Parses arguments and runs the rebuild."""
    args = _parse_args()
    model = ModelSettings(name=args.model, effort=cast("ReasoningEffort", args.effort))
    asyncio.run(main=_regen_all(model=model, target=args.target, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
