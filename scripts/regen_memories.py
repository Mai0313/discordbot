"""Offline regeneration of per-user consolidated memory from cold-tier evidence.

Rebuilds `main.md` for one user or every user with a caller-specified model,
using only the detail tail window plus unconsumed raw entries (never the
existing main file). Run from the repo root: the `folder` argument only picks
single-user versus all-users and enumerates user ids; reads and writes always
go through the memory store rooted at `./data/memories`.
"""

from typing import TYPE_CHECKING, cast
import asyncio
from pathlib import Path
import argparse

from rich.console import Console

from discordbot.utils.llm import create_litellm_client
from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs._memory.store import (
    user_scope,
    read_detail_tail,
    read_main_memory,
    read_raw_entries,
    count_raw_entries,
    read_main_identity,
)
from discordbot.cogs._memory.pipeline import regenerate_main_memory
from discordbot.cogs._memory.constants import MEMORY_DETAIL_CONTEXT_MAX_CHARS
from discordbot.cogs._memory.extraction import MemoryExtractorAI, observation_keys_from_text

if TYPE_CHECKING:
    from openai.types.shared.reasoning_effort import ReasoningEffort

console = Console()


def _resolve_user_ids(folder: Path) -> list[int]:
    """Returns the user ids for a single-user directory or a memories root.

    Args:
        folder: Either one user's memory directory (numeric basename) or the
            memories root containing one numeric subdirectory per user.
    """
    if folder.name.isdigit():
        return [int(folder.name)]
    return sorted(
        int(path.name) for path in folder.iterdir() if path.is_dir() and path.name.isdigit()
    )


def _preview_one(user_id: int) -> None:
    """Prints privacy-preserving memory evidence statistics for one user."""
    scope = user_scope(user_id=user_id)
    main_text = read_main_memory(scope=scope)
    raw_text = read_raw_entries(scope=scope)
    detail_text = read_detail_tail(scope=scope, max_chars=MEMORY_DETAIL_CONTEXT_MAX_CHARS)
    raw_keys = observation_keys_from_text(text=raw_text)
    detail_keys = observation_keys_from_text(text=detail_text)
    duplicate_keys = raw_keys & detail_keys
    console.print(
        f"[cyan]{user_id}: dry-run[/cyan] "
        f"main_chars={len(main_text)} "
        f"raw_entries={count_raw_entries(scope=scope)} "
        f"raw_keys={len(raw_keys)} detail_keys={len(detail_keys)} "
        f"duplicate_keys={len(duplicate_keys)}"
    )


async def _regen_one(extractor: MemoryExtractorAI, user_id: int) -> None:
    """Regenerates one user's main memory file and prints the outcome.

    Args:
        extractor: Memory extractor whose consolidate model performs the rewrite.
        user_id: Discord user id whose memory directory is rebuilt.
    """
    scope = user_scope(user_id=user_id)
    identity = read_main_identity(scope=scope) or f"[id: {user_id}]"
    try:
        result = await regenerate_main_memory(scope=scope, extractor=extractor, identity=identity)
    except Exception as error:
        console.print(f"[red]{user_id}: error ({error})[/red]")
        return
    styles = {
        "regenerated": "green",
        "no_evidence": "yellow",
        "failed": "red",
        "cooldown": "yellow",
    }
    console.print(f"[{styles[result]}]{user_id}: {result}[/{styles[result]}]")


async def _regen_all(model: ModelSettings, folder: str, apply: bool) -> None:
    """Regenerates the main memory file for every resolved user concurrently.

    Concurrency is bounded by the pipeline's global memory semaphore
    (`MEMORY_GLOBAL_CONCURRENCY`), not by this script.

    Args:
        model: Model settings (LiteLLM model string plus reasoning effort)
            used for the consolidation rewrite.
        folder: Single-user memory directory or the memories root.
        apply: Whether to rewrite memory files; false prints a dry-run preview.
    """
    user_ids = _resolve_user_ids(folder=Path(folder))
    if not apply:
        console.print(
            f"Dry-run preview for {len(user_ids)} user(s). Pass --apply to rewrite main.md."
        )
        for user_id in user_ids:
            _preview_one(user_id=user_id)
        return
    extractor = MemoryExtractorAI(
        client=create_litellm_client(config=LLMConfig()),
        extract_model=model,
        evaluate_model=model,
        consolidate_model=model,
    )
    console.print(
        f"Regenerating {len(user_ids)} user(s) with [bold]{model.name}[/bold] "
        f"(effort: {model.effort})"
    )
    tasks = []
    for user_id in user_ids:
        tasks.append(_regen_one(extractor=extractor, user_id=user_id))
    await asyncio.gather(*tasks)


def regen_memories(model: ModelSettings, folder: str, apply: bool = False) -> None:
    """Regenerates `main.md` from evidence for one user or every user.

    Args:
        model: Model settings (LiteLLM model string plus reasoning effort)
            used for the consolidation rewrite.
        folder: Single-user memory directory (e.g. `./data/memories/<id>`) or
            the memories root (`./data/memories`) for every user.
        apply: Whether to rewrite memory files; false prints a dry-run preview.
    """
    asyncio.run(main=_regen_all(model=model, folder=folder, apply=apply))


def _parse_args() -> argparse.Namespace:
    """Parses the offline regeneration CLI arguments."""
    parser = argparse.ArgumentParser(description="Preview or regenerate file-backed memories.")
    parser.add_argument("--folder", default="./data/memories")
    parser.add_argument("--model", default="azure/gpt-5.5")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_settings = ModelSettings(name=args.model, effort=cast("ReasoningEffort", args.effort))
    regen_memories(model=model_settings, folder=args.folder, apply=args.apply)
