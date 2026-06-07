"""Offline regeneration of per-user consolidated memory from cold-tier evidence.

Rebuilds `main.md` for one user or every user with a caller-specified model,
using only the detail tail window plus unconsumed raw entries (never the
existing main file). Run from the repo root: the `folder` argument only picks
single-user versus all-users and enumerates user ids; reads and writes always
go through the memory store rooted at `./data/memories`.
"""

import asyncio
from pathlib import Path

from rich.console import Console

from discordbot.utils.llm import create_litellm_client
from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs._memory.store import read_main_identity
from discordbot.cogs._memory.pipeline import regenerate_main_memory
from discordbot.cogs._memory.extraction import MemoryExtractorAI

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


async def _regen_all(model: ModelSettings, folder: str) -> None:
    """Regenerates the main memory file for every resolved user sequentially.

    Args:
        model: Model settings (LiteLLM model string plus reasoning effort)
            used for the consolidation rewrite.
        folder: Single-user memory directory or the memories root.
    """
    extractor = MemoryExtractorAI(
        client=create_litellm_client(config=LLMConfig()),
        extract_model=model,
        consolidate_model=model,
    )
    user_ids = _resolve_user_ids(folder=Path(folder))
    console.print(
        f"Regenerating {len(user_ids)} user(s) with [bold]{model.name}[/bold] "
        f"(effort: {model.effort})"
    )
    for user_id in user_ids:
        identity = read_main_identity(user_id=user_id) or f"[id: {user_id}]"
        try:
            result = await regenerate_main_memory(
                user_id=user_id, extractor=extractor, identity=identity
            )
        except Exception as error:
            console.print(f"[red]{user_id}: error ({error})[/red]")
            continue
        styles = {"regenerated": "green", "no_evidence": "yellow", "failed": "red"}
        console.print(f"[{styles[result]}]{user_id}: {result}[/{styles[result]}]")


def regen_memories(model: ModelSettings, folder: str) -> None:
    """Regenerates `main.md` from evidence for one user or every user.

    Args:
        model: Model settings (LiteLLM model string plus reasoning effort)
            used for the consolidation rewrite.
        folder: Single-user memory directory (e.g. `./data/memories/<id>`) or
            the memories root (`./data/memories`) for every user.
    """
    asyncio.run(main=_regen_all(model=model, folder=folder))


if __name__ == "__main__":
    model_settings = ModelSettings(name="azure/gpt-5.5", effort="xhigh")
    regen_memories(model=model_settings, folder="./data/memories")
