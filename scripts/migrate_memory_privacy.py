"""One-shot offline migration of per-user memory to the source-tagged privacy format.

Rewrites every user's `main.md` so each bullet outside `## 使用者輪廓` carries a
`[src:...]` tag (`[src:*]` for anywhere-safe general facts, `[src:legacy]` for
everything else — the pre-migration file never recorded where a fact was
learned, so a real source id is unrecoverable), rewrites the profile paragraph
to anywhere-safe content only, and seeds `tone.md` from any tone/delivery
bullets it extracts. Run once at the privacy-redesign deploy, after deleting
the untagged user-scope `raw.md` / `detail.md` / `main.bak.md` evidence; the
rewrite itself stores the pre-migration file as the fresh `main.bak.md` escape
hatch. Default is a free deterministic dry-run preview; pass `--apply` for the
LLM rewrite.
"""

import re
from typing import TYPE_CHECKING, cast
import asyncio
from pathlib import Path
import argparse

from openai import AsyncOpenAI
from rich.console import Console

from discordbot.utils.llm import parse_responses_or_none
from discordbot.typings.llm import LLMConfig

# Module-namespace import on purpose: `_repoint_store_root` reassigns the store's
# root so `--folder` governs reads and writes, not just id enumeration.
from discordbot.cogs._memory import store
from discordbot.typings.models import ModelSettings
from discordbot.cogs._memory.store import (
    user_scope,
    write_tone,
    read_main_memory,
    write_main_memory,
    read_main_identity,
)
from discordbot.cogs._memory.extraction import ConsolidatedMemory

if TYPE_CHECKING:
    from openai.types.shared.reasoning_effort import ReasoningEffort

console = Console()

# A liveness backstop only, like the pipeline's consolidation timeout: the
# rewrite of a tens-of-KB file at high effort can legitimately take minutes.
MIGRATION_TIMEOUT_SECONDS = 600.0

# Local bound: unlike regen_memories.py this path does not ride the pipeline's
# global memory semaphore, so the fan-out over every user is capped here.
MIGRATION_CONCURRENCY = 8

# A well-formed trailing source tag; mirrors the read-side filter's grammar.
_SRC_TAG_RE = re.compile(r"\[src:[0-9a-z*,]+\]\s*$")

_PROFILE_HEADER = "## 使用者輪廓"

MIGRATION_PROMPT = """
You are migrating ONE user's consolidated memory file to a source-tagged privacy format.

INPUT (in the user message):
* `<memory_file>`: the current consolidated file, starting with `v1`. Sections: `## 使用者輪廓` (one short paragraph), then bullet sections (`## 永久事實`, `## 穩定偏好`, `## 穩定事實`, `## 互動筆記`, `## 近期脈絡`), any of which may be absent.

YOUR ONLY JOB — rewrite the file with exactly three changes and NOTHING else:
1. TAG every bullet in every section except `## 使用者輪廓` with exactly one trailing `[src:...]` tag as the bullet's last token:
   * `[src:*]` ONLY for clearly harmless, anywhere-safe general facts: reply language / format preferences, how the user wants to be addressed, broad interests and hobbies, tech background, which bot features they use.
   * `[src:legacy]` for EVERYTHING else: secrets, feelings, health, relationships, money, work or project specifics, plans and trips, ongoing situations, opinions about people, and anything involving another person. This file predates source tracking, so the real source of a fact is unrecoverable — NEVER invent a guild id. When in doubt, `[src:legacy]`.
2. TONE EXTRACTION: move every tone/delivery preference (how the bot should SOUND: banter / sarcasm / profanity tolerance, formality, warmth, terse vs verbose, emoji use) OUT of `memory_markdown` and into `tone_markdown`: a short note starting exactly with `## 語氣偏好`, holding persona-independent bullets with no dates and no `[src:...]` tags. Return an empty `tone_markdown` when the file carries no tone signal.
3. PROFILE SAFETY: `## 使用者輪廓` stays untagged and is shown in every conversation, so rewrite the paragraph to contain ONLY anywhere-safe content (language, broad interests, tech background, how to address them); move any private content it currently carries into a `[src:legacy]` bullet in the fitting section instead of deleting it.

HARD RULES:
* Do NOT add, drop, merge, reword, reorder, or re-date anything else. Keep every leading date tag (`[~YYYY-MM]`, `[YYYY-MM-DD]`) and every [REDACTED_SECRET] marker exactly as-is.
* Keep the section order and headers; drop a section header only when tone extraction emptied it. The output must start exactly with `v1`.
* Set `changed=true`. The content language stays Traditional Chinese.
"""


def _repoint_store_root(folder: Path) -> None:
    """Points the memory store at the requested root so `--folder` governs reads AND writes.

    Without this, ids enumerated from a backup or staging copy would read — and on
    `--apply`, rewrite — the LIVE `./data/memories` files instead of the requested
    folder. A single-user directory repoints to its parent (the memories root).
    """
    root = folder.parent if folder.name.isdigit() else folder
    store._MEMORY_DIR = root  # noqa: SLF001 -- deliberate offline repoint; the same knob the test fixture uses


def _resolve_user_ids(folder: Path) -> list[int]:
    """Returns the user ids for a single-user directory or a memories root.

    Mirrors `regen_memories._resolve_user_ids`; the bot's own server-memory
    parent directory is numeric too but carries no top-level `main.md`, so the
    per-user no-main skip below drops it naturally.
    """
    if folder.name.isdigit():
        return [int(folder.name)]
    return sorted(
        int(path.name) for path in folder.iterdir() if path.is_dir() and path.name.isdigit()
    )


def count_untagged_bullets(text: str) -> tuple[int, int]:
    """Returns `(bullets, untagged)` over every bullet line outside 使用者輪廓."""
    bullets = 0
    untagged = 0
    in_profile = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_profile = stripped == _PROFILE_HEADER
            continue
        if in_profile or not stripped.startswith(("* ", "- ")):
            continue
        bullets += 1
        if not _SRC_TAG_RE.search(stripped):
            untagged += 1
    return bullets, untagged


def tag_untagged_bullets(text: str) -> tuple[str, int]:
    """Deterministic backstop: appends `[src:legacy]` to any bullet the LLM left untagged.

    The read-side filter fail-closes on untagged bullets anyway; this stamp only
    makes the stored file honest about it so `/memory show` and later rewrites
    see an explicit tag instead of an accidental omission.
    """
    lines: list[str] = []
    fixed = 0
    in_profile = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_profile = stripped == _PROFILE_HEADER
        elif (
            not in_profile
            and stripped.startswith(("* ", "- "))
            and not _SRC_TAG_RE.search(stripped)
        ):
            lines.append(f"{line.rstrip()} [src:legacy]")
            fixed += 1
            continue
        lines.append(line)
    return "\n".join(lines), fixed


def _preview_one(user_id: int) -> None:
    """Prints privacy-preserving migration statistics for one user."""
    scope = user_scope(user_id=user_id)
    main_text = read_main_memory(scope=scope)
    if not main_text:
        console.print(f"[yellow]{user_id}: skip (no main.md)[/yellow]")
        return
    bullets, untagged = count_untagged_bullets(text=main_text)
    console.print(
        f"[cyan]{user_id}: dry-run[/cyan] main_chars={len(main_text)} "
        f"bullets={bullets} untagged={untagged}"
    )


async def _migrate_one(
    client: AsyncOpenAI, model: ModelSettings, semaphore: asyncio.Semaphore, user_id: int
) -> None:
    """Rewrites one user's main.md into the tagged format and seeds tone.md."""
    scope = user_scope(user_id=user_id)
    existing_main = read_main_memory(scope=scope)
    if not existing_main:
        console.print(f"[yellow]{user_id}: skip (no main.md)[/yellow]")
        return
    async with semaphore:
        result = await parse_responses_or_none(
            client=client,
            model=model,
            instructions=MIGRATION_PROMPT,
            user_text=f"<memory_file>\n{existing_main}\n</memory_file>",
            end_user_id="memory_migrate",
            text_format=ConsolidatedMemory,
            timeout_seconds=MIGRATION_TIMEOUT_SECONDS,
        )
    if result is None or not result.memory_markdown.strip().startswith("v1\n"):
        console.print(f"[red]{user_id}: failed (LLM path or malformed rewrite)[/red]")
        return
    rewritten, backstop_tagged = tag_untagged_bullets(text=result.memory_markdown.strip())
    # Same shape as the pipeline's non-compact shrink guard: the migration only
    # re-tags, extracts tone, and rewrites the profile, so losing over half of a
    # non-trivial file reads as dropped content, not migration.
    if len(existing_main) > 2_000 and len(rewritten) < len(existing_main) // 2:
        console.print(
            f"[red]{user_id}: failed (shrank {len(existing_main)} -> {len(rewritten)})[/red]"
        )
        return
    write_main_memory(
        scope=scope,
        content=rewritten,
        identity=read_main_identity(scope=scope) or f"[id: {user_id}]",
    )
    tone = result.tone_markdown.strip()
    if tone.startswith("## 語氣偏好"):
        write_tone(scope=scope, content=tone)
    bullets, untagged = count_untagged_bullets(text=rewritten)
    console.print(
        f"[green]{user_id}: migrated[/green] bullets={bullets} untagged={untagged} "
        f"backstop_tagged={backstop_tagged} tone={'yes' if tone else 'no'}"
    )


async def _migrate_all(model: ModelSettings, folder: str, apply: bool) -> None:
    """Migrates every resolved user's main.md, bounded by the local semaphore."""
    resolved_folder = Path(folder)
    _repoint_store_root(folder=resolved_folder)
    user_ids = _resolve_user_ids(folder=resolved_folder)
    if not apply:
        console.print(
            f"Dry-run preview for {len(user_ids)} user(s). Pass --apply to rewrite main.md."
        )
        for user_id in user_ids:
            _preview_one(user_id=user_id)
        return
    config = LLMConfig()
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    semaphore = asyncio.Semaphore(MIGRATION_CONCURRENCY)
    console.print(
        f"Migrating {len(user_ids)} user(s) with [bold]{model.name}[/bold] "
        f"(effort: {model.effort})"
    )
    tasks = []
    for user_id in user_ids:
        tasks.append(
            _migrate_one(client=client, model=model, semaphore=semaphore, user_id=user_id)
        )
    await asyncio.gather(*tasks)


def _parse_args() -> argparse.Namespace:
    """Parses the offline migration CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Tag existing memories with [src:...] and seed tone.md (one-shot migration)."
    )
    parser.add_argument("--folder", default="./data/memories")
    parser.add_argument("--model", default="azure/gpt-5.5")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_settings = ModelSettings(name=args.model, effort=cast("ReasoningEffort", args.effort))
    asyncio.run(main=_migrate_all(model=model_settings, folder=args.folder, apply=args.apply))
