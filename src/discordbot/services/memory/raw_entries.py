"""The entry grammar of `raw.md` and `detail.md`: how evidence is rendered, stamped and read back.

Both files hold entries under a `## <timestamp>` header, and an entry holds blocks that each
open with a `### <category>` header followed by `- name: value` fields. Detail entries are
retired raw entries verbatim, so one grammar serves both files.
"""

import re
from datetime import UTC, datetime
from itertools import groupby, pairwise

from discordbot.typings.memory import FORGET_REQUEST_CATEGORY
from discordbot.services.memory.writer import MemoryObservation, note_text

# Raw entries start with a `## <ISO-8601 timestamp>` header line. An entry's
# body is bullet-style prose, so the date prefix doubles as the split marker;
# `timestamp` is the rest of the header line.
RAW_ENTRY_HEADER_RE = re.compile(
    r"^## (?P<timestamp>\d{4}-\d{2}-\d{2}T.*?)\s*$", flags=re.MULTILINE
)
# One observation block's header inside a raw entry.
_OBSERVATION_HEADER_RE = re.compile(r"^### (?P<category>\S+)")
_FIELD_RE = re.compile(r"^\s*-\s*(?P<name>[a-z_]+):\s*(?P<value>.*?)\s*$")
_GUILD_SOURCE_RE = re.compile(r"^guild (?P<guild_id>\d+)$")


def render_memory_observations(
    observations: tuple[MemoryObservation, ...], source: str | None
) -> str:
    """Renders structured observations as timestamp-entry body markdown.

    `source` names the conversation the observations came from (`guild <id>` /
    `dm`), stamped deterministically here — never LLM-echoed — so consolidation
    can scope each bullet. None is the server flavor, whose subject carries no
    source line; it renders neither the source nor the sharing field.
    """
    blocks: list[str] = []
    for observation in observations:
        ttl_text = "null" if observation.ttl_days is None else str(observation.ttl_days)
        lines = [
            f"### {observation.category}",
            f"- normalized_key: {observation.normalized_key}",
            f"- evidence_kind: {observation.evidence_kind}",
            f"- confidence: {observation.confidence}",
            f"- durability: {observation.durability}",
            f"- promotion_eligible: {str(observation.promotion_eligible).lower()}",
            f"- ttl_days: {ttl_text}",
        ]
        if source is not None:
            lines.append(f"- source: {source}")
            lines.append(f"- sharing: {observation.sharing}")
        lines.append(f"- summary_zh: {observation.summary_zh}")
        lines.append(f"- evidence_quote: {observation.evidence_quote}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_forget_requests(notes: tuple[str, ...], source: str | None) -> str:
    """Renders `<forget-memory>` notes as raw entries consolidation can act on.

    A forget is deliberately NOT a `MemoryObservation`. It is not something to store, so it needs
    no category, durability, sharing or dedupe key, and running it through the gates that decide
    whether a fact is worth keeping would only find reasons to drop it. Its own
    `### forget_request` header — deliberately not a `MemoryCategory` — is what keeps it
    invisible to every reader that walks observation fields.

    `source` is stamped for the record rather than for routing: routing a forget by its source
    would leave it unable to reach a fact stored anywhere else.
    """
    blocks = [
        "\n".join([
            f"### {FORGET_REQUEST_CATEGORY}",
            *([f"- source: {source}"] if source is not None else []),
            f"- text: {note_text(note=note)}",
        ])
        for note in notes
        if note_text(note=note)
    ]
    return "\n\n".join(blocks)


def stamp_entry(body: str) -> str:
    """Returns `body` as one new entry, headed by the current time.

    Microseconds, so the stamp orders entries the way they were written: a forget is set
    against everything stamped before it, and a deferred turn can land in the same second as
    the one it follows.
    """
    timestamp = datetime.now(UTC).isoformat(timespec="microseconds")
    return f"## {timestamp}\n{body.strip()}"


def split_raw_entries(text: str) -> list[str]:
    """Splits raw file text into stripped per-entry blocks including headers."""
    starts = [match.start() for match in RAW_ENTRY_HEADER_RE.finditer(text)]
    if not starts:
        return []
    bounds = [*starts, len(text)]
    blocks = [text[begin:end].strip() for begin, end in pairwise(bounds)]
    return [block for block in blocks if block]


def iter_observations(text: str) -> list[tuple[str, str]]:
    """Splits a raw or detail file into `(entry timestamp, observation block)` pairs."""
    pairs: list[tuple[str, str]] = []
    timestamp = ""
    current: list[str] = []

    def flush() -> None:
        block = "\n".join(current).strip()
        if block:
            pairs.append((timestamp, block))
        current.clear()

    for line in text.splitlines():
        header = RAW_ENTRY_HEADER_RE.match(line)
        if header is not None:
            flush()
            timestamp = header.group("timestamp")
            continue
        if _OBSERVATION_HEADER_RE.match(line):
            flush()
        current.append(line)
    flush()
    return pairs


def observation_category(block: str) -> str | None:
    """Returns the category in one block's `### <category>` header, or None without one."""
    header = _OBSERVATION_HEADER_RE.match(block)
    return header.group("category") if header is not None else None


def is_forget_request(block: str) -> bool:
    """Whether one raw block is a forget request rather than an observation."""
    return observation_category(block=block) == FORGET_REQUEST_CATEGORY


def fields_of(block: str) -> dict[str, str]:
    """Extracts one observation block's `- name: value` fields."""
    fields: dict[str, str] = {}
    for line in block.splitlines():
        match = _FIELD_RE.match(line)
        if match is not None:
            fields[match.group("name")] = match.group("value")
    return fields


def source_guild_id(source: str) -> int | None:
    """Returns the guild a `source` field names, or None for `dm` or any other value."""
    match = _GUILD_SOURCE_RE.match(source)
    return int(match.group("guild_id")) if match is not None else None


def newest_stamp(text: str) -> str:
    """Returns the latest entry stamp in a raw or detail text, or "" when it has none."""
    return max((timestamp for timestamp, _ in iter_observations(text=text)), default="")


def render_entries(blocks: list[tuple[str, str]]) -> str:
    """Re-renders bucketed observation blocks under their original entry headers.

    This is the form a consolidation prompt reads, with a blank line after each header;
    `render_file_entries` writes the on-disk form without one.
    """
    rendered: list[str] = []
    previous = ""
    for timestamp, block in blocks:
        if timestamp and timestamp != previous:
            rendered.append(f"## {timestamp}")
            previous = timestamp
        rendered.append(block)
    return "\n\n".join(rendered)


def render_file_entries(pairs: list[tuple[str, str]]) -> str:
    """Renders observation blocks the way `raw.md` and `detail.md` hold them on disk."""
    entries: list[str] = []
    for timestamp, group in groupby(pairs, key=lambda pair: pair[0]):
        body = "\n\n".join(block for _, block in group)
        entries.append(f"## {timestamp}\n{body}" if timestamp else body)
    return "\n\n".join(entries)
