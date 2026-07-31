"""Serialization and rendering for one-fact-per-file memory.

A fact file is a ``---`` fenced header of single-line ``key: value`` pairs followed by
the body. The header is deliberately not YAML: every value is a plain scalar, so a
hand-rolled reader needs no dependency (``pyyaml`` reaches this project only
transitively) and cannot be talked into constructing objects. Values are written with
their whitespace collapsed, so the reader never has to handle continuations.

Rendering turns a compartment's files back into the same Traditional Chinese document
shape the reply prompt has always been given, which is what keeps every downstream
consumer working unchanged: ``allowlist_ids_from_server_memory`` still finds
``## 成員稱呼``, and the four prompts that tell the model to read that table still point
at something real. Section keys are ASCII so the structured LLM schema stays English;
the headings below are the only place the two vocabularies meet.
"""

import re
from typing import Literal
import hashlib
from datetime import UTC, datetime

import logfire

from discordbot.typings.memory import MemoryFact, MemoryOwner, MemorySection, MemoryNodeType

type MemoryFlavor = Literal["user", "server"]

_FENCE = "---"
# The `<name> [id: <N>]` line `render_author_identity` / `render_server_identity`
# produce. It survives the `memory_job` round-trip as one string, so it is split into
# its two stamped fields here rather than being threaded as a pair through the DB.
_IDENTITY_RE = re.compile(r"^(?P<name>.*?)\s*\[id:\s*(?P<owner_id>\d+)\]\s*$")
_HEADER_LINE_RE = re.compile(r"^(?P<key>[a-z_]+):[ ]?(?P<value>.*)$")
# A fact id is minted by code and is the filename stem, so it must never be able to
# carry a path separator, a dot, or anything else the filesystem reads structurally.
FACT_ID_RE = re.compile(r"^[0-9a-f]{16}$")
# An `[id: N]` token anywhere in a fact body. Alias rows get theirs appended from the
# code-stamped `subject_id`, and the allowlist parser takes the FIRST match on a line,
# so a body that carries its own would silently win — and that body is distilled from
# messages anyone in the server can write.
_ID_TOKEN_RE = re.compile(r"\[id:\s*\d+\]")

# Rendered headings per flavor, in document order. The tuples double as the per-flavor
# section allowlist: a delta naming a section absent here is dropped.
_USER_SECTION_HEADINGS: tuple[tuple[MemorySection, str], ...] = (
    ("profile", "使用者輪廓"),
    ("permanent", "永久事實"),
    ("preference", "穩定偏好"),
    ("fact", "穩定事實"),
    ("interaction", "互動筆記"),
    ("recent", "近期脈絡"),
)
_SERVER_SECTION_HEADINGS: tuple[tuple[MemorySection, str], ...] = (
    ("profile", "伺服器輪廓"),
    ("culture", "社群文化"),
    ("topic", "常見話題"),
    ("fact", "重要事實"),
    ("member_alias", "成員稱呼"),
    ("recent", "近期脈絡"),
)

# Appended when the size governor stops rendering, so the model is told the document is
# partial instead of silently reading a truncated profile as the whole person.
TRUNCATION_NOTICE = "（記憶已達可注入上限，較舊的內容未列出）"


def section_headings(flavor: MemoryFlavor) -> tuple[tuple[MemorySection, str], ...]:
    """Returns the flavor's sections in document order, paired with their headings."""
    return _USER_SECTION_HEADINGS if flavor == "user" else _SERVER_SECTION_HEADINGS


def sections_for_flavor(flavor: MemoryFlavor) -> frozenset[MemorySection]:
    """Returns the sections a delta may name for this flavor."""
    return frozenset(section for section, _ in section_headings(flavor=flavor))


def node_type_for(section: MemorySection) -> MemoryNodeType:
    """Derives the stored node type from the section so the two cannot drift."""
    return "member_alias" if section == "member_alias" else "memory"


def parse_identity(identity: str, fallback_owner_id: int) -> MemoryOwner:
    """Splits a rendered identity line into the owner fields stamped onto a fact.

    A line that does not parse (a job persisted before the format existed, a
    hand-edited row) keeps the id the scope key already carries and drops the name,
    which the next online write fills back in.
    """
    match = _IDENTITY_RE.match(identity.strip())
    if match is None:
        return MemoryOwner(owner_id=fallback_owner_id, owner_name=_one_line(text=identity))
    return MemoryOwner(
        owner_id=int(match.group("owner_id")), owner_name=_one_line(text=match.group("name"))
    )


def render_owner_identity(owner: MemoryOwner) -> str:
    """Renders a stored owner back into the identity line the pipeline threads around.

    The inverse of `parse_identity`, for the offline and restart paths that have a
    stored scope but no Discord context to rebuild the label from.
    """
    return f"{owner.owner_name} [id: {owner.owner_id}]".strip()


def mint_fact_id(compartment: str, summary: str) -> str:
    """Mints the code-owned id for a new fact.

    Derived from the compartment plus the summary rather than chosen by the model: the
    id is the filename, so letting conversation content reach it would put path
    traversal one prompt injection away. Including the compartment also means the same
    sentence distilled in two compartments gets two ids, so a cross-compartment
    collision cannot happen and the reader never has to arbitrate one.
    """
    digest = hashlib.sha256(f"{compartment}\0{' '.join(summary.split())}".encode())
    return digest.hexdigest()[:16]


def render_fact_file(fact: MemoryFact) -> str:
    """Renders one fact as its on-disk file."""
    header = {
        "id": fact.fact_id,
        "summary": _one_line(text=fact.summary),
        "section": fact.section,
        "durability": fact.durability,
        "compartment": fact.compartment,
        "owner_id": str(fact.owner_id),
        "owner_name": _one_line(text=fact.owner_name),
        "node_type": fact.node_type,
        "created": fact.created.isoformat(timespec="seconds"),
        "last_confirmed": fact.last_confirmed.isoformat(timespec="seconds"),
        "keys": ",".join(fact.keys),
    }
    if fact.subject_id is not None:
        header["subject_id"] = str(fact.subject_id)
    lines = [
        _FENCE,
        *(f"{key}: {value}" for key, value in header.items()),
        _FENCE,
        fact.text.strip(),
    ]
    return "\n".join(lines) + "\n"


def parse_fact_file(text: str, compartment: str) -> MemoryFact | None:
    """Parses one fact file, or returns None when it is unusable.

    `compartment` is the directory the file was found in and is authoritative: a stored
    `compartment` that disagrees means the tree was hand-edited or a migration stopped
    half way, and there is no safe way to guess which side is right. Returning None
    (the caller logs it) keeps the fact out of every reply rather than guessing the
    permissive answer.
    """
    header, body = _split_front_matter(text=text)
    if header is None:
        return None
    stored_compartment = header.get("compartment", "")
    if stored_compartment != compartment:
        logfire.error(
            "Memory fact compartment disagrees with its directory; skipping",
            directory=compartment,
            stored=stored_compartment,
            fact_id=header.get("id", ""),
        )
        return None
    try:
        return MemoryFact(
            fact_id=header["id"],
            summary=header["summary"],
            section=header["section"],  # ty: ignore[invalid-argument-type] -- validated by pydantic
            durability=header["durability"],  # ty: ignore[invalid-argument-type] -- validated by pydantic
            text=body,
            compartment=compartment,
            owner_id=int(header["owner_id"]),
            owner_name=header.get("owner_name", ""),
            subject_id=_optional_int(value=header.get("subject_id", "")),
            node_type=node_type_for(section=header["section"]),  # ty: ignore[invalid-argument-type] -- validated by pydantic
            created=datetime.fromisoformat(header["created"]),
            last_confirmed=datetime.fromisoformat(header["last_confirmed"]),
            keys=tuple(key for key in header.get("keys", "").split(",") if key),
        )
    except (KeyError, ValueError) as error:
        # Broad over the two shapes a bad header takes (a missing key, an unparsable
        # scalar or a value outside its Literal); either way the file is not a fact.
        logfire.warn(
            "Memory fact file is malformed; skipping",
            compartment=compartment,
            error_type=type(error).__name__,
        )
        return None


def render_memory_document(facts: list[MemoryFact], flavor: MemoryFlavor, max_chars: int) -> str:
    """Renders a compartment set as the Traditional Chinese document the prompts expect.

    Sections come out in the flavor's fixed order and facts within a section newest
    first, so a document trimmed at `max_chars` loses the stalest content rather than
    an arbitrary tail. Truncation only ever affects this rendering; nothing is deleted
    from disk, which is what stops the size cap fighting the next consolidation over
    facts it would immediately write back.
    """
    by_section: dict[MemorySection, list[MemoryFact]] = {}
    for fact in facts:
        by_section.setdefault(fact.section, []).append(fact)
    rendered: list[str] = []
    used = 0
    truncated = False
    for section, heading in section_headings(flavor=flavor):
        section_facts = sorted(
            by_section.get(section, []),
            key=lambda item: (item.last_confirmed, item.fact_id),
            reverse=True,
        )
        if not section_facts:
            continue
        block: list[str] = []
        for fact in section_facts:
            line = _render_fact_line(fact=fact, section=section)
            # +1 for the newline this line costs inside the joined document.
            if used + len(line) + 1 > max_chars:
                truncated = True
                break
            block.append(line)
            used += len(line) + 1
        if not block:
            continue
        rendered.append(f"## {heading}\n" + "\n".join(block))
        used += len(heading) + 4
    document = "\n\n".join(rendered).strip()
    if truncated and document:
        document = f"{document}\n\n{TRUNCATION_NOTICE}"
    return document


def _render_fact_line(fact: MemoryFact, section: MemorySection) -> str:
    """Renders one fact as its document line.

    The profile is a paragraph rather than a bullet (it always has been), a recent-context
    line carries the code-stamped date the model used to write itself, and an alias row
    has every id token stripped from its body before the real `subject_id` is appended —
    so the id can never be hallucinated (or injected by a member) onto the wrong person,
    and the table stays parseable by the allowlist reader.
    """
    body = " ".join(fact.text.split()) if section == "profile" else fact.text.strip()
    if section == "profile":
        return body
    if section == "recent":
        return f"* [{fact.last_confirmed.date().isoformat()}] {body}"
    if section == "member_alias" and fact.subject_id is not None:
        # Stripped, not escaped: the row's whole meaning is the name-to-id mapping, and
        # the only id that may appear in it is the one code stamped.
        return f"* {_ID_TOKEN_RE.sub('', body).strip()}[id: {fact.subject_id}]"
    return f"* {body}"


def _split_front_matter(text: str) -> tuple[dict[str, str] | None, str]:
    """Splits a fact file into its header mapping and its body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FENCE:
        return None, ""
    header: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == _FENCE:
            return header, "\n".join(lines[index + 1 :]).strip()
        match = _HEADER_LINE_RE.match(line)
        if match is None:
            return None, ""
        header[match.group("key")] = match.group("value").strip()
    return None, ""


def _optional_int(value: str) -> int | None:
    """Parses an optional numeric header value, treating an empty one as absent."""
    return int(value) if value else None


def _one_line(text: str) -> str:
    """Collapses a header value so it can never span lines or forge a fence."""
    return " ".join(text.split())


def utc_now() -> datetime:
    """Returns the timestamp used for every code-stamped fact date."""
    return datetime.now(UTC).replace(microsecond=0)
