"""Serialization and rendering for one-fact-per-file memory.

A fact file has a `---` fenced header of single-line `key: value` scalar pairs (not YAML)
followed by the body.

Rendering turns a compartment's files into a Traditional Chinese document whose `## 成員稱呼`
section is parsed back out by `parse_member_alias_table`. Section keys remain ASCII for the
structured LLM schema; the heading tables below map between the two.
"""

import re
import hashlib
from datetime import UTC, datetime, timedelta

import logfire

from discordbot.typings.memory import (
    MemoryFact,
    MemoryOwner,
    MemoryFlavor,
    MemorySection,
    MemoryNodeType,
)
from discordbot.services.memory.constants import RECENT_CONTEXT_TTL_DAYS

_FENCE = "---"
# Parses `<name> [id: <N>]` identity lines carried across memory_job rows.
_IDENTITY_RE = re.compile(r"^(?P<name>.*?)\s*\[id:\s*(?P<owner_id>\d+)\]\s*$")
_HEADER_LINE_RE = re.compile(r"^(?P<key>[a-z_]+):[ ]?(?P<value>.*)$")
# Code-minted hex filename stem; disallows path separators and dots.
FACT_ID_RE = re.compile(r"^[0-9a-f]{16}$")
# Matches `[id: N]` tokens stripped from alias bodies before appending the code-stamped subject_id.
_ID_TOKEN_RE = re.compile(pattern=r"\[id:\s*(?P<user_id>\d+)\]")

# Nickname table heading shared by renderer and parser.
_MEMBER_ALIAS_HEADING = "成員稱呼"
# Matches the nickname table section only at block start so bare profile lines cannot forge it.
_MEMBER_ALIAS_SECTION_RE = re.compile(
    pattern=rf"(?:\A|(?<=\n\n))##\s*{re.escape(pattern=_MEMBER_ALIAS_HEADING)}\s*$(?P<body>.*?)(?=^##\s|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)

# Rendered headings per flavor in document order; also defines the per-flavor section allowlist.
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
    ("member_alias", _MEMBER_ALIAS_HEADING),
    ("recent", "近期脈絡"),
)

# Appended when rendering hits max_chars so the model sees the document is partial.
TRUNCATION_NOTICE = "（記憶已達可注入上限，較舊的內容未列出）"


def section_headings(flavor: MemoryFlavor) -> tuple[tuple[MemorySection, str], ...]:
    """Returns the flavor's sections in document order, paired with their headings.

    Args:
        flavor (MemoryFlavor): Scope flavor (`'user'` or `'server'`).
    """
    return _USER_SECTION_HEADINGS if flavor == "user" else _SERVER_SECTION_HEADINGS


def sections_for_flavor(flavor: MemoryFlavor) -> frozenset[MemorySection]:
    """Returns the sections a delta may name for `flavor`.

    Args:
        flavor (MemoryFlavor): Scope flavor (`'user'` or `'server'`).
    """
    return frozenset(section for section, _ in section_headings(flavor=flavor))


def node_type_for(section: MemorySection) -> MemoryNodeType:
    """Returns the stored node type for `section`.

    Args:
        section (MemorySection): Fact section name.
    """
    return "member_alias" if section == "member_alias" else "memory"


def render_member_alias_text(display_name: str, aliases: tuple[str, ...]) -> str:
    """Returns a single-line alias body for `display_name` and `aliases`, or `""` when empty.

    Allows an empty `display_name`; an empty return causes the caller to drop the delta.

    Args:
        display_name (str): Member's current display name.
        aliases (tuple[str, ...]): Colloquial aliases the community calls that member.
    """
    collapsed = (_one_line(text=alias) for alias in aliases)
    unique = list(dict.fromkeys(alias for alias in collapsed if alias))
    if not unique:
        return ""
    return f"{_one_line(text=display_name)}(社群暱稱:{'、'.join(unique)})"


def parse_identity(identity: str, fallback_owner_id: int) -> MemoryOwner:
    """Returns the `MemoryOwner` parsed from a rendered identity line.

    A line that does not match `<name> [id: <N>]` uses `fallback_owner_id` and takes the
    collapsed line as `owner_name`.

    Args:
        identity (str): Rendered identity line.
        fallback_owner_id (int): Owner id from the scope key used when `identity` carries none.
    """
    match = _IDENTITY_RE.match(identity.strip())
    if match is None:
        return MemoryOwner(owner_id=fallback_owner_id, owner_name=_one_line(text=identity))
    return MemoryOwner(
        owner_id=int(match.group("owner_id")), owner_name=_one_line(text=match.group("name"))
    )


def render_owner_identity(owner: MemoryOwner) -> str:
    """Returns `owner` rendered into the `<name> [id: <N>]` identity line.

    The inverse of `parse_identity` for offline and restart paths that have no Discord context.

    Args:
        owner (MemoryOwner): Stored owner identity.
    """
    return f"{owner.owner_name} [id: {owner.owner_id}]".strip()


def mint_fact_id(compartment: str, summary: str) -> str:
    """Returns a 16-char hex fact id hashed from `compartment` and normalized `summary`.

    Args:
        compartment (str): Compartment the fact belongs to.
        summary (str): Fact summary line.
    """
    digest = hashlib.sha256(f"{compartment}\0{' '.join(summary.split())}".encode())
    return digest.hexdigest()[:16]


def render_fact_file(fact: MemoryFact) -> str:
    """Returns `fact` serialized as its on-disk file content.

    Args:
        fact (MemoryFact): Fact to serialize.
    """
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


def parse_fact_file(text: str, compartment: str, scope: str, filename: str) -> MemoryFact | None:
    """Returns the `MemoryFact` parsed from `text`, or None when unusable.

    `compartment` is authoritative: a disagreeing header `compartment` logs an error and returns
    None. Malformed headers log a warning and return None; missing front matter returns None
    silently.

    Args:
        text (str): Raw file content.
        compartment (str): Compartment directory the file was read from.
        scope (str): Memory scope key for logging.
        filename (str): Fact filename for logging.
    """
    header, body = _split_front_matter(text=text)
    if header is None:
        return None
    stored_compartment = header.get("compartment", "")
    if stored_compartment != compartment:
        logfire.error(
            "Memory fact compartment disagrees with its directory; skipping",
            scope=scope,
            filename=filename,
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
        logfire.warn(
            "Memory fact file is malformed; skipping",
            scope=scope,
            compartment=compartment,
            filename=filename,
            error_type=type(error).__name__,
            _exc_info=error,
        )
        return None


def render_memory_document(facts: list[MemoryFact], flavor: MemoryFlavor, max_chars: int) -> str:
    """Returns `facts` rendered as the Traditional Chinese document the prompts expect.

    Sections render in `flavor`'s fixed order and facts newest-first within each section; entries
    past `max_chars` are omitted with `TRUNCATION_NOTICE` appended.

    Args:
        facts (list[MemoryFact]): Facts to render.
        flavor (MemoryFlavor): Scope flavor (`'user'` or `'server'`).
        max_chars (int): Character budget before truncation.
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
            # +1 for the newline added when joining lines.
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
    """Renders one fact as a single line, collapsing whitespace and stripping id tokens."""
    body = " ".join(fact.text.split())
    if section == "profile":
        return body
    if section == "recent":
        return f"* [{fact.last_confirmed.date().isoformat()}] {body}"
    if section == "member_alias" and fact.subject_id is not None:
        # Loop because removing an inner `[id: N]` can join surrounding chars into a new token.
        while _ID_TOKEN_RE.search(string=body):
            body = _ID_TOKEN_RE.sub(repl="", string=body)
        return f"* {body.strip()}[id: {fact.subject_id}]"
    return f"* {body}"


def parse_member_alias_table(memory: str) -> dict[int, str]:
    """Returns the `## 成員稱呼` table from a rendered server memory document as `{user_id: label}`.

    Maps each row's `[id: USER_ID]` to the row minus its id token and bullet, unescaped. The
    first row naming an id wins, and a document without the section yields an empty dict.

    Args:
        memory (str): Rendered server memory document.
    """
    section = _MEMBER_ALIAS_SECTION_RE.search(string=memory)
    if section is None:
        return {}
    rows: dict[int, str] = {}
    for line in section.group("body").splitlines():
        match = _ID_TOKEN_RE.search(string=line)
        if match is None:
            continue
        user_id = int(match.group("user_id"))
        if user_id in rows:
            continue
        rows[user_id] = _ID_TOKEN_RE.sub(repl="", string=line).strip().lstrip("*").strip()
    return rows


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
    """Collapses whitespace in `text` into single spaces on one line."""
    return " ".join(text.split())


def utc_now() -> datetime:
    """Returns the timestamp used for every code-stamped fact date."""
    return datetime.now(UTC).replace(microsecond=0)


def today_utc() -> datetime:
    """Returns the current UTC day boundary the freshness rules compare against."""
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def is_past_recent_ttl(fact: MemoryFact, today: datetime) -> bool:
    """Returns whether `fact` is subject to the date TTL and has expired as of `today`.

    Exempts `permanent` durability or section facts, `member_alias` rows, and `stable` facts
    outside `recent` (which age by compartment displacement instead).

    Args:
        fact (MemoryFact): Stored fact to classify.
        today (datetime): UTC day boundary to compare against.
    """
    if (
        fact.durability == "permanent"
        or fact.section == "permanent"
        or fact.node_type == "member_alias"
        or (fact.durability == "stable" and fact.section != "recent")
    ):
        return False
    return today - fact.last_confirmed > timedelta(days=RECENT_CONTEXT_TTL_DAYS)
