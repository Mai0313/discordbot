"""LLM review and consolidation for long-term memory, in either flavor.

Shares validation gates, transcript rendering, and secret redaction across user and server flavors.
"""

import re
from typing import TYPE_CHECKING, cast

from openai import AsyncOpenAI
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.utils.llm import parse_responses_or_none
from discordbot.typings.memory import (
    TONE_HEADER,
    MemoryFlavor,
    MemorySection,
    MemorySharing,
    MemoryCategory,
    MemoryConfidence,
    MemoryDurability,
    MemoryDeltaAction,
    MemoryEvidenceKind,
)
from discordbot.typings.models import ModelSettings
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE, FORWARDED_MESSAGE_MARKER
from discordbot.services.memory.prompts import (
    PHASE2_PROMPT,
    TONE_FORGET_PROMPT,
    PHASE1_EVALUATOR_PROMPT,
    PHASE2_COMPACTION_BLOCK,
)
from discordbot.typings.context_budgets import (
    MEMORY_NOTE_MAX_CHARS,
    MEMORY_REPLY_MAX_CHARS,
    MEMORY_TRANSCRIPT_MAX_CHARS,
)
from discordbot.services.memory.constants import (
    OBSERVATION_MAX_TTL_DAYS,
    OBSERVATION_QUOTE_MAX_CHARS,
    OBSERVATION_DEFAULT_TTL_DAYS,
    OBSERVATION_SUMMARY_MAX_CHARS,
)
from discordbot.services.memory.server_prompts import (
    SERVER_PHASE2_PROMPT,
    SERVER_CATCHUP_PROMPT,
    SERVER_PHASE1_EVALUATOR_PROMPT,
)

if TYPE_CHECKING:
    from openai.types.responses.response_input_text_param import ResponseInputTextParam

# Shape-specific secret patterns scrubbed on input and output (bare hex excluded to preserve git
# SHAs). Dotted tokens use ASCII lookbehinds rather than `\b` because Python treats CJK as `\w`.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"mfa\.[A-Za-z0-9_-]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9_])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9_-]{23,28}\.[A-Za-z0-9_-]{6,7}\.[A-Za-z0-9_-]{27,}"),
)

_AUTHOR_PREFIX_RE = re.compile(r"^[^\n]*?\[id: (?P<user_id>\d+)\]:")
# Matches indented author prefixes in rendered transcripts to build the participant roster.
_PARTICIPANT_PREFIX_RE = re.compile(
    r"^[ \t]*(?P<display>.+?) \((?P<username>[^()\n]+)\) \[id: (?P<user_id>\d+)\]:",
    flags=re.MULTILINE,
)
_LATIN_NAME_RE = re.compile(r"^[\w.\- ]+$", flags=re.ASCII)
# Matches `[id: N]` and Discord mentions in observation text; captures id to exempt target and bot.
_OTHER_PERSON_TOKEN_RE = re.compile(r"\[id:\s*(?P<user_id>\d+)\]|<@!?(?P<mention_id>\d+)>")
# Extracts `target_user_id` from a phase-1 subject; absent for server flavor.
_SUBJECT_TARGET_USER_RE = re.compile(r"^target_user_id:\s*(?P<user_id>\d+)", flags=re.MULTILINE)
# Co-located parser for the optional `source` line written by `user_subject`.
_SUBJECT_SOURCE_RE = re.compile(r"^source: (?P<source>guild \d+|dm)$", flags=re.MULTILINE)
_KEY_SAFE_RE = re.compile(r"[^a-z0-9._:-]+")
# Column-0 message marker used to align middle-truncated tails to a trusted block boundary.
_BLOCK_MARKER_RE = re.compile(r"^\[message \d+ \| ", flags=re.MULTILINE)
# Matches `[memory notes | remember|forget]` blocks and their indented bodies.
_NOTES_BLOCK_RE = re.compile(
    r"^\[memory notes \| (?P<kind>remember|forget)\]$(?P<body>(?:\n(?:[ \t]+.*)?)*)",
    flags=re.MULTILINE,
)
_REJECTED_EVIDENCE_KINDS = frozenset({
    "casual_mention",
    "hypothetical",
    "bot_suggestion",
    "other_user_context",
    "unknown",
})
_STABLE_EVIDENCE_KINDS = frozenset({
    "explicit_preference",
    "repeated_behavior",
    "correction",
    "stable_fact",
    "recurring_pattern",
    "tool_usage",
})


class MemoryObservation(BaseModel):
    """One validated phase-1 observation before markdown rendering."""

    model_config = ConfigDict(frozen=True)

    category: MemoryCategory = Field(
        ...,
        description="The memory section this observation belongs to.",
        examples=["stable_preference", "recent_context"],
    )
    subject_is_target_user: bool = Field(
        ..., description="Whether the evidence is about the target user, not another participant."
    )
    evidence_kind: MemoryEvidenceKind = Field(
        ...,
        description="The evidence shape supporting or rejecting this observation.",
        examples=["explicit_preference", "casual_mention"],
    )
    confidence: MemoryConfidence = Field(
        ..., description="Confidence after attribution and durability checks.", examples=["high"]
    )
    durability: MemoryDurability = Field(
        ...,
        description="How long the observation should influence memory.",
        examples=["stable", "recent"],
    )
    promotion_eligible: bool = Field(
        ..., description="Whether this may be promoted into stable memory during consolidation."
    )
    normalized_key: str = Field(
        ...,
        description="Stable dedupe key for the same underlying observation.",
        examples=["preference.reply_language.zh_tw"],
    )
    sharing: MemorySharing = Field(
        ...,
        description=(
            "Whether the observation is safe to use in any conversation (`global`: harmless "
            "general facts like language preference, interests, tech background) or must stay "
            "confined to the conversation source it was learned in (`source_only`: secrets, "
            "feelings, plans, anything personal or involving another person; when unsure, "
            "source_only)."
        ),
        examples=["source_only"],
    )
    summary_zh: str = Field(..., description="Traditional Chinese memory delta.")
    evidence_quote: str = Field(..., description="Short evidence quote from the target user.")
    ttl_days: int | None = Field(
        default=None,
        description="Positive TTL for recent context; null for stable observations.",
        examples=[30],
    )


class RawMemoryDraft(BaseModel):
    """Structured phase-1 review output for one turn's memory notes."""

    model_config = ConfigDict(frozen=True)

    has_signal: bool = Field(
        ...,
        description="Whether the conversation contained durable memory-worthy signal about the target user",
    )
    observations: tuple[MemoryObservation, ...] = Field(
        default=(),
        description="Validated structured memory observations; empty when has_signal is false",
    )


class ServerCatchupNotes(BaseModel):
    """Memory notes about a server's community, proposed from a channel the bot was not part of."""

    model_config = ConfigDict(frozen=True)

    notes: tuple[str, ...] = Field(
        default=(),
        description=(
            "At most five one-sentence notes about the community, the most durable first; empty "
            "when nothing in the conversation is worth remembering."
        ),
        examples=[("This community plays Valorant together every Friday night.",)],
    )


class MemoryFactDelta(BaseModel):
    """One change a consolidation asks for against a single compartment.

    A bad pass loses one fact instead of a whole file, and a rejected batch is retried
    without re-deciding everything. Deltas are keyed by `fact_id`, which code mints and the
    model only ever echoes back.
    """

    model_config = ConfigDict(frozen=True)

    action: MemoryDeltaAction = Field(
        ..., description="Whether to add a fact, rewrite one, or drop one.", examples=["create"]
    )
    fact_id: str = Field(
        default="",
        description="Existing id for update/delete; empty for create (code mints it).",
        examples=["9f2c41a7be03d5e8"],
    )
    section: MemorySection = Field(
        ..., description="Which document section the fact belongs to.", examples=["preference"]
    )
    durability: MemoryDurability = Field(
        ..., description="Permanent, stable, or time-bound.", examples=["stable"]
    )
    summary: str = Field(..., description="One-line Traditional Chinese gist of the fact.")
    text: str = Field(..., description="The Traditional Chinese fact body, as it will be read.")
    from_keys: tuple[str, ...] = Field(
        default=(),
        description="normalized_keys of the observations this fact rests on.",
        examples=[("preference.reply_language.zh_tw",)],
    )
    subject_id: str = Field(
        default="",
        description="Member id a nickname row refers to; empty for any other section.",
        examples=["987654321098765432"],
    )
    display_name: str = Field(
        default="",
        description="Nickname row only: the member's current display name.",
        examples=["小李"],
    )
    aliases: tuple[str, ...] = Field(
        default=(),
        description="Nickname row only: the aliases the community calls that member.",
        examples=[("李董", "破貓親爹")],
    )


class ConsolidatedMemory(BaseModel):
    """Structured phase-2 consolidation output for one compartment."""

    model_config = ConfigDict(frozen=True)

    deltas: tuple[MemoryFactDelta, ...] = Field(
        default=(), description="Changes to apply; empty is a valid no-op."
    )
    tone_markdown: str = Field(
        default="",
        description=(
            f"Full rewritten per-user tone note starting with `{TONE_HEADER}`; empty when the "
            "corpus carries no tone signal. Only the request carrying `<tone_evidence>` "
            "emits one."
        ),
    )


class ToneForget(BaseModel):
    """Which tone-note lines and which tone evidence the forget requests name."""

    model_config = ConfigDict(frozen=True)

    drop_lines: tuple[int, ...] = Field(
        default=(),
        description="Numbers of the `<tone_note>` lines a forget request names; empty when none.",
        examples=[(2,)],
    )
    drop_evidence: tuple[int, ...] = Field(
        default=(),
        description=(
            "Numbers of the `<tone_evidence>` entries a forget request names; empty when none."
        ),
        examples=[(1, 4)],
    )


class ConsolidationRequest(BaseModel):
    """Inputs for one compartment's consolidation call.

    Empty optional blocks are rendered explicitly as `(empty)` by `_tagged`.
    """

    model_config = ConfigDict(frozen=True)

    compartment_note: str = Field(
        ..., description="Plain-English description of who may read this compartment."
    )
    allowed_sections: tuple[MemorySection, ...] = Field(
        ..., description="Sections a delta may name for this flavor."
    )
    existing_facts: str = Field(
        default="", description="The compartment's current facts, rendered with their ids."
    )
    existing_tone: str = Field(
        default="", description="Current tone note; ignored unless emit_tone."
    )
    raw_entries: str = Field(..., description="This compartment's share of the raw batch.")
    recent_detail: str = Field(
        default="", description="Cold evidence filtered to this compartment."
    )
    tone_evidence: str = Field(
        default="", description="Unpartitioned tone signal; carried only by the tone-note call."
    )
    global_reference: str = Field(
        default="",
        description="Global facts already stored, so a guild compartment does not restate them.",
    )
    today: str = Field(..., description="ISO date used for dating and aging.")
    compact: bool = Field(..., description="Whether the compartment is large enough to compact.")
    net_removal_limit: int | None = Field(
        default=None,
        description="Facts a compaction pass may remove net before it is refused; None when uncapped.",
    )
    emit_tone: bool = Field(..., description="Whether this call owns the tone note.")


class MemoryWriterAI(BaseModel):
    """Runs the memory LLM calls with best-effort fallbacks, for a scope of either flavor."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Async OpenAI client for the Responses API memory calls."
    )
    model: ModelSettings = Field(
        ...,
        description=(
            "Model running every memory call. Required rather than optional: the note review "
            "is the only step that authors a raw entry's fields, so omitting it would turn "
            "memory writing into a silent no-op."
        ),
    )
    bot_user_id: int | None = Field(
        default=None,
        description=(
            "The bot's own user id, which the sharing gate never counts as another "
            "participant, by name or by id. None for a writer that only consolidates."
        ),
    )

    async def evaluate(
        self, flavor: MemoryFlavor, subject: str, transcript: str, notes: tuple[str, ...]
    ) -> RawMemoryDraft | None:
        """Returns validated observations reviewed from `notes` against `transcript`, or None on failure.

        `<forget-memory>` notes bypass this call and are rendered directly into `raw.md` by
        `render_forget_requests`.

        Args:
            flavor (MemoryFlavor): Scope flavor (`'user'` or `'server'`), selecting the evaluator
                prompt.
            subject (str): Leading directive naming the target (`target_user_id: <id>` or
                `target_server_id: <id>`).
            transcript (str): Rendered conversation transcript.
            notes (tuple[str, ...]): Candidate memory notes to review.
        """
        if not notes:
            return RawMemoryDraft(has_signal=False)
        target_match = _SUBJECT_TARGET_USER_RE.search(subject)
        target_user_id = int(target_match.group("user_id")) if target_match else None
        roster = (
            participant_names_from_transcript(
                transcript=transcript, target_user_id=target_user_id, bot_user_id=self.bot_user_id
            )
            if target_user_id is not None
            else ()
        )
        draft = await self._parse(
            instructions=(
                SERVER_PHASE1_EVALUATOR_PROMPT if flavor == "server" else PHASE1_EVALUATOR_PROMPT
            ),
            user_text=(
                f"{subject}\n\n"
                f"Conversation transcript:\n{transcript}\n\n"
                f"<memory_notes>\n{render_memory_notes(notes=notes)}\n</memory_notes>"
            ),
            text_format=RawMemoryDraft,
            end_user_label="memory_evaluate",
        )
        if draft is None:
            return None
        return _validated_draft(
            draft=draft, target_user_id=target_user_id, bot_user_id=self.bot_user_id, roster=roster
        )

    async def propose_server_notes(self, subject: str, transcript: str) -> tuple[str, ...] | None:
        """Returns up to five community notes proposed from `transcript`, or None when the call fails.

        Used by `/memory server catchup` on channels the bot was not addressed in; the returned
        notes are then reviewed through `evaluate`.

        Args:
            subject (str): Phase-1 directive naming the target server.
            transcript (str): Rendered channel transcript.
        """
        max_notes = 5
        proposal = await self._parse(
            instructions=SERVER_CATCHUP_PROMPT,
            user_text=f"{subject}\n\nConversation transcript:\n{transcript}",
            text_format=ServerCatchupNotes,
            end_user_label="memory_catchup",
        )
        if proposal is None:
            return None
        return tuple(note.strip() for note in proposal.notes if note.strip())[:max_notes]

    async def consolidate(
        self, flavor: MemoryFlavor, request: ConsolidationRequest
    ) -> ConsolidatedMemory | None:
        """Returns one compartment's consolidation output, or None when the LLM path fails.

        Args:
            flavor (MemoryFlavor): Scope flavor (`'user'` or `'server'`), selecting the
                consolidation prompt.
            request (ConsolidationRequest): Per-compartment inputs and flags for the call.
        """
        sections = ", ".join(request.allowed_sections)
        blocks = [
            f"today: {request.today}",
            f"compartment: {request.compartment_note}",
            f"allowed sections: {sections}",
        ]
        if request.net_removal_limit is not None:
            blocks.append(f"net fact removals allowed: {request.net_removal_limit}")
        blocks += [
            _tagged(tag="existing_facts", body=request.existing_facts),
            _tagged(tag="raw_entries", body=request.raw_entries),
            _tagged(tag="recent_detail", body=request.recent_detail),
        ]
        if request.global_reference:
            blocks.append(_tagged(tag="global_reference", body=request.global_reference))
        if request.emit_tone:
            blocks.append(_tagged(tag="existing_tone", body=request.existing_tone))
            blocks.append(_tagged(tag="tone_evidence", body=request.tone_evidence))
        prompt = SERVER_PHASE2_PROMPT if flavor == "server" else PHASE2_PROMPT
        instructions = prompt + PHASE2_COMPACTION_BLOCK if request.compact else prompt
        result = await self._parse(
            instructions=instructions,
            user_text="\n\n".join(blocks),
            text_format=ConsolidatedMemory,
            end_user_label="memory_consolidate",
        )
        if result is None:
            return None
        return result.model_copy(
            update={
                "deltas": tuple(_redacted_delta(delta=delta) for delta in result.deltas),
                "tone_markdown": redact_secrets(text=result.tone_markdown).strip(),
            }
        )

    async def forget_tone(
        self, forgets: str, note_lines: tuple[str, ...], evidence: tuple[str, ...]
    ) -> ToneForget | None:
        """Returns which `note_lines` and `evidence` numbers `forgets` name, or None on failure.

        Args:
            forgets (str): Raw forget-request entries.
            note_lines (tuple[str, ...]): Current tone note lines (without the header).
            evidence (tuple[str, ...]): Tone-bearing observation lines predating the newest forget.
        """
        return await self._parse(
            instructions=TONE_FORGET_PROMPT,
            user_text="\n\n".join([
                _tagged(tag="forget_requests", body=forgets),
                _tagged(tag="tone_note", body=_numbered(lines=note_lines)),
                _tagged(tag="tone_evidence", body=_numbered(lines=evidence)),
            ]),
            text_format=ToneForget,
            end_user_label="memory_tone_forget",
        )

    async def _parse[OutputT: BaseModel](
        self, instructions: str, user_text: str, text_format: type[OutputT], end_user_label: str
    ) -> OutputT | None:
        """Runs one structured Responses API call via `parse_responses_or_none`."""
        return await parse_responses_or_none(
            client=self.client,
            model=self.model,
            instructions=instructions,
            user_text=user_text,
            end_user_id=end_user_label,
            text_format=text_format,
        )


def _tagged(tag: str, body: str) -> str:
    """Wraps one consolidation input block, marking an absent one explicitly."""
    return f"<{tag}>\n{body.strip() or '(empty)'}\n</{tag}>"


def _numbered(lines: tuple[str, ...]) -> str:
    """Numbers lines from 1, the handles a `ToneForget` answer points back with."""
    return "\n".join(f"[{number}] {line}" for number, line in enumerate(lines, start=1))


def _redacted_delta(delta: MemoryFactDelta) -> MemoryFactDelta:
    """Scrubs secret-shaped strings out of one delta's model-authored text."""
    return delta.model_copy(
        update={
            "summary": redact_secrets(text=delta.summary).strip(),
            "text": redact_secrets(text=delta.text).strip(),
            "display_name": redact_secrets(text=delta.display_name).strip(),
            "aliases": tuple(redact_secrets(text=alias).strip() for alias in delta.aliases),
        }
    )


def participant_names_from_transcript(
    transcript: str, target_user_id: int | None, bot_user_id: int | None
) -> tuple[str, ...]:
    """Returns matchable display names and usernames of participants other than target and bot.

    Args:
        transcript (str): Rendered conversation transcript.
        target_user_id (int | None): Target user's Discord id, or None for server memory.
        bot_user_id (int | None): Bot's own Discord user id, or None when unknown.
    """
    names: set[str] = set()
    for match in _PARTICIPANT_PREFIX_RE.finditer(transcript):
        if int(match.group("user_id")) in (target_user_id, bot_user_id):
            continue
        names.update((match.group("display").strip(), match.group("username").strip()))
    return tuple(sorted(name for name in names if _is_matchable_name(name=name)))


def _is_matchable_name(name: str) -> bool:
    """Whether a roster name is distinctive enough to lock an observation on."""
    # Minimum lengths by script: Latin uses ASCII word boundaries; CJK matches by substring.
    min_latin_name = 3
    min_other_name = 2
    floor = min_latin_name if _LATIN_NAME_RE.match(name) else min_other_name
    return len(name) >= floor


def _mentions_roster_name(text: str, roster: tuple[str, ...]) -> bool:
    """Returns whether `text` mentions any roster name (ASCII-bounded Latin or CJK substring)."""
    folded = text.casefold()
    for name in roster:
        candidate = name.casefold()
        if _LATIN_NAME_RE.match(name):
            if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", folded, flags=re.ASCII):
                return True
        elif candidate in folded:
            return True
    return False


def transcript_from_messages(message_list: list[EasyInputMessageParam], full_reply: str) -> str:
    """Returns `message_list` and `full_reply` rendered as an indented block transcript.

    Prefixes each message with `[message <n> | <role>]` at column 0 and indents body lines by
    two spaces.

    Args:
        message_list (list[EasyInputMessageParam]): Input messages from the reply context.
        full_reply (str): Assistant reply text for the turn, or `""` when none.
    """
    blocks: list[str] = []
    for message in message_list:
        text = _strip_forwarded_payload(text=_message_text(message=message))
        if not text:
            continue
        marker = f"[message {len(blocks) + 1} | {message['role']}]"
        blocks.append(f"{marker}\n{_indent_block(text=text)}")
    reply = USAGE_FOOTER_RE.sub("", full_reply).strip()
    if len(reply) > MEMORY_REPLY_MAX_CHARS:
        # Cap reply text so middle truncation preserves the current user message.
        reply = f"{reply[:MEMORY_REPLY_MAX_CHARS]}\n[... reply truncated ...]"
    if reply:
        # Omit empty reply blocks (e.g., server catchup) so the reviewer sees no bot turn.
        blocks.append(
            f"[message {len(blocks) + 1} | assistant reply (this turn)]\n{_indent_block(text=reply)}"
        )
    transcript = redact_secrets(text="\n\n".join(blocks))
    return _truncate_middle(text=transcript, max_chars=MEMORY_TRANSCRIPT_MAX_CHARS)


def target_centered_memory_messages(
    hist_messages: list[EasyInputMessageParam],
    reference_messages: list[EasyInputMessageParam],
    current_message: list[EasyInputMessageParam],
    target_user_id: int,
) -> list[EasyInputMessageParam]:
    """Returns reply context narrowed to target-centered evidence for the memory review.

    Args:
        hist_messages (list[EasyInputMessageParam]): Channel history messages (header first).
        reference_messages (list[EasyInputMessageParam]): Referenced reply-chain messages.
        current_message (list[EasyInputMessageParam]): Current turn's user message(s).
        target_user_id (int): Target user's Discord id.
    """
    return [
        *_target_centered_history_messages(
            hist_messages=hist_messages, target_user_id=target_user_id
        ),
        *reference_messages,
        *current_message,
    ]


# One turn's inline memory notes `(remember, forget)`; merged payloads hold rounds oldest-first.
type NoteRound = tuple[tuple[str, ...], tuple[str, ...]]


def render_turn_payload(transcript: str, rounds: tuple[NoteRound, ...]) -> str:
    """Returns `transcript` and `rounds` bundled into a single string for `memory_job.transcript`.

    Appends indented `[memory notes | <kind>]` blocks after `transcript` using column-0 markers.

    Args:
        transcript (str): Rendered conversation transcript.
        rounds (tuple[NoteRound, ...]): Ordered `(remember_notes, forget_notes)` rounds.
    """
    blocks = [transcript]
    for position, (remember, forget) in enumerate(rounds, start=1):
        for kind, notes in (("remember", remember), ("forget", forget)):
            lines = [text for note in notes if (text := note_text(note=note))]
            # Emit a forget delimiter before later rounds so parse_turn_payload splits rounds.
            if lines or (kind == "forget" and position < len(rounds)):
                body = _indent_block(text="\n".join(lines))
                blocks.append(f"[memory notes | {kind}]\n{body}".rstrip())
    return "\n\n".join(blocks)


def parse_turn_payload(payload: str) -> tuple[str, tuple[NoteRound, ...]]:
    """Returns `payload` split back into `(transcript, rounds)` in round order.

    Args:
        payload (str): Combined turn payload created by `render_turn_payload`.
    """
    rounds: list[NoteRound] = []
    remember: list[str] = []
    for match in _NOTES_BLOCK_RE.finditer(payload):
        notes = [
            stripped for line in match.group("body").splitlines() if (stripped := line.strip())
        ]
        if match.group("kind") == "remember":
            remember.extend(notes)
            continue
        rounds.append((tuple(remember), tuple(notes)))
        remember = []
    if remember:
        rounds.append((tuple(remember), ()))
    return _NOTES_BLOCK_RE.sub("", payload).rstrip(), tuple(rounds)


def render_memory_notes(notes: tuple[str, ...]) -> str:
    """Returns `notes` rendered as the numbered candidate list the evaluator reviews.

    Args:
        notes (tuple[str, ...]): Candidate memory notes.
    """
    return "\n".join(
        f"{index}. {note_text(note=note)}" for index, note in enumerate(notes, start=1)
    )


def note_text(note: str) -> str:
    """Returns `note` collapsed to a single redacted line bounded by `MEMORY_NOTE_MAX_CHARS`.

    Args:
        note (str): Raw inline memory note.
    """
    return _trim_text(text=redact_secrets(text=note), max_chars=MEMORY_NOTE_MAX_CHARS)


def user_subject(user_id: int, guild_id: int | None) -> str:
    """Returns the phase-1 subject directive for `user_id` and conversation source.

    Args:
        user_id (int): Target Discord user id.
        guild_id (int | None): Guild id where the turn occurred, or None for DMs.
    """
    source = f"guild {guild_id}" if guild_id is not None else "dm"
    return f"target_user_id: {user_id}\nsource: {source}"


def server_subject(server_id: int) -> str:
    """Returns the phase-1 subject directive for `server_id` (with no source line).

    Args:
        server_id (int): Target Discord guild id.
    """
    return f"target_server_id: {server_id}"


def parse_subject_source(subject: str) -> str | None:
    """Returns the conversation source from `subject` (`'guild <id>'` or `'dm'`), or None.

    Args:
        subject (str): Phase-1 subject directive string.
    """
    match = _SUBJECT_SOURCE_RE.search(subject)
    return match.group("source") if match else None


def redact_secrets(text: str) -> str:
    """Returns `text` with token-, key-, and credential-shaped strings replaced by `[REDACTED_SECRET]`.

    Args:
        text (str): Text to scrub.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


def _validated_draft(
    draft: RawMemoryDraft,
    target_user_id: int | None,
    bot_user_id: int | None,
    roster: tuple[str, ...],
) -> RawMemoryDraft:
    """Applies deterministic high-precision gates to model observations."""
    observations: list[MemoryObservation] = []
    seen_keys: set[str] = set()
    for observation in draft.observations:
        sanitized = _sanitize_observation(
            observation=observation,
            target_user_id=target_user_id,
            bot_user_id=bot_user_id,
            roster=roster,
        )
        if sanitized.normalized_key in seen_keys:
            continue
        if not _is_accepted_observation(observation=sanitized):
            continue
        observations.append(sanitized)
        seen_keys.add(sanitized.normalized_key)
    return RawMemoryDraft(has_signal=bool(observations), observations=tuple(observations))


def _mentions_other_person(text: str, target_user_id: int | None, bot_user_id: int | None) -> bool:
    """Whether the text references anyone other than the target user and the bot."""
    for match in _OTHER_PERSON_TOKEN_RE.finditer(text):
        mentioned = int(match.group("user_id") or match.group("mention_id"))
        if mentioned not in (target_user_id, bot_user_id):
            return True
    return False


def _sanitize_observation(
    observation: MemoryObservation,
    target_user_id: int | None,
    bot_user_id: int | None,
    roster: tuple[str, ...],
) -> MemoryObservation:
    """Normalizes text, keys, TTL, and sharing fields before validation."""
    category = observation.category
    ttl_days = observation.ttl_days
    promotion_eligible = observation.promotion_eligible
    durability = observation.durability
    if category == "recent_context":
        promotion_eligible = False
        durability = "recent"
        ttl_days = (
            OBSERVATION_DEFAULT_TTL_DAYS
            if ttl_days is None or ttl_days <= 0
            else min(ttl_days, OBSERVATION_MAX_TTL_DAYS)
        )
    else:
        ttl_days = None
    summary_zh = _trim_text(
        text=redact_secrets(text=observation.summary_zh), max_chars=OBSERVATION_SUMMARY_MAX_CHARS
    )
    evidence_quote = _trim_text(
        text=redact_secrets(text=observation.evidence_quote), max_chars=OBSERVATION_QUOTE_MAX_CHARS
    )
    # Deterministic sharing backstop: scan pre-trim text so truncated tokens cannot bypass the
    # gate. Only tightens personal observations to `source_only`; server memory has no sharing
    # compartments and never loosens a model-chosen `source_only` to `global`.
    sharing = observation.sharing
    if target_user_id is not None:
        scanned = f"{observation.summary_zh}\n{observation.evidence_quote}"
        if (
            category == "recent_context"
            or observation.evidence_kind == "ongoing_situation"
            or _mentions_other_person(
                text=scanned, target_user_id=target_user_id, bot_user_id=bot_user_id
            )
            or _mentions_roster_name(text=scanned, roster=roster)
        ):
            sharing = "source_only"
    return MemoryObservation(
        category=category,
        subject_is_target_user=observation.subject_is_target_user,
        evidence_kind=observation.evidence_kind,
        confidence=observation.confidence,
        durability=durability,
        promotion_eligible=promotion_eligible,
        normalized_key=_clean_normalized_key(value=observation.normalized_key),
        sharing=sharing,
        summary_zh=summary_zh,
        evidence_quote=evidence_quote,
        ttl_days=ttl_days,
    )


def _is_accepted_observation(observation: MemoryObservation) -> bool:
    """Returns whether an observation is precise enough to enter raw memory."""
    if not observation.subject_is_target_user:
        return False
    if observation.evidence_kind in _REJECTED_EVIDENCE_KINDS:
        return False
    if (
        not observation.normalized_key
        or not observation.summary_zh
        or not observation.evidence_quote
    ):
        return False
    if observation.category == "recent_context":
        return observation.confidence in {"medium", "high"} and observation.ttl_days is not None
    return (
        observation.promotion_eligible
        and observation.confidence == "high"
        and observation.durability in {"stable", "permanent"}
        and observation.evidence_kind in _STABLE_EVIDENCE_KINDS
    )


def _clean_normalized_key(value: str) -> str:
    """Normalizes a model-provided dedupe key into a compact safe token."""
    key = _KEY_SAFE_RE.sub(".", redact_secrets(text=value).strip().lower())
    key = re.sub(r"\.+", ".", key).strip(".")
    return key[:120]


def _trim_text(text: str, max_chars: int) -> str:
    """Collapses whitespace and caps one observation field."""
    trimmed = " ".join(text.split())
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[: max_chars - 3].rstrip() + "..."


def _target_centered_history_messages(
    hist_messages: list[EasyInputMessageParam], target_user_id: int
) -> list[EasyInputMessageParam]:
    """Keeps target history plus local neighboring context."""
    if not hist_messages:
        return []
    header, body = hist_messages[0], hist_messages[1:]
    keep_indexes: set[int] = set()
    for index, message in enumerate(body):
        if not _is_target_user_message(message=message, target_user_id=target_user_id):
            continue
        keep_indexes.update(range(max(0, index - 1), min(len(body), index + 2)))
    if not keep_indexes:
        return []
    centered: list[EasyInputMessageParam] = [header]
    previous = -1
    for index in sorted(keep_indexes):
        omitted = index - previous - 1
        if omitted > 0:
            centered.append(_omission_message(omitted_count=omitted))
        centered.append(body[index])
        previous = index
    trailing = len(body) - previous - 1
    if trailing > 0:
        centered.append(_omission_message(omitted_count=trailing))
    return centered


def _is_target_user_message(message: EasyInputMessageParam, target_user_id: int) -> bool:
    """Returns whether the trusted author prefix names the target user."""
    match = _AUTHOR_PREFIX_RE.match(_message_text(message=message))
    return match is not None and int(match.group("user_id")) == target_user_id


def _omission_message(omitted_count: int) -> EasyInputMessageParam:
    """Builds a neutral marker for omitted non-target history."""
    return EasyInputMessageParam(
        role="system", content=f"[{omitted_count} non-target history message(s) omitted]"
    )


def _indent_block(text: str) -> str:
    """Indents content lines so column-0 block markers cannot be forged in bodies."""
    return "\n".join(f"  {line}" for line in text.splitlines())


def _strip_forwarded_payload(text: str) -> str:
    """Strips text from `FORWARDED_MESSAGE_MARKER` onward to exclude forwarded content."""
    index = text.find(FORWARDED_MESSAGE_MARKER)
    if index == -1:
        return text
    return text[:index].rstrip()


def _message_text(message: EasyInputMessageParam) -> str:
    """Extracts the plain text from one input message, dropping non-text parts."""
    content = message["content"]
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for part in content:
        if part.get("type") != "input_text":
            continue
        # Narrow after runtime type check so text_part["text"] types as str.
        text_part = cast("ResponseInputTextParam", part)
        parts.append(text_part["text"])
    return "\n".join(parts).strip()


def _truncate_middle(text: str, max_chars: int) -> str:
    """Middle-truncates `text` to `max_chars`, aligning the tail to a column-0 block marker."""
    if len(text) <= max_chars:
        return text
    marker = "\n\n[... transcript truncated ...]\n\n"
    budget = max_chars - len(marker)
    head = budget * 2 // 3
    tail = budget - head
    raw_tail = text[len(text) - tail :]
    aligned = _BLOCK_MARKER_RE.search(raw_tail)
    tail_text = raw_tail[aligned.start() :] if aligned else raw_tail
    return f"{text[:head]}{marker}{tail_text}"
