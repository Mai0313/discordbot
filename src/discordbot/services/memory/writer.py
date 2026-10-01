"""LLM review and consolidation for long-term memory, in either flavor.

Each review and consolidation call is handed its scope's flavor, which picks the phase prompt,
so the per-user and the per-server memory share every gate, renderer and redaction here.
"""

import re
from typing import TYPE_CHECKING, TypeVar, cast

from openai import AsyncOpenAI
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.utils.llm import parse_responses_or_none
from discordbot.typings.memory import (
    TONE_HEADER,
    FORGET_REQUEST_CATEGORY,
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
    SERVER_PHASE1_EVALUATOR_PROMPT,
)

if TYPE_CHECKING:
    from openai.types.responses.response_input_text_param import ResponseInputTextParam

_OutputT = TypeVar("_OutputT", bound=BaseModel)

# Both phases run on model output that originated in user conversations, so
# secrets are scrubbed before upload and again on the model output. Patterns
# stay shape-specific on purpose: a bare-hex rule would also eat git SHAs,
# which are common non-secret content in a developer Discord. The prompts
# instruct the model to redact anything token-like as the generic backstop.
# The dotted shapes anchor their start on ASCII word characters, not `\b`, which counts
# Chinese as a word character and misses a token typed straight against it (#904).
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
# The trusted author prefix as it appears inside a rendered transcript block, indented
# by `_indent_block`. Read to recover who else took part in the conversation, so the
# sharing gate can recognise a third party named in plain prose rather than by id.
_PARTICIPANT_PREFIX_RE = re.compile(
    r"^[ \t]*(?P<display>.+?) \((?P<username>[^()\n]+)\) \[id: (?P<user_id>\d+)\]:",
    flags=re.MULTILINE,
)
_LATIN_NAME_RE = re.compile(r"^[\w.\- ]+$", flags=re.ASCII)
# Another participant referenced inside an observation's text (an id token or a raw
# Discord mention). Such an observation is about a relationship or someone else's
# business, so the sharing gate locks it to its source conversation. The id is captured
# so the gate can exempt the TARGET's own id (the transcript's author prefix makes it
# the most likely token to be quoted into evidence, and it names nobody else).
_OTHER_PERSON_TOKEN_RE = re.compile(r"\[id:\s*(?P<user_id>\d+)\]|<@!?(?P<mention_id>\d+)>")
# The target-user id inside a phase-1 subject; None for the server flavor.
_SUBJECT_TARGET_USER_RE = re.compile(r"^target_user_id:\s*(?P<user_id>\d+)", flags=re.MULTILINE)
# The optional second subject line naming where the conversation happened. Format and
# parser are co-located so the writer (`user_subject`) and the reader
# (`parse_subject_source`) cannot drift apart across the memory_job round-trip.
_SUBJECT_SOURCE_RE = re.compile(r"^source: (?P<source>guild \d+|dm)$", flags=re.MULTILINE)
_KEY_SAFE_RE = re.compile(r"[^a-z0-9._:-]+")
# Column-0 transcript block marker (`[message N | role]`). Used to realign a middle-
# truncated tail to a trusted block boundary so a sliced indent never leaves user
# content at column 0, where the marker scheme reserves the trusted authorship signal.
_BLOCK_MARKER_RE = re.compile(r"^\[message \d+ \| ", flags=re.MULTILINE)
# One `[memory notes | <kind>]` block plus its indented body, as `render_turn_payload` writes it.
# The body runs to the first line that is neither indented nor blank, which is the next column-0
# marker or the end of the payload.
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
    """Everything one compartment's consolidation call is given.

    A block that defaults to `""` is one some calls have nothing to put in. `_tagged` marks
    an absent block explicitly, so an omitted one and one passed as `""` reach the model
    identically.
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
            "The bot's own user id, which the sharing gate's roster leaves out. None for a "
            "writer that only consolidates."
        ),
    )

    async def evaluate(
        self, flavor: MemoryFlavor, subject: str, transcript: str, notes: tuple[str, ...]
    ) -> RawMemoryDraft | None:
        """Turns the answer model's own memory notes into validated observations.

        `notes` are the `<write-memory>` sentences the answer model wrote inside the reply it had
        just given, so nothing here guesses at what mattered in a conversation it was not part of.
        What is left for this call is the half that never worked well from the outside: reviewing
        each note against the transcript it came from, and authoring the structured fields a raw
        entry needs — which is why the evaluator prompt carries the field rules.

        `flavor` is the scope's, and picks the evaluator prompt. `subject` is the leading directive
        naming the memory's target (`target_user_id: <id>` or `target_server_id: <id>`). The server
        flavor deliberately parses to no target user, which leaves the roster empty too: a server
        memory has no single subject for the sharing gate to protect, and its observations carry
        no sharing field at all.

        `<forget-memory>` notes do NOT come through here. A forget is an instruction to
        consolidation rather than something to store, so it needs none of the fields this call
        authors and none of the gates that decide whether a fact is worth keeping;
        `render_forget_requests` writes it straight into the raw batch.
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
        return _validated_draft(draft=draft, target_user_id=target_user_id, roster=roster)

    async def consolidate(
        self, flavor: MemoryFlavor, request: ConsolidationRequest
    ) -> ConsolidatedMemory | None:
        """Returns one compartment's consolidation deltas, or None when the LLM path fails.

        `flavor` is the scope's, and picks the consolidation prompt.
        """
        sections = ", ".join(request.allowed_sections)
        blocks = [
            f"today: {request.today}",
            f"compartment: {request.compartment_note}",
            f"allowed sections: {sections}",
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
        """Asks which tone-note lines and which tone evidence the forget requests name.

        The answer is numbers into the two lists and nothing else, so a caller can only ever
        drop what it rendered here: nothing a forget says, and nothing the model writes, can
        land in the note. None means the LLM path failed.
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

    async def _parse(
        self, instructions: str, user_text: str, text_format: type[_OutputT], end_user_label: str
    ) -> _OutputT | None:
        """Runs one structured Responses API call, returning None on any failure.

        Delegates to the shared `parse_responses_or_none`, which owns the call surface and
        the degrade-to-None handling (refused output, an incomplete/truncated response — the
        last matters here because a half-emitted delta batch is indistinguishable from a
        complete one, and a caller may delete every fact the batch did not re-emit).

        No deadline is passed: every phase runs in the background with nobody waiting on it,
        so the client's own ceiling is the right bound. What the fan-out AROUND these calls
        needs is a different question.
        """
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
    """Returns the display names and usernames of everyone but the target and the bot.

    The trusted author prefix is the only authorship signal in a rendered transcript,
    so the roster is read from it rather than threaded down from the reply pipeline —
    which also means a resumed job rebuilds the same roster from its stored transcript
    with no extra column. The bot's own message carries that prefix when it had
    attachments, so its id is skipped like the target's.

    A forged prefix inside someone's message body can only ADD a name, and an extra name
    can only tighten an observation's sharing, so the untrusted position costs nothing.
    """
    names: set[str] = set()
    for match in _PARTICIPANT_PREFIX_RE.finditer(transcript):
        if int(match.group("user_id")) in (target_user_id, bot_user_id):
            continue
        names.update((match.group("display").strip(), match.group("username").strip()))
    return tuple(sorted(name for name in names if _is_matchable_name(name=name)))


def _is_matchable_name(name: str) -> bool:
    """Whether a roster name is distinctive enough to lock an observation on."""
    # Shortest roster name the gate will match on, split by script. A Latin name also has
    # to land on a word boundary, which a CJK name cannot (there are no spaces), so the CJK
    # floor carries that burden on its own.
    min_latin_name = 3
    min_other_name = 2
    floor = min_latin_name if _LATIN_NAME_RE.match(name) else min_other_name
    return len(name) >= floor


def _mentions_roster_name(text: str, roster: tuple[str, ...]) -> bool:
    """Whether the text names another participant in plain prose.

    Latin names must land on an ASCII word boundary, so `amy` does not fire on `amylase` but
    does on `跟Amy吵架`, where Chinese puts no space around a name; a CJK
    name has no boundaries to anchor to and is matched as a substring, which is the
    deliberate asymmetry — a false positive keeps a harmless fact inside one guild,
    while a false negative publishes a private one everywhere.
    """
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
    """Renders the reply-pipeline input messages plus the streamed reply as plain text.

    Each message becomes a block whose `[message <n> | <role>]` marker sits at
    column 0 while every content line is indented, so user-authored text can
    never forge a block boundary or plant an author prefix at content start.
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
        # The reply is secondary evidence; capping it keeps the tail of the
        # middle-truncation budget free for the current user message.
        reply = f"{reply[:MEMORY_REPLY_MAX_CHARS]}\n[... reply truncated ...]"
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
    """Narrows reply context to target-centered evidence for the memory review."""
    return [
        *_target_centered_history_messages(
            hist_messages=hist_messages, target_user_id=target_user_id
        ),
        *reference_messages,
        *current_message,
    ]


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


# One turn's inline memory notes, `(remember, forget)`. A payload merged from several waiting
# turns carries one per turn, oldest first; `parse_turn_payload` has why.
type NoteRound = tuple[tuple[str, ...], tuple[str, ...]]


def render_turn_payload(transcript: str, rounds: tuple[NoteRound, ...]) -> str:
    """Bundles one turn's transcript and its inline memory notes into a single stored string.

    The notes ride inside the `transcript` column rather than in columns of their own: nothing
    migrates this schema, so a new column would take `/memory clear` down on a deployed bot.
    What the column holds is still one thing — everything the background turn needs to run.

    Appended AFTER the transcript's own truncation, so a long conversation can never push the
    notes out of the payload. Each block reuses the column-0 marker shape the transcript
    already uses, with the notes indented under it, so the split back out cannot be forged by
    conversation content that happens to contain the header line.
    """
    blocks = [transcript]
    for position, (remember, forget) in enumerate(rounds, start=1):
        for kind, notes in (("remember", remember), ("forget", forget)):
            lines = [text for note in notes if (text := _note_text(note=note))]
            # A round with another after it always ends in its forget block, empty or not,
            # since that block is where `parse_turn_payload` closes a round.
            if lines or (kind == "forget" and position < len(rounds)):
                body = _indent_block(text="\n".join(lines))
                blocks.append(f"[memory notes | {kind}]\n{body}".rstrip())
    return "\n\n".join(blocks)


def parse_turn_payload(payload: str) -> tuple[str, tuple[NoteRound, ...]]:
    """Splits a stored payload back into its transcript and its rounds of notes, in order.

    The inverse of `render_turn_payload`. Rounds stay apart because they are staged in order: a
    newer turn's forget can name what an older one asked to remember, which has to be in
    `raw.md` ahead of it for the forget to reach. Each round writes its remember block before
    its forget block, and a forget block closes a round, so a row stored as one turn with at
    most one block of each kind still reads as the single round it was.
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
    """Renders the answer model's notes as the numbered candidate list the evaluator reviews."""
    return "\n".join(
        f"{index}. {_note_text(note=note)}" for index, note in enumerate(notes, start=1)
    )


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
            f"- text: {_note_text(note=note)}",
        ])
        for note in notes
        if _note_text(note=note)
    ]
    return "\n\n".join(blocks)


def _note_text(note: str) -> str:
    """Collapses one inline memory note to a single redacted, bounded line."""
    return _trim_text(text=redact_secrets(text=note), max_chars=MEMORY_NOTE_MAX_CHARS)


def user_subject(user_id: int, guild_id: int | None) -> str:
    """Renders the subject of one user's memory review.

    The second line names where the conversation happened: the guild, or `dm` when
    `guild_id` is None.
    """
    source = f"guild {guild_id}" if guild_id is not None else "dm"
    return f"target_user_id: {user_id}\nsource: {source}"


def server_subject(server_id: int) -> str:
    """Renders the subject of one server's memory review, which carries no source line."""
    return f"target_server_id: {server_id}"


def parse_subject_source(subject: str) -> str | None:
    """Extracts the conversation source from a persisted subject, or None when absent.

    None is the server flavor, whose subject never carries a source line: a server memory
    is one guild by construction, so there is nothing to scope its observations by and they
    render without per-observation source stamping.
    """
    match = _SUBJECT_SOURCE_RE.search(subject)
    return match.group("source") if match else None


def redact_secrets(text: str) -> str:
    """Replaces token-, key-, and password-like strings with a redaction marker."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


def _validated_draft(
    draft: RawMemoryDraft, target_user_id: int | None, roster: tuple[str, ...]
) -> RawMemoryDraft:
    """Applies deterministic high-precision gates to model observations."""
    observations: list[MemoryObservation] = []
    seen_keys: set[str] = set()
    for observation in draft.observations:
        sanitized = _sanitize_observation(
            observation=observation, target_user_id=target_user_id, roster=roster
        )
        if sanitized.normalized_key in seen_keys:
            continue
        if not _is_accepted_observation(observation=sanitized):
            continue
        observations.append(sanitized)
        seen_keys.add(sanitized.normalized_key)
    return RawMemoryDraft(has_signal=bool(observations), observations=tuple(observations))


def _mentions_other_person(text: str, target_user_id: int | None) -> bool:
    """Whether the text references any participant other than the target user."""
    for match in _OTHER_PERSON_TOKEN_RE.finditer(text):
        mentioned = int(match.group("user_id") or match.group("mention_id"))
        if target_user_id is None or mentioned != target_user_id:
            return True
    return False


def _sanitize_observation(
    observation: MemoryObservation, target_user_id: int | None, roster: tuple[str, ...]
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
    # Deterministic privacy backstop over the LLM's sharing call: ongoing situations
    # are private by construction, and an observation about ANOTHER participant is
    # about a relationship, not a portable fact (the target's own id — e.g. a quoted
    # author prefix — names nobody else and stays exempt). Scans the pre-trim text so
    # a token past the truncation point cannot dodge the gate. Code only ever tightens
    # sharing to source_only; it never loosens a source_only call back to global.
    #
    # The roster half exists because `global` is permanent cross-server reach with no
    # read-time filter behind it, so "他跟女友吵架" — which carries no id token at all —
    # cannot be left entirely to the model's own judgement. Matching the conversation's other
    # participants literally is the deterministic half; `PHASE1_EVALUATOR_PROMPT` covers whoever
    # is named but absent.
    scanned = f"{observation.summary_zh}\n{observation.evidence_quote}"
    sharing = observation.sharing
    if (
        category == "recent_context"
        or observation.evidence_kind == "ongoing_situation"
        or _mentions_other_person(text=scanned, target_user_id=target_user_id)
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
    """Drops a block's forwarded snapshot span so memory never attributes it to the forwarder.

    `get_cleaned_content` appends forwarded text last under `FORWARDED_MESSAGE_MARKER`, so the
    first marker is the suffix boundary: everything from it to end-of-body is someone else's
    words and must not become a fact about the (target) forwarder. The answer still sees the
    full body; only this memory-evidence transcript excludes it.
    """
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
        # Narrow to the concrete text part type after the runtime type check, so the
        # `text` key reads as str instead of widening every part to dict[str, object].
        text_part = cast("ResponseInputTextParam", part)
        parts.append(text_part["text"])
    return "\n".join(parts).strip()


def _truncate_middle(text: str, max_chars: int) -> str:
    """Keeps the head and tail of an oversized transcript, dropping the middle.

    The tail is realigned forward to the next column-0 block marker so the resumed
    region always starts at a trusted `[message N | role]` boundary; without this a
    cut landing inside an indented body could leave user content at column 0 and forge
    a block boundary. When no marker lands inside the tail it is returned as a best
    effort.
    """
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
