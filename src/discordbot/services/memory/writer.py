"""LLM review and consolidation for long-term memory, in either flavor.

The phase prompts ride on `MemoryWriterAI` as fields, so the per-user and the
per-server memory share every gate, renderer and redaction here.
"""

import re
from typing import TYPE_CHECKING, TypeVar, cast

from openai import AsyncOpenAI
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.utils.llm import parse_responses_or_none
from discordbot.typings.memory import (
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

if TYPE_CHECKING:
    from openai.types.responses.response_input_text_param import ResponseInputTextParam

_OutputT = TypeVar("_OutputT", bound=BaseModel)

# Both phases run on model output that originated in user conversations, so
# secrets are scrubbed before upload and again on the model output. Patterns
# stay shape-specific on purpose: a bare-hex rule would also eat git SHAs,
# which are common non-secret content in a developer Discord. The prompts
# instruct the model to redact anything token-like as the generic backstop.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"mfa\.[A-Za-z0-9_-]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{23,28}\.[A-Za-z0-9_-]{6,7}\.[A-Za-z0-9_-]{27,}\b"),
)

_AUTHOR_PREFIX_RE = re.compile(r"^[^\n]*?\[id: (?P<user_id>\d+)\]:")
# The trusted author prefix as it appears inside a rendered transcript block, indented
# by `_indent_block`. Read to recover who else took part in the conversation, so the
# sharing gate can recognise a third party named in plain prose rather than by id.
_PARTICIPANT_PREFIX_RE = re.compile(
    r"^[ \t]*(?P<display>.+?) \((?P<username>[^()\n]+)\) \[id: (?P<user_id>\d+)\]:",
    flags=re.MULTILINE,
)
# Shortest roster name the gate will match on, split by script. A Latin name also has
# to land on a word boundary, which a CJK name cannot (there are no spaces), so the CJK
# floor carries that burden on its own.
_MIN_LATIN_ROSTER_NAME = 3
_MIN_OTHER_ROSTER_NAME = 2
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
# parser are co-located so the writer (`subject_source_line`) and the reader
# (`parse_subject_source`) cannot drift apart across the memory_job round-trip.
_SUBJECT_SOURCE_RE = re.compile(r"^source: (?P<source>guild \d+|dm)$", flags=re.MULTILINE)
_KEY_SAFE_RE = re.compile(r"[^a-z0-9._:-]+")
_STRUCTURED_KEY_RE = re.compile(r"^\s*-\s*normalized_key:\s*(?P<key>\S+)\s*$", flags=re.MULTILINE)
# The code-stamped `- source:` field inside one observation block; paired with the
# block's normalized_key by `observation_key_sources_from_text`.
_STRUCTURED_SOURCE_RE = re.compile(r"^\s*-\s*source:\s*(?P<source>guild \d+|dm)\s*$")
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
# The `### <category>` header a forget request carries inside `raw.md`. Deliberately not a
# `MemoryCategory`: a forget is an instruction to consolidation, not an observation to store, and
# keeping it out of that vocabulary is what keeps it out of every reader that walks observation
# fields. `render_forget_requests` has the rest.
FORGET_REQUEST_CATEGORY = "forget_request"

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

    Deltas replaced the whole-file rewrite so a bad pass can lose one fact instead of a
    file, and so a rejected batch can be retried without re-deciding everything. They
    are keyed by `fact_id`, which code mints and the model only ever echoes back.
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
            "Full rewritten per-user tone note starting with `## 語氣偏好`; empty when the "
            "corpus carries no tone signal. Only the request carrying `<tone_evidence>` "
            "emits one."
        ),
    )


class ConsolidationRequest(BaseModel):
    """Everything one compartment's consolidation call is given.

    Bundled rather than passed as a dozen arguments because the fan-out builds these
    per compartment and the differences between them (which raw bucket, whether tone is
    wanted, which sections are legal) are exactly what the caller has to get right.

    A block that defaults to `""` is one most calls have nothing to put in: the tone note's
    call carries no facts and no cold evidence, and a from-scratch rebuild deliberately
    withholds both. `_tagged` marks an absent block explicitly, so an omitted one and one
    passed as `""` reach the model identically.
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
    """Runs the memory LLM calls with best-effort fallbacks.

    The phase prompts are instance fields so the same engine can drive a
    different memory flavor (e.g. the bot's per-server memory) by swapping the
    prompts while reusing the evaluation, consolidation, validation, and
    redaction logic unchanged. They default to the per-user prompts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Async OpenAI client for the Responses API memory calls."
    )
    consolidate_model: ModelSettings = Field(
        ..., description="Model running the phase-2 consolidation call."
    )
    evaluate_model: ModelSettings = Field(
        ...,
        description=(
            "Model reviewing the answer model's memory notes. Required rather than optional: "
            "with the extraction pass gone it is the only step that authors a raw entry's "
            "fields, so a caller that omitted it would turn memory writing into a silent no-op."
        ),
    )
    evaluator_prompt: str = Field(
        default=PHASE1_EVALUATOR_PROMPT, description="Instructions for the evaluator call."
    )
    consolidate_prompt: str = Field(
        default=PHASE2_PROMPT, description="Instructions for the phase-2 consolidation call."
    )
    compaction_block: str = Field(
        default=PHASE2_COMPACTION_BLOCK,
        description="Extra block appended to the consolidation prompt when compacting.",
    )

    async def evaluate(
        self, subject: str, transcript: str, notes: tuple[str, ...]
    ) -> RawMemoryDraft | None:
        """Turns the answer model's own memory notes into validated observations.

        This replaced the phase-1 extraction pass (#596). `notes` are the `<write-memory>`
        sentences the answer model wrote inside the reply it had just given, so nothing here
        guesses at what mattered in a conversation it was not part of. What is left for this
        call is the half that never worked well from the outside: reviewing each note against
        the transcript it came from, and authoring the structured fields a raw entry needs.
        The second half used to belong to the deleted `extract`, which is why the evaluator
        prompt now carries its field rules.

        `subject` is the leading directive naming the memory's target (`target_user_id: <id>` or
        `target_server_id: <id>`). The server flavor deliberately parses to no target user, which
        leaves the roster empty too: a server memory has no single subject for the sharing gate
        to protect, and its observations carry no sharing field at all.

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
            participant_names_from_transcript(transcript=transcript, target_user_id=target_user_id)
            if target_user_id is not None
            else ()
        )
        draft = await self._parse(
            model=self.evaluate_model,
            instructions=self.evaluator_prompt,
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

    async def consolidate(self, request: ConsolidationRequest) -> ConsolidatedMemory | None:
        """Returns one compartment's consolidation deltas, or None when the LLM path fails."""
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
        instructions = (
            self.consolidate_prompt + self.compaction_block
            if request.compact
            else self.consolidate_prompt
        )
        result = await self._parse(
            model=self.consolidate_model,
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

    async def _parse(
        self,
        model: ModelSettings,
        instructions: str,
        user_text: str,
        text_format: type[_OutputT],
        end_user_label: str,
    ) -> _OutputT | None:
        """Runs one structured Responses API call, returning None on any failure.

        Delegates to the shared `parse_responses_or_none`, which owns the call surface and
        the degrade-to-None handling (refused output, an incomplete/truncated response — the
        last matters here because a half-emitted delta batch is indistinguishable from a
        complete one, and the rebuild path deletes every fact its batch did not re-emit).

        No deadline is passed: every phase runs in the background with nobody waiting on it,
        so the client's own ceiling is the right bound (`constants.py` has what a stuck call
        costs while it holds the scope lock). What the fan-out AROUND these calls needs is a
        different question, answered in `consolidation.py`.
        """
        return await parse_responses_or_none(
            client=self.client,
            model=model,
            instructions=instructions,
            user_text=user_text,
            end_user_id=end_user_label,
            text_format=text_format,
        )


def _tagged(tag: str, body: str) -> str:
    """Wraps one consolidation input block, marking an absent one explicitly."""
    return f"<{tag}>\n{body.strip() or '(empty)'}\n</{tag}>"


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
    transcript: str, target_user_id: int | None
) -> tuple[str, ...]:
    """Returns the display names and usernames of everyone in the transcript but the target.

    The trusted author prefix is the only authorship signal in a rendered transcript,
    so the roster is read from it rather than threaded down from the reply pipeline —
    which also means a resumed job rebuilds the same roster from its stored transcript
    with no extra column. The bot never carries an author prefix, so it is absent by
    construction rather than by an exclusion rule.

    A forged prefix inside someone's message body can only ADD a name, and an extra name
    can only tighten an observation's sharing, so the untrusted position costs nothing.
    """
    names: set[str] = set()
    for match in _PARTICIPANT_PREFIX_RE.finditer(transcript):
        if target_user_id is not None and int(match.group("user_id")) == target_user_id:
            continue
        names.update((match.group("display").strip(), match.group("username").strip()))
    return tuple(sorted(name for name in names if _is_matchable_name(name=name)))


def _is_matchable_name(name: str) -> bool:
    """Whether a roster name is distinctive enough to lock an observation on."""
    floor = _MIN_LATIN_ROSTER_NAME if _LATIN_NAME_RE.match(name) else _MIN_OTHER_ROSTER_NAME
    return len(name) >= floor


def _mentions_roster_name(text: str, roster: tuple[str, ...]) -> bool:
    """Whether the text names another participant in plain prose.

    Latin names must land on a word boundary so `amy` does not fire on `dynamic`; a CJK
    name has no boundaries to anchor to and is matched as a substring, which is the
    deliberate asymmetry — a false positive keeps a harmless fact inside one guild,
    while a false negative publishes a private one everywhere.
    """
    folded = text.casefold()
    for name in roster:
        candidate = name.casefold()
        if _LATIN_NAME_RE.match(name):
            if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", folded):
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


def render_turn_payload(
    transcript: str, remember: tuple[str, ...], forget: tuple[str, ...]
) -> str:
    """Bundles one turn's transcript and its inline memory notes into a single stored string.

    The notes ride inside the `transcript` column rather than in columns of their own. This
    repo has no migration mechanism (`_ensure_schema` is one `create_all`, which never alters
    an existing table) and `clear_job` is the one memory DB call not wrapped in best-effort
    handling, so a new column would take `/memory clear` down on a deployed bot. What the
    column holds is still one thing: everything the background turn needs to run.

    Appended AFTER the transcript's own truncation, so a long conversation can never push the
    notes out of the payload. Each block reuses the column-0 marker shape the transcript
    already uses, with the notes indented under it, so the split back out cannot be forged by
    conversation content that happens to contain the header line.
    """
    blocks = [transcript]
    for kind, notes in (("remember", remember), ("forget", forget)):
        lines = [text for note in notes if (text := _note_text(note=note))]
        if lines:
            body = _indent_block(text="\n".join(lines))
            blocks.append(f"[memory notes | {kind}]\n{body}")
    return "\n\n".join(blocks)


def parse_turn_payload(payload: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Splits a stored payload back into `(transcript, remember notes, forget notes)`.

    The inverse of `render_turn_payload`, and the reason a resumed job needs no extra column:
    a row persisted before restart carries its notes in the same string.
    """
    collected: dict[str, list[str]] = {"remember": [], "forget": []}
    for match in _NOTES_BLOCK_RE.finditer(payload):
        collected[match.group("kind")].extend(
            stripped for line in match.group("body").splitlines() if (stripped := line.strip())
        )
    transcript = _NOTES_BLOCK_RE.sub("", payload).rstrip()
    return transcript, tuple(collected["remember"]), tuple(collected["forget"])


def render_memory_notes(notes: tuple[str, ...]) -> str:
    """Renders the answer model's notes as the numbered candidate list the evaluator reviews."""
    return "\n".join(
        f"{index}. {_note_text(note=note)}" for index, note in enumerate(notes, start=1)
    )


def render_forget_requests(notes: tuple[str, ...], source: str | None) -> str:
    """Renders `<forget-memory>` notes as raw entries consolidation can act on.

    A forget is deliberately NOT a `MemoryObservation`. It is not something to store, so it needs
    no category, durability, sharing or dedupe key, and running it through the gates that decide
    whether a fact is worth keeping would only find reasons to drop it. Giving it its own
    `### forget_request` header instead keeps it invisible to every reader that walks observation
    fields: `tone_evidence_from_raw` skips it because the header is not a tone category, and
    `observation_key_sources_from_text` finds no `normalized_key` to pair it with.

    `source` is stamped for the record rather than for routing. Routing a forget by its source
    would leave it unable to reach a fact stored anywhere else, so `partition_forget_requests`
    broadcasts it to every compartment its speaker could read from instead.
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


def subject_source_line(guild_id: int | None) -> str:
    """Renders the subject's second line naming where the conversation happened."""
    return f"source: guild {guild_id}" if guild_id is not None else "source: dm"


def parse_subject_source(subject: str) -> str | None:
    """Extracts the conversation source from a persisted subject, or None when absent.

    None is the server flavor, whose subject never carries a source line: a server memory
    is one guild by construction, so there is nothing to scope its observations by and they
    render without per-observation source stamping.
    """
    match = _SUBJECT_SOURCE_RE.search(subject)
    return match.group("source") if match else None


def observation_key_sources_from_text(text: str) -> set[tuple[str, str | None]]:
    """Extracts `(normalized_key, source)` pairs from raw/detail evidence.

    The renderer emits `- source:` after `- normalized_key:` inside one block, so a
    line walk can pair each key with its block's source; entries written before
    source stamping (or by the server flavor) pair with None.
    """
    pairs: set[tuple[str, str | None]] = set()
    pending_key: str | None = None
    for line in text.splitlines():
        key_match = _STRUCTURED_KEY_RE.match(line)
        if key_match:
            if pending_key is not None:
                pairs.add((pending_key, None))
            pending_key = key_match.group("key")
            continue
        source_match = _STRUCTURED_SOURCE_RE.match(line)
        if source_match and pending_key is not None:
            pairs.add((pending_key, source_match.group("source")))
            pending_key = None
    if pending_key is not None:
        pairs.add((pending_key, None))
    return pairs


def filter_duplicate_observations(
    observations: tuple[MemoryObservation, ...], existing_text: str, source: str | None
) -> tuple[MemoryObservation, ...]:
    """Drops observations already evidenced from the SAME conversation source.

    The dedupe key is `(normalized_key, source)`, not the key alone: a fact re-stated in
    another guild (or a DM) must re-enter raw so `partition_raw_entries` can file it in
    that conversation's own compartment; key-only dedupe would lock every fact to the
    first source that ever observed it.
    """
    existing_pairs = observation_key_sources_from_text(text=existing_text)
    kept: list[MemoryObservation] = []
    for observation in observations:
        if (observation.normalized_key, source) in existing_pairs:
            continue
        kept.append(observation)
        existing_pairs.add((observation.normalized_key, source))
    return tuple(kept)


def redact_secrets(text: str) -> str:
    """Replaces token-, key-, and password-like strings with a redaction marker."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


def _validated_draft(
    draft: RawMemoryDraft, target_user_id: int | None, roster: tuple[str, ...] = ()
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
    observation: MemoryObservation, target_user_id: int | None, roster: tuple[str, ...] = ()
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
    # The roster half is what the directory boundary made necessary: with no read-time
    # filter left, `global` is permanent cross-server reach, so "他跟女友吵架" — which
    # carries no id token at all — can no longer be left entirely to the model's own
    # judgement. Matching the conversation's other participants literally is the
    # deterministic half; the phase-1.5 evaluator covers whoever is named but absent.
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
        role="system",
        content=f"[{omitted_count} non-target history message(s) omitted from memory extraction]",
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
    effort (mirrors `store.read_detail_tail`).
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
