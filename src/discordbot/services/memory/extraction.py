"""Phase-1 extraction, phase-2 consolidation, and the deterministic gates around both.

This is the LLM half of long-term memory. `MemoryExtractorAI` runs the three model calls the
pipeline makes — phase 1 (a conversation to structured observations), the optional phase-1.5
evaluator that re-reads its own candidates, and phase 2 (one compartment's raw batch to a set of
deltas) — and each of them degrades to None instead of raising, so a failed call keeps the
previous memory state. The phase prompts are instance fields, which is what lets the same engine
drive the bot's per-server memory by swapping prompts alone.

The structured schemas live here too: `MemoryObservation` / `RawMemoryDraft` are what phase 1
returns, `MemoryFactDelta` / `ConsolidatedMemory` what phase 2 returns, and `ConsolidationRequest`
is the input bundle one compartment call is given. The four output schemas are prompt surface —
their `Field(description=...)` strings are read by the model, so editing one changes behavior;
`ConsolidationRequest` is built in Python and its descriptions are only documentation.

Around the calls sit the parts that deliberately are NOT prompt rules:

* The sharing gate. `_sanitize_observation` forces `source_only` on a recent-context
  observation, on `ongoing_situation` evidence, on an id token or mention naming anyone but the
  target, and on text that literally names another participant of the same conversation (the
  roster is read back out of the transcript's own trusted author prefixes by
  `participant_names_from_transcript`). Code only ever TIGHTENS sharing; it never turns a
  `source_only` call back into `global`. Since the compartment directory became the privacy
  boundary there is no read-time filter left, so `global` means permanent cross-server reach and
  a false positive here costs one harmless fact its portability while a false negative publishes
  a private one everywhere.
* The acceptance gate. `_is_accepted_observation` drops casual mentions, hypotheticals, bot
  suggestions, observations about somebody else, and anything under the per-category confidence
  and durability floor, so raw only ever stages what survived it.
* Anti-injection. `transcript_from_messages` renders each turn as a column-0
  `[message <n> | <role>]` marker over an indented body, so nothing a user writes can forge a
  block boundary or plant an author prefix; `_truncate_middle` realigns a cut tail to the next
  marker for the same reason; `redact_secrets` scrubs secret-shaped strings off the transcript on
  the way in and off every model-authored string on the way out.

The rest is the raw-entry format: `render_memory_observations` writes one block per accepted
observation with the conversation source stamped by code rather than echoed by the model, and
`observation_keys_from_text` / `observation_key_sources_from_text` /
`filter_duplicate_observations` read it back, keyed on `(normalized_key, source)` so a fact
re-stated in another guild re-enters raw instead of being deduped away.

Nothing here touches the filesystem, takes a lock, or knows about Discord: `pipeline.py` owns the
ordering, the locking and every write, `store.py` the files, and `deltas.py` what a delta is
allowed to do to them. The other callers only build inputs or the engine —
`cogs/gen_reply/cog.py` for the target-centered message list and the subject's source line,
`cogs/memory/cog.py` for the extractor itself, and the offline `scripts/`.
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
    PHASE1_PROMPT,
    PHASE2_PROMPT,
    PHASE1_EVALUATOR_PROMPT,
    PHASE2_COMPACTION_BLOCK,
)
from discordbot.services.memory.constants import (
    MEMORY_REPLY_MAX_CHARS,
    MEMORY_TRANSCRIPT_MAX_CHARS,
    MEMORY_EXTRACT_TIMEOUT_SECONDS,
    MEMORY_COMPARTMENT_TIMEOUT_SECONDS,
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
    """One phase-1 observation, as the model emits it and as the gates hand it back.

    The same type carries both: `_sanitize_observation` returns a new instance with the text
    redacted and trimmed, the key normalized, the TTL settled and `sharing` tightened where code
    says it must be, and `_is_accepted_observation` then decides whether it reaches raw at all.
    Every description below is prompt text.
    """

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
    """Structured phase-1 extraction output for one conversation.

    Also the phase-1.5 evaluator's schema and what `_validated_draft` rebuilds afterwards, so
    `has_signal` on a draft the pipeline sees means "something survived the gates", not "the model
    thought it saw something".
    """

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


class ConsolidatedMemory(BaseModel):
    """Structured phase-2 consolidation output for one compartment.

    Only one of the two fields is ever read per call: a compartment call's `tone_markdown` is
    ignored, and the dedicated tone-note call's `deltas` are discarded by code, which is what lets
    that call be handed unpartitioned tone evidence without it becoming a stored fact.
    """

    model_config = ConfigDict(frozen=True)

    deltas: tuple[MemoryFactDelta, ...] = Field(
        default=(), description="Changes to apply; empty is a valid no-op."
    )
    tone_markdown: str = Field(
        default="",
        description=(
            "Full rewritten per-user tone note starting with `## 語氣偏好`; empty when the "
            "corpus carries no tone signal. Only the global compartment emits one."
        ),
    )


class ConsolidationRequest(BaseModel):
    """Everything one compartment's consolidation call is given.

    Bundled rather than passed as a dozen arguments because the fan-out builds these
    per compartment and the differences between them (which raw bucket, whether tone is
    wanted, which sections are legal) are exactly what the caller has to get right.
    """

    model_config = ConfigDict(frozen=True)

    compartment_note: str = Field(
        ..., description="Plain-English description of who may read this compartment."
    )
    allowed_sections: tuple[MemorySection, ...] = Field(
        ..., description="Sections a delta may name for this flavor."
    )
    existing_facts: str = Field(
        ..., description="The compartment's current facts, rendered with their ids."
    )
    existing_tone: str = Field(..., description="Current tone note; ignored unless emit_tone.")
    raw_entries: str = Field(..., description="This compartment's share of the raw batch.")
    recent_detail: str = Field(..., description="Cold evidence filtered to this compartment.")
    tone_evidence: str = Field(
        default="",
        description="Unpartitioned tone signal; the tone-note call sets it, no compartment does.",
    )
    global_reference: str = Field(
        default="",
        description="Global facts already stored, so a guild compartment does not restate them.",
    )
    today: str = Field(..., description="ISO date used for dating and aging.")
    compact: bool = Field(..., description="Whether the compartment is large enough to compact.")
    emit_tone: bool = Field(..., description="Whether this call owns the tone note.")


class MemoryExtractorAI(BaseModel):
    """Runs the two-phase memory LLM calls with best-effort fallbacks.

    The phase prompts are instance fields so the same engine can drive a
    different memory flavor (e.g. the bot's per-server memory) by swapping the
    prompts while reusing the extraction, consolidation, validation, and
    redaction logic unchanged. They default to the per-user prompts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI] = Field(
        ..., description="Async OpenAI client for the Responses API memory calls."
    )
    extract_model: ModelSettings = Field(
        ..., description="Model running the phase-1 extraction call."
    )
    consolidate_model: ModelSettings = Field(
        ..., description="Model running the phase-2 consolidation call."
    )
    evaluate_model: ModelSettings | None = Field(
        default=None, description="Optional model for the phase-1.5 evaluator review."
    )
    phase1_prompt: str = Field(
        default=PHASE1_PROMPT, description="Instructions for the phase-1 extraction call."
    )
    evaluator_prompt: str = Field(
        default=PHASE1_EVALUATOR_PROMPT,
        description="Instructions for the phase-1.5 evaluator call.",
    )
    consolidate_prompt: str = Field(
        default=PHASE2_PROMPT, description="Instructions for the phase-2 consolidation call."
    )
    compaction_block: str = Field(
        default=PHASE2_COMPACTION_BLOCK,
        description="Extra block appended to the consolidation prompt when compacting.",
    )

    async def extract(self, subject: str, transcript: str) -> RawMemoryDraft | None:
        """Runs phase 1, then the phase-1.5 evaluator when one is configured.

        Both model outputs go through `_validated_draft`, so the evaluator can only narrow what
        phase 1 proposed: it never sees an ungated observation and its own answer is gated again.
        The roster the sharing gate matches names against is read out of `transcript` here rather
        than threaded down from the reply pipeline, so a job resumed from its stored transcript
        rebuilds exactly the same one.

        Args:
            subject (str): The leading directive naming the memory's target
                (`target_user_id: <id>` or `target_server_id: <id>`, plus the optional `source:`
                line); the phase-1 prompt explains how to read it. A server subject carries no
                user id, so no roster is built for it.
            transcript (str): The rendered conversation, already redacted and capped by
                `transcript_from_messages`.

        Returns:
            The gated draft, or None when either LLM call failed — which the caller must read as
            "keep the previous state and retry later", never as "no signal".
        """
        user_text = f"{subject}\n\nConversation transcript:\n{transcript}"
        target_match = _SUBJECT_TARGET_USER_RE.search(subject)
        target_user_id = int(target_match.group("user_id")) if target_match else None
        roster = (
            participant_names_from_transcript(transcript=transcript, target_user_id=target_user_id)
            if target_user_id is not None
            else ()
        )
        draft = await self._parse(
            model=self.extract_model,
            instructions=self.phase1_prompt,
            user_text=user_text,
            text_format=RawMemoryDraft,
            timeout_seconds=MEMORY_EXTRACT_TIMEOUT_SECONDS,
            end_user_label="memory_extract",
        )
        if draft is None:
            return None
        draft = _validated_draft(draft=draft, target_user_id=target_user_id, roster=roster)
        if not draft.has_signal:
            return draft
        evaluate_model = self.evaluate_model
        if evaluate_model is None:
            return draft
        evaluated = await self._parse(
            model=evaluate_model,
            instructions=self.evaluator_prompt,
            user_text=(
                f"{subject}\n\n"
                f"Conversation transcript:\n{transcript}\n\n"
                f"Candidate observations:\n{draft.model_dump_json()}"
            ),
            text_format=RawMemoryDraft,
            timeout_seconds=MEMORY_EXTRACT_TIMEOUT_SECONDS,
            end_user_label="memory_evaluate",
        )
        if evaluated is None:
            return None
        return _validated_draft(draft=evaluated, target_user_id=target_user_id, roster=roster)

    async def consolidate(self, request: ConsolidationRequest) -> ConsolidatedMemory | None:
        """Runs phase 2 for one compartment and redacts every model-authored string.

        The compaction block is appended to the prompt only when the compartment is large enough
        to be worth deep-summarizing. `existing_tone` and `tone_evidence` are put in front of the
        model only on the call that owns the tone note; every other call is handed neither, which
        is what keeps unpartitioned tone evidence out of a call that can write facts.

        Args:
            request (ConsolidationRequest): The compartment's identity, its evidence, and the two
                flags that shape the call.

        Returns:
            The deltas plus the tone note with their prose redacted, or None when the LLM path
            failed (the caller then keeps the raw batch for the next attempt).
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
            timeout_seconds=MEMORY_COMPARTMENT_TIMEOUT_SECONDS,
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

    async def _parse(  # noqa: PLR0913 -- thin delegate mirroring the 3 phase call sites
        self,
        model: ModelSettings,
        instructions: str,
        user_text: str,
        text_format: type[_OutputT],
        timeout_seconds: float,
        end_user_label: str,
    ) -> _OutputT | None:
        """Runs one structured Responses API call, returning None on any failure.

        Delegates to the shared `parse_responses_or_none`, which owns the call surface, the
        timeout, and the degrade-to-None handling (timeout, refusal, off-schema body, a response
        the proxy reported incomplete). Every phase here maps that None onto "keep the previous
        memory state", so nothing partial is ever written.

        Args:
            model (ModelSettings): Tier this phase runs on.
            instructions (str): Developer-authority prompt for the phase.
            user_text (str): Body of the single user-role message.
            text_format (type[_OutputT]): Schema the reply is parsed into.
            timeout_seconds (float): Wall-clock budget for the call.
            end_user_label (str): Per-phase label sent as the LiteLLM end-user header; a phase
                name, never a Discord identity.

        Returns:
            The parsed instance, or None when the call failed.
        """
        return await parse_responses_or_none(
            client=self.client,
            model=model,
            instructions=instructions,
            user_text=user_text,
            end_user_id=end_user_label,
            text_format=text_format,
            timeout_seconds=timeout_seconds,
        )


def _tagged(tag: str, body: str) -> str:
    """Wraps one consolidation input block, marking an absent one explicitly.

    An empty body renders as `(empty)` rather than as nothing, so the model reads "there is none
    of this" instead of having to guess whether the block was omitted from the request.

    Args:
        tag (str): Element name the block is wrapped in.
        body (str): Block contents; whitespace-only counts as absent.

    Returns:
        The tagged block.
    """
    return f"<{tag}>\n{body.strip() or '(empty)'}\n</{tag}>"


def _redacted_delta(delta: MemoryFactDelta) -> MemoryFactDelta:
    """Scrubs secret-shaped strings out of one delta's model-authored text.

    Only `summary` and `text` are free prose, and they are the two fields that become the stored
    fact's readable body; the remaining fields are ids, keys and enum members and are left alone.

    Args:
        delta (MemoryFactDelta): The delta as the model returned it.

    Returns:
        A copy with both prose fields redacted and stripped.
    """
    return delta.model_copy(
        update={
            "summary": redact_secrets(text=delta.summary).strip(),
            "text": redact_secrets(text=delta.text).strip(),
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

    Args:
        transcript (str): The rendered transcript to read author prefixes out of.
        target_user_id (int | None): The observation's target, left out of the roster; None keeps
            everyone, since there is nobody to exclude.

    Returns:
        The sorted, deduplicated display names and usernames distinctive enough to match on.
    """
    names: set[str] = set()
    for match in _PARTICIPANT_PREFIX_RE.finditer(transcript):
        if target_user_id is not None and int(match.group("user_id")) == target_user_id:
            continue
        names.update((match.group("display").strip(), match.group("username").strip()))
    return tuple(sorted(name for name in names if _is_matchable_name(name=name)))


def _is_matchable_name(name: str) -> bool:
    """Whether a roster name is distinctive enough to lock an observation on.

    Args:
        name (str): One display name or username read off the transcript.

    Returns:
        True when the name clears the length floor for its script.
    """
    floor = _MIN_LATIN_ROSTER_NAME if _LATIN_NAME_RE.match(name) else _MIN_OTHER_ROSTER_NAME
    return len(name) >= floor


def _mentions_roster_name(text: str, roster: tuple[str, ...]) -> bool:
    """Whether the text names another participant in plain prose.

    Latin names must land on a word boundary so `amy` does not fire on `dynamic`; a CJK
    name has no boundaries to anchor to and is matched as a substring, which is the
    deliberate asymmetry — a false positive keeps a harmless fact inside one guild,
    while a false negative publishes a private one everywhere.

    Args:
        text (str): The observation's summary and evidence quote, before trimming.
        roster (tuple[str, ...]): The conversation's other participants.

    Returns:
        True when any roster name appears in the text.
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

    Pure and sub-millisecond, which is why the pipeline renders it eagerly on the reply path and
    persists the string: a resumed job then re-renders nothing and reads the same evidence.

    Args:
        message_list (list[EasyInputMessageParam]): The turn's input messages, already narrowed
            by `target_centered_memory_messages`.
        full_reply (str): The streamed reply text; its usage footer is stripped first, being
            delivery chrome rather than evidence about anyone.

    Returns:
        The transcript, redacted and capped at `MEMORY_TRANSCRIPT_MAX_CHARS`.
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
    """Narrows reply context to target-centered evidence for memory extraction.

    Only the history is thinned. The reply-reference chain and the current message are kept whole
    because they are what the turn is actually about.

    Args:
        hist_messages (list[EasyInputMessageParam]): Channel history, header message first.
        reference_messages (list[EasyInputMessageParam]): The replied-to chain, kept whole.
        current_message (list[EasyInputMessageParam]): The triggering message, kept whole.
        target_user_id (int): Whose memory is being extracted.

    Returns:
        The narrowed message list, in the order the reply pipeline built it.
    """
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
    """Renders accepted observations as the body of one raw entry.

    The `## <timestamp>` header above them belongs to `store.append_raw_entry`; this is only the
    blocks under it. `source` names the conversation the observations came from (`guild <id>` /
    `dm`), stamped deterministically here — never LLM-echoed — so `partition_raw_entries` can
    later route each block to a compartment off it and the code-stamped sharing field. None (the
    server flavor, or a legacy job with no source line) renders neither field.

    Args:
        observations (tuple[MemoryObservation, ...]): The observations that survived the gates.
        source (str | None): `guild <id>` / `dm`, or None to omit source and sharing.

    Returns:
        The blocks joined by a blank line; empty when there are no observations.
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


def subject_source_line(guild_id: int | None) -> str:
    """Renders the subject's second line naming where the conversation happened.

    The writing half of the format `parse_subject_source` reads back after the `memory_job`
    round-trip; both live beside `_SUBJECT_SOURCE_RE` so they cannot drift apart.

    Args:
        guild_id (int | None): The guild the turn happened in; None for a DM.

    Returns:
        `source: guild <id>`, or `source: dm`.
    """
    return f"source: guild {guild_id}" if guild_id is not None else "source: dm"


def parse_subject_source(subject: str) -> str | None:
    """Extracts the conversation source from a persisted subject, or None when absent.

    None covers the server flavor (its subject never carries a source line) and
    user jobs persisted before the source line existed; both render without
    per-observation source stamping.

    Args:
        subject (str): The subject as it was persisted with the job.

    Returns:
        `guild <id>` or `dm`, or None when the subject names no source.
    """
    match = _SUBJECT_SOURCE_RE.search(subject)
    return match.group("source") if match else None


def observation_keys_from_text(text: str) -> set[str]:
    """Extracts structured observation keys already present in raw/detail evidence.

    Args:
        text (str): Raw or detail markdown to scan.

    Returns:
        Every `normalized_key` the text stages.
    """
    return {match.group("key") for match in _STRUCTURED_KEY_RE.finditer(text)}


def observation_key_sources_from_text(text: str) -> set[tuple[str, str | None]]:
    """Extracts `(normalized_key, source)` pairs from raw/detail evidence.

    The renderer emits `- source:` after `- normalized_key:` inside one block, so a
    line walk can pair each key with its block's source; entries written before
    source stamping (or by the server flavor) pair with None.

    Args:
        text (str): Raw or detail markdown to scan.

    Returns:
        One `(normalized_key, source)` pair per staged observation.
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

    The dedupe key is `(normalized_key, source)`, not the key alone: a fact
    re-stated in another guild (or a DM) must re-enter raw so consolidation can
    widen that bullet's source tag; key-only dedupe would lock every fact to the
    first source that ever observed it.

    The batch is also deduped against itself, so one draft cannot stage the same key twice.

    Args:
        observations (tuple[MemoryObservation, ...]): The accepted observations of one turn.
        existing_text (str): Raw plus the detail tail, the evidence already on disk.
        source (str | None): This turn's conversation source, or None when it has none.

    Returns:
        The observations not yet evidenced from this source, in their original order.
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
    """Replaces token-, key-, and password-like strings with a redaction marker.

    Args:
        text (str): Any string on its way into a prompt or out of a model.

    Returns:
        The text with every matched secret shape replaced by `[REDACTED_SECRET]`.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


def _validated_draft(
    draft: RawMemoryDraft, target_user_id: int | None, roster: tuple[str, ...] = ()
) -> RawMemoryDraft:
    """Applies the deterministic high-precision gates to one model draft.

    Sanitizing runs before the dedupe, so two observations the model keyed differently that
    normalize to the same token collapse into one. `has_signal` is recomputed from what survived
    rather than carried over, so a draft the gates emptied reports no signal to the caller.

    Args:
        draft (RawMemoryDraft): The model's own output, from phase 1 or the evaluator.
        target_user_id (int | None): Whose memory this is; None for the server flavor.
        roster (tuple[str, ...]): The conversation's other participants, for the sharing gate.

    Returns:
        A rebuilt draft holding only the accepted observations.
    """
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
    """Whether the text references any participant other than the target user.

    Args:
        text (str): Observation text, scanned for `[id: N]` and `<@N>` tokens.
        target_user_id (int | None): The one exempt id; None makes every token count, since
            there is no target to quote.

    Returns:
        True when a token naming somebody else is present.
    """
    for match in _OTHER_PERSON_TOKEN_RE.finditer(text):
        mentioned = int(match.group("user_id") or match.group("mention_id"))
        if target_user_id is None or mentioned != target_user_id:
            return True
    return False


def _sanitize_observation(
    observation: MemoryObservation, target_user_id: int | None, roster: tuple[str, ...] = ()
) -> MemoryObservation:
    """Normalizes text, keys, TTL, and sharing fields before validation.

    Recent context is forced unpromotable, `recent`, and TTL-bound whatever the model said; every
    other category loses its TTL entirely. The sharing half is the privacy backstop and the
    reasoning for it sits at the block that applies it.

    Args:
        observation (MemoryObservation): The observation as the model wrote it.
        target_user_id (int | None): The id exempt from the other-person check; None for the
            server flavor.
        roster (tuple[str, ...]): The conversation's other participants, matched literally.

    Returns:
        A new observation with the normalized fields, tightened to `source_only` where a rule
        fired.
    """
    category = observation.category
    ttl_days = observation.ttl_days
    promotion_eligible = observation.promotion_eligible
    durability = observation.durability
    if category == "recent_context":
        promotion_eligible = False
        durability = "recent"
        ttl_days = 30 if ttl_days is None or ttl_days <= 0 else min(ttl_days, 90)
    else:
        ttl_days = None
    summary_zh = _trim_text(text=redact_secrets(text=observation.summary_zh), max_chars=800)
    evidence_quote = _trim_text(
        text=redact_secrets(text=observation.evidence_quote), max_chars=240
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
    """Returns whether an observation is precise enough to enter raw memory.

    Recent context is held to a lower bar (medium confidence plus a TTL), which the sanitizer has
    already source-locked and dated; everything else has to be a promotable, high-confidence,
    durable observation resting on one of the stable evidence kinds.

    Args:
        observation (MemoryObservation): The sanitized observation.

    Returns:
        True when it may be staged in raw.
    """
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
    """Normalizes a model-provided dedupe key into a compact safe token.

    Everything outside `[a-z0-9._:-]` collapses to a dot and the result is capped, so two
    spellings of one key still dedupe together and a key the model wrote as a sentence cannot
    grow unbounded.

    Args:
        value (str): The key as the model wrote it.

    Returns:
        The lowercase dotted token, at most 120 characters.
    """
    key = _KEY_SAFE_RE.sub(".", redact_secrets(text=value).strip().lower())
    key = re.sub(r"\.+", ".", key).strip(".")
    return key[:120]


def _trim_text(text: str, max_chars: int) -> str:
    """Collapses whitespace and caps one observation field.

    Args:
        text (str): The field value, already redacted.
        max_chars (int): Ceiling, ellipsis included.

    Returns:
        The single-line value, ellipsized when it was over the cap.
    """
    trimmed = " ".join(text.split())
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[: max_chars - 3].rstrip() + "..."


def _target_centered_history_messages(
    hist_messages: list[EasyInputMessageParam], target_user_id: int
) -> list[EasyInputMessageParam]:
    """Keeps target history plus local neighboring context.

    The first message is the history header and is always kept ahead of the window. Each target
    message brings the one either side of it, so a reply still reads with the line it answers,
    and every dropped run becomes a count marker rather than a silent join — otherwise the model
    reads two unrelated moments as one continuous exchange.

    Args:
        hist_messages (list[EasyInputMessageParam]): Channel history, header message first.
        target_user_id (int): Whose messages anchor the window.

    Returns:
        The narrowed history, or an empty list when the target never spoke in it (the header
        alone is not evidence).
    """
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
    """Returns whether the trusted author prefix names the target user.

    Anchored to the first line, where the input builder puts the prefix, so an `[id: N]` quoted
    deeper in the body cannot claim authorship.

    Args:
        message (EasyInputMessageParam): One input message.
        target_user_id (int): The id being looked for.

    Returns:
        True when the message's own prefix carries that id.
    """
    match = _AUTHOR_PREFIX_RE.match(_message_text(message=message))
    return match is not None and int(match.group("user_id")) == target_user_id


def _omission_message(omitted_count: int) -> EasyInputMessageParam:
    """Builds a neutral marker for omitted non-target history.

    Args:
        omitted_count (int): How many consecutive messages the window dropped here.

    Returns:
        A `system` message stating the size of the gap.
    """
    return EasyInputMessageParam(
        role="system",
        content=f"[{omitted_count} non-target history message(s) omitted from memory extraction]",
    )


def _indent_block(text: str) -> str:
    """Indents content lines so column-0 block markers cannot be forged in bodies.

    Args:
        text (str): One message body.

    Returns:
        The body with every line indented by two spaces.
    """
    return "\n".join(f"  {line}" for line in text.splitlines())


def _strip_forwarded_payload(text: str) -> str:
    """Drops a block's forwarded snapshot span so memory never attributes it to the forwarder.

    `get_cleaned_content` appends forwarded text last under `FORWARDED_MESSAGE_MARKER`, so the
    first marker is the suffix boundary: everything from it to end-of-body is someone else's
    words and must not become a fact about the (target) forwarder. The answer still sees the
    full body; only this memory-evidence transcript excludes it.

    Args:
        text (str): One rendered message body.

    Returns:
        The body up to the first forward marker, or unchanged when it carries none.
    """
    index = text.find(FORWARDED_MESSAGE_MARKER)
    if index == -1:
        return text
    return text[:index].rstrip()


def _message_text(message: EasyInputMessageParam) -> str:
    """Extracts the plain text from one input message, dropping non-text parts.

    A non-text part is a file or image handle whose id is opaque, so the transcript keeps only
    what was actually written.

    Args:
        message (EasyInputMessageParam): One input message, string- or parts-shaped.

    Returns:
        The message's text parts joined by newlines, stripped.
    """
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

    Args:
        text (str): The rendered transcript.
        max_chars (int): Ceiling, the truncation marker included.

    Returns:
        The text unchanged when it fits, else head + marker + boundary-aligned tail.
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
