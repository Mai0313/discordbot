"""LLM extraction and consolidation for per-user long-term memory."""

import re
from typing import TYPE_CHECKING, Literal, TypeVar, cast

from openai import AsyncOpenAI
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.utils.llm import parse_responses_or_none
from discordbot.typings.models import ModelSettings
from discordbot.cogs._memory.prompts import (
    PHASE1_PROMPT,
    PHASE2_PROMPT,
    PHASE1_EVALUATOR_PROMPT,
    PHASE2_COMPACTION_BLOCK,
)
from discordbot.cogs._gen_reply.input import USAGE_FOOTER_RE, FORWARDED_MESSAGE_MARKER
from discordbot.cogs._memory.constants import (
    MEMORY_REPLY_MAX_CHARS,
    MEMORY_TRANSCRIPT_MAX_CHARS,
    MEMORY_EXTRACT_TIMEOUT_SECONDS,
    MEMORY_CONSOLIDATE_TIMEOUT_SECONDS,
)

if TYPE_CHECKING:
    from openai.types.responses.response_input_text_param import ResponseInputTextParam

_OutputT = TypeVar("_OutputT", bound=BaseModel)

type MemoryCategory = Literal[
    "stable_preference", "stable_fact", "interaction_style", "recurring_pattern", "recent_context"
]
type MemoryEvidenceKind = Literal[
    "explicit_preference",
    "repeated_behavior",
    "correction",
    "stable_fact",
    "recurring_pattern",
    "ongoing_situation",
    "tool_usage",
    "casual_mention",
    "hypothetical",
    "bot_suggestion",
    "other_user_context",
    "unknown",
]
type MemoryConfidence = Literal["low", "medium", "high"]
type MemoryDurability = Literal["volatile", "session", "recent", "stable", "permanent"]
type MemorySharing = Literal["global", "source_only"]

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
# Another participant referenced inside an observation's text (an id token or a raw
# Discord mention). Such an observation is about a relationship or someone else's
# business, so the sharing gate locks it to its source conversation.
_OTHER_PERSON_TOKEN_RE = re.compile(r"\[id:\s*\d+\]|<@!?\d+>")
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
    """Structured phase-1 extraction output for one conversation."""

    model_config = ConfigDict(frozen=True)

    has_signal: bool = Field(
        ...,
        description="Whether the conversation contained durable memory-worthy signal about the target user",
    )
    observations: tuple[MemoryObservation, ...] = Field(
        default=(),
        description="Validated structured memory observations; empty when has_signal is false",
    )


class ConsolidatedMemory(BaseModel):
    """Structured phase-2 consolidation output."""

    model_config = ConfigDict(frozen=True)

    # Kept for the prompt's no-op contract (the model emits changed=false with empty
    # memory_markdown for a no-op); the runtime consolidation path intentionally ignores
    # this bool and decides off memory_markdown's well-formedness, so do not branch on it.
    changed: bool = Field(
        ...,
        description="Whether the consolidated memory file materially changed from the existing one",
    )
    memory_markdown: str = Field(
        ...,
        description="Full rewritten memory file starting with `v1`; empty when changed is false",
    )
    tone_markdown: str = Field(
        ...,
        description=(
            "Full rewritten per-user tone note starting with `## 語氣偏好`; empty when the "
            "corpus carries no tone signal (server-flavor consolidations always return empty)."
        ),
    )


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
        """Returns the phase-1 raw memory draft, or None when the LLM path fails.

        `subject` is the leading directive naming the memory's target (e.g.
        `target_user_id: <id>` or `target_server_id: <id>`); the phase-1 prompt
        explains how to read it.
        """
        user_text = f"{subject}\n\nConversation transcript:\n{transcript}"
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
        draft = _validated_draft(draft=draft)
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
        return _validated_draft(draft=evaluated)

    async def consolidate(  # noqa: PLR0913 -- the phase-2 corpus (main/tone/raw/detail) plus dating and compaction flags
        self,
        existing_main: str,
        existing_tone: str,
        raw_entries: str,
        recent_detail: str,
        today: str,
        compact: bool,
    ) -> ConsolidatedMemory | None:
        """Returns the phase-2 consolidation result, or None when the LLM path fails."""
        user_text = (
            f"today: {today}\n\n"
            f"<existing_memory>\n{existing_main.strip() or '(empty)'}\n</existing_memory>\n\n"
            f"<existing_tone>\n{existing_tone.strip() or '(empty)'}\n</existing_tone>\n\n"
            f"<raw_entries>\n{raw_entries.strip()}\n</raw_entries>\n\n"
            f"<recent_detail>\n{recent_detail.strip() or '(empty)'}\n</recent_detail>"
        )
        instructions = (
            self.consolidate_prompt + self.compaction_block if compact else self.consolidate_prompt
        )
        result = await self._parse(
            model=self.consolidate_model,
            instructions=instructions,
            user_text=user_text,
            text_format=ConsolidatedMemory,
            timeout_seconds=MEMORY_CONSOLIDATE_TIMEOUT_SECONDS,
            end_user_label="memory_consolidate",
        )
        if result is None:
            return None
        return result.model_copy(
            update={
                "memory_markdown": redact_secrets(text=result.memory_markdown).strip(),
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

        Delegates to the shared `parse_responses_or_none`, which owns the call surface,
        the timeout, and the degrade-to-None handling (timeout, refused output, an
        incomplete/truncated response — the last matters here because a model that closed
        the JSON early could otherwise pass the `v1` header check with an amputated file).
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
    """Narrows reply context to target-centered evidence for memory extraction."""
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
    can scope each bullet. None (the server flavor, or a legacy job with no
    source line) renders neither the source nor the sharing field.
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
    """Renders the subject's second line naming where the conversation happened."""
    return f"source: guild {guild_id}" if guild_id is not None else "source: dm"


def parse_subject_source(subject: str) -> str | None:
    """Extracts the conversation source from a persisted subject, or None when absent.

    None covers the server flavor (its subject never carries a source line) and
    user jobs persisted before the source line existed; both render without
    per-observation source stamping.
    """
    match = _SUBJECT_SOURCE_RE.search(subject)
    return match.group("source") if match else None


def observation_keys_from_text(text: str) -> set[str]:
    """Extracts structured observation keys already present in raw/detail evidence."""
    return {match.group("key") for match in _STRUCTURED_KEY_RE.finditer(text)}


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

    The dedupe key is `(normalized_key, source)`, not the key alone: a fact
    re-stated in another guild (or a DM) must re-enter raw so consolidation can
    widen that bullet's source tag; key-only dedupe would lock every fact to the
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


def _validated_draft(draft: RawMemoryDraft) -> RawMemoryDraft:
    """Applies deterministic high-precision gates to model observations."""
    observations: list[MemoryObservation] = []
    seen_keys: set[str] = set()
    for observation in draft.observations:
        sanitized = _sanitize_observation(observation=observation)
        if sanitized.normalized_key in seen_keys:
            continue
        if not _is_accepted_observation(observation=sanitized):
            continue
        observations.append(sanitized)
        seen_keys.add(sanitized.normalized_key)
    return RawMemoryDraft(has_signal=bool(observations), observations=tuple(observations))


def _sanitize_observation(observation: MemoryObservation) -> MemoryObservation:
    """Normalizes text, keys, TTL, and sharing fields before validation."""
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
    # are private by construction, and an observation naming another participant is
    # about a relationship, not a portable fact. Code only ever tightens sharing to
    # source_only; it never loosens a source_only call back to global.
    sharing = observation.sharing
    if category == "recent_context" or _OTHER_PERSON_TOKEN_RE.search(
        f"{summary_zh}\n{evidence_quote}"
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
