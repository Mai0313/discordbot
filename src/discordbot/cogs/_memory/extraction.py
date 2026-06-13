"""LLM extraction and consolidation for per-user long-term memory."""

import re
from typing import Literal, TypeVar, cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.models import ModelSettings
from discordbot.cogs._memory.prompts import (
    PHASE1_PROMPT,
    PHASE2_PROMPT,
    PHASE1_EVALUATOR_PROMPT,
    PHASE2_COMPACTION_BLOCK,
)
from discordbot.cogs._gen_reply.input import USAGE_FOOTER_RE
from discordbot.cogs._memory.constants import (
    MEMORY_REPLY_MAX_CHARS,
    MEMORY_TRANSCRIPT_MAX_CHARS,
    MEMORY_EXTRACT_TIMEOUT_SECONDS,
    MEMORY_CONSOLIDATE_TIMEOUT_SECONDS,
)

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
type MemoryDurability = Literal["volatile", "session", "recent", "stable"]

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
_KEY_SAFE_RE = re.compile(r"[^a-z0-9._:-]+")
_STRUCTURED_KEY_RE = re.compile(r"^\s*-\s*normalized_key:\s*(?P<key>\S+)\s*$", flags=re.MULTILINE)
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
        description="The memory section this observation belongs to.",
        examples=["stable_preference", "recent_context"],
    )
    subject_is_target_user: bool = Field(
        description="Whether the evidence is about the target user, not another participant."
    )
    evidence_kind: MemoryEvidenceKind = Field(
        description="The evidence shape supporting or rejecting this observation.",
        examples=["explicit_preference", "casual_mention"],
    )
    confidence: MemoryConfidence = Field(
        description="Confidence after attribution and durability checks.", examples=["high"]
    )
    durability: MemoryDurability = Field(
        description="How long the observation should influence memory.",
        examples=["stable", "recent"],
    )
    promotion_eligible: bool = Field(
        description="Whether this may be promoted into stable memory during consolidation."
    )
    normalized_key: str = Field(
        description="Stable dedupe key for the same underlying observation.",
        examples=["preference.reply_language.zh_tw"],
    )
    summary_zh: str = Field(description="Traditional Chinese memory delta.")
    evidence_quote: str = Field(description="Short evidence quote from the target user.")
    ttl_days: int | None = Field(
        default=None,
        description="Positive TTL for recent context; null for stable observations.",
        examples=[30],
    )


class RawMemoryDraft(BaseModel):
    """Structured phase-1 extraction output for one conversation."""

    model_config = ConfigDict(frozen=True)

    has_signal: bool = Field(
        description="Whether the conversation contained durable memory-worthy signal about the target user"
    )
    observations: tuple[MemoryObservation, ...] = Field(
        default=(),
        description="Validated structured memory observations; empty when has_signal is false",
    )

    @property
    def memory_markdown(self) -> str:
        """Renders accepted observations into the raw markdown evidence format."""
        return render_memory_observations(observations=self.observations)


class ConsolidatedMemory(BaseModel):
    """Structured phase-2 consolidation output."""

    model_config = ConfigDict(frozen=True)

    changed: bool = Field(
        description="Whether the consolidated memory file materially changed from the existing one"
    )
    memory_markdown: str = Field(
        description="Full rewritten memory file starting with `v1`; empty when changed is false"
    )


class MemoryExtractorAI(BaseModel):
    """Runs the two-phase memory LLM calls with best-effort fallbacks.

    The phase prompts are instance fields so the same engine can drive a
    different memory flavor (e.g. the bot's per-server memory) by swapping the
    prompts while reusing the extraction, consolidation, validation, and
    redaction logic unchanged. They default to the per-user prompts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI]
    extract_model: ModelSettings
    consolidate_model: ModelSettings
    evaluate_model: ModelSettings | None = None
    phase1_prompt: str = PHASE1_PROMPT
    evaluator_prompt: str = PHASE1_EVALUATOR_PROMPT
    consolidate_prompt: str = PHASE2_PROMPT
    compaction_block: str = PHASE2_COMPACTION_BLOCK

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

    async def consolidate(
        self, existing_main: str, raw_entries: str, recent_detail: str, today: str, compact: bool
    ) -> ConsolidatedMemory | None:
        """Returns the phase-2 consolidation result, or None when the LLM path fails."""
        user_text = (
            f"today: {today}\n\n"
            f"<existing_memory>\n{existing_main.strip() or '(empty)'}\n</existing_memory>\n\n"
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
        return ConsolidatedMemory(
            changed=result.changed,
            memory_markdown=redact_secrets(text=result.memory_markdown).strip(),
        )

    async def _parse(  # noqa: PLR0913 -- one shared Responses call surface for both phases
        self,
        model: ModelSettings,
        instructions: str,
        user_text: str,
        text_format: type[_OutputT],
        timeout_seconds: float,
        end_user_label: str,
    ) -> _OutputT | None:
        """Runs one structured Responses API call, returning None on any failure."""
        try:
            async with asyncio.timeout(delay=timeout_seconds):
                responses = await self.client.responses.parse(
                    model=model.name,
                    instructions=instructions,
                    input=cast(
                        "ResponseInputParam",
                        [EasyInputMessageParam(role="user", content=user_text)],
                    ),
                    text_format=text_format,
                    reasoning=model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": end_user_label},
                    extra_body={"mock_testing_fallbacks": False},
                )
        except TimeoutError:
            logfire.warn(
                "Memory LLM request timed out; skipping update",
                timeout_seconds=timeout_seconds,
                end_user_label=end_user_label,
            )
            return None
        except ValidationError:
            # The model returned no text output (e.g. safety filter); parse raises
            # before output_parsed can be inspected, same as RouteDecision handling.
            logfire.warn("Memory LLM parse failed; skipping update", end_user_label=end_user_label)
            return None
        except Exception:
            logfire.warn(
                "Memory LLM request failed; skipping update",
                end_user_label=end_user_label,
                _exc_info=True,
            )
            return None
        if responses.status == "incomplete":
            # The response hit the answer model's own output-token ceiling (the
            # memory calls set no explicit lower cap). A truncated JSON body
            # already fails parsing above, but a model that closed the JSON
            # early can still pass the `v1` header check downstream with a
            # silently amputated memory file; refuse it so raw entries are
            # kept for retry instead.
            logfire.warn(
                "Memory LLM response incomplete; skipping update",
                end_user_label=end_user_label,
                incomplete_details=str(responses.incomplete_details),
            )
            return None
        return responses.output_parsed


def transcript_from_messages(message_list: list[EasyInputMessageParam], full_reply: str) -> str:
    """Renders the reply-pipeline input messages plus the streamed reply as plain text.

    Each message becomes a block whose `[message <n> | <role>]` marker sits at
    column 0 while every content line is indented, so user-authored text can
    never forge a block boundary or plant an author prefix at content start.
    """
    blocks: list[str] = []
    for message in message_list:
        text = _message_text(message=message)
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


def render_memory_observations(observations: tuple[MemoryObservation, ...]) -> str:
    """Renders structured observations as timestamp-entry body markdown."""
    blocks: list[str] = []
    for observation in observations:
        ttl_text = "null" if observation.ttl_days is None else str(observation.ttl_days)
        blocks.append(
            "\n".join((
                f"### {observation.category}",
                f"- normalized_key: {observation.normalized_key}",
                f"- evidence_kind: {observation.evidence_kind}",
                f"- confidence: {observation.confidence}",
                f"- durability: {observation.durability}",
                f"- promotion_eligible: {str(observation.promotion_eligible).lower()}",
                f"- ttl_days: {ttl_text}",
                f"- summary_zh: {observation.summary_zh}",
                f"- evidence_quote: {observation.evidence_quote}",
            ))
        )
    return "\n\n".join(blocks)


def observation_keys_from_text(text: str) -> set[str]:
    """Extracts structured observation keys already present in raw/detail evidence."""
    return {match.group("key") for match in _STRUCTURED_KEY_RE.finditer(text)}


def filter_duplicate_observations(
    observations: tuple[MemoryObservation, ...], existing_text: str
) -> tuple[MemoryObservation, ...]:
    """Drops observations whose normalized key already exists in evidence."""
    existing_keys = observation_keys_from_text(text=existing_text)
    kept: list[MemoryObservation] = []
    for observation in observations:
        if observation.normalized_key in existing_keys:
            continue
        kept.append(observation)
        existing_keys.add(observation.normalized_key)
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
    """Normalizes text, keys, and TTL fields before validation."""
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
    return MemoryObservation(
        category=category,
        subject_is_target_user=observation.subject_is_target_user,
        evidence_kind=observation.evidence_kind,
        confidence=observation.confidence,
        durability=durability,
        promotion_eligible=promotion_eligible,
        normalized_key=_clean_normalized_key(value=observation.normalized_key),
        summary_zh=_trim_text(text=redact_secrets(text=observation.summary_zh), max_chars=800),
        evidence_quote=_trim_text(
            text=redact_secrets(text=observation.evidence_quote), max_chars=240
        ),
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
        and observation.durability == "stable"
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


def _message_text(message: EasyInputMessageParam) -> str:
    """Extracts the plain text from one input message, dropping non-text parts."""
    content = message["content"]
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for part in content:
        part_dict = cast("dict[str, object]", part)
        if part_dict.get("type") != "input_text":
            continue
        text_value = part_dict.get("text")
        if isinstance(text_value, str):
            parts.append(text_value)
    return "\n".join(parts).strip()


def _truncate_middle(text: str, max_chars: int) -> str:
    """Keeps the head and tail of an oversized transcript, dropping the middle."""
    if len(text) <= max_chars:
        return text
    marker = "\n\n[... transcript truncated ...]\n\n"
    budget = max_chars - len(marker)
    head = budget * 2 // 3
    tail = budget - head
    return f"{text[:head]}{marker}{text[len(text) - tail :]}"
