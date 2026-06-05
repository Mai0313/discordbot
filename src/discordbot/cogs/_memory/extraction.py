"""LLM extraction and consolidation for per-user long-term memory."""

import re
from typing import TypeVar, cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.models import ModelSettings
from discordbot.cogs._memory.prompts import PHASE1_PROMPT, PHASE2_PROMPT, PHASE2_COMPACTION_BLOCK
from discordbot.cogs._gen_reply.input import USAGE_FOOTER_RE
from discordbot.cogs._memory.constants import (
    MEMORY_REPLY_MAX_CHARS,
    MEMORY_TRANSCRIPT_MAX_CHARS,
    MEMORY_EXTRACT_TIMEOUT_SECONDS,
    MEMORY_CONSOLIDATE_TIMEOUT_SECONDS,
)

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


class RawMemoryDraft(BaseModel):
    """Structured phase-1 extraction output for one conversation."""

    model_config = ConfigDict(frozen=True)

    has_signal: bool = Field(
        description="Whether the conversation contained durable memory-worthy signal about the target user"
    )
    memory_markdown: str = Field(
        description="Traditional Chinese raw memory bullets; empty when has_signal is false"
    )


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
    """Runs the two-phase memory LLM calls with best-effort fallbacks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI]
    extract_model: ModelSettings
    consolidate_model: ModelSettings

    async def extract(self, target_user_id: int, transcript: str) -> RawMemoryDraft | None:
        """Returns the phase-1 raw memory draft, or None when the LLM path fails."""
        user_text = f"target_user_id: {target_user_id}\n\nConversation transcript:\n{transcript}"
        draft = await self._parse(
            model=self.extract_model,
            instructions=PHASE1_PROMPT,
            user_text=user_text,
            text_format=RawMemoryDraft,
            timeout_seconds=MEMORY_EXTRACT_TIMEOUT_SECONDS,
            end_user_label="memory_extract",
        )
        if draft is None:
            return None
        return RawMemoryDraft(
            has_signal=draft.has_signal,
            memory_markdown=redact_secrets(text=draft.memory_markdown).strip(),
        )

    async def consolidate(
        self, existing_main: str, raw_entries: str, today: str, compact: bool
    ) -> ConsolidatedMemory | None:
        """Returns the phase-2 consolidation result, or None when the LLM path fails."""
        user_text = (
            f"today: {today}\n\n"
            f"<existing_memory>\n{existing_main.strip() or '(empty)'}\n</existing_memory>\n\n"
            f"<raw_entries>\n{raw_entries.strip()}\n</raw_entries>"
        )
        instructions = PHASE2_PROMPT + PHASE2_COMPACTION_BLOCK if compact else PHASE2_PROMPT
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


def redact_secrets(text: str) -> str:
    """Replaces token-, key-, and password-like strings with a redaction marker."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


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
