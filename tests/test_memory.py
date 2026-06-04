"""Tests for the per-user long-term memory helpers."""

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
import asyncio
from pathlib import Path

import pytest
from nextcord import Embed, Locale
from pydantic import BaseModel, ValidationError
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.cogs.memory import MemoryCogs
from discordbot.cogs._memory import store, pipeline
from discordbot.typings.models import ModelSettings
from discordbot.cogs._memory.constants import MEMORY_INJECTION_MAX_CHARS
from discordbot.cogs._memory.extraction import (
    RawMemoryDraft,
    MemoryExtractorAI,
    ConsolidatedMemory,
    redact_secrets,
    transcript_from_messages,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from nextcord import Interaction
    from nextcord.ext import commands

USER_ID = 123456789

TEST_MEMORY_MODEL = ModelSettings(name="test-memories-model", effort="none")


class FakeMemoryResponses:
    """Fake Responses API resource recording parse calls for memory tests."""

    def __init__(self) -> None:
        """Initializes recorded calls and the configured parsed output."""
        self.parse_models: list[str] = []
        self.parse_instructions: list[str] = []
        self.parse_inputs: list[list[dict[str, str]]] = []
        self.output_parsed: BaseModel | None = None
        self.raises: Exception | None = None

    async def parse(  # noqa: PLR0913 -- mirrors Responses API parse signature
        self,
        model: str,
        instructions: str,
        input: list[dict[str, str]],  # noqa: A002 -- SDK parameter
        text_format: type[BaseModel],
        reasoning: dict[str, str],
        service_tier: str,
        extra_headers: dict[str, str],
        extra_body: dict[str, bool],
    ) -> SimpleNamespace:
        """Records the call and returns or raises the configured result."""
        del text_format, reasoning, service_tier, extra_headers, extra_body
        self.parse_models.append(model)
        self.parse_instructions.append(instructions)
        self.parse_inputs.append(input)
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(output_parsed=self.output_parsed)


class FakeMemoryClient:
    """Fake OpenAI client exposing only the responses resource."""

    def __init__(self) -> None:
        """Initializes the fake responses resource."""
        self.responses = FakeMemoryResponses()


def _extractor() -> tuple[MemoryExtractorAI, FakeMemoryClient]:
    """Builds a MemoryExtractorAI bound to a fake client."""
    fake_client = FakeMemoryClient()
    extractor = MemoryExtractorAI(client=cast("AsyncOpenAI", fake_client), model=TEST_MEMORY_MODEL)
    return extractor, fake_client


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


def test_read_main_memory_missing_file_returns_empty(memory_isolated_dir: Path) -> None:
    assert store.read_main_memory(user_id=USER_ID) == ""
    assert store.read_main_memory_full(user_id=USER_ID) == ""


def test_write_main_memory_roundtrip_and_atomic(memory_isolated_dir: Path) -> None:
    store.write_main_memory(user_id=USER_ID, content="v1\n\n## 使用者輪廓\n測試內容\n")
    assert store.read_main_memory(user_id=USER_ID) == "v1\n\n## 使用者輪廓\n測試內容"
    leftovers = list(memory_isolated_dir.glob("*.tmp"))
    assert leftovers == []


def test_read_main_memory_truncates_to_injection_limit(memory_isolated_dir: Path) -> None:
    store.write_main_memory(user_id=USER_ID, content="x" * (MEMORY_INJECTION_MAX_CHARS + 500))
    assert len(store.read_main_memory(user_id=USER_ID)) == MEMORY_INJECTION_MAX_CHARS
    assert len(store.read_main_memory_full(user_id=USER_ID)) == MEMORY_INJECTION_MAX_CHARS + 500


def test_append_raw_entry_creates_timestamped_entries(memory_isolated_dir: Path) -> None:
    store.append_raw_entry(user_id=USER_ID, entry_text="偏好訊號:\n- 喜歡簡短回覆")
    store.append_raw_entry(user_id=USER_ID, entry_text="穩定事實:\n- 慣用繁體中文")
    assert store.count_raw_entries(user_id=USER_ID) == 2
    raw_text = store.read_raw_entries(user_id=USER_ID)
    assert raw_text.startswith("## ")
    assert "喜歡簡短回覆" in raw_text
    assert "慣用繁體中文" in raw_text


def test_append_raw_entry_evicts_oldest_on_overflow(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("discordbot.cogs._memory.store.RAW_FILE_MAX_BYTES", 220)
    store.append_raw_entry(user_id=USER_ID, entry_text="first entry " + "a" * 100)
    store.append_raw_entry(user_id=USER_ID, entry_text="second entry " + "b" * 100)
    raw_text = store.read_raw_entries(user_id=USER_ID)
    assert "first entry" not in raw_text
    assert "second entry" in raw_text
    assert store.count_raw_entries(user_id=USER_ID) == 1


def test_append_raw_entry_keeps_single_oversized_entry(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("discordbot.cogs._memory.store.RAW_FILE_MAX_BYTES", 50)
    store.append_raw_entry(user_id=USER_ID, entry_text="oversized " + "c" * 200)
    assert store.count_raw_entries(user_id=USER_ID) == 1


def test_raw_file_bytes_missing_file_is_zero(memory_isolated_dir: Path) -> None:
    assert store.raw_file_bytes(user_id=USER_ID) == 0
    store.append_raw_entry(user_id=USER_ID, entry_text="something")
    assert store.raw_file_bytes(user_id=USER_ID) > 0


def test_clear_raw_removes_only_raw_file(memory_isolated_dir: Path) -> None:
    store.write_main_memory(user_id=USER_ID, content="v1\n\nmain")
    store.append_raw_entry(user_id=USER_ID, entry_text="raw entry")
    store.clear_raw(user_id=USER_ID)
    assert store.count_raw_entries(user_id=USER_ID) == 0
    assert store.read_main_memory(user_id=USER_ID) != ""


def test_clear_user_memory_removes_both_files(memory_isolated_dir: Path) -> None:
    store.write_main_memory(user_id=USER_ID, content="v1\n\nmain")
    store.append_raw_entry(user_id=USER_ID, entry_text="raw entry")
    assert store.clear_user_memory(user_id=USER_ID) is True
    assert store.read_main_memory(user_id=USER_ID) == ""
    assert store.count_raw_entries(user_id=USER_ID) == 0
    assert store.clear_user_memory(user_id=USER_ID) is False


def test_clear_user_memory_flags_in_flight_updates(memory_isolated_dir: Path) -> None:
    started_at = time.monotonic()
    assert store.cleared_since(user_id=USER_ID, started_at=started_at) is False
    store.clear_user_memory(user_id=USER_ID)
    assert store.cleared_since(user_id=USER_ID, started_at=started_at) is True
    later = time.monotonic()
    assert store.cleared_since(user_id=USER_ID, started_at=later) is False


async def test_user_lock_is_stable_per_user(memory_isolated_dir: Path) -> None:
    lock_a = store.user_lock(user_id=USER_ID)
    lock_b = store.user_lock(user_id=USER_ID)
    lock_other = store.user_lock(user_id=USER_ID + 1)
    assert lock_a is lock_b
    assert lock_a is not lock_other


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


async def test_extract_returns_redacted_draft() -> None:
    extractor, fake_client = _extractor()
    fake_client.responses.output_parsed = RawMemoryDraft(
        has_signal=True, memory_markdown="偏好訊號:\n- 提到 token sk-aaaabbbbccccddddeeee 的事"
    )
    draft = await extractor.extract(target_user_id=USER_ID, transcript="some transcript")
    assert draft is not None
    assert draft.has_signal is True
    assert "sk-aaaabbbbccccddddeeee" not in draft.memory_markdown
    assert "[REDACTED_SECRET]" in draft.memory_markdown
    assert fake_client.responses.parse_models == [TEST_MEMORY_MODEL.name]
    user_text = fake_client.responses.parse_inputs[0][0]["content"]
    assert f"target_user_id: {USER_ID}" in user_text


async def test_extract_no_signal_passthrough() -> None:
    extractor, fake_client = _extractor()
    fake_client.responses.output_parsed = RawMemoryDraft(has_signal=False, memory_markdown="")
    draft = await extractor.extract(target_user_id=USER_ID, transcript="hi")
    assert draft is not None
    assert draft.has_signal is False
    assert draft.memory_markdown == ""


async def test_extract_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("discordbot.cogs._memory.extraction.MEMORY_EXTRACT_TIMEOUT_SECONDS", 0.01)
    extractor, fake_client = _extractor()

    async def hang(**kwargs: object) -> SimpleNamespace:
        await asyncio.sleep(10)
        return SimpleNamespace(output_parsed=None)

    monkeypatch.setattr(fake_client.responses, "parse", hang)
    assert await extractor.extract(target_user_id=USER_ID, transcript="hi") is None


async def test_extract_returns_none_on_validation_error() -> None:
    extractor, fake_client = _extractor()
    try:
        RawMemoryDraft.model_validate({})
    except ValidationError as exc:
        fake_client.responses.raises = exc
    assert await extractor.extract(target_user_id=USER_ID, transcript="hi") is None


async def test_extract_returns_none_on_generic_failure() -> None:
    extractor, fake_client = _extractor()
    fake_client.responses.raises = RuntimeError("boom")
    assert await extractor.extract(target_user_id=USER_ID, transcript="hi") is None


async def test_extract_returns_none_on_empty_parse() -> None:
    extractor, fake_client = _extractor()
    fake_client.responses.output_parsed = None
    assert await extractor.extract(target_user_id=USER_ID, transcript="hi") is None


async def test_consolidate_marks_empty_existing_memory() -> None:
    extractor, fake_client = _extractor()
    fake_client.responses.output_parsed = ConsolidatedMemory(
        changed=True, memory_markdown="v1\n\n## 使用者輪廓\n新檔案"
    )
    result = await extractor.consolidate(existing_main="", raw_entries="## 2026-01-01T00:00:00\nx")
    assert result is not None
    assert result.changed is True
    assert result.memory_markdown.startswith("v1")
    user_text = fake_client.responses.parse_inputs[0][0]["content"]
    assert "(empty)" in user_text


async def test_consolidate_unchanged_result_passthrough() -> None:
    extractor, fake_client = _extractor()
    fake_client.responses.output_parsed = ConsolidatedMemory(changed=False, memory_markdown="")
    result = await extractor.consolidate(existing_main="v1\n\nold", raw_entries="## t\nx")
    assert result is not None
    assert result.changed is False


def test_redact_secrets_masks_token_shapes() -> None:
    text = (
        "my key is sk-abcdefghijklmnop123 and AIzaSyA1234567890abcdefghijklmnopqrstu "
        "plus Bearer abcdefghijklmnopqrstuvwxyz and xoxb-1234567890-abcdefghij "
        "and ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    )
    redacted = redact_secrets(text=text)
    assert "sk-abcdefghijklmnop123" not in redacted
    assert "AIzaSyA1234567890abcdefghijklmnopqrstu" not in redacted
    assert "xoxb-1234567890-abcdefghij" not in redacted
    assert "ghp_abcdefghijklmnopqrstuvwxyz1234567890" not in redacted
    assert redacted.count("[REDACTED_SECRET]") >= 4


def test_transcript_from_messages_drops_non_text_parts() -> None:
    message_list = [
        EasyInputMessageParam(
            role="system", content=[{"type": "input_text", "text": "==== separator ===="}]
        ),
        EasyInputMessageParam(role="user", content="Alice (alice) [id: 1]: 哈囉"),
        EasyInputMessageParam(role="assistant", content="舊回覆"),
        EasyInputMessageParam(
            role="user",
            content=[
                {"type": "input_text", "text": "Bob (bob) [id: 2]: 看圖"},
                {
                    "type": "input_image",
                    "image_url": "data:image/jpeg;base64,xxx",
                    "detail": "auto",
                },
            ],
        ),
    ]
    transcript = transcript_from_messages(
        message_list=message_list, full_reply="新回覆\n\n-# model · ⬆ 1 ⬇ 2 · $0.00000001 · +1"
    )
    assert "==== separator ====" in transcript
    assert "Alice (alice) [id: 1]: 哈囉" in transcript
    assert "Assistant: 舊回覆" in transcript
    assert "Bob (bob) [id: 2]: 看圖" in transcript
    assert "data:image/jpeg" not in transcript
    assert "Assistant reply (this turn): 新回覆" in transcript
    assert "⬆" not in transcript


def test_transcript_from_messages_truncates_middle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("discordbot.cogs._memory.extraction.MEMORY_TRANSCRIPT_MAX_CHARS", 200)
    message_list = [
        EasyInputMessageParam(role="user", content=f"user message {index} " + "x" * 50)
        for index in range(20)
    ]
    transcript = transcript_from_messages(message_list=message_list, full_reply="tail reply")
    assert len(transcript) <= 200
    assert "[... transcript truncated ...]" in transcript
    assert transcript.endswith("tail reply")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def _user_message() -> list[EasyInputMessageParam]:
    """Builds a minimal message list for pipeline tests."""
    return [EasyInputMessageParam(role="user", content=f"Alice (alice) [id: {USER_ID}]: 哈囉")]


async def _wait_for_inflight() -> None:
    """Awaits the scheduled background memory task for the test user."""
    task = pipeline._inflight_tasks.get(USER_ID)
    if task is not None:
        await task


async def test_pipeline_appends_raw_entry_on_signal(memory_isolated_dir: Path) -> None:
    extractor, fake_client = _extractor()
    fake_client.responses.output_parsed = RawMemoryDraft(
        has_signal=True, memory_markdown="偏好訊號:\n- 喜歡簡短"
    )
    pipeline.schedule_memory_update(
        user_id=USER_ID, message_list=_user_message(), full_reply="回覆", extractor=extractor
    )
    await _wait_for_inflight()
    assert store.count_raw_entries(user_id=USER_ID) == 1
    assert store.read_main_memory(user_id=USER_ID) == ""


async def test_pipeline_no_op_gate_writes_nothing(memory_isolated_dir: Path) -> None:
    extractor, fake_client = _extractor()
    fake_client.responses.output_parsed = RawMemoryDraft(has_signal=False, memory_markdown="")
    pipeline.schedule_memory_update(
        user_id=USER_ID, message_list=_user_message(), full_reply="回覆", extractor=extractor
    )
    await _wait_for_inflight()
    assert store.count_raw_entries(user_id=USER_ID) == 0
    assert store.raw_file_bytes(user_id=USER_ID) == 0


async def test_pipeline_skips_when_update_already_in_flight(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor, fake_client = _extractor()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_parse(**kwargs: object) -> SimpleNamespace:
        started.set()
        await release.wait()
        return SimpleNamespace(output_parsed=RawMemoryDraft(has_signal=True, memory_markdown="x"))

    monkeypatch.setattr(fake_client.responses, "parse", slow_parse)
    pipeline.schedule_memory_update(
        user_id=USER_ID, message_list=_user_message(), full_reply="第一", extractor=extractor
    )
    await started.wait()
    first_task = pipeline._inflight_tasks[USER_ID]
    pipeline.schedule_memory_update(
        user_id=USER_ID, message_list=_user_message(), full_reply="第二", extractor=extractor
    )
    assert pipeline._inflight_tasks[USER_ID] is first_task
    release.set()
    await first_task
    assert store.count_raw_entries(user_id=USER_ID) == 1


async def test_pipeline_consolidates_at_threshold(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("discordbot.cogs._memory.pipeline.RAW_CONSOLIDATION_THRESHOLD", 2)
    extractor, fake_client = _extractor()
    fake_client.responses.output_parsed = RawMemoryDraft(
        has_signal=True, memory_markdown="偏好訊號:\n- 第一筆"
    )
    pipeline.schedule_memory_update(
        user_id=USER_ID, message_list=_user_message(), full_reply="回覆一", extractor=extractor
    )
    await _wait_for_inflight()
    assert store.count_raw_entries(user_id=USER_ID) == 1

    parsed_outputs = [
        RawMemoryDraft(has_signal=True, memory_markdown="偏好訊號:\n- 第二筆"),
        ConsolidatedMemory(changed=True, memory_markdown="v1\n\n## 使用者輪廓\n合併後"),
    ]

    async def staged_parse(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(output_parsed=parsed_outputs.pop(0))

    monkeypatch.setattr(fake_client.responses, "parse", staged_parse)
    pipeline.schedule_memory_update(
        user_id=USER_ID, message_list=_user_message(), full_reply="回覆二", extractor=extractor
    )
    await _wait_for_inflight()
    assert store.read_main_memory(user_id=USER_ID).startswith("v1")
    assert "合併後" in store.read_main_memory(user_id=USER_ID)
    assert store.count_raw_entries(user_id=USER_ID) == 0


async def test_pipeline_keeps_raw_when_consolidation_fails(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("discordbot.cogs._memory.pipeline.RAW_CONSOLIDATION_THRESHOLD", 1)
    extractor, fake_client = _extractor()

    parse_results: list[SimpleNamespace | None] = [
        SimpleNamespace(
            output_parsed=RawMemoryDraft(has_signal=True, memory_markdown="偏好訊號:\n- 訊號")
        ),
        None,
    ]

    async def staged_parse(**kwargs: object) -> SimpleNamespace:
        result = parse_results.pop(0)
        if result is None:
            raise RuntimeError("consolidation down")
        return result

    monkeypatch.setattr(fake_client.responses, "parse", staged_parse)
    pipeline.schedule_memory_update(
        user_id=USER_ID, message_list=_user_message(), full_reply="回覆", extractor=extractor
    )
    await _wait_for_inflight()
    assert store.count_raw_entries(user_id=USER_ID) == 1
    assert store.read_main_memory(user_id=USER_ID) == ""


async def test_pipeline_unchanged_consolidation_still_clears_raw(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("discordbot.cogs._memory.pipeline.RAW_CONSOLIDATION_THRESHOLD", 1)
    store.write_main_memory(user_id=USER_ID, content="v1\n\n## 使用者輪廓\n既有內容")
    extractor, fake_client = _extractor()

    parsed_outputs: list[BaseModel] = [
        RawMemoryDraft(has_signal=True, memory_markdown="偏好訊號:\n- 已知資訊"),
        ConsolidatedMemory(changed=False, memory_markdown=""),
    ]

    async def staged_parse(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(output_parsed=parsed_outputs.pop(0))

    monkeypatch.setattr(fake_client.responses, "parse", staged_parse)
    pipeline.schedule_memory_update(
        user_id=USER_ID, message_list=_user_message(), full_reply="回覆", extractor=extractor
    )
    await _wait_for_inflight()
    assert "既有內容" in store.read_main_memory(user_id=USER_ID)
    assert store.count_raw_entries(user_id=USER_ID) == 0


async def test_pipeline_aborts_write_after_clear(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor, fake_client = _extractor()
    parse_started = asyncio.Event()
    release = asyncio.Event()

    async def slow_parse(**kwargs: object) -> SimpleNamespace:
        parse_started.set()
        await release.wait()
        return SimpleNamespace(
            output_parsed=RawMemoryDraft(has_signal=True, memory_markdown="不該被寫入")
        )

    monkeypatch.setattr(fake_client.responses, "parse", slow_parse)
    pipeline.schedule_memory_update(
        user_id=USER_ID, message_list=_user_message(), full_reply="回覆", extractor=extractor
    )
    await parse_started.wait()
    store.mark_cleared(user_id=USER_ID)
    release.set()
    await _wait_for_inflight()
    assert store.count_raw_entries(user_id=USER_ID) == 0


async def test_pipeline_background_failure_is_swallowed(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor, fake_client = _extractor()

    async def exploding_parse(**kwargs: object) -> SimpleNamespace:
        raise MemoryError("unexpected")

    monkeypatch.setattr(fake_client.responses, "parse", exploding_parse)
    pipeline.schedule_memory_update(
        user_id=USER_ID, message_list=_user_message(), full_reply="回覆", extractor=extractor
    )
    task = pipeline._inflight_tasks.get(USER_ID)
    assert task is not None
    await asyncio.wait([task])
    assert pipeline._inflight_tasks.get(USER_ID) is None
    assert store.count_raw_entries(user_id=USER_ID) == 0


# ---------------------------------------------------------------------------
# /memory cog
# ---------------------------------------------------------------------------


class ResponseStub:
    """Records the initial interaction response payload."""

    def __init__(self) -> None:
        """Initializes the recorded payload."""
        self.sent: dict[str, object] = {}

    async def send_message(self, **kwargs: object) -> None:
        """Records the response payload."""
        self.sent = kwargs


def _interaction(user_id: int = USER_ID) -> SimpleNamespace:
    """Builds a minimal interaction stub for the memory cog."""
    return SimpleNamespace(user=SimpleNamespace(id=user_id), response=ResponseStub())


def _memory_cog() -> MemoryCogs:
    """Builds a MemoryCogs instance with a stub bot."""
    return MemoryCogs(bot=cast("commands.Bot", SimpleNamespace()))


async def test_memory_show_displays_stored_memory(memory_isolated_dir: Path) -> None:
    store.write_main_memory(user_id=USER_ID, content="v1\n\n## 使用者輪廓\n愛開玩笑")
    cog = _memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, cast("Interaction", interaction))
    assert interaction.response.sent["ephemeral"] is True
    embed = interaction.response.sent["embed"]
    assert isinstance(embed, Embed)
    assert "愛開玩笑" in (embed.description or "")


async def test_memory_show_handles_empty_memory(memory_isolated_dir: Path) -> None:
    cog = _memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_show.callback(cog, cast("Interaction", interaction))
    assert interaction.response.sent["ephemeral"] is True
    embed = interaction.response.sent["embed"]
    assert isinstance(embed, Embed)
    assert "還沒有任何記憶" in (embed.description or "")


async def test_memory_clear_removes_files_and_reports(memory_isolated_dir: Path) -> None:
    store.write_main_memory(user_id=USER_ID, content="v1\n\nmain")
    store.append_raw_entry(user_id=USER_ID, entry_text="raw")
    cog = _memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_clear.callback(cog, cast("Interaction", interaction))
    embed = interaction.response.sent["embed"]
    assert isinstance(embed, Embed)
    assert "已清除" in (embed.description or "")
    assert store.read_main_memory(user_id=USER_ID) == ""
    assert store.count_raw_entries(user_id=USER_ID) == 0
    started_at = 0.0
    assert store.cleared_since(user_id=USER_ID, started_at=started_at) is True


async def test_memory_clear_without_memory_reports_noop(memory_isolated_dir: Path) -> None:
    cog = _memory_cog()
    interaction = _interaction()
    await MemoryCogs.memory_clear.callback(cog, cast("Interaction", interaction))
    embed = interaction.response.sent["embed"]
    assert isinstance(embed, Embed)
    assert "無事發生" in (embed.description or "")


def test_memory_commands_have_localizations() -> None:
    for command in (MemoryCogs.memory, MemoryCogs.memory_show, MemoryCogs.memory_clear):
        assert command.name_localizations is not None
        assert Locale.zh_TW in command.name_localizations
        assert Locale.ja in command.name_localizations
        assert command.description_localizations is not None
        assert Locale.zh_TW in command.description_localizations
        assert Locale.ja in command.description_localizations
