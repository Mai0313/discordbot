"""Tests for the per-user long-term memory helpers."""

from pathlib import Path

import pytest

from discordbot.cogs._memory import store
from discordbot.cogs._memory.constants import MEMORY_INJECTION_MAX_CHARS

USER_ID = 123456789


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
    import time

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
