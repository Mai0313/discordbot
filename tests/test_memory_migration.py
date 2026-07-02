"""Tests for the deterministic helpers of the one-shot memory privacy migration."""

from pathlib import Path

from scripts.migrate_memory_privacy import (
    _resolve_user_ids,
    tag_untagged_bullets,
    count_untagged_bullets,
)

# A migrated-in-progress main.md: the profile paragraph stays untagged by design,
# two bullets already carry a tag, and two (one `* `, one `- `) still lack one.
SAMPLE_MAIN = (
    "v1\n"
    "\n"
    "## 使用者輪廓\n"
    "喜歡繁體中文,對 Python 有興趣。\n"
    "\n"
    "## 永久事實\n"
    "* 使用者是男性 [src:*]\n"
    "\n"
    "## 穩定偏好\n"
    "* [~2026-06] 喜歡簡短回覆 [src:987654321098765432]\n"
    "- [~2026-05] 提過搬家計畫\n"
    "* 常揪 [id: 42] 打遊戲\n"
)


def test_count_untagged_bullets_counts_both_bullet_shapes() -> None:
    assert count_untagged_bullets(text=SAMPLE_MAIN) == (4, 2)


def test_count_untagged_bullets_skips_profile_section() -> None:
    profile_only = "v1\n\n## 使用者輪廓\n* 輪廓段落裡的條列不計\n"
    assert count_untagged_bullets(text=profile_only) == (0, 0)


def test_tag_untagged_bullets_appends_legacy_tag_outside_profile() -> None:
    rewritten, fixed = tag_untagged_bullets(text=SAMPLE_MAIN)
    assert fixed == 2
    # Already-tagged bullets and the profile paragraph stay byte-identical.
    assert "* 使用者是男性 [src:*]\n" in rewritten
    assert "* [~2026-06] 喜歡簡短回覆 [src:987654321098765432]\n" in rewritten
    assert "喜歡繁體中文,對 Python 有興趣。\n" in rewritten
    # Both bullet shapes gain the legacy tag as the last token.
    assert "- [~2026-05] 提過搬家計畫 [src:legacy]" in rewritten
    assert "* 常揪 [id: 42] 打遊戲 [src:legacy]" in rewritten
    assert count_untagged_bullets(text=rewritten) == (4, 0)


def test_tag_untagged_bullets_keeps_profile_bullets_untouched() -> None:
    text = "## 使用者輪廓\n* 輪廓內條列\n\n## 穩定偏好\n* 未標記偏好\n"
    rewritten, fixed = tag_untagged_bullets(text=text)
    assert fixed == 1
    assert "* 輪廓內條列\n" in rewritten
    assert "* 輪廓內條列 [src:legacy]" not in rewritten
    assert "* 未標記偏好 [src:legacy]" in rewritten


def test_tag_untagged_bullets_is_idempotent() -> None:
    once, _ = tag_untagged_bullets(text=SAMPLE_MAIN)
    twice, fixed = tag_untagged_bullets(text=once)
    assert fixed == 0
    assert twice == once


def test_resolve_user_ids_handles_single_user_and_root_folder(tmp_path: Path) -> None:
    single = tmp_path / "123"
    single.mkdir()
    assert _resolve_user_ids(folder=single) == [123]
    root = tmp_path / "memories"
    (root / "42").mkdir(parents=True)
    (root / "7").mkdir()
    # Non-numeric dirs (e.g. the bot's server-memory parent named after a word)
    # and plain files are skipped; the result is sorted numerically.
    (root / "not-a-user").mkdir()
    (root / "99").write_text(data="a file, not a user dir", encoding="utf-8")
    assert _resolve_user_ids(folder=root) == [7, 42]
