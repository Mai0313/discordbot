"""Tests for the offline `uv run feedback-report` CLI."""

from io import StringIO
from pathlib import Path
import sqlite3

import pytest
from rich.console import Console

from discordbot.admin.feedback_report import run_report


def _seed_db(db_path: Path) -> None:
    """Populates a migrated `reply.db` copy with turns and feedback rows."""
    conn = sqlite3.connect(database=db_path)
    try:
        conn.execute(
            """
            INSERT INTO reply_turn (
                reply_message_id, source_message_id, guild_id, channel_id, author_id,
                surface, route, model, effort, backend, question, answer, created_at
            ) VALUES
                (101, 1001, 1, 10, 500, 'gateway', 'QA', 'gemini-4-neon', 'medium', 'responses',
                 'Explain WAL mode', 'WAL lets readers and writers run together.', '2026-11-20T10:00:00+08:00'),
                (102, 1002, 1, 10, 500, 'ask', 'QA', 'gemini-4-carbon', 'high', 'responses',
                 'Fix my async bug', 'Await the coroutine before returning.', '2026-11-20T11:00:00+08:00'),
                (103, 1003, 1, 10, 600, 'gateway', 'IMAGE', 'gemini-nano-banana-2.1', '', 'responses',
                 'Draw a cat', 'Here is your cat!', '2026-11-20T12:00:00+08:00')
            """
        )
        conn.execute(
            """
            INSERT INTO reply_feedback (
                reply_message_id, user_id, user_name, vote, reason_tag, reason_text,
                created_at, updated_at
            ) VALUES
                (101, 500, 'mai9999', 'up', NULL, NULL,
                 '2026-11-20T10:01:00+08:00', '2026-11-20T10:01:00+08:00'),
                (102, 500, 'mai9999', 'down', 'inaccurate', NULL,
                 '2026-11-20T11:02:00+08:00', '2026-11-20T11:02:00+08:00'),
                (102, 777, 'bystander', 'down', 'other', 'wrong line number',
                 '2026-11-20T11:05:00+08:00', '2026-11-20T11:05:00+08:00')
            """
        )
        conn.commit()
    finally:
        conn.close()


def _copy_reply_db(tmp_path: Path, migrated_databases: dict[str, Path]) -> Path:
    """Copies the migrated `reply.db` template into `tmp_path` and seeds sample rows."""
    db_path = tmp_path / "reply.db"
    db_path.write_bytes(migrated_databases["reply"].read_bytes())
    _seed_db(db_path=db_path)
    return db_path


def _capture(argv: list[str]) -> str:
    """Runs `run_report` into an in-memory Rich console and returns plain output."""
    buf = StringIO()
    console = Console(file=buf, width=160, force_terminal=False, color_system=None)
    run_report(argv=argv, out=console)
    return buf.getvalue()


def test_feedback_report_summarizes_by_model_and_filters(
    tmp_path: Path, migrated_databases: dict[str, Path]
) -> None:
    """Default view prints model totals and details; filters narrow turns and voters."""
    db_path = _copy_reply_db(tmp_path=tmp_path, migrated_databases=migrated_databases)

    full = _capture(argv=["--db", str(db_path)])
    assert "gemini-4-neon" in full
    assert "gemini-4-carbon" in full
    assert "gemini-nano-banana-2.1" in full
    assert "100.0%" in full
    assert "0.0%" in full
    assert "資訊錯誤" in full
    assert "wrong line number" in full

    asker_down = _capture(
        argv=["--db", str(db_path), "--route", "QA", "--down-only", "--asker-only"]
    )
    assert "mai9999 *" in asker_down
    assert "bystander" not in asker_down
    assert "gemini-nano-banana-2.1" not in asker_down


def test_feedback_report_show_prints_full_qa_and_votes(
    tmp_path: Path, migrated_databases: dict[str, Path]
) -> None:
    """`--show <id>` prints full untruncated question, answer, and every vote on that reply."""
    db_path = _copy_reply_db(tmp_path=tmp_path, migrated_databases=migrated_databases)

    detail = _capture(argv=["--db", str(db_path), "--show", "102"])
    assert "Fix my async bug" in detail
    assert "Await the coroutine before returning." in detail
    assert "mai9999 (asker)" in detail
    assert "bystander" in detail
    assert "wrong line number" in detail


def test_feedback_report_exits_cleanly_on_missing_db_or_turn(
    tmp_path: Path, migrated_databases: dict[str, Path]
) -> None:
    """Missing database or unknown `--show` id fails cleanly with `SystemExit`."""
    with pytest.raises(SystemExit, match="no reply database"):
        _capture(argv=["--db", str(tmp_path / "missing.db")])

    db_path = _copy_reply_db(tmp_path=tmp_path, migrated_databases=migrated_databases)
    with pytest.raises(SystemExit, match="no reply_turn found"):
        _capture(argv=["--db", str(db_path), "--show", "999999"])


def test_feedback_report_preserves_bracketed_text_without_rich_markup_errors(
    tmp_path: Path, migrated_databases: dict[str, Path]
) -> None:
    """Square brackets such as `[attachment]` and `[/]` render literally without Rich errors."""
    db_path = _copy_reply_db(tmp_path=tmp_path, migrated_databases=migrated_databases)
    conn = sqlite3.connect(database=db_path)
    try:
        conn.execute(
            """
            INSERT INTO reply_turn (
                reply_message_id, source_message_id, guild_id, channel_id, author_id,
                surface, route, model, effort, backend, question, answer, created_at
            ) VALUES (
                104, 1004, 1, 10, 500, 'gateway', 'QA', 'gemini-4-neon', 'low', 'responses',
                '[attachment] check [/]', 'See [docs](https://example.com) [/b]', '2026-11-20T13:00:00+08:00'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO reply_feedback (
                reply_message_id, user_id, user_name, vote, reason_tag, reason_text,
                created_at, updated_at
            ) VALUES (
                104, 500, 'user[/]name', 'down', 'other', 'broken [/] [b]tag',
                '2026-11-20T13:01:00+08:00', '2026-11-20T13:01:00+08:00'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    summary = _capture(argv=["--db", str(db_path)])
    assert "[attachment] check [/]" in summary
    assert "user[/]name" in summary
    assert "broken [/] [b]tag" in summary

    detail = _capture(argv=["--db", str(db_path), "--show", "104"])
    assert "[attachment] check [/]" in detail
    assert "See [docs](https://example.com) [/b]" in detail
    assert "broken [/] [b]tag" in detail
