"""Tests for the offline `uv run feedback-report` CLI."""

from io import StringIO
from pathlib import Path
import sqlite3

import pytest
from rich.console import Console

from discordbot.admin.feedback_report import (
    run_report,
    _format_identity,
    _resolve_console,
    _render_model_summary,
)


def _seed_db(db_path: Path) -> None:
    """Populates a migrated `reply.db` copy with turns and feedback rows."""
    conn = sqlite3.connect(database=db_path)
    try:
        conn.execute(
            """
            INSERT INTO reply_turn (
                reply_message_id, source_message_id, guild_id, channel_id, author_id,
                author_name, author_display_name, surface, route, model, effort, backend,
                question, answer, created_at
            ) VALUES
                (101, 1001, 1, 10, 500, 'mai9999', '麥麥', 'gateway', 'QA', 'gemini-4-neon',
                 'medium', 'responses', 'Explain WAL mode',
                 'WAL lets readers and writers run together.', '2026-11-20T10:00:00+08:00'),
                (102, 1002, 1, 10, 500, NULL, NULL, 'ask', 'QA', 'gemini-4-carbon',
                 'high', 'responses', 'Fix my async bug',
                 'Await the coroutine before returning.', '2026-11-20T11:00:00+08:00'),
                (103, 1003, 1, 10, 600, 'catlover', 'catlover', 'gateway', 'IMAGE',
                 'gemini-nano-banana-2.1', '', 'responses', 'Draw a cat',
                 'Here is your cat!', '2026-11-18T12:00:00+08:00'),
                (105, 1005, NULL, 999, 800, 'dmuser', 'DM Nick', 'gateway', 'QA',
                 'gemini-4-neon', 'low', 'responses', 'Remember what I said earlier?',
                 'Yes, you asked about Rust lifetimes.', '2026-11-21T09:00:00+08:00')
            """
        )
        conn.execute(
            """
            INSERT INTO reply_feedback (
                reply_message_id, user_id, user_name, user_display_name, vote,
                reason_tag, reason_text, created_at, updated_at
            ) VALUES
                (101, 500, 'mai9999', '麥麥', 'up', NULL, NULL,
                 '2026-11-20T10:01:00+08:00', '2026-11-20T10:01:00+08:00'),
                (102, 500, 'mai9999', '麥麥', 'down', 'inaccurate', NULL,
                 '2026-11-20T11:02:00+08:00', '2026-11-20T11:02:00+08:00'),
                (102, 777, 'bystander', NULL, 'down', 'other', 'wrong line number',
                 '2026-11-20T11:05:00+08:00', '2026-11-20T11:05:00+08:00'),
                (105, 800, 'dmuser', 'DM Nick', 'down', 'verbose', NULL,
                 '2026-11-21T09:02:00+08:00', '2026-11-21T09:02:00+08:00')
            """
        )
        conn.commit()
    finally:
        conn.close()


def _seed_messages_db(db_path: Path) -> None:
    """Populates a migrated `messages.db` copy with guild and DM channel history."""
    conn = sqlite3.connect(database=db_path)
    try:
        conn.execute(
            """
            INSERT INTO messages (
                discord_message_id, source_type, author, author_id, content, created_at,
                channel_name, channel_id, attachments, stickers
            ) VALUES
                ('990', 'guild', 'other_chan', '900', 'Unrelated channel message',
                 '2026-11-20 01:55:00', 'channel_other_20', '20', '', ''),
                ('995', 'guild', 'alice', '501', 'First context line',
                 '2026-11-20 01:57:00', 'channel_general_10', '10',
                 'https://cdn.example/a.png;https://cdn.example/b.png', ''),
                ('996', 'guild', 'bob', '502', '',
                 '2026-11-20 01:58:00', 'channel_general_10', '10',
                 '', 'https://cdn.example/sticker.png'),
                ('997', 'guild', 'mai9999', '500', 'Same-second earlier message [/] [b]',
                 '2026-11-20 02:00:00', 'channel_general_10', '10',
                 'https://cdn.example/c.png', ''),
                ('1001', 'guild', 'mai9999', '500', 'Explain WAL mode',
                 '2026-11-20 02:00:00', 'channel_general_10', '10', '', ''),
                ('1009', 'guild', 'later_user', '503', 'Message after source',
                 '2026-11-20 02:01:00', 'channel_general_10', '10', '', ''),
                ('2001', 'dm', 'dmuser', '800', 'Earlier DM about Rust lifetimes',
                 '2026-11-21 00:58:00', 'DM_dmuser_800', '800', '', ''),
                ('1005', 'dm', 'dmuser', '800', 'Remember what I said earlier?',
                 '2026-11-21 01:00:00', 'DM_dmuser_800', '800', '', '')
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


def _copy_messages_db(tmp_path: Path, migrated_databases: dict[str, Path]) -> Path:
    """Copies the migrated `messages.db` template into `tmp_path` and seeds channel history."""
    db_path = tmp_path / "messages.db"
    db_path.write_bytes(migrated_databases["messages"].read_bytes())
    _seed_messages_db(db_path=db_path)
    return db_path


def _capture(argv: list[str], width: int = 240) -> str:
    """Runs `run_report` into an in-memory Rich console and returns plain output."""
    buf = StringIO()
    console = Console(file=buf, width=width, force_terminal=False, color_system=None)
    run_report(argv=argv, out=console)
    return buf.getvalue()


def test_feedback_report_summarizes_by_model_and_filters(
    tmp_path: Path, migrated_databases: dict[str, Path]
) -> None:
    """Default view prints collapsed ASCII reasons, answer snippets, and supports filters."""
    db_path = _copy_reply_db(tmp_path=tmp_path, migrated_databases=migrated_databases)

    full = _capture(argv=["--db", str(db_path), "--messages-db", str(tmp_path / "none.db")])
    assert "gemini-4-neon" in full
    assert "gemini-4-carbon" in full
    assert "gemini-nano-banana-2.1" in full
    assert "Appr" in full
    assert "Reasons" in full
    assert "Answer" in full
    assert "資訊錯誤x1, 其他x1" in full
    assert "太囉嗦x1" in full
    assert "\u00d7" not in full
    assert "wrong line number" in full
    assert "WAL lets readers and writers run together." in full
    assert "麥麥 (@mai9999, 500) *" in full
    assert "bystander (777)" in full

    asker_down = _capture(
        argv=[
            "--db",
            str(db_path),
            "--messages-db",
            str(tmp_path / "none.db"),
            "--route",
            "QA",
            "--down-only",
            "--asker-only",
        ]
    )
    assert "麥麥 (@mai9999, 500) *" in asker_down
    assert "bystander" not in asker_down
    assert "gemini-nano-banana-2.1" not in asker_down

    reason_filtered = _capture(
        argv=[
            "--db",
            str(db_path),
            "--messages-db",
            str(tmp_path / "none.db"),
            "--reason",
            "verbose",
        ]
    )
    assert "DM Nick (@dmuser, 800) *" in reason_filtered
    assert "wrong line number" not in reason_filtered

    since_filtered = _capture(
        argv=[
            "--db",
            str(db_path),
            "--messages-db",
            str(tmp_path / "none.db"),
            "--since",
            "2026-11-20",
        ]
    )
    assert "gemini-nano-banana-2.1" not in since_filtered
    assert "gemini-4-carbon" in since_filtered


def test_feedback_report_no_wrap_and_non_terminal_widening(
    tmp_path: Path, migrated_databases: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Key summary/detail columns disable wrapping and non-terminal consoles widen to at least 160."""
    table = _render_model_summary(turns=[], feedback=[])
    no_wrap_headers = {col.header: col.no_wrap for col in table.columns}
    for header in ("Route", "Model", "Turns", "👍", "👎", "Appr", "Regen"):
        assert no_wrap_headers[header] is True

    db_path = _copy_reply_db(tmp_path=tmp_path, migrated_databases=migrated_databases)
    narrow = _capture(
        argv=["--db", str(db_path), "--messages-db", str(tmp_path / "none.db")], width=80
    )
    assert "gemini-nano-banana-2.1" in narrow
    assert "gemini-4-carbon" in narrow

    class _FakeConsole:
        def __init__(self, width: int = 80) -> None:
            self.width = width
            self.is_terminal = False

    monkeypatch.setattr("discordbot.admin.feedback_report.Console", _FakeConsole)
    widened = _resolve_console(out=None)
    assert widened.width == 160


def test_feedback_report_show_reads_guild_and_dm_context_and_falls_back_author(
    tmp_path: Path, migrated_databases: dict[str, Path]
) -> None:
    """`--show` prints preceding guild/DM messages, media markers, and falls back author name."""
    db_path = _copy_reply_db(tmp_path=tmp_path, migrated_databases=migrated_databases)
    msg_db = _copy_messages_db(tmp_path=tmp_path, migrated_databases=migrated_databases)

    # Guild turn with preceding context (including same-second id tie-break, attachments, stickers).
    guild_scene = _capture(
        argv=[
            "--db",
            str(db_path),
            "--messages-db",
            str(msg_db),
            "--show",
            "101",
            "--context",
            "5",
        ]
    )
    assert "麥麥 (@mai9999, 500)" in guild_scene
    assert "First context line [attachments x2]" in guild_scene
    assert "bob (502): [sticker]" in guild_scene
    assert "Same-second earlier message [/] [b] [attachment]" in guild_scene
    assert "Unrelated channel message" not in guild_scene
    assert "Message after source" not in guild_scene
    assert guild_scene.index("First context line") < guild_scene.index("Same-second earlier")

    # Older turn with author_name IS NULL and `/ask` source_message_id absent from messages.db.
    older_scene = _capture(
        argv=["--db", str(db_path), "--messages-db", str(msg_db), "--show", "102"]
    )
    assert "mai9999 (500)" in older_scene
    assert "Fix my async bug" in older_scene
    assert "Await the coroutine before returning." in older_scene
    assert "麥麥 (@mai9999, 500) (asker)" in older_scene
    assert "bystander (777)" in older_scene
    assert "wrong line number" in older_scene
    assert "source message not found in messages.db" in older_scene

    # DM turn queries source_type='dm' AND channel_id=str(turn.author_id).
    dm_scene = _capture(argv=["--db", str(db_path), "--messages-db", str(msg_db), "--show", "105"])
    assert "DM / 999" in dm_scene
    assert "Earlier DM about Rust lifetimes" in dm_scene
    assert "DM Nick (@dmuser, 800)" in dm_scene
    assert "DM Nick (@dmuser, 800) (asker)" in dm_scene


def test_feedback_report_inspect_deduplicates_turns_and_degrades_without_messages_db(
    tmp_path: Path, migrated_databases: dict[str, Path]
) -> None:
    """`--inspect` prints one scene card per matched turn and degrades cleanly when messages.db is missing."""
    db_path = _copy_reply_db(tmp_path=tmp_path, migrated_databases=migrated_databases)
    missing_msg_db = tmp_path / "missing_messages.db"

    inspected = _capture(
        argv=[
            "--db",
            str(db_path),
            "--messages-db",
            str(missing_msg_db),
            "--inspect",
            "--down-only",
            "--context",
            "3",
        ]
    )
    # Turn 102 has two downvotes (asker + bystander) but appears once with both votes listed.
    assert inspected.count("reply 102") == 1
    assert inspected.count("reply 105") == 1
    assert "reply 101" not in inspected
    assert "messages.db not found" in inspected
    assert "bystander (777)" in inspected
    assert "wrong line number" in inspected
    # Without messages.db, NULL author_name falls back to bare numeric id.
    assert "Question (500)" in inspected

    empty = _capture(
        argv=[
            "--db",
            str(db_path),
            "--messages-db",
            str(missing_msg_db),
            "--inspect",
            "--reason",
            "tone",
        ]
    )
    assert "No matching feedback turns." in empty

    zero_limit = _capture(
        argv=[
            "--db",
            str(db_path),
            "--messages-db",
            str(missing_msg_db),
            "--inspect",
            "--limit",
            "0",
        ]
    )
    assert "No matching feedback turns." in zero_limit


def test_feedback_report_exits_cleanly_on_missing_db_or_bad_args(
    tmp_path: Path, migrated_databases: dict[str, Path]
) -> None:
    """Missing database, unknown `--show` id, invalid `--since`, or negative bounds exit cleanly."""
    missing_msg_db = str(tmp_path / "none.db")
    with pytest.raises(SystemExit, match="no reply database"):
        _capture(argv=["--db", str(tmp_path / "missing.db"), "--messages-db", missing_msg_db])

    db_path = _copy_reply_db(tmp_path=tmp_path, migrated_databases=migrated_databases)
    with pytest.raises(SystemExit, match="no reply_turn found"):
        _capture(argv=["--db", str(db_path), "--messages-db", missing_msg_db, "--show", "999999"])

    with pytest.raises(SystemExit):
        _capture(
            argv=["--db", str(db_path), "--messages-db", missing_msg_db, "--since", "20261120"]
        )

    with pytest.raises(SystemExit):
        _capture(argv=["--db", str(db_path), "--messages-db", missing_msg_db, "--context", "-1"])

    with pytest.raises(SystemExit):
        _capture(argv=["--db", str(db_path), "--messages-db", missing_msg_db, "--limit", "-1"])

    empty_db = tmp_path / "empty_reply.db"
    sqlite3.connect(database=empty_db).close()
    with pytest.raises(SystemExit, match="reply feedback tables missing"):
        _capture(argv=["--db", str(empty_db), "--messages-db", missing_msg_db, "--show", "101"])


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
                author_name, author_display_name, surface, route, model, effort, backend,
                question, answer, created_at
            ) VALUES (
                104, 1004, 1, 10, 500, 'user[/]name', '[Owner] [/b]Nick', 'gateway', 'QA',
                'gemini-4-neon', 'low', 'responses', '[attachment] check [/]',
                'See [docs](https://example.com) [/b]', '2026-11-20T13:00:00+08:00'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO reply_feedback (
                reply_message_id, user_id, user_name, user_display_name, vote, reason_tag,
                reason_text, created_at, updated_at
            ) VALUES (
                104, 500, 'user[/]name', '[Owner] [/b]Nick', 'down', 'other', 'broken [/] [b]tag',
                '2026-11-20T13:01:00+08:00', '2026-11-20T13:01:00+08:00'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    summary = _capture(argv=["--db", str(db_path), "--messages-db", str(tmp_path / "none.db")])
    assert "[attachment] check [/]" in summary
    assert "[Owner] [/b]Nick (@user[/]name, 500)" in summary
    assert "broken [/] [b]tag" in summary

    detail = _capture(
        argv=["--db", str(db_path), "--messages-db", str(tmp_path / "none.db"), "--show", "104"]
    )
    assert "[attachment] check [/]" in detail
    assert "See [docs](https://example.com) [/b]" in detail
    assert "[Owner] [/b]Nick (@user[/]name, 500)" in detail
    assert "broken [/] [b]tag" in detail


def test_format_identity_handles_all_combinations() -> None:
    """Identity formatter covers distinct display name, identical name, username-only, and id-only."""
    assert (
        _format_identity(user_id=1, username="alice", display_name="Alice") == "Alice (@alice, 1)"
    )
    assert _format_identity(user_id=1, username="alice", display_name="alice") == "alice (1)"
    assert _format_identity(user_id=1, username="alice", display_name=None) == "alice (1)"
    assert _format_identity(user_id=1, username=None, display_name="Alice") == "Alice (1)"
    assert _format_identity(user_id=1, username=None, display_name=None) == "1"


def _seed_regen_db(db_path: Path) -> None:
    """Seeds regen chains covering explicit win, soft accept, both failed, and prior explicit downvote."""
    conn = sqlite3.connect(database=db_path)
    try:
        conn.execute(
            """
            INSERT INTO reply_turn (
                reply_message_id, source_message_id, guild_id, channel_id, author_id,
                author_name, author_display_name, surface, route, model, effort, backend,
                question, answer, regen_of_reply_message_id, created_at
            ) VALUES
                -- Chain 1: carbon (explicit down before regen) -> neon (explicit up => Explicit Win)
                (301, 3001, 1, 10, 500, 'mai9999', '麥麥', 'gateway', 'QA', 'gemini-4-carbon',
                 'high', 'responses', 'Q1', 'A1 carbon', NULL, '2026-11-20T10:00:00+08:00'),
                (302, 3001, 1, 10, 500, 'mai9999', '麥麥', 'gateway', 'QA', 'gemini-4-neon',
                 'high', 'responses', 'Q1', 'A2 neon', 301, '2026-11-20T10:00:30+08:00'),
                -- Chain 2: carbon (implicit regen) -> neon (no votes => Soft Accept)
                (303, 3002, 1, 10, 500, 'mai9999', '麥麥', 'gateway', 'QA', 'gemini-4-carbon',
                 'medium', 'responses', 'Q2', 'A1 carbon', NULL, '2026-11-20T11:00:00+08:00'),
                (304, 3002, 1, 10, 500, 'mai9999', '麥麥', 'gateway', 'QA', 'gemini-4-neon',
                 'medium', 'responses', 'Q2', 'A2 neon', 303, '2026-11-20T11:00:20+08:00'),
                -- Chain 3: carbon -> hydrogen (further regen child => Both Failed) -> neon (bystander down only)
                (305, 3003, 1, 10, 500, 'mai9999', '麥麥', 'gateway', 'QA', 'gemini-4-carbon',
                 'medium', 'responses', 'Q3', 'A1 carbon', NULL, '2026-11-21T12:00:00+08:00'),
                (306, 3003, 1, 10, 500, 'mai9999', '麥麥', 'gateway', 'QA', 'gemini-4-hydrogen',
                 'medium', 'responses', 'Q3', 'A2 hydrogen', 305, '2026-11-21T12:00:20+08:00'),
                (307, 3003, 1, 10, 500, 'mai9999', '麥麥', 'gateway', 'QA', 'gemini-4-neon',
                 'medium', 'responses', 'Q3', 'A3 neon', 306, '2026-11-21T12:00:40+08:00')
            """
        )
        conn.execute(
            """
            INSERT INTO reply_feedback (
                reply_message_id, user_id, user_name, user_display_name, vote,
                reason_tag, reason_text, created_at, updated_at
            ) VALUES
                (301, 500, 'mai9999', '麥麥', 'down', 'inaccurate', NULL,
                 '2026-11-20T10:00:10+08:00', '2026-11-20T10:00:10+08:00'),
                (302, 500, 'mai9999', '麥麥', 'up', NULL, NULL,
                 '2026-11-20T10:00:40+08:00', '2026-11-20T10:00:40+08:00'),
                (303, 500, 'mai9999', '麥麥', 'down', 'regen', NULL,
                 '2026-11-20T11:00:21+08:00', '2026-11-20T11:00:21+08:00'),
                (305, 500, 'mai9999', '麥麥', 'down', 'regen', NULL,
                 '2026-11-21T12:00:21+08:00', '2026-11-21T12:00:21+08:00'),
                (306, 500, 'mai9999', '麥麥', 'down', 'regen', NULL,
                 '2026-11-21T12:00:41+08:00', '2026-11-21T12:00:41+08:00'),
                (307, 777, 'bystander', NULL, 'down', 'verbose', NULL,
                 '2026-11-21T12:01:00+08:00', '2026-11-21T12:01:00+08:00')
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_feedback_report_separates_regen_from_explicit_votes_and_renders_pairwise(
    tmp_path: Path, migrated_databases: dict[str, Path]
) -> None:
    """Explicit votes exclude `reason_tag='regen'`, `Regen` counts structurally, and `Pairwise` respects filters."""
    # Without any regen edges, the Pairwise table is omitted.
    plain_db = _copy_reply_db(tmp_path=tmp_path, migrated_databases=migrated_databases)
    plain_out = _capture(argv=["--db", str(plain_db), "--messages-db", str(tmp_path / "none.db")])
    assert "Pairwise" not in plain_out
    assert "regens" in plain_out
    assert "0 (0.0%)" in plain_out

    regen_db = tmp_path / "regen_reply.db"
    regen_db.write_bytes(migrated_databases["reply"].read_bytes())
    _seed_regen_db(db_path=regen_db)

    full = _capture(argv=["--db", str(regen_db), "--messages-db", str(tmp_path / "none.db")])
    # Overview: 7 turns, 3 explicit votes (1 up, 2 explicit down; 3 regen rows excluded), 4 regens (57.1%).
    assert "3  (👍 1 / 👎 2, 2 voter(s))" in full
    assert "4 (57.1%)" in full
    # By Model: carbon has 3 turns, 1 explicit down (301), 3 regens (100.0%), and reasons exclude 'regen'.
    assert "3 (100.0%)" in full
    assert "資訊錯誤x1" in full
    assert "重新生成x" not in full
    # Pairwise table present with all 3 model pairs:
    # carbon -> neon: 2 pairs (1 Explicit Win, 1 Soft Accept, 0 Both Failed, 100.0%)
    # carbon -> hydrogen: 1 pair (0, 0, 1 Both Failed, 0.0%)
    # hydrogen -> neon: 1 pair (bystander down => Both Failed, 0.0%)
    assert "Pairwise" in full
    assert "Explicit Win" in full
    assert "Soft Accept" in full
    assert "Both Failed" in full
    assert "Accept Rate" in full
    assert "100.0%" in full
    # Recent Feedback renders reason_tag='regen' as 重新生成.
    assert "重新生成" in full

    # --asker-only drops bystander's downvote on 307, turning hydrogen -> neon into Soft Accept (100.0%).
    asker_only = _capture(
        argv=[
            "--db",
            str(regen_db),
            "--messages-db",
            str(tmp_path / "none.db"),
            "--asker-only",
            "--since",
            "2026-11-21",
        ]
    )
    assert "gemini-4-hydrogen" in asker_only
    # On 2026-11-21, hydrogen -> neon has 1 pair with Soft Accept => 100.0%.
    assert "100.0%" in asker_only

    # --model keeps structural Regen count and cross-model Pairwise rows touching that model.
    carbon_only = _capture(
        argv=[
            "--db",
            str(regen_db),
            "--messages-db",
            str(tmp_path / "none.db"),
            "--model",
            "gemini-4-carbon",
        ]
    )
    assert "3 (100.0%)" in carbon_only
    assert "Pairwise" in carbon_only
    assert "gemini-4-neon" in carbon_only
    assert "gemini-4-hydrogen" in carbon_only

    # --show displays regen_of / regenerated_to links and renders 'regen' reason as 重新生成.
    show_middle = _capture(
        argv=["--db", str(regen_db), "--messages-db", str(tmp_path / "none.db"), "--show", "306"]
    )
    assert "regen_of" in show_middle
    assert "305" in show_middle
    assert "regenerated_to" in show_middle
    assert "307" in show_middle
    assert "重新生成" in show_middle
