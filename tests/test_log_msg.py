"""Tests for the canonical message logging path."""

import asyncio
from pathlib import Path
import sqlite3
from collections.abc import Iterator

import pytest
from scripts import migrate_messages_db
from sqlalchemy import Engine, text, create_engine

from discordbot.cogs import log_msg


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """Replaces the module-level engine with a per-test SQLite file."""
    db_path = tmp_path / "messages.db"
    engine = create_engine(url=f"sqlite:///{db_path}")
    monkeypatch.setattr(target=log_msg, name="_sql_engine", value=engine)
    monkeypatch.setattr(target=log_msg, name="_MESSAGES_TABLE_READY_FOR", value=None)
    yield engine
    engine.dispose()


_SAMPLE_ROW: dict[str, str] = {
    "source_type": "guild",
    "author": "alice",
    "author_id": "42",
    "content": "hello world",
    "created_at": "2026-05-11 12:00:00",
    "channel_name": "channel_general_99",
    "channel_id": "99",
    "attachments": "",
    "stickers": "",
}


def test_write_row_creates_table_and_inserts(isolated_db: Engine) -> None:
    """First write creates the canonical messages table, then inserts the row."""
    log_msg._write_row_sync(row=_SAMPLE_ROW)
    with isolated_db.connect() as conn:
        rows = conn.execute(
            text('SELECT source_type, author, author_id, content FROM "messages"')
        ).all()
        legacy_tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB 'channel_*'")
        ).all()
    assert rows == [("guild", "alice", "42", "hello world")]
    assert legacy_tables == []
    assert log_msg._MESSAGES_TABLE_READY_FOR is isolated_db


def test_write_row_appends_to_existing_table(isolated_db: Engine) -> None:
    """Subsequent writes append to the same canonical table."""
    log_msg._write_row_sync(row=_SAMPLE_ROW)
    second_row = {**_SAMPLE_ROW, "content": "second message"}
    log_msg._write_row_sync(row=second_row)

    with isolated_db.connect() as conn:
        rows = conn.execute(text('SELECT content FROM "messages" ORDER BY id')).all()
    assert rows == [("hello world",), ("second message",)]


def test_write_row_stores_different_sources_in_one_table(isolated_db: Engine) -> None:
    """Different channel and DM rows land in one messages table."""
    log_msg._write_row_sync(row=_SAMPLE_ROW)
    other_row = {**_SAMPLE_ROW, "channel_id": "100", "content": "from another channel"}
    dm_row = {
        **_SAMPLE_ROW,
        "source_type": "dm",
        "channel_name": "DM_alice_42",
        "content": "from dm",
    }
    log_msg._write_row_sync(row=other_row)
    log_msg._write_row_sync(row=dm_row)

    with isolated_db.connect() as conn:
        rows = conn.execute(
            text('SELECT source_type, channel_id, content FROM "messages" ORDER BY id')
        ).all()
        user_tables = conn.execute(
            text("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND (name GLOB 'channel_*' OR name GLOB 'DM_*')
            """)
        ).all()
    assert rows == [
        ("guild", "99", "hello world"),
        ("guild", "100", "from another channel"),
        ("dm", "99", "from dm"),
    ]
    assert user_tables == []


async def test_write_row_concurrent_inserts_all_land(isolated_db: Engine) -> None:
    """Twenty concurrent writes via `asyncio.to_thread` all land in the table.

    The file-level write lock plus `busy_timeout` serializes the threads; the
    test fails fast if any insert is silently dropped.
    """
    rows = [{**_SAMPLE_ROW, "content": f"msg-{i}"} for i in range(20)]
    await asyncio.gather(*[asyncio.to_thread(log_msg._write_row_sync, row=row) for row in rows])

    with isolated_db.connect() as conn:
        count = conn.execute(text('SELECT COUNT(*) FROM "messages"')).scalar_one()
    assert count == 20


def test_migrate_messages_db_merges_legacy_tables(tmp_path: Path) -> None:
    """Legacy per-source tables are copied into one canonical messages table."""
    source = tmp_path / "messages.backup.db"
    dest = tmp_path / "messages.db"
    with sqlite3.connect(database=source) as conn:
        conn.execute("""
            CREATE TABLE "channel_99" (
                author TEXT, author_id BIGINT, content TEXT,
                created_at TEXT, channel_name TEXT, channel_id BIGINT,
                attachments TEXT, stickers TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE "DM_42" (
                author TEXT, author_id TEXT, content TEXT,
                created_at TEXT, channel_name TEXT, channel_id TEXT,
                attachments TEXT, stickers TEXT
            )
        """)
        conn.execute(
            """
            INSERT INTO "channel_99"
                (author, author_id, content, created_at, channel_name, channel_id, attachments, stickers)
            VALUES
                ('alice', 42, 'guild msg', '2026-05-11 12:00:00', 'channel_general_99', 99, '', '')
            """
        )
        conn.execute(
            """
            INSERT INTO "DM_42"
                (author, author_id, content, created_at, channel_name, channel_id, attachments, stickers)
            VALUES
                ('alice', '42', 'dm msg', '2026-05-11 12:01:00', 'DM_alice_42', '42', '', '')
            """
        )

    summary = migrate_messages_db.migrate_messages_db(source=source, dest=dest, progress_every=0)

    assert summary == migrate_messages_db.MigrationSummary(
        table_count=2, row_count=2, integrity_check="ok"
    )
    with sqlite3.connect(database=dest) as conn:
        rows = conn.execute("""
            SELECT source_type, author, author_id, content, channel_id
            FROM messages
            ORDER BY id
        """).fetchall()
        legacy_tables = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND (name GLOB 'channel_*' OR name GLOB 'DM_*')
        """).fetchall()
    assert rows == [
        ("dm", "alice", "42", "dm msg", "42"),
        ("guild", "alice", "42", "guild msg", "99"),
    ]
    assert legacy_tables == []
