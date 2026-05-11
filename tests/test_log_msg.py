"""Tests for the canonical message logging path."""

import asyncio
from pathlib import Path
from collections.abc import Iterator

import pytest
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
    "discord_message_id": "1001",
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
            text(
                "SELECT discord_message_id, source_type, author, author_id, content "
                'FROM "messages"'
            )
        ).all()
        legacy_tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB 'channel_*'")
        ).all()
    assert rows == [("1001", "guild", "alice", "42", "hello world")]
    assert legacy_tables == []
    assert log_msg._MESSAGES_TABLE_READY_FOR is isolated_db


def test_write_row_appends_to_existing_table(isolated_db: Engine) -> None:
    """Subsequent writes with distinct discord_message_ids append fresh rows."""
    log_msg._write_row_sync(row=_SAMPLE_ROW)
    second_row = {**_SAMPLE_ROW, "discord_message_id": "1002", "content": "second message"}
    log_msg._write_row_sync(row=second_row)

    with isolated_db.connect() as conn:
        rows = conn.execute(text('SELECT content FROM "messages" ORDER BY id')).all()
    assert rows == [("hello world",), ("second message",)]


def test_write_row_upserts_on_same_discord_message_id(isolated_db: Engine) -> None:
    """Two writes sharing a discord_message_id collapse into one updated row.

    Mirrors the streaming-edit flow where bot replies edit themselves multiple
    times; messages.db should converge to the final on-Discord state, not
    accumulate the intermediate fragments. created_at stays pinned to the
    original send-time even after the content update.
    """
    log_msg._write_row_sync(row=_SAMPLE_ROW)
    edited_row = {
        **_SAMPLE_ROW,
        "content": "final streamed content with footer",
        "created_at": "2099-01-01 00:00:00",
    }
    log_msg._write_row_sync(row=edited_row)

    with isolated_db.connect() as conn:
        rows = conn.execute(text('SELECT content, created_at FROM "messages"')).all()
    assert rows == [("final streamed content with footer", "2026-05-11 12:00:00")]


def test_write_row_stores_different_sources_in_one_table(isolated_db: Engine) -> None:
    """Different channel and DM rows land in one messages table."""
    log_msg._write_row_sync(row=_SAMPLE_ROW)
    other_row = {
        **_SAMPLE_ROW,
        "discord_message_id": "1002",
        "channel_id": "100",
        "content": "from another channel",
    }
    dm_row = {
        **_SAMPLE_ROW,
        "discord_message_id": "1003",
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
    rows = [
        {**_SAMPLE_ROW, "discord_message_id": f"{2000 + i}", "content": f"msg-{i}"}
        for i in range(20)
    ]
    await asyncio.gather(*[asyncio.to_thread(log_msg._write_row_sync, row=row) for row in rows])

    with isolated_db.connect() as conn:
        count = conn.execute(text('SELECT COUNT(*) FROM "messages"')).scalar_one()
    assert count == 20


def test_write_row_migrates_legacy_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A legacy DB without the discord_message_id column is migrated in place."""
    db_path = tmp_path / "legacy.db"
    legacy_engine = create_engine(url=f"sqlite:///{db_path}")
    try:
        with legacy_engine.begin() as conn:
            conn.execute(
                text(
                    text="""
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    author TEXT,
                    author_id TEXT,
                    content TEXT,
                    created_at TEXT,
                    channel_name TEXT,
                    channel_id TEXT,
                    attachments TEXT,
                    stickers TEXT
                )
            """
                )
            )
            conn.execute(
                text(
                    text="""
                INSERT INTO messages
                    (source_type, author, author_id, content, created_at,
                     channel_name, channel_id, attachments, stickers)
                VALUES
                    ('guild', 'pre-existing', '1', 'legacy row',
                     '2024-01-01 00:00:00', 'channel_old_1', '1', '', '')
            """
                )
            )
    finally:
        legacy_engine.dispose()

    engine = create_engine(url=f"sqlite:///{db_path}")
    monkeypatch.setattr(target=log_msg, name="_sql_engine", value=engine)
    monkeypatch.setattr(target=log_msg, name="_MESSAGES_TABLE_READY_FOR", value=None)
    try:
        log_msg._write_row_sync(row=_SAMPLE_ROW)
        with engine.connect() as conn:
            columns = {
                row_info[1] for row_info in conn.execute(text(text="PRAGMA table_info(messages)"))
            }
            rows = conn.execute(
                text('SELECT discord_message_id, content FROM "messages" ORDER BY id')
            ).all()
        assert "discord_message_id" in columns
        assert rows == [(None, "legacy row"), ("1001", "hello world")]
    finally:
        engine.dispose()
