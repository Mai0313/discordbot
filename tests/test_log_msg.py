"""Tests for the raw-INSERT message logging path."""

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
    monkeypatch.setattr(target=log_msg, name="_INITIALIZED_TABLES", value=set())
    yield engine
    engine.dispose()


_SAMPLE_ROW: dict[str, str] = {
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
    """First write to a new table creates it, then inserts the row."""
    log_msg._write_row_sync(table_name="channel_99", row=_SAMPLE_ROW)
    with isolated_db.connect() as conn:
        rows = conn.execute(text('SELECT author, content FROM "channel_99"')).all()
    assert rows == [("alice", "hello world")]
    assert "channel_99" in log_msg._INITIALIZED_TABLES


def test_write_row_appends_to_existing_table(isolated_db: Engine) -> None:
    """Subsequent writes append without re-running CREATE TABLE."""
    log_msg._write_row_sync(table_name="channel_99", row=_SAMPLE_ROW)
    second_row = {**_SAMPLE_ROW, "content": "second message"}
    log_msg._write_row_sync(table_name="channel_99", row=second_row)

    with isolated_db.connect() as conn:
        rows = conn.execute(text('SELECT content FROM "channel_99" ORDER BY rowid')).all()
    assert rows == [("hello world",), ("second message",)]


def test_write_row_isolates_tables_per_channel(isolated_db: Engine) -> None:
    """Different channel IDs land in different tables."""
    log_msg._write_row_sync(table_name="channel_99", row=_SAMPLE_ROW)
    other_row = {**_SAMPLE_ROW, "channel_id": "100", "content": "from another channel"}
    log_msg._write_row_sync(table_name="channel_100", row=other_row)

    with isolated_db.connect() as conn:
        first = conn.execute(text('SELECT content FROM "channel_99"')).all()
        second = conn.execute(text('SELECT content FROM "channel_100"')).all()
    assert first == [("hello world",)]
    assert second == [("from another channel",)]


def test_write_row_accepts_legacy_bigint_schema(isolated_db: Engine) -> None:
    """Older tables built by pandas had BIGINT id columns; affinity must cope.

    Some historical tables in `data/messages.db` were built with
    ``author_id BIGINT``/``channel_id BIGINT`` because pandas inferred those
    columns from raw ints before `.astype(str)` was added. SQLite's INTEGER
    affinity transparently casts well-formed numeric TEXT into INTEGER on
    INSERT, which matches what the legacy rows already store — so reads
    against existing BIGINT tables keep returning ints just like before.
    """
    with isolated_db.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE "channel_legacy" (
                author TEXT, author_id BIGINT, content TEXT,
                created_at TEXT, channel_name TEXT, channel_id BIGINT,
                attachments TEXT, stickers TEXT
            )
        """)
        )

    log_msg._write_row_sync(table_name="channel_legacy", row=_SAMPLE_ROW)

    with isolated_db.connect() as conn:
        rows = conn.execute(text('SELECT author_id, channel_id FROM "channel_legacy"')).all()
    # Affinity cast TEXT "42" → INTEGER 42 on store; consistent with the
    # historical rows that were originally INSERTed as ints.
    assert rows == [(42, 99)]


async def test_write_row_concurrent_inserts_all_land(isolated_db: Engine) -> None:
    """Twenty concurrent writes via `asyncio.to_thread` all land in the table.

    The file-level write lock plus `busy_timeout` serializes the threads;
    the test fails fast if any insert is silently dropped or if the
    table-init cache mis-skips the CREATE statement under contention.
    """
    rows = [{**_SAMPLE_ROW, "content": f"msg-{i}"} for i in range(20)]
    await asyncio.gather(*[
        asyncio.to_thread(log_msg._write_row_sync, "channel_99", row) for row in rows
    ])

    with isolated_db.connect() as conn:
        count = conn.execute(text('SELECT COUNT(*) FROM "channel_99"')).scalar_one()
    assert count == 20
