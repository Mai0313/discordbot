"""Pins the on-disk half of the message archive: `_write_row_sync` in `cogs/log_msg/cog.py`.

Every test drives that function directly rather than the cog. The listeners only schedule a
detached task and the row-building half needs a real `nextcord.Message`, while everything that
can go wrong on disk — the schema, the table layout, the UPSERT, the threading — lives in that
one function.

Nothing in the running bot ever reads this table back, so a regression here is invisible while
it happens: the archive quietly loses, duplicates or scrambles rows, and nobody finds out until
someone queries it offline. Four behaviours carry the weight. The schema is created on the first
write against an engine, keyed on engine identity, which is also what lets a test point the
module at a temp file at all. Every source shares one flat `messages` table, with the
pre-migration per-channel `channel_*` / `DM_*` tables asserted absent so that layout cannot creep
back. A repeat write on the same `discord_message_id` UPSERTs instead of appending and leaves
`created_at` at the original send time, which is what makes a streaming reply's many
`on_message_edit` writes converge on one row rather than pile up. And concurrent writes all land,
because the real caller hands every one of them to `asyncio.to_thread`.
"""

import asyncio
from pathlib import Path
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text, create_engine

from discordbot.cogs.log_msg import cog as log_msg


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """Points the cog module at a per-test SQLite file, engine and schema marker both.

    `_write_row_sync` writes `_MESSAGES_TABLE_READY_FOR` itself, so the marker is monkeypatched
    rather than merely reset: that is what restores this module-level global at teardown instead
    of leaving it pointed at an engine this fixture has already disposed.

    Yields:
        The engine now installed on the cog module, for reading the written rows back.
    """
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
    """First write creates the canonical `messages` table alone, inserts, and marks the engine."""
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
    """A repeat write on one `discord_message_id` updates that row and leaves `created_at`."""
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
    """Two channels and a DM share the one `messages` table, with no per-channel table made."""
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
    """Concurrent writes from worker threads all land, none lost to the first-write DDL race."""
    rows = [
        {**_SAMPLE_ROW, "discord_message_id": f"{2000 + i}", "content": f"msg-{i}"}
        for i in range(20)
    ]
    await asyncio.gather(*[asyncio.to_thread(log_msg._write_row_sync, row=row) for row in rows])

    with isolated_db.connect() as conn:
        count = conn.execute(text('SELECT COUNT(*) FROM "messages"')).scalar_one()
    assert count == 20
