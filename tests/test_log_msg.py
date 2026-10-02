"""Tests for the canonical message logging path."""

from typing import Any
import asyncio

from sqlalchemy import text

from discordbot.cogs.log_msg import cog as log_msg

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


async def _query(sql: str) -> list[tuple[Any, ...]]:
    """Reads rows back through the module's own (swapped) engine."""
    async with log_msg._engine.connect() as conn:
        result = await conn.execute(statement=text(text=sql))
        return [tuple(row) for row in result.all()]


async def test_write_row_creates_table_and_inserts() -> None:
    """First write creates the canonical messages table, then inserts the row."""
    await log_msg._write_row(row=_SAMPLE_ROW)

    rows = await _query(
        sql='SELECT discord_message_id, source_type, author, author_id, content FROM "messages"'
    )
    legacy_tables = await _query(
        sql="SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB 'channel_*'"
    )
    assert rows == [("1001", "guild", "alice", "42", "hello world")]
    assert legacy_tables == []


async def test_write_row_appends_to_existing_table() -> None:
    """Subsequent writes with distinct discord_message_ids append fresh rows."""
    await log_msg._write_row(row=_SAMPLE_ROW)
    second_row = {**_SAMPLE_ROW, "discord_message_id": "1002", "content": "second message"}
    await log_msg._write_row(row=second_row)

    rows = await _query(sql='SELECT content FROM "messages" ORDER BY id')
    assert rows == [("hello world",), ("second message",)]


async def test_write_row_upserts_on_same_discord_message_id() -> None:
    """Verifies that duplicate discord_message_id writes update one row."""
    await log_msg._write_row(row=_SAMPLE_ROW)
    edited_row = {
        **_SAMPLE_ROW,
        "content": "final streamed content with footer",
        "created_at": "2099-01-01 00:00:00",
    }
    await log_msg._write_row(row=edited_row)

    rows = await _query(sql='SELECT content, created_at FROM "messages"')
    assert rows == [("final streamed content with footer", "2026-05-11 12:00:00")]


async def test_write_row_stores_different_sources_in_one_table() -> None:
    """Different channel and DM rows land in one messages table."""
    await log_msg._write_row(row=_SAMPLE_ROW)
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
    await log_msg._write_row(row=other_row)
    await log_msg._write_row(row=dm_row)

    rows = await _query(sql='SELECT source_type, channel_id, content FROM "messages" ORDER BY id')
    user_tables = await _query(
        sql="""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND (name GLOB 'channel_*' OR name GLOB 'DM_*')
        """
    )
    assert rows == [
        ("guild", "99", "hello world"),
        ("guild", "100", "from another channel"),
        ("dm", "99", "from dm"),
    ]
    assert user_tables == []


async def test_write_row_concurrent_inserts_all_land() -> None:
    """Verifies that concurrent writes, the first one's schema bootstrap included, all land."""
    rows = [
        {**_SAMPLE_ROW, "discord_message_id": f"{2000 + i}", "content": f"msg-{i}"}
        for i in range(20)
    ]
    await asyncio.gather(*[log_msg._write_row(row=row) for row in rows])

    assert await _query(sql='SELECT COUNT(*) FROM "messages"') == [(20,)]
