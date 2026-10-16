"""Tests for the store that keeps `/ask` conversations."""

import pytest

from discordbot.cogs.gen_reply import ask_store
from discordbot.cogs.gen_reply.ask_store import load_ask_turns, record_ask_turn

from tests.helpers.ask_interaction import ASKER_ID, CHANNEL_ID, ASK_SNOWFLAKE


async def test_the_store_replays_a_conversation_oldest_first() -> None:
    """What went in comes back in transcript order, scoped to one person in one channel."""
    for index, (question, answer) in enumerate([("一", "1"), ("二", "2"), ("三", "3")]):
        await record_ask_turn(
            channel_id=CHANNEL_ID,
            user_id=ASKER_ID,
            message_id=ASK_SNOWFLAKE + index * 2,
            question=question,
            answer=answer,
        )
    await record_ask_turn(
        channel_id=CHANNEL_ID,
        user_id=ASKER_ID + 1,
        message_id=ASK_SNOWFLAKE,
        question="別人的",
        answer="別人的回覆",
    )

    turns = await load_ask_turns(channel_id=CHANNEL_ID, user_id=ASKER_ID, limit=10)

    assert [turn.question for turn in turns] == ["一", "二", "三"]
    assert [turn.answer for turn in turns] == ["1", "2", "3"]


async def test_the_store_keeps_only_its_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    """A conversation someone keeps up for a year must not grow the table without limit."""
    monkeypatch.setattr(ask_store, "ASK_TURN_RETENTION", 2)
    for index in range(4):
        await record_ask_turn(
            channel_id=CHANNEL_ID,
            user_id=ASKER_ID,
            message_id=ASK_SNOWFLAKE + index * 2,
            question=str(index),
            answer=str(index),
        )

    turns = await load_ask_turns(channel_id=CHANNEL_ID, user_id=ASKER_ID, limit=10)

    assert [turn.question for turn in turns] == ["2", "3"]
