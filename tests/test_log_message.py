from datetime import datetime

import pytest
from src.sdk.log_message import MessageLogger


class DummyUser:
    def __init__(self, user_id: int, name: str, nick: str | None = None) -> None:
        self.id = user_id
        self.name = name
        self.nick = nick
        self.bot = False


class DummyDMChannel:
    def __init__(self, channel_id: int) -> None:
        self.id = channel_id
        self.name = "dm"


class DummyTextChannel:
    def __init__(self, channel_id: int, name: str) -> None:
        self.id = channel_id
        self.name = name


class DummyAttachment:
    def __init__(self, url: str) -> None:
        self.url = url


class DummyMessage:
    def __init__(self, author, channel) -> None:
        self.author = author
        self.channel = channel
        self.content = "hello"
        self.created_at = datetime.utcnow()
        self.attachments: list[DummyAttachment] = []
        self.stickers: list[DummyAttachment] = []


@pytest.mark.asyncio
async def test_computed_fields_dm_channel() -> None:
    author = DummyUser(1, "tester", "tester_nick")
    message = DummyMessage(author, DummyDMChannel(10))
    logger = MessageLogger(message=message)

    assert logger.table_name == "DM_1"
    assert logger.channel_name_or_author_name == "DM_tester_nick_1"
    assert logger.channel_id_or_author_id == "1"


@pytest.mark.asyncio
async def test_computed_fields_text_channel() -> None:
    author = DummyUser(2, "tester")
    channel = DummyTextChannel(20, "general")
    message = DummyMessage(author, channel)
    logger = MessageLogger(message=message)

    assert logger.table_name == "channel_20"
    assert logger.channel_name_or_author_name == "channel_general_20"
    assert logger.channel_id_or_author_id == "20"


@pytest.mark.asyncio
async def test_save_attachments_and_stickers() -> None:
    author = DummyUser(3, "tester")
    channel = DummyTextChannel(30, "general")
    message = DummyMessage(author, channel)
    message.attachments = [DummyAttachment("a.png"), DummyAttachment("b.png")]
    message.stickers = [DummyAttachment("s.png")]
    logger = MessageLogger(message=message)

    attachments = await logger._save_attachments()  # noqa: SLF001
    stickers = await logger._save_stickers()  # noqa: SLF001

    assert attachments == ["a.png", "b.png"]
    assert stickers == ["s.png"]
