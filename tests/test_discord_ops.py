"""Tests for DiscordMessageOps and DiscordStreamOps in utils.discord_ops."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import base64
from typing import TYPE_CHECKING

from PIL import Image
import pytest
from nextcord import File, Embed

from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.discord_ops import (
    _USAGE_FOOTER_RE,
    _DISCORD_MESSAGE_LIMIT,
    DiscordMessageOps,
    DiscordStreamOps,
)

TEST_LLM_MODEL = "test-llm-model"

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class FakeGuild:
    """Minimal guild stub with a stable ID."""

    def __init__(self, guild_id: int = 1) -> None:
        """Initializes the fake guild ID."""
        self.id = guild_id


class FakeReference:
    """Minimal message reference stub."""

    def __init__(self, resolved: FakeMessage) -> None:
        """Initializes the resolved referenced message."""
        self.resolved = resolved


class FakeReply:
    """Provides a fake reply object that records edited content and follow-up replies."""

    def __init__(self) -> None:
        """Initializes the fake reply with empty content and no follow-up chain."""
        self.content: str | None = ""
        self.file: File | None = None
        self.embed: Embed | None = None
        self.replies: list[FakeReply] = []

    async def edit(self, content: str) -> None:
        """Records the replacement content passed to edit."""
        self.content = content

    async def reply(self, content: str) -> FakeReply:
        """Creates and records a follow-up reply in the chain."""
        child = FakeReply()
        child.content = content
        self.replies.append(child)
        return child


class FakeAuthor:
    """Minimal stand-in for `Message.author`."""

    def __init__(self, bot: bool = False, user_id: int = 12345) -> None:
        """Initializes the fake author with stable id and name fields."""
        self.id = user_id
        self.name = "tester"
        self.display_name = "Tester"
        self.mention = f"<@{user_id}>"
        self.bot = bot
        self.display_avatar = SimpleNamespace(url="https://example.test/avatar.png")


class FakeMessage:
    """Provides a fake message object that records created replies."""

    def __init__(self, content: str = "", author: FakeAuthor | None = None) -> None:
        """Initializes the fake message with no recorded replies."""
        self.replies: list[FakeReply] = []
        self.author = author or FakeAuthor()
        self.content = content
        self.embeds: list[Embed] = []
        self.attachments: list[FakeAttachment] = []
        self.stickers: list[FakeAttachment] = []
        self.reference: FakeReference | None = None
        self.guild: FakeGuild | None = FakeGuild()
        self.channel = SimpleNamespace(history=self._history)
        self.id = 987
        self.system_content = ""
        self.added_reactions: list[str] = []
        self.removed_reactions: list[tuple[str, FakeAuthor]] = []

    async def _history(
        self, limit: int, before: FakeMessage, oldest_first: bool
    ) -> AsyncIterator[FakeMessage]:
        """Yields no history by default."""
        if False:
            yield self

    async def reply(
        self, content: str | None, file: File | None = None, embed: Embed | None = None
    ) -> FakeReply:
        """Creates and records a fake reply with the requested content."""
        reply = FakeReply()
        reply.content = content
        reply.file = file
        reply.embed = embed
        self.replies.append(reply)
        return reply

    async def add_reaction(self, emoji: str) -> None:
        """Records a reaction added to the fake message."""
        self.added_reactions.append(emoji)

    async def remove_reaction(self, emoji: str, member: FakeAuthor) -> None:
        """Records a reaction removal from the fake message."""
        self.removed_reactions.append((emoji, member))

    def is_system(self) -> bool:
        """Returns whether the fake message carries system content."""
        return bool(self.system_content)


class FakeAttachment:
    """Minimal Discord attachment or sticker stub."""

    def __init__(
        self,
        filename: str = "file.txt",
        content_type: str | None = "text/plain",
        payload: bytes = b"hello",
        url: str = "https://example.test/file.txt",
    ) -> None:
        """Initializes attachment metadata and payload bytes."""
        self.filename = filename
        self.content_type = content_type
        self._payload = payload
        self.url = url

    async def read(self) -> bytes:
        """Returns the configured attachment bytes."""
        return self._payload


def _png_b64() -> str:
    """Returns a base64-encoded one-pixel PNG."""
    image = Image.new(mode="RGB", size=(1, 1), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG")
    return base64.b64encode(s=buffer.getvalue()).decode(encoding="utf-8")


def _ops(bot_user_id: int = 999) -> tuple[DiscordMessageOps, DiscordStreamOps]:
    """Builds DiscordMessageOps + DiscordStreamOps instances for tests."""
    bot = SimpleNamespace(user=SimpleNamespace(id=bot_user_id, name="bot"))
    msg_ops = DiscordMessageOps(bot=bot, runtime_models=RuntimeModelCatalog())
    stream_ops = DiscordStreamOps(bot=bot, msg_ops=msg_ops)
    return msg_ops, stream_ops


async def _stream_events() -> AsyncIterator[SimpleNamespace]:
    """Yields a minimal streaming completion with token usage."""
    yield SimpleNamespace(type="response.output_text.delta", delta="hello from stream")
    yield SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            model=TEST_LLM_MODEL,
            usage=SimpleNamespace(input_tokens=12, output_tokens=34, output_tokens_details=None),
        ),
    )


async def _stream_events_from(events: list[SimpleNamespace]) -> AsyncIterator[SimpleNamespace]:
    """Yields the provided fake streaming events in order."""
    for event in events:
        yield event


async def test_handle_reaction_returns_emoji_for_chaining() -> None:
    """Verifies handle_reaction returns the applied emoji so callers can chain."""
    msg_ops, _stream_ops = _ops()
    message = FakeMessage()

    first = await msg_ops.handle_reaction(message=message, emoji="🔀")
    assert first == "🔀"
    assert message.added_reactions == ["🔀"]

    second = await msg_ops.handle_reaction(message=message, emoji="🆗", previous=first)
    assert second == "🆗"
    assert message.added_reactions == ["🔀", "🆗"]
    assert message.removed_reactions[0][0] == "🔀"


async def test_handle_streaming_allows_missing_output_token_details(
    economy_isolated_db: None,
) -> None:
    """Regression: LiteLLM may return usage with output_tokens_details=null."""
    del economy_isolated_db
    _msg_ops, stream_ops = _ops()
    message = FakeMessage()

    result = await stream_ops.handle_streaming(responses=_stream_events(), message=message)

    expected = (
        f"hello from stream\n\n-# {TEST_LLM_MODEL} · ⬆ 12 ⬇ 34 · $0.00000000"
        " · 46 虛擬歡樂豆 (+46 虛擬歡樂豆)"
    )
    assert result == expected
    assert message.replies[0].content == result


async def test_handle_streaming_continues_long_reply_as_reply_chain(
    economy_isolated_db: None,
) -> None:
    """Verifies replies over Discord's content limit continue as a reply chain."""
    del economy_isolated_db
    _msg_ops, stream_ops = _ops()
    message = FakeMessage(content="<@999> explain how long Discord replies are handled")
    body = "x" * 4500

    result = await stream_ops.handle_streaming(
        responses=_stream_events_from(
            events=[
                SimpleNamespace(type="response.output_text.delta", delta=body),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        model=TEST_LLM_MODEL,
                        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
                    ),
                ),
            ]
        ),
        message=message,
    )

    usage_footer = (
        f"\n\n-# {TEST_LLM_MODEL} · ⬆ 1 ⬇ 2 · $0.00000000 · 3 虛擬歡樂豆 (+3 虛擬歡樂豆)"
    )
    assert result == f"{body}{usage_footer}"

    parent = message.replies[0]
    assert parent.content == body[:_DISCORD_MESSAGE_LIMIT]

    first_follow_up = parent.replies[0]
    assert first_follow_up.content == body[_DISCORD_MESSAGE_LIMIT : _DISCORD_MESSAGE_LIMIT * 2]

    second_follow_up = first_follow_up.replies[0]
    assert second_follow_up.content == f"{body[_DISCORD_MESSAGE_LIMIT * 2 :]}{usage_footer}"
    assert second_follow_up.replies == []

    chain_chunks = [parent.content, first_follow_up.content, second_follow_up.content]
    assert all(len(chunk) <= _DISCORD_MESSAGE_LIMIT for chunk in chain_chunks)


async def test_handle_streaming_marks_web_search_from_call_event(
    economy_isolated_db: None,
) -> None:
    """Verifies native web_search_call events trigger the web reaction."""
    del economy_isolated_db
    _msg_ops, stream_ops = _ops()
    message = FakeMessage()

    await stream_ops.handle_streaming(
        responses=_stream_events_from(
            events=[
                SimpleNamespace(type="response.output_text.delta", delta="answer"),
                SimpleNamespace(type="response.web_search_call.completed"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        model=TEST_LLM_MODEL,
                        usage=SimpleNamespace(input_tokens=12, output_tokens=34),
                    ),
                ),
            ]
        ),
        message=message,
    )

    assert message.added_reactions == ["🌐"]


async def test_handle_streaming_marks_web_search_from_annotation(
    economy_isolated_db: None,
) -> None:
    """Verifies any annotation event also triggers the web reaction.

    LiteLLM-proxied Gemini grounds via url_citation annotations without
    emitting web_search_call.* events, so annotation alone is treated as
    a search signal.
    """
    del economy_isolated_db
    _msg_ops, stream_ops = _ops()
    message = FakeMessage()

    await stream_ops.handle_streaming(
        responses=_stream_events_from(
            events=[
                SimpleNamespace(type="response.output_text.delta", delta="grounded answer"),
                SimpleNamespace(
                    type="response.output_text.annotation.added",
                    annotation={"type": "url_citation", "url": "https://example.com/article"},
                ),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        model=TEST_LLM_MODEL,
                        usage=SimpleNamespace(input_tokens=12, output_tokens=34),
                    ),
                ),
            ]
        ),
        message=message,
    )

    assert message.added_reactions == ["🌐"]


async def test_message_content_and_attachment_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies prompt cleanup, embed extraction, and attachment conversion."""
    msg_ops, _stream_ops = _ops()
    embed = Embed(title="Title", description="Body")
    embed.set_author(name="Author")
    embed.add_field(name="Field", value="Value")
    embed.set_footer(text="Footer")

    assert await msg_ops.get_user_prompt(content="hi <@999>") == "hi"
    assert "Author" in msg_ops.extract_embed_text(embeds=[embed])

    bot_message = FakeMessage(
        content="answer\n\n-# model · ⬆ 1 ⬇ 2 · $0.0 · +3",
        author=FakeAuthor(bot=True, user_id=999),
    )
    assert await msg_ops.get_cleaned_content(message=bot_message) == "answer"
    assert _USAGE_FOOTER_RE.search(string=bot_message.content)

    embed_message = FakeMessage()
    embed_message.embeds = [embed]
    assert "Title" in await msg_ops.get_cleaned_content(message=embed_message)

    system_message = FakeMessage()
    system_message.system_content = "joined"
    assert await msg_ops.get_cleaned_content(message=system_message) == "joined"

    assert msg_ops._required_modality(content_type="video/mp4") == "video"
    assert msg_ops._required_modality(content_type="audio/mpeg") == "audio"
    assert msg_ops._required_modality(content_type="application/pdf") == "image"

    file_part = await msg_ops._attachment_to_part(
        attachment=FakeAttachment(filename="note.txt", content_type="text/plain", payload=b"abc")
    )
    assert file_part is not None
    assert file_part["file_data"] == "data:text/plain;base64,YWJj"

    image_part = await msg_ops._image_to_part(
        source=FakeAttachment(
            filename="pixel.png", content_type="image/png", payload=base64.b64decode(_png_b64())
        )
    )
    assert image_part is not None
    assert image_part["type"] == "input_image"

    monkeypatch.setattr(
        "discordbot.utils.discord_ops.get_supported_modalities", lambda model_name: {"image"}
    )
    message = FakeMessage()
    message.attachments = [
        FakeAttachment(
            filename="pixel.png", content_type="image/png", payload=base64.b64decode(_png_b64())
        ),
        FakeAttachment(filename="clip.mp4", content_type="video/mp4", payload=b"video"),
    ]
    message.stickers = [
        FakeAttachment(
            filename="sticker.png", content_type="image/png", payload=base64.b64decode(_png_b64())
        )
    ]
    img_embed = Embed()
    img_embed.set_image(url="https://example.test/image.png")
    message.embeds = [img_embed]
    monkeypatch.setattr(
        "discordbot.utils.discord_ops.get_pil_image",
        lambda image_file: Image.new(mode="RGB", size=(1, 1), color=(0, 0, 0)),
    )
    parts = await msg_ops.get_attachment_parts(message=message)
    assert [part["type"] for part in parts] == ["input_image", "input_image", "input_image"]


async def test_get_attachment_parts_requires_runtime_models() -> None:
    """Verifies the runtime_models guard raises a clear ValueError."""
    bot = SimpleNamespace(user=SimpleNamespace(id=999, name="bot"))
    msg_ops = DiscordMessageOps(bot=bot)
    with pytest.raises(ValueError, match="runtime_models"):
        await msg_ops.get_attachment_parts(message=FakeMessage())


async def test_process_single_message_role_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies role dispatch (assistant vs user) and attachment shape."""
    msg_ops, _stream_ops = _ops()
    monkeypatch.setattr(
        "discordbot.utils.discord_ops.get_supported_modalities",
        lambda model_name: {"text", "image"},
    )
    bot_msg = FakeMessage(content="bot answer", author=FakeAuthor(bot=True, user_id=999))
    user_msg = FakeMessage(content="hello", author=FakeAuthor(user_id=1))
    with_attachment = FakeMessage(content="see file", author=FakeAuthor(user_id=2))
    with_attachment.attachments = [FakeAttachment(filename="note.txt", content_type="text/plain")]

    bot_processed = await msg_ops.process_single_message(message=bot_msg)
    user_processed = await msg_ops.process_single_message(message=user_msg)
    attachment_processed = await msg_ops.process_single_message(message=with_attachment)
    assert bot_processed["role"] == "assistant"
    assert user_processed["role"] == "user"
    assert attachment_processed["role"] == "user"
    assert isinstance(attachment_processed["content"], list)
