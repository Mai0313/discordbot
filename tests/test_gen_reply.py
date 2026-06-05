"""Tests for AI reply routing, attachment handling, streaming, and regeneration."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import base64
from typing import TYPE_CHECKING, Literal
from datetime import UTC, datetime

from PIL import Image
import pytest
from nextcord import File, Embed

from discordbot.cogs.gen_reply import ReplyGeneratorCogs
from discordbot.typings.config import MemoryConfig
from discordbot.typings.models import ModelSettings, RouteDecision, RuntimeModelCatalog
from discordbot.cogs._memory.store import write_main_memory
from discordbot.cogs._gen_reply.input import USAGE_FOOTER_RE
from discordbot.cogs._gen_reply.streaming import DISCORD_MESSAGE_LIMIT, ResponseStreamer
from discordbot.cogs._gen_reply.exceptions import extract_friendly_error

TEST_LLM_MODEL = "test-llm-model"

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai.types.responses.response_input_param import ResponseInputParam


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
        self.files: list[File] | None = None
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
    """Minimal stand-in for `Message.author` used by the streaming helper."""

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
        self,
        content: str | None,
        file: File | None = None,
        embed: Embed | None = None,
        files: list[File] | None = None,
    ) -> FakeReply:
        """Creates and records a fake reply with the requested content."""
        reply = FakeReply()
        reply.content = content
        reply.file = file
        reply.files = files
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


class FakeResponses:
    """Fake Responses API resource for routing and caption calls."""

    def __init__(self) -> None:
        """Initializes recorded calls and default outputs."""
        self.create_streams: list[bool] = []
        self.create_models: list[str] = []
        self.create_instructions: list[str] = []
        self.create_inputs: list[ResponseInputParam | str] = []
        self.parse_models: list[str] = []
        self.output_text = "caption"
        self.output_parsed = SimpleNamespace(decision="SUMMARY")

    async def create(  # noqa: PLR0913 -- mirrors Responses API create signature
        self,
        model: str,
        instructions: str,
        input: ResponseInputParam | str,  # noqa: A002 -- SDK parameter
        reasoning: dict[str, str],
        service_tier: str,
        extra_headers: dict[str, str],
        extra_body: dict[str, bool],
        stream: bool = False,
        tools: list[dict[str, str | dict[str, str]]] | None = None,
    ) -> SimpleNamespace:
        """Records streaming mode and returns configured output text."""
        del reasoning, service_tier, extra_headers, extra_body, tools
        self.create_models.append(model)
        self.create_instructions.append(instructions)
        self.create_inputs.append(input)
        self.create_streams.append(stream)
        return SimpleNamespace(output_text=self.output_text)

    async def parse(  # noqa: PLR0913 -- mirrors Responses API parse signature
        self,
        model: str,
        instructions: str,
        input: list[dict[str, str | list[dict[str, str]]]],  # noqa: A002 -- SDK parameter
        text_format: type[RouteDecision],
        reasoning: dict[str, str],
        service_tier: str,
        extra_headers: dict[str, str],
        extra_body: dict[str, bool],
    ) -> SimpleNamespace:
        """Records the route model and returns configured parsed output."""
        self.parse_models.append(model)
        return SimpleNamespace(output_parsed=self.output_parsed)


class FakeImages:
    """Fake Images API resource for generation and edit calls."""

    def __init__(self) -> None:
        """Initializes image API call counters."""
        self.generate_calls = 0
        self.edit_calls = 0

    async def generate(  # noqa: PLR0913 -- mirrors Images API generate signature
        self,
        prompt: str,
        model: str,
        n: int,
        response_format: Literal["b64_json"],
        quality: str,
        size: str,
        extra_headers: dict[str, str],
    ) -> SimpleNamespace:
        """Records an image generation call and returns a tiny PNG."""
        self.generate_calls += 1
        return SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64())])

    async def edit(  # noqa: PLR0913 -- mirrors Images API edit signature
        self,
        image: list[bytes],
        prompt: str,
        model: str,
        n: int,
        response_format: Literal["b64_json"],
        quality: str,
        size: str,
        extra_headers: dict[str, str],
    ) -> SimpleNamespace:
        """Records an image edit call and returns a tiny PNG."""
        self.edit_calls += 1
        return SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64())])


class FakeVideos:
    """Fake Videos API resource that completes after one poll."""

    def __init__(self) -> None:
        """Initializes video retrieve call count."""
        self.retrieve_calls = 0

    async def create(
        self, model: str, prompt: str, extra_headers: dict[str, str]
    ) -> SimpleNamespace:
        """Returns an in-progress fake video job."""
        return SimpleNamespace(id="video-1", status="processing")

    async def retrieve(self, video_id: str, extra_headers: dict[str, str]) -> SimpleNamespace:
        """Records a poll and returns the completed fake video job."""
        self.retrieve_calls += 1
        return SimpleNamespace(id="video-1", status="completed")

    async def download_content(
        self, video_id: str, extra_headers: dict[str, str]
    ) -> SimpleNamespace:
        """Returns fake MP4 bytes."""
        return SimpleNamespace(content=b"mp4")


class FakeClient:
    """Fake OpenAI client with responses, images, and videos resources."""

    def __init__(self) -> None:
        """Initializes fake OpenAI resource objects."""
        self.responses = FakeResponses()
        self.images = FakeImages()
        self.videos = FakeVideos()


def _png_b64() -> str:
    """Returns a base64-encoded one-pixel PNG."""
    image = Image.new(mode="RGB", size=(1, 1), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG")
    return base64.b64encode(s=buffer.getvalue()).decode(encoding="utf-8")


def _cog(bot_user_id: int = 999, memory_enabled: bool = False) -> ReplyGeneratorCogs:
    """Builds a ReplyGeneratorCogs instance with a fake client."""
    cog = ReplyGeneratorCogs.__new__(ReplyGeneratorCogs)
    cog.bot = SimpleNamespace(user=SimpleNamespace(id=bot_user_id, name="bot"))
    cog.runtime_models = RuntimeModelCatalog()
    cog.memory_config = MemoryConfig(MEMORY_ENABLED=memory_enabled)
    cog.__dict__["client"] = FakeClient()
    return cog


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


async def test_handle_streaming_allows_missing_output_token_details(
    economy_isolated_db: None,
) -> None:
    """Regression: LiteLLM may return usage with output_tokens_details=null."""
    del economy_isolated_db
    message = FakeMessage()

    result = await ResponseStreamer(message=message, responses=_stream_events()).stream()

    # 46 tokens // 100 divisor rounds down to a 0 chat reward.
    expected = (
        f"hello from stream\n\n-# {TEST_LLM_MODEL} · ⬆ 12 ⬇ 34 · $0.00000000"
        " · 0 虛擬歡樂豆 (0 虛擬歡樂豆)"
    )
    assert result == expected
    assert message.replies[0].content == result


async def test_handle_streaming_chat_reward_divided_and_capped(economy_isolated_db: None) -> None:
    """A long reply's chat reward is divided by the token divisor and capped."""
    del economy_isolated_db
    message = FakeMessage()
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="hi"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                model=TEST_LLM_MODEL,
                usage=SimpleNamespace(
                    input_tokens=3_000, output_tokens=3_000, output_tokens_details=None
                ),
            ),
        ),
    ]

    result = await ResponseStreamer(
        message=message, responses=_stream_events_from(events=events)
    ).stream()

    # 6,000 tokens // 100 = 60, capped at 50; the footer shows the credited amount.
    assert "⬆ 3,000 ⬇ 3,000" in result
    assert "· 50 虛擬歡樂豆 (+50 虛擬歡樂豆)" in result


async def test_handle_streaming_continues_long_reply_as_reply_chain(
    economy_isolated_db: None,
) -> None:
    """Verifies replies over Discord's content limit continue as a reply chain."""
    del economy_isolated_db
    cog = _cog()
    message = FakeMessage(content="<@999> explain how long Discord replies are handled")
    body = "x" * 4500

    result = await ResponseStreamer(
        message=message,
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
    ).stream()

    # 3 tokens // 100 divisor rounds down to a 0 chat reward.
    usage_footer = f"\n\n-# {TEST_LLM_MODEL} · ⬆ 1 ⬇ 2 · $0.00000000 · 0 虛擬歡樂豆 (0 虛擬歡樂豆)"
    assert result == f"{body}{usage_footer}"

    parent = message.replies[0]
    assert parent.content == body[:DISCORD_MESSAGE_LIMIT]

    first_follow_up = parent.replies[0]
    assert first_follow_up.content == body[DISCORD_MESSAGE_LIMIT : DISCORD_MESSAGE_LIMIT * 2]

    second_follow_up = first_follow_up.replies[0]
    assert second_follow_up.content == f"{body[DISCORD_MESSAGE_LIMIT * 2 :]}{usage_footer}"
    assert second_follow_up.replies == []

    chain_chunks = [parent.content, first_follow_up.content, second_follow_up.content]
    assert all(len(chunk) <= DISCORD_MESSAGE_LIMIT for chunk in chain_chunks)
    assert cog.client.responses.create_models == []


async def test_handle_streaming_marks_web_search_from_call_event(
    economy_isolated_db: None,
) -> None:
    """Verifies native web_search_call events trigger the web reaction."""
    del economy_isolated_db
    message = FakeMessage()

    await ResponseStreamer(
        message=message,
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
    ).stream()

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
    message = FakeMessage()

    await ResponseStreamer(
        message=message,
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
    ).stream()

    assert message.added_reactions == ["🌐"]


def test_extract_friendly_error_prefers_nested_provider_message() -> None:
    """Verifies nested provider errors are preferred over wrapper text."""
    raw = """wrapper b'{"error": {"message": "quota exceeded"}}'"""
    assert extract_friendly_error(exc=RuntimeError(raw)) == "quota exceeded"
    assert extract_friendly_error(exc=RuntimeError("plain failure")) == "plain failure"
    assert extract_friendly_error(exc=RuntimeError("bad b'not json'")) == "bad b'not json'"


async def test_gen_reply_message_content_and_attachment_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies prompt cleanup, embed extraction, and attachment conversion."""
    cog = _cog()
    embed = Embed(title="Title", description="Body")
    embed.set_author(name="Author")
    embed.add_field(name="Field", value="Value")
    embed.set_footer(text="Footer")

    assert await cog.input_builder.get_user_prompt(content="hi <@999>") == "hi"
    assert "Author" in cog.input_builder.extract_embed_text(embeds=[embed])

    bot_message = FakeMessage(
        content="answer\n\n-# model · ⬆ 1 ⬇ 2 · $0.0 · +3",
        author=FakeAuthor(bot=True, user_id=999),
    )
    assert await cog.input_builder.get_cleaned_content(message=bot_message) == "answer"
    assert USAGE_FOOTER_RE.search(string=bot_message.content)

    embed_message = FakeMessage()
    embed_message.embeds = [embed]
    assert "Title" in await cog.input_builder.get_cleaned_content(message=embed_message)

    system_message = FakeMessage()
    system_message.system_content = "joined"
    assert await cog.input_builder.get_cleaned_content(message=system_message) == "joined"

    assert cog.input_builder.required_modality(content_type="video/mp4") == "video"
    assert cog.input_builder.required_modality(content_type="audio/mpeg") == "audio"
    assert cog.input_builder.required_modality(content_type="application/pdf") == "image"

    file_part = await cog.input_builder.attachment_to_part(
        attachment=FakeAttachment(filename="note.txt", content_type="text/plain", payload=b"abc")
    )
    assert file_part is not None
    assert file_part["file_data"] == "data:text/plain;base64,YWJj"

    image_part = await cog.input_builder.image_to_part(
        source=FakeAttachment(
            filename="pixel.png", content_type="image/png", payload=base64.b64decode(_png_b64())
        )
    )
    assert image_part is not None
    assert image_part["type"] == "input_image"

    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
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
        "discordbot.cogs._gen_reply.input.get_image_data", lambda image_file: _png_b64()
    )
    parts = await cog.input_builder.get_attachment_parts(message=message)
    assert [part["type"] for part in parts] == ["input_image", "input_image", "input_image"]


async def test_gen_reply_processes_history_reference_and_current_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies message processing for history, references, and current prompts."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.input.get_supported_modalities",
        lambda model_name: {"text", "image"},
    )
    bot_msg = FakeMessage(content="bot answer", author=FakeAuthor(bot=True, user_id=999))
    user_msg = FakeMessage(content="hello", author=FakeAuthor(user_id=1))
    with_attachment = FakeMessage(content="see file", author=FakeAuthor(user_id=2))
    with_attachment.attachments = [FakeAttachment(filename="note.txt", content_type="text/plain")]

    bot_processed = await cog.input_builder.process_single_message(message=bot_msg)
    user_processed = await cog.input_builder.process_single_message(message=user_msg)
    attachment_processed = await cog.input_builder.process_single_message(message=with_attachment)
    assert bot_processed["role"] == "assistant"
    assert user_processed["role"] == "user"
    assert attachment_processed["role"] == "user"
    assert isinstance(attachment_processed["content"], list)

    async def fake_history(
        limit: int, before: FakeMessage, oldest_first: bool
    ) -> AsyncIterator[FakeMessage]:
        """Yields two messages for history assembly."""
        yield user_msg
        yield bot_msg

    current = FakeMessage(content="current", author=FakeAuthor(user_id=3))
    current.channel = SimpleNamespace(history=fake_history)
    history = await cog._get_history_message(message=current, limit=30)
    assert len(history) == 3
    assert history[0]["role"] == "system"

    parent = FakeMessage(content="parent", author=FakeAuthor(user_id=4))
    grandparent = FakeMessage(content="grandparent", author=FakeAuthor(user_id=5))
    parent.id = 988
    grandparent.id = 989
    parent.reference = FakeReference(resolved=grandparent)
    current.reference = FakeReference(resolved=parent)
    monkeypatch.setattr("discordbot.cogs.gen_reply.Message", FakeMessage)
    reference = await cog._get_reference_message(message=current)
    assert len(reference) == 4
    assert reference[0]["role"] == "system"
    assert len(await cog._get_current_message(message=current)) == 2


async def test_gen_reply_routes_and_handlers_without_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies route, video, image, and slow-reply handlers using fake APIs."""
    cog = _cog()
    message = FakeMessage(content="make a summary", author=FakeAuthor(user_id=1))
    assert await cog._route_message(message=message) == "SUMMARY"
    assert cog.client.responses.parse_models[0] == cog.runtime_models.fast_model.name

    async def fake_sleep(delay: float) -> None:
        """Skips video polling delay."""

    monkeypatch.setattr("discordbot.cogs.gen_reply.asyncio.sleep", fake_sleep)
    await cog._handle_video_reply(message=message, user_prompt="video")
    assert len(message.replies) == 1

    await cog._handle_image_reply(message=message, user_prompt="image")
    assert cog.client.images.generate_calls
    assert isinstance(message.replies[-1].content, str)
    assert message.replies[-1].content.startswith("<@1> caption")

    streamed: list[FakeMessage] = []

    class FakeResponder:
        """Records the message handed to the streaming responder."""

        def __init__(self, message: FakeMessage, responses: SimpleNamespace) -> None:
            """Stores the streaming inputs."""
            self.message = message
            self.responses = responses

        async def stream(self) -> str:
            """Records the message and returns placeholder content."""
            streamed.append(self.message)
            return "done"

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    await cog._handle_message_reply(message=message, system_prompt="system", history_limit=2)
    assert cog.client.responses.create_streams[-1] is True
    assert streamed[-1] is message


@pytest.mark.parametrize(
    argnames=("route", "expected_call"),
    argvalues=[
        ("IMAGE", "_handle_image_reply"),
        ("VIDEO", "_handle_video_reply"),
        ("SUMMARY", "_handle_message_reply"),
        ("QA", "_handle_message_reply"),
    ],
)
async def test_gen_reply_on_message_dispatches_routes(
    monkeypatch: pytest.MonkeyPatch, route: str, expected_call: str
) -> None:
    """Verifies on_message dispatches each route to the expected handler."""
    cog = _cog()
    calls: list[str] = []

    async def fake_route(message: FakeMessage) -> str:
        """Returns the parametrized route."""
        return route

    async def fake_reaction(
        message: FakeMessage, bot_user: object, emoji: str, previous: str | None = None
    ) -> str:
        """Records reaction state transitions."""
        calls.append(f"reaction:{emoji}")
        return emoji

    async def fake_image_handler(message: FakeMessage, user_prompt: str) -> None:
        """Records image handler dispatch."""
        calls.append("_handle_image_reply")

    async def fake_video_handler(message: FakeMessage, user_prompt: str) -> None:
        """Records video handler dispatch."""
        calls.append("_handle_video_reply")

    memory_flags: list[bool] = []

    async def fake_message_handler(
        message: FakeMessage, system_prompt: str, history_limit: int, memory_enabled: bool = True
    ) -> None:
        """Records slow message handler dispatch."""
        calls.append("_handle_message_reply")
        memory_flags.append(memory_enabled)

    monkeypatch.setattr(cog, "_route_message", fake_route)
    monkeypatch.setattr("discordbot.cogs.gen_reply.update_reaction", fake_reaction)
    monkeypatch.setattr(cog, "_handle_image_reply", fake_image_handler)
    monkeypatch.setattr(cog, "_handle_video_reply", fake_video_handler)
    monkeypatch.setattr(cog, "_handle_message_reply", fake_message_handler)

    message = FakeMessage(content="<@999> hello", author=FakeAuthor(user_id=1))
    await cog.on_message(message=message)
    assert expected_call in calls
    assert calls[-1] == "reaction:🆗"
    # Summaries opt out of per-user memory; QA keeps the default.
    if route == "SUMMARY":
        assert memory_flags == [False]
    elif route == "QA":
        assert memory_flags == [True]


async def test_gen_reply_on_message_early_returns_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies bot messages, unmentioned guild messages, empty prompts, and errors."""
    cog = _cog()
    bot_authored = FakeMessage(content="<@999> hi", author=FakeAuthor(bot=True))
    await cog.on_message(message=bot_authored)
    assert bot_authored.replies == []

    unmentioned = FakeMessage(content="hello", author=FakeAuthor(user_id=1))
    await cog.on_message(message=unmentioned)
    assert unmentioned.replies == []

    dm_empty = FakeMessage(content="<@999>", author=FakeAuthor(user_id=1))
    dm_empty.guild = None
    await cog.on_message(message=dm_empty)
    assert dm_empty.replies[0].content == "?"

    async def boom(message: FakeMessage) -> str:
        """Raises to exercise error handling."""
        raise RuntimeError("boom")

    monkeypatch.setattr(cog, "_route_message", boom)
    failed = FakeMessage(content="<@999> fail", author=FakeAuthor(user_id=1))
    await cog.on_message(message=failed)
    assert failed.replies[0].content is None


def test_model_settings_and_config_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies model properties and provider-specific tool dispatch."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    catalog = RuntimeModelCatalog()
    cog = ReplyGeneratorCogs(bot=SimpleNamespace(user=SimpleNamespace(id=999)))
    assert cog.runtime_models.fast_model == catalog.fast_model
    assert isinstance(catalog.fast_model, ModelSettings)
    assert catalog.image_model.name.endswith("image-preview")
    assert catalog.video_model.name.startswith("veo")
    assert catalog.slow_model.effort == "high"
    assert ModelSettings(name="gemini-test").tools == [{"googleSearch": {}}, {"urlContext": {}}]
    assert ModelSettings(name="claude-test").tools[0]["name"] == "web_search"
    assert ModelSettings(name="openai-test").tools == [{"type": "web_search"}]


def test_runtime_model_catalog_dispatches_slow_model_by_peak_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies slow-model peak-hour and off-peak dispatch."""

    def model_snapshot_at(now: datetime) -> tuple[ModelSettings, bool, bool]:
        """Returns peak-sensitive model settings with the catalog clock pinned to `now`."""

        def fixed_now(tz: object) -> datetime:
            """Returns the pinned timestamp."""
            assert tz is UTC
            return now

        monkeypatch.setattr("discordbot.typings.models.datetime", SimpleNamespace(now=fixed_now))
        catalog = RuntimeModelCatalog()
        return catalog.slow_model, catalog.is_peak, catalog.model_dump()["is_peak"] is True

    peak_start = model_snapshot_at(now=datetime(year=2026, month=5, day=18, hour=8, tzinfo=UTC))
    peak_end = model_snapshot_at(now=datetime(year=2026, month=5, day=18, hour=16, tzinfo=UTC))
    before_peak = model_snapshot_at(now=datetime(year=2026, month=5, day=18, hour=7, tzinfo=UTC))
    after_peak = model_snapshot_at(now=datetime(year=2026, month=5, day=18, hour=17, tzinfo=UTC))
    weekend = model_snapshot_at(now=datetime(year=2026, month=5, day=23, hour=12, tzinfo=UTC))

    assert peak_start[1:] == (True, True)
    assert peak_end[1:] == (True, True)
    assert before_peak[1:] == (False, False)
    assert after_peak[1:] == (False, False)
    assert weekend[1:] == (False, False)
    assert peak_start[0].name == "gemini-flash-latest"
    assert after_peak[0].name == "gemini-3.5-flash"
    assert peak_start[0] == peak_end[0]
    assert before_peak[0] == after_peak[0] == weekend[0]
    assert peak_start[0] != after_peak[0]
    assert {peak_start[0].effort, after_peak[0].effort} == {"high"}


async def test_handle_message_reply_injects_memory_as_trailing_system_message(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies stored memory rides as a trailing role=system input, not in instructions."""
    cog = _cog(memory_enabled=True)
    write_main_memory(
        user_id=1, content="v1\n\n## 使用者輪廓\n喜歡簡短回覆", identity="Tester (tester) [id: 1]"
    )

    class FakeResponder:
        """Returns a fixed completed reply."""

        def __init__(self, message: FakeMessage, responses: SimpleNamespace) -> None:
            """Stores the streaming inputs."""
            self.message = message
            self.responses = responses

        async def stream(self) -> str:
            """Returns placeholder reply content."""
            return "完整回覆"

    scheduled: list[dict[str, object]] = []

    def fake_schedule(
        user_id: int, message_list: list[object], full_reply: str, extractor: object, identity: str
    ) -> None:
        """Records the scheduled memory update arguments."""
        scheduled.append({
            "user_id": user_id,
            "message_list": message_list,
            "full_reply": full_reply,
            "extractor": extractor,
            "identity": identity,
        })

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", fake_schedule)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await cog._handle_message_reply(message=message, system_prompt="SYS", history_limit=2)

    # Top-level instructions must stay the clean developer-controlled persona.
    instructions = cog.client.responses.create_instructions[-1]
    assert instructions == "SYS"
    assert "喜歡簡短回覆" not in instructions

    # Memory rides as the LAST input item, role=system, carrying the wrapper.
    llm_input = cog.client.responses.create_inputs[-1]
    memory_item = llm_input[-1]
    assert memory_item["role"] == "system"
    memory_text = memory_item["content"][0]["text"]
    assert "喜歡簡短回覆" in memory_text
    assert "Long-term memory" in memory_text

    # The extraction message_list must NOT contain the memory block (no self-feeding).
    scheduled_list = scheduled[0]["message_list"]
    assert isinstance(scheduled_list, list)
    assert "Long-term memory" not in str(scheduled_list)
    assert "喜歡簡短回覆" not in str(scheduled_list)
    assert len(scheduled_list) == len(llm_input) - 1
    assert scheduled[0]["full_reply"] == "完整回覆"
    assert scheduled[0]["extractor"] is cog.memory_extractor
    assert scheduled[0]["identity"] == "Tester (tester) [id: 1]"
    assert cog.memory_extractor.extract_model.name == cog.runtime_models.extract_model.name
    assert cog.memory_extractor.consolidate_model.name == cog.runtime_models.memories_model.name


async def test_handle_message_reply_without_stored_memory_keeps_instructions(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies a memory-less user gets untouched instructions but still schedules."""
    cog = _cog(memory_enabled=True)

    class FakeResponder:
        """Returns a fixed completed reply."""

        def __init__(self, message: FakeMessage, responses: SimpleNamespace) -> None:
            """Stores the streaming inputs."""
            self.message = message
            self.responses = responses

        async def stream(self) -> str:
            """Returns placeholder reply content."""
            return "回覆"

    scheduled: list[int] = []

    def fake_schedule(
        user_id: int, message_list: list[object], full_reply: str, extractor: object, identity: str
    ) -> None:
        """Records that a memory update was scheduled."""
        scheduled.append(user_id)

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", fake_schedule)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await cog._handle_message_reply(message=message, system_prompt="SYS", history_limit=2)

    assert cog.client.responses.create_instructions[-1] == "SYS"
    assert scheduled == [1]


async def test_handle_message_reply_disabled_memory_skips_pipeline(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies the kill switch bypasses injection and extraction entirely."""
    cog = _cog(memory_enabled=False)
    write_main_memory(
        user_id=1, content="v1\n\n## 使用者輪廓\n不該被注入", identity="Tester (tester) [id: 1]"
    )

    class FakeResponder:
        """Returns a fixed completed reply."""

        def __init__(self, message: FakeMessage, responses: SimpleNamespace) -> None:
            """Stores the streaming inputs."""
            self.message = message
            self.responses = responses

        async def stream(self) -> str:
            """Returns placeholder reply content."""
            return "回覆"

    scheduled: list[int] = []

    def fake_schedule(
        user_id: int, message_list: list[object], full_reply: str, extractor: object, identity: str
    ) -> None:
        """Records that a memory update was scheduled."""
        scheduled.append(user_id)

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", fake_schedule)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await cog._handle_message_reply(message=message, system_prompt="SYS", history_limit=2)

    assert cog.client.responses.create_instructions[-1] == "SYS"
    assert "不該被注入" not in cog.client.responses.create_instructions[-1]
    assert scheduled == []


async def test_handle_message_reply_memory_disabled_arg_skips_pipeline(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies memory_enabled=False (summary route) bypasses injection and extraction."""
    cog = _cog(memory_enabled=True)
    write_main_memory(
        user_id=1, content="v1\n\n## 使用者輪廓\n不該被注入", identity="Tester (tester) [id: 1]"
    )

    class FakeResponder:
        """Returns a fixed completed reply."""

        def __init__(self, message: FakeMessage, responses: SimpleNamespace) -> None:
            """Stores the streaming inputs."""
            self.message = message
            self.responses = responses

        async def stream(self) -> str:
            """Returns placeholder reply content."""
            return "回覆"

    scheduled: list[int] = []

    def fake_schedule(
        user_id: int, message_list: list[object], full_reply: str, extractor: object, identity: str
    ) -> None:
        """Records that a memory update was scheduled."""
        scheduled.append(user_id)

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", fake_schedule)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await cog._handle_message_reply(
        message=message, system_prompt="SYS", history_limit=2, memory_enabled=False
    )

    assert "不該被注入" not in str(cog.client.responses.create_inputs[-1])
    assert scheduled == []


async def test_process_single_message_neutralizes_spoofed_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies id-prefix lookalikes in display names cannot forge authorship."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.input.get_supported_modalities", lambda model_name: {"text"}
    )
    author = FakeAuthor(user_id=555)
    author.display_name = "Mallory (mallory) [id: 1]:"
    message = FakeMessage(content="假冒攻擊", author=author)

    processed = await cog.input_builder.process_single_message(message=message)
    rendered = processed["content"]
    assert isinstance(rendered, str)
    assert "[id: 1]" not in rendered
    assert "[id: 555]:" in rendered

    current_messages = await cog._get_current_message(message=message)
    separator = current_messages[0]["content"]
    assert isinstance(separator, list)
    assert "[id: 1]" not in separator[0]["text"]
