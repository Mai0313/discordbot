"""Tests for AI reply routing, image/video handlers, and on_message dispatch."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import base64
from typing import TYPE_CHECKING, Literal
from datetime import UTC, datetime

from PIL import Image
import pytest

from discordbot.cogs.gen_reply import ReplyGeneratorCogs
from discordbot.typings.models import ModelSettings, RouteDecision, RuntimeModelCatalog
from discordbot.utils.discord_ops import DiscordStreamOps, DiscordMessageOps
from discordbot.cogs._gen_reply.exceptions import extract_friendly_error

TEST_LLM_MODEL = "test-llm-model"

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nextcord import File, Embed
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


def _cog(bot_user_id: int = 999) -> ReplyGeneratorCogs:
    """Builds a ReplyGeneratorCogs instance with a fake client."""
    cog = ReplyGeneratorCogs.__new__(ReplyGeneratorCogs)
    cog.bot = SimpleNamespace(user=SimpleNamespace(id=bot_user_id, name="bot"))
    cog.runtime_models = RuntimeModelCatalog()
    cog.__dict__["client"] = FakeClient()
    cog.discord_messages = DiscordMessageOps(bot=cog.bot, runtime_models=cog.runtime_models)
    cog.discord_stream = DiscordStreamOps(bot=cog.bot, msg_ops=cog.discord_messages)
    return cog


def test_extract_friendly_error_prefers_nested_provider_message() -> None:
    """Verifies nested provider errors are preferred over wrapper text."""
    raw = """wrapper b'{"error": {"message": "quota exceeded"}}'"""
    assert extract_friendly_error(exc=RuntimeError(raw)) == "quota exceeded"
    assert extract_friendly_error(exc=RuntimeError("plain failure")) == "plain failure"
    assert extract_friendly_error(exc=RuntimeError("bad b'not json'")) == "bad b'not json'"


async def test_gen_reply_assembles_history_reference_and_current_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies history, reference chain, and current-message orchestration on the cog."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.utils.discord_ops.get_supported_modalities",
        lambda model_name: {"text", "image"},
    )
    bot_msg = FakeMessage(content="bot answer", author=FakeAuthor(bot=True, user_id=999))
    user_msg = FakeMessage(content="hello", author=FakeAuthor(user_id=1))

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

    async def fake_streaming(responses: SimpleNamespace, message: FakeMessage) -> str:
        """Records the message passed to streaming."""
        streamed.append(message)
        return "done"

    streamed: list[FakeMessage] = []
    monkeypatch.setattr(cog, "discord_stream", SimpleNamespace(handle_streaming=fake_streaming))
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

    async def fake_reaction(message: FakeMessage, emoji: str, previous: str | None = None) -> str:
        """Records reaction state transitions and returns the emoji for chaining."""
        calls.append(f"reaction:{emoji}")
        return emoji

    async def fake_image_handler(message: FakeMessage, user_prompt: str) -> None:
        """Records image handler dispatch."""
        calls.append("_handle_image_reply")

    async def fake_video_handler(message: FakeMessage, user_prompt: str) -> None:
        """Records video handler dispatch."""
        calls.append("_handle_video_reply")

    async def fake_message_handler(
        message: FakeMessage, system_prompt: str, history_limit: int
    ) -> None:
        """Records slow message handler dispatch."""
        calls.append("_handle_message_reply")

    monkeypatch.setattr(cog, "_route_message", fake_route)
    monkeypatch.setattr(
        cog,
        "discord_messages",
        SimpleNamespace(
            get_user_prompt=cog.discord_messages.get_user_prompt,
            handle_reaction=fake_reaction,
        ),
    )
    monkeypatch.setattr(cog, "_handle_image_reply", fake_image_handler)
    monkeypatch.setattr(cog, "_handle_video_reply", fake_video_handler)
    monkeypatch.setattr(cog, "_handle_message_reply", fake_message_handler)

    message = FakeMessage(content="<@999> hello", author=FakeAuthor(user_id=1))
    await cog.on_message(message=message)
    assert expected_call in calls
    assert calls[-1] == "reaction:🆗"


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
