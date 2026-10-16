"""Discord and model doubles for the reply pipeline, and builders for its parts."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import base64
from typing import TYPE_CHECKING, Any, Literal, cast
import asyncio
from datetime import UTC, datetime, timedelta
import itertools

from PIL import Image

# openai 3.x builds its exceptions on httpx2, so a request or response handed to one has to
# come from there. google-genai is still on httpx 0.x, and both live in the environment.
import httpx2
from openai import APIError
import nextcord
from nextcord import File, Embed, Message
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.gen_reply import streaming as streaming_module
from discordbot.typings.memory import MemoryOwner, MemorySection, MemoryDurability
from discordbot.typings.models import RouteClassification, RuntimeModelCatalog
from discordbot.utils.reactions import ReactionStatusChain
from discordbot.typings.timeouts import INTERACTION_DELIVERY_MARGIN_SECONDS
from discordbot.cogs.gen_reply.cog import ReplyGeneratorCogs
from discordbot.cogs.gen_reply.input import MessageInputBuilder
from discordbot.cogs.gen_reply.answer import AnswerTurn
from discordbot.services.memory.facts import mint_fact_id
from discordbot.services.memory.store import (
    GLOBAL_COMPARTMENT,
    write_fact,
    server_scope,
    scope_owner_id,
)
from discordbot.cogs.gen_reply.context import ReplyContext, ReplyContextBuilder
from discordbot.cogs.gen_reply.markers import InlineMarkers
from discordbot.cogs.gen_reply.routing import RouteClassifier
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.pipeline import ReplyPipeline
from discordbot.cogs.gen_reply.streaming import ResponseStreamer
from discordbot.cogs.gen_reply.media_reply import MediaReplyRoutes
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES
from discordbot.cogs.gen_reply.attachment.gemini_file_api import GeminiFileUploader
from discordbot.cogs.gen_reply.attachment.openai_file_api import OpenAIFileUploader

from tests.helpers.memory import make_fact
from tests.helpers.casting import as_bot, as_client, as_message
from tests.helpers.gen_reply import FakeGeminiFiles, FakeGeminiClient, event_stream
from tests.helpers.link_sources import hosting_off_planner

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable, AsyncIterator

    import pytest
    from nextcord import Attachment
    from nextcord.ext import commands
    from openai.types.responses import ResponseStreamEvent
    from openai.types.responses.response_input_param import ResponseInputParam

    from discordbot.cogs.gen_reply.link_sources import LinkContextSource


TEST_LLM_MODEL = "test-llm-model"
FAKE_MESSAGE_CREATED_AT = datetime(2026, 6, 10, 3, 4, 5, tzinfo=UTC)
# Every `FakeMessage` takes the next id, so two messages in one test are never the same message.
_FAKE_MESSAGE_IDS = itertools.count(start=1000)


class FakeGuild:
    """Minimal guild stub with a stable ID, name, and member lookup."""

    def __init__(
        self,
        guild_id: int = 1,
        name: str = "Test Guild",
        members: dict[int, SimpleNamespace] | None = None,
        filesize_limit: int = 25 * 1024 * 1024,
    ) -> None:
        """Initializes the fake guild ID, name, @everyone sentinel, member map, upload limit."""
        self.id = guild_id
        self.name = name
        self.default_role = SimpleNamespace()
        self._members = members or {}
        self.filesize_limit = filesize_limit

    def get_member(self, user_id: int) -> SimpleNamespace | None:
        """Returns a registered member stub for mention-name resolution, else None."""
        return self._members.get(user_id)

    def get_role(self, role_id: int) -> None:
        """No roles are registered in the stub."""
        del role_id

    def get_channel(self, channel_id: int) -> None:
        """No channels are registered in the stub."""
        del channel_id


class FakeChannel:
    """Minimal channel stub: history plus an @everyone view-permission flag."""

    def __init__(self, history: object, view_channel: bool = True) -> None:
        """Initializes the channel stub with its history coroutine and visibility."""
        self.history = history
        self.parent = None
        self.id = 555
        self._view_channel = view_channel
        self.sent: list[FakeReply] = []

    def permissions_for(self, role: object) -> SimpleNamespace:
        """Returns the @everyone permissions for this channel."""
        del role
        return SimpleNamespace(view_channel=self._view_channel)

    async def send(
        self,
        content: str | None = None,
        embed: Embed | None = None,
        file: File | None = None,
        files: list[File] | None = None,
        allowed_mentions: object | None = None,
    ) -> FakeReply:
        """Records an unparented channel send (the deleted-source fallback target)."""
        del allowed_mentions
        sent = FakeReply()
        sent.content = content
        sent.embed = embed
        sent.file = file
        sent.files = files
        self.sent.append(sent)
        return sent


class FakeReference:
    """Minimal message reference stub."""

    def __init__(self, resolved: FakeMessage) -> None:
        """Initializes the resolved referenced message."""
        self.resolved = resolved


class FakeReply:
    """Provides a fake reply object that records edited content and follow-up replies."""

    def __init__(self) -> None:
        """Initializes the fake reply with empty content and no follow-up chain."""
        self.id = 654
        self.content: str | None = ""
        self.file: File | None = None
        self.files: list[File] | None = None
        self.embed: Embed | None = None
        self.replies: list[FakeReply] = []
        self.edits: list[str] = []
        self.deleted = False
        # When set, edit() raises this instead of recording (simulates a deleted reply).
        self.edit_error: Exception | None = None
        # When set, reply() raises this instead of recording (simulates a failed follow-up).
        self.reply_error: Exception | None = None
        # Records the allowed_mentions arg of each edit/reply so tests can prove a media edit /
        # follow-up keeps AllowedMentions.none() (dropping it would re-ping the author).
        self.allowed_mentions_seen: list[object | None] = []

    async def delete(self) -> None:
        """Records that this reply was deleted (e.g. the orphaned persona-base cleanup)."""
        self.deleted = True

    async def edit(  # noqa: PLR0913 -- one keyword per field of `Message.edit` a caller writes
        self,
        content: str | None = None,
        file: File | None = None,
        files: list[File] | None = None,
        embed: Embed | None = None,
        attachments: list[object] | None = None,
        allowed_mentions: object | None = None,
    ) -> None:
        """Records edited content, embed and/or newly attached media (voice clip / inline image)."""
        del attachments
        if self.edit_error is not None:
            raise self.edit_error
        self.allowed_mentions_seen.append(allowed_mentions)
        if content is not None:
            self.content = content
            self.edits.append(content)
        if embed is not None:
            self.embed = embed
        if file is not None:
            self.file = file
        if files is not None:
            self.files = files
            # Convenience for single-attachment assertions (the voice-only common case).
            if len(files) == 1:
                self.file = files[0]

    async def reply(self, content: str, allowed_mentions: object | None = None) -> FakeReply:
        """Creates and records a follow-up reply in the chain."""
        if self.reply_error is not None:
            raise self.reply_error
        self.allowed_mentions_seen.append(allowed_mentions)
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

    def __init__(
        self, content: str = "", author: FakeAuthor | None = None, channel_public: bool = True
    ) -> None:
        """Initializes the fake message with no recorded replies."""
        self.replies: list[FakeReply] = []
        self.author = author or FakeAuthor()
        self.content = content
        self.embeds: list[Embed] = []
        self.attachments: list[FakeAttachment] = []
        self.stickers: list[FakeAttachment] = []
        self.snapshots: list[FakeSnapshot] = []
        self.reference: FakeReference | None = None
        self.guild: FakeGuild | None = FakeGuild()
        self.channel = FakeChannel(history=self._history, view_channel=channel_public)
        self.mentions: list[FakeAuthor] = []
        self.id = next(_FAKE_MESSAGE_IDS)
        self.created_at = FAKE_MESSAGE_CREATED_AT
        self.edited_at: datetime | None = None
        self.system_content = ""
        self.added_reactions: list[str] = []
        self.removed_reactions: list[tuple[str, FakeAuthor]] = []
        # When set, reply() raises this instead of recording (simulates a deleted source).
        self.reply_error: Exception | None = None

    async def _history(self, limit: int, before: FakeMessage) -> AsyncIterator[FakeMessage]:
        """Yields no history by default."""
        if False:
            yield self

    async def reply(
        self,
        content: str | None = None,
        file: File | None = None,
        embed: Embed | None = None,
        files: list[File] | None = None,
        allowed_mentions: object | None = None,
    ) -> FakeReply:
        """Creates and records a fake reply with the requested content."""
        del allowed_mentions
        if self.reply_error is not None:
            raise self.reply_error
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
        attachment_id: int = 555,
    ) -> None:
        """Initializes attachment metadata and payload bytes."""
        self.id = attachment_id
        self.filename = filename
        self.content_type = content_type
        self._payload = payload
        self.url = url
        self.size = len(payload)
        self.read_count = 0

    async def read(self) -> bytes:
        """Returns the configured attachment bytes."""
        self.read_count += 1
        return self._payload


class FakeSnapshot:
    """Minimal stand-in for a `nextcord.MessageSnapshot` (a forwarded message's payload)."""

    def __init__(
        self,
        content: str = "",
        embeds: list[Embed] | None = None,
        attachments: list[FakeAttachment] | None = None,
        sticker_items: list[FakeAttachment] | None = None,
    ) -> None:
        """Initializes the forwarded snapshot's content and media (stickers as sticker_items)."""
        self.content = content
        self.embeds = embeds or []
        self.attachments = attachments or []
        self.sticker_items = sticker_items or []


class FakeResponses:
    """Fake Responses API resource for the triage, prompt-director and streamed reply calls."""

    def __init__(self) -> None:
        """Initializes recorded calls and default outputs."""
        self.create_streams: list[bool] = []
        self.create_models: list[str] = []
        self.create_instructions: list[str] = []
        self.create_inputs: list[ResponseInputParam | str] = []
        self.create_tools: list[list[object] | None] = []
        self.create_reasonings: list[dict[str, str]] = []
        self.parse_models: list[str] = []
        self.parse_instructions: list[str] = []
        self.parse_inputs: list[ResponseInputParam] = []
        self.parse_text_formats: list[type[RouteClassification]] = []
        # What the triage model "answered", re-validated into whichever schema each parse()
        # asks for, so staged picks survive only a call that actually offered candidates.
        self.output_parsed: RouteClassification | None = RouteClassification(decision="QA")
        # Each entry is the event list for one streaming create(), popped in order. An entry
        # may instead be an Exception, which makes that stream raise instead of yielding, as a
        # provider error frame does -- the only way to drive the answer turn's retry end to end.
        self.stream_queue: list[list[SimpleNamespace] | Exception] = []
        # `.output_text` returned by each non-streaming create(); the prompt director reads it.
        # None (the default) leaves it empty so `refine` falls back to the raw prompt.
        self.refine_output_text: str | None = None

    async def create(  # noqa: PLR0913 -- mirrors Responses API create signature
        self,
        model: str,
        instructions: str,
        input: ResponseInputParam | str,  # noqa: A002 -- SDK parameter
        reasoning: dict[str, str],
        service_tier: str,
        extra_headers: dict[str, str],
        stream: bool = False,
        tools: list[object] | None = None,
    ) -> object:
        """Records the call; returns a streamed event iterator or non-stream output."""
        del service_tier, extra_headers
        self.create_reasonings.append(reasoning)
        self.create_models.append(model)
        self.create_instructions.append(instructions)
        self.create_inputs.append(input)
        self.create_streams.append(stream)
        self.create_tools.append(tools)
        if stream:
            events = (
                self.stream_queue.pop(0) if self.stream_queue else list(_default_turn_events())
            )
            if isinstance(events, Exception):
                return _stream_events_then_raise(events=[], error=events)
            return _stream_events_from(events=events)
        output: list[SimpleNamespace] = []
        if self.refine_output_text is not None:
            # The prompt director reads text via `output_text_or_empty`, which aggregates the
            # structured `.output` message parts (mirroring how the real Response derives
            # `.output_text`), so carry the refine text as an output_text content part.
            output = [
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text=self.refine_output_text)],
                )
            ]
        return SimpleNamespace(output=output, output_text=self.refine_output_text)

    async def parse(  # noqa: PLR0913 -- mirrors Responses API parse signature
        self,
        model: str,
        instructions: str,
        input: ResponseInputParam,  # noqa: A002 -- SDK parameter
        text_format: type[RouteClassification],
        reasoning: dict[str, str],
        service_tier: str,
        extra_headers: dict[str, str],
    ) -> SimpleNamespace:
        """Records the call and returns the staged output parsed into the requested schema.

        Parsed the way the SDK does, into whatever `text_format` asked for: a staged
        `RecallRouteClassification` loses its picks when the call asked for the plain schema.
        """
        self.parse_models.append(model)
        self.parse_instructions.append(instructions)
        self.parse_inputs.append(input)
        self.parse_text_formats.append(text_format)
        if self.output_parsed is None:
            return SimpleNamespace(output_parsed=None)
        return SimpleNamespace(
            output_parsed=text_format.model_validate(obj=self.output_parsed.model_dump())
        )


class FakeImages:
    """Fake Images API resource for generation and edit calls."""

    def __init__(self) -> None:
        """Initializes image API call counters."""
        self.generate_calls = 0
        self.edit_calls = 0
        self.generate_prompts: list[str] = []
        self.edit_prompts: list[str] = []

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
        del model, n, response_format, quality, size, extra_headers
        self.generate_calls += 1
        self.generate_prompts.append(prompt)
        png = base64.b64encode(s=_png_bytes()).decode(encoding="utf-8")
        return SimpleNamespace(data=[SimpleNamespace(b64_json=png)])

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
        del image, model, n, response_format, quality, size, extra_headers
        self.edit_calls += 1
        self.edit_prompts.append(prompt)
        png = base64.b64encode(s=_png_bytes()).decode(encoding="utf-8")
        return SimpleNamespace(data=[SimpleNamespace(b64_json=png)])


class FakeGeminiVideoClient(FakeGeminiClient):
    """Fake native Gemini client whose async omni Interactions video API renders one clip.

    `interactions.create` returns a completed interaction carrying one output video uri; the
    Files resource is a `FakeGeminiFiles`, which downloads that uri as fake MP4 bytes and takes
    both the source-video edit upload and the post-generation "watch the video" reply. Records
    each call's `input`, `response_format`, and `generation_config` (mirroring the real
    `create(**body)`) so tests can assert the task, aspect ratio, and reference-image /
    source-video wiring.
    """

    def __init__(self) -> None:
        """Initializes call records and the async-namespace resources."""
        self.create_inputs: list[Any] = []
        self.create_response_formats: list[Any] = []
        self.create_configs: list[Any] = []
        super().__init__(interactions=SimpleNamespace(create=self._interactions_create))

    async def _interactions_create(self, **body: object) -> SimpleNamespace:
        """Records the request body and returns a completed interaction with one output video."""
        self.create_inputs.append(body.get("input"))
        self.create_response_formats.append(body.get("response_format"))
        self.create_configs.append(body.get("generation_config"))
        return SimpleNamespace(
            status="completed",
            output_text=None,
            output_video=SimpleNamespace(
                uri="https://files.test/video", data=None, mime_type="video/mp4"
            ),
        )


class FakeOpenAIFiles:
    """Fake OpenAI Files API resource that records uploads."""

    def __init__(
        self,
        status: str = "uploaded",
        file_id: str = "file-test",
        expires_at: int | None = 4_070_908_800,
    ) -> None:
        """Initializes fake upload output fields."""
        self.status = status
        self.file_id = file_id
        self.expires_at = expires_at
        self.create_calls: list[
            tuple[str, bytes, str, str, dict[str, object], dict[str, object] | None]
        ] = []

    async def create(
        self,
        file: tuple[str, BytesIO, str],
        purpose: str,
        expires_after: dict[str, object],
        extra_body: dict[str, object] | None = None,
    ) -> SimpleNamespace:
        """Records an upload and returns a fake OpenAI file object."""
        filename, data, content_type = file
        self.create_calls.append((
            filename,
            data.read(),
            content_type,
            purpose,
            expires_after,
            extra_body,
        ))
        return SimpleNamespace(
            id=self.file_id, status=self.status, expires_at=self.expires_at, purpose=purpose
        )


class FakeOpenAIClient:
    """Fake OpenAI client exposing the async Files API used by OpenAIFileUploader."""

    def __init__(self, files: FakeOpenAIFiles | None = None) -> None:
        """Initializes the file resource."""
        self.files = files or FakeOpenAIFiles()


class FakeClient:
    """Fake OpenAI client with responses and images resources."""

    def __init__(self) -> None:
        """Initializes fake OpenAI resource objects."""
        self.responses = FakeResponses()
        self.images = FakeImages()


def _png_bytes() -> bytes:
    """Returns a one-pixel PNG."""
    image = Image.new(mode="RGB", size=(1, 1), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG")
    return buffer.getvalue()


def _fake_uploader(files: FakeGeminiFiles | None = None) -> GeminiFileUploader:
    """A GeminiFileUploader whose client is a fake, so the upload path runs against it."""
    client = as_client(fake=FakeGeminiClient(files=files))
    return GeminiFileUploader(gemini_client=lambda: client)


def _fake_openai_uploader(files: FakeOpenAIFiles | None = None) -> OpenAIFileUploader:
    """An OpenAIFileUploader with its lazy client pre-seeded to a fake."""
    uploader = OpenAIFileUploader(model_name=TEST_LLM_MODEL)
    uploader.__dict__["client"] = FakeOpenAIClient(files=files)
    return uploader


def _cog(bot_user_id: int = 999) -> ReplyGeneratorCogs:
    """Builds a ReplyGeneratorCogs over fake clients, its LLMConfig pinned.

    The config is every field's declared default, so a checkout's `.env` cannot decide a test;
    a test about it sets it on the cog. Everything else, the media planner (which the suite
    keeps from hosting), the usage recorder and the attachment handler choice included, still
    reads the environment. The toolkit alone holds a key, whose client is a fake, so a Gemini
    answer model gets the uploading handler a keyed deployment does.
    """
    cog = ReplyGeneratorCogs(
        bot=as_bot(
            fake=SimpleNamespace(
                user=SimpleNamespace(id=bot_user_id, name="bot"), get_cog=lambda name: None
            )
        )
    )
    cog.config = LLMConfig.model_construct()
    cog.__dict__["openai_client"] = FakeClient()
    toolkit = ReplyToolkit(bot=cog.bot, openai_client=cog.openai_client, gemini_api_key="test-key")
    toolkit.__dict__["gemini_client"] = FakeGeminiVideoClient()
    # Seeded into the cached_property's slot, so every path reads this one rather than
    # building a real toolkit against the test deployment's empty credentials.
    cog.__dict__["toolkit"] = toolkit
    return cog


def _install_streamer(
    monkeypatch: pytest.MonkeyPatch,
    reply: str | Exception = "完整回覆",
    memory_notes: tuple[str, ...] = (),
    forget_notes: tuple[str, ...] = (),
    server_memory_notes: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """Swaps `ResponseStreamer` for a double that returns `reply`, or raises it, unstreamed.

    The double accepts whatever the answer path constructs it with, and the returned list gets
    those kwargs once per construction. It carries only what a turn with no research launch and
    no retryable failure reads back (`carries_turn_notices` is the one a failing `consume` needs);
    a test driving research or a retry needs the attributes those paths read, since a missing one
    fails with an AttributeError the reply path's own handler would swallow.
    """
    built: list[dict[str, object]] = []

    class StreamerDouble:
        """Stands in for `ResponseStreamer` without touching Discord or the event stream."""

        carries_turn_notices = False
        content_ever_started = False

        def __init__(self, **kwargs: object) -> None:
            """Records the constructor kwargs and seeds the marker notes."""
            built.append(kwargs)
            self.markers = InlineMarkers(
                cleaned_text="",
                memory_notes=list(memory_notes),
                forget_notes=list(forget_notes),
                server_memory_notes=list(server_memory_notes),
            )

        async def consume(self, responses: object) -> None:
            """Raises the canned failure, if that is what `reply` is."""
            del responses
            if isinstance(reply, Exception):
                raise reply

        async def finalize_reply(self) -> str | Exception:
            """Returns the canned reply, which `consume` already raised if it was a failure."""
            return reply

    monkeypatch.setattr("discordbot.cogs.gen_reply.answer.ResponseStreamer", StreamerDouble)
    return built


def _context_builder(
    cog: ReplyGeneratorCogs, message: Message, toolkit: ReplyToolkit | None = None
) -> ReplyContextBuilder:
    """The context builder `ReplyPipeline` would build for this message."""
    return ReplyContextBuilder(
        toolkit=toolkit or cog.toolkit, surface=TurnSurface.for_message(message=message)
    )


def _classifier(
    cog: ReplyGeneratorCogs, message: Message, toolkit: ReplyToolkit | None = None
) -> RouteClassifier:
    """The route/effort classifier `ReplyPipeline` would build for this message."""
    return RouteClassifier(
        toolkit=toolkit or cog.toolkit,
        message=message,
        inline_image_enabled=cog.config.inline_image_enabled,
    )


def _streamer(message: object, **fields: Any) -> ResponseStreamer:  # noqa: ANN401 -- the streamer's own fields, passed through
    """A streamer answering `message` on the gateway surface `ReplyPipeline` would give it.

    Its media planner never hosts unless `fields` hands it one.
    """
    fields.setdefault("media_delivery", hosting_off_planner())
    return ResponseStreamer(
        message=message,
        surface=TurnSurface.for_message(message=as_message(fake=message)),
        **fields,
    )


async def _stream(
    streamer: ResponseStreamer, responses: AsyncIterator[ResponseStreamEvent]
) -> str:
    """Streams `responses` onto `streamer` and finalizes it, as one attempt that succeeds does."""
    await streamer.consume(responses=responses)
    return await streamer.finalize_reply()


def _answer(
    cog: ReplyGeneratorCogs,
    message: Message,
    toolkit: ReplyToolkit | None = None,
    surface: TurnSurface | None = None,
) -> AnswerTurn:
    """The answer turn `ReplyPipeline` would build for this message."""
    return AnswerTurn(
        config=cog.config,
        media_delivery=cog.media_delivery,
        toolkit=toolkit or cog.toolkit,
        surface=surface or TurnSurface.for_message(message=message),
    )


def _media_routes(
    cog: ReplyGeneratorCogs,
    message: Message,
    toolkit: ReplyToolkit | None = None,
    surface: TurnSurface | None = None,
) -> MediaReplyRoutes:
    """The IMAGE / VIDEO routes `ReplyPipeline` would build for this message."""
    return MediaReplyRoutes(
        answer=_answer(cog=cog, message=message, toolkit=toolkit, surface=surface)
    )


def _expiring_surface(message: Message, seconds_left: float) -> TurnSurface:
    """A `/ask` surface with `seconds_left` of useful time before its token runs out.

    Only `expires_at` is modelled, because the media routes read the interaction for nothing
    else and a route that reaches this bound never gets as far as sending.
    """
    return TurnSurface(
        message=message,
        interaction=cast(
            "nextcord.Interaction[commands.Bot]",
            SimpleNamespace(
                expires_at=nextcord.utils.utcnow()
                + timedelta(seconds=seconds_left + INTERACTION_DELIVERY_MARGIN_SECONDS)
            ),
        ),
    )


def _recorded(cog: ReplyGeneratorCogs) -> FakeClient:
    """Reads the recorder client back off the cog's typed openai_client slot."""
    return cast("FakeClient", cog.openai_client)


def _recorded_video(cog: ReplyGeneratorCogs) -> FakeGeminiVideoClient:
    """Reads the recorder video client back off the seeded toolkit's gemini_client slot."""
    return cast("FakeGeminiVideoClient", cog.toolkit.gemini_client)


def _config_stub(**fields: object) -> LLMConfig:
    """The cog's LLMConfig with `fields` set and every other field at its declared default.

    Never read from the environment. Every inline marker is off unless `fields` turns it on, so
    nothing is appended to the answer instructions a test did not ask for.
    """
    markers_off = {
        "inline_voice_enabled": False,
        "inline_image_enabled": False,
        "inline_music_enabled": False,
        "inline_video_enabled": False,
    }
    assert fields.keys() <= LLMConfig.model_fields.keys(), fields
    return LLMConfig.model_construct().model_copy(update=markers_off | fields)


def _seed_fact(  # noqa: PLR0913 -- one keyword per stored-fact field a test varies
    scope: str,
    text: str,
    compartment: str = GLOBAL_COMPARTMENT,
    section: MemorySection = "preference",
    durability: MemoryDurability = "stable",
    subject_id: int | None = None,
) -> None:
    """Seeds one stored fact, stamping everything consolidation owns.

    Memory is one fact per file, so a test states the body it wants injected and the
    compartment it must be readable from; the id, the node type and the owner follow from those
    exactly as the pipeline derives them, and every seeded fact shares one date.
    """
    owner_id = scope_owner_id(scope=scope)
    write_fact(
        scope=scope,
        fact=make_fact(
            owner=MemoryOwner(owner_id=owner_id, owner_name=f"U{owner_id} (u{owner_id})"),
            fact_id=mint_fact_id(compartment=compartment, summary=text),
            summary=text,
            section=section,
            durability=durability,
            text=text,
            compartment=compartment,
            subject_id=subject_id,
        ),
    )


def _seed_alias(subject_id: int, text: str) -> None:
    """Seeds one `## 成員稱呼` row of guild 1's server memory, naming `subject_id`."""
    _seed_fact(
        scope=server_scope(server_id=1),
        text=text,
        section="member_alias",
        durability="permanent",
        subject_id=subject_id,
    )


def _att(
    filename: str = "file.txt", content_type: str | None = "text/plain", payload: bytes = b"hello"
) -> Attachment:
    """Builds a FakeAttachment viewed as the nextcord Attachment a renderer expects."""
    return cast(
        "Attachment", FakeAttachment(filename=filename, content_type=content_type, payload=payload)
    )


async def _route(cog: ReplyGeneratorCogs, message: FakeMessage) -> RouteClassification:
    """Classifies a message after building the shared text-only reference/current parts."""
    msg = as_message(fake=message)
    reference_messages, current_message = await _context_builder(
        cog=cog, message=msg
    ).render_parts(text_only=True)
    return await _classifier(cog=cog, message=msg).classify(
        reference_messages=reference_messages,
        current_message=current_message,
        recall_candidates={},
        server_memory_block=None,
    )


async def _run_pipeline(
    cog: ReplyGeneratorCogs, message: FakeMessage, surface: TurnSurface | None = None
) -> None:
    """Runs one whole turn through `ReplyPipeline`, route call included, and lets it raise.

    Built directly rather than through `on_message`, whose failure notice would swallow a
    broken turn; reactions are off because no assertion here reads them.
    """
    msg = as_message(fake=message)
    await ReplyPipeline(
        config=cog.config,
        media_delivery=cog.media_delivery,
        usage_recorder=cog.usage_recorder,
        toolkit=cog.toolkit,
        surface=surface or TurnSurface.for_message(message=msg),
        user_prompt=message.content,
        reactions=ReactionStatusChain(message=msg, bot_user=cog.bot.user, enabled=False),
    ).run()


def _classify_stub(
    route: RouteClassification | Exception,
) -> Callable[..., Awaitable[RouteClassification]]:
    """A `RouteClassifier.classify` that returns `route`, or raises it, after one yield.

    The yield stands in for the real call's network round trip, which is what lets the
    speculative build start before the route is known.
    """

    async def classify(self: RouteClassifier, **kwargs: object) -> RouteClassification:
        """Answers every message with the staged route."""
        del self, kwargs
        await asyncio.sleep(0)
        if isinstance(route, Exception):
            raise route
        return route

    return classify


def _build_stub(context: ReplyContext) -> Callable[..., Awaitable[ReplyContext]]:
    """A `ReplyContextBuilder.build` that returns `context` at once, off memory and history."""

    async def build(self: ReplyContextBuilder, **kwargs: object) -> ReplyContext:
        """Hands back the staged context."""
        del self, kwargs
        return context

    return build


def _assert_runtime_time_context(instructions: str, system_prompt: str) -> None:
    """Verifies that per-request time context follows the base instructions, last."""
    assert instructions.startswith(system_prompt)
    assert "* Treat `message_created_at_asia_taipei` as now for this reply." in instructions
    assert instructions.endswith("* `message_created_at_asia_taipei`: 2026-06-10T11:04:05+08:00")


def _stream_events() -> AsyncIterator[ResponseStreamEvent]:
    """Yields a minimal streaming completion with token usage."""
    return _stream_events_from(
        events=[
            SimpleNamespace(type="response.output_text.delta", delta="hello from stream"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    model=TEST_LLM_MODEL,
                    usage=SimpleNamespace(
                        input_tokens=12, output_tokens=34, output_tokens_details=None
                    ),
                    output=[],
                ),
            ),
        ]
    )


def _stream_events_from(events: list[SimpleNamespace]) -> AsyncIterator[ResponseStreamEvent]:
    """Yields the provided fake streaming events in order.

    Typed as the SDK stream union: production discriminates on the `.type` string, so
    fabricated SimpleNamespace events stand in for the real stream events.
    """
    return cast("AsyncIterator[ResponseStreamEvent]", event_stream(events=events))


def _text_event(delta: str) -> SimpleNamespace:
    """Builds a fake text-delta streaming event."""
    return SimpleNamespace(type="response.output_text.delta", delta=delta)


def _completed_event(input_tokens: int, output_tokens: int) -> SimpleNamespace:
    """Builds a fake response.completed event carrying token usage."""
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            model=TEST_LLM_MODEL,
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
            output=[],
        ),
    )


def _default_turn_events() -> list[SimpleNamespace]:
    """A minimal single-turn stream: one text delta and a completed event."""
    return [_text_event(delta="done"), _completed_event(input_tokens=1, output_tokens=1)]


def _ready_context_task() -> asyncio.Task[ReplyContext]:
    """An empty reply context already built, as the IMAGE and VIDEO handlers are handed one."""

    async def ready() -> ReplyContext:
        """Hands over the empty context at once."""
        return ReplyContext()

    return asyncio.create_task(coro=ready())


def _link_source(name: str) -> LinkContextSource:
    """The live registry entry for one linked-content source, so the tests pin the real wiring."""
    return next(source for source in LINK_CONTEXT_SOURCES if source.name == name)


def _media_builder() -> MessageInputBuilder:
    """A MessageInputBuilder wired with a fake Gemini client for media-path tests."""
    return MessageInputBuilder(
        bot=SimpleNamespace(user=SimpleNamespace(id=999, name="bot")),
        runtime_models=RuntimeModelCatalog(),
        attachment_handler=_fake_uploader(),
    )


def _mid_stream_unavailable() -> APIError:
    """The exact exception a Vertex 503 reaches the bot as, through LiteLLM and openai."""
    return APIError(
        message="litellm.MidStreamFallbackError: litellm.ServiceUnavailableError: ...",
        request=httpx2.Request(method="POST", url="http://proxy/v1/responses"),
        body={"message": "high demand", "type": "None", "param": "None", "code": "503"},
    )


def _stream_events_then_raise(
    events: list[SimpleNamespace], error: Exception, pause: float = 0.0
) -> AsyncIterator[ResponseStreamEvent]:
    """Yields the given events and then dies, as a stream carrying an SSE error frame does.

    `pause` holds the stream open before the raise, which is the only way to let the preview
    editor get a tick in on an attempt that then fails.
    """

    async def _iter() -> AsyncIterator[SimpleNamespace]:
        for event in events:
            yield event
        if pause:
            await asyncio.sleep(pause)
        raise error

    return cast("AsyncIterator[ResponseStreamEvent]", _iter())


def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes the answer retry sleepless.

    Both halves are needed: the jitter is a `wait_random` added on top of the fixed interval,
    independent of it, so zeroing the interval alone still sleeps up to a second per attempt
    and the test only looks instant.
    """
    monkeypatch.setattr(streaming_module, "ANSWER_RETRY_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(streaming_module, "ANSWER_RETRY_JITTER_SECONDS", 0.0)


def _stalled(call: Callable[..., Awaitable[object]]) -> Callable[..., Awaitable[object]]:
    """Wraps a fake provider call so it answers only after outlasting any bound under test.

    It does answer, and successfully, so a test can tell a call that was given up on from one
    that was waited out.
    """

    async def stalled(**kwargs: object) -> object:
        """Sleeps past the bound, then answers as the wrapped call does."""
        await asyncio.sleep(5)
        return await call(**kwargs)

    return stalled


def _image_post(index: int, count: int) -> FakeMessage:
    """A history message carrying `count` image attachments with distinct ids."""
    message = FakeMessage(content=f"post {index}", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(
            filename=f"{index}-{n}.png", content_type="image/png", attachment_id=index * 100 + n
        )
        for n in range(count)
    ]
    return message


_LINK_SOURCES = [source.name for source in LINK_CONTEXT_SOURCES]


_ROUTE_REFERENCE = [EasyInputMessageParam(role="user", content="parent (p) [id: 7]: 原訊息")]
_ROUTE_CURRENT = [EasyInputMessageParam(role="user", content="Tester (tester) [id: 1]: 李董在嗎")]
