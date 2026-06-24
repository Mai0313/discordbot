"""Tests for AI reply routing, attachment handling, streaming, and regeneration."""

from __future__ import annotations

from io import BytesIO
import json
from types import SimpleNamespace
import base64
from typing import TYPE_CHECKING, Literal
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from PIL import Image
import httpx
from openai import APITimeoutError
import pytest
import nextcord
from nextcord import File, Embed
from google.genai.types import FileState
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.cogs._memory import database as memory_db
from discordbot.cogs.gen_reply import (
    ReplyGeneratorCogs,
    _discard_task,
    _find_youtube_url,
    _can_launch_research,
    _build_runtime_instructions,
)
from discordbot.typings.models import (
    EffortGrade,
    ModelSettings,
    RouteClassification,
    RuntimeModelCatalog,
)
from discordbot.utils.reactions import ReactionStatusChain
from discordbot.cogs._memory.store import user_scope, server_scope, write_main_memory
from discordbot.cogs._gen_reply.input import USAGE_FOOTER_RE, MessageInputBuilder
from discordbot.cogs._gen_reply.voice import (
    VOICE_TIMEOUT_SECONDS,
    VoiceClip,
    VoiceOutcome,
    VoiceSynthesizer,
    speechify_discord_markup,
)
from discordbot.cogs._gen_reply.context import ReplyContext
from discordbot.cogs._gen_reply.markers import (
    MAX_INLINE_IMAGES,
    extract_inline_markers,
    scrub_markers_for_preview,
)
from discordbot.cogs._gen_reply.prompts import MEMORY_SELECT_PROMPT
from discordbot.cogs._gen_reply.streaming import DISCORD_MESSAGE_LIMIT, ResponseStreamer
from discordbot.cogs._gen_reply.exceptions import extract_friendly_error
from discordbot.cogs._gen_reply.generation import (
    MusicClip,
    ImageGenerator,
    MusicGenerator,
    music_filename,
)
from discordbot.cogs._gen_reply.memory_tool import (
    NO_STORED_MEMORY,
    parse_user_id_list,
    resolve_user_memories,
    build_memory_allowlist,
    widen_allowlist_with_aliases,
    allowlist_ids_from_server_memory,
)
from discordbot.cogs._memory.server_prompts import SERVER_PHASE1_PROMPT, SERVER_PHASE2_PROMPT
from discordbot.cogs._parse_threads.builder import THREADS_CONTEXT_SEPARATOR
from discordbot.cogs._gen_reply.attachment.base import DEAD_SOURCE_TTL, loggable_cache_key
from discordbot.cogs._gen_reply.attachment.inline import InlineRenderer
from discordbot.cogs._gen_reply.attachment.select import build_attachment_handler
from discordbot.cogs._gen_reply.attachment.gemini_file_api import PendingUpload, GeminiFileUploader
from discordbot.cogs._gen_reply.attachment.openai_file_api import OpenAIFileUploader

from tests.helpers.llm_input import (
    request_index,
    request_input,
    iter_text_blocks,
    tool_names_for_call,
    has_memory_context_block,
    extract_callable_user_ids,
    has_threads_context_block,
    extract_user_memory_blocks,
    extract_server_memory_block,
    extract_threads_context_block,
)

TEST_LLM_MODEL = "test-llm-model"
FAKE_MESSAGE_CREATED_AT = datetime(2026, 6, 10, 3, 4, 5, tzinfo=UTC)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai.types.responses.response_input_param import ResponseInputParam


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
    ) -> FakeReply:
        """Records an unparented channel send (the deleted-source fallback target)."""
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
        self.content: str | None = ""
        self.file: File | None = None
        self.files: list[File] | None = None
        self.embed: Embed | None = None
        self.replies: list[FakeReply] = []
        self.edits: list[str] = []

    async def edit(
        self, content: str | None = None, file: File | None = None, files: list[File] | None = None
    ) -> None:
        """Records edited content and/or newly attached media (voice clip / inline image)."""
        if content is not None:
            self.content = content
            self.edits.append(content)
        if file is not None:
            self.file = file
        if files is not None:
            self.files = files
            # Convenience for single-attachment assertions (the voice-only common case).
            if len(files) == 1:
                self.file = files[0]

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
        self.reference: FakeReference | None = None
        self.guild: FakeGuild | None = FakeGuild()
        self.channel = FakeChannel(history=self._history, view_channel=channel_public)
        self.mentions: list[FakeAuthor] = []
        self.id = 987
        self.created_at = FAKE_MESSAGE_CREATED_AT
        self.edited_at: datetime | None = None
        self.system_content = ""
        self.added_reactions: list[str] = []
        self.removed_reactions: list[tuple[str, FakeAuthor]] = []
        # When set, reply() raises this instead of recording (simulates a deleted source).
        self.reply_error: Exception | None = None

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
        self.read_count = 0

    async def read(self) -> bytes:
        """Returns the configured attachment bytes."""
        self.read_count += 1
        return self._payload


class FakeResponses:
    """Fake Responses API resource for routing, memory selection, and streamed reply calls."""

    def __init__(self) -> None:
        """Initializes recorded calls and default outputs."""
        self.create_streams: list[bool] = []
        self.create_models: list[str] = []
        self.create_instructions: list[str] = []
        self.create_inputs: list[ResponseInputParam | str] = []
        self.create_tools: list[list[object] | None] = []
        self.create_reasonings: list[dict[str, str]] = []
        self.parse_models: list[str] = []
        self.parse_inputs: list[object] = []
        # parse() serves both the route classifier and the effort grader; each picks its
        # own parsed output by the requested text_format.
        self.output_parsed: RouteClassification | None = RouteClassification(decision="SUMMARY")
        self.effort_parsed: EffortGrade | None = EffortGrade(effort="high")
        # Each entry is the event list for one streaming create(); popped in order.
        self.stream_queue: list[list[SimpleNamespace]] = []
        # Each entry is the `.output` item list for one non-streaming (memory selection)
        # create(); popped in order.
        self.select_queue: list[list[SimpleNamespace]] = []
        # `.usage` returned by each non-streaming (memory selection) create().
        self.select_usage: SimpleNamespace | None = None

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
        tools: list[object] | None = None,
    ) -> object:
        """Records the call; returns a streamed event iterator or non-stream output."""
        del service_tier, extra_headers, extra_body
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
            return _stream_events_from(events=events)
        output = self.select_queue.pop(0) if self.select_queue else []
        return SimpleNamespace(output=output, usage=self.select_usage)

    async def parse(  # noqa: PLR0913 -- mirrors Responses API parse signature
        self,
        model: str,
        instructions: str,
        input: list[dict[str, str | list[dict[str, str]]]],  # noqa: A002 -- SDK parameter
        text_format: type[RouteClassification | EffortGrade],
        reasoning: dict[str, str],
        service_tier: str,
        extra_headers: dict[str, str],
        extra_body: dict[str, bool],
    ) -> SimpleNamespace:
        """Records the model and returns the parsed output for the requested schema."""
        self.parse_models.append(model)
        self.parse_inputs.append(input)
        if text_format is EffortGrade:
            return SimpleNamespace(output_parsed=self.effort_parsed)
        return SimpleNamespace(output_parsed=self.output_parsed)


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
        del image, model, n, response_format, quality, size, extra_headers
        self.edit_calls += 1
        self.edit_prompts.append(prompt)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64())])


class FakeGeminiVideoClient:
    """Fake native Gemini client exposing the async Veo generation API.

    `generate_videos` returns an in-progress operation that the first `operations.get` flips
    to done with one generated video; `files.download` returns fake MP4 bytes; `files.upload`
    /`get` return an ACTIVE file for the post-generation "watch the video" reply. Records the
    prompt and the config so tests can assert reference-image wiring.
    """

    def __init__(self) -> None:
        """Initializes call records and the async-namespace resources."""
        self.generate_prompts: list[str] = []
        self.generate_configs: list[object] = []
        self.get_calls = 0
        self.aio = SimpleNamespace(
            models=SimpleNamespace(generate_videos=self._generate_videos),
            operations=SimpleNamespace(get=self._operations_get),
            files=SimpleNamespace(
                download=self._files_download, upload=self._files_upload, get=self._files_get
            ),
        )

    async def _generate_videos(
        self, *, model: str, prompt: str, config: object = None
    ) -> SimpleNamespace:
        """Records the request and returns an in-progress operation."""
        del model
        self.generate_prompts.append(prompt)
        self.generate_configs.append(config)
        return SimpleNamespace(name="op-1", done=False, error=None, response=None)

    async def _operations_get(self, operation: object) -> SimpleNamespace:
        """Records a poll and returns the completed operation with one generated video."""
        del operation
        self.get_calls += 1
        video = SimpleNamespace(uri="https://files.test/video", video_bytes=None)
        return SimpleNamespace(
            name="op-1",
            done=True,
            error=None,
            response=SimpleNamespace(generated_videos=[SimpleNamespace(video=video)]),
        )

    async def _files_download(self, *, file: object) -> bytes:
        """Returns fake MP4 bytes for the completed video."""
        del file
        return b"mp4"

    async def _files_upload(self, *, file: object, config: dict[str, str]) -> SimpleNamespace:
        """Returns an ACTIVE uploaded file for the post-generation video reply."""
        del file, config
        return SimpleNamespace(
            name="files/vid", uri="https://files.test/files/vid", state=FileState.ACTIVE
        )

    async def _files_get(self, *, name: str) -> SimpleNamespace:
        """Returns the ACTIVE uploaded file when the reply polls it."""
        del name
        return SimpleNamespace(
            name="files/vid", uri="https://files.test/files/vid", state=FileState.ACTIVE
        )


class FakeGeminiFiles:
    """Fake Gemini Files API resource that records uploads and returns ACTIVE files.

    `processing_rounds` makes `upload` return a PROCESSING file that flips to ACTIVE
    after that many `get` polls, so the activation poll loop is exercised. A negative
    `final_state` (e.g. FAILED) lets a test drive the failed-processing branch.
    """

    def __init__(
        self,
        processing_rounds: int = 0,
        final_state: FileState = FileState.ACTIVE,
        expiration_time: datetime = datetime(2099, 1, 1, tzinfo=UTC),
    ) -> None:
        """Initializes upload records and the processing-to-active schedule."""
        self.upload_calls: list[tuple[str, str]] = []
        self.processing_rounds = processing_rounds
        self.final_state = final_state
        self.expiration_time = expiration_time
        self._remaining = 0

    def _file(self, name: str, state: FileState) -> SimpleNamespace:
        """Builds a fake uploaded-file object with the URI the answer references."""
        return SimpleNamespace(
            name=name,
            uri=f"https://files.test/{name}",
            state=state,
            error=None,
            expiration_time=self.expiration_time,
        )

    async def upload(self, file: BytesIO, config: dict[str, str]) -> SimpleNamespace:
        """Records an upload and returns a file keyed on its display name."""
        del file
        display_name = config["display_name"]
        self.upload_calls.append((display_name, config["mime_type"]))
        self._remaining = self.processing_rounds
        state = FileState.PROCESSING if self.processing_rounds else self.final_state
        return self._file(name=display_name, state=state)

    async def get(self, name: str) -> SimpleNamespace:
        """Returns the polled file, flipping to the final state once rounds elapse."""
        self._remaining -= 1
        state = FileState.PROCESSING if self._remaining > 0 else self.final_state
        return self._file(name=name, state=state)


class FakeGeminiClient:
    """Fake Gemini client exposing the async Files API used for attachment uploads."""

    def __init__(self, files: FakeGeminiFiles | None = None) -> None:
        """Initializes the async-namespace file resource."""
        self.aio = SimpleNamespace(files=files or FakeGeminiFiles())


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


def _png_b64() -> str:
    """Returns a base64-encoded one-pixel PNG."""
    image = Image.new(mode="RGB", size=(1, 1), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG")
    return base64.b64encode(s=buffer.getvalue()).decode(encoding="utf-8")


def _fake_uploader(files: FakeGeminiFiles | None = None) -> GeminiFileUploader:
    """A GeminiFileUploader with its lazy Gemini client pre-seeded to a fake.

    `gemini_client` is a cached_property, so seeding `__dict__` bypasses the real
    factory and the upload path runs against the fake instead.
    """
    uploader = GeminiFileUploader()
    uploader.__dict__["gemini_client"] = FakeGeminiClient(files=files)
    return uploader


def _fake_openai_uploader(files: FakeOpenAIFiles | None = None) -> OpenAIFileUploader:
    """An OpenAIFileUploader with its lazy client pre-seeded to a fake."""
    uploader = OpenAIFileUploader(model_name=TEST_LLM_MODEL)
    uploader.__dict__["client"] = FakeOpenAIClient(files=files)
    return uploader


def _cog(bot_user_id: int = 999) -> ReplyGeneratorCogs:
    """Builds a ReplyGeneratorCogs instance with a fake client."""
    cog = ReplyGeneratorCogs.__new__(ReplyGeneratorCogs)
    cog.bot = SimpleNamespace(user=SimpleNamespace(id=bot_user_id, name="bot"))
    cog.runtime_models = RuntimeModelCatalog()
    cog.config = LLMConfig()
    cog.__dict__["openai_client"] = FakeClient()
    cog.__dict__["gemini_client"] = FakeGeminiVideoClient()
    handler = cog.input_builder.attachment_handler
    if isinstance(handler, GeminiFileUploader):
        handler.__dict__["gemini_client"] = FakeGeminiClient()
    return cog


async def _route(cog: ReplyGeneratorCogs, message: FakeMessage) -> RouteClassification:
    """Classifies a message after building the shared text-only reference/current parts."""
    reference_messages, current_message = await cog._get_reference_and_current(
        message=message, text_only=True
    )
    return await cog._route_classify(
        message=message, reference_messages=reference_messages, current_message=current_message
    )


async def _grade(cog: ReplyGeneratorCogs, message: FakeMessage) -> EffortGrade:
    """Grades a message's answer effort after building the shared text-only parts."""
    reference_messages, current_message = await cog._get_reference_and_current(
        message=message, text_only=True
    )
    return await cog._grade_effort(
        message=message, reference_messages=reference_messages, current_message=current_message
    )


async def _reply_via_pipeline(  # noqa: PLR0913 -- mirrors _handle_message_reply's signature
    cog: ReplyGeneratorCogs,
    message: FakeMessage,
    system_prompt: str = "SYS",
    history_limit: int = 2,
    memory_enabled: bool = True,
    effort: Literal["low", "medium", "high"] = "high",
) -> None:
    """Drives prepare-context plus answer the way on_message does for the QA route."""
    parts_task = asyncio.create_task(coro=cog._get_reference_and_current(message=message))
    text_parts = await cog._get_reference_and_current(message=message, text_only=True)
    route_done = asyncio.Event()
    route_done.set()
    context = await cog._prepare_reply_context(
        message=message,
        history_limit=history_limit,
        memory_enabled=memory_enabled,
        parts_task=parts_task,
        text_parts=text_parts,
        route_done=route_done,
    )
    await cog._handle_message_reply(
        message=message,
        system_prompt=system_prompt,
        context=context,
        memory_enabled=memory_enabled,
        effort=effort,
    )


def _assert_runtime_time_context(instructions: str, system_prompt: str) -> None:
    """Verifies that per-request time context wraps the base instructions."""
    assert instructions.startswith("Current request time:")
    assert "* Treat `message_created_at_asia_taipei` as now for this reply." in instructions
    assert "* `message_created_at_asia_taipei`: 2026-06-10T11:04:05+08:00" in instructions
    assert instructions.endswith(system_prompt)


def test_build_runtime_instructions_adds_request_time_context() -> None:
    """Request time context uses Discord's message creation timestamp."""
    message = FakeMessage(content="hi")

    instructions = _build_runtime_instructions(system_prompt="SYS", message=message)

    _assert_runtime_time_context(instructions=instructions, system_prompt="SYS")


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
        ),
    )


def _function_call_item(
    call_id: str, arguments: str, name: str = "get_user_memory"
) -> SimpleNamespace:
    """Builds a fake non-streaming `.output` function-call item for the selection phase."""
    return SimpleNamespace(type="function_call", name=name, call_id=call_id, arguments=arguments)


def _default_turn_events() -> list[SimpleNamespace]:
    """A minimal single-turn stream: one text delta and a completed event."""
    return [_text_event(delta="done"), _completed_event(input_tokens=1, output_tokens=1)]


async def _ready_reply_context() -> ReplyContext:
    """An empty reply context for directly exercising `_handle_image_reply`."""
    return ReplyContext()


async def test_handle_streaming_allows_missing_output_token_details(
    economy_isolated_db: None,
) -> None:
    """Regression: LiteLLM may return usage with output_tokens_details=null."""
    del economy_isolated_db
    message = FakeMessage()

    result = await ResponseStreamer(message=message).stream(responses=_stream_events())

    expected = f"hello from stream\n\n-# {TEST_LLM_MODEL} · ⬆ 12 ⬇ 34 · $0.00000000"
    assert result == expected
    assert message.replies[0].content == result


async def test_handle_streaming_continues_long_reply_as_reply_chain(
    economy_isolated_db: None,
) -> None:
    """Verifies replies over Discord's content limit continue as a reply chain."""
    del economy_isolated_db
    cog = _cog()
    message = FakeMessage(content="<@999> explain how long Discord replies are handled")
    body = "x" * 4500

    result = await ResponseStreamer(message=message).stream(
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
        )
    )

    usage_footer = f"\n\n-# {TEST_LLM_MODEL} · ⬆ 1 ⬇ 2 · $0.00000000"
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
    assert cog.openai_client.responses.create_models == []


def _deleted_source_error() -> nextcord.HTTPException:
    """Builds the Discord 400 50035 raised when replying to a since-deleted source."""
    return nextcord.HTTPException(
        SimpleNamespace(status=400, reason="Bad Request"),
        {"code": 50035, "message": "Invalid Form Body"},
    )


def _unknown_message_notfound() -> nextcord.NotFound:
    """Builds the 404 10008 a deleted source can raise on some Discord paths."""
    return nextcord.NotFound(
        SimpleNamespace(status=404, reason="Not Found"),
        {"code": 10008, "message": "Unknown Message"},
    )


@pytest.mark.parametrize("error", [_deleted_source_error(), _unknown_message_notfound()])
async def test_streaming_falls_back_to_channel_send_when_source_deleted(
    economy_isolated_db: None, error: nextcord.HTTPException
) -> None:
    """A deleted source makes the final reply land unparented via channel.send, not crash."""
    del economy_isolated_db
    message = FakeMessage()
    message.reply_error = error

    result = await ResponseStreamer(message=message).stream(responses=_stream_events())

    assert message.replies == []  # reply() raised, so nothing was recorded there
    assert message.channel.sent[0].content == result


async def test_streaming_followup_chain_intact_after_channel_send_fallback(
    economy_isolated_db: None,
) -> None:
    """Overflow follow-ups still chain off the unparented parent when the source is gone."""
    del economy_isolated_db
    message = FakeMessage(content="<@999> explain")
    message.reply_error = _deleted_source_error()
    body = "x" * 4500

    await ResponseStreamer(message=message).stream(
        responses=_stream_events_from(
            events=[_text_event(delta=body), _completed_event(input_tokens=1, output_tokens=2)]
        )
    )

    assert message.replies == []
    parent = message.channel.sent[0]
    assert parent.content == body[:DISCORD_MESSAGE_LIMIT]
    # The chain continues off the channel-sent parent, not the deleted source.
    assert parent.replies[0].content == body[DISCORD_MESSAGE_LIMIT : DISCORD_MESSAGE_LIMIT * 2]


async def test_streaming_reraises_non_deletion_http_errors(economy_isolated_db: None) -> None:
    """A non-deletion HTTP error (e.g. Forbidden) propagates instead of silently channel.send."""
    del economy_isolated_db
    message = FakeMessage()
    message.reply_error = nextcord.HTTPException(
        SimpleNamespace(status=403, reason="Forbidden"),
        {"code": 50013, "message": "Missing Permissions"},
    )

    with pytest.raises(nextcord.HTTPException):
        await ResponseStreamer(message=message).stream(responses=_stream_events())
    assert message.channel.sent == []


# ---- voice (spoken reply) ----


class _FakeVoiceSynthesizer:
    """Records synthesize calls and returns a configurable VoiceClip for streamer voice tests."""

    def __init__(
        self, audio: bytes | None = b"RIFFfake-wav", outcome: VoiceOutcome = VoiceOutcome.OK
    ) -> None:
        """Stores the audio bytes (None to simulate failure) and the reported outcome."""
        self.audio = audio
        self.outcome = outcome
        self.calls: list[dict[str, str]] = []

    async def synthesize(self, *, text: str, end_user_id: str) -> VoiceClip:
        """Records the spoken-text request and returns the preset VoiceClip."""
        self.calls.append({"text": text, "end_user_id": end_user_id})
        return VoiceClip(audio=self.audio, outcome=self.outcome)


def _voice_marker_events() -> list[SimpleNamespace]:
    """A single-turn stream whose reply wraps one segment in <voice> tags."""
    return [
        _text_event(delta="閉嘴啦白痴 "),
        _text_event(delta="<voice>嗆爆你</voice>"),
        _text_event(delta=" 滾"),
        _completed_event(input_tokens=3, output_tokens=4),
    ]


def _assert_no_voice_tags(text: str) -> None:
    """Asserts neither voice tag leaked into the visible reply."""
    assert "<voice>" not in text
    assert "</voice>" not in text


async def test_voice_marker_triggers_synthesis_and_strips_tag(economy_isolated_db: None) -> None:
    """A <voice> segment is spoken (only that part), its tags stripped, the clip attached."""
    del economy_isolated_db
    message = FakeMessage()
    synthesizer = _FakeVoiceSynthesizer()

    result = await ResponseStreamer(message=message, voice_synthesizer=synthesizer).stream(
        responses=_stream_events_from(_voice_marker_events())
    )

    _assert_no_voice_tags(result)
    # The wrapped content stays visible alongside the rest of the reply.
    assert "嗆爆你" in result
    assert "閉嘴啦白痴" in result
    # Only the wrapped segment (not the whole reply) is spoken.
    assert synthesizer.calls == [{"text": "嗆爆你", "end_user_id": message.author.name}]
    assert message.replies[0].file is not None
    assert message.replies[0].file.filename == "reply.wav"
    # The source message is marked with the voice app emoji while the clip is produced.
    assert message.added_reactions == ["<:voice:1517558121092878376>"]


async def test_voice_marker_absent_no_synthesis(economy_isolated_db: None) -> None:
    """A normal reply (no <voice>) never calls the synthesizer and attaches no file."""
    del economy_isolated_db
    message = FakeMessage()
    synthesizer = _FakeVoiceSynthesizer()

    await ResponseStreamer(message=message, voice_synthesizer=synthesizer).stream(
        responses=_stream_events()
    )

    assert synthesizer.calls == []
    assert message.replies[0].file is None
    # The model chose no voice, so there is nothing to hint about.
    assert message.added_reactions == []


async def test_voice_disabled_still_strips_marker(economy_isolated_db: None) -> None:
    """With no synthesizer (voice off) the tags are still stripped and no file attaches."""
    del economy_isolated_db
    message = FakeMessage()

    result = await ResponseStreamer(message=message).stream(
        responses=_stream_events_from(_voice_marker_events())
    )

    _assert_no_voice_tags(result)
    assert "嗆爆你" in result
    assert message.replies[0].file is None


async def test_voice_synthesis_failure_leaves_text_reply(economy_isolated_db: None) -> None:
    """A synthesis error leaves a clean text reply, no file, and hints with a warning emoji."""
    del economy_isolated_db
    message = FakeMessage()
    synthesizer = _FakeVoiceSynthesizer(audio=None, outcome=VoiceOutcome.ERROR)

    result = await ResponseStreamer(message=message, voice_synthesizer=synthesizer).stream(
        responses=_stream_events_from(_voice_marker_events())
    )

    _assert_no_voice_tags(result)
    assert message.replies[0].file is None
    # The voice marker is added before synth; a non-timeout failure then hints with the warning.
    assert message.added_reactions == ["<:voice:1517558121092878376>", "⚠️"]


async def test_voice_synthesis_timeout_hints_with_clock(economy_isolated_db: None) -> None:
    """A synthesis timeout leaves a text reply and hints with the clock emoji, staying silent."""
    del economy_isolated_db
    message = FakeMessage()
    synthesizer = _FakeVoiceSynthesizer(audio=None, outcome=VoiceOutcome.TIMEOUT)

    result = await ResponseStreamer(message=message, voice_synthesizer=synthesizer).stream(
        responses=_stream_events_from(_voice_marker_events())
    )

    _assert_no_voice_tags(result)
    assert message.replies[0].file is None
    assert message.added_reactions == ["<:voice:1517558121092878376>", "⏱️"]


def test_extract_inline_markers_voice_keeps_content() -> None:
    """A <voice> segment stays in the visible text; only the tags are stripped."""
    markers = extract_inline_markers(text="嗆爆你 <voice>聽好了</voice> 滾")
    assert markers.cleaned_text == "嗆爆你 聽好了 滾"
    assert markers.voice_text == "聽好了"
    assert markers.voice_requested is True
    assert markers.image_prompts == []


def test_extract_inline_markers_multiple_voice_segments_concatenate() -> None:
    """Multiple <voice> segments concatenate into one spoken input, all content kept."""
    markers = extract_inline_markers(text="<voice>第一</voice>中間<voice>第二</voice>")
    assert markers.voice_text == "第一\n第二"
    assert markers.cleaned_text == "第一中間第二"


def test_extract_inline_markers_image_block_removed() -> None:
    """An <image> block (tags AND content) is pulled from the visible reply."""
    markers = extract_inline_markers(text="看這張\n<image>a red cat on a sofa</image>")
    assert markers.image_prompts == ["a red cat on a sofa"]
    assert "<image>" not in markers.cleaned_text
    assert "a red cat" not in markers.cleaned_text
    assert markers.cleaned_text == "看這張"
    assert markers.voice_requested is False


def test_extract_inline_markers_multiple_image_blocks_in_order() -> None:
    """Every <image> block becomes an image request, kept in document order."""
    markers = extract_inline_markers(
        text="先看\n<image>a red cat</image>\n再看\n<image>a blue dog</image>"
    )
    assert markers.image_prompts == ["a red cat", "a blue dog"]
    assert "<image>" not in markers.cleaned_text
    assert "red cat" not in markers.cleaned_text
    assert "blue dog" not in markers.cleaned_text


def test_extract_inline_markers_closed_then_unclosed_image_both_pulled() -> None:
    """A complete block plus a trailing unclosed <image> are both captured, in order."""
    markers = extract_inline_markers(text="看\n<image>a red cat</image>\n還有\n<image>a blue dog")
    assert markers.image_prompts == ["a red cat", "a blue dog"]
    assert "<image>" not in markers.cleaned_text


def test_extract_inline_markers_unclosed_image_is_pulled() -> None:
    """An unclosed trailing <image> (model forgot to close) never leaks its description."""
    markers = extract_inline_markers(text="來囉\n<image>a sunset over the sea")
    assert markers.image_prompts == ["a sunset over the sea"]
    assert "<image>" not in markers.cleaned_text
    assert "sunset" not in markers.cleaned_text
    assert markers.cleaned_text == "來囉"


def test_extract_inline_markers_music_block_removed() -> None:
    """A <music> block (tags AND content) is pulled from the visible reply."""
    markers = extract_inline_markers(
        text="這首給你\n<music>upbeat anime J-pop, female vocals</music>"
    )
    assert markers.music_prompt == "upbeat anime J-pop, female vocals"
    assert "<music>" not in markers.cleaned_text
    assert "anime" not in markers.cleaned_text
    assert markers.cleaned_text == "這首給你"


def test_extract_inline_markers_only_first_music_block_kept() -> None:
    """Only the first non-empty <music> block is kept (a single clip per reply)."""
    markers = extract_inline_markers(
        text="<music>first track</music>中間<music>second track</music>"
    )
    assert markers.music_prompt == "first track"
    assert "<music>" not in markers.cleaned_text
    assert "second track" not in markers.cleaned_text


def test_extract_inline_markers_unclosed_music_is_pulled() -> None:
    """An unclosed trailing <music> (model forgot to close) never leaks its description."""
    markers = extract_inline_markers(text="等我一下\n<music>a calm lo-fi beat")
    assert markers.music_prompt == "a calm lo-fi beat"
    assert "<music>" not in markers.cleaned_text
    assert "lo-fi" not in markers.cleaned_text
    assert markers.cleaned_text == "等我一下"


def test_speechify_discord_markup_rewrites_and_drops() -> None:
    """Mentions resolve to names; emoji / timestamps drop; slash commands keep their words."""
    names = {239270225441193986: "小明", 42: "管理員", 7: "general"}

    def _resolve(*, target_id: int) -> str | None:
        return names.get(target_id)

    assert speechify_discord_markup(text="嗆爆 <@239270225441193986>", resolve_name=_resolve) == (
        "嗆爆 小明"
    )
    # Role and channel mentions resolve through the same snowflake lookup.
    assert speechify_discord_markup(text="<@&42> 去 <#7> 集合", resolve_name=_resolve) == (
        "管理員 去 general 集合"
    )
    # An unresolved mention is dropped, leaving no doubled space behind.
    assert speechify_discord_markup(text="哈囉 <@999> 你好", resolve_name=_resolve) == "哈囉 你好"
    # Custom emoji and timestamp tags are dropped; a slash-command reference keeps its words.
    assert speechify_discord_markup(text="讚啦 <:blobcheer:123>", resolve_name=_resolve) == "讚啦"
    assert speechify_discord_markup(
        text="活動在 <t:1700000000:F> 開始", resolve_name=_resolve
    ) == ("活動在 開始")
    assert (
        speechify_discord_markup(text="用 </play:456> 點歌", resolve_name=_resolve)
        == "用 play 點歌"
    )


def _voice_marker_mention_events() -> list[SimpleNamespace]:
    """A stream whose <voice> segment contains a raw user mention."""
    return [
        _text_event(delta="<voice>嗆爆 <@239270225441193986></voice>"),
        _completed_event(input_tokens=3, output_tokens=4),
    ]


async def test_voice_text_strips_discord_markup(economy_isolated_db: None) -> None:
    """The spoken clip narrates the resolved name while the visible reply keeps the mention."""
    del economy_isolated_db
    message = FakeMessage()
    message.guild = FakeGuild(members={239270225441193986: SimpleNamespace(display_name="小明")})
    synthesizer = _FakeVoiceSynthesizer()

    result = await ResponseStreamer(message=message, voice_synthesizer=synthesizer).stream(
        responses=_stream_events_from(_voice_marker_mention_events())
    )

    # The visible reply keeps the clickable mention; only the spoken text is normalised.
    assert "<@239270225441193986>" in result
    assert synthesizer.calls == [{"text": "嗆爆 小明", "end_user_id": message.author.name}]


def test_scrub_markers_for_preview_hides_streaming_fragments() -> None:
    """Markers arriving mid-stream are hidden from the live preview before the final extract."""
    # A partial trailing tag is trimmed; the content before it stays.
    assert scrub_markers_for_preview(text="嗆你 <voi") == "嗆你"
    # A complete <voice> pair is stripped but its content stays visible.
    assert scrub_markers_for_preview(text="嗆你 <voice>聽好</voice>") == "嗆你 聽好"
    # An unclosed <image> open and everything after it is hidden whole (the block is pulled).
    assert scrub_markers_for_preview(text="看這 <image>a red ca") == "看這"
    # A complete <image> block is removed whole.
    assert scrub_markers_for_preview(text="看這<image>a cat</image>之後") == "看這之後"
    assert scrub_markers_for_preview(text="正常文字") == "正常文字"


# ---- inline image (<image>) ----


class _FakeImageGenerator:
    """Records generate calls and returns configurable PNG bytes for streamer image tests."""

    def __init__(self, image: bytes | None = b"\x89PNG-fake") -> None:
        """Stores the PNG bytes (None to simulate a failed render) returned by generate."""
        self.image = image
        self.calls: list[dict[str, str]] = []

    async def generate(self, *, user_prompt: str, end_user_id: str) -> bytes | None:
        """Records the rough description request and returns the preset image bytes."""
        self.calls.append({"user_prompt": user_prompt, "end_user_id": end_user_id})
        return self.image


def _image_marker_events() -> list[SimpleNamespace]:
    """A single-turn stream whose reply wraps an <image> description."""
    return [
        _text_event(delta="這是你要的圖 "),
        _text_event(delta="<image>a cute black cat</image>"),
        _completed_event(input_tokens=3, output_tokens=4),
    ]


async def test_image_marker_generates_and_attaches(economy_isolated_db: None) -> None:
    """An <image> block is pulled from the reply, rendered, and the PNG attached to the reply."""
    del economy_isolated_db
    message = FakeMessage()
    generator = _FakeImageGenerator()

    result = await ResponseStreamer(message=message, image_generator=generator).stream(
        responses=_stream_events_from(_image_marker_events())
    )

    # The block (tags AND description) never shows in chat.
    assert "<image>" not in result
    assert "a cute black cat" not in result
    assert "這是你要的圖" in result
    # The rough description is handed to the generator and the PNG attached afterward.
    assert generator.calls == [
        {"user_prompt": "a cute black cat", "end_user_id": message.author.name}
    ]
    assert message.replies[0].file is not None
    assert message.replies[0].file.filename == "generated.png"
    # The source message is marked with the image app emoji while the image is rendered.
    assert message.added_reactions == ["<:image:1517559727880667226>"]


async def test_image_disabled_still_strips_marker(economy_isolated_db: None) -> None:
    """With no generator (inline image off) the block is still pulled and no file attaches."""
    del economy_isolated_db
    message = FakeMessage()

    result = await ResponseStreamer(message=message).stream(
        responses=_stream_events_from(_image_marker_events())
    )

    assert "<image>" not in result
    assert "a cute black cat" not in result
    assert message.replies[0].file is None


async def test_image_generation_failure_hints(economy_isolated_db: None) -> None:
    """A failed render leaves a clean text reply with no file and a warning hint."""
    del economy_isolated_db
    message = FakeMessage()
    generator = _FakeImageGenerator(image=None)

    result = await ResponseStreamer(message=message, image_generator=generator).stream(
        responses=_stream_events_from(_image_marker_events())
    )

    assert "a cute black cat" not in result
    assert message.replies[0].file is None
    assert message.added_reactions == ["<:image:1517559727880667226>", "⚠️"]


async def test_voice_and_image_attach_in_one_edit(economy_isolated_db: None) -> None:
    """A reply with both markers rides a single edit carrying the WAV and the PNG together."""
    del economy_isolated_db
    message = FakeMessage()
    synthesizer = _FakeVoiceSynthesizer()
    generator = _FakeImageGenerator()

    result = await ResponseStreamer(
        message=message, voice_synthesizer=synthesizer, image_generator=generator
    ).stream(
        responses=_stream_events_from([
            _text_event(delta="看 <voice>聽好</voice> "),
            _text_event(delta="<image>a red balloon</image>"),
            _completed_event(input_tokens=3, output_tokens=4),
        ])
    )

    assert "聽好" in result
    assert "<image>" not in result
    assert "a red balloon" not in result
    files = message.replies[0].files
    assert files is not None
    assert {item.filename for item in files} == {"reply.wav", "generated.png"}


async def test_multiple_image_markers_attach_distinct_files(economy_isolated_db: None) -> None:
    """Several <image> blocks each render and attach under distinct filenames in one edit."""
    del economy_isolated_db
    message = FakeMessage()
    generator = _FakeImageGenerator()

    result = await ResponseStreamer(message=message, image_generator=generator).stream(
        responses=_stream_events_from([
            _text_event(delta="兩張圖 "),
            _text_event(delta="<image>a red cat</image><image>a blue dog</image>"),
            _completed_event(input_tokens=3, output_tokens=4),
        ])
    )

    assert "<image>" not in result
    # Each description renders independently, in order.
    assert [call["user_prompt"] for call in generator.calls] == ["a red cat", "a blue dog"]
    files = message.replies[0].files
    assert files is not None
    assert [item.filename for item in files] == ["generated_1.png", "generated_2.png"]


async def test_image_markers_capped_at_limit(economy_isolated_db: None) -> None:
    """More <image> blocks than the per-reply cap render only up to MAX_INLINE_IMAGES."""
    del economy_isolated_db
    message = FakeMessage()
    generator = _FakeImageGenerator()
    blocks = "".join(f"<image>image {index}</image>" for index in range(MAX_INLINE_IMAGES + 3))

    await ResponseStreamer(message=message, image_generator=generator).stream(
        responses=_stream_events_from([
            _text_event(delta=f"好多圖 {blocks}"),
            _completed_event(input_tokens=3, output_tokens=4),
        ])
    )

    # Only the first MAX_INLINE_IMAGES render and attach; the extra blocks are dropped.
    assert len(generator.calls) == MAX_INLINE_IMAGES
    files = message.replies[0].files
    assert files is not None
    assert len(files) == MAX_INLINE_IMAGES


# ---- inline music (<music>) ----


class _FakeMusicGenerator:
    """Records generate calls and returns a configurable MusicClip (or None) for streamer tests."""

    def __init__(
        self, audio: bytes | None = b"ID3-fake-mp3", mime_type: str = "audio/mp3"
    ) -> None:
        """Stores the clip (None audio simulates a failed render) returned by generate."""
        self.clip = MusicClip(audio=audio, mime_type=mime_type) if audio is not None else None
        self.calls: list[str] = []

    async def generate(self, *, user_prompt: str) -> MusicClip | None:
        """Records the music description request and returns the preset clip."""
        self.calls.append(user_prompt)
        return self.clip


def _music_marker_events() -> list[SimpleNamespace]:
    """A single-turn stream whose reply wraps a <music> description."""
    return [
        _text_event(delta="這首給你 "),
        _text_event(delta="<music>upbeat anime J-pop, female vocals</music>"),
        _completed_event(input_tokens=3, output_tokens=4),
    ]


async def test_music_marker_generates_and_attaches(economy_isolated_db: None) -> None:
    """A <music> block is pulled from the reply, generated, and the clip attached to the reply."""
    del economy_isolated_db
    message = FakeMessage()
    generator = _FakeMusicGenerator()

    result = await ResponseStreamer(message=message, music_generator=generator).stream(
        responses=_stream_events_from(_music_marker_events())
    )

    # The block (tags AND description) never shows in chat.
    assert "<music>" not in result
    assert "anime" not in result
    assert "這首給你" in result
    # The description is handed to the generator and the clip attached afterward.
    assert generator.calls == ["upbeat anime J-pop, female vocals"]
    assert message.replies[0].file is not None
    assert message.replies[0].file.filename == "music.mp3"
    # The source message is marked with the music emoji while the clip renders.
    assert message.added_reactions == ["🎵"]


async def test_music_disabled_still_strips_marker(economy_isolated_db: None) -> None:
    """With no generator (music off) the block is still pulled and no file attaches."""
    del economy_isolated_db
    message = FakeMessage()

    result = await ResponseStreamer(message=message).stream(
        responses=_stream_events_from(_music_marker_events())
    )

    assert "<music>" not in result
    assert "anime" not in result
    assert message.replies[0].file is None


async def test_music_generation_failure_hints(economy_isolated_db: None) -> None:
    """A failed render leaves a clean text reply with no file and a warning hint."""
    del economy_isolated_db
    message = FakeMessage()
    generator = _FakeMusicGenerator(audio=None)

    result = await ResponseStreamer(message=message, music_generator=generator).stream(
        responses=_stream_events_from(_music_marker_events())
    )

    assert "anime" not in result
    assert message.replies[0].file is None
    assert message.added_reactions == ["🎵", "⚠️"]


async def test_music_filename_follows_returned_mime() -> None:
    """The attachment extension follows the returned audio mime, falling back to .mp3."""
    assert music_filename(mime_type="audio/wav") == "music.wav"
    assert music_filename(mime_type="audio/mpeg") == "music.mp3"
    assert music_filename(mime_type="audio/ogg") == "music.ogg"
    assert music_filename(mime_type=None) == "music.mp3"


async def test_voice_music_image_attach_in_one_edit(economy_isolated_db: None) -> None:
    """A reply with all three markers rides one edit carrying the WAV, the clip, and the PNG."""
    del economy_isolated_db
    message = FakeMessage()
    synthesizer = _FakeVoiceSynthesizer()
    music_generator = _FakeMusicGenerator()
    image_generator = _FakeImageGenerator()

    result = await ResponseStreamer(
        message=message,
        voice_synthesizer=synthesizer,
        music_generator=music_generator,
        image_generator=image_generator,
    ).stream(
        responses=_stream_events_from([
            _text_event(delta="來囉 <voice>聽好</voice> "),
            _text_event(delta="<music>a calm lo-fi beat</music><image>a red balloon</image>"),
            _completed_event(input_tokens=3, output_tokens=4),
        ])
    )

    assert "聽好" in result
    assert "<music>" not in result
    assert "lo-fi" not in result
    assert "a red balloon" not in result
    files = message.replies[0].files
    assert files is not None
    assert {item.filename for item in files} == {"reply.wav", "music.mp3", "generated.png"}


async def test_music_generator_drops_clip_on_bad_audio_payload() -> None:
    """A non-decodable audio payload returns None instead of raising into the attach gather."""

    class _Interactions:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            """Returns an interaction whose audio data cannot be base64-decoded."""
            del kwargs
            return SimpleNamespace(
                output_audio=SimpleNamespace(data="not-valid-base64-x", mime_type="audio/mpeg")
            )

    client = SimpleNamespace(aio=SimpleNamespace(interactions=_Interactions()))
    generator = MusicGenerator(client=client, music_model=RuntimeModelCatalog().music_model)

    # The decode failure is swallowed (best-effort), so the streamer's media gather is never aborted.
    assert await generator.generate(user_prompt="a calm beat") is None


class _FakeSpeechResponse:
    """Async binary-response stand-in exposing aread() like the OpenAI speech result."""

    def __init__(self, data: bytes) -> None:
        """Stores the audio bytes to return from aread()."""
        self._data = data

    async def aread(self) -> bytes:
        """Returns the preset audio bytes."""
        return self._data


class _FakeSpeech:
    """Records audio.speech.create calls and returns or raises a preset result."""

    def __init__(self, data: bytes = b"RIFFwav", error: Exception | None = None) -> None:
        """Stores the bytes to return and an optional error to raise."""
        self.data = data
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> _FakeSpeechResponse:
        """Records the call and returns the preset response or raises the preset error."""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _FakeSpeechResponse(self.data)


def _fake_audio_client(speech: _FakeSpeech) -> SimpleNamespace:
    """A minimal AsyncOpenAI stand-in exposing client.audio.speech.create."""
    return SimpleNamespace(audio=SimpleNamespace(speech=speech))


async def test_voice_synthesizer_prepends_style_and_returns_bytes() -> None:
    """A normal reply renders to bytes with the style directive prepended to the input."""
    speech = _FakeSpeech(data=b"RIFFwav")
    synth = VoiceSynthesizer(client=_fake_audio_client(speech=speech))

    clip = await synth.synthesize(text="閉嘴", end_user_id="tester")

    assert clip.outcome is VoiceOutcome.OK
    assert clip.audio == b"RIFFwav"
    assert speech.calls[0]["input"].endswith("閉嘴")
    assert speech.calls[0]["input"] != "閉嘴"
    # response_format is intentionally never sent (the proxy 500s on it).
    assert "response_format" not in speech.calls[0]
    # The per-request timeout is applied so a slow clip cannot stall the message pipeline.
    assert speech.calls[0]["timeout"] == VOICE_TIMEOUT_SECONDS


async def test_voice_synthesizer_swallows_provider_errors() -> None:
    """A provider error reports ERROR with no audio so the reply stays text-only."""
    speech = _FakeSpeech(error=RuntimeError("boom"))
    synth = VoiceSynthesizer(client=_fake_audio_client(speech=speech))

    clip = await synth.synthesize(text="嗆你", end_user_id="tester")

    assert clip.audio is None
    assert clip.outcome is VoiceOutcome.ERROR


async def test_voice_synthesizer_reports_timeout() -> None:
    """A request timeout is reported as TIMEOUT so the caller can hint distinctly."""
    speech = _FakeSpeech(error=APITimeoutError(request=httpx.Request("POST", "http://proxy")))
    synth = VoiceSynthesizer(client=_fake_audio_client(speech=speech))

    clip = await synth.synthesize(text="嗆你", end_user_id="tester")

    assert clip.audio is None
    assert clip.outcome is VoiceOutcome.TIMEOUT


async def test_voice_oversized_clip_not_attached(economy_isolated_db: None) -> None:
    """A clip past the guild's upload limit is dropped, leaving a text-only reply."""
    del economy_isolated_db
    message = FakeMessage()
    message.guild = FakeGuild(filesize_limit=8)
    synthesizer = _FakeVoiceSynthesizer(audio=b"x" * 16)

    result = await ResponseStreamer(message=message, voice_synthesizer=synthesizer).stream(
        responses=_stream_events_from(_voice_marker_events())
    )

    _assert_no_voice_tags(result)
    assert message.replies[0].file is None
    # An oversized clip is dropped for a non-timeout reason, so it hints with the warning emoji.
    assert message.added_reactions == ["<:voice:1517558121092878376>", "⚠️"]


@pytest.mark.parametrize(("enabled", "expect_synth"), [(True, True), (False, False)])
async def test_voice_config_gate_controls_synthesizer(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, expect_synth: bool
) -> None:
    """config.voice_reply_enabled gates whether the QA streamer receives a synthesizer."""
    cog = _cog()
    cog.config = SimpleNamespace(voice_reply_enabled=enabled)
    captured: list[object] = []

    class FakeResponder:
        """Captures the synthesizer the cog wires into the streamer."""

        def __init__(  # noqa: PLR0913 -- stub mirrors ResponseStreamer's constructor kwargs
            self,
            message: FakeMessage,
            memory_lookups: list[str] | None = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
            model_effort: str = "",
            voice_synthesizer: object | None = None,
            image_generator: object | None = None,
            music_generator: object | None = None,
        ) -> None:
            """Records the synthesizer the cog passed."""
            del message, memory_lookups, input_tokens, output_tokens, model_effort
            del image_generator, music_generator
            captured.append(voice_synthesizer)

        async def stream(self, *, responses: object) -> str:
            """Returns placeholder reply content."""
            del responses
            return "回覆"

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **_: None)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await cog._handle_message_reply(
        message=message,
        system_prompt="SYS",
        context=ReplyContext(),
        memory_enabled=False,
        allow_voice=True,
    )

    assert (captured[0] is not None) == expect_synth
    if expect_synth:
        assert isinstance(captured[0], VoiceSynthesizer)


@pytest.mark.parametrize(("enabled", "expect_gen"), [(True, True), (False, False)])
async def test_image_config_gate_controls_generator(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, expect_gen: bool
) -> None:
    """config.inline_image_enabled gates whether the QA streamer receives an image generator."""
    cog = _cog()
    cog.config = SimpleNamespace(voice_reply_enabled=False, inline_image_enabled=enabled)
    captured: list[object] = []

    class FakeResponder:
        """Captures the image generator the cog wires into the streamer."""

        def __init__(  # noqa: PLR0913 -- stub mirrors ResponseStreamer's constructor kwargs
            self,
            message: FakeMessage,
            memory_lookups: list[str] | None = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
            model_effort: str = "",
            voice_synthesizer: object | None = None,
            image_generator: object | None = None,
            music_generator: object | None = None,
        ) -> None:
            """Records the generator the cog passed."""
            del message, memory_lookups, input_tokens, output_tokens, model_effort
            del voice_synthesizer, music_generator
            captured.append(image_generator)

        async def stream(self, *, responses: object) -> str:
            """Returns placeholder reply content."""
            del responses
            return "回覆"

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **_: None)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await cog._handle_message_reply(
        message=message,
        system_prompt="SYS",
        context=ReplyContext(),
        memory_enabled=False,
        allow_image=True,
    )

    assert (captured[0] is not None) == expect_gen
    if expect_gen:
        assert isinstance(captured[0], ImageGenerator)


class _FakeInteractionsResource:
    """Records Interactions answer calls and returns a fake event stream."""

    def __init__(self, events: list[SimpleNamespace]) -> None:
        """Stores the events each create() will stream and a call recorder."""
        self._events = events
        self.calls: list[SimpleNamespace] = []

    async def create(  # noqa: PLR0913 -- mirrors the Interactions create signature
        self,
        model: str,
        system_instruction: str,
        input: list[object],  # noqa: A002 -- SDK parameter
        environment: str,
        generation_config: object,
        tools: list[object],
        stream: bool,
    ) -> AsyncIterator[SimpleNamespace]:
        """Records the call and returns the fake Interactions event stream."""
        del environment, tools, stream
        self.calls.append(
            SimpleNamespace(
                model=model,
                system_instruction=system_instruction,
                input=input,
                generation_config=generation_config,
            )
        )
        return _stream_events_from(events=self._events)


class _FakeInteractionsClient:
    """Fake Gemini client exposing the async Interactions resource."""

    def __init__(self, events: list[SimpleNamespace]) -> None:
        """Wires the recorder under `aio.interactions` like the real client."""
        self.recorder = _FakeInteractionsResource(events=events)
        self.aio = SimpleNamespace(interactions=self.recorder)


def _interactions_turn_events() -> list[SimpleNamespace]:
    """A minimal Interactions turn: created, one text delta, completed with usage."""
    return [
        SimpleNamespace(
            event_type="interaction.created", interaction=SimpleNamespace(model=TEST_LLM_MODEL)
        ),
        SimpleNamespace(
            event_type="step.delta", delta=SimpleNamespace(type="text", text="watched it")
        ),
        SimpleNamespace(
            event_type="interaction.completed",
            interaction=SimpleNamespace(
                model=TEST_LLM_MODEL,
                usage=SimpleNamespace(total_input_tokens=12, total_output_tokens=34),
            ),
            metadata=None,
        ),
    ]


async def test_youtube_qa_uses_interactions_backend(
    economy_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A watched YouTube URL streams the answer through Interactions, not Responses."""
    del economy_isolated_db
    cog = _cog()
    cog.config = SimpleNamespace(
        voice_reply_enabled=False, inline_image_enabled=False, youtube_video_enabled=True
    )
    fake = _FakeInteractionsClient(events=_interactions_turn_events())
    cog.__dict__["gemini_client"] = fake
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **_: None)

    url = "https://youtu.be/jNQXAC9IVRw"
    message = FakeMessage(content=f"<@999> 總結這影片 {url}", author=FakeAuthor(user_id=1))
    await cog._handle_message_reply(
        message=message,
        system_prompt="SYS",
        context=ReplyContext(),
        memory_enabled=False,
        yt_url=url,
    )

    # The Responses answer stream was never used; the Interactions one was, with the video part.
    assert cog.openai_client.responses.create_streams == []
    assert len(fake.recorder.calls) == 1
    last_step_parts = fake.recorder.calls[0].input[-1]["content"]
    assert {"type": "video", "uri": url} in last_step_parts
    # The shared streamer rendered the reply and a footer from the Interactions usage.
    assert "watched it" in message.replies[0].content
    assert "⬆ 12 ⬇ 34" in message.replies[0].content
    # A persistent watch reaction marks that the reply was grounded in the video.
    assert "<:youtube:1517546722535018596>" in message.added_reactions


async def test_youtube_interactions_passes_effort_as_thinking_level(
    economy_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graded effort is sent straight through as the Interactions thinking_level."""
    del economy_isolated_db
    cog = _cog()
    cog.config = SimpleNamespace(
        voice_reply_enabled=False, inline_image_enabled=False, youtube_video_enabled=True
    )
    fake = _FakeInteractionsClient(events=_interactions_turn_events())
    cog.__dict__["gemini_client"] = fake
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **_: None)

    url = "https://youtu.be/jNQXAC9IVRw"
    message = FakeMessage(content=f"<@999> {url}", author=FakeAuthor(user_id=1))
    await cog._handle_message_reply(
        message=message,
        system_prompt="SYS",
        context=ReplyContext(),
        memory_enabled=False,
        effort="medium",
        yt_url=url,
    )

    assert fake.recorder.calls[0].generation_config["thinking_level"] == "medium"


@pytest.mark.parametrize("scenario", ["kill_switch_off", "non_gemini_model", "no_url"])
async def test_youtube_qa_falls_back_to_responses(
    economy_isolated_db: None, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    """Without a watchable Gemini video turn, the answer stays on the Responses path."""
    del economy_isolated_db
    cog = _cog()
    cog.config = SimpleNamespace(
        voice_reply_enabled=False,
        inline_image_enabled=False,
        youtube_video_enabled=scenario != "kill_switch_off",
    )
    if scenario == "non_gemini_model":
        monkeypatch.setattr(
            RuntimeModelCatalog,
            "slow_model",
            property(lambda _self: ModelSettings(name="gpt-5-mini", effort="high")),
        )
    fake = _FakeInteractionsClient(events=_interactions_turn_events())
    cog.__dict__["gemini_client"] = fake
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **_: None)

    url = "https://youtu.be/jNQXAC9IVRw"
    yt_url = None if scenario == "no_url" else url
    message = FakeMessage(content=f"<@999> {url}", author=FakeAuthor(user_id=1))
    await cog._handle_message_reply(
        message=message,
        system_prompt="SYS",
        context=ReplyContext(),
        memory_enabled=False,
        yt_url=yt_url,
    )

    assert fake.recorder.calls == []
    assert cog.openai_client.responses.create_streams == [True]


def test_find_youtube_url_searches_reference_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A YouTube link in the replied-to message is found even when the reply omits it."""
    monkeypatch.setattr("discordbot.cogs.gen_reply.Message", FakeMessage)
    url = "https://youtu.be/jNQXAC9IVRw"
    referenced = FakeMessage(content=f"look at this {url}")
    referenced.id = 555
    message = FakeMessage(content="<@999> 總結這影片")
    message.reference = FakeReference(resolved=referenced)

    assert _find_youtube_url(message=message) == url


def test_find_youtube_url_none_without_link(monkeypatch: pytest.MonkeyPatch) -> None:
    """No YouTube link in the message or its reference chain returns None."""
    monkeypatch.setattr("discordbot.cogs.gen_reply.Message", FakeMessage)
    message = FakeMessage(content="<@999> hi")
    message.reference = FakeReference(resolved=FakeMessage(content="just chatting"))

    assert _find_youtube_url(message=message) is None


def _media_builder() -> MessageInputBuilder:
    """A MessageInputBuilder wired with a fake Gemini client for media-path tests."""
    return MessageInputBuilder(
        bot=SimpleNamespace(user=SimpleNamespace(id=999, name="bot")),
        runtime_models=RuntimeModelCatalog(),
        attachment_handler=_fake_uploader(),
    )


def test_collect_sources_skips_bot_own_voice_clip() -> None:
    """The bot's own generated voice clip is dropped from history input; others survive."""
    builder = _media_builder()  # bot user id 999

    bot_msg = FakeMessage(author=FakeAuthor(user_id=999))
    bot_msg.attachments = [
        FakeAttachment(filename="reply.wav", content_type="audio/wav", attachment_id=1),
        FakeAttachment(filename="note.txt", content_type="text/plain", attachment_id=2),
    ]
    # The bot's voice clip is skipped; a normal attachment on its message is kept.
    assert [s.cache_key for s in builder.collect_attachment_sources(message=bot_msg)] == [2]

    # The same filename on a human's message is NOT skipped (only the bot's own clip is).
    user_msg = FakeMessage(author=FakeAuthor(user_id=1))
    user_msg.attachments = [
        FakeAttachment(filename="reply.wav", content_type="audio/wav", attachment_id=3)
    ]
    assert [s.cache_key for s in builder.collect_attachment_sources(message=user_msg)] == [3]


def test_collect_sources_keeps_bot_own_music_clip() -> None:
    """The bot's own generated music clip is deliberately retained (unlike the voice clip).

    The `<music>` description is stripped from the visible reply, so the clip is the only trace
    of the song the bot made; keeping it lets a later turn reference it. Only the spoken `reply.wav`
    (whose text is already in the transcript) is skipped.
    """
    builder = _media_builder()  # bot user id 999

    bot_msg = FakeMessage(author=FakeAuthor(user_id=999))
    bot_msg.attachments = [
        FakeAttachment(filename="music.mp3", content_type="audio/mpeg", attachment_id=1),
        FakeAttachment(filename="reply.wav", content_type="audio/wav", attachment_id=2),
    ]
    # The music clip is kept (cache_key 1); only the voice clip (cache_key 2) is skipped.
    assert [s.cache_key for s in builder.collect_attachment_sources(message=bot_msg)] == [1]


async def test_dead_source_skipped_within_ttl_then_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing source is skipped (no re-fetch) for the TTL, then retried once after it."""
    calls = {"n": 0}

    def _raise_get_image_data(image_file: str) -> bytes:
        del image_file
        calls["n"] += 1
        raise RuntimeError("CDN url expired")

    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.attachment.loaders.get_image_data", _raise_get_image_data
    )
    uploader = _fake_uploader()
    url = "https://example.test/dead.png"

    assert await uploader.render_image(source=url, cache_key=url, allow_dead_cache=True) is None
    assert calls["n"] == 1
    # Within the TTL the source is skipped without another fetch.
    assert await uploader.render_image(source=url, cache_key=url, allow_dead_cache=True) is None
    assert calls["n"] == 1
    # Backdating the marker past the TTL retries the fetch exactly once (self-heal).
    uploader._dead_sources[url] = datetime.now(tz=UTC) - DEAD_SOURCE_TTL - timedelta(seconds=1)
    assert await uploader.render_image(source=url, cache_key=url, allow_dead_cache=True) is None
    assert calls["n"] == 2


async def test_non_history_render_does_not_dead_cache_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current/reference renders (allow_dead_cache off) retry a transient failure, not poison it."""
    calls = {"n": 0}

    def _raise_get_image_data(image_file: str) -> bytes:
        del image_file
        calls["n"] += 1
        raise RuntimeError("transient blip")

    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.attachment.loaders.get_image_data", _raise_get_image_data
    )
    uploader = _fake_uploader()
    url = "https://example.test/fresh.png"

    # Default path (current/reference): each call re-attempts the fetch and never marks dead.
    assert await uploader.render_image(source=url, cache_key=url) is None
    assert await uploader.render_image(source=url, cache_key=url) is None
    assert calls["n"] == 2
    assert url not in uploader._dead_sources


async def test_media_semaphore_bounds_media_io_concurrency() -> None:
    """The shared semaphore caps the whole download+upload sequence, not just the upload.

    Counting concurrency in the byte loader proves non-image downloads (which run before the
    Gemini upload) are bounded too, so concurrent pipelines cannot buffer every file at once.
    """
    uploader = _fake_uploader()
    uploader._media_semaphore = asyncio.Semaphore(2)
    state = {"active": 0, "peak": 0}

    async def _slow_load() -> tuple[bytes, str]:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return b"x", "image/png"

    results = await asyncio.gather(*[
        uploader._resolve_file_upload(
            cache_key=f"k{index}", filename=f"f{index}", load_data=_slow_load
        )
        for index in range(6)
    ])

    assert all(result is not None for result in results)
    assert state["peak"] == 2


def test_extract_friendly_error_prefers_nested_provider_message() -> None:
    """Verifies nested provider errors are preferred over wrapper text."""
    raw = """wrapper b'{"error": {"message": "quota exceeded"}}'"""
    assert extract_friendly_error(exc=RuntimeError(raw)) == "quota exceeded"
    assert extract_friendly_error(exc=RuntimeError("plain failure")) == "plain failure"
    assert extract_friendly_error(exc=RuntimeError("bad b'not json'")) == "bad b'not json'"


def test_required_modality_gate_keeps_code_and_text() -> None:
    """The MIME gate drops unknown binaries but keeps source-code / structured-text types."""
    modality = MessageInputBuilder.required_modality
    # Known binary application types are dropped before any upload.
    assert modality(content_type="application/octet-stream") == "unknown"
    assert modality(content_type="application/x-tar") == "unknown"
    # Source-code / script application types still proxy through (.rb -> application/x-ruby).
    assert modality(content_type="application/x-ruby") == "image"
    assert modality(content_type="application/x-perl") == "image"
    # Structured-text suffixes and text/* pass too.
    assert modality(content_type="application/geo+json") == "image"
    assert modality(content_type="application/atom+xml") == "image"
    assert modality(content_type="text/x-go") == "image"


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
    assert await cog.input_builder.get_user_prompt(content="hi <@!999>") == "hi"
    assert cog.input_builder.has_bot_mention(content="hi <@999>")
    assert cog.input_builder.has_bot_mention(content="hi <@!999>")
    assert "Author" in cog.input_builder.extract_embed_text(embeds=[embed])

    self_mention = FakeMessage(content="你的審美跟 <@999> 一樣", author=FakeAuthor(user_id=1))
    assert (
        await cog.input_builder.get_cleaned_content(message=self_mention) == self_mention.content
    )

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

    file_rendered = await cog.input_builder.attachment_handler.render_file(
        attachment=FakeAttachment(filename="note.txt", content_type="text/plain", payload=b"abc"),
        cache_key="note.txt",
    )
    assert file_rendered is not None
    file_part, file_expiry = file_rendered
    assert file_part["type"] == "input_file"
    assert file_part["file_id"] == "https://files.test/note.txt"
    assert file_expiry == datetime(2099, 1, 1, tzinfo=UTC)

    image_rendered = await cog.input_builder.attachment_handler.render_image(
        source=FakeAttachment(
            filename="pixel.png", content_type="image/png", payload=base64.b64decode(_png_b64())
        ),
        cache_key="pixel.png",
    )
    assert image_rendered is not None
    image_part, _image_expiry = image_rendered
    assert image_part["type"] == "input_file"
    assert image_part["file_id"] == "https://files.test/pixel.png"

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
        "discordbot.cogs._gen_reply.attachment.loaders.get_image_data",
        lambda image_file: base64.b64decode(_png_b64()),
    )
    parts = await cog.input_builder.get_attachment_parts(message=message)
    assert [part["type"] for part in parts] == ["input_file", "input_file", "input_file"]


async def test_upload_file_polls_active_and_drops_unready_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the upload polls to ACTIVE and drops files that never become usable."""

    async def _no_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.attachment.gemini_file_api.asyncio.sleep", _no_sleep
    )

    def _uploader(files: FakeGeminiFiles) -> GeminiFileUploader:
        return _fake_uploader(files=files)

    # PROCESSING for two polls, then ACTIVE: the file URI and its expiry are returned.
    active = _uploader(FakeGeminiFiles(processing_rounds=2))
    uploaded = await active._upload_file(
        filename="doc.pdf", data=b"x", content_type="application/pdf"
    )
    assert uploaded == ("https://files.test/doc.pdf", datetime(2099, 1, 1, tzinfo=UTC))

    # Terminal non-active state: the file is dropped.
    failed = _uploader(FakeGeminiFiles(final_state=FileState.FAILED))
    assert (
        await failed._upload_file(filename="bad.pdf", data=b"x", content_type="application/pdf")
        is None
    )

    # Never leaves PROCESSING within the bound: the timeout drops the file. An auto-advancing
    # clock jumps past the 15s bound on each read, so the deadline trips regardless of how many
    # monotonic() calls the upload path makes (e.g. for latency logging).
    clock = {"now": 0.0}

    def _fake_monotonic() -> float:
        clock["now"] += 50.0
        return clock["now"]

    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.attachment.gemini_file_api.time.monotonic", _fake_monotonic
    )
    stuck = _uploader(FakeGeminiFiles(processing_rounds=99))
    pending = await stuck._upload_file(filename="slow.mp4", data=b"x", content_type="video/mp4")
    assert isinstance(pending, PendingUpload)
    assert pending.name == "slow.mp4"
    assert pending.uri == "https://files.test/slow.mp4"

    # Upload raises: the file is dropped instead of aborting the reply.
    async def _raise(file: BytesIO, config: dict[str, str]) -> SimpleNamespace:
        del file, config
        raise RuntimeError("upload failed")

    boom = _uploader(FakeGeminiFiles())
    monkeypatch.setattr(boom.gemini_client.aio.files, "upload", _raise)
    assert await boom._upload_file(filename="x.txt", data=b"x", content_type="text/plain") is None


async def test_resolve_file_upload_recovers_pending_on_next_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out upload is cached as pending and re-polled, not re-uploaded, next time."""

    async def _no_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.attachment.gemini_file_api.asyncio.sleep", _no_sleep
    )

    # Auto-advancing clock: each call jumps well past the 15s activation bound, so the first
    # reference times out to PENDING regardless of how many monotonic() calls the upload path
    # makes (e.g. for latency logging). Robust to instrumentation, unlike a hand-counted list.
    clock = {"now": 0.0}

    def _fake_monotonic() -> float:
        clock["now"] += 50.0
        return clock["now"]

    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.attachment.gemini_file_api.time.monotonic", _fake_monotonic
    )

    files = FakeGeminiFiles(processing_rounds=99)
    uploader = _fake_uploader(files=files)

    load_calls = 0

    async def _load() -> tuple[bytes, str]:
        nonlocal load_calls
        load_calls += 1
        return b"x", "video/mp4"

    # First reference times out while still PROCESSING: dropped for now, cached as pending.
    first = await uploader._resolve_file_upload(cache_key="vid", filename="v.mp4", load_data=_load)
    assert first is None
    assert "vid" in uploader._pending_uploads
    assert files.upload_calls == [("v.mp4", "video/mp4")]
    assert load_calls == 1  # downloaded once for the fresh upload

    # The file finished processing in the background; the next reference re-polls the same
    # file once and adopts it, without re-downloading or re-uploading the bytes.
    async def _active_get(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            name=name,
            uri=f"https://files.test/{name}",
            state=FileState.ACTIVE,
            error=None,
            expiration_time=datetime(2099, 1, 1, tzinfo=UTC),
        )

    monkeypatch.setattr(files, "get", _active_get)
    second = await uploader._resolve_file_upload(
        cache_key="vid", filename="v.mp4", load_data=_load
    )
    assert second == ("https://files.test/v.mp4", datetime(2099, 1, 1, tzinfo=UTC))
    assert "vid" not in uploader._pending_uploads
    assert files.upload_calls == [("v.mp4", "video/mp4")]  # no second upload
    assert load_calls == 1  # adopt path did not re-download the source


def test_loggable_cache_key_strips_url_query_token() -> None:
    """An int key logs unchanged; a URL key drops its (possibly signed) query string."""
    assert loggable_cache_key(cache_key=12345) == 12345
    assert (
        loggable_cache_key(cache_key="https://media.discordapp.net/x/y.png?ex=1&hm=secrettoken")
        == "https://media.discordapp.net/x/y.png"
    )
    assert loggable_cache_key(cache_key="https://cdn.example/a.png") == "https://cdn.example/a.png"


async def test_openai_file_uploader_renders_image_and_file_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI uploads return file-id content parts for images and files."""
    files = FakeOpenAIFiles()
    renderer = _fake_openai_uploader(files=files)

    image_rendered = await renderer.render_image(
        source=FakeAttachment(
            filename="pic.png", content_type="image/png", payload=base64.b64decode(_png_b64())
        ),
        cache_key="pic.png",
    )
    assert image_rendered is not None
    image_part, image_expiry = image_rendered
    assert image_part["type"] == "input_image"
    assert image_part["file_id"] == "file-test"
    assert image_part["detail"] == "auto"
    assert image_expiry == datetime(2099, 1, 1, tzinfo=UTC)

    url = "https://example.test/image.png"
    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.attachment.loaders.get_image_data", lambda image_file: b"jpeg"
    )
    url_image_rendered = await renderer.render_image(source=url, cache_key=url)
    assert url_image_rendered is not None
    url_image_part, _url_image_expiry = url_image_rendered
    assert url_image_part["type"] == "input_image"
    assert url_image_part["file_id"] == "file-test"

    file_rendered = await renderer.render_file(
        attachment=FakeAttachment(
            filename="notes.txt", content_type="text/plain", payload=b"hello world"
        ),
        cache_key="notes.txt",
    )
    assert file_rendered is not None
    file_part, file_expiry = file_rendered
    assert file_part["type"] == "input_file"
    assert file_part["file_id"] == "file-test"
    assert file_part["filename"] == "notes.txt"
    assert file_expiry == datetime(2099, 1, 1, tzinfo=UTC)

    assert files.create_calls[0][0] == "pic.png"
    assert files.create_calls[0][2] == "image/jpeg"
    assert files.create_calls[0][3] == "vision"
    assert files.create_calls[0][4] == {"anchor": "created_at", "seconds": 2_592_000}
    assert files.create_calls[0][5] == {"model": TEST_LLM_MODEL}
    assert files.create_calls[1] == (
        "image.jpg",
        b"jpeg",
        "image/jpeg",
        "vision",
        {"anchor": "created_at", "seconds": 2_592_000},
        {"model": TEST_LLM_MODEL},
    )
    assert files.create_calls[2] == (
        "notes.txt",
        b"hello world",
        "text/plain",
        "user_data",
        {"anchor": "created_at", "seconds": 2_592_000},
        {"model": TEST_LLM_MODEL},
    )


async def test_openai_file_uploader_drops_failed_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI upload errors degrade to a dropped attachment."""
    errored = _fake_openai_uploader(files=FakeOpenAIFiles(status="error"))
    assert (
        await errored._upload_file(
            filename="bad.txt", data=b"x", content_type="text/plain", purpose="user_data"
        )
        is None
    )

    boom = _fake_openai_uploader(files=FakeOpenAIFiles())

    async def _raise(
        file: tuple[str, BytesIO, str],
        purpose: str,
        expires_after: dict[str, object],
        extra_body: dict[str, object] | None = None,
    ) -> SimpleNamespace:
        del file, purpose, expires_after, extra_body
        raise RuntimeError("upload failed")

    monkeypatch.setattr(boom.client.files, "create", _raise)
    assert (
        await boom._upload_file(
            filename="x.txt", data=b"x", content_type="text/plain", purpose="user_data"
        )
        is None
    )


def test_gpt_attachment_handler_path_stays_disabled() -> None:
    """GPT models still use inline attachments until the OpenAI uploader branch is enabled."""
    assert isinstance(build_attachment_handler(model_name="gpt-5.1"), InlineRenderer)


async def test_non_gemini_answer_model_inlines_attachments() -> None:
    """A non-Gemini answer model inlines attachments instead of using the Gemini Files API."""
    renderer = InlineRenderer()

    # Image -> base64 input_image (no Files API upload).
    image_rendered = await renderer.render_image(
        source=FakeAttachment(
            filename="pic.png", content_type="image/png", payload=base64.b64decode(_png_b64())
        ),
        cache_key="pic.png",
    )
    assert image_rendered is not None
    image_part, _image_expiry = image_rendered
    assert image_part["type"] == "input_image"
    assert image_part["image_url"].startswith("data:image/")
    assert ";base64," in image_part["image_url"]

    # Text/code file -> inlined as input_text with a filename header.
    text_rendered = await renderer.render_file(
        attachment=FakeAttachment(
            filename="notes.txt", content_type="text/plain", payload=b"hello world"
        ),
        cache_key="notes.txt",
    )
    assert text_rendered is not None
    text_part, _text_expiry = text_rendered
    assert text_part["type"] == "input_text"
    assert "hello world" in text_part["text"]
    assert "notes.txt" in text_part["text"]

    # PDF -> inlined as base64 input_file file_data (not a Files-API file_id).
    pdf_rendered = await renderer.render_file(
        attachment=FakeAttachment(
            filename="doc.pdf", content_type="application/pdf", payload=b"%PDF-1.4 fake"
        ),
        cache_key="doc.pdf",
    )
    assert pdf_rendered is not None
    pdf_part, _pdf_expiry = pdf_rendered
    assert pdf_part["type"] == "input_file"
    assert pdf_part["file_data"].startswith("data:application/pdf;base64,")
    assert "file_id" not in pdf_part

    # Non-text, non-PDF binary -> dropped.
    binary_rendered = await renderer.render_file(
        attachment=FakeAttachment(
            filename="blob.bin", content_type="application/octet-stream", payload=b"\x00\x01\xff"
        ),
        cache_key="blob.bin",
    )
    assert binary_rendered is None


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
    current.channel = FakeChannel(history=fake_history)
    raw_history = await cog._fetch_history(message=current, limit=30)
    rendered = await cog._render_history(raw_history, text_only=False)
    assert len(rendered) == 3
    assert rendered[0]["role"] == "system"
    assert [m.content for m in raw_history] == ["hello", "bot answer"]

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


async def test_gen_reply_preserves_bot_mention_in_text_context() -> None:
    """Regression: self-mentions can be the subject of a normal QA message."""
    cog = _cog()
    message = FakeMessage(
        content="你的審美跟 <@999> 一樣 這樣算誇獎嗎", author=FakeAuthor(user_id=1)
    )

    processed = await cog.input_builder.process_single_message(message=message)
    rendered = processed["content"]

    assert isinstance(rendered, str)
    assert "你的審美跟 <@999> 一樣 這樣算誇獎嗎" in rendered


async def test_gen_reply_routes_and_handlers_without_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies route, video, image, and slow-reply handlers using fake APIs."""
    cog = _cog()
    message = FakeMessage(content="make a summary", author=FakeAuthor(user_id=1))
    assert (await _route(cog=cog, message=message)).decision == "SUMMARY"
    assert cog.openai_client.responses.parse_models[0] == cog.runtime_models.route_model.name

    async def fake_sleep(delay: float) -> None:
        """Skips video polling delay."""

    monkeypatch.setattr("discordbot.cogs.gen_reply.asyncio.sleep", fake_sleep)
    await cog._handle_video_reply(
        message=message,
        user_prompt="video",
        context_task=asyncio.create_task(_ready_reply_context()),
    )
    assert len(message.replies) == 1
    # Video has no director: the raw request reaches native generate_videos directly.
    assert cog.gemini_client.generate_prompts == ["video"]

    await cog._handle_image_reply(
        message=message,
        user_prompt="image",
        context_task=asyncio.create_task(_ready_reply_context()),
    )
    assert cog.openai_client.images.generate_calls
    # No director: the raw request reaches images.generate directly.
    assert cog.openai_client.images.generate_prompts == ["image"]
    # The image is delivered first, then a conversational reply streams onto that same
    # message via the flash image_reply_model with no tools.
    assert message.replies[-1].file is not None
    assert (
        cog.openai_client.responses.create_models[-1] == cog.runtime_models.image_reply_model.name
    )
    assert cog.openai_client.responses.create_streams[-1] is True
    assert cog.openai_client.responses.create_tools[-1] is None

    streamed: list[FakeMessage] = []

    class FakeResponder:
        """Records the message handed to the streaming responder."""

        def __init__(  # noqa: PLR0913 -- stub mirrors ResponseStreamer's constructor kwargs
            self,
            message: FakeMessage,
            memory_lookups: list[str] | None = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
            model_effort: str = "",
            voice_synthesizer: object | None = None,
            image_generator: object | None = None,
            music_generator: object | None = None,
        ) -> None:
            """Stores the streaming target message."""
            del memory_lookups, input_tokens, output_tokens, model_effort
            del voice_synthesizer, image_generator, music_generator
            self.message = message

        async def stream(self, *, responses: object) -> str:
            """Records the message and returns placeholder content."""
            del responses
            streamed.append(self.message)
            return "done"

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    # memory_enabled=False keeps this routing test off the real memory path,
    # which is not isolated here.
    await _reply_via_pipeline(
        cog=cog, message=message, system_prompt="system", memory_enabled=False
    )
    assert cog.openai_client.responses.create_streams[-1] is True
    assert streamed[-1] is message


async def test_uploaded_image_without_extension_marks_as_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image attachment whose filename lacks an extension still marks as an image."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
    )
    message = FakeMessage(content="<@999> see", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(
            filename="screenshot",
            content_type="image/png",
            payload=base64.b64decode(_png_b64()),
            url="https://example.test/screenshot",
        )
    ]

    # Classification is by content_type, not filename, so the marker render needs no upload.
    rendered = await cog.input_builder.process_single_message_text_only(message=message)
    parts = rendered["content"]
    assert isinstance(parts, list)
    assert parts[-1]["text"] == "[attachment: image]"


async def test_text_only_and_full_render_agree_on_attachment_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker render and the upload render keep the same supported-attachment slots."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
    )
    message = FakeMessage(content="<@999> mix", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(
            filename="pic.png", content_type="image/png", payload=base64.b64decode(_png_b64())
        ),
        FakeAttachment(filename="clip.mp4", content_type="video/mp4", payload=b"v"),
    ]

    text_only = await cog.input_builder.process_single_message_text_only(message=message)
    full = await cog.input_builder.process_single_message(message=message)

    text_markers = [
        part
        for part in text_only["content"]
        if isinstance(part, dict) and str(part.get("text", "")).startswith("[attachment:")
    ]
    full_files = [
        part
        for part in full["content"]
        if isinstance(part, dict) and part.get("type") == "input_file"
    ]
    assert len(text_markers) == len(full_files) == 1


async def test_text_only_render_degrades_when_modality_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold-start modality lookup failure degrades to empty text, not a pipeline abort."""
    cog = _cog()

    def boom(model_name: str) -> set[str]:
        """Simulates the LiteLLM model-info fetch failing on a cold cache."""
        del model_name
        raise RuntimeError("model info unreachable")

    monkeypatch.setattr("discordbot.cogs._gen_reply.input.get_supported_modalities", boom)
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(filename="pic.png", content_type="image/png", payload=b"x")
    ]

    rendered = await cog.input_builder.process_single_message_text_only(message=message)

    assert rendered == EasyInputMessageParam(role="user", content="")


async def test_handle_image_reply_edits_attached_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """An attached image routes the IMAGE handler through images.edit with raw bytes."""
    cog = _cog()
    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.input.get_supported_modalities", lambda model_name: {"image"}
    )
    message = FakeMessage(content="改這張圖", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(
            filename="pic.png", content_type="image/png", payload=base64.b64decode(_png_b64())
        )
    ]

    await cog._handle_image_reply(
        message=message,
        user_prompt="make it blue",
        context_task=asyncio.create_task(_ready_reply_context()),
    )

    assert cog.openai_client.images.edit_calls == 1
    assert cog.openai_client.images.generate_calls == 0


async def test_handle_image_reply_sends_raw_prompt_to_generate() -> None:
    """The raw request reaches images.generate directly, with no prompt director call."""
    cog = _cog()
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))

    await cog._handle_image_reply(
        message=message,
        user_prompt="draw a cat",
        context_task=asyncio.create_task(_ready_reply_context()),
    )

    # The raw prompt reaches images.generate; the only responses.create is the streaming persona
    # reply (no non-streaming director ran).
    assert cog.openai_client.images.generate_prompts == ["draw a cat"]
    assert cog.openai_client.responses.create_streams == [True]
    assert cog.openai_client.responses.create_models == [cog.runtime_models.image_reply_model.name]


async def test_handle_image_reply_injects_only_user_memory() -> None:
    """The conversational reply carries the requester's memory, never the server memory."""
    cog = _cog()
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))
    context = ReplyContext(
        memory_block=EasyInputMessageParam(role="assistant", content="USER_MEM_MARKER"),
        server_memory_block=EasyInputMessageParam(role="assistant", content="SERVER_MEM_MARKER"),
    )

    async def _ready() -> ReplyContext:
        """Hands the prepared context to the handler."""
        return context

    await cog._handle_image_reply(
        message=message, user_prompt="draw a cat", context_task=asyncio.create_task(_ready())
    )

    # The streamed reply is the last create; only the user memory block rides in it.
    reply_input = cog.openai_client.responses.create_inputs[-1]
    contents = [block.get("content") for block in reply_input]
    assert "USER_MEM_MARKER" in contents
    assert "SERVER_MEM_MARKER" not in contents


async def test_handle_image_reply_best_effort_when_reply_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure producing the conversational reply still leaves the image delivered."""
    cog = _cog()
    message = FakeMessage(content="畫一隻貓", author=FakeAuthor(user_id=1))

    class BoomResponder:
        """Stands in for ResponseStreamer and fails while streaming the reply."""

        def __init__(self, **kwargs: object) -> None:
            """Ignores the streamer kwargs."""
            del kwargs

        async def stream(self, *, responses: object) -> str:
            """Simulates a streaming failure after the image is already delivered."""
            del responses
            raise RuntimeError("stream boom")

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", BoomResponder)

    await cog._handle_image_reply(
        message=message,
        user_prompt="draw a cat",
        context_task=asyncio.create_task(_ready_reply_context()),
    )

    # The image is delivered even though the reply stream raised; the error never surfaced.
    assert message.replies[-1].file is not None


async def test_handle_video_reply_uses_raw_prompt_without_director(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video generation sends the raw request to native generate_videos, with no director call."""
    cog = _cog()

    async def fake_sleep(delay: float) -> None:
        """Skips video polling delay."""

    monkeypatch.setattr("discordbot.cogs.gen_reply.asyncio.sleep", fake_sleep)
    message = FakeMessage(content="拍一段影片", author=FakeAuthor(user_id=1))

    await cog._handle_video_reply(
        message=message,
        user_prompt="video",
        context_task=asyncio.create_task(_ready_reply_context()),
    )

    # The raw prompt reaches generate_videos; the only responses.create is the streaming reply
    # about the video (no non-streaming director ran).
    assert cog.gemini_client.generate_prompts == ["video"]
    assert cog.openai_client.responses.create_streams == [True]
    assert cog.openai_client.responses.create_models == [cog.runtime_models.video_reply_model.name]
    # The reply watches the generated video: it is referenced as an uploaded input_file part.
    reply_parts = cog.openai_client.responses.create_inputs[0][-1]["content"]
    assert any(part.get("type") == "input_file" for part in reply_parts)
    # No attachments: a plain text-to-video generation at the configured 1080p, MP4 delivered.
    config = cog.gemini_client.generate_configs[0]
    assert config.resolution == "1080p"
    assert config.reference_images is None
    assert message.replies[-1].file is not None


async def test_handle_video_reply_includes_video_thumbnail_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A referenced video's poster frame is wired in as an asset reference image."""
    cog = _cog()

    async def fake_sleep(delay: float) -> None:
        """Skips video polling delay."""

    async def fake_thumbs(builder: object, message: object) -> list[tuple[bytes, str]]:
        """Returns a fake video poster frame for the message."""
        del builder, message
        return [(b"poster", "image/jpeg")]

    monkeypatch.setattr("discordbot.cogs.gen_reply.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.input.MessageInputBuilder.get_video_thumbnail_sources",
        fake_thumbs,
    )
    message = FakeMessage(content="把這部影片做成新的", author=FakeAuthor(user_id=1))

    await cog._handle_video_reply(
        message=message,
        user_prompt="video",
        context_task=asyncio.create_task(_ready_reply_context()),
    )

    config = cog.gemini_client.generate_configs[0]
    assert len(config.reference_images) == 1


async def test_handle_video_reply_passes_reference_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attached images ride as asset reference images (capped at three)."""
    cog = _cog()

    async def fake_sleep(delay: float) -> None:
        """Skips video polling delay."""

    monkeypatch.setattr("discordbot.cogs.gen_reply.asyncio.sleep", fake_sleep)
    message = FakeMessage(content="把這些做成影片", author=FakeAuthor(user_id=1))
    message.attachments = [
        FakeAttachment(
            filename=f"pic{index}.png",
            content_type="image/png",
            payload=base64.b64decode(_png_b64()),
        )
        for index in range(4)
    ]

    await cog._handle_video_reply(
        message=message,
        user_prompt="video",
        context_task=asyncio.create_task(_ready_reply_context()),
    )

    # All attached images ride as asset reference images, capped at three.
    config = cog.gemini_client.generate_configs[0]
    assert len(config.reference_images) == 3
    assert all(ref.image.image_bytes for ref in config.reference_images)


@pytest.mark.parametrize(
    argnames="content",
    argvalues=[
        "整理懶人包 https://example.test/post",
        "這裡面又在說啥 整理給我聽 https://example.test/post",
    ],
)
async def test_gen_reply_routes_url_summary_requests_to_qa(content: str) -> None:
    """Regression: URL summaries should use the normal QA route, not chat SUMMARY."""
    cog = _cog()
    message = FakeMessage(content=content, author=FakeAuthor(user_id=1))

    routed = await _route(cog=cog, message=message)
    assert routed.decision == "QA"
    assert cog.openai_client.responses.parse_models[0] == cog.runtime_models.route_model.name


@pytest.mark.parametrize(
    argnames=(
        "route",
        "expected_call",
        "expected_prep",
        "expected_flags",
        "expected_voice",
        "expected_image",
        "expected_music",
    ),
    argvalues=[
        ("IMAGE", "_handle_image_reply", [(30, True)], [], [], [], []),
        ("VIDEO", "_handle_video_reply", [(30, True)], [], [], [], []),
        (
            "SUMMARY",
            "_handle_message_reply",
            [(30, True), (100, False)],
            [False],
            [True],
            [False],
            [False],
        ),
        ("QA", "_handle_message_reply", [(30, True)], [True], [True], [True], [True]),
    ],
)
async def test_gen_reply_on_message_dispatches_routes(  # noqa: PLR0913, PLR0915 -- parametrized columns; orchestrates per-route stubs
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    expected_call: str,
    expected_prep: list[tuple[int, bool]],
    expected_flags: list[bool],
    expected_voice: list[bool],
    expected_image: list[bool],
    expected_music: list[bool],
) -> None:
    """Verifies on_message dispatches each route to the expected handler."""
    cog = _cog()
    # Distinctive non-fallback grade so the effort reaching the answer model is checked to
    # be the graded value, not the "high" default that timeout/error would also produce.
    cog.openai_client.responses.effort_parsed = EffortGrade(effort="low")
    calls: list[str] = []
    prompts: list[str] = []
    prep_requests: list[tuple[int, bool]] = []
    prepared_context = ReplyContext()

    async def fake_route(
        message: FakeMessage, reference_messages: list[object], current_message: list[object]
    ) -> RouteClassification:
        """Returns the parametrized route."""
        del reference_messages, current_message
        # Yield like a real route I/O call so the speculative prep task gets scheduled.
        await asyncio.sleep(0)
        return RouteClassification(decision=route)

    async def fake_prepare(  # noqa: PLR0913 -- mirrors _prepare_reply_context's signature
        message: FakeMessage,
        history_limit: int,
        memory_enabled: bool,
        parts_task: object,
        text_parts: object,
        route_done: object,
    ) -> ReplyContext:
        """Records context requests while staying off the memory and history paths."""
        del message, parts_task, text_parts, route_done
        prep_requests.append((history_limit, memory_enabled))
        return prepared_context

    async def fake_reaction(
        message: FakeMessage, bot_user: object, emoji: str, previous: str | None = None
    ) -> str:
        """Records reaction state transitions."""
        calls.append(f"reaction:{emoji}")
        return emoji

    async def fake_image_handler(
        message: FakeMessage, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Records image handler dispatch and drains the handed-over context task."""
        del message
        await context_task
        prompts.append(user_prompt)
        calls.append("_handle_image_reply")

    async def fake_video_handler(
        message: FakeMessage, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Records video handler dispatch and drains the handed-over context task."""
        del message
        await context_task
        prompts.append(user_prompt)
        calls.append("_handle_video_reply")

    memory_flags: list[bool] = []
    voice_flags: list[bool] = []
    image_flags: list[bool] = []
    music_flags: list[bool] = []
    effort_flags: list[str] = []
    contexts: list[ReplyContext] = []

    async def fake_message_handler(  # noqa: PLR0913 -- stub mirrors _handle_message_reply's signature
        message: FakeMessage,
        system_prompt: str,
        context: ReplyContext,
        memory_enabled: bool = True,
        effort: str = "high",
        allow_voice: bool = False,
        allow_image: bool = False,
        allow_music: bool = False,
        allow_research: bool = False,
        yt_url: str | None = None,
    ) -> None:
        """Records slow message handler dispatch."""
        del yt_url, allow_research
        calls.append("_handle_message_reply")
        memory_flags.append(memory_enabled)
        voice_flags.append(allow_voice)
        image_flags.append(allow_image)
        music_flags.append(allow_music)
        effort_flags.append(effort)
        contexts.append(context)

    monkeypatch.setattr(cog, "_route_classify", fake_route)
    monkeypatch.setattr(cog, "_prepare_reply_context", fake_prepare)
    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", fake_reaction)
    monkeypatch.setattr(cog, "_handle_image_reply", fake_image_handler)
    monkeypatch.setattr(cog, "_handle_video_reply", fake_video_handler)
    monkeypatch.setattr(cog, "_handle_message_reply", fake_message_handler)

    message = FakeMessage(content="<@!999> hello", author=FakeAuthor(user_id=1))
    await cog.on_message(message=message)
    assert expected_call in calls
    assert calls[-1] == "reaction:<:greencheck:1517565102424068226>"
    # The speculative QA context always builds first; SUMMARY rebuilds at its own
    # history depth without memory, and QA consumes the speculative context as-is.
    assert prep_requests == expected_prep
    assert memory_flags == expected_flags
    # Voice is enabled on QA and SUMMARY (both stream a reply); IMAGE/VIDEO never reach here.
    assert voice_flags == expected_voice
    # Inline image is QA-only; SUMMARY stays text and IMAGE/VIDEO never reach here.
    assert image_flags == expected_image
    # Inline music is QA-only, like inline image; SUMMARY stays text.
    assert music_flags == expected_music
    if route in {"IMAGE", "VIDEO"}:
        assert prompts == ["hello"]
        assert effort_flags == []
    else:
        assert contexts == [prepared_context]
        # The parallel grade flows end-to-end into the answer model on QA and SUMMARY.
        assert effort_flags == ["low"]


async def test_prepare_reply_context_shields_shared_parts_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the speculative prep must not cancel the shared upload task.

    A SUMMARY route cancels the speculative prep while still reusing `parts_task`; an
    unshielded `await parts_task` inside prep would propagate the cancellation and make
    the rebuilt summary context fail with CancelledError.
    """
    cog = _cog()
    release = asyncio.Event()

    async def slow_parts() -> tuple[list[object], list[object]]:
        """Stands in for an upload still activating when the route is decided."""
        await release.wait()
        return ([], [])

    async def fake_history(message: FakeMessage, limit: int) -> list[object]:
        """Returns empty history so prep parks directly on the shared parts task."""
        del message, limit
        return []

    monkeypatch.setattr(cog, "_fetch_history", fake_history)
    parts_task = asyncio.create_task(coro=slow_parts())
    prep_task = asyncio.create_task(
        coro=cog._prepare_reply_context(
            message=FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1)),
            history_limit=100,
            memory_enabled=False,
            parts_task=parts_task,
            text_parts=([], []),
            route_done=asyncio.Event(),
        )
    )
    # Let prep run its empty history and park on `await asyncio.shield(parts_task)`.
    for _ in range(5):
        await asyncio.sleep(0)

    await _discard_task(task=prep_task)

    assert not parts_task.cancelled()
    release.set()
    reference_messages, current_message = await parts_task
    assert (reference_messages, current_message) == ([], [])


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

    async def boom(
        message: FakeMessage, reference_messages: list[object], current_message: list[object]
    ) -> str:
        """Raises to exercise error handling."""
        del reference_messages, current_message
        raise RuntimeError("boom")

    async def fake_prepare(  # noqa: PLR0913 -- mirrors _prepare_reply_context's signature
        message: FakeMessage,
        history_limit: int,
        memory_enabled: bool,
        parts_task: object,
        text_parts: object,
        route_done: object,
    ) -> ReplyContext:
        """Keeps the speculative prep off the real memory and history paths."""
        del message, history_limit, memory_enabled, parts_task, text_parts, route_done
        return ReplyContext()

    monkeypatch.setattr(cog, "_route_classify", boom)
    monkeypatch.setattr(cog, "_prepare_reply_context", fake_prepare)
    failed = FakeMessage(content="<@999> fail", author=FakeAuthor(user_id=1))
    await cog.on_message(message=failed)
    assert failed.replies[0].content is None

    # Source deleted before the error embed lands: it falls back to an unparented send.
    deleted = FakeMessage(content="<@999> fail", author=FakeAuthor(user_id=1))
    deleted.reply_error = _deleted_source_error()
    await cog.on_message(message=deleted)
    assert deleted.replies == []
    assert deleted.channel.sent[0].embed is not None


async def test_reaction_status_chain_orders_and_replaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advance schedules ordered swaps without blocking; flush waits for the tail."""
    events: list[tuple[str, str | None]] = []

    async def fake_reaction(
        message: FakeMessage, bot_user: object, emoji: str, previous: str | None = None
    ) -> str:
        """Records each scheduled reaction swap."""
        del message, bot_user
        events.append((emoji, previous))
        return emoji

    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", fake_reaction)
    chain = ReactionStatusChain(
        message=FakeMessage(content="hi"), bot_user=SimpleNamespace(id=999)
    )
    chain.advance(emoji="🔀")
    chain.advance(emoji="❓")
    chain.advance(emoji="🆗")
    assert events == []  # nothing awaited yet: scheduling never blocks the caller
    await chain.flush()
    assert events == [("🔀", None), ("❓", "🔀"), ("🆗", "❓")]


async def test_on_message_consumes_speculative_context_on_image_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The IMAGE route hands its speculative context to the image handler, not discards it."""
    cog = _cog()
    prepared = ReplyContext()
    received: list[ReplyContext] = []

    async def fake_route(
        message: FakeMessage, reference_messages: list[object], current_message: list[object]
    ) -> RouteClassification:
        """Routes every message to IMAGE."""
        del reference_messages, current_message
        # Yield like a real route I/O call so the speculative prep task starts.
        await asyncio.sleep(0)
        return RouteClassification(decision="IMAGE")

    async def fake_prepare(  # noqa: PLR0913 -- mirrors _prepare_reply_context's signature
        message: FakeMessage,
        history_limit: int,
        memory_enabled: bool,
        parts_task: object,
        text_parts: object,
        route_done: object,
    ) -> ReplyContext:
        """Returns the prepared context the image handler should consume."""
        del message, history_limit, memory_enabled, parts_task, text_parts, route_done
        return prepared

    async def fake_image_handler(
        message: FakeMessage, user_prompt: str, context_task: asyncio.Task[ReplyContext]
    ) -> None:
        """Records the context the dispatch handed over."""
        del message, user_prompt
        received.append(await context_task)

    async def fake_reaction(
        message: FakeMessage, bot_user: object, emoji: str, previous: str | None = None
    ) -> str:
        """Skips real reaction calls."""
        del message, bot_user, previous
        return emoji

    monkeypatch.setattr(cog, "_route_classify", fake_route)
    monkeypatch.setattr(cog, "_prepare_reply_context", fake_prepare)
    monkeypatch.setattr(cog, "_handle_image_reply", fake_image_handler)
    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", fake_reaction)

    message = FakeMessage(content="<@!999> draw", author=FakeAuthor(user_id=1))
    await cog.on_message(message=message)
    assert received == [prepared]


class _ThreadsStreamer:
    """Answer-phase streamer stub returning a fixed reply without real streaming."""

    def __init__(  # noqa: PLR0913 -- stub mirrors ResponseStreamer's constructor kwargs
        self,
        message: FakeMessage,
        memory_lookups: list[str] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model_effort: str = "",
        voice_synthesizer: object | None = None,
        image_generator: object | None = None,
        music_generator: object | None = None,
    ) -> None:
        """Stores the streaming target message and ignores the rest."""
        del memory_lookups, input_tokens, output_tokens, model_effort
        del voice_synthesizer, image_generator, music_generator
        self.message = message

    async def stream(self, *, responses: object) -> str:
        """Returns placeholder reply content."""
        del responses
        return "完整回覆"


async def _silent_reaction(
    message: FakeMessage, bot_user: object, emoji: str, previous: str | None = None
) -> str:
    """Skips real reaction calls during pipeline integration tests."""
    del message, bot_user, previous
    return emoji


def _threads_block(body: str = "MOCK THREADS POST BODY") -> list[dict[str, object]]:
    """Builds a builder-shaped Threads block: the real separator plus a user content message."""
    return [
        {"role": "system", "content": [{"type": "input_text", "text": THREADS_CONTEXT_SEPARATOR}]},
        {"role": "user", "content": [{"type": "input_text", "text": body}]},
    ]


async def test_on_message_injects_threads_context_before_current(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A QA message with a Threads URL injects the parsed post just before the current message."""
    cog = _cog()
    cog.openai_client.responses.output_parsed = RouteClassification(decision="QA")
    cog.config = SimpleNamespace(
        voice_reply_enabled=False,
        inline_image_enabled=False,
        music_available=False,
        deep_research_enabled=False,
    )
    seen_urls: list[str] = []

    async def fake_builder(*, url: str, answer_model_is_gemini: bool) -> list[dict[str, object]]:
        """Returns a recognizable Threads block instead of hitting the network."""
        del answer_model_is_gemini
        seen_urls.append(url)
        return _threads_block()

    monkeypatch.setattr("discordbot.cogs.gen_reply.build_threads_context_messages", fake_builder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", _ThreadsStreamer)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **_: None)
    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", _silent_reaction)

    url = "https://www.threads.com/@a/post/ABC123"
    message = FakeMessage(content=f"<@999> what is this {url}", author=FakeAuthor(user_id=1))
    await cog.on_message(message=message)

    assert seen_urls == [url]
    answer = request_input(responses=cog.openai_client.responses, phase="answer")
    assert has_threads_context_block(request=answer)
    assert extract_threads_context_block(request=answer) == "MOCK THREADS POST BODY"

    # The block lands after memory but before the current message (which stays last).
    headers = [text.split("\n", 1)[0] for _role, text in iter_text_blocks(request=answer)]
    separator_index = headers.index(THREADS_CONTEXT_SEPARATOR.split("\n", 1)[0])
    current_index = next(
        index for index, head in enumerate(headers) if head.startswith("==== Current Message")
    )
    assert separator_index < current_index


async def test_on_message_cancels_threads_context_on_image_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-QA route cancels the in-flight Threads parse instead of orphaning it."""
    cog = _cog()
    cancelled: list[bool] = []

    async def hanging_builder(
        *, url: str, answer_model_is_gemini: bool
    ) -> list[dict[str, object]]:
        """Blocks until cancelled, recording the cancellation."""
        del url, answer_model_is_gemini
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return []

    async def fake_route(
        message: FakeMessage, reference_messages: list[object], current_message: list[object]
    ) -> RouteClassification:
        """Routes every message to IMAGE after yielding so the parse starts."""
        del reference_messages, current_message
        await asyncio.sleep(0)
        return RouteClassification(decision="IMAGE")

    async def fake_image_handler(message: FakeMessage, user_prompt: str) -> None:
        """Accepts the dispatched image request."""
        del message, user_prompt

    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.build_threads_context_messages", hanging_builder
    )
    monkeypatch.setattr(cog, "_route_classify", fake_route)
    monkeypatch.setattr(cog, "_handle_image_reply", fake_image_handler)
    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", _silent_reaction)

    message = FakeMessage(
        content="<@999> draw https://www.threads.com/@a/post/ABC123", author=FakeAuthor(user_id=1)
    )
    await cog.on_message(message=message)
    assert cancelled == [True]


async def test_on_message_skips_threads_context_without_url(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A message with no Threads URL never starts the parse and injects no block."""
    cog = _cog()
    cog.openai_client.responses.output_parsed = RouteClassification(decision="QA")
    cog.config = SimpleNamespace(
        voice_reply_enabled=False,
        inline_image_enabled=False,
        music_available=False,
        deep_research_enabled=False,
    )
    called: list[str] = []

    async def fake_builder(*, url: str, answer_model_is_gemini: bool) -> list[dict[str, object]]:
        """Records any call so the test can assert it never runs."""
        del answer_model_is_gemini
        called.append(url)
        return _threads_block()

    monkeypatch.setattr("discordbot.cogs.gen_reply.build_threads_context_messages", fake_builder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", _ThreadsStreamer)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **_: None)
    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", _silent_reaction)

    message = FakeMessage(content="<@999> just a plain question", author=FakeAuthor(user_id=1))
    await cog.on_message(message=message)

    assert called == []
    assert not has_threads_context_block(
        request=request_input(responses=cog.openai_client.responses, phase="answer")
    )


async def test_on_message_threads_context_grace_timeout_injects_notice(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parse slower than the post-route grace injects a timeout notice; the answer streams."""
    cog = _cog()
    cog.openai_client.responses.output_parsed = RouteClassification(decision="QA")
    cog.config = SimpleNamespace(
        voice_reply_enabled=False,
        inline_image_enabled=False,
        music_available=False,
        deep_research_enabled=False,
    )
    monkeypatch.setattr("discordbot.cogs.gen_reply.THREADS_GRACE_SECONDS", 0.01)

    async def slow_builder(*, url: str, answer_model_is_gemini: bool) -> list[dict[str, object]]:
        """Outlasts the grace so the gate drops it."""
        del url, answer_model_is_gemini
        await asyncio.sleep(5)
        return _threads_block()

    monkeypatch.setattr("discordbot.cogs.gen_reply.build_threads_context_messages", slow_builder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", _ThreadsStreamer)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **_: None)
    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", _silent_reaction)

    message = FakeMessage(
        content="<@999> what is this https://www.threads.com/@a/post/ABC123",
        author=FakeAuthor(user_id=1),
    )
    await cog.on_message(message=message)

    # The slow parse is dropped, but a deterministic timeout notice keeps the model from
    # claiming it cannot open the link, and the answer still streams.
    answer = request_input(responses=cog.openai_client.responses, phase="answer")
    assert has_threads_context_block(request=answer)
    assert "did not respond in time" in str(answer)


def test_reply_context_message_list_orders_hist_ref_current() -> None:
    """message_list keeps transcript order: history, reference, current."""
    context = ReplyContext(
        hist_messages=[{"role": "system", "content": "hist"}],
        reference_messages=[{"role": "system", "content": "ref"}],
        current_message=[{"role": "user", "content": "now"}],
    )
    assert [part["content"] for part in context.message_list] == ["hist", "ref", "now"]


async def test_handle_message_reply_orders_reference_after_memory_before_current(
    economy_isolated_db: None, memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer input puts memory first, then the reference message, then the current message.

    The reference (the message being replied to) rides just above the current message so the
    reply pair stays adjacent and reads as the primary context, and the strengthened headers
    spell out the reply relationship.
    """
    del economy_isolated_db, memory_isolated_dir
    cog = _cog()
    write_main_memory(
        scope=user_scope(user_id=1),
        content="v1\n\n## 使用者輪廓\n喜歡簡短回覆",
        identity="U1 (u1) [id: 1]",
    )
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **kwargs: None)
    monkeypatch.setattr("discordbot.cogs.gen_reply.Message", FakeMessage)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    parent_author = FakeAuthor(user_id=4)
    parent_author.name, parent_author.display_name = "parent", "Parent"
    parent = FakeMessage(content="原訊息", author=parent_author)
    parent.id = 988
    message.reference = FakeReference(resolved=parent)

    cog.openai_client.responses.select_queue = [
        [_function_call_item(call_id="c0", arguments=json.dumps({"user_id_list": ["1"]}))]
    ]
    cog.openai_client.responses.stream_queue = [
        [_text_event(delta="好"), _completed_event(input_tokens=1, output_tokens=1)]
    ]

    await _reply_via_pipeline(cog=cog, message=message)

    answer = request_input(responses=cog.openai_client.responses, phase="answer")
    blocks = list(iter_text_blocks(request=answer))
    memory_index = next(
        index
        for index, (role, text) in enumerate(blocks)
        if role == "assistant" and text.startswith("(My long-term memory about participants")
    )
    reference_index = next(
        index
        for index, (_role, text) in enumerate(blocks)
        if text.startswith("==== Reference Message")
    )
    current_index = next(
        index
        for index, (_role, text) in enumerate(blocks)
        if text.startswith("==== Current Message")
    )
    assert memory_index < reference_index < current_index
    assert "directly replying to this message" in blocks[reference_index][1]
    assert "reply to the Reference Message above" in blocks[current_index][1]


def test_model_settings_and_config_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies model properties and provider-specific tool dispatch."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    catalog = RuntimeModelCatalog()
    cog = ReplyGeneratorCogs(bot=SimpleNamespace(user=SimpleNamespace(id=999)))
    assert cog.runtime_models.fast_model == catalog.fast_model
    assert isinstance(catalog.fast_model, ModelSettings)
    assert "image" in catalog.image_model.name
    assert catalog.video_model.name.startswith("veo")
    assert catalog.slow_model.effort == "high"
    # Code execution is omitted on purpose: it 400s the request on file attachments.
    assert ModelSettings(name="gemini-test").tools == [{"googleSearch": {}}, {"urlContext": {}}]
    assert ModelSettings(name="claude-test").tools == [
        {"type": "web_search_20260209", "name": "web_search"},
        {"type": "web_fetch_20260209", "name": "web_fetch"},
    ]
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
    assert peak_start[0] == ModelSettings(name="gemini-3.1-pro-preview", effort="high")
    assert peak_start[0] == peak_end[0] == before_peak[0] == after_peak[0] == weekend[0]


async def test_handle_message_reply_selection_offers_tool_then_answers_with_builtins(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selection phase offers get_user_memory + callable users; the answer phase keeps built-ins."""
    cog = _cog()
    write_main_memory(
        scope=user_scope(user_id=1),
        content="v1\n\n## 使用者輪廓\n喜歡簡短回覆",
        identity="Tester (tester) [id: 1]",
    )

    class FakeResponder:
        """Stands in for the answer-phase streamer without real streaming."""

        def __init__(  # noqa: PLR0913 -- stub mirrors ResponseStreamer's constructor kwargs
            self,
            message: FakeMessage,
            memory_lookups: list[str] | None = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
            model_effort: str = "",
            voice_synthesizer: object | None = None,
            image_generator: object | None = None,
            music_generator: object | None = None,
        ) -> None:
            """Stores the streaming target message."""
            del memory_lookups, input_tokens, output_tokens, model_effort
            del voice_synthesizer, image_generator, music_generator
            self.message = message

        async def stream(self, *, responses: object) -> str:
            """Returns placeholder reply content."""
            del responses
            return "完整回覆"

    scheduled: list[dict[str, object]] = []

    def fake_schedule(**kwargs: object) -> None:
        """Records the scheduled memory update arguments."""
        scheduled.append(kwargs)

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", fake_schedule)

    # The selection model declines (no calls), so nothing is injected into the answer.
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await _reply_via_pipeline(cog=cog, message=message)

    # Two requests: selection (non-streaming) then the answer (streaming).
    assert cog.openai_client.responses.create_streams == [False, True]

    # Selection runs on the fast tool_model; only the answer pays for slow_model.
    assert cog.openai_client.responses.create_models == [
        cog.runtime_models.tool_model.name,
        cog.runtime_models.slow_model.name,
    ]

    # Selection request offers only get_user_memory and lists the author as callable.
    selection_idx = request_index(responses=cog.openai_client.responses, phase="selection")
    assert tool_names_for_call(responses=cog.openai_client.responses, n=selection_idx) == [
        "get_user_memory"
    ]
    assert extract_callable_user_ids(
        request=request_input(responses=cog.openai_client.responses, phase="selection")
    ) == {1}
    assert cog.openai_client.responses.create_instructions[selection_idx] == MEMORY_SELECT_PROMPT

    # Answer request keeps the built-in tools (no get_user_memory) and the clean persona: the
    # author declined selection, so their stored memory is not injected.
    answer_idx = request_index(responses=cog.openai_client.responses, phase="answer")
    assert "get_user_memory" not in tool_names_for_call(
        responses=cog.openai_client.responses, n=answer_idx
    )
    _assert_runtime_time_context(
        instructions=cog.openai_client.responses.create_instructions[answer_idx],
        system_prompt="SYS",
    )
    assert not has_memory_context_block(
        request=request_input(responses=cog.openai_client.responses, phase="answer")
    )

    # Extraction still scheduled for the author with a memory-free, tool-free list.
    scheduled_list = scheduled[0]["message_list"]
    assert isinstance(scheduled_list, list)
    assert "get_user_memory" not in str(scheduled_list)
    assert "喜歡簡短回覆" not in str(scheduled_list)
    assert scheduled[0]["scope"] == user_scope(user_id=1)
    assert scheduled[0]["full_reply"] == "完整回覆"
    assert scheduled[0]["extractor"] is cog.memory_extractor
    assert scheduled[0]["identity"] == "Tester (tester) [id: 1]"
    assert cog.memory_extractor.extract_model.name == cog.runtime_models.extract_model.name
    assert (
        cog.memory_extractor.evaluate_model.name == cog.runtime_models.memory_evaluator_model.name
    )
    assert cog.memory_extractor.consolidate_model.name == cog.runtime_models.memories_model.name


async def test_handle_message_reply_without_stored_memory_keeps_instructions(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies a memory-less user gets untouched instructions but still schedules."""
    cog = _cog()

    class FakeResponder:
        """Stands in for the answer-phase streamer without real streaming."""

        def __init__(  # noqa: PLR0913 -- stub mirrors ResponseStreamer's constructor kwargs
            self,
            message: FakeMessage,
            memory_lookups: list[str] | None = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
            model_effort: str = "",
            voice_synthesizer: object | None = None,
            image_generator: object | None = None,
            music_generator: object | None = None,
        ) -> None:
            """Stores the streaming target message."""
            del memory_lookups, input_tokens, output_tokens, model_effort
            del voice_synthesizer, image_generator, music_generator
            self.message = message

        async def stream(self, *, responses: object) -> str:
            """Returns placeholder reply content."""
            del responses
            return "回覆"

    scheduled: list[object] = []

    def fake_schedule(**kwargs: object) -> None:
        """Records that a memory update was scheduled."""
        scheduled.append(kwargs["scope"])

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", fake_schedule)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await _reply_via_pipeline(cog=cog, message=message)

    # The selection phase still offers the tool even when nobody has stored memory; the
    # answer phase keeps the clean persona and the built-in tools.
    selection_idx = request_index(responses=cog.openai_client.responses, phase="selection")
    answer_idx = request_index(responses=cog.openai_client.responses, phase="answer")
    assert "get_user_memory" in tool_names_for_call(
        responses=cog.openai_client.responses, n=selection_idx
    )
    _assert_runtime_time_context(
        instructions=cog.openai_client.responses.create_instructions[answer_idx],
        system_prompt="SYS",
    )
    assert "get_user_memory" not in tool_names_for_call(
        responses=cog.openai_client.responses, n=answer_idx
    )
    assert scheduled == [user_scope(user_id=1), server_scope(bot_id=999, server_id=1)]


async def test_handle_message_reply_memory_disabled_arg_skips_user_memory(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies memory_enabled=False (summary route) skips user memory but still records server."""
    cog = _cog()
    write_main_memory(
        scope=user_scope(user_id=1),
        content="v1\n\n## 使用者輪廓\n不該被注入",
        identity="Tester (tester) [id: 1]",
    )

    class FakeResponder:
        """Stands in for the answer-phase streamer without real streaming."""

        def __init__(  # noqa: PLR0913 -- stub mirrors ResponseStreamer's constructor kwargs
            self,
            message: FakeMessage,
            memory_lookups: list[str] | None = None,
            input_tokens: int = 0,
            output_tokens: int = 0,
            model_effort: str = "",
            voice_synthesizer: object | None = None,
            image_generator: object | None = None,
            music_generator: object | None = None,
        ) -> None:
            """Stores the streaming target message."""
            del memory_lookups, input_tokens, output_tokens, model_effort
            del voice_synthesizer, image_generator, music_generator
            self.message = message

        async def stream(self, *, responses: object) -> str:
            """Returns placeholder reply content."""
            del responses
            return "回覆"

    scheduled: list[object] = []

    def fake_schedule(**kwargs: object) -> None:
        """Records that a memory update was scheduled."""
        scheduled.append(kwargs["scope"])

    monkeypatch.setattr("discordbot.cogs.gen_reply.ResponseStreamer", FakeResponder)
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", fake_schedule)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await _reply_via_pipeline(cog=cog, message=message, memory_enabled=False)

    # memory_enabled=False runs no selection phase: a single answer request, no tool, no memory.
    assert cog.openai_client.responses.create_streams == [True]
    answer = request_input(responses=cog.openai_client.responses, phase="answer")
    assert not has_memory_context_block(request=answer)
    assert "get_user_memory" not in tool_names_for_call(responses=cog.openai_client.responses, n=0)
    # The per-user update is skipped, but the server-scope update still runs in a public guild.
    assert scheduled == [server_scope(bot_id=999, server_id=1)]


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


def test_build_memory_allowlist_collects_authors_and_mentions_excluding_bot() -> None:
    """Authors and mentioned users are collected, deduped, and the bot is excluded."""
    author = FakeAuthor(user_id=1)
    mentioned = FakeAuthor(user_id=2)
    mentioned.name = "alice"
    mentioned.display_name = "Alice"
    bot = FakeAuthor(user_id=999)

    msg_with_mentions = FakeMessage(author=author)
    msg_with_mentions.mentions = [mentioned, bot]
    duplicate_author = FakeMessage(author=author)
    bot_authored = FakeMessage(author=bot)

    allowed = build_memory_allowlist(
        messages=[msg_with_mentions, duplicate_author, bot_authored], bot_user_id=999
    )

    # Insertion order preserved, bot (999) excluded from both author and mention slots.
    assert list(allowed.keys()) == [1, 2]
    assert allowed[1] == "Tester (tester)"
    assert allowed[2] == "Alice (alice)"


def test_build_memory_allowlist_escapes_mention_labels() -> None:
    """Mention syntax in a display name is neutralized so a label cannot ping."""
    author = FakeAuthor(user_id=1)
    author.display_name = "@everyone"
    allowed = build_memory_allowlist(messages=[FakeMessage(author=author)], bot_user_id=999)

    # The active @everyone is broken (zero-width space) while the text survives.
    assert "@everyone" not in allowed[1]
    assert "everyone" in allowed[1]


def test_parse_user_id_list_handles_valid_and_malformed() -> None:
    """Valid payloads parse to string ids; malformed payloads degrade to an empty list."""
    assert parse_user_id_list(arguments='{"user_id_list": ["1", "2"]}') == ["1", "2"]
    assert parse_user_id_list(arguments='{"user_id_list": [1, 2]}') == ["1", "2"]
    assert parse_user_id_list(arguments="not json") == []
    assert parse_user_id_list(arguments='{"other": 1}') == []
    assert parse_user_id_list(arguments='{"user_id_list": "nope"}') == []


def test_resolve_user_memories_enforces_allowlist(memory_isolated_dir: object) -> None:
    """Ids outside the allowlist drop, mention wrappers and dupes collapse, gaps signal clearly."""
    del memory_isolated_dir
    write_main_memory(
        scope=user_scope(user_id=1),
        content="v1\n\n## 使用者輪廓\n甲的記憶",
        identity="A (a) [id: 1]",
    )
    allowed = {1: "A (a)", 2: "B (b)"}

    memories = resolve_user_memories(user_id_list=["1", "<@1>", "3", "abc", "2"], allowed=allowed)

    by_id = {memory.user_id: memory for memory in memories}
    assert set(by_id) == {"1", "2"}
    assert "甲的記憶" in by_id["1"].memory
    assert by_id["1"].username == "A (a)"
    assert by_id["2"].memory == "(no stored memory for this user)"


@pytest.mark.parametrize(
    (
        "seeded",
        "server_nick",
        "mention_ids",
        "reference_author_id",
        "channel_public",
        "select_id_lists",
        "expected_injected",
        "callable_includes",
        "callable_excludes",
    ),
    [
        ({1: "喜歡被叫阿狗"}, None, [], None, True, [["1"]], {1}, {1}, set()),
        ({}, None, [], None, True, [["1"]], set(), {1}, set()),
        ({1: "機密"}, None, [], None, True, [], set(), {1}, set()),
        ({42: "機密外人記憶"}, None, [], None, True, [["42"]], set(), {1}, {42}),
        ({1: "甲記憶", 2: "乙記憶"}, None, [2], None, True, [["1"], ["2"]], {1, 2}, {1, 2}, set()),
        (
            {uid: f"記憶{uid}" for uid in range(1, 11)},
            None,
            list(range(2, 11)),
            None,
            True,
            [[str(uid) for uid in range(1, 11)]],
            set(range(1, 9)),
            {1},
            set(),
        ),
        ({}, None, [], 7, True, [], set(), {1, 7}, set()),
        ({42: "李董的祕密"}, (42, "Boss", "李董"), [], None, True, [["42"]], {42}, {42}, set()),
        ({42: "李董的祕密"}, (42, "Boss", "李董"), [], None, False, [["42"]], set(), {1}, {42}),
    ],
    ids=[
        "inject-selected-memory",
        "no-stored-memory",
        "selection-declines",
        "non-allowlisted-id-dropped",
        "multiple-selection-calls",
        "caps-injected-memories",
        "reference-author-callable",
        "nickname-table-widens-public",
        "nickname-table-no-widen-private",
    ],
)
async def test_handle_message_reply_user_memory_injection(  # noqa: PLR0913 -- parametrized columns
    economy_isolated_db: None,
    memory_isolated_dir: object,
    monkeypatch: pytest.MonkeyPatch,
    seeded: dict[int, str],
    server_nick: tuple[int, str, str] | None,
    mention_ids: list[int],
    reference_author_id: int | None,
    channel_public: bool,
    select_id_lists: list[list[str]],
    expected_injected: set[int],
    callable_includes: set[int],
    callable_excludes: set[int],
) -> None:
    """The answer gets exactly the allowlisted, selected memories; everything else is dropped.

    One matrix over the user-memory boundary: a plain selection, an empty/declined selection, an
    id outside the conversation, multiple calls, the per-reply cap, a reference author joining the
    allowlist, and the public-only nickname-table widening. Injection is asserted by id
    (extract_user_memory_blocks) and the allowlist by the ids offered to the selection model
    (extract_callable_user_ids), never by a sentinel substring over a serialized blob.
    """
    del economy_isolated_db, memory_isolated_dir
    cog = _cog()
    for uid, body in seeded.items():
        write_main_memory(
            scope=user_scope(user_id=uid),
            content=f"v1\n\n## 使用者輪廓\n{body}",
            identity=f"U{uid} (u{uid}) [id: {uid}]",
        )
    if server_nick is not None:
        nick_id, nick_name, nick_alias = server_nick
        write_main_memory(
            scope=server_scope(bot_id=999, server_id=1),
            content=f"v1\n\n## 成員稱呼\n* {nick_name}(社群暱稱:{nick_alias})[id: {nick_id}]",
            identity="Test Guild [id: 1]",
        )
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **kwargs: None)
    if reference_author_id is not None:
        monkeypatch.setattr("discordbot.cogs.gen_reply.Message", FakeMessage)

    message = FakeMessage(
        content="<@999> hi", author=FakeAuthor(user_id=1), channel_public=channel_public
    )
    message.mentions = [FakeAuthor(user_id=uid) for uid in mention_ids]
    if reference_author_id is not None:
        parent_author = FakeAuthor(user_id=reference_author_id)
        parent_author.name, parent_author.display_name = "parent", "Parent"
        parent = FakeMessage(content="原訊息", author=parent_author)
        parent.id = 988
        message.reference = FakeReference(resolved=parent)

    cog.openai_client.responses.select_queue = [
        [
            _function_call_item(call_id=f"c{index}", arguments=json.dumps({"user_id_list": ids}))
            for index, ids in enumerate(select_id_lists)
        ]
    ]
    cog.openai_client.responses.stream_queue = [
        [_text_event(delta="好"), _completed_event(input_tokens=1, output_tokens=1)]
    ]

    await _reply_via_pipeline(cog=cog, message=message)

    answer = request_input(responses=cog.openai_client.responses, phase="answer")
    # An allowlisted-but-memoryless user gets a placeholder block, not a leak; the boundary is
    # which ids' real memory reaches the model, so placeholder sections are filtered out.
    injected = {
        uid
        for uid, body in extract_user_memory_blocks(request=answer).items()
        if body != NO_STORED_MEMORY
    }
    assert injected == expected_injected
    # The current user message stays last so the model answers it, and no internal selection
    # artifact (a function_call_output) ever leaks into the answer request.
    assert isinstance(answer, list)
    assert answer[-1].get("role") == "user"
    assert not any(
        isinstance(item, dict) and item.get("type") == "function_call_output" for item in answer
    )

    callable_ids = extract_callable_user_ids(
        request=request_input(responses=cog.openai_client.responses, phase="selection")
    )
    assert callable_includes <= callable_ids
    assert callable_excludes.isdisjoint(callable_ids)


@pytest.mark.parametrize(
    (
        "seeded_ids",
        "mentions",
        "select_id_lists",
        "select_usage",
        "stream_usage",
        "present",
        "absent",
        "credited_once",
    ),
    [
        (
            [1],
            [],
            [["1"]],
            None,
            (5, 6),
            ["⬆ 5 ⬇ 6", "\n-# <:tag:1517563887573143595> Tester (tester) 的記憶"],
            [],
            None,
        ),
        ([1], [], [["1"]], (100, 20), (5, 6), ["⬆ 105 ⬇ 26"], [], None),
        (
            [1, 2, 3],
            [(2, "alice", "Alice"), (3, "bob", "Bob")],
            [["1", "2", "3"]],
            None,
            (1, 1),
            ["\n-# <:tag:1517563887573143595> Tester (tester), Alice (alice) 等 3 人的記憶"],
            [],
            None,
        ),
        (
            [1],
            [],
            [["1"], ["1"]],
            None,
            (1, 1),
            ["\n-# <:tag:1517563887573143595> Tester (tester) 的記憶"],
            [],
            "Tester (tester)",
        ),
        ([], [], [["1"]], None, (5, 6), [], ["<:tag:1517563887573143595>"], None),
    ],
    ids=[
        "single-owner-credit",
        "selection-usage-folded-in",
        "owners-collapse-past-two",
        "repeat-lookups-credited-once",
        "no-memory-no-credit",
    ],
)
async def test_handle_message_reply_memory_footer(  # noqa: PLR0913 -- parametrized columns
    economy_isolated_db: None,
    memory_isolated_dir: object,
    monkeypatch: pytest.MonkeyPatch,
    seeded_ids: list[int],
    mentions: list[tuple[int, str, str]],
    select_id_lists: list[list[str]],
    select_usage: tuple[int, int] | None,
    stream_usage: tuple[int, int],
    present: list[str],
    absent: list[str],
    credited_once: str | None,
) -> None:
    """The footer credits the memory owners actually read and folds selection tokens into usage.

    Reads the user-visible reply text (the feature's small, real output surface): the single-owner
    credit, the selection-request token contribution, the collapse to "等 N 人" past two owners,
    repeat-lookup de-duplication, and the no-credit case.
    """
    del economy_isolated_db, memory_isolated_dir
    cog = _cog()
    labels = {1: "Tester (tester)", 2: "Alice (alice)", 3: "Bob (bob)"}
    for uid in seeded_ids:
        write_main_memory(
            scope=user_scope(user_id=uid),
            content=f"v1\n\n## 使用者輪廓\n記憶{uid}",
            identity=f"{labels[uid]} [id: {uid}]",
        )
    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **kwargs: None)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    mention_authors: list[FakeAuthor] = []
    for uid, name, display in mentions:
        author = FakeAuthor(user_id=uid)
        author.name, author.display_name = name, display
        mention_authors.append(author)
    message.mentions = mention_authors

    if select_usage is not None:
        cog.openai_client.responses.select_usage = SimpleNamespace(
            input_tokens=select_usage[0], output_tokens=select_usage[1]
        )
    cog.openai_client.responses.select_queue = [
        [
            _function_call_item(call_id=f"c{index}", arguments=json.dumps({"user_id_list": ids}))
            for index, ids in enumerate(select_id_lists)
        ]
    ]
    cog.openai_client.responses.stream_queue = [
        [
            _text_event(delta="好"),
            _completed_event(input_tokens=stream_usage[0], output_tokens=stream_usage[1]),
        ]
    ]

    await _reply_via_pipeline(cog=cog, message=message)

    content = message.replies[0].content or ""
    for fragment in present:
        assert fragment in content
    for fragment in absent:
        assert fragment not in content
    if credited_once is not None:
        assert content.count(credited_once) == 1


async def test_handle_message_reply_falls_back_to_author_memory_when_selection_fails(
    economy_isolated_db: None, memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing memory-selection request still replies and falls back to the author's own memory."""
    del economy_isolated_db, memory_isolated_dir
    cog = _cog()
    write_main_memory(
        scope=user_scope(user_id=1),
        content="v1\n\n## 使用者輪廓\n甲",
        identity="Tester (tester) [id: 1]",
    )

    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", lambda **kwargs: None)

    async def boom(**kwargs: object) -> object:
        """Simulates a selection-request failure."""
        del kwargs
        raise RuntimeError("selection provider error")

    monkeypatch.setattr(cog, "_select_user_memories", boom)

    cog.openai_client.responses.stream_queue = [
        [_text_event(delta="照常回答"), _completed_event(input_tokens=5, output_tokens=6)]
    ]

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await _reply_via_pipeline(cog=cog, message=message)

    # The answer request still ran, and the author's own memory was injected as the fallback.
    assert (message.replies[0].content or "").startswith("照常回答")
    answer = request_input(responses=cog.openai_client.responses, phase="answer")
    assert "甲" in (extract_user_memory_blocks(request=answer).get(1) or "")


def test_usage_footer_re_strips_memory_credit_second_line() -> None:
    """The optional second -# memory line is stripped together with the usage footer."""
    body = "答案內容"
    double = "\n\n-# model · ⬆ 1 ⬇ 2 · $0.00000000 · +3\n-# <:tag:1517563887573143595> Tester (tester) 的記憶"
    assert USAGE_FOOTER_RE.sub("", f"{body}{double}") == body
    # Backward compatible: a single-line footer still strips cleanly.
    single = "\n\n-# model · ⬆ 1 ⬇ 2 · $0.00000000 · +3"
    assert USAGE_FOOTER_RE.sub("", f"{body}{single}") == body


@pytest.mark.parametrize(
    ("memory_enabled", "has_guild", "channel_public", "expect_server_read", "expect_scopes"),
    [
        (True, True, True, True, ["user", "server"]),
        (True, True, False, True, ["user"]),
        (True, False, True, False, ["user"]),
        (False, True, True, False, ["server"]),
        (False, False, True, False, []),
        (False, True, False, False, []),
    ],
    ids=[
        "qa-guild-public",
        "qa-guild-private",
        "qa-dm",
        "summary-guild-public",
        "summary-dm",
        "summary-guild-private",
    ],
)
async def test_handle_message_reply_server_memory_gating(  # noqa: PLR0913 -- parametrized columns
    economy_isolated_db: None,
    memory_isolated_dir: object,
    monkeypatch: pytest.MonkeyPatch,
    memory_enabled: bool,
    has_guild: bool,
    channel_public: bool,
    expect_server_read: bool,
    expect_scopes: list[str],
) -> None:
    """Server memory is read on a guild QA turn and written only from a public guild channel.

    One matrix over (route, guild/DM, public/private): the read block rides the answer (and the
    selection request) only on a memory-enabled guild turn; the per-user write follows
    memory_enabled; the per-server write needs a public guild channel. Read is asserted
    structurally via extract_server_memory_block, writes via the scheduled scopes.
    """
    del economy_isolated_db, memory_isolated_dir
    cog = _cog()
    write_main_memory(
        scope=server_scope(bot_id=999, server_id=1),
        content="v1\n\n## 伺服器輪廓\n社群風格",
        identity="Test Guild [id: 1]",
    )
    scheduled: list[dict[str, object]] = []

    def fake_schedule(**kwargs: object) -> None:
        """Records each scheduled memory update."""
        scheduled.append(kwargs)

    monkeypatch.setattr("discordbot.cogs.gen_reply.schedule_memory_update", fake_schedule)

    message = FakeMessage(
        content="<@999> hi", author=FakeAuthor(user_id=1), channel_public=channel_public
    )
    if not has_guild:
        message.guild = None
    cog.openai_client.responses.stream_queue = [
        [_text_event(delta="好"), _completed_event(input_tokens=1, output_tokens=1)]
    ]

    await _reply_via_pipeline(cog=cog, message=message, memory_enabled=memory_enabled)

    answer = request_input(responses=cog.openai_client.responses, phase="answer")
    assert (extract_server_memory_block(request=answer) is not None) == expect_server_read

    server_scope_value = server_scope(bot_id=999, server_id=1)
    name_to_scope = {"user": user_scope(user_id=1), "server": server_scope_value}
    assert [update["scope"] for update in scheduled] == [
        name_to_scope[name] for name in expect_scopes
    ]
    for update in scheduled:
        if update["scope"] == server_scope_value:
            assert update["subject"] == "target_server_id: 1"
            assert update["extractor"] is cog.server_memory_extractor
            assert update["identity"] == "Test Guild [id: 1]"
            assert cog.server_memory_extractor.phase1_prompt is SERVER_PHASE1_PROMPT
            assert cog.server_memory_extractor.consolidate_prompt is SERVER_PHASE2_PROMPT

    # On a memory-enabled guild turn the selection request also sees the server memory so it can
    # resolve nicknames; non-guild or SUMMARY turns run no selection phase.
    if memory_enabled and has_guild:
        selection = request_input(responses=cog.openai_client.responses, phase="selection")
        assert extract_server_memory_block(request=selection) is not None


def test_allowlist_ids_from_server_memory_parses_nickname_table() -> None:
    """Only ids under the `## 成員稱呼` section are returned, labelled by the table row."""
    memory = (
        "v1\n\n## 伺服器輪廓\n社群\n\n"
        "## 成員稱呼\n"
        "* Mai(社群暱稱:李董、破貓親爹)[id: 123]\n"
        "* Bob(社群暱稱:阿伯)[id: 456]\n\n"
        "## 近期脈絡\n* [2026-06-10] 某人 [id: 789] 提到活動\n"
    )
    allowed = allowlist_ids_from_server_memory(memory=memory)
    assert set(allowed) == {123, 456}
    assert "李董" in allowed[123]
    assert "[id:" not in allowed[123]
    # An id outside the nickname section (e.g. in 近期脈絡) is never exposed.
    assert 789 not in allowed


def test_widen_allowlist_with_aliases_merges_participant_labels() -> None:
    """A participant keeps their label and gains aliases; absent members are added."""
    memory = (
        "v1\n\n## 成員稱呼\n"
        "* Mai(社群暱稱:李董、破貓親爹)[id: 123]\n"
        "* Bob(社群暱稱:阿伯)[id: 456]\n"
    )
    allowed = {123: "Mai (mai9999)"}
    widen_allowlist_with_aliases(allowed=allowed, memory=memory, include_absent=True)

    # The conversation label leads and the table row rides behind it on the same line.
    assert allowed[123].startswith("Mai (mai9999)")
    assert "李董" in allowed[123]
    # A member absent from the conversation is added with the table row as label.
    assert "阿伯" in allowed[456]


def test_widen_allowlist_with_aliases_skips_absent_when_not_public() -> None:
    """Without include_absent, participants are still enriched but absent members stay out.

    A private channel must not gain read access to an absent member's personal memory by
    naming a public nickname, even though the nickname table itself is public content.
    """
    memory = (
        "v1\n\n## 成員稱呼\n"
        "* Mai(社群暱稱:李董、破貓親爹)[id: 123]\n"
        "* Bob(社群暱稱:阿伯)[id: 456]\n"
    )
    allowed = {123: "Mai (mai9999)"}
    widen_allowlist_with_aliases(allowed=allowed, memory=memory, include_absent=False)

    # The present participant is still enriched with community aliases.
    assert allowed[123].startswith("Mai (mai9999)")
    assert "李董" in allowed[123]
    # The absent member is not added, so their personal memory stays unreachable here.
    assert 456 not in allowed


async def test_streamer_reasoning_preview_then_content_overwrites() -> None:
    """The reasoning preview renders as -# subtext and real content replaces it in place."""
    message = FakeMessage()
    streamer = ResponseStreamer(message=message)
    streamer.reasoning_content = "first thought\n\nsecond thought"

    await streamer._write_preview_snapshot()
    assert len(message.replies) == 1
    preview = message.replies[0].content
    assert isinstance(preview, str)
    assert preview.splitlines()[0] == "-# <:message:1517560873000898860> Thinking..."
    assert "-# first thought" in preview
    assert "-# second thought" in preview

    streamer.content_started = True
    streamer.stored_content = "real answer"
    await streamer._write_preview_snapshot()
    assert len(message.replies) == 1
    assert message.replies[0].content == "real answer"


def test_streamer_reasoning_preview_keeps_newest_lines_within_limit() -> None:
    """A long think keeps only its newest tail lines under the Discord limit."""
    streamer = ResponseStreamer(message=FakeMessage())
    streamer.reasoning_content = "\n".join(f"thought line {i} " + "x" * 80 for i in range(60))

    preview = streamer._render_preview()

    assert len(preview) <= DISCORD_MESSAGE_LIMIT
    lines = preview.splitlines()
    assert lines[0] == "-# <:message:1517560873000898860> Thinking..."
    assert all(line.startswith("-# ") for line in lines)
    assert "thought line 59" in preview
    assert "thought line 9 " not in preview


def test_streamer_reasoning_preview_escapes_mentions() -> None:
    """Transient thought text can never ping people or roles."""
    streamer = ResponseStreamer(message=FakeMessage())
    streamer.reasoning_content = "should I ping @everyone or <@123456789012345678>?"

    preview = streamer._render_preview()

    assert "@everyone" not in preview
    assert "<@123456789012345678>" not in preview


async def test_streamer_strips_leading_newlines_from_first_reasoning_delta(
    economy_isolated_db: None,
) -> None:
    """Gemini's leading reasoning newlines are dropped like content newlines."""
    del economy_isolated_db
    events = [
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="\n\n"),
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="\nthought"),
        _text_event(delta="answer"),
        _completed_event(input_tokens=1, output_tokens=1),
    ]
    streamer = ResponseStreamer(message=FakeMessage())

    await streamer.stream(responses=_stream_events_from(events=events))

    assert streamer.reasoning_content == "thought"


async def test_streamer_edits_are_time_throttled(economy_isolated_db: None) -> None:
    """The snapshot editor writes far fewer Discord edits than stream deltas."""
    del economy_isolated_db
    message = FakeMessage()

    async def _events() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(type="response.reasoning_summary_text.delta", delta="thinking hard")
        await asyncio.sleep(0.06)
        for index in range(40):
            yield SimpleNamespace(type="response.output_text.delta", delta=f"chunk{index} ")
            await asyncio.sleep(0.002)
        yield _completed_event(input_tokens=1, output_tokens=1)

    streamer = ResponseStreamer(message=message, preview_interval_seconds=0.02)
    result = await streamer.stream(responses=_events())

    assert len(message.replies) == 1
    reply = message.replies[0]
    assert 1 + len(reply.edits) < 40
    assert result.startswith("chunk0 ")
    assert isinstance(reply.content, str)
    assert reply.content.startswith("chunk0 ")


async def test_streamer_footer_shows_route_effort(economy_isolated_db: None) -> None:
    """The usage footer labels the model with the route-decided effort."""
    del economy_isolated_db
    message = FakeMessage()

    result = await ResponseStreamer(message=message, model_effort="low").stream(
        responses=_stream_events()
    )

    assert f"\n\n-# {TEST_LLM_MODEL} (low) · ⬆ 12 ⬇ 34" in result
    assert USAGE_FOOTER_RE.sub("", result) == "hello from stream"


async def test_route_classify_carries_decision_and_defaults_qa() -> None:
    """The route classifies the reply mode; unparsed output falls back to QA."""
    cog = _cog()
    cog.openai_client.responses.output_parsed = RouteClassification(decision="IMAGE")
    message = FakeMessage(content="draw a cat", author=FakeAuthor(user_id=1))
    assert (await _route(cog=cog, message=message)).decision == "IMAGE"

    cog.openai_client.responses.output_parsed = None
    assert (await _route(cog=cog, message=message)).decision == "QA"


async def test_route_url_summary_downgrades_to_qa() -> None:
    """A SUMMARY classification on a message carrying a URL is steered back to QA."""
    cog = _cog()
    cog.openai_client.responses.output_parsed = RouteClassification(decision="SUMMARY")
    message = FakeMessage(content="整理 https://example.test/a", author=FakeAuthor(user_id=1))

    assert (await _route(cog=cog, message=message)).decision == "QA"


async def test_grade_effort_carries_grade_and_defaults_high() -> None:
    """The effort grader returns the model's grade; unparsed output falls back to high."""
    cog = _cog()
    cog.openai_client.responses.effort_parsed = EffortGrade(effort="low")
    message = FakeMessage(content="hi", author=FakeAuthor(user_id=1))
    assert (await _grade(cog=cog, message=message)).effort == "low"

    cog.openai_client.responses.effort_parsed = None
    assert (await _grade(cog=cog, message=message)).effort == "high"


async def test_resolve_effort_returns_graded_effort_on_success() -> None:
    """A completed grade flows through _resolve_effort as the answer model's effort."""
    cog = _cog()
    route_done = asyncio.Event()
    route_done.set()

    async def graded() -> EffortGrade:
        """Returns a non-default grade so the success path is pinned."""
        return EffortGrade(effort="low")

    effort_task = asyncio.create_task(coro=graded())
    assert (
        await cog._resolve_effort(
            message=FakeMessage(), effort_task=effort_task, route_done=route_done
        )
        == "low"
    )


async def test_resolve_effort_defaults_high_on_error() -> None:
    """A failed effort grade resolves to high effort rather than stalling the reply."""
    cog = _cog()
    route_done = asyncio.Event()
    route_done.set()

    async def boom() -> EffortGrade:
        """Fails the grade to exercise the fallback."""
        raise RuntimeError("boom")

    effort_task = asyncio.create_task(coro=boom())
    assert (
        await cog._resolve_effort(
            message=FakeMessage(), effort_task=effort_task, route_done=route_done
        )
        == "high"
    )


async def test_resolve_effort_defaults_high_on_grace_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grade still running past the post-route grace resolves to high effort."""
    cog = _cog()
    monkeypatch.setattr("discordbot.cogs.gen_reply.EFFORT_GRACE_SECONDS", 0.01)
    route_done = asyncio.Event()
    route_done.set()

    async def slow() -> EffortGrade:
        """Outlives the grace window."""
        await asyncio.sleep(30)
        return EffortGrade(effort="low")

    effort_task = asyncio.create_task(coro=slow())
    assert (
        await cog._resolve_effort(
            message=FakeMessage(), effort_task=effort_task, route_done=route_done
        )
        == "high"
    )


async def test_on_message_cancels_effort_task_on_image_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The IMAGE route cancels the parallel effort grade it will never consume."""
    cog = _cog()
    cancelled: list[bool] = []

    async def fake_route(
        message: FakeMessage, reference_messages: list[object], current_message: list[object]
    ) -> RouteClassification:
        """Routes every message to IMAGE after yielding so the effort task starts."""
        del reference_messages, current_message
        await asyncio.sleep(0)
        return RouteClassification(decision="IMAGE")

    async def fake_grade(
        message: FakeMessage, reference_messages: list[object], current_message: list[object]
    ) -> EffortGrade:
        """Blocks until cancelled, recording the cancellation."""
        del message, reference_messages, current_message
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return EffortGrade(effort="low")

    async def fake_prepare(  # noqa: PLR0913 -- mirrors _prepare_reply_context's signature
        message: FakeMessage,
        history_limit: int,
        memory_enabled: bool,
        parts_task: object,
        text_parts: object,
        route_done: object,
    ) -> ReplyContext:
        """Keeps the speculative prep off the real memory and history paths."""
        del message, history_limit, memory_enabled, parts_task, text_parts, route_done
        return ReplyContext()

    async def fake_image_handler(message: FakeMessage, user_prompt: str) -> None:
        """Accepts the dispatched image request."""
        del message, user_prompt

    async def fake_reaction(
        message: FakeMessage, bot_user: object, emoji: str, previous: str | None = None
    ) -> str:
        """Skips real reaction calls."""
        del message, bot_user, previous
        return emoji

    monkeypatch.setattr(cog, "_route_classify", fake_route)
    monkeypatch.setattr(cog, "_grade_effort", fake_grade)
    monkeypatch.setattr(cog, "_prepare_reply_context", fake_prepare)
    monkeypatch.setattr(cog, "_handle_image_reply", fake_image_handler)
    monkeypatch.setattr("discordbot.utils.reactions.update_reaction", fake_reaction)

    message = FakeMessage(content="<@!999> draw", author=FakeAuthor(user_id=1))
    await cog.on_message(message=message)
    assert cancelled == [True]


async def test_handle_message_reply_uses_route_effort(economy_isolated_db: None) -> None:
    """The answer request's reasoning effort follows the route decision."""
    del economy_isolated_db
    cog = _cog()
    message = FakeMessage(content="<@999> why", author=FakeAuthor(user_id=1))

    await _reply_via_pipeline(cog=cog, message=message, memory_enabled=False, effort="low")

    assert cog.openai_client.responses.create_reasonings[-1]["effort"] == "low"


async def test_route_input_excludes_attachment_payloads() -> None:
    """The route request sees an attachment marker instead of the file payload."""
    cog = _cog()
    message = FakeMessage(content="<@999> see", author=FakeAuthor(user_id=1))
    message.attachments = [FakeAttachment(filename="note.txt", content_type="text/plain")]

    await _route(cog=cog, message=message)

    rendered = str(cog.openai_client.responses.parse_inputs[-1])
    assert "input_file" not in rendered
    assert "[attachment: file]" in rendered


async def test_select_user_memories_uses_text_only_transcript() -> None:
    """The selection request carries the text-only transcript verbatim, no payloads."""
    cog = _cog()
    cog.openai_client.responses.select_queue = [[]]
    message_list = [
        EasyInputMessageParam(
            role="user",
            content=[
                {"type": "input_text", "text": "user (u) [id: 1]: look"},
                {"type": "input_text", "text": "[attachment: image]"},
            ],
        )
    ]

    await cog._select_user_memories(
        message=FakeMessage(), message_list=message_list, allowed={1: "u"}
    )

    rendered = str(cog.openai_client.responses.create_inputs[-1])
    assert "input_image" not in rendered
    assert "input_file" not in rendered
    assert "[attachment: image]" in rendered


async def test_attachment_parts_cached_until_message_changes() -> None:
    """Rendered attachment parts are cached per message and refresh on edit."""
    cog = _cog()
    message = FakeMessage(content="doc", author=FakeAuthor(user_id=2))
    attachment = FakeAttachment(filename="note.txt", content_type="text/plain")
    message.attachments = [attachment]

    first = await cog.input_builder.get_attachment_parts(message=message)
    again = await cog.input_builder.get_attachment_parts(message=message)

    assert attachment.read_count == 1
    assert again == first

    message.edited_at = datetime.now(tz=UTC)
    await cog.input_builder.get_attachment_parts(message=message)
    assert attachment.read_count == 2


async def test_attachment_cache_reuploads_expired_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached file_id past its real expiry is re-rendered, not served stale."""
    cog = _cog()
    builder = cog.input_builder
    message = FakeMessage(content="doc", author=FakeAuthor(user_id=2))
    attachment = FakeAttachment(filename="note.txt", content_type="text/plain")
    message.attachments = [attachment]

    await builder.get_attachment_parts(message=message)
    assert attachment.read_count == 1

    # Within expiry: the cached handle is reused, so no second download.
    await builder.get_attachment_parts(message=message)
    assert attachment.read_count == 1

    # Force the entry past its stored expiry: the next render re-downloads and re-uploads.
    (cache_key, (_expiry, cached_parts)) = next(iter(builder._attachment_cache.items()))
    builder._attachment_cache[cache_key] = (datetime(2000, 1, 1, tzinfo=UTC), cached_parts)
    await builder.get_attachment_parts(message=message)
    assert attachment.read_count == 2


async def test_attachment_cache_refreshes_on_embed_url_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late embed unfurl swapping an image URL at constant count re-renders."""
    cog = _cog()
    message = FakeMessage(content="link", author=FakeAuthor(user_id=2))
    rendered_urls: list[str] = []

    async def fake_render_image(
        self: object, source: object, cache_key: object, allow_dead_cache: bool = False
    ) -> tuple[dict[str, str], datetime]:
        """Records each rendered source instead of hitting the network."""
        del self, cache_key, allow_dead_cache
        rendered_urls.append(str(source))
        return {"type": "input_image", "image_url": str(source)}, datetime(2099, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(
        "discordbot.cogs._gen_reply.attachment.gemini_file_api.GeminiFileUploader.render_image",
        fake_render_image,
    )

    def _embed(url: str) -> SimpleNamespace:
        """Builds a fake embed whose image carries a swappable proxy URL."""
        return SimpleNamespace(image=SimpleNamespace(proxy_url=url, url=url), thumbnail=None)

    message.embeds = [_embed("https://media.test/a.png")]
    await cog.input_builder.get_attachment_parts(message=message)
    await cog.input_builder.get_attachment_parts(message=message)
    assert rendered_urls == ["https://media.test/a.png"]

    # Same embed count, different image URL: the cache must not serve the stale part.
    message.embeds = [_embed("https://media.test/b.png")]
    await cog.input_builder.get_attachment_parts(message=message)
    assert rendered_urls == ["https://media.test/a.png", "https://media.test/b.png"]


async def _prepare_context_with_hanging_selection(
    cog: ReplyGeneratorCogs, message: FakeMessage, monkeypatch: pytest.MonkeyPatch
) -> ReplyContext:
    """Builds reply context where selection hangs past the grace, so the fallback fires."""
    monkeypatch.setattr("discordbot.cogs.gen_reply.MEMORY_SELECT_GRACE_SECONDS", 0.01)

    async def slow_selection(**kwargs: object) -> None:
        """Simulates a proxy hang far past the selection grace."""
        del kwargs
        await asyncio.sleep(1)

    monkeypatch.setattr(cog, "_select_user_memories", slow_selection)
    parts_task = asyncio.create_task(coro=cog._get_reference_and_current(message=message))
    text_parts = await cog._get_reference_and_current(message=message, text_only=True)
    # The route has already returned, so selection gets only the tiny grace before it times out.
    route_done = asyncio.Event()
    route_done.set()
    return await cog._prepare_reply_context(
        message=message,
        history_limit=2,
        memory_enabled=True,
        parts_task=parts_task,
        text_parts=text_parts,
        route_done=route_done,
    )


async def test_memory_selection_timeout_falls_back_to_author_memory(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selection slower than the grace falls back to the message author's own memory."""
    del memory_isolated_dir
    cog = _cog()
    write_main_memory(
        scope=user_scope(user_id=1),
        content="v1\n\n## 使用者輪廓\n甲",
        identity="Tester (tester) [id: 1]",
    )
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))

    context = await _prepare_context_with_hanging_selection(
        cog=cog, message=message, monkeypatch=monkeypatch
    )

    assert context.memory_block is not None
    assert "甲" in (extract_user_memory_blocks(request=[context.memory_block]).get(1) or "")
    assert context.memory_labels


async def test_memory_selection_timeout_without_author_memory_injects_nothing(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback injects nothing when the author has no stored memory."""
    del memory_isolated_dir
    cog = _cog()
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))

    context = await _prepare_context_with_hanging_selection(
        cog=cog, message=message, monkeypatch=monkeypatch
    )

    assert context.memory_block is None
    assert context.memory_labels == []


async def test_memory_selection_timeout_falls_back_to_author_and_reference_memory(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reply makes the fallback read both the author and the referenced message's author."""
    del memory_isolated_dir
    cog = _cog()
    write_main_memory(
        scope=user_scope(user_id=1),
        content="v1\n\n## 使用者輪廓\n甲",
        identity="Author (author) [id: 1]",
    )
    write_main_memory(
        scope=user_scope(user_id=2),
        content="v1\n\n## 使用者輪廓\n乙",
        identity="Parent (parent) [id: 2]",
    )
    # _walk_reference_chain only follows a resolved message that passes isinstance(_, Message).
    monkeypatch.setattr("discordbot.cogs.gen_reply.Message", FakeMessage)
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    parent = FakeMessage(content="原訊息", author=FakeAuthor(user_id=2))
    parent.id = 988
    message.reference = FakeReference(resolved=parent)

    context = await _prepare_context_with_hanging_selection(
        cog=cog, message=message, monkeypatch=monkeypatch
    )

    assert context.memory_block is not None
    blocks = extract_user_memory_blocks(request=[context.memory_block])
    assert "甲" in (blocks.get(1) or "")
    assert "乙" in (blocks.get(2) or "")
    assert len(context.memory_labels) == 2


def test_can_launch_research_requires_guild_text_channel() -> None:
    text = SimpleNamespace(guild=object(), channel=MagicMock(spec=nextcord.TextChannel))
    assert _can_launch_research(message=text) is True  # type: ignore[arg-type]  # SimpleNamespace stub
    thread = SimpleNamespace(guild=object(), channel=MagicMock(spec=nextcord.Thread))
    assert _can_launch_research(message=thread) is False  # type: ignore[arg-type]  # SimpleNamespace stub
    dm = SimpleNamespace(guild=None, channel=MagicMock(spec=nextcord.TextChannel))
    assert _can_launch_research(message=dm) is False  # type: ignore[arg-type]  # SimpleNamespace stub


async def test_resume_memory_reenqueues_jobs_and_sweeps_other_scopes(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """on_ready resume re-enqueues persisted jobs (by flavor) and sweeps every over-threshold scope."""
    cog = _cog(bot_user_id=999)
    cog._tasks = set()
    cog._resume_started = False
    user_sentinel = object()
    server_sentinel = object()
    cog.__dict__["memory_extractor"] = user_sentinel
    cog.__dict__["server_memory_extractor"] = server_sentinel

    user_job_scope = user_scope(user_id=1)
    server_job_scope = server_scope(bot_id=999, server_id=2)
    sweep_scope = user_scope(user_id=3)
    jobs = [
        memory_db.MemoryJob(
            scope=user_job_scope,
            flavor="user",
            subject="target_user_id: 1",
            transcript="u-transcript",
            identity="id-u",
            status="failed",
            token=11,
            last_error="boom",
        ),
        memory_db.MemoryJob(
            scope=server_job_scope,
            flavor="server",
            subject="target_server_id: 2",
            transcript="s-transcript",
            identity="id-s",
            status="pending",
            token=22,
            last_error=None,
        ),
    ]
    resumed: list[dict[str, object]] = []
    swept: list[str] = []

    async def fake_list() -> list[memory_db.MemoryJob]:
        return jobs

    def fake_resume(**kwargs: object) -> None:
        resumed.append(kwargs)

    async def fake_consolidate(scope: str, extractor: object, identity: str) -> None:
        swept.append(scope)

    monkeypatch.setattr("discordbot.cogs.gen_reply.safe_list_resumable", fake_list)
    monkeypatch.setattr("discordbot.cogs.gen_reply.resume_memory_update", fake_resume)
    monkeypatch.setattr("discordbot.cogs.gen_reply.consolidate_if_needed", fake_consolidate)
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.iter_scopes",
        lambda: [user_job_scope, server_job_scope, sweep_scope],
    )
    monkeypatch.setattr("discordbot.cogs.gen_reply.needs_consolidation", lambda scope: True)
    monkeypatch.setattr("discordbot.cogs.gen_reply.read_main_identity", lambda scope: "")

    await cog._resume_memory()
    # Wait for spawned sweep tasks to finish.
    while cog._tasks:
        await asyncio.gather(*list(cog._tasks))

    assert {kwargs["scope"] for kwargs in resumed} == {user_job_scope, server_job_scope}
    by_scope = {kwargs["scope"]: kwargs for kwargs in resumed}
    assert by_scope[user_job_scope]["extractor"] is user_sentinel
    assert by_scope[user_job_scope]["token"] == 11
    assert by_scope[server_job_scope]["extractor"] is server_sentinel
    # Every over-threshold scope is swept, including the resumed ones: the scope
    # lock makes the resumed extraction and the consolidation sweep idempotent.
    assert set(swept) == {user_job_scope, server_job_scope, sweep_scope}


async def test_on_ready_resume_runs_once(
    memory_isolated_dir: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """on_ready guards the resume so a gateway reconnect does not re-sweep."""
    cog = _cog(bot_user_id=999)
    cog._tasks = set()
    cog._resume_started = False
    calls = 0

    async def fake_resume_memory() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(cog, "_resume_memory", fake_resume_memory)
    await cog.on_ready()
    await cog.on_ready()
    while cog._tasks:
        await asyncio.gather(*list(cog._tasks))
    assert calls == 1
