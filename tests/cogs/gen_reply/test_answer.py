"""Tests for the QA answer turn."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from nextcord.enums import InteractionContextType
from openai.types.responses.response_input_param import EasyInputMessageParam
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.utils.llm_transcript import USAGE_FOOTER_RE
from discordbot.cogs.gen_reply.answer import (
    AnswerTurn,
    count_media_parts,
    build_runtime_instructions,
)
from discordbot.services.memory.store import user_scope, server_scope
from discordbot.cogs.gen_reply.context import ReplyContext
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.services.memory.writer import RawMemoryDraft
from discordbot.services.memory.prompts import PHASE1_EVALUATOR_PROMPT
from discordbot.cogs.gen_reply.streaming import ResponseStreamer
from discordbot.cogs.gen_reply.generation import ImageGenerator, VoiceGenerator
from discordbot.cogs.gen_reply.status_marks import YOUTUBE_EMOJI
from discordbot.services.memory.server_prompts import SERVER_PHASE1_EVALUATOR_PROMPT

from tests.helpers.memory import drain_memory_turns, wait_for_persisted_writes
from tests.helpers.casting import as_message
from tests.helpers.gen_reply import FakeGeminiClient, interactions_turn_events
from tests.helpers.reply_fakes import (
    FakeAuthor,
    FakeMessage,
    _cog,
    _answer,
    _recorded,
    _config_stub,
    _run_pipeline,
    _install_streamer,
    _no_retry_backoff,
    _stream_events_from,
    _default_turn_events,
    _mid_stream_unavailable,
    _assert_runtime_time_context,
)
from tests.helpers.link_sources import hosting_off_planner
from tests.helpers.ask_interaction import ASKER_ID, GUILD_ID, _toolkit, _interaction
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai.types.responses import ResponseStreamEvent
    from openai.types.responses.response_input_param import ResponseInputParam


def test_build_runtime_instructions_adds_request_time_context() -> None:
    """Request time context uses Discord's message creation timestamp."""
    message = FakeMessage(content="hi")

    instructions = build_runtime_instructions(
        system_prompt="SYS", message=as_message(fake=message), guild_id=1
    )

    _assert_runtime_time_context(instructions=instructions, system_prompt="SYS")


def test_build_runtime_instructions_names_conversation_location() -> None:
    """Instructions carry the guild id for a guild message and the DM marker otherwise.

    Deliberately id-only: the guild NAME is owner-controlled text and this block rides
    the developer-authority instructions, so it must never appear there.
    """
    guild_message = FakeMessage(content="hi")
    instructions = build_runtime_instructions(
        system_prompt="SYS", message=as_message(fake=guild_message), guild_id=424242
    )
    assert "424242" in instructions
    assert "Test Guild" not in instructions

    dm_message = FakeMessage(content="hi")
    dm_message.guild = None
    dm_instructions = build_runtime_instructions(
        system_prompt="SYS", message=as_message(fake=dm_message), guild_id=None
    )
    assert "424242" not in dm_instructions
    assert dm_instructions != instructions


@pytest.mark.parametrize(argnames="guild_id", argvalues=[424242, None], ids=["guild", "dm"])
@pytest.mark.usefixtures("no_memory_review")
async def test_the_answer_is_told_where_its_turn_happens(guild_id: int | None) -> None:
    """The QA answer's instructions name the turn's own server off the surface, or a DM."""
    cog = _cog()
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    message.guild = None
    typed_message = as_message(fake=message)
    surface = TurnSurface(message=typed_message, guild_id=guild_id)

    await _answer(cog=cog, message=typed_message, surface=surface).stream_answer(
        system_prompt="SYS", context=ReplyContext()
    )

    (instructions,) = _recorded(cog).responses.create_instructions
    assert ("424242" in instructions) is (guild_id is not None)


@pytest.mark.parametrize(
    ("flag", "kwarg", "generator_type"),
    [
        ("inline_voice_enabled", "voice_generator", VoiceGenerator),
        ("inline_image_enabled", "image_generator", ImageGenerator),
    ],
)
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.usefixtures("no_memory_review")
async def test_a_marker_switch_controls_its_generator(
    monkeypatch: pytest.MonkeyPatch, flag: str, kwarg: str, generator_type: type, enabled: bool
) -> None:
    """Each inline marker's switch gates whether the QA streamer receives its generator."""
    cog = _cog()
    cog.config = _config_stub(**{flag: enabled})
    built = _install_streamer(monkeypatch=monkeypatch)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await _answer(cog=cog, message=as_message(fake=message)).stream_answer(
        system_prompt="SYS", context=ReplyContext()
    )

    if enabled:
        assert isinstance(built[0][kwarg], generator_type)
    else:
        assert built[0][kwarg] is None


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
    ) -> AsyncIterator[ResponseStreamEvent]:
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


@pytest.mark.usefixtures("no_memory_review")
async def test_youtube_qa_uses_interactions_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A watched YouTube URL streams the answer through Interactions, not Responses.

    The graded effort is sent straight through as the Interactions thinking_level.
    """
    monkeypatch.setattr(
        RuntimeModelCatalog,
        "slow_model",
        property(lambda _self: ModelSettings(name="gemini-test", effort="high")),
    )
    cog = _cog()
    cog.config = _config_stub(youtube_video_enabled=True, gemini_api_key="test-key")
    interactions = _FakeInteractionsResource(events=interactions_turn_events())
    cog.toolkit.__dict__["gemini_client"] = FakeGeminiClient(interactions=interactions)

    url = "https://youtu.be/jNQXAC9IVRw"
    message = FakeMessage(content=f"<@999> 總結這影片 {url}", author=FakeAuthor(user_id=1))
    assert message.guild is not None
    message.guild.id = 424242
    await _answer(cog=cog, message=as_message(fake=message)).stream_answer(
        system_prompt="SYS", context=ReplyContext(), effort="low", yt_url=url
    )

    # The Responses answer stream was never used; the Interactions one was, with the video part.
    assert _recorded(cog).responses.create_streams == []
    assert len(interactions.calls) == 1
    assert interactions.calls[0].generation_config["thinking_level"] == "low"
    last_step_parts = interactions.calls[0].input[-1]["content"]
    assert {"type": "video", "uri": url, "processing": "agentic"} in last_step_parts
    # The video turn is told the request's time and place, exactly as a Responses one is.
    _assert_runtime_time_context(
        instructions=interactions.calls[0].system_instruction, system_prompt="SYS"
    )
    assert "424242" in interactions.calls[0].system_instruction
    # The shared streamer rendered the reply and a footer from the Interactions usage.
    reply_content = message.replies[0].content or ""
    assert "Hello world" in reply_content
    footer_match = USAGE_FOOTER_RE.search(reply_content)
    assert footer_match is not None
    assert "12" in footer_match.group()
    assert "34" in footer_match.group()
    # A persistent watch reaction marks that the reply was grounded in the video.
    assert YOUTUBE_EMOJI in message.added_reactions


def test_count_media_parts_counts_only_the_shapes_media_reaches_the_model_in() -> None:
    """`media_parts` is the one number saying an attachment survived into the request.

    A silent zero would be worse than no field, so the walk is pinned against the shapes the
    assembled input actually mixes: string shorthand, text parts, and both media parts.
    """
    answer_input = cast(
        "ResponseInputParam",
        [
            EasyInputMessageParam(role="user", content="string shorthand carries no parts"),
            EasyInputMessageParam(
                role="user",
                content=[
                    ResponseInputTextParam(type="input_text", text="look at this"),
                    ResponseInputImageParam(type="input_image", detail="auto", image_url="data:"),
                    ResponseInputFileParam(type="input_file", file_id="https://x/files/a"),
                ],
            ),
            EasyInputMessageParam(
                role="user",
                content=[ResponseInputFileParam(type="input_file", file_id="https://x/files/b")],
            ),
        ],
    )

    assert count_media_parts(answer_input=answer_input) == 3


@pytest.mark.parametrize("scenario", ["kill_switch_off", "non_gemini_model", "no_url", "no_key"])
@pytest.mark.usefixtures("no_memory_review")
async def test_youtube_qa_falls_back_to_responses(
    monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    """Without a watchable Gemini video turn, the answer stays on the Responses path."""
    cog = _cog()
    cog.config = _config_stub(
        youtube_video_enabled=scenario != "kill_switch_off",
        gemini_api_key="" if scenario == "no_key" else "test-key",
    )
    slow_name = "gpt-5-mini" if scenario == "non_gemini_model" else "gemini-test"
    monkeypatch.setattr(
        RuntimeModelCatalog,
        "slow_model",
        property(lambda _self: ModelSettings(name=slow_name, effort="high")),
    )
    interactions = _FakeInteractionsResource(events=interactions_turn_events())
    cog.toolkit.__dict__["gemini_client"] = FakeGeminiClient(interactions=interactions)
    logged = capture_logs(monkeypatch=monkeypatch, level="info")

    url = "https://youtu.be/jNQXAC9IVRw"
    yt_url = None if scenario == "no_url" else url
    message = FakeMessage(content=f"<@999> {url}", author=FakeAuthor(user_id=1))
    await _answer(cog=cog, message=as_message(fake=message)).stream_answer(
        system_prompt="SYS", context=ReplyContext(), yt_url=yt_url
    )

    assert interactions.calls == []
    assert _recorded(cog).responses.create_streams == [True]
    # The fallback is silent to the user, so the log is the only place the reason survives. A
    # `no_url` turn never asked for the swap here, so it names no reason.
    declines = [fields for text, fields in logged if "youtube watch declined" in text]
    expected_reason = {
        "non_gemini_model": "model",
        "kill_switch_off": "kill-switch",
        "no_key": "no-gemini-key",
    }.get(scenario)
    assert [fields.get("reason") for fields in declines] == (
        [expected_reason] if expected_reason else []
    )
    dispatch = next(fields for text, fields in logged if text == "gen_reply answer dispatch")
    assert dispatch["backend"] == "responses"


async def test_the_answer_turn_itself_is_retried_and_still_delivers_the_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the wiring, not the helper: the QA answer path must go through the retry.

    Both the helper's own tests and this one would stay green if `AnswerTurn.stream_answer` were
    quietly put back on a bare `streamer.consume(...)`, except for the second `create` this
    asserts on.
    """
    _no_retry_backoff(monkeypatch=monkeypatch)
    cog = _cog()
    message = FakeMessage(content="hi")
    _recorded(cog).responses.stream_queue = [
        _mid_stream_unavailable(),
        list(_default_turn_events()),
    ]

    await _run_pipeline(cog=cog, message=message)

    # Two streaming dispatches for one answer, and the reply still landed.
    assert _recorded(cog).responses.create_streams.count(True) == 2
    assert message.replies
    assert (message.replies[0].content or "").startswith("done")


async def test_memory_markers_route_by_the_message_not_by_the_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whose memory a note lands in is decided from the message, never from the marker body.

    This is what keeps the compartment boundary structural now that the answer model, rather
    than a separate extraction pass, proposes what to write: a note claiming to be about
    someone else still goes to the author's scope, and the community note goes to the guild
    the message was sent in.
    """
    cog = _cog()

    scheduled: list[dict[str, object]] = []

    def fake_schedule(**kwargs: object) -> None:
        """Records each scheduled memory update."""
        scheduled.append(kwargs)

    _install_streamer(
        monkeypatch=monkeypatch,
        memory_notes=("使用者偏好繁體中文",),
        forget_notes=("使用者不再玩那款遊戲",),
        server_memory_notes=("這個社群週五都在講炸雞",),
    )
    monkeypatch.setattr("discordbot.cogs.gen_reply.answer.schedule_memory_update", fake_schedule)

    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    await _run_pipeline(cog=cog, message=message)

    by_scope = {str(update["scope"]): update for update in scheduled}
    personal = by_scope[user_scope(user_id=1)]
    assert personal["remember_notes"] == ("使用者偏好繁體中文",)
    assert personal["forget_notes"] == ("使用者不再玩那款遊戲",)
    community = by_scope[server_scope(server_id=1)]
    assert community["remember_notes"] == ("這個社群週五都在講炸雞",)
    # The community update never carries a forget: `<forget-memory>` is a per-user marker.
    assert "forget_notes" not in community


@pytest.mark.parametrize(
    ("memory_notes", "server_memory_notes", "prompt"),
    [
        (("使用者偏好繁體中文",), (), PHASE1_EVALUATOR_PROMPT),
        ((), ("這個社群週五都在講炸雞",), SERVER_PHASE1_EVALUATOR_PROMPT),
    ],
    ids=["user", "server"],
)
async def test_a_memory_note_is_reviewed_under_its_scopes_own_prompt(
    monkeypatch: pytest.MonkeyPatch,
    memory_notes: tuple[str, ...],
    server_memory_notes: tuple[str, ...],
    prompt: str,
) -> None:
    """A note is reviewed under the evaluator prompt of the scope it lands in."""
    cog = _cog()
    _install_streamer(
        monkeypatch=monkeypatch, memory_notes=memory_notes, server_memory_notes=server_memory_notes
    )

    await _run_pipeline(
        cog=cog, message=FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))
    )
    await drain_memory_turns(scopes=(user_scope(user_id=1), server_scope(server_id=1)))
    await wait_for_persisted_writes()

    responses = _recorded(cog).responses
    reviews = [
        instructions
        for instructions, text_format in zip(
            responses.parse_instructions, responses.parse_text_formats, strict=True
        )
        if text_format is RawMemoryDraft
    ]
    assert reviews == [prompt]


@pytest.mark.parametrize(
    ("context", "guild_id", "expected_source"),
    [
        (InteractionContextType.guild, GUILD_ID, f"source: guild {GUILD_ID}"),
        (InteractionContextType.bot_dm, None, "source: dm"),
    ],
)
def test_an_ask_turn_stamps_its_memory_with_where_it_happens(
    monkeypatch: pytest.MonkeyPatch,
    context: InteractionContextType,
    guild_id: int | None,
    expected_source: str,
) -> None:
    """The read and the write must name the same compartment, or memory goes in and never out.

    `Message.guild` is None here whatever the interaction says, so a source stamp taken from
    the message would file a server turn's `source_only` observations under `dm/` while the next
    turn in the same server read only `global` and that guild's own.
    """
    scheduled: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.answer.schedule_memory_update",
        lambda **kwargs: scheduled.append(kwargs),
    )
    interaction = _interaction(guild_id=guild_id, context=context)
    surface = TurnSurface.for_interaction(interaction=interaction, question="在幹嘛")
    turn = AnswerTurn(
        config=LLMConfig.model_construct(),
        media_delivery=hosting_off_planner(),
        toolkit=_toolkit(interaction=interaction),
        surface=surface,
    )

    turn._schedule_memory_updates(
        context=ReplyContext(),
        full_reply="好",
        streamer=ResponseStreamer(
            message=surface.message, surface=surface, media_delivery=hosting_off_planner()
        ),
    )

    assert [update["subject"] for update in scheduled] == [
        f"target_user_id: {ASKER_ID}\n{expected_source}"
    ]
