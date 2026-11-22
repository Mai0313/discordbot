"""Tests for the reply cog's listeners and its `/ask` command."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
import asyncio
import contextlib

import pytest
from nextcord import Attachment

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.gen_reply import cog as gen_reply_cog
from discordbot.typings.models import ModelSettings, RouteClassification, RuntimeModelCatalog
from discordbot.cogs.gen_reply.cog import ReplyGeneratorCogs
from discordbot.cogs.gen_reply.context import ReplyContext, ReplyContextBuilder
from discordbot.cogs.gen_reply.routing import RouteClassifier
from discordbot.cogs.gen_reply.pipeline import ReplyPipeline
from discordbot.cogs.gen_reply.turn_state import dispatched_model, current_answer_streamer
from discordbot.cogs.gen_reply.status_marks import FAILED_EMOJI, EMPTY_PROMPT_EMOJI

from tests.helpers.casting import as_bot, as_message, make_invalid_form_body
from tests.helpers.reply_fakes import (
    FakeAuthor,
    FakeMessage,
    FakeSnapshot,
    _cog,
    _route,
    _recorded,
    _build_stub,
    _media_routes,
    _classify_stub,
    _recorded_video,
    _ready_context_task,
)
from tests.helpers.ask_interaction import (
    GUILD_ID,
    _CAT_PNG,
    BOT_USER_ID,
    ASK_SNOWFLAKE,
    _toolkit,
    _interaction,
)
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from collections.abc import Callable, Awaitable

    from nextcord.types.message import Attachment as AttachmentPayload

    from discordbot.cogs.gen_reply.surface import TurnSurface


async def test_gen_reply_routes_and_handlers_without_api() -> None:
    """Verifies route, video, and image handlers using fake APIs."""
    cog = _cog()
    message = FakeMessage(content="make a summary", author=FakeAuthor(user_id=1))
    _recorded(cog).responses.output_parsed = RouteClassification(decision="IMAGE")
    assert (await _route(cog=cog, message=message)).decision == "IMAGE"
    assert _recorded(cog).responses.parse_models[0] == cog.toolkit.runtime_models.triage_model.name

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_video(
        user_prompt="video", context_task=_ready_context_task()
    )
    assert len(message.replies) == 1
    # Text-to-video: the fake director returns no draft, so `refine` falls back to the raw
    # request, which reaches omni as the interaction input text.
    create_input = _recorded_video(cog).create_inputs[0]
    assert [part["text"] for part in create_input if part["type"] == "text"] == ["video"]

    await _media_routes(cog=cog, message=as_message(fake=message)).handle_image(
        user_prompt="image", context_task=_ready_context_task()
    )
    assert _recorded(cog).images.generate_calls
    # The director returns no draft here either, so images.generate gets the raw request.
    assert _recorded(cog).images.generate_prompts == ["image"]
    # The image is delivered first, then a conversational reply streams onto that same
    # message via the flash fast_model with no tools.
    assert message.replies[-1].file is not None
    assert _recorded(cog).responses.create_models[-1] == cog.toolkit.runtime_models.fast_model.name
    assert _recorded(cog).responses.create_streams[-1] is True
    assert _recorded(cog).responses.create_tools[-1] is None


async def test_gen_reply_on_message_early_returns_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies bot messages, unmentioned guild messages, empty prompts, and errors."""
    cog = _cog()
    bot_authored = FakeMessage(content="<@999> hi", author=FakeAuthor(bot=True))
    await cog.on_message(message=as_message(fake=bot_authored))
    assert bot_authored.replies == []

    unmentioned = FakeMessage(content="hello", author=FakeAuthor(user_id=1))
    await cog.on_message(message=as_message(fake=unmentioned))
    assert unmentioned.replies == []

    dm_empty = FakeMessage(content="<@999>", author=FakeAuthor(user_id=1))
    dm_empty.guild = None
    await cog.on_message(message=as_message(fake=dm_empty))
    assert dm_empty.replies[0].embed is None
    assert _recorded(cog).responses.parse_models == []

    monkeypatch.setattr(RouteClassifier, "classify", _classify_stub(route=RuntimeError("boom")))
    monkeypatch.setattr(ReplyContextBuilder, "build", _build_stub(context=ReplyContext()))
    failed = FakeMessage(content="<@999> fail", author=FakeAuthor(user_id=1))
    await cog.on_message(message=as_message(fake=failed))
    notice = failed.replies[0].embed
    assert notice is not None
    assert notice.footer.text == "RuntimeError"

    # Source deleted before the error embed lands: it falls back to an unparented send.
    deleted = FakeMessage(content="<@999> fail", author=FakeAuthor(user_id=1))
    deleted.reply_error = make_invalid_form_body()
    await cog.on_message(message=as_message(fake=deleted))
    assert deleted.replies == []
    assert deleted.channel.sent[0].embed is not None


async def test_a_turn_without_a_proxy_key_still_reports_its_failure() -> None:
    """Reading the request builds the toolkit, whose client the SDK refuses without a key."""
    cog = ReplyGeneratorCogs(
        bot=as_bot(
            fake=SimpleNamespace(
                user=SimpleNamespace(id=999, name="bot"), get_cog=lambda name: None
            )
        )
    )
    cog.config = LLMConfig.model_construct()
    message = FakeMessage(content="<@999> hi", author=FakeAuthor(user_id=1))

    await cog.on_message(message=as_message(fake=message))

    notice = message.replies[0].embed
    assert notice is not None
    assert notice.footer.text == "OpenAIError"
    assert message.added_reactions == [FAILED_EMOJI]


async def test_an_empty_mention_is_marked_as_well_as_answered() -> None:
    """A mention with nothing else in it gets marked on the message beside its plain reply."""
    cog = _cog()
    message = FakeMessage(content="<@999>", author=FakeAuthor(user_id=1))

    await cog.on_message(message=as_message(fake=message))

    assert message.added_reactions == [EMPTY_PROMPT_EMOJI]
    assert len(message.replies) == 1
    assert message.replies[0].embed is None


async def test_an_empty_mention_deleted_before_its_answer_gets_nothing() -> None:
    """An empty mention answers nothing once its message is gone, so no error notice lands."""
    cog = _cog()
    message = FakeMessage(content="<@999>", author=FakeAuthor(user_id=1))
    message.reply_error = make_invalid_form_body()

    await cog.on_message(message=as_message(fake=message))

    assert message.added_reactions == [EMPTY_PROMPT_EMOJI]
    assert message.channel.sent == []


@pytest.mark.parametrize(
    ("content", "in_guild", "expected_prompt"),
    [("", False, "draw a cat"), ("<@999> please", True, "please\ndraw a cat")],
    ids=["pure-dm-forward", "commented-guild-forward"],
)
async def test_on_message_folds_a_forward_into_the_prompt(
    monkeypatch: pytest.MonkeyPatch, content: str, in_guild: bool, expected_prompt: str
) -> None:
    """A forward's request reaches the pipeline as the prompt, after any comment of its own.

    A pure forward (empty content, payload only in snapshots) is not gated as an empty `?`, so an
    IMAGE/VIDEO route is not blank, and a commented one keeps the forwarded request rather than
    dropping it because the comment is non-empty. A guild forward triggers only via the mention.
    """
    cog = _cog()
    calls: list[tuple[FakeMessage, str]] = []

    async def record_pipeline(self: ReplyPipeline) -> None:
        """Records the prompt the pipeline receives."""
        calls.append((cast("FakeMessage", self.message), self.user_prompt))

    monkeypatch.setattr(ReplyPipeline, "run", record_pipeline)

    message = FakeMessage(content=content, author=FakeAuthor(user_id=1))
    if not in_guild:
        message.guild = None
    message.snapshots = [FakeSnapshot(content="draw a cat")]
    await cog.on_message(message=as_message(fake=message))

    assert message.replies == []
    assert calls == [(message, expected_prompt)]


@pytest.mark.usefixtures("no_memory_review")
async def test_a_failed_turn_records_the_model_it_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`gen_reply failed` names the model the turn was on when it broke.

    The failure surfaces in `on_message`, frames above every place that picks a model, and a
    provider error rarely names the model it refused ("This model is currently experiencing
    high demand" names none). With many turns interleaved in one log file, re-deriving it from
    a neighbouring record means trusting that nothing else wrote between them.
    """
    cog = _cog()
    monkeypatch.setattr(
        RuntimeModelCatalog,
        "slow_model",
        property(lambda _self: ModelSettings(name="gemini-answer-tier", effort="high")),
    )

    async def failing_create(**kwargs: object) -> object:
        """Fails the answer turn the way a provider outage does."""
        del kwargs
        raise RuntimeError("This model is currently experiencing high demand")

    monkeypatch.setattr(_recorded(cog).responses, "create", failing_create)
    failures = capture_logs(monkeypatch=monkeypatch, level="error")

    # The route fake answers QA, so the answer is the turn's only `responses.create`.
    message = FakeMessage(content="<@999> 幫我總結", author=FakeAuthor(user_id=1))
    await cog.on_message(message=as_message(fake=message))

    # The answer tier, not the triage tier the route ran on a moment earlier.
    assert [fields.get("model") for text, fields in failures if text == "gen_reply failed"] == [
        "gemini-answer-tier"
    ]
    assert dispatched_model.get() is None
    assert current_answer_streamer.get() is None


@pytest.mark.usefixtures("no_memory_review")
async def test_run_turn_clears_contextvars_after_success() -> None:
    """Normal turns clear `dispatched_model` and `current_answer_streamer` before returning."""
    cog = _cog()
    message = FakeMessage(content="<@999> 你好", author=FakeAuthor(user_id=1))
    await cog.on_message(message=as_message(fake=message))

    assert dispatched_model.get() is None
    assert current_answer_streamer.get() is None


def test_the_cog_builds_no_toolkit_until_a_reply_needs_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured deployment never constructs a Gemini client it can never use."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    cog = ReplyGeneratorCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    assert "toolkit" not in cog.__dict__


async def test_on_ready_resume_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """on_ready guards the resume and price-table loop so reconnects start each only once."""
    cog = _cog(bot_user_id=999)
    calls = 0

    async def fake_resume_after_restart(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(gen_reply_cog, "resume_after_restart", fake_resume_after_restart)
    starts = 0
    real_start = cog.price_table_task.start

    def recording_start() -> asyncio.Task[None]:
        nonlocal starts
        starts += 1
        return real_start()

    monkeypatch.setattr(cog.price_table_task, "start", recording_start)
    try:
        await cog.on_ready()
        await cog.on_ready()
        while cog._tasks:
            await asyncio.gather(*list(cog._tasks))
        assert calls == 1
        assert starts == 1
        assert cog.price_table_task.is_running()
    finally:
        task = cog.price_table_task.get_task()
        cog.cog_unload()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
    assert not cog.price_table_task.is_running()


async def test_price_table_task_logs_error_type_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broad catch in `price_table_task` attaches both `error_type` and `_exc_info`."""
    errors = capture_logs(monkeypatch=monkeypatch, level="error")

    def boom() -> None:
        raise ValueError("price refresh blew up")

    monkeypatch.setattr(target=gen_reply_cog, name="refresh_model_info", value=boom)

    await ReplyGeneratorCogs.price_table_task.coro(_cog(bot_user_id=999))

    assert len(errors) == 1
    _, fields = errors[0]
    assert fields["error_type"] == "ValueError"
    assert isinstance(fields["_exc_info"], ValueError)


def _ask_cog(
    interaction: Any,  # noqa: ANN401 -- see `_interaction`
    monkeypatch: pytest.MonkeyPatch,
    run: Callable[[ReplyPipeline], Awaitable[None]],
) -> ReplyGeneratorCogs:
    """A cog whose `/ask` gets as far as handing the pipeline the turn, and no further."""
    cog = ReplyGeneratorCogs(
        bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=BOT_USER_ID, name="pocat")))
    )
    cog.__dict__["toolkit"] = _toolkit(interaction=interaction)
    monkeypatch.setattr(ReplyPipeline, "run", run)
    return cog


async def test_ask_defers_before_anything_slower_than_three_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The token dies after three seconds, and every phase of a turn is slower than that."""
    ran: list[tuple[TurnSurface, str, bool]] = []

    async def _run(pipeline: ReplyPipeline) -> None:
        """Records the turn the command would have run and whether the defer had already landed."""
        ran.append((pipeline.surface, pipeline.user_prompt, interaction.deferred))

    interaction = _interaction()
    cog = _ask_cog(interaction=interaction, monkeypatch=monkeypatch, run=_run)

    await cog.ask(interaction, question="在幹嘛", attachment=None)

    surface, user_prompt, deferred_first = ran[0]
    assert deferred_first is True
    assert user_prompt == "在幹嘛"
    assert surface.interaction is interaction
    assert surface.guild_id == GUILD_ID
    assert surface.message.id == ASK_SNOWFLAKE


async def test_an_ask_turn_never_reacts_to_its_synthesized_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nobody posted that message, so every status reaction would be a REST call that 404s."""
    enabled: list[bool] = []

    async def _run(pipeline: ReplyPipeline) -> None:
        """Records whether the turn's status chain would react."""
        enabled.append(pipeline.reactions.enabled)

    interaction = _interaction()
    cog = _ask_cog(interaction=interaction, monkeypatch=monkeypatch, run=_run)

    await cog.ask(interaction, question="在幹嘛", attachment=None)

    assert enabled == [False]


async def test_ask_answers_a_blank_question_without_running_a_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whitespace-only option would otherwise route on nothing at all."""

    async def _never(pipeline: ReplyPipeline) -> None:
        """Fails if a blank question ever reaches the pipeline."""
        del pipeline
        raise AssertionError("a blank question must not run a turn")

    interaction = _interaction()
    cog = _ask_cog(interaction=interaction, monkeypatch=monkeypatch, run=_never)

    await cog.ask(interaction, question="   ", attachment=None)

    (edit,) = interaction.edits
    assert "embed" not in edit


async def test_ask_answers_an_attachment_sent_without_a_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file on its own is something to answer, not an empty prompt.

    The attachment reaches the turn only through the synthesized message, so that message is
    what the empty-prompt check has to read.
    """
    ran: list[tuple[TurnSurface, str]] = []

    async def _run(pipeline: ReplyPipeline) -> None:
        """Records the turn the command would have run."""
        ran.append((pipeline.surface, pipeline.user_prompt))

    interaction = _interaction()
    cog = _ask_cog(interaction=interaction, monkeypatch=monkeypatch, run=_run)
    interaction.data = {"resolved": {"attachments": {"9": _CAT_PNG}}}

    await cog.ask(
        interaction,
        question="   ",
        attachment=Attachment(data=cast("AttachmentPayload", _CAT_PNG), state=interaction._state),
    )

    assert interaction.edits == []
    ((surface, user_prompt),) = ran
    assert user_prompt == ""
    assert [attachment.filename for attachment in surface.message.attachments] == ["cat.png"]


async def test_ask_without_a_proxy_key_answers_its_deferred_response() -> None:
    """Otherwise the asker is left on Discord's "thinking" state with nothing ever landing."""
    cog = ReplyGeneratorCogs(
        bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=BOT_USER_ID, name="pocat")))
    )
    cog.config = LLMConfig.model_construct()
    interaction = _interaction()

    await cog.ask(interaction, question="在幹嘛", attachment=None)

    (edit,) = interaction.edits
    assert edit["embed"].footer.text == "OpenAIError"
