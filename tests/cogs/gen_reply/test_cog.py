"""Tests for the reply cog's listeners and its `/ask` command."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
import asyncio

import pytest
from nextcord import Attachment

from discordbot.typings.llm import LLMConfig
from discordbot.typings.memory import MemoryOwner
from discordbot.typings.models import ModelSettings, RuntimeModelCatalog
from discordbot.services.memory import database as memory_db
from discordbot.services.memory import consolidation
from discordbot.cogs.gen_reply.cog import ReplyGeneratorCogs
from discordbot.services.memory.store import (
    flavor_of,
    user_scope,
    server_scope,
    scope_owner_id,
    append_raw_entry,
)
from discordbot.cogs.gen_reply.context import ReplyContext, ReplyContextBuilder
from discordbot.cogs.gen_reply.routing import RouteClassifier
from discordbot.services.memory.writer import render_turn_payload
from discordbot.cogs.gen_reply.pipeline import ReplyPipeline
from discordbot.services.memory.prompts import PHASE2_PROMPT, PHASE1_EVALUATOR_PROMPT
from discordbot.cogs.gen_reply.status_marks import FAILED_EMOJI
from discordbot.services.memory.server_prompts import (
    SERVER_PHASE2_PROMPT,
    SERVER_PHASE1_EVALUATOR_PROMPT,
)

from tests.helpers.memory import drain_memory_turns, wait_for_persisted_writes
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
    _run_pipeline,
    _classify_stub,
    _recorded_video,
    _install_streamer,
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
    from pathlib import Path
    from datetime import datetime
    from collections.abc import Callable, Awaitable

    from nextcord.types.message import Attachment as AttachmentPayload

    from discordbot.cogs.gen_reply.surface import TurnSurface


async def test_gen_reply_routes_and_handlers_without_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies route, video, image, and slow-reply handlers using fake APIs."""
    cog = _cog()
    message = FakeMessage(content="make a summary", author=FakeAuthor(user_id=1))
    assert (await _route(cog=cog, message=message)).decision == "QA"
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

    streams_before = _recorded(cog).responses.create_streams.count(True)
    built = _install_streamer(monkeypatch=monkeypatch)
    await _run_pipeline(cog=cog, message=message)
    assert _recorded(cog).responses.create_streams.count(True) == streams_before + 1
    assert built[-1]["message"] is message


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
    assert dm_empty.replies[0].content == "?"

    monkeypatch.setattr(RouteClassifier, "classify", _classify_stub(route=RuntimeError("boom")))
    monkeypatch.setattr(ReplyContextBuilder, "build", _build_stub(context=ReplyContext()))
    failed = FakeMessage(content="<@999> fail", author=FakeAuthor(user_id=1))
    await cog.on_message(message=as_message(fake=failed))
    notice = failed.replies[0].embed
    assert notice is not None
    assert notice.title == "Something went wrong"

    # Source deleted before the error embed lands: it falls back to an unparented send.
    deleted = FakeMessage(content="<@999> fail", author=FakeAuthor(user_id=1))
    deleted.reply_error = make_invalid_form_body()
    await cog.on_message(message=as_message(fake=deleted))
    assert deleted.replies == []
    assert deleted.channel.sent[0].embed is not None


async def test_a_turn_without_a_proxy_key_still_reports_its_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the request builds the toolkit, whose client the SDK refuses without a key."""
    # The SDK also accepts `OPENAI_ADMIN_KEY` from the environment, which would build the client.
    monkeypatch.delenv(name="OPENAI_ADMIN_KEY", raising=False)
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
    assert notice.title == "Something went wrong"
    assert notice.footer.text == "OpenAIError"
    assert message.added_reactions == [FAILED_EMOJI]


async def test_an_empty_mention_is_marked_as_well_as_answered() -> None:
    """A mention with nothing else in it gets ❓ on the message beside its `?` reply."""
    cog = _cog()
    message = FakeMessage(content="<@999>", author=FakeAuthor(user_id=1))

    await cog.on_message(message=as_message(fake=message))

    assert message.added_reactions == ["❓"]
    assert [reply.content for reply in message.replies] == ["?"]


async def test_an_empty_mention_deleted_before_its_answer_gets_nothing() -> None:
    """A `?` answers nothing once its message is gone, so no error notice lands instead."""
    cog = _cog()
    message = FakeMessage(content="<@999>", author=FakeAuthor(user_id=1))
    message.reply_error = make_invalid_form_body()

    await cog.on_message(message=as_message(fake=message))

    assert message.added_reactions == ["❓"]
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


def test_the_cog_builds_no_toolkit_until_a_reply_needs_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured deployment never constructs a Gemini client it can never use."""
    monkeypatch.setenv(name="OPENAI_BASE_URL", value="https://example.test/v1")
    monkeypatch.setenv(name="OPENAI_API_KEY", value="test-key")
    cog = ReplyGeneratorCogs(bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=999))))
    assert "toolkit" not in cog.__dict__


async def test_resume_memory_reenqueues_jobs_and_sweeps_other_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_ready resume re-enqueues persisted jobs and sweeps every over-threshold scope."""
    cog = _cog(bot_user_id=999)
    writer_sentinel = object()
    cog.toolkit.__dict__["memory_writer"] = writer_sentinel

    user_job_scope = user_scope(user_id=1)
    server_job_scope = server_scope(server_id=2)
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

    async def fake_list(updated_before: datetime) -> list[memory_db.MemoryJob]:
        del updated_before
        return jobs

    def fake_resume(**kwargs: object) -> None:
        resumed.append(kwargs)

    async def fake_consolidate(scope: str, writer: object, identity: str) -> None:
        swept.append(scope)

    monkeypatch.setattr("discordbot.cogs.gen_reply.cog.safe_list_resumable", fake_list)
    monkeypatch.setattr("discordbot.cogs.gen_reply.cog.resume_memory_update", fake_resume)
    monkeypatch.setattr("discordbot.cogs.gen_reply.cog.consolidate_if_needed", fake_consolidate)
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.cog.iter_scopes",
        lambda: [user_job_scope, server_job_scope, sweep_scope],
    )
    monkeypatch.setattr("discordbot.cogs.gen_reply.cog.needs_consolidation", lambda scope: True)
    monkeypatch.setattr(
        "discordbot.cogs.gen_reply.cog.read_owner",
        lambda scope: MemoryOwner(owner_id=scope_owner_id(scope=scope), owner_name=""),
    )

    await cog._resume_memory()
    # Wait for spawned sweep tasks to finish.
    while cog._tasks:
        await asyncio.gather(*list(cog._tasks))

    assert {kwargs["scope"] for kwargs in resumed} == {user_job_scope, server_job_scope}
    by_scope = {kwargs["scope"]: kwargs for kwargs in resumed}
    assert by_scope[user_job_scope]["writer"] is writer_sentinel
    assert by_scope[user_job_scope]["token"] == 11
    assert by_scope[server_job_scope]["writer"] is writer_sentinel
    # The row's status decides whether the resumed review is its one retry.
    assert by_scope[user_job_scope]["status"] == "failed"
    assert by_scope[server_job_scope]["status"] == "pending"
    # Every over-threshold scope is swept, including the resumed ones: the scope
    # lock makes the resumed review and the consolidation sweep idempotent.
    assert set(swept) == {user_job_scope, server_job_scope, sweep_scope}


async def test_resume_memory_sweeps_every_scope_past_a_backup_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy like `111.bak` under the store must not abort the sweep for the scopes after it."""
    cog = _cog(bot_user_id=999)
    scopes = [user_scope(user_id=111), user_scope(user_id=222), server_scope(server_id=333)]
    for key in [*scopes, "111.bak"]:
        for entry in ("- a", "- b"):
            append_raw_entry(scope=key, entry_text=entry)
    swept: list[str] = []

    async def fake_consolidate(scope: str, writer: object, identity: str) -> None:
        swept.append(scope)

    monkeypatch.setattr("discordbot.cogs.gen_reply.cog.consolidate_if_needed", fake_consolidate)

    await cog._resume_memory()
    while cog._tasks:
        await asyncio.gather(*list(cog._tasks))

    assert sorted(swept) == sorted(scopes)


async def test_resume_memory_sweeps_every_scope_past_an_undecodable_raw_file(
    memory_isolated_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `raw.md` saved in the wrong encoding leaves out its own scope, not the ones after it."""
    cog = _cog(bot_user_id=999)
    scopes = [user_scope(user_id=111), user_scope(user_id=222), server_scope(server_id=333)]
    for scope in scopes:
        for entry in ("- a", "- b"):
            append_raw_entry(scope=scope, entry_text=entry)
    (memory_isolated_dir / "222" / "raw.md").write_bytes("- a\n\n- 喜歡簡短回覆".encode("big5"))
    swept: list[str] = []

    async def fake_consolidate(scope: str, writer: object, identity: str) -> None:
        swept.append(scope)

    monkeypatch.setattr("discordbot.cogs.gen_reply.cog.consolidate_if_needed", fake_consolidate)

    await cog._resume_memory()
    while cog._tasks:
        await asyncio.gather(*list(cog._tasks))

    assert sorted(swept) == [user_scope(user_id=111), server_scope(server_id=333)]


@pytest.mark.parametrize(
    ("scope", "subject", "prompts"),
    [
        (user_scope(user_id=1), "target_user_id: 1", {PHASE1_EVALUATOR_PROMPT, PHASE2_PROMPT}),
        (
            server_scope(server_id=2),
            "target_server_id: 2",
            {SERVER_PHASE1_EVALUATOR_PROMPT, SERVER_PHASE2_PROMPT},
        ),
    ],
    ids=["user", "server"],
)
async def test_resume_memory_reaches_the_model_under_each_scopes_own_prompts(
    monkeypatch: pytest.MonkeyPatch, scope: str, subject: str, prompts: set[str]
) -> None:
    """The resumed review and the consolidation sweep both run under the scope's flavor.

    The recorder answers the review with no valid draft, and a failed review with no forget
    never consolidates, so the one consolidation call is the sweep's.
    """
    monkeypatch.setattr(consolidation, "RAW_CONSOLIDATION_THRESHOLD", 1)
    append_raw_entry(scope=scope, entry_text="### stable_preference\n- summary_zh: 喜歡簡短回覆")
    await memory_db.upsert_pending(
        scope=scope,
        flavor=flavor_of(scope=scope),
        subject=subject,
        transcript=render_turn_payload(
            transcript="Alice (alice) [id: 1]: 哈囉", rounds=((("喜歡簡短回覆",), ()),)
        ),
        identity="",
        token=memory_db.new_token(),
    )
    # Built after the row, so the row is an earlier process's.
    cog = _cog(bot_user_id=999)

    await cog._resume_memory()
    while cog._tasks:
        await asyncio.gather(*list(cog._tasks))
    await drain_memory_turns(scopes=(scope,))
    await wait_for_persisted_writes()

    assert set(_recorded(cog).responses.parse_instructions) == prompts


async def test_resume_memory_leaves_a_turn_this_process_staged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reply runs before `on_ready` does, and the review it staged is still running.

    The live scope's row was first written by an earlier process, so only `updated_at`, which
    the upsert moves while keeping `created_at`, says its transcript is this process's own. The
    scope an earlier process left is the control that the sweep ran at all.
    """
    live_scope = user_scope(user_id=1)
    left_scope = user_scope(user_id=2)
    for scope in (live_scope, left_scope):
        await memory_db.upsert_pending(
            scope=scope,
            flavor="user",
            subject="target_user_id: 1",
            transcript="earlier",
            identity="",
            token=memory_db.new_token(),
        )
    cog = _cog(bot_user_id=999)
    await memory_db.upsert_pending(
        scope=live_scope,
        flavor="user",
        subject="target_user_id: 1",
        transcript="live",
        identity="",
        token=memory_db.new_token(),
    )
    resumed: list[object] = []

    def fake_resume(**kwargs: object) -> None:
        resumed.append(kwargs["scope"])

    monkeypatch.setattr("discordbot.cogs.gen_reply.cog.resume_memory_update", fake_resume)

    await cog._resume_memory()
    while cog._tasks:
        await asyncio.gather(*list(cog._tasks))

    assert resumed == [left_scope]


async def test_on_ready_resume_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """on_ready guards the resume so a gateway reconnect does not re-sweep."""
    cog = _cog(bot_user_id=999)
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
    ran: list[tuple[TurnSurface, str]] = []

    async def _run(pipeline: ReplyPipeline) -> None:
        """Records the turn the command would have run."""
        ran.append((pipeline.surface, pipeline.user_prompt))

    interaction = _interaction()
    cog = _ask_cog(interaction=interaction, monkeypatch=monkeypatch, run=_run)

    await cog.ask(interaction, question="在幹嘛", attachment=None)

    assert interaction.deferred is True
    surface, user_prompt = ran[0]
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

    assert [edit["content"] for edit in interaction.edits] == ["?"]


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


async def test_ask_without_a_proxy_key_answers_its_deferred_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise the asker is left on Discord's "thinking" state with nothing ever landing."""
    # The SDK also accepts `OPENAI_ADMIN_KEY` from the environment, which would build the client.
    monkeypatch.delenv(name="OPENAI_ADMIN_KEY", raising=False)
    cog = ReplyGeneratorCogs(
        bot=as_bot(fake=SimpleNamespace(user=SimpleNamespace(id=BOT_USER_ID, name="pocat")))
    )
    cog.config = LLMConfig.model_construct()
    interaction = _interaction()

    await cog.ask(interaction, question="在幹嘛", attachment=None)

    (edit,) = interaction.edits
    assert edit["embed"].title == "Something went wrong"
    assert edit["embed"].footer.text == "OpenAIError"
