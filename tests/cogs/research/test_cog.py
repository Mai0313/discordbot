"""Tests for the deep-research cog: launches, refusals, the resume sweep and every run exit."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn, cast
import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest
from nextcord import Embed, Thread, Permissions, AllowedMentions, PartialMessageable
from sqlalchemy.exc import OperationalError

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.research import cog as research_cog
from discordbot.cogs.research import database as rdb
from discordbot.utils.timezone import database_now
from discordbot.utils.model_pricing import ModelPriceEntry

from tests.helpers.casting import (
    as_bot,
    as_client,
    as_message,
    as_interaction,
    make_forbidden,
    make_not_found,
    make_server_error,
    make_invalid_form_body,
)
from tests.helpers.research import (
    _OWNER_ID,
    _THREAD_ID,
    _RunThread,
    _FakeStream,
    _fake_client,
    _created_event,
    _completed_event,
    _terminal_interaction,
)
from tests.helpers.discord_mocks import (
    FakeUser,
    FakeInteraction,
    FakeDiscordMessage,
    text_channel_granting,
)
from tests.helpers.logfire_capture import capture_logs

if TYPE_CHECKING:
    from discordbot.utils.media_delivery import MediaDeliveryPlanner

# ----- research module helpers --------------------------------------------------------------


def test_fallback_thread_name_uses_first_line() -> None:
    name = research_cog._fallback_thread_name(brief="研究 TPU 的歷史與競爭格局\n更多細節")
    assert name.startswith("研究 TPU")
    assert "\n" not in name
    assert research_cog._fallback_thread_name(brief="   ") == "深度研究"


def test_terminal_phase_mapping() -> None:
    assert research_cog._terminal_phase(status="completed") == "done"
    assert research_cog._terminal_phase(status="cancelled") == "cancelled"
    assert research_cog._terminal_phase(status="budget_exceeded") == "failed"


def test_failure_text_distinguishes_budget() -> None:
    assert "成本上限" in research_cog._failure_text(status="budget_exceeded")
    assert "取消" in research_cog._failure_text(status="cancelled")
    assert research_cog._failure_text(status="failed")


# ----- restart resume sweep -----------------------------------------------------------------


def _research_cog(enabled: bool) -> research_cog.ResearchCogs:
    """A cog on a bot with no gateway, whose config no deployment's `.env` reaches.

    The key is always present so the switch alone decides `deep_research_available`.
    """
    cog = research_cog.ResearchCogs(bot=as_bot(fake=SimpleNamespace()))
    cog.config = LLMConfig.model_construct(
        deep_research_enabled=enabled, gemini_api_key="AIza-key"
    )
    return cog


async def _seed_researching(
    thread_id: int,
    owner_id: int,
    stored_id: bool = True,
    agent: str = "antigravity-preview-09-2026",
) -> None:
    """Seeds one in-flight row, as a launch that never reached a terminal phase left it.

    `stored_id=False` is a launch that restarted before its interaction id was persisted.
    """
    await rdb.insert_session(
        thread_id=thread_id,
        owner_id=owner_id,
        channel_id=1,
        guild_id=1,
        source_message_id=1,
        agent=agent,
        brief="b",
    )
    if stored_id:
        await rdb.set_interaction(thread_id=thread_id, interaction_id=f"int_{thread_id}")


async def test_resume_sweep_reattaches_to_nothing_while_the_switch_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(**_kwargs: object) -> None:
        raise AssertionError("the resume must not reach the provider while the switch is off")

    monkeypatch.setattr(research_cog, "resume_research_stream", _boom)
    await _seed_researching(thread_id=30, owner_id=300)
    cog = _research_cog(enabled=False)

    await cog._resume_all()

    # Nothing is attached and nothing is delivered, and the row stays `researching` because that
    # is what it is: the interaction runs server-side and a later start with the switch on may
    # still deliver it.
    assert not cog._tasks
    assert cog._active_threads == set()
    assert [
        session.thread_id for session in await rdb.list_resumable(created_before=database_now())
    ] == [30]


async def test_resume_sweep_stays_off_without_a_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(**_kwargs: object) -> None:
        raise AssertionError("a keyless deployment must not reach the provider either")

    monkeypatch.setattr(research_cog, "resume_research_stream", _boom)
    # The gate is `deep_research_available`, so a switched-on deployment with no key is refused
    # here rather than at `genai.Client` inside the resume's own try.
    await _seed_researching(thread_id=35, owner_id=350)
    cog = _research_cog(enabled=True)
    cog.config.gemini_api_key = "   "

    await cog._resume_all()

    assert not cog._tasks
    assert [
        session.thread_id for session in await rdb.list_resumable(created_before=database_now())
    ] == [35]


async def test_resume_sweep_still_resumes_when_the_switch_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_researching(thread_id=40, owner_id=400)
    cog = _research_cog(enabled=True)
    resumed: list[int] = []

    async def _fake_resume_one(session: rdb.PersistentResearchSession) -> None:
        resumed.append(session.thread_id)

    monkeypatch.setattr(cog, "_resume_one", _fake_resume_one)

    await cog._resume_all()
    await asyncio.gather(*cog._tasks)

    assert resumed == [40]
    assert cog._active_threads == {40}
    # The sweep leaves the phase alone: the resumed run decides its own terminal phase.
    assert [
        session.thread_id for session in await rdb.list_resumable(created_before=database_now())
    ] == [40]


async def test_resume_sweep_leaves_the_runs_this_process_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch runs before `on_ready` does, and the research it started is still running.

    Its row is `researching` exactly like one an earlier process left, so only the cog's own
    start tells them apart; resuming it would run the report twice, or abandon a live run.
    """
    await _seed_researching(thread_id=50, owner_id=500)
    cog = _research_cog(enabled=True)
    await _seed_researching(thread_id=51, owner_id=510, stored_id=False)
    resumed: list[int] = []

    async def _fake_resume_one(session: rdb.PersistentResearchSession) -> None:
        resumed.append(session.thread_id)

    monkeypatch.setattr(cog, "_resume_one", _fake_resume_one)

    await cog._resume_all()
    await asyncio.gather(*cog._tasks)

    assert resumed == [50]
    assert cog._active_threads == {50}
    assert await rdb.active_thread_for_owner(owner_id=510) == 51


# ----- permission refusals ------------------------------------------------------------------


class _ResearchInteraction(FakeInteraction):
    """A `/deep_research` invocation from a guild text channel whose posts the test decides."""

    def __init__(self, channel: MagicMock) -> None:
        """Initializes the shared fake plus the channel the command was run in."""
        super().__init__()
        self.channel = channel


class _Anchor(FakeDiscordMessage):
    """A message the research thread hangs off, refusing it with `error` or opening `thread`."""

    def __init__(
        self, channel: MagicMock, error: Exception | None = None, thread: object = None
    ) -> None:
        """Initializes the identity `_start_for` reads on top of the shared message fake."""
        super().__init__()
        self.id = 10
        self.guild = SimpleNamespace(id=1)
        self.channel = channel
        self.author = FakeUser(user_id=300)
        self.error = error
        self.thread = thread

    async def create_thread(self, **_kwargs: object) -> object:
        """Opens `thread`, or fails the way Discord fails the thread."""
        if self.error is not None:
            raise self.error
        return self.thread


def _launching_cog(monkeypatch: pytest.MonkeyPatch) -> research_cog.ResearchCogs:
    """A cog that gets as far as `create_thread` without a title model behind it."""
    cog = _research_cog(enabled=True)

    async def _title(brief: str) -> str:
        del brief
        return "research"

    monkeypatch.setattr(target=cog, name="_generate_thread_name", value=_title)
    return cog


async def test_deep_research_answers_when_the_bot_cannot_post_in_the_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slash command reaches a channel the bot may not post in, so the answer rides the token."""
    channel = text_channel_granting()
    channel.send = AsyncMock(side_effect=make_forbidden(message="Missing Access"))
    interaction = _ResearchInteraction(channel=channel)
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")

    await _launching_cog(monkeypatch=monkeypatch).deep_research(
        as_interaction(fake=interaction), topic="topic"
    )

    assert interaction.response.deferred is True
    assert [edit.get("content") for edit in interaction.edits] == [
        "我在這個頻道的權限不夠,開不了研究串"
    ]
    assert warns == [
        ("deep research cannot post its anchor in this channel", {"channel_id": 20, "owner_id": 1})
    ], "a permission the bot cannot earn needs the ids and no traceback"


@pytest.mark.parametrize(
    ("error", "error_type"),
    [(make_server_error(), "HTTPException"), (OSError("connection reset"), "OSError")],
    ids=["server_error", "transport"],
)
async def test_deep_research_answers_when_its_anchor_fails_for_another_reason(
    monkeypatch: pytest.MonkeyPatch, error: Exception, error_type: str
) -> None:
    """Only a refusal drops the traceback; any other failure still answers on the token."""
    channel = text_channel_granting()
    channel.send = AsyncMock(side_effect=error)
    interaction = _ResearchInteraction(channel=channel)
    errors = capture_logs(monkeypatch=monkeypatch, level="error")

    await _launching_cog(monkeypatch=monkeypatch).deep_research(
        as_interaction(fake=interaction), topic="topic"
    )

    assert [edit.get("content") for edit in interaction.edits] == ["開研究串失敗了,等等再試一次"]
    assert errors == [
        (
            "failed to post the deep research anchor",
            {"channel_id": 20, "owner_id": 1, "error_type": error_type, "_exc_info": error},
        )
    ]


async def test_deep_research_withdraws_its_anchor_when_the_thread_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anchor posted but the thread did not, so the requester is told why, not to retry."""
    channel = text_channel_granting()
    anchor = _Anchor(error=make_forbidden(message="Missing Permissions"), channel=channel)
    channel.send = AsyncMock(return_value=anchor)
    interaction = _ResearchInteraction(channel=channel)
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")

    await _launching_cog(monkeypatch=monkeypatch).deep_research(
        as_interaction(fake=interaction), topic="topic"
    )

    assert anchor.deleted is True
    assert [edit.get("content") for edit in interaction.edits] == [
        "我在這個頻道的權限不夠,開不了研究串"
    ]
    assert [fields for _, fields in warns] == [{"message_id": 10, "owner_id": 1, "channel_id": 20}]


async def test_a_marker_launch_says_so_when_the_thread_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `<deep-research>` entry shares `_start_for`, so it answers the refusal the same way."""
    anchor = _Anchor(
        error=make_forbidden(message="Missing Permissions"), channel=text_channel_granting()
    )
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")

    await _launching_cog(monkeypatch=monkeypatch).launch(
        message=as_message(fake=anchor), brief="b"
    )

    assert [reply.get("content") for reply in anchor.replies] == [
        "我在這個頻道的權限不夠,開不了研究串"
    ]
    assert warns == [
        (
            "deep research cannot open a thread in this channel",
            {"message_id": 10, "owner_id": 300, "channel_id": 20},
        )
    ]


async def test_both_entry_points_name_the_owners_running_research_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second launch points at the running thread in one wording, however it was asked for."""
    await _seed_researching(thread_id=_THREAD_ID, owner_id=_OWNER_ID)
    channel = text_channel_granting()
    request = _Anchor(channel=channel)
    slash_anchor = _Anchor(channel=channel)
    channel.send = AsyncMock(return_value=slash_anchor)
    interaction = _ResearchInteraction(channel=channel)
    interaction.user = FakeUser(user_id=_OWNER_ID)
    cog = _launching_cog(monkeypatch=monkeypatch)

    await cog.launch(message=as_message(fake=request), brief="b")
    await cog.deep_research(as_interaction(fake=interaction), topic="topic")

    running = f"你已經有一個深度研究在進行了:<#{_THREAD_ID}>"
    assert [reply.get("content") for reply in request.replies] == [running]
    assert [edit.get("content") for edit in interaction.edits] == [running]
    assert slash_anchor.deleted is True


async def test_a_thread_failure_that_is_not_a_refusal_keeps_its_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The carve-out is for `Forbidden` alone; a 5xx is still something to look at."""
    anchor = _Anchor(error=make_server_error(), channel=text_channel_granting())
    errors = capture_logs(monkeypatch=monkeypatch, level="error")

    await _launching_cog(monkeypatch=monkeypatch).launch(
        message=as_message(fake=anchor), brief="b"
    )

    assert [reply.get("content") for reply in anchor.replies] == ["開研究串失敗了,等等再試一次"]
    assert len(errors) == 1
    assert errors[0][1].get("_exc_info") is not None, (
        "a transport failure still needs its traceback"
    )


@pytest.mark.parametrize(
    ("failure", "level", "traceback"),
    [
        (make_forbidden(message="Missing Permissions"), "warn", False),
        (make_not_found(message="Unknown Message"), "info", False),
        (make_invalid_form_body(), "info", False),
        (make_server_error(), "warn", True),
    ],
    ids=["refused", "message_gone", "reply_target_gone", "broke"],
)
async def test_a_launch_that_cannot_say_why_it_stopped_logs_it(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, level: str, traceback: bool
) -> None:
    """The launch has already ended, so its notice failing is logged here and raises nothing."""
    anchor = _Anchor(error=make_server_error(), channel=text_channel_granting())

    async def refuse(**kwargs: object) -> NoReturn:
        """Fails the reply the way Discord does."""
        del kwargs
        raise failure

    anchor.reply = refuse  # ty: ignore[invalid-assignment]
    records = {
        name: capture_logs(monkeypatch=monkeypatch, level=name) for name in ("info", "warn")
    }

    await _launching_cog(monkeypatch=monkeypatch).launch(
        message=as_message(fake=anchor), brief="b"
    )

    assert [
        (name, fields["message_id"], fields["channel_id"], "_exc_info" in fields)
        for name, found in records.items()
        for _, fields in found
    ] == [(level, 10, 20, traceback)]


@pytest.mark.parametrize(
    ("failure", "level"),
    [(make_not_found(message="Unknown Channel"), "info"), (make_forbidden(), "warn")],
    ids=["deleted", "shut_out"],
)
async def test_a_thread_the_bot_lost_access_to_is_not_reported_as_deleted(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, level: str
) -> None:
    """Both end the resume the same way, but only a deletion is the owner's doing."""
    cog = _research_cog(enabled=True)
    cog.bot = as_bot(
        fake=SimpleNamespace(
            get_channel=lambda channel_id: None, fetch_channel=AsyncMock(side_effect=failure)
        )
    )
    records = {
        name: capture_logs(monkeypatch=monkeypatch, level=name) for name in ("info", "warn")
    }

    assert await cog._fetch_thread(thread_id=5) is None
    assert [(name, fields) for name, found in records.items() for _, fields in found] == [
        (level, {"thread_id": 5})
    ]


async def test_deep_research_refuses_up_front_where_it_cannot_open_a_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The channel already says so, so nothing is posted, pinged or titled before the refusal."""
    channel = text_channel_granting(
        permissions=Permissions(
            view_channel=True, send_messages=True, send_messages_in_threads=True, attach_files=True
        )
    )
    channel.send = AsyncMock()
    interaction = _ResearchInteraction(channel=channel)

    await _launching_cog(monkeypatch=monkeypatch).deep_research(
        as_interaction(fake=interaction), topic="topic"
    )

    assert interaction.response.deferred is False
    assert interaction.response.sent == [
        {"content": "我在這個頻道的權限不夠,開不了研究串", "ephemeral": True}
    ]
    channel.send.assert_not_called()
    channel.permissions_for.assert_called_once_with(channel.guild.me)


@pytest.mark.parametrize(
    ("channel_type", "bot_in_guild", "guild_id", "reply"),
    [
        (
            PartialMessageable,
            False,
            100,
            "我沒有被加進這個伺服器,深度研究只能在我所在的伺服器裡開",
        ),
        (
            PartialMessageable,
            False,
            None,
            "深度研究只能在伺服器的一般文字頻道開(私訊或討論串裡開不了新的 thread)",
        ),
        (
            Thread,
            True,
            100,
            "深度研究只能在伺服器的一般文字頻道開(私訊或討論串裡開不了新的 thread)",
        ),
    ],
    ids=["server_without_the_bot", "dm", "thread"],
)
async def test_deep_research_names_why_it_cannot_open_here(
    monkeypatch: pytest.MonkeyPatch,
    channel_type: type,
    bot_in_guild: bool,
    guild_id: int | None,
    reply: str,
) -> None:
    """A server the bot was never added to is named as the reason the command refuses there.

    A user install reaches such servers, whose channels never resolve; a DM or a thread keeps the
    channel-type reply.
    """
    channel = MagicMock(spec=channel_type)
    channel.send = AsyncMock()
    interaction = _ResearchInteraction(channel=channel)
    if not bot_in_guild:
        interaction.guild = None
    interaction.guild_id = guild_id

    await _launching_cog(monkeypatch=monkeypatch).deep_research(
        as_interaction(fake=interaction), topic="topic"
    )

    assert interaction.response.sent == [{"content": reply, "ephemeral": True}]
    channel.send.assert_not_called()


async def test_a_refused_thread_write_is_reported_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permission that changed mid-run is expected, so it is a warn carrying only the id."""
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    thread = _RunThread(error=make_forbidden(message="Missing Permissions"), status_posts=False)

    sent = await _research_cog(enabled=True)._safe_send(
        thread=cast("Thread", thread), content="-# Researching..."
    )

    assert sent is None
    assert warns == [("research thread refused a message", {"thread_id": _THREAD_ID})]


# ----- run exits ----------------------------------------------------------------------------


class _ThreadBot:
    """The bot surface a run reads: its own user, and the thread lookup a resume starts from."""

    user = FakeUser(user_id=900, name="bot", bot=True)

    def __init__(self, thread: _RunThread | None) -> None:
        """Initializes the one thread the cache holds; None is a thread deleted meanwhile."""
        self.thread = thread

    def get_channel(self, channel_id: int) -> MagicMock | None:
        """Answers the cache with the thread dressed as the nextcord `Thread` a resume wants."""
        if self.thread is None or channel_id != self.thread.id:
            return None
        channel = MagicMock(spec=Thread)
        channel.id = self.thread.id
        channel.guild = self.thread.guild
        channel.send = self.thread.send
        channel.history = self.thread.history
        return channel

    async def fetch_channel(self, channel_id: int) -> None:
        """Answers the REST lookup the way Discord does for a deleted thread."""
        del channel_id
        raise make_not_found(message="Unknown Channel")


def _settling_client(status: str) -> SimpleNamespace:
    """A Gemini client whose research streams to its end and settles with `status`."""
    return _fake_client(
        streams=[_FakeStream([_created_event(), _completed_event()])],
        terminal=_terminal_interaction(status=status, input_tokens=1234, output_tokens=567),
    )


def _failing_client(error: Exception) -> SimpleNamespace:
    """A Gemini client whose research create fails outright."""

    async def create(**_kwargs: object) -> None:
        raise error

    return SimpleNamespace(aio=SimpleNamespace(interactions=SimpleNamespace(create=create)))


def _running_cog(
    monkeypatch: pytest.MonkeyPatch, client: object, thread: _RunThread | None = None
) -> research_cog.ResearchCogs:
    """A cog that runs research on `client` end to end and delivers with hosting off.

    `thread` is what a resume finds by id. The price table lists no agent, as production's does
    not, so the usage footer is what a report really ends with rather than whatever price
    table the worker holds.
    """
    cog = _launching_cog(monkeypatch=monkeypatch)
    cog.bot = as_bot(fake=_ThreadBot(thread=thread))
    cog.interactions_client = as_client(fake=client)
    monkeypatch.setattr("discordbot.utils.model_pricing.load_model_info", lambda: {})
    return cog


async def _launch_run(cog: research_cog.ResearchCogs, thread: _RunThread) -> None:
    """Launches a research from a marker and waits out the run it spawned."""
    anchor = _Anchor(channel=text_channel_granting(), thread=thread)
    await cog.launch(message=as_message(fake=anchor), brief="b")
    await asyncio.gather(*cog._tasks)


async def _resume_run(
    cog: research_cog.ResearchCogs,
    stored_id: bool = True,
    agent: str = "antigravity-preview-09-2026",
) -> None:
    """Resumes the owner's in-flight row after a restart and waits out the run it spawned."""
    await _seed_researching(
        thread_id=_THREAD_ID, owner_id=_OWNER_ID, stored_id=stored_id, agent=agent
    )
    # The restart comes after the run that left the row.
    cog._started_at = database_now()
    await cog._resume_all()
    await asyncio.gather(*cog._tasks)


async def _assert_owner_released(cog: research_cog.ResearchCogs, phase: str) -> None:
    """The run's row ended in `phase` and nothing holds its owner's one slot any more."""
    async with rdb.open_session() as session:
        row = await session.get(entity=rdb.ResearchSessionRow, ident=_THREAD_ID)
    assert row is not None
    assert row.phase == phase
    assert await rdb.active_thread_for_owner(owner_id=_OWNER_ID) is None
    assert cog._active_threads == set()


def _assert_pings_only_the_owner(write: dict[str, object]) -> None:
    """The write mentions its owner, and its mention policy lets nobody else be pinged."""
    assert f"<@{_OWNER_ID}>" in str(write["content"])
    mentions = cast("AllowedMentions", write["allowed_mentions"])
    assert mentions.everyone is False
    assert mentions.roles is False
    assert isinstance(mentions.users, list)
    assert [user.id for user in mentions.users] == [_OWNER_ID]


@pytest.mark.parametrize(
    ("settles", "phase"),
    [
        ("completed", "done"),
        ("cancelled", "cancelled"),
        ("budget_exceeded", "failed"),
        (RuntimeError("quota"), "failed"),
    ],
    ids=["completed", "cancelled", "budget_exceeded", "create_fails"],
)
async def test_every_exit_of_a_launched_run_records_its_phase_and_frees_the_owner(
    monkeypatch: pytest.MonkeyPatch, settles: str | Exception, phase: str
) -> None:
    client = (
        _failing_client(error=settles)
        if isinstance(settles, Exception)
        else _settling_client(status=settles)
    )
    cog = _running_cog(monkeypatch=monkeypatch, client=client)

    await _launch_run(cog=cog, thread=_RunThread())

    await _assert_owner_released(cog=cog, phase=phase)


@pytest.mark.parametrize("resumed", [False, True], ids=["launched", "resumed"])
async def test_a_run_whose_delivery_raises_still_ends_failed_and_frees_the_owner(
    monkeypatch: pytest.MonkeyPatch, resumed: bool
) -> None:
    thread = _RunThread()
    cog = _running_cog(
        monkeypatch=monkeypatch, client=_settling_client(status="completed"), thread=thread
    )
    cog.media_delivery = cast(
        "MediaDeliveryPlanner", SimpleNamespace(plan=AsyncMock(side_effect=OSError("host down")))
    )

    if resumed:
        await _resume_run(cog=cog)
    else:
        await _launch_run(cog=cog, thread=thread)

    await _assert_owner_released(cog=cog, phase="failed")
    assert [write["content"] for write in thread.writes] == [
        "-# Researching... (Antigravity)",
        "<@300> ⚠️",
        "-# Research failed (Antigravity)",
    ]
    _assert_pings_only_the_owner(write=thread.writes[1])


def _lock_reply_db(
    monkeypatch: pytest.MonkeyPatch, call: str
) -> list[tuple[str, dict[str, object]]]:
    """Makes one research store call fail the way a locked `reply.db` does; returns the errors."""

    async def _locked(**_kwargs: object) -> None:
        raise OperationalError("research", None, sqlite3.OperationalError("database is locked"))

    monkeypatch.setattr(target=rdb, name=call, value=_locked)
    return capture_logs(monkeypatch=monkeypatch, level="error")


# What `_release` logs when the store refuses a run's terminal phase.
_UNRECORDED = "failed to record how a research run ended"


@pytest.mark.parametrize("resumed", [False, True], ids=["launched", "resumed"])
async def test_a_delivered_report_stays_delivered_when_its_phase_write_fails(
    monkeypatch: pytest.MonkeyPatch, resumed: bool
) -> None:
    errors = _lock_reply_db(monkeypatch=monkeypatch, call="set_phase")
    thread = _RunThread()
    cog = _running_cog(
        monkeypatch=monkeypatch, client=_settling_client(status="completed"), thread=thread
    )

    if resumed:
        await _resume_run(cog=cog)
    else:
        await _launch_run(cog=cog, thread=thread)

    # The status line became the report, and nothing followed it.
    assert [str(write["content"]).split("\n")[0] for write in thread.writes] == [
        "-# Researching... (Antigravity)",
        "# Report",
    ]
    assert [(message, fields.get("phase")) for message, fields in errors] == [
        (_UNRECORDED, "done")
    ]
    assert errors[0][1].get("_exc_info") is not None
    assert cog._active_threads == set()


@pytest.mark.parametrize(
    ("settles", "writes", "logged"),
    [
        (
            "cancelled",
            ["-# Researching... (Antigravity)", "<@300> ⚠️", "-# Research failed (Antigravity)"],
            [(_UNRECORDED, "cancelled")],
        ),
        (
            RuntimeError("quota"),
            ["-# Researching... (Antigravity)", "<@300> ⚠️", "-# Research failed (Antigravity)"],
            [("research failed", None), (_UNRECORDED, "failed")],
        ),
        (
            None,
            [
                "-# Research failed (Antigravity)",
                "<@300> 重啟後沒辦法接回剛剛的研究,麻煩重新發起一次",
            ],
            [(_UNRECORDED, "failed")],
        ),
    ],
    ids=["cancelled", "create_fails", "resume_lost"],
)
async def test_a_failed_run_whose_phase_write_fails_still_tells_its_owner_and_frees_its_thread(
    monkeypatch: pytest.MonkeyPatch,
    settles: str | Exception | None,
    writes: list[str],
    logged: list[tuple[str, str | None]],
) -> None:
    errors = _lock_reply_db(monkeypatch=monkeypatch, call="set_phase")
    thread = _RunThread()
    if settles is None:
        # A resumed row whose interaction id was never stored has nothing to re-attach to.
        cog = _running_cog(monkeypatch=monkeypatch, client=SimpleNamespace(), thread=thread)
        await _resume_run(cog=cog, stored_id=False)
    else:
        client = (
            _failing_client(error=settles)
            if isinstance(settles, Exception)
            else _settling_client(status=settles)
        )
        cog = _running_cog(monkeypatch=monkeypatch, client=client)
        await _launch_run(cog=cog, thread=thread)

    assert [write["content"] for write in thread.writes] == writes
    assert [(message, fields.get("phase")) for message, fields in errors] == logged
    assert cog._active_threads == set()


@pytest.mark.parametrize("call", ["active_thread_for_owner", "insert_session"])
async def test_deep_research_answers_and_withdraws_its_posts_when_reply_db_fails(
    monkeypatch: pytest.MonkeyPatch, call: str
) -> None:
    errors = _lock_reply_db(monkeypatch=monkeypatch, call=call)
    channel = text_channel_granting()
    thread = _RunThread()
    anchor = _Anchor(channel=channel, thread=thread)
    channel.send = AsyncMock(return_value=anchor)
    interaction = _ResearchInteraction(channel=channel)
    cog = _running_cog(monkeypatch=monkeypatch, client=SimpleNamespace())

    await cog.deep_research(as_interaction(fake=interaction), topic="topic")

    assert [edit.get("content") for edit in interaction.edits] == ["開研究串失敗了,等等再試一次"]
    assert anchor.deleted is True
    # Only the insert comes after the thread is opened, and a thread with no row is not a run.
    assert thread.deleted is (call == "insert_session")
    assert thread.writes == []
    assert cog._active_threads == set()
    assert not cog._tasks
    assert len(errors) == 1
    assert errors[0][1].get("_exc_info") is not None


@pytest.mark.parametrize("refused", [False, True], ids=["deleted", "delete_refused"])
async def test_a_marker_launch_says_so_and_withdraws_its_thread_when_reply_db_fails(
    monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    _lock_reply_db(monkeypatch=monkeypatch, call="insert_session")
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    thread = _RunThread(error=make_forbidden(message="Missing Permissions") if refused else None)
    anchor = _Anchor(channel=text_channel_granting(), thread=thread)
    cog = _running_cog(monkeypatch=monkeypatch, client=SimpleNamespace())

    await cog.launch(message=as_message(fake=anchor), brief="b")

    assert [reply.get("content") for reply in anchor.replies] == ["開研究串失敗了,等等再試一次"]
    assert thread.deleted is not refused
    # The marker's anchor is the reply that promised the run, so it stays.
    assert anchor.deleted is False
    assert cog._active_threads == set()
    # A launch never asks for the Manage Threads a delete takes, so a refusal logs the ids alone.
    assert warns == (
        [
            (
                "research thread of a failed launch could not be deleted",
                {"thread_id": _THREAD_ID, "owner_id": _OWNER_ID},
            )
        ]
        if refused
        else []
    )


@pytest.mark.parametrize("stored_id", [True, False], ids=["resume_fails", "no_stored_id"])
async def test_a_resume_that_cannot_reattach_frees_the_owner_and_tells_only_them(
    monkeypatch: pytest.MonkeyPatch, stored_id: bool
) -> None:
    async def _expired(**_kwargs: object) -> None:
        raise RuntimeError("interaction expired")

    monkeypatch.setattr(target=research_cog, name="resume_research_stream", value=_expired)
    thread = _RunThread()
    cog = _running_cog(monkeypatch=monkeypatch, client=SimpleNamespace(), thread=thread)

    await _resume_run(cog=cog, stored_id=stored_id)

    await _assert_owner_released(cog=cog, phase="failed")
    notice = thread.writes[-1]
    assert notice["content"] == "<@300> 重啟後沒辦法接回剛剛的研究,麻煩重新發起一次"
    _assert_pings_only_the_owner(write=notice)


async def test_a_resume_that_cannot_reattach_ends_its_own_status_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _expired(**_kwargs: object) -> None:
        raise RuntimeError("interaction expired")

    monkeypatch.setattr(target=research_cog, name="resume_research_stream", value=_expired)
    thread = _RunThread()
    cog = _running_cog(monkeypatch=monkeypatch, client=SimpleNamespace(), thread=thread)
    errors = capture_logs(monkeypatch=monkeypatch, level="error")

    await _resume_run(cog=cog)

    # The same failure a fresh run reports at `error`: the report it was waiting for is lost.
    assert [(message, fields["error_type"]) for message, fields in errors] == [
        ("research resume failed", "RuntimeError")
    ]
    assert [write["content"] for write in thread.writes] == [
        "-# Researching... (Antigravity)",
        "-# Research failed (Antigravity)",
        "<@300> 重啟後沒辦法接回剛剛的研究,麻煩重新發起一次",
    ]
    assert thread.sends == 2, "the failed line is the resume's own status edited, not a new post"


@pytest.mark.parametrize(
    ("settles", "stored_id", "ends_as"),
    [
        ("completed", True, "# Report\nbody"),
        (RuntimeError("interaction expired"), True, "-# Research failed (Antigravity)"),
        ("completed", False, "-# Research failed (Antigravity)"),
    ],
    ids=["delivers", "resume_fails", "no_stored_id"],
)
async def test_a_resume_ends_the_status_line_posted_before_the_restart(
    monkeypatch: pytest.MonkeyPatch, settles: str | Exception, stored_id: bool, ends_as: str
) -> None:
    """The resume takes the pre-restart line over, so nothing is left claiming to research."""
    if isinstance(settles, Exception):

        async def _expired(**_kwargs: object) -> None:
            raise settles

        monkeypatch.setattr(target=research_cog, name="resume_research_stream", value=_expired)
    before_restart = FakeDiscordMessage(
        author=_ThreadBot.user, content="-# Researching... (Antigravity, 12m30s)\n-# Weighing"
    )
    # Neither a member's copy of the line nor a bot line of another kind is the status.
    thread = _RunThread(
        earlier=[
            FakeDiscordMessage(author=FakeUser(user_id=_OWNER_ID), content="-# Researching... ?"),
            FakeDiscordMessage(author=_ThreadBot.user, content="-# Research failed (Antigravity)"),
            before_restart,
        ]
    )
    cog = _running_cog(
        monkeypatch=monkeypatch, client=_settling_client(status="completed"), thread=thread
    )

    await _resume_run(cog=cog, stored_id=stored_id)

    assert str(before_restart.edits[-1]["content"]).startswith(ends_as)
    assert not any(str(write["content"]).startswith("-# Researching") for write in thread.writes)


@pytest.mark.parametrize("refused", [True, False], ids=["missing_access", "failing"])
async def test_a_resume_whose_history_read_fails_runs_on_a_status_line_of_its_own(
    monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    """A 403 is the server's setting and logs the id alone; any other failure keeps its trace."""
    error = make_forbidden(message="Missing Access") if refused else make_server_error()
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    thread = _RunThread()

    def _failed(**_kwargs: object) -> NoReturn:
        raise error

    monkeypatch.setattr(target=thread, name="history", value=_failed)
    cog = _running_cog(
        monkeypatch=monkeypatch, client=_settling_client(status="completed"), thread=thread
    )

    await _resume_run(cog=cog)

    await _assert_owner_released(cog=cog, phase="done")
    assert thread.writes[0]["content"] == "-# Researching... (Antigravity)"
    assert warns == [
        ("research thread refused the history read", {"thread_id": _THREAD_ID})
        if refused
        else (
            "failed to read research thread history",
            {"thread_id": _THREAD_ID, "error_type": "HTTPException", "_exc_info": error},
        )
    ]


async def test_a_resume_whose_thread_is_gone_still_records_how_the_run_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cog = _running_cog(monkeypatch=monkeypatch, client=_settling_client(status="cancelled"))

    await _resume_run(cog=cog)

    await _assert_owner_released(cog=cog, phase="cancelled")


@pytest.mark.parametrize(
    ("settles", "reason", "footer"),
    [(RuntimeError("quota"), "quota", "RuntimeError"), ("cancelled", "研究被取消了", None)],
    ids=["create_fails", "cancelled"],
)
async def test_a_failed_run_tells_only_its_owner_why(
    monkeypatch: pytest.MonkeyPatch, settles: str | Exception, reason: str, footer: str | None
) -> None:
    client = (
        _failing_client(error=settles)
        if isinstance(settles, Exception)
        else _settling_client(status=settles)
    )
    thread = _RunThread()

    await _launch_run(cog=_running_cog(monkeypatch=monkeypatch, client=client), thread=thread)

    assert [write["content"] for write in thread.writes] == [
        "-# Researching... (Antigravity)",
        "<@300> ⚠️",
        "-# Research failed (Antigravity)",
    ]
    notice = thread.writes[1]
    _assert_pings_only_the_owner(write=notice)
    embed = cast("Embed", notice["embed"])
    assert embed.description == f"```\n{reason}\n```"
    assert embed.footer.text == footer


async def test_a_delivered_report_pings_only_its_owner_over_the_runs_own_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cog = _running_cog(monkeypatch=monkeypatch, client=_settling_client(status="completed"))
    thread = _RunThread()

    await _launch_run(cog=cog, thread=thread)

    # The counts the interaction reported, and no cost: the table has no entry for the agent.
    report = thread.writes[-1]
    agent_name = cog.runtime_models.antigravity_model.name
    assert report["content"] == f"# Report\nbody\n\n<@300>\n\n-# {agent_name} · ⬆ 1,234 ⬇ 567"
    _assert_pings_only_the_owner(write=report)


async def test_a_resumed_report_pings_only_its_owner_over_the_runs_own_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = _RunThread()
    cog = _running_cog(
        monkeypatch=monkeypatch, client=_settling_client(status="completed"), thread=thread
    )

    # The row's own agent, not the catalog's: a restart after a repoint still names and prices
    # the agent the run was launched on, which a table pricing only the catalog's leaves unpriced.
    rates = {
        cog.runtime_models.antigravity_model.name: ModelPriceEntry(
            input_cost_per_token=1e-6, output_cost_per_token=2e-6
        )
    }
    monkeypatch.setattr("discordbot.utils.model_pricing.load_model_info", lambda: rates)
    await _resume_run(cog=cog, agent="antigravity-launched-agent")

    await _assert_owner_released(cog=cog, phase="done")
    report = thread.writes[-1]
    assert report["content"] == (
        "# Report\nbody\n\n<@300>\n\n-# antigravity-launched-agent · ⬆ 1,234 ⬇ 567"
    )
    _assert_pings_only_the_owner(write=report)


# What each write of a failing run logs when Discord refuses it, and when it fails otherwise.
_FAILED_RUN_WRITE_LOGS = {
    "status": ("research thread refused a message", "failed to send research thread message"),
    "notice": (
        "research thread refused the failure notice",
        "failed to post research failure notice",
    ),
    "status edit": (
        "research thread refused the status edit",
        "failed to edit research status message",
    ),
    "terminal status": (
        "research thread refused the terminal status",
        "failed to post terminal research status",
    ),
}


@pytest.mark.parametrize("refused", [True, False], ids=["refused", "failing"])
@pytest.mark.parametrize(
    ("status_posts", "writes"),
    [
        (True, ("notice", "status edit", "terminal status")),
        (False, ("status", "notice", "terminal status")),
    ],
    ids=["status_posted", "status_lost"],
)
async def test_a_failed_run_on_a_broken_thread_logs_each_write_and_still_ends(
    monkeypatch: pytest.MonkeyPatch, refused: bool, status_posts: bool, writes: tuple[str, ...]
) -> None:
    """A refusal is the server's setting and logs the id alone; any other failure keeps its trace."""
    error = make_forbidden(message="Missing Access") if refused else make_server_error()
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    errors = capture_logs(monkeypatch=monkeypatch, level="error")
    cog = _running_cog(
        monkeypatch=monkeypatch, client=_failing_client(error=RuntimeError("quota"))
    )

    await _launch_run(cog=cog, thread=_RunThread(error=error, status_posts=status_posts))

    trace = {} if refused else {"error_type": "HTTPException", "_exc_info": error}
    assert warns == [
        (_FAILED_RUN_WRITE_LOGS[write][0 if refused else 1], {"thread_id": _THREAD_ID, **trace})
        for write in writes
    ]
    assert [message for message, _ in errors] == ["research failed"]
    await _assert_owner_released(cog=cog, phase="failed")


async def test_a_refused_report_logs_each_write_without_a_traceback_and_still_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    errors = capture_logs(monkeypatch=monkeypatch, level="error")
    cog = _running_cog(monkeypatch=monkeypatch, client=_settling_client(status="completed"))

    await _launch_run(cog=cog, thread=_RunThread(error=make_forbidden(message="Missing Access")))

    where = {"thread_id": _THREAD_ID, "chunk_index": 0}
    assert warns == [
        ("research thread refused the report edit", where),
        ("research thread refused a report message", {**where, "is_last": True}),
    ]
    assert errors == []
    await _assert_owner_released(cog=cog, phase="done")


async def test_a_failing_report_logs_each_write_with_its_traceback_and_still_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = make_server_error()
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    errors = capture_logs(monkeypatch=monkeypatch, level="error")
    cog = _running_cog(monkeypatch=monkeypatch, client=_settling_client(status="completed"))

    await _launch_run(cog=cog, thread=_RunThread(error=error))

    where = {"thread_id": _THREAD_ID, "chunk_index": 0}
    trace = {"error_type": "HTTPException", "_exc_info": error}
    assert warns == [("failed to edit research status into report", {**where, **trace})]
    # The last message carries the file, ping and footer, so losing it is an error.
    assert errors == [
        (
            "failed to post research report message",
            {**where, "is_last": True, "has_files": True, **trace},
        )
    ]
    await _assert_owner_released(cog=cog, phase="done")


async def test_deep_research_starts_under_its_first_line_when_no_proxy_client_can_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The title is best-effort: an empty proxy key costs the generated title, not the run."""
    # The SDK also accepts `OPENAI_ADMIN_KEY` from the environment, which would build the client.
    monkeypatch.delenv(name="OPENAI_ADMIN_KEY", raising=False)
    thread = _RunThread()
    cog = _running_cog(monkeypatch=monkeypatch, client=_settling_client(status="completed"))
    # The real title generator, which `_launching_cog` stubs out.
    monkeypatch.delattr(target=cog, name="_generate_thread_name")
    cog.config.api_key = ""
    channel = text_channel_granting()
    anchor = _Anchor(channel=channel, thread=thread)
    anchor.create_thread = AsyncMock(return_value=thread)
    channel.send = AsyncMock(return_value=anchor)
    interaction = _ResearchInteraction(channel=channel)
    infos = capture_logs(monkeypatch=monkeypatch, level="info")

    await cog.deep_research(as_interaction(fake=interaction), topic="TPU landscape\nand history")
    await asyncio.gather(*cog._tasks)

    anchor.create_thread.assert_awaited_once_with(name="TPU landscape", auto_archive_duration=1440)
    assert [edit.get("content") for edit in interaction.edits] == [f"開好了:<#{_THREAD_ID}>"]
    assert anchor.deleted is False
    assert (
        "no proxy key for the deep research thread title; using the brief's first line",
        {},
    ) in infos, "an unset key is an unconfigured-feature skip, so it carries no traceback"
