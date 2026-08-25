"""Tests for `/feedback`: the store, the issue text, the panel, and the submit ordering.

Nothing here touches the network or a real credential. The GitHub side is a fake object
handed to the cog, and the write-up model is a stub, which is also how the feature is
meant to be reasoned about: every external call is allowed to fail, and the report has to
survive all of them.
"""

from typing import Any, cast
import asyncio
from pathlib import Path
from datetime import UTC, datetime, timedelta

import httpx
from openai import AsyncOpenAI
import pytest
from nextcord import Embed, Interaction
from pydantic import Field
from nextcord.ui import Button, StringSelect
from nextcord.ext import commands

from discordbot.typings.config import FeedbackConfig
from discordbot.typings.models import ModelSettings
from discordbot.utils.timezone import database_now
from discordbot.cogs.feedback.cog import FeedbackCogs
from discordbot.cogs.feedback.auth import AppCredentials, GitHubAuthError, TokenCredentials
from discordbot.cogs.feedback.views import (
    PanelRows,
    TicketRow,
    ReportModal,
    TicketDetail,
    TicketDetailView,
    FeedbackPanelView,
    build_panel_embed,
    build_detail_embed,
)
from discordbot.cogs.feedback.github import (
    _COMMENTS_PER_PAGE,
    REPORTER_COMMENT_MARKER,
    GitHubIssues,
    IssueComment,
    IssueSnapshot,
    GitHubIssuesError,
    select_conversation,
)
from discordbot.cogs.feedback.notice import closing_comment
from discordbot.cogs.feedback.writeup import (
    StoredDraft,
    ReportWriteUp,
    stored_draft,
    render_issue_body,
    label_for_category,
    initial_issue_title,
)
from discordbot.cogs.feedback.database import (
    FeedbackTicket,
    get_ticket,
    create_ticket,
    store_write_up,
    list_user_tickets,
    attach_issue_number,
    count_relayed_reply,
    tickets_awaiting_issue,
    seconds_since_last_ticket,
    tickets_awaiting_close_check,
    close_notices_awaiting_delivery,
)

from tests.helpers.discord_mocks import FakeUser, FakeInteraction

# Never a real credential: nothing holding it ever reaches the network.
_FAKE_TOKEN = "not-a-real-value"  # noqa: S105 -- a placeholder, not a secret


class FakeIssues:
    """Stand-in for the GitHub REST surface, recording what the cog asked for."""

    def __init__(self, *, fail_create: bool = False) -> None:
        """Initializes an empty repository whose create can be made to fail."""
        self.fail_create = fail_create
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.comments: list[dict[str, Any]] = []
        self.labelled: list[dict[str, Any]] = []
        self.snapshots: dict[int, IssueSnapshot] = {}
        self.conversation: list[Any] = []
        self.next_number = 460

    async def create_issue(self, *, title: str, body: str, labels: list[str]) -> int:
        """Records a created issue and hands back its number."""
        if self.fail_create:
            raise GitHubIssuesError("POST /issues answered 503")
        self.created.append({"title": title, "body": body, "labels": labels})
        number = self.next_number
        self.next_number += 1
        return number

    async def update_issue(self, *, number: int, title: str, body: str) -> None:
        """Records an edited issue."""
        self.updated.append({"number": number, "title": title, "body": body})

    async def add_labels(self, *, number: int, labels: list[str]) -> None:
        """Records labels added after the write-up."""
        self.labelled.append({"number": number, "labels": labels})

    async def add_comment(self, *, number: int, body: str) -> None:
        """Records a relayed comment."""
        self.comments.append({"number": number, "body": body})

    async def read_issue(self, *, number: int) -> IssueSnapshot:
        """Returns a prepared snapshot, or an open issue with no comments."""
        return self.snapshots.get(
            number, IssueSnapshot(number=number, state="open", state_reason=None, comment_count=0)
        )

    async def read_conversation(self, *, number: int) -> list[Any]:
        """Returns the prepared conversation."""
        return self.conversation


def _config(**overrides: object) -> FeedbackConfig:
    """Builds a reporting config from alias keys, hermetically.

    `model_validate` rather than the constructor on purpose: it skips the settings
    sources, so a checkout or a CI runner that happens to export FEEDBACK_GITHUB_TOKEN cannot
    change what a test is asserting about.
    """
    values: dict[str, object] = {
        "FEEDBACK_ENABLED": True,
        "FEEDBACK_GITHUB_TOKEN": "fake-credential",
        "FEEDBACK_GITHUB_APP_ID": "",
        "FEEDBACK_GITHUB_APP_PRIVATE_KEY_PATH": "",
        "FEEDBACK_GITHUB_REPOSITORY": "owner/name",
        "FEEDBACK_CONTACT": "",
        "FEEDBACK_MAX_OPEN_REPORTS": 3,
        "FEEDBACK_SUBMIT_COOLDOWN_SECONDS": 300,
    }
    values.update(overrides)
    return FeedbackConfig.model_validate(values)


def _cog(*, issues: FakeIssues, config: FeedbackConfig | None = None) -> FeedbackCogs:
    """Builds a cog with the GitHub side faked out and no bot behind it."""
    cog = FeedbackCogs(cast("commands.Bot", object()))
    cog.config = config or _config()
    # Both are cached_property, so writing the instance attribute is the injection point;
    # nothing in the cog reads either one any other way. The LLM client matters even
    # where no model call is expected: building the real one needs a credential, and a
    # test that quietly depends on the developer's own .env passes locally and fails in
    # CI, which is exactly what happened to this file.
    cog.issues = cast("Any", issues)
    cog.client = cast("Any", object())
    return cog


async def _tidy_write_up(
    *, client: AsyncOpenAI, model: ModelSettings, ticket: FeedbackTicket
) -> ReportWriteUp:
    """Stands in for the model, returning one fixed write-up."""
    return ReportWriteUp(
        label="下載會卡住",
        category="bug",
        title="fix: download stalls midway",
        body="Downloading a long video stalls.",
    )


async def _drain(*, cog: FeedbackCogs) -> None:
    """Waits for the background tasks the cog spawned during a call."""
    while cog._background:
        await asyncio.gather(*tuple(cog._background), return_exceptions=True)


async def _no_cooldown(*, user_id: int) -> float | None:
    """Stands in for the submission cooldown when a test files several in a row."""
    return None


def _ticket(**overrides: object) -> FeedbackTicket:
    """Builds a stored report with sensible defaults."""
    values: dict[str, Any] = {
        "ticket_id": 1,
        "issue_number": 460,
        "user_id": 7,
        "user_name": "alice",
        "display_name": "Alice",
        "guild_id": 100,
        "guild_name": "test guild",
        "channel_id": 200,
        "locale": "zh-TW",
        "raw_text": "下載長影片會卡住",
        "label": "",
        "category": "",
        "draft_title": "",
        "draft_body": "",
        "relayed_replies": 0,
        "created_at": database_now(),
    }
    values.update(overrides)
    return FeedbackTicket(**values)


def _row(*, snapshot: IssueSnapshot | None, **overrides: object) -> TicketRow:
    """Builds one panel row."""
    return TicketRow(ticket=_ticket(**overrides), snapshot=snapshot)


# --------------------------------------------------------------------------- store


async def test_a_report_is_stored_before_it_has_an_issue(feedback_isolated_db: None) -> None:
    """A fresh report is durable immediately, with no issue number yet."""
    ticket = await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=100,
        guild_name="test guild",
        channel_id=200,
        locale="zh-TW",
        raw_text="壞掉了",
    )
    assert ticket.issue_number is None
    queued = await tickets_awaiting_issue(limit=10)
    assert [row.ticket_id for row in queued] == [ticket.ticket_id]
    assert queued[0].raw_text == "壞掉了"
    await attach_issue_number(ticket_id=ticket.ticket_id, issue_number=460)
    assert await tickets_awaiting_issue(limit=10) == []
    stored = await get_ticket(ticket_id=ticket.ticket_id)
    assert stored is not None
    assert stored.issue_number == 460


async def test_the_store_lists_only_the_asking_users_reports(feedback_isolated_db: None) -> None:
    """One person's panel never shows another person's reports."""
    for user_id in (7, 8, 7):
        await create_ticket(
            user_id=user_id,
            user_name="u",
            display_name="U",
            guild_id=None,
            guild_name="",
            channel_id=None,
            locale="",
            raw_text=f"report from {user_id}",
        )
    mine = await list_user_tickets(user_id=7, limit=10)
    assert len(mine) == 2
    assert {ticket.user_id for ticket in mine} == {7}
    # Newest first, so the report someone just filed is the one they see at the top.
    assert mine[0].ticket_id > mine[1].ticket_id


async def test_the_write_up_lands_next_to_the_original(feedback_isolated_db: None) -> None:
    """Storing a write-up never overwrites what the reporter wrote."""
    ticket = await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="原本寫的字",
    )
    await store_write_up(
        ticket_id=ticket.ticket_id,
        label="下載會卡住",
        category="bug",
        draft_title="fix: download stalls",
        draft_body="The download stalls.",
    )
    stored = await get_ticket(ticket_id=ticket.ticket_id)
    assert stored is not None
    assert stored.raw_text == "原本寫的字"
    assert stored.label == "下載會卡住"
    assert stored.draft_title == "fix: download stalls"


async def test_a_relayed_reply_is_counted(feedback_isolated_db: None) -> None:
    """Replies this bot posts are counted so they never read back as an answer."""
    ticket = await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="x",
    )
    await count_relayed_reply(ticket_id=ticket.ticket_id)
    await count_relayed_reply(ticket_id=ticket.ticket_id)
    stored = await get_ticket(ticket_id=ticket.ticket_id)
    assert stored is not None
    assert stored.relayed_replies == 2


async def test_the_cooldown_reads_the_last_submission(feedback_isolated_db: None) -> None:
    """Someone with no reports has no cooldown; a fresh one starts near zero."""
    assert await seconds_since_last_ticket(user_id=7) is None
    await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="x",
    )
    elapsed = await seconds_since_last_ticket(user_id=7)
    assert elapsed is not None
    assert elapsed < 60


# ----------------------------------------------------------------------- issue text


def test_the_original_wording_survives_a_code_fence_in_the_report() -> None:
    """A reporter pasting a fence cannot break out of the quoted block.

    Fencing is what keeps `@name` and `#123` inert on GitHub, so a report that closes
    the fence early would turn its own text back into live references.
    """
    body = render_issue_body(ticket=_ticket(raw_text="see ```py\nprint(1)\n``` and @someone #12"))
    original = body.split("<summary>Original wording</summary>")[1]
    assert "````text" in original
    assert "@someone #12" in original
    # The opening fence is longer than any run of backticks the reporter wrote, so the
    # inner one cannot close it: everything they typed stays inside exactly one block.
    lines = [line for line in original.splitlines() if line.strip()]
    assert lines[0] == "````text"
    assert lines[-2] == "````"
    assert lines[-1] == "</details>"


def test_the_issue_body_names_the_reporter() -> None:
    """The maintainer can tell who filed a report and from where without a lookup."""
    body = render_issue_body(ticket=_ticket())
    assert "Alice" in body
    assert "alice" in body
    assert "test guild" in body


def test_a_write_up_leads_the_body_and_keeps_the_original() -> None:
    """The tidied text goes on top; the reporter's words stay underneath."""
    write_up = ReportWriteUp(
        label="下載會卡住",
        category="bug",
        title="fix: download stalls midway",
        body="Downloading a long video stalls.",
    )
    body = render_issue_body(ticket=_ticket(), lead=write_up.body)
    assert body.startswith("Downloading a long video stalls.")
    assert "下載長影片會卡住" in body


def test_the_first_issue_title_is_the_reporters_own_first_line() -> None:
    """Before any write-up exists, the only accurate title is what they wrote."""
    assert initial_issue_title(ticket=_ticket(raw_text="\n\n第一行\n第二行")) == "第一行"


def test_an_empty_report_still_gets_a_title() -> None:
    """A report with nothing usable in it still opens an identifiable issue."""
    assert initial_issue_title(ticket=_ticket(raw_text="   \n ")) == "user report #1"


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("bug", ["user-report", "bug"]),
        ("feature", ["user-report", "feature"]),
        ("question", ["user-report", "question"]),
        ("nonsense", ["user-report"]),
    ],
)
def test_labels_follow_the_category(category: str, expected: list[str]) -> None:
    """Every report is findable as one, and a category the repo has no label for is dropped."""
    assert label_for_category(category=category) == expected


# ------------------------------------------------------------------- comment filter


def test_only_the_developer_and_the_reporter_are_shown() -> None:
    """A passer-by on a public repository is never presented as the developer's reply."""
    conversation = select_conversation(
        comments=[
            {
                "user": {"login": "mai", "type": "User"},
                "author_association": "OWNER",
                "body": "找到原因了",
                "created_at": "2026-02-04T10:00:00Z",
            },
            {
                "user": {"login": "stranger", "type": "User"},
                "author_association": "NONE",
                "body": "+1 我也遇到",
                "created_at": "2026-02-04T11:00:00Z",
            },
            {
                "user": {"login": "some-bot", "type": "Bot"},
                "author_association": "OWNER",
                "body": "CI failed",
                "created_at": "2026-02-04T12:00:00Z",
            },
            {
                "user": {"login": "mai", "type": "User"},
                "author_association": "OWNER",
                "body": f"{REPORTER_COMMENT_MARKER}\n**Alice** wrote from Discord:\n\n補充一下",
                "created_at": "2026-02-05T09:00:00Z",
            },
        ]
    )
    assert [comment.from_reporter for comment in conversation] == [False, True]
    assert conversation[0].body == "找到原因了"
    # The marker is plumbing, so it never reaches the panel.
    assert REPORTER_COMMENT_MARKER not in conversation[1].body


# ------------------------------------------------------------------------- statuses


@pytest.mark.parametrize(
    ("snapshot", "overrides", "expected", "outstanding"),
    [
        (None, {"issue_number": None}, "⏳ 建立中", True),
        (None, {}, "❔ 讀不到狀態", False),
        (
            IssueSnapshot(number=460, state="open", state_reason=None, comment_count=0),
            {},
            "🟡 還沒回覆",
            True,
        ),
        (
            IssueSnapshot(number=460, state="open", state_reason=None, comment_count=1),
            {},
            "🟢 處理中",
            True,
        ),
        (
            IssueSnapshot(number=460, state="closed", state_reason="completed", comment_count=2),
            {},
            "✅ 已處理",
            False,
        ),
        (
            IssueSnapshot(number=460, state="closed", state_reason="not_planned", comment_count=1),
            {},
            "⚪ 不處理",
            False,
        ),
        # Merged into another issue. Reading this as work that got done is what the plain
        # else branch used to do, and it is the one closed state that is not an outcome.
        (
            IssueSnapshot(number=460, state="closed", state_reason="duplicate", comment_count=1),
            {},
            "🔁 併入其他單",
            False,
        ),
    ],
)
def test_the_status_comes_from_the_issue(
    snapshot: IssueSnapshot | None, overrides: dict[str, Any], expected: str, outstanding: bool
) -> None:
    """Nothing about the status is stored, so it can never drift from the issue."""
    row = _row(snapshot=snapshot, **overrides)
    assert row.status.text == expected
    assert row.status.outstanding is outstanding


def test_a_reporters_own_reply_does_not_read_as_an_answer() -> None:
    """Someone adding detail to their own report must not look like a reply to them."""
    row = _row(
        snapshot=IssueSnapshot(number=460, state="open", state_reason=None, comment_count=1),
        relayed_replies=1,
    )
    assert row.status.text == "🟡 還沒回覆"


# ---------------------------------------------------------------------------- panel


def test_the_empty_panel_still_explains_where_a_report_goes() -> None:
    """Someone opening the panel for the first time is told what filing one means."""
    embed = build_panel_embed(rows=[])
    assert embed.description is not None
    assert "公開" in embed.description
    assert not embed.fields


def test_the_panel_lists_a_report_by_its_number() -> None:
    """The ticket number is the issue number, and it leads the row."""
    embed = build_panel_embed(rows=[_row(snapshot=None)])
    assert embed.fields[0].name is not None
    assert embed.fields[0].name.startswith("#460")


def test_a_report_without_a_write_up_is_listed_by_its_own_first_line() -> None:
    """The panel is usable during the window before the background write-up lands."""
    row = _row(snapshot=None, raw_text="下載長影片會卡住\n第二行")
    assert row.ticket.summary_line == "下載長影片會卡住"
    row_with_label = _row(snapshot=None, label="下載會卡住")
    assert row_with_label.ticket.summary_line == "下載會卡住"


def test_the_detail_says_so_when_nobody_has_replied() -> None:
    """Silence is stated rather than shown as an empty screen."""
    embed = build_detail_embed(detail=TicketDetail(row=_row(snapshot=None), comments=[]))
    assert embed.fields[0].name == "還沒有回覆"


async def test_the_open_report_cap_reads_what_the_panel_already_fetched() -> None:
    """The submission cap costs no extra request, because the panel just measured it."""
    closed = IssueSnapshot(number=1, state="closed", state_reason="completed", comment_count=0)
    open_issue = IssueSnapshot(number=2, state="open", state_reason=None, comment_count=0)
    view = FeedbackPanelView(
        host=cast("Any", object()),
        rows=[_row(snapshot=closed), _row(snapshot=open_issue), _row(snapshot=None)],
    )
    # The closed one is done and the unreadable one is not known to be open, so only the
    # middle row counts. Counting the unreadable one would mean a GitHub outage stops
    # people filing reports, which is the situation the local store exists to survive.
    assert view.outstanding_count() == 1


# --------------------------------------------------------------------------- submit


async def test_submitting_stores_the_report_and_opens_the_issue(
    feedback_isolated_db: None,
) -> None:
    """The happy path: stored, filed, and answered with the number."""
    issues = FakeIssues()
    cog = _cog(issues=issues)
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(
        interaction=cast("Any", interaction), text="下載長影片會卡住", outstanding=0
    )
    stored = await list_user_tickets(user_id=7, limit=10)
    assert len(stored) == 1
    assert stored[0].issue_number == 460
    assert stored[0].raw_text == "下載長影片會卡住"
    assert issues.created[0]["labels"] == ["user-report"]
    assert "下載長影片會卡住" in issues.created[0]["body"]
    embed = interaction.followup.sent[0]["embed"]
    assert "#460" in str(embed.title)


async def test_a_failed_issue_never_loses_the_report(feedback_isolated_db: None) -> None:
    """GitHub being down delays the number; it does not cost the report."""
    issues = FakeIssues(fail_create=True)
    cog = _cog(issues=issues)
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(interaction=cast("Any", interaction), text="壞掉了", outstanding=0)
    queued = await tickets_awaiting_issue(limit=10)
    assert len(queued) == 1
    assert queued[0].raw_text == "壞掉了"
    embed = interaction.followup.sent[0]["embed"]
    assert "存下來了" in str(embed.title)


async def test_the_retry_sweep_files_what_the_submit_path_could_not(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queued report is what makes writing locally first worth anything."""
    cog = _cog(issues=FakeIssues(fail_create=True))
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(interaction=cast("Any", interaction), text="壞掉了", outstanding=0)
    recovered = FakeIssues()
    cog.issues = cast("Any", recovered)
    # The sweep leaves very fresh rows to the submit that may still be filing them; this
    # one was filed a moment ago, so the age gate is dropped for the test.
    monkeypatch.setattr("discordbot.cogs.feedback.cog.RETRY_MIN_AGE_SECONDS", 0)
    await cog.retry_unfiled_reports()
    assert await tickets_awaiting_issue(limit=10) == []
    assert len(recovered.created) == 1


async def test_a_report_is_refused_once_too_many_are_open(feedback_isolated_db: None) -> None:
    """The cap is enforced before anything is written or filed."""
    issues = FakeIssues()
    cog = _cog(issues=issues)
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(interaction=cast("Any", interaction), text="第四張單", outstanding=3)
    assert await list_user_tickets(user_id=7, limit=10) == []
    assert issues.created == []
    assert "還沒處理完" in str(interaction.followup.sent[0]["embed"].description)


async def test_a_second_report_within_the_cooldown_is_refused(feedback_isolated_db: None) -> None:
    """Back-to-back submissions are throttled even when nothing is open."""
    issues = FakeIssues()
    cog = _cog(issues=issues)
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(interaction=cast("Any", interaction), text="第一張", outstanding=0)
    second = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(interaction=cast("Any", second), text="第二張", outstanding=0)
    assert len(await list_user_tickets(user_id=7, limit=10)) == 1
    assert "再送下一張" in str(second.followup.sent[0]["embed"].description)


async def test_the_cooldown_lets_a_later_report_through(feedback_isolated_db: None) -> None:
    """The throttle is a gap, not a lock: the next report goes through afterwards."""
    issues = FakeIssues()
    cog = _cog(issues=issues)
    cog.config = _config(FEEDBACK_SUBMIT_COOLDOWN_SECONDS=0)
    for text in ("第一張", "第二張"):
        interaction = FakeInteraction(user=FakeUser(user_id=7))
        await cog.submit_report(interaction=cast("Any", interaction), text=text, outstanding=0)
    assert len(await list_user_tickets(user_id=7, limit=10)) == 2


# ---------------------------------------------------------------------------- reply


async def test_a_reply_is_relayed_and_marked_as_the_reporters(feedback_isolated_db: None) -> None:
    """A relayed line is tagged so it never comes back as the developer's answer."""
    issues = FakeIssues()
    cog = _cog(issues=issues)
    ticket = await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="x",
    )
    await attach_issue_number(ticket_id=ticket.ticket_id, issue_number=460)
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_reply(
        interaction=cast("Any", interaction), ticket_id=ticket.ticket_id, text="補充一下"
    )
    assert issues.comments[0]["number"] == 460
    assert issues.comments[0]["body"].startswith(REPORTER_COMMENT_MARKER)
    assert "補充一下" in issues.comments[0]["body"]
    stored = await get_ticket(ticket_id=ticket.ticket_id)
    assert stored is not None
    assert stored.relayed_replies == 1


async def test_nobody_can_reply_on_someone_elses_report(feedback_isolated_db: None) -> None:
    """The panel is private, and the write is checked anyway before it reaches GitHub."""
    issues = FakeIssues()
    cog = _cog(issues=issues)
    ticket = await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="x",
    )
    await attach_issue_number(ticket_id=ticket.ticket_id, issue_number=460)
    interaction = FakeInteraction(user=FakeUser(user_id=8))
    await cog.submit_reply(
        interaction=cast("Any", interaction), ticket_id=ticket.ticket_id, text="我來亂"
    )
    assert issues.comments == []


# -------------------------------------------------------------------------- write-up


async def test_the_write_up_rewrites_the_issue_and_stores_the_draft(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful write-up replaces the raw issue text and labels it."""
    issues = FakeIssues()
    cog = _cog(issues=issues)
    ticket = await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="下載長影片會卡住",
    )
    await attach_issue_number(ticket_id=ticket.ticket_id, issue_number=460)
    ticket = ticket.model_copy(update={"issue_number": 460})

    async def _fake_write_up(
        *, client: AsyncOpenAI, model: ModelSettings, ticket: FeedbackTicket
    ) -> ReportWriteUp:
        return ReportWriteUp(
            label="下載會卡住",
            category="bug",
            title="fix: download stalls midway",
            body="Downloading a long video stalls.",
        )

    monkeypatch.setattr("discordbot.cogs.feedback.cog.write_up_report", _fake_write_up)
    await cog._apply_write_up(ticket=ticket)
    assert issues.updated[0]["title"] == "fix: download stalls midway"
    assert "下載長影片會卡住" in issues.updated[0]["body"]
    assert issues.labelled[0]["labels"] == ["user-report", "bug"]
    stored = await get_ticket(ticket_id=ticket.ticket_id)
    assert stored is not None
    assert stored.label == "下載會卡住"


async def test_a_failed_write_up_leaves_the_issue_as_the_reporter_wrote_it(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model failing is an ordinary outcome with nothing to clean up."""
    issues = FakeIssues()
    cog = _cog(issues=issues)
    ticket = _ticket()

    async def _no_write_up(
        *, client: AsyncOpenAI, model: ModelSettings, ticket: FeedbackTicket
    ) -> None:
        return None

    monkeypatch.setattr("discordbot.cogs.feedback.cog.write_up_report", _no_write_up)
    await cog._apply_write_up(ticket=ticket)
    assert issues.updated == []
    assert issues.labelled == []


async def test_a_raising_write_up_never_escapes_the_background_task(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing in the background half may surface as an unhandled task exception."""
    cog = _cog(issues=FakeIssues())

    async def _boom(
        *, client: AsyncOpenAI, model: ModelSettings, ticket: FeedbackTicket
    ) -> ReportWriteUp:
        raise RuntimeError("proxy exploded")

    monkeypatch.setattr("discordbot.cogs.feedback.cog.write_up_report", _boom)
    await cog._apply_write_up(ticket=_ticket())


# ----------------------------------------------------------------------- the command


async def test_the_command_offers_a_contact_when_the_feature_is_switched_off() -> None:
    """The kill-switch names someone to talk to instead of a setting."""
    cog = _cog(issues=FakeIssues())
    cog.config = _config(FEEDBACK_ENABLED=False, FEEDBACK_CONTACT="mai9999")
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.feedback(cast("Any", interaction))
    description = str(interaction.response.sent[0]["embed"].description)
    assert "mai9999" in description
    assert interaction.response.deferred is False


async def test_the_command_shows_the_panel(feedback_isolated_db: None) -> None:
    """The panel replaces the deferred placeholder, with the caller's own reports on it."""
    issues = FakeIssues()
    cog = _cog(issues=issues)
    await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="下載長影片會卡住",
    )
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.feedback(cast("Any", interaction))
    assert interaction.response.deferred_ephemeral is True
    embed = interaction.edits[0]["embed"]
    assert embed.fields[0].value is not None
    assert "下載長影片會卡住" in embed.fields[0].value


def test_github_is_not_ready_without_both_halves() -> None:
    """A token with no repository, or a repository with no token, opens nothing."""
    assert not _config(FEEDBACK_GITHUB_REPOSITORY="").github_ready
    assert not _config(FEEDBACK_GITHUB_TOKEN="").github_ready
    assert not _config(FEEDBACK_ENABLED=False).github_ready
    assert _config().github_ready
    # A slug missing its owner is not a repository, however non-empty it looks.
    assert not _config(FEEDBACK_GITHUB_REPOSITORY="discordbot").github_ready


# ------------------------------------------------------------------------ REST logic


class _ScriptedIssues(GitHubIssues):
    """A GitHub client whose transport is scripted, so the REST logic runs without a network.

    Attributes:
        calls: Every request the client attempted, in order.
        fail_status: Status the scripted transport answers with, 0 to always succeed.
        fail_once: Whether only the first attempt fails.
    """

    calls: list[dict[str, Any]] = Field(default_factory=list)
    fail_status: int = 0
    fail_once: bool = True

    async def _request(
        self, *, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:  # noqa: ANN401 -- mirrors the signature it overrides
        """Records the attempt and answers from the script."""
        self.calls.append({"method": method, "path": path, "payload": payload})
        if self.fail_status and (not self.fail_once or len(self.calls) == 1):
            raise GitHubIssuesError(
                f"{method} {path} answered {self.fail_status}: nope", self.fail_status
            )
        if method == "POST" and path == "/issues":
            return {"number": 460}
        if method == "GET" and path.endswith("/comments"):
            return []
        if method == "GET":
            return {
                "number": 460,
                "title": "t",
                "state": "closed",
                "state_reason": "not_planned",
                "comments": 3,
            }
        return {}


async def test_a_missing_label_never_costs_the_report() -> None:
    """A label the repository does not carry must not take the whole issue down with it."""
    client = _scripted(fail_status=422)
    number = await client.create_issue(title="t", body="b", labels=["user-report"])
    assert number == 460
    assert len(client.calls) == 2
    assert "labels" not in client.calls[1]["payload"]


async def test_a_real_create_failure_is_not_swallowed() -> None:
    """Anything other than a rejected label is the caller's problem to handle."""
    client = _scripted(fail_status=503)
    with pytest.raises(GitHubIssuesError):
        await client.create_issue(title="t", body="b", labels=["user-report"])
    assert len(client.calls) == 1


async def test_the_snapshot_carries_why_an_issue_was_closed() -> None:
    """A closed issue cannot tell finished from not-doing-it, so the reason rides along."""
    client = _scripted()
    snapshot = await client.read_issue(number=460)
    assert snapshot.state == "closed"
    assert snapshot.state_reason == "not_planned"
    assert snapshot.comment_count == 3


def _scripted(*, fail_status: int = 0) -> "_ScriptedIssues":
    """Builds a scripted client authorized by a placeholder token."""
    return _ScriptedIssues(
        credentials=TokenCredentials(token=_FAKE_TOKEN),
        repository="owner/name",
        fail_status=fail_status,
    )


class _PagingIssues(GitHubIssues):
    """A client whose scripted transport answers a conversation longer than one page.

    Scripted after the endpoint rather than after what the reader wants from it: the
    comments come back oldest first however they are asked for, and how deep the thread
    goes is only in the `Link` header, whose `rel="last"` entry sits behind a `rel="next"`
    pointing at a different page and is absent on the last page altogether. All of it was
    measured against api.github.com, and each part earns its place — a fake answering the
    newest first would let the bug this file pins go unnoticed, and one emitting a lone
    `rel="last"` would never exercise the entry the header parser has to pick it out of.

    Attributes:
        total: How many comments the scripted thread holds.
        requested: The pages the reader asked for, in order.
    """

    total: int = Field(default=120, description="How many comments the scripted thread holds.")
    requested: list[int] = Field(
        default_factory=list, description="The pages the reader asked for, in order."
    )

    async def _send(
        self, *, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> httpx.Response:
        """Answers one page of the thread, with the header that says how many there are."""
        page = int(path.rsplit("page=", maxsplit=1)[1])
        self.requested.append(page)
        last_page = max(1, -(-self.total // _COMMENTS_PER_PAGE))
        start = (page - 1) * _COMMENTS_PER_PAGE
        body = [
            {
                "user": {"login": "mai", "type": "User"},
                "author_association": "OWNER",
                "body": f"第 {index} 則",
                "created_at": "2026-02-04T10:00:00Z",
            }
            for index in range(start, min(start + _COMMENTS_PER_PAGE, self.total))
        ]
        entries = (
            ([(page - 1, "prev")] if page > 1 else [])
            + ([(page + 1, "next"), (last_page, "last")] if page < last_page else [])
            + ([(1, "first")] if page > 1 else [])
        )
        link = ", ".join(
            f"<https://api.github.com/repositories/1/issues/460/comments"
            f'?per_page={_COMMENTS_PER_PAGE}&page={target}>; rel="{relation}"'
            for target, relation in entries
        )
        return httpx.Response(status_code=200, json=body, headers={"Link": link} if link else {})


def _paging(*, total: int) -> _PagingIssues:
    """Builds a paging client over a scripted thread of `total` comments."""
    return _PagingIssues(
        credentials=TokenCredentials(token=_FAKE_TOKEN), repository="owner/name", total=total
    )


# ---------------------------------------------------------------------- view wiring


class _FakeHost:
    """Records what the views ask the cog to do."""

    def __init__(self, *, detail: TicketDetail | None) -> None:
        """Initializes the host with the detail its lookups will return."""
        self.detail = detail
        self.rows: list[TicketRow] = []
        self.viewed: list[dict[str, Any]] = []
        self.submitted: list[dict[str, Any]] = []
        self.replied: list[dict[str, Any]] = []

    async def load_rows(self, *, user_id: int) -> PanelRows:
        """Returns the prepared rows."""
        return PanelRows(rows=self.rows, total=len(self.rows))

    async def load_detail(self, *, ticket_id: int, viewer_id: int) -> TicketDetail | None:
        """Returns the prepared detail."""
        self.viewed.append({"ticket_id": ticket_id, "viewer_id": viewer_id})
        return self.detail

    async def submit_report(
        self, *, interaction: Interaction[commands.Bot], text: str, outstanding: int
    ) -> None:
        """Records a submitted report."""
        self.submitted.append({"text": text, "outstanding": outstanding})

    async def submit_reply(
        self, *, interaction: Interaction[commands.Bot], ticket_id: int, text: str
    ) -> None:
        """Records a relayed reply."""
        self.replied.append({"ticket_id": ticket_id, "text": text})


async def test_picking_a_report_opens_it_in_place() -> None:
    """The panel and the detail share one ephemeral message, so the picker edits it."""
    detail = TicketDetail(row=_row(snapshot=None), comments=[])
    host = _FakeHost(detail=detail)
    view = FeedbackPanelView(host=cast("Any", host), rows=[_row(snapshot=None)])
    view.stop()
    select = next(child for child in view.children if isinstance(child, StringSelect))
    select._selected_values = ["1"]
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await select.callback(cast("Any", interaction))
    # Deferred, then edited: reading the report is two GitHub calls, well past Discord's
    # three-second window for a first response.
    assert interaction.response.deferred is True
    assert "#460" in str(interaction.edits[0]["embed"].title)
    assert host.viewed[0]["viewer_id"] == 7


async def test_the_report_form_carries_the_open_count_it_was_opened_with() -> None:
    """The cap is measured against the list the person was looking at."""
    host = _FakeHost(detail=None)
    open_issue = IssueSnapshot(number=460, state="open", state_reason=None, comment_count=0)
    view = FeedbackPanelView(host=cast("Any", host), rows=[_row(snapshot=open_issue)])
    view.stop()
    button = next(child for child in view.children if isinstance(child, Button))
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await button.callback(cast("Any", interaction))
    modal = interaction.response.modals[0]
    assert isinstance(modal, ReportModal)
    assert modal.outstanding == 1


async def test_a_report_without_a_number_cannot_be_replied_to_yet() -> None:
    """There is nowhere to put a reply until the issue exists, so the form stays shut."""
    detail = TicketDetail(row=_row(snapshot=None, issue_number=None), comments=[])
    host = _FakeHost(detail=detail)
    view = TicketDetailView(host=cast("Any", host), detail=detail)
    reply_button = next(
        child
        for child in view.children
        if isinstance(child, Button) and "回一句" in str(child.label)
    )
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await reply_button.callback(cast("Any", interaction))
    assert interaction.response.modals == []
    assert "建立中" in str(interaction.response.sent[0]["embed"].description)


# ------------------------------------------------------- what the review pass found


def test_a_long_report_cannot_break_the_panel() -> None:
    """A 2000-character report used to build a field Discord rejects, locking the panel.

    The form takes 2000 characters and the write-up may never land, so the panel's own
    summary line is the reporter's raw text. Discord caps a field value at 1024 and the
    whole message at 6000, and it answers a violation with a 400 that would leave this
    person unable to open their reports at all, every time, permanently.
    """
    rows = [_row(snapshot=None, ticket_id=index, raw_text="壞" * 2000) for index in range(10)]
    embed = build_panel_embed(rows=rows, total=len(rows))
    assert all(len(str(field.value)) <= 1024 for field in embed.fields)
    assert len(embed) < 6000


def test_a_long_conversation_cannot_break_the_detail_screen() -> None:
    """Same budget, the other screen: several long maintainer replies used to overflow it."""
    comments = [
        IssueComment(
            author="mai", body="說" * 2000, created_at="2026-02-04T10:00:00Z", from_reporter=False
        )
        for _ in range(8)
    ]
    embed = build_detail_embed(
        detail=TicketDetail(row=_row(snapshot=None, raw_text="壞" * 2000), comments=comments)
    )
    assert all(len(str(field.value)) <= 1024 for field in embed.fields)
    assert len(embed) < 6000


def test_what_is_cut_off_is_said_rather_than_silently_dropped() -> None:
    """Truncation is visible: a report has to be recognisable to the person who wrote it."""
    embed = build_panel_embed(rows=[_row(snapshot=None, raw_text="壞" * 2000)], total=25)
    assert "…" in str(embed.fields[0].value)
    assert "24" in str(embed.footer.text)
    comments = [
        IssueComment(
            author="mai",
            body=f"第 {index} 則",
            created_at="2026-02-04T10:00:00Z",
            from_reporter=False,
        )
        for index in range(9)
    ]
    detail = build_detail_embed(detail=TicketDetail(row=_row(snapshot=None), comments=comments))
    assert any("比較早的對話" in str(field.name) for field in detail.fields)


def test_unreadable_replies_are_not_reported_as_no_replies() -> None:
    """The screen exists to answer "has anyone replied", so it must not answer it wrongly.

    A failed comment read used to collapse into an empty list, which rendered as the
    developer not having answered — while the panel next to it said someone had.
    """
    embed = build_detail_embed(detail=TicketDetail(row=_row(snapshot=None), comments=None))
    assert embed.fields[0].name == "現在讀不到回覆"
    assert not any("還沒有回覆" in str(field.name) for field in embed.fields)


async def test_an_issue_number_is_recorded_once(feedback_isolated_db: None) -> None:
    """Two writers race for one row, and the number the reporter was given has to win."""
    ticket = await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="x",
    )
    assert await attach_issue_number(ticket_id=ticket.ticket_id, issue_number=460) is True
    assert await attach_issue_number(ticket_id=ticket.ticket_id, issue_number=461) is False
    stored = await get_ticket(ticket_id=ticket.ticket_id)
    assert stored is not None
    assert stored.issue_number == 460


async def test_the_sweep_leaves_a_submit_in_flight_alone(feedback_isolated_db: None) -> None:
    """A row mid-submit looks exactly like a failed one, and filing it twice is the cost."""
    await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="x",
    )
    assert await tickets_awaiting_issue(limit=10, min_age_seconds=120) == []
    assert len(await tickets_awaiting_issue(limit=10, min_age_seconds=0)) == 1


async def test_the_sweep_survives_a_failure_and_runs_again(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception escaping the loop would stop it for the life of the process.

    nextcord only retries a short list of connection errors; anything else ends the task
    and nothing restarts it, so every queued report would sit there until the next deploy.
    """
    cog = _cog(issues=FakeIssues(fail_create=True))
    monkeypatch.setattr("discordbot.cogs.feedback.cog.write_up_report", _tidy_write_up)
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(interaction=cast("Any", interaction), text="壞掉了", outstanding=0)
    await _drain(cog=cog)

    async def _boom(**_kwargs: object) -> list[FeedbackTicket]:
        raise RuntimeError("database is locked")

    monkeypatch.setattr("discordbot.cogs.feedback.cog.tickets_awaiting_issue", _boom)
    await cog.retry_unfiled_reports()

    # The failure cost this report one cycle, not the loop: the next pass still files it.
    monkeypatch.setattr(
        "discordbot.cogs.feedback.cog.tickets_awaiting_issue", tickets_awaiting_issue
    )
    monkeypatch.setattr("discordbot.cogs.feedback.cog.RETRY_MIN_AGE_SECONDS", 0)
    recovered = FakeIssues()
    cog.issues = cast("Any", recovered)
    await cog.retry_unfiled_reports()
    assert len(recovered.created) == 1
    assert await tickets_awaiting_issue(limit=10, min_age_seconds=0) == []


async def test_a_report_that_cannot_be_stored_says_so(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one failure that loses a report outright must not be a silent spinner."""
    cog = _cog(issues=FakeIssues())

    async def _boom(**_kwargs: object) -> FeedbackTicket:
        raise RuntimeError("disk full")

    monkeypatch.setattr("discordbot.cogs.feedback.cog.create_ticket", _boom)
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(interaction=cast("Any", interaction), text="壞掉了", outstanding=0)
    assert "沒有存下來" in str(interaction.followup.sent[0]["embed"].description)


async def test_nobody_can_read_someone_elses_report(feedback_isolated_db: None) -> None:
    """The panel is private, and the read path is where the other person's words are."""
    cog = _cog(issues=FakeIssues())
    ticket = await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="私密的內容",
    )
    assert await cog.load_detail(ticket_id=ticket.ticket_id, viewer_id=8) is None
    assert await cog.load_detail(ticket_id=ticket.ticket_id, viewer_id=7) is not None


async def test_every_comment_page_is_read() -> None:
    """The newest reply is the one being looked for, and it is on the last page."""
    client = _paging(total=120)
    conversation = await client.read_conversation(number=460)
    assert len(conversation) == 120
    assert conversation[-1].body == "第 119 則"
    assert client.requested == [1, 2]


async def test_the_comment_bound_drops_the_oldest_end_and_not_the_newest() -> None:
    """A thread past the bound has to lose something, and it must not be what is shown.

    The endpoint answers oldest first and ignores `direction`, so a bound counted from
    page 1 keeps the opening of the thread and drops the reply the reporter came back to
    read — while the status line beside it, derived from the issue's own comment count,
    still says someone answered.
    """
    client = _paging(total=450)
    conversation = await client.read_conversation(number=460)
    assert conversation[-1].body == "第 449 則"
    assert conversation[0].body == "第 200 則"
    assert client.requested == [1, 3, 4, 5]


async def test_the_panel_shows_the_newest_end_of_what_was_read(feedback_isolated_db: None) -> None:
    """Which end the screen takes is why the read has to keep that end, so it is pinned.

    This is the panel half alone: the conversation is handed to the cog rather than
    fetched, so nothing here can go red over the page bound. What it fixes in place is
    the premise that bound is chosen for — `build_detail_embed` renders the LAST few
    comments, so a reader that dropped the newest ones would be dropping the screen.
    """
    issues = FakeIssues()
    issues.conversation = [
        IssueComment(
            author="mai",
            body=f"第 {index} 則",
            created_at="2026-02-04T10:00:00Z",
            from_reporter=False,
        )
        for index in range(9)
    ]
    ticket = await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="壞掉了",
    )
    await attach_issue_number(ticket_id=ticket.ticket_id, issue_number=460)
    detail = await _cog(issues=issues).load_detail(ticket_id=ticket.ticket_id, viewer_id=7)
    assert detail is not None
    assert detail.comments == issues.conversation
    embed = build_detail_embed(detail=detail)
    values = [str(field.value) for field in embed.fields]
    assert "第 8 則" in values
    assert "第 3 則" not in values
    assert "上面還有 4 則比較早的對話" in [str(field.name) for field in embed.fields]


# ------------------------------------------------------------- before a token exists


async def test_a_report_is_taken_before_any_token_is_configured(
    feedback_isolated_db: None,
) -> None:
    """A deployment mid-setup still collects reports; they wait rather than being refused.

    The token is an operational state, not a switch. Refusing here would throw away the
    reports filed between the bot going live and the credentials landing, and those are
    exactly the ones a new deployment gets most of.
    """
    issues = FakeIssues()
    cog = _cog(issues=issues)
    cog.config = _config(FEEDBACK_GITHUB_TOKEN="")
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(interaction=cast("Any", interaction), text="壞掉了", outstanding=0)
    stored = await list_user_tickets(user_id=7, limit=10)
    assert len(stored) == 1
    assert stored[0].issue_number is None
    assert stored[0].raw_text == "壞掉了"
    # Nothing was sent anywhere: there is nowhere to send it to yet.
    assert issues.created == []
    assert "存下來了" in str(interaction.followup.sent[0]["embed"].title)


async def test_the_panel_opens_before_any_token_is_configured(feedback_isolated_db: None) -> None:
    """The queued reports are still the reporter's own list, and still readable."""
    cog = _cog(issues=FakeIssues())
    cog.config = _config(FEEDBACK_GITHUB_TOKEN="")
    await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=None,
        guild_name="",
        channel_id=None,
        locale="",
        raw_text="下載長影片會卡住",
    )
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.feedback(cast("Any", interaction))
    embed = interaction.edits[0]["embed"]
    assert "建立中" in str(embed.fields[0].name)


async def test_the_queue_drains_once_a_token_arrives(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything filed during the setup gap is opened by the first sweep afterwards."""
    cog = _cog(issues=FakeIssues())
    cog.config = _config(FEEDBACK_GITHUB_TOKEN="")
    for text in ("第一張", "第二張"):
        interaction = FakeInteraction(user=FakeUser(user_id=7))
        await cog.submit_report(interaction=cast("Any", interaction), text=text, outstanding=0)
        monkeypatch.setattr("discordbot.cogs.feedback.cog.seconds_since_last_ticket", _no_cooldown)
    assert len(await tickets_awaiting_issue(limit=10, min_age_seconds=0)) == 2

    configured = FakeIssues()
    cog.issues = cast("Any", configured)
    cog.config = _config()
    monkeypatch.setattr("discordbot.cogs.feedback.cog.RETRY_MIN_AGE_SECONDS", 0)
    await cog.retry_unfiled_reports()
    assert len(configured.created) == 2
    assert await tickets_awaiting_issue(limit=10, min_age_seconds=0) == []


async def test_the_sweep_waits_while_the_token_is_still_missing(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unconfigured sweep leaves the queue alone rather than burning attempts on it."""
    issues = FakeIssues()
    cog = _cog(issues=issues)
    cog.config = _config(FEEDBACK_GITHUB_TOKEN="")
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(interaction=cast("Any", interaction), text="壞掉了", outstanding=0)
    monkeypatch.setattr("discordbot.cogs.feedback.cog.RETRY_MIN_AGE_SECONDS", 0)
    await cog.retry_unfiled_reports()
    assert issues.created == []
    assert len(await tickets_awaiting_issue(limit=10, min_age_seconds=0)) == 1


async def test_a_queued_report_is_written_up_without_waiting_for_a_token(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tidy-up needs nothing from GitHub, so it must not wait for it.

    Half of what it produces is local: the line the reporter's own panel shows, and the
    classification. Holding it back until a token appears would leave the panel showing
    raw first lines for as long as the setup takes, which is the whole benefit of storing
    the report locally in the first place.
    """
    cog = _cog(issues=FakeIssues())
    cog.config = _config(FEEDBACK_GITHUB_TOKEN="")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.write_up_report", _tidy_write_up)
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(
        interaction=cast("Any", interaction), text="下載長影片會卡住", outstanding=0
    )
    await _drain(cog=cog)
    stored = (await list_user_tickets(user_id=7, limit=10))[0]
    assert stored.issue_number is None
    assert stored.label == "下載會卡住"
    assert stored.draft_title == "fix: download stalls midway"
    assert stored.category == "bug"


async def test_a_queued_issue_opens_from_the_draft_instead_of_raw_text(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep opens the issue in its finished form, with no follow-up edit."""
    cog = _cog(issues=FakeIssues())
    cog.config = _config(FEEDBACK_GITHUB_TOKEN="")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.write_up_report", _tidy_write_up)
    interaction = FakeInteraction(user=FakeUser(user_id=7))
    await cog.submit_report(
        interaction=cast("Any", interaction), text="下載長影片會卡住", outstanding=0
    )
    await _drain(cog=cog)

    configured = FakeIssues()
    cog.issues = cast("Any", configured)
    cog.config = _config()
    monkeypatch.setattr("discordbot.cogs.feedback.cog.RETRY_MIN_AGE_SECONDS", 0)
    await cog.retry_unfiled_reports()
    await _drain(cog=cog)
    assert configured.created[0]["title"] == "fix: download stalls midway"
    assert configured.created[0]["labels"] == ["user-report", "bug"]
    assert "下載長影片會卡住" in configured.created[0]["body"]
    # Nothing to rewrite: it was opened finished, so the model is not called a second time.
    assert configured.updated == []


def test_a_half_written_draft_is_not_used() -> None:
    """A row with only one half of the draft is not a draft."""
    assert stored_draft(ticket=_ticket()) is None
    assert stored_draft(ticket=_ticket(draft_title="t")) is None
    assert stored_draft(ticket=_ticket(draft_title="t", draft_body="b")) == StoredDraft(
        title="t", body="b"
    )


# ------------------------------------------------------------------ how it authorizes


def test_a_token_authorizes_directly() -> None:
    """The account path has nothing to exchange; the token is the credential."""
    assert (
        asyncio.run(TokenCredentials(token=_FAKE_TOKEN).authorization()) == f"Bearer {_FAKE_TOKEN}"
    )


def test_a_missing_private_key_is_reported_as_an_auth_failure(tmp_path: Path) -> None:
    """Every credential failure arrives as one type, so callers catch one thing."""
    credentials = AppCredentials(
        app_id="Iv23li", private_key_path=tmp_path / "absent.pem", repository="owner/name"
    )
    with pytest.raises(GitHubAuthError):
        asyncio.run(credentials.authorization())


def test_an_unusable_private_key_is_reported_as_an_auth_failure(tmp_path: Path) -> None:
    """A file that is not a key fails at signing rather than somewhere further in."""
    key = tmp_path / "not-a-key.pem"
    key.write_text("hello", encoding="utf-8")
    credentials = AppCredentials(app_id="Iv23li", private_key_path=key, repository="owner/name")
    with pytest.raises(GitHubAuthError):
        asyncio.run(credentials.authorization())


async def test_an_installation_token_is_reused_until_it_nearly_expires(tmp_path: Path) -> None:
    """One exchange per hour, not one per request: the panel makes several calls a time."""
    minted: list[str] = []

    class _Exchanging(AppCredentials):
        """An app credential whose GitHub side is scripted."""

        def _signed_jwt(self) -> str:
            return "jwt"

        async def _call(self, *, method: str, path: str, authorization: str) -> Any:  # noqa: ANN401 -- mirrors the signature it overrides
            minted.append(path)
            if path.endswith("/installation"):
                return {"id": 42}
            return {
                "token": "ghs_installation",
                "expires_at": (datetime.now(tz=UTC) + timedelta(hours=1)).isoformat(),
            }

    credentials = _Exchanging(
        app_id="Iv23li", private_key_path=tmp_path / "key.pem", repository="owner/name"
    )
    assert await credentials.authorization() == "Bearer ghs_installation"
    assert await credentials.authorization() == "Bearer ghs_installation"
    # Two calls the first time (find the installation, mint a token), nothing the second; a
    # token cannot be asked for before the installation it belongs to is known.
    # order-contract: one coroutine awaits the lookup and only then mints.
    assert minted == ["/repos/owner/name/installation", "/app/installations/42/access_tokens"]


async def test_an_expired_installation_token_is_replaced(tmp_path: Path) -> None:
    """A token near its end is exchanged again rather than sent and refused."""
    exchanges = 0

    class _Expiring(AppCredentials):
        """An app credential whose tokens are already at their expiry."""

        def _signed_jwt(self) -> str:
            return "jwt"

        async def _call(self, *, method: str, path: str, authorization: str) -> Any:  # noqa: ANN401 -- mirrors the signature it overrides
            nonlocal exchanges
            if path.endswith("/installation"):
                return {"id": 42}
            exchanges += 1
            return {"token": f"ghs_{exchanges}", "expires_at": datetime.now(tz=UTC).isoformat()}

    credentials = _Expiring(
        app_id="Iv23li", private_key_path=tmp_path / "key.pem", repository="owner/name"
    )
    assert await credentials.authorization() == "Bearer ghs_1"
    assert await credentials.authorization() == "Bearer ghs_2"


def test_the_app_is_preferred_over_a_token() -> None:
    """Both configured means the app wins: it files under its own name and rotates itself."""
    cog = _cog(issues=FakeIssues())
    cog.config = _config(
        FEEDBACK_GITHUB_APP_ID="Iv23li", FEEDBACK_GITHUB_APP_PRIVATE_KEY_PATH="feedback-app.pem"
    )
    del cog.__dict__["issues"]
    assert isinstance(cog.issues.credentials, AppCredentials)


def test_an_app_alone_is_enough_to_be_ready() -> None:
    """The app is a credential in its own right; no token needs to be set beside it."""
    assert _config(
        FEEDBACK_GITHUB_TOKEN="",
        FEEDBACK_GITHUB_APP_ID="Iv23li",
        FEEDBACK_GITHUB_APP_PRIVATE_KEY_PATH="feedback-app.pem",
    ).github_ready
    # Half of an app is not an app.
    assert not _config(FEEDBACK_GITHUB_TOKEN="", FEEDBACK_GITHUB_APP_ID="Iv23li").github_ready


def test_a_relayed_reply_survives_being_posted_by_an_app() -> None:
    """An app comments as a Bot, and the reporter's own words must not be filtered as noise.

    This is the one thing switching credentials would have broken silently: the bot
    filter used to run before the marker check, so every line a reporter added through
    the panel would have vanished from their own view of the report.
    """
    conversation = select_conversation(
        comments=[
            {
                "user": {"login": "pocat-feedback-bot[bot]", "type": "Bot"},
                "author_association": "NONE",
                "body": f"{REPORTER_COMMENT_MARKER}\n**Alice** wrote from Discord:\n\n補充一下",
                "created_at": "2026-02-05T09:00:00Z",
            },
            {
                "user": {"login": "dependabot[bot]", "type": "Bot"},
                "author_association": "NONE",
                "body": "Bumps httpx",
                "created_at": "2026-02-05T10:00:00Z",
            },
        ]
    )
    assert len(conversation) == 1
    assert conversation[0].from_reporter is True
    assert "補充一下" in conversation[0].body


def test_a_stranger_cannot_forge_a_reporter_reply() -> None:
    """The marker is plumbing, not authentication; who posted it is what decides."""
    conversation = select_conversation(
        comments=[
            {
                "user": {"login": "stranger", "type": "User"},
                "author_association": "NONE",
                "body": f"{REPORTER_COMMENT_MARKER}\n**Alice** wrote from Discord:\n\n我不是 Alice",
                "created_at": "2026-02-05T09:00:00Z",
            }
        ]
    )
    assert conversation == []


# ------------------------------------------------------------------- close notices


class _FakeReporter:
    """The person who filed a report, recording what was sent to them."""

    def __init__(self, *, refuse: bool = False) -> None:
        """Initializes a reporter whose DMs can be made to fail."""
        self.refuse = refuse
        self.embeds: list[Embed] = []

    async def send(self, *, embed: Embed) -> None:
        """Records one direct message, or refuses like a closed DM would."""
        if self.refuse:
            raise RuntimeError("Cannot send messages to this user")
        self.embeds.append(embed)


class _FakeBot:
    """Enough of a bot to look one reporter up."""

    def __init__(self, *, reporter: _FakeReporter | None) -> None:
        """Initializes a bot that finds `reporter`, or nobody at all."""
        self.reporter = reporter

    async def fetch_user(self, user_id: int, /) -> _FakeReporter:
        """Hands back the reporter, or fails the way an unknown id does."""
        if self.reporter is None:
            raise RuntimeError(f"Unknown user {user_id}")
        return self.reporter


def _notice_cog(*, issues: FakeIssues, reporter: _FakeReporter | None) -> FeedbackCogs:
    """Builds a cog that can reach exactly one reporter."""
    cog = _cog(issues=issues)
    cog.bot = cast("Any", _FakeBot(reporter=reporter))
    return cog


def _github_stamp(*, ago: timedelta = timedelta()) -> str:
    """A GitHub timestamp that far in the past, in the shape its API answers with.

    Relative to now on purpose. Both windows this feeds are measured against the real
    clock (`_CLOSING_WINDOW` from the close, `BACKFILL_CUTOFF` from today), so an absolute
    date here is a test that goes red on a calendar date with nobody having touched the
    code. What each case actually asserts is a distance between two moments, and that is
    what this expresses.
    """
    return (datetime.now(tz=UTC) - ago).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _filed_report(
    *, issues: FakeIssues, state_reason: str | None = None, closed_at: str | None = None
) -> FeedbackTicket:
    """Stores one report with an issue behind it, closed for the given reason."""
    ticket = await create_ticket(
        user_id=7,
        user_name="alice",
        display_name="Alice",
        guild_id=100,
        guild_name="test guild",
        channel_id=200,
        locale="zh-TW",
        raw_text="簽到功能不見了",
    )
    await attach_issue_number(ticket_id=ticket.ticket_id, issue_number=460)
    if state_reason is not None:
        issues.snapshots[460] = IssueSnapshot(
            number=460,
            state="closed",
            state_reason=state_reason,
            comment_count=1,
            closed_at=closed_at or _github_stamp(),
        )
    return ticket


def _maintainer_said(*, body: str, ago: timedelta = timedelta()) -> IssueComment:
    """One comment from the developer, written that long before now."""
    return IssueComment(
        author="maintainer", body=body, created_at=_github_stamp(ago=ago), from_reporter=False
    )


async def _translates_verbatim(
    *, client: AsyncOpenAI, model: ModelSettings, ticket: FeedbackTicket, comment: IssueComment
) -> str:
    """Stands in for the translation, handing the comment back unchanged."""
    return comment.body


async def _translation_fails(
    *, client: AsyncOpenAI, model: ModelSettings, ticket: FeedbackTicket, comment: IssueComment
) -> str | None:
    """Stands in for a translation that did not come back."""
    return None


async def test_an_open_report_is_never_notified(feedback_isolated_db: None) -> None:
    """Nothing is announced while the developer is still working on it."""
    issues = FakeIssues()
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(issues=issues)

    await cog.notify_closed_reports()

    assert reporter.embeds == []
    assert await close_notices_awaiting_delivery(min_age_seconds=0) == []


async def test_a_close_is_not_announced_on_the_pass_that_finds_it(
    feedback_isolated_db: None,
) -> None:
    """The wait between the passes is what lets a comment written after the close ride along."""
    issues = FakeIssues()
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(issues=issues, state_reason="completed")

    await cog.notify_closed_reports()

    assert reporter.embeds == []
    # Recorded and owed, just not yet due.
    assert len(await close_notices_awaiting_delivery(min_age_seconds=0)) == 1


async def test_the_second_pass_sends_the_close_and_marks_it_done(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The developer's own closing words reach the reporter, once."""
    issues = FakeIssues()
    issues.conversation = [_maintainer_said(body="簽到指令補回去了。")]
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(issues=issues, state_reason="completed")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.translate_comment", _translates_verbatim)

    await cog.notify_closed_reports()
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)
    await cog.notify_closed_reports()

    assert len(reporter.embeds) == 1
    assert "已完成" in str(reporter.embeds[0].title)
    assert "簽到指令補回去了。" in str(reporter.embeds[0].fields[0].value)

    # A third pass has nothing left to say, which is what "one per report" means.
    await cog.notify_closed_reports()
    assert len(reporter.embeds) == 1


async def test_a_reopened_and_reclosed_report_says_nothing_further(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One message per report, ever: reopening does not buy a second one."""
    issues = FakeIssues()
    issues.conversation = [_maintainer_said(body="Fixed.")]
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(issues=issues, state_reason="completed")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.translate_comment", _translates_verbatim)
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)
    await cog.notify_closed_reports()
    assert len(reporter.embeds) == 1

    issues.snapshots[460] = IssueSnapshot(
        number=460, state="open", state_reason="reopened", comment_count=2
    )
    await cog.notify_closed_reports()
    issues.snapshots[460] = IssueSnapshot(
        number=460, state="closed", state_reason="not_planned", comment_count=3
    )
    await cog.notify_closed_reports()

    assert len(reporter.embeds) == 1


async def test_a_duplicate_tells_nobody_but_is_still_finished_with(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A merged report has no outcome of its own yet, and must not be rediscovered forever."""
    issues = FakeIssues()
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(issues=issues, state_reason="duplicate")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)

    await cog.notify_closed_reports()
    await cog.notify_closed_reports()

    assert reporter.embeds == []
    assert await close_notices_awaiting_delivery(min_age_seconds=0) == []
    assert await tickets_awaiting_close_check() == []


async def test_a_failed_translation_leaves_the_notice_for_the_next_pass(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sending the English instead would defeat the whole point of translating it."""
    issues = FakeIssues()
    issues.conversation = [_maintainer_said(body="The command is back in this release.")]
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(issues=issues, state_reason="completed")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)
    monkeypatch.setattr("discordbot.cogs.feedback.cog.translate_comment", _translation_fails)

    await cog.notify_closed_reports()
    assert reporter.embeds == []
    assert len(await close_notices_awaiting_delivery(min_age_seconds=0)) == 1

    monkeypatch.setattr("discordbot.cogs.feedback.cog.translate_comment", _translates_verbatim)
    await cog.notify_closed_reports()
    assert len(reporter.embeds) == 1


async def test_a_close_with_no_comment_still_reaches_the_reporter(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal with no explanation still beats silence, and needs no model call."""
    issues = FakeIssues()
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(issues=issues, state_reason="not_planned")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)
    monkeypatch.setattr("discordbot.cogs.feedback.cog.translate_comment", _translation_fails)

    await cog.notify_closed_reports()

    assert len(reporter.embeds) == 1
    assert "不列入計劃" in str(reporter.embeds[0].title)
    assert "沒有另外留話" in str(reporter.embeds[0].description)


async def test_an_undeliverable_message_still_finishes_the_report(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closed DM does not reopen because we tried again; retrying only re-runs the model."""
    issues = FakeIssues()
    issues.conversation = [_maintainer_said(body="Fixed.")]
    cog = _notice_cog(issues=issues, reporter=_FakeReporter(refuse=True))
    await _filed_report(issues=issues, state_reason="completed")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)
    monkeypatch.setattr("discordbot.cogs.feedback.cog.translate_comment", _translates_verbatim)

    await cog.notify_closed_reports()

    assert await close_notices_awaiting_delivery(min_age_seconds=0) == []


async def test_an_unreadable_conversation_leaves_the_notice_for_the_next_pass(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad minute at GitHub must not cost the reporter their answer."""

    class _UnreadableIssues(FakeIssues):
        async def read_conversation(self, *, number: int) -> list[Any]:
            raise GitHubIssuesError(f"GET /issues/{number}/comments answered 503", 503)

    issues = _UnreadableIssues()
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(issues=issues, state_reason="completed")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)

    await cog.notify_closed_reports()

    assert reporter.embeds == []
    assert len(await close_notices_awaiting_delivery(min_age_seconds=0)) == 1


async def test_a_question_the_reporter_already_answered_is_not_the_verdict(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Being the developer's newest line does not make it the reason the report was closed.

    The shape this guards is common: the developer asks something, the reporter answers
    through the panel, and the report is closed later with nothing further said. Handing
    that question over as the verdict is worse than saying there was no comment.
    """
    issues = FakeIssues()
    issues.conversation = [
        _maintainer_said(body="Can you try again after restarting your client?"),
        IssueComment(
            author="alice", body="還是不行", created_at=_github_stamp(), from_reporter=True
        ),
    ]
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(issues=issues, state_reason="not_planned")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)

    await cog.notify_closed_reports()

    assert len(reporter.embeds) == 1
    assert "沒有另外留話" in str(reporter.embeds[0].description)
    assert reporter.embeds[0].fields == []


async def test_a_comment_from_long_before_the_close_is_not_the_verdict(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nobody spoke after it, but it still predates the close by months."""
    issues = FakeIssues()
    issues.conversation = [_maintainer_said(body="Taking a look.", ago=timedelta(days=90))]
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(issues=issues, state_reason="completed")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)
    monkeypatch.setattr("discordbot.cogs.feedback.cog.translate_comment", _translates_verbatim)

    await cog.notify_closed_reports()

    assert len(reporter.embeds) == 1
    assert reporter.embeds[0].fields == []


async def test_reports_closed_long_ago_are_not_announced_on_the_first_sweep(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every report ever closed is in this set the first time the sweep runs."""
    issues = FakeIssues()
    issues.conversation = [_maintainer_said(body="Fixed.")]
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    await _filed_report(
        issues=issues, state_reason="completed", closed_at=_github_stamp(ago=timedelta(days=400))
    )
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)
    monkeypatch.setattr("discordbot.cogs.feedback.cog.translate_comment", _translates_verbatim)

    await cog.notify_closed_reports()

    assert reporter.embeds == []
    # Marked rather than skipped, or it would be rediscovered on every later pass.
    assert await close_notices_awaiting_delivery(min_age_seconds=0) == []
    assert await tickets_awaiting_close_check() == []


@pytest.mark.parametrize(
    ("closed_at", "comment_at", "expects_comment"),
    [
        # GitHub's own shape, on both sides.
        ("2026-08-21T09:05:00Z", "2026-08-21T09:00:00Z", True),
        # A timestamp without a zone would be naive, and comparing it against an aware one
        # raises TypeError inside a sweep whose broad except would swallow it whole.
        ("2026-08-21T09:05:00", "2026-08-21T09:00:00Z", True),
        ("2026-08-21T09:05:00Z", "2026-08-21T09:00:00", True),
        # Unreadable: the ordering rule still stands on its own.
        ("not a timestamp", "2026-08-21T09:00:00Z", True),
        (None, "2026-08-21T09:00:00Z", True),
        # Readable on both sides, and months apart.
        ("2026-11-21T09:00:00Z", "2026-08-21T09:00:00Z", False),
    ],
)
def test_the_closing_comment_survives_every_timestamp_shape(
    closed_at: str | None, comment_at: str, expects_comment: bool
) -> None:
    """A timestamp nobody can parse must cost the comment, never the whole notice."""
    comment = IssueComment(
        author="maintainer", body="Fixed.", created_at=comment_at, from_reporter=False
    )
    found = closing_comment(comments=[comment], closed_at=closed_at)
    assert (found is not None) is expects_comment


async def test_a_report_reopened_during_the_wait_is_not_announced(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Announcing a close that was taken back would spend the one message it gets."""
    issues = FakeIssues()
    issues.conversation = [_maintainer_said(body="Fixed.")]
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    ticket = await _filed_report(issues=issues, state_reason="completed")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.translate_comment", _translates_verbatim)

    await cog.notify_closed_reports()
    issues.snapshots[460] = IssueSnapshot(
        number=460, state="open", state_reason="reopened", comment_count=1
    )
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)
    await cog.notify_closed_reports()

    assert reporter.embeds == []
    # Forgotten rather than marked, so closing it for real still reaches the reporter.
    assert [row.ticket_id for row in await tickets_awaiting_close_check()] == [ticket.ticket_id]

    issues.snapshots[460] = IssueSnapshot(
        number=460,
        state="closed",
        state_reason="completed",
        comment_count=1,
        closed_at=_github_stamp(),
    )
    await cog.notify_closed_reports()
    assert len(reporter.embeds) == 1


async def test_the_close_sweep_stops_when_the_feature_is_switched_off(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsolicited message must not send its reader to a command that will refuse them."""
    issues = FakeIssues()
    issues.conversation = [_maintainer_said(body="Fixed.")]
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    cog.config = _config(FEEDBACK_ENABLED=False)
    await _filed_report(issues=issues, state_reason="completed")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)
    monkeypatch.setattr("discordbot.cogs.feedback.cog.translate_comment", _translates_verbatim)

    await cog.notify_closed_reports()

    assert reporter.embeds == []
    # Owed rather than dropped: switching the feature back on still tells them.
    assert len(await tickets_awaiting_close_check()) == 1


async def test_the_close_sweep_waits_while_the_token_is_still_missing(
    feedback_isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no credential there is nothing to read, so nothing is decided about a report."""
    issues = FakeIssues()
    reporter = _FakeReporter()
    cog = _notice_cog(issues=issues, reporter=reporter)
    cog.config = _config(FEEDBACK_GITHUB_TOKEN="")
    await _filed_report(issues=issues, state_reason="completed")
    monkeypatch.setattr("discordbot.cogs.feedback.cog.CLOSE_NOTICE_MIN_AGE_SECONDS", 0)

    await cog.notify_closed_reports()

    assert reporter.embeds == []
    assert len(await tickets_awaiting_close_check()) == 1
