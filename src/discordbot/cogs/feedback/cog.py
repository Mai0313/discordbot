"""`/feedback`: a user's own reports, and the form that files a new one.

The submit path writes the local row **before** it talks to GitHub. After that first
write nothing outside this process can lose the report: a failed create leaves a row the
retry loop finishes later, and the reporter is told their report is stored even when the
number is not there yet.

The issue is opened from the reporter's raw words, and the LLM write-up rewrites it
afterwards in the background. Nobody is waiting on that call, so it takes as long as it
takes, and a failure needs no fallback text — the issue already reads as the reporter
wrote it.

There is no admin surface here on purpose. Replies are written on the issue, which is
also where the panel reads status from, so the developer keeps one inbox instead of two.
"""

from typing import Any
import asyncio
from functools import cached_property
from collections.abc import Coroutine

from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import Embed, Locale, Interaction
from nextcord.ext import tasks, commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.colors import NEUTRAL_BLUE, DISCORD_GREEN, DISCORD_YELLOW
from discordbot.typings.config import FeedbackConfig
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.timezone import as_taipei, database_now
from discordbot.cogs.feedback.views import (
    MAX_PANEL_TICKETS,
    PanelRows,
    TicketRow,
    TicketDetail,
    FeedbackPanelView,
    build_panel_embed,
    build_notice_embed,
    build_submitted_embed,
)
from discordbot.cogs.feedback.github import (
    REPORTER_COMMENT_MARKER,
    GitHubIssues,
    IssueComment,
    IssueSnapshot,
    GitHubIssuesError,
)
from discordbot.cogs.feedback.writeup import (
    write_up_report,
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
    count_user_tickets,
    attach_issue_number,
    count_relayed_reply,
    tickets_awaiting_issue,
    seconds_since_last_ticket,
)

# How often the sweep tries again for reports whose issue was never opened, and how many
# it takes per pass. Small: this only ever runs after a GitHub outage.
RETRY_INTERVAL_MINUTES = 10
RETRY_BATCH_SIZE = 10

# A submit that is still in flight has already opened its issue and has not recorded the
# number yet; anything younger than this is left to finish rather than filed twice.
RETRY_MIN_AGE_SECONDS = 120

# Past this, an unfiled report is not waiting out an outage any more. The queue is ordered,
# so one report GitHub will never accept holds up every report behind it.
RETRY_STALLED_AFTER_SECONDS = 24 * 60 * 60


def _locale_text(*, interaction: Interaction[commands.Bot]) -> str:
    """The reporter's Discord locale as a plain string, or empty when unknown."""
    locale = interaction.locale
    if isinstance(locale, Locale):
        return str(locale.value)
    return str(locale or "")


def _age_seconds(*, ticket: FeedbackTicket) -> float:
    """How long ago a report was filed, in seconds."""
    created = as_taipei(dt=ticket.created_at)
    return max((database_now() - created).total_seconds(), 0.0)


class FeedbackCogs(commands.Cog):
    """Provides the user-report panel and files reports as GitHub issues.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: Reporting settings, including whether reports can be filed at all.
        llm_config: Credentials for the background write-up.
        runtime_models: Catalog providing the write-up model tier.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the feedback cog.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = FeedbackConfig()
        self.llm_config = LLMConfig()
        self.runtime_models = RuntimeModelCatalog()
        self._started = False
        # Background work is held here so a task is not garbage collected mid-flight;
        # each one removes itself when it finishes.
        self._background: set[asyncio.Task[None]] = set()

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The cached AsyncOpenAI client used for the background write-up."""
        return AsyncOpenAI(base_url=self.llm_config.base_url, api_key=self.llm_config.api_key)

    @cached_property
    def issues(self) -> GitHubIssues:
        """The cached GitHub issues client for the configured repository."""
        return GitHubIssues(token=self.config.github_token, repository=self.config.repository_slug)

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Runs a coroutine in the background, keeping a reference until it finishes."""
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _read_snapshot(self, *, ticket: FeedbackTicket) -> IssueSnapshot | None:
        """Reads one report's issue, returning None when there is nothing to read.

        An unreadable issue is an ordinary outcome here (it has no number yet, or GitHub
        is having a bad minute); the panel says so rather than failing the whole list.
        """
        if ticket.issue_number is None:
            return None
        try:
            return await self.issues.read_issue(number=ticket.issue_number)
        except GitHubIssuesError as exc:
            logfire.warn(
                "Could not read a report's issue; showing it without a status",
                ticket_id=ticket.ticket_id,
                issue_number=ticket.issue_number,
                _exc_info=exc,
            )
            return None

    async def load_rows(self, *, user_id: int) -> PanelRows:
        """Returns one person's newest reports with their current issue state."""
        tickets = await list_user_tickets(user_id=user_id, limit=MAX_PANEL_TICKETS)
        total = await count_user_tickets(user_id=user_id)
        snapshots = await asyncio.gather(
            *(self._read_snapshot(ticket=ticket) for ticket in tickets)
        )
        return PanelRows(
            rows=[
                TicketRow(ticket=ticket, snapshot=snapshot)
                for ticket, snapshot in zip(tickets, snapshots, strict=True)
            ],
            total=total,
        )

    async def load_detail(self, *, ticket_id: int, viewer_id: int) -> TicketDetail | None:
        """Returns one report with its conversation, or None when the viewer may not see it.

        An unread conversation stays `None` rather than collapsing to an empty list: the
        screen it feeds exists to answer "has anyone replied", and an empty list there
        would be an answer we did not actually obtain.
        """
        ticket = await get_ticket(ticket_id=ticket_id)
        if ticket is None:
            return None
        if ticket.user_id != viewer_id:
            logfire.warn(
                "Refused to show a report to someone who did not file it",
                ticket_id=ticket_id,
                viewer_id=viewer_id,
            )
            return None
        snapshot = await self._read_snapshot(ticket=ticket)
        comments: list[IssueComment] | None = []
        if ticket.issue_number is not None:
            try:
                comments = await self.issues.read_conversation(number=ticket.issue_number)
            except GitHubIssuesError as exc:
                logfire.warn(
                    "Could not read a report's replies; the panel says so rather than guessing",
                    ticket_id=ticket.ticket_id,
                    issue_number=ticket.issue_number,
                    _exc_info=exc,
                )
                comments = None
        return TicketDetail(row=TicketRow(ticket=ticket, snapshot=snapshot), comments=comments)

    async def _open_issue(self, *, ticket: FeedbackTicket) -> int | None:
        """Opens the issue for a stored report, returning None when GitHub refuses.

        Returning None is safe precisely because the report is already stored: the sweep
        picks the row up again, and the reporter was told their words were kept.

        Failing to record the number afterwards is the one case that is not safe, because
        the row goes back into the sweep with an issue already open against it and gets a
        second one. It cannot be undone from here — the issue exists — so it is reported
        at `error` with the number in it, which is what a human needs to reconcile it.
        """
        try:
            number = await self.issues.create_issue(
                title=initial_issue_title(ticket=ticket),
                body=render_issue_body(ticket=ticket, write_up=None),
                labels=["user-report"],
            )
        except GitHubIssuesError as exc:
            logfire.warn(
                "Could not open the issue for a report; it stays queued",
                ticket_id=ticket.ticket_id,
                _exc_info=exc,
            )
            return None
        try:
            recorded = await attach_issue_number(ticket_id=ticket.ticket_id, issue_number=number)
        # Broad on purpose: every storage error means the same thing here — an issue
        # exists that this process can no longer connect to the report that caused it.
        except Exception as exc:
            logfire.error(
                "Opened an issue but could not record it against the report",
                ticket_id=ticket.ticket_id,
                issue_number=number,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        if not recorded:
            # Another pass got there first. Whichever number the row already carries is
            # the one the reporter was told, so this issue is the spare.
            logfire.warn(
                "A second issue was opened for a report that already had one",
                ticket_id=ticket.ticket_id,
                issue_number=number,
            )
            return None
        return number

    async def _apply_write_up(self, *, ticket: FeedbackTicket) -> None:
        """Writes the report up and rewrites its issue, best-effort.

        Every failure here leaves the issue exactly as the reporter wrote it, which is
        why none of them is escalated.

        The issue is rewritten before the draft is stored, not after. Nothing retries a
        failed rewrite, so storing first would leave the panel showing a tidy summary
        line for an issue that still reads as the raw report — two descriptions of the
        same thing, with no way to tell which one the developer is looking at.
        """
        try:
            write_up = await write_up_report(
                client=self.client, model=self.runtime_models.slow_model, ticket=ticket
            )
            if write_up is None:
                logfire.info(
                    "Report write-up produced nothing; the issue keeps the original wording",
                    ticket_id=ticket.ticket_id,
                )
                return
            if ticket.issue_number is None:
                logfire.warn(
                    "A report was written up before its issue existed; dropping the draft",
                    ticket_id=ticket.ticket_id,
                )
                return
            try:
                await self.issues.update_issue(
                    number=ticket.issue_number,
                    title=write_up.title,
                    body=render_issue_body(ticket=ticket, write_up=write_up),
                )
            except GitHubIssuesError as exc:
                # Named separately from the model call above: the write-up worked, and
                # what did not is the edit. Nothing retries it, so the issue keeps the
                # reporter's own words and the panel keeps showing their first line.
                logfire.warn(
                    "Could not rewrite a report's issue; it keeps the original wording",
                    ticket_id=ticket.ticket_id,
                    issue_number=ticket.issue_number,
                    _exc_info=exc,
                )
                return
            await store_write_up(
                ticket_id=ticket.ticket_id,
                label=write_up.label,
                category=write_up.category,
                draft_title=write_up.title,
                draft_body=write_up.body,
            )
            try:
                await self.issues.add_labels(
                    number=ticket.issue_number,
                    labels=label_for_category(category=write_up.category),
                )
            except GitHubIssuesError as exc:
                # Its own step so the log says what actually failed: the issue is already
                # rewritten by this point, and a rejected label costs only the label.
                logfire.warn(
                    "Could not label a report's issue",
                    ticket_id=ticket.ticket_id,
                    issue_number=ticket.issue_number,
                    _exc_info=exc,
                )
        # Broad on purpose: this is a background task boundary. Anything escaping it
        # would surface as an unhandled task exception with nothing to act on, while the
        # report itself is already filed in the reporter's own words.
        except Exception as exc:
            logfire.warn(
                "Report write-up failed; the issue keeps the original wording",
                ticket_id=ticket.ticket_id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    async def _notify_owner(self, *, ticket: FeedbackTicket) -> None:
        """Tells the bot's owner a report came in.

        GitHub does not notify you about issues you opened yourself, and the token may
        well be yours, so this is what keeps a new report from arriving silently.
        """
        try:
            application = await self.bot.application_info()
            owner = application.owner
            if owner is None:
                # A team-owned application has no single owner to write to. This is a
                # standing configuration state, not a blip, so it is said out loud once
                # per report rather than returning as if nothing was meant to happen.
                logfire.warn(
                    "This application has no owner to notify; new reports arrive silently",
                    ticket_id=ticket.ticket_id,
                )
                return
            number = f"#{ticket.issue_number}" if ticket.issue_number else "（還沒有單號）"
            origin = ticket.guild_name or "私訊"
            embed = Embed(
                title=f"🎫 新的回報 {number}",
                description=ticket.raw_text[:2000],
                color=NEUTRAL_BLUE,
            )
            embed.set_footer(text=f"{ticket.display_name} ({ticket.user_name}) · {origin}")
            await owner.send(embed=embed)
        # Broad on purpose: a closed DM, a missing shared server, or a transport hiccup
        # all mean the same thing here, and none of them should touch the report itself.
        # `warn`, not `info`: this is the only channel that announces a new report, and a
        # closed DM keeps failing, so every report after it would arrive unannounced.
        except Exception as exc:
            logfire.warn(
                "Could not notify the owner about a new report",
                ticket_id=ticket.ticket_id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    async def _throttle_notice(self, *, user_id: int, outstanding: int) -> str:
        """Returns why this person may not file right now, or an empty string."""
        if outstanding >= self.config.max_open_reports:
            return f"你已經有 {outstanding} 張還沒處理完的單了，先等其中一張有結果再回報新的吧。"
        elapsed = await seconds_since_last_ticket(user_id=user_id)
        if elapsed is not None and elapsed < self.config.submit_cooldown_seconds:
            wait_minutes = max(1, int((self.config.submit_cooldown_seconds - elapsed) // 60) + 1)
            return f"你剛剛才回報過，大約 {wait_minutes} 分鐘後再送下一張。"
        return ""

    async def submit_report(
        self, *, interaction: Interaction[commands.Bot], text: str, outstanding: int
    ) -> None:
        """Files a new report and answers the person who wrote it."""
        if interaction.user is None:
            return
        # Acked first: the local write plus one GitHub call can outrun Discord's window,
        # and a late answer would look like the report was lost.
        await interaction.response.defer(ephemeral=True)
        notice = await self._throttle_notice(user_id=interaction.user.id, outstanding=outstanding)
        if notice:
            await interaction.followup.send(
                embed=build_notice_embed(description=notice), ephemeral=True
            )
            return
        guild = interaction.guild
        try:
            ticket = await create_ticket(
                user_id=interaction.user.id,
                user_name=interaction.user.name,
                display_name=interaction.user.display_name,
                guild_id=guild.id if guild is not None else None,
                guild_name=guild.name if guild is not None else "",
                channel_id=interaction.channel_id,
                locale=_locale_text(interaction=interaction),
                raw_text=text.strip(),
            )
        # Broad on purpose: this is the one step whose failure loses the report outright,
        # and the interaction is already deferred, so anything escaping here would leave
        # the person watching a spinner that never resolves with nothing in the log.
        except Exception as exc:
            logfire.error(
                "Could not store a user report; it is lost",
                user_id=interaction.user.id,
                guild_id=guild.id if guild is not None else None,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await interaction.followup.send(
                embed=build_notice_embed(
                    description="出了點問題，你寫的東西沒有存下來，再試一次看看。"
                ),
                ephemeral=True,
            )
            return
        number = await self._open_issue(ticket=ticket)
        if number is not None:
            ticket = ticket.model_copy(update={"issue_number": number})
        # Logged before the answer, not after: a failed followup would otherwise leave a
        # filed report with no trace of it anywhere.
        logfire.info(
            "A user report was filed",
            ticket_id=ticket.ticket_id,
            issue_number=ticket.issue_number,
            user_id=ticket.user_id,
            guild_id=ticket.guild_id,
        )
        self._spawn(self._notify_owner(ticket=ticket))
        if number is not None:
            self._spawn(self._apply_write_up(ticket=ticket))
        await interaction.followup.send(embed=build_submitted_embed(ticket=ticket), ephemeral=True)

    async def submit_reply(
        self, *, interaction: Interaction[commands.Bot], ticket_id: int, text: str
    ) -> None:
        """Relays one more line from the reporter onto their own report."""
        if interaction.user is None:
            return
        await interaction.response.defer(ephemeral=True)
        ticket = await get_ticket(ticket_id=ticket_id)
        if ticket is None:
            await interaction.followup.send(
                embed=build_notice_embed(description="這張單找不到了。"), ephemeral=True
            )
            return
        # The panel is ephemeral, so only its owner can reach this; the check is here
        # because the call writes to a public issue and cheap certainty is worth more
        # than the assumption. It is loud because nothing should ever reach it.
        if ticket.user_id != interaction.user.id:
            logfire.warn(
                "Refused to relay a reply onto someone else's report",
                ticket_id=ticket.ticket_id,
                viewer_id=interaction.user.id,
                owner_id=ticket.user_id,
            )
            await interaction.followup.send(
                embed=build_notice_embed(description="這不是你的單。"), ephemeral=True
            )
            return
        if ticket.issue_number is None:
            await interaction.followup.send(
                embed=build_notice_embed(
                    description="這張單還在建立中，等它有單號之後就可以補充了。"
                ),
                ephemeral=True,
            )
            return
        body = (
            f"{REPORTER_COMMENT_MARKER}\n"
            f"**{ticket.display_name}** wrote from Discord:\n\n"
            f"{text.strip()}"
        )
        try:
            await self.issues.add_comment(number=ticket.issue_number, body=body)
        except GitHubIssuesError as exc:
            logfire.warn(
                "Could not relay a reporter's reply",
                ticket_id=ticket.ticket_id,
                issue_number=ticket.issue_number,
                _exc_info=exc,
            )
            await interaction.followup.send(
                embed=build_notice_embed(description="送不出去，等一下再試一次。"), ephemeral=True
            )
            return
        try:
            await count_relayed_reply(ticket_id=ticket.ticket_id)
        # Broad on purpose: the comment is already on the issue, so nothing here can be
        # undone. Losing the count would make the reporter's own line read back to them
        # as the developer's answer, which is the one thing this counter exists to stop.
        except Exception as exc:
            logfire.error(
                "Relayed a reply but could not count it; the panel may read it as an answer",
                ticket_id=ticket.ticket_id,
                issue_number=ticket.issue_number,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        await interaction.followup.send(
            embed=build_notice_embed(
                description=f"補上去了，開發者會在 #{ticket.issue_number} 看到。",
                color=DISCORD_GREEN,
            ),
            ephemeral=True,
        )

    def _unavailable_embed(self) -> Embed:
        """The answer when reports cannot be filed at all.

        Naming someone to talk to beats naming a switch: the person wanted to reach the
        developer, and a report accepted here would be one nobody could ever read.
        """
        if self.config.contact:
            description = (
                "這台 bot 還沒接上回報系統，我沒辦法幫你開單。\n"
                f"有問題的話直接找 **{self.config.contact}**，他就是做這隻 bot 的人。"
            )
        else:
            description = "回報功能目前沒有開，我沒辦法幫你開單。"
        return build_notice_embed(description=description, color=DISCORD_YELLOW)

    @nextcord.slash_command(
        name="feedback",
        description="Report a problem or ask for a feature, and read the developer's replies.",
        name_localizations={Locale.zh_TW: "回報", Locale.ja: "フィードバック"},
        description_localizations={
            Locale.zh_TW: "回報問題或許願，也可以看開發者的回覆",
            Locale.ja: "不具合や要望を開発者に送り、返信を確認します。",
        },
        nsfw=False,
    )
    async def feedback(self, interaction: Interaction[commands.Bot]) -> None:
        """Opens the caller's own report panel."""
        if interaction.user is None:
            return
        if not self.config.available:
            await interaction.response.send_message(
                embed=self._unavailable_embed(), ephemeral=True
            )
            return
        # Deferred because the panel reads every listed report from GitHub; the edit
        # below turns the placeholder into the panel itself.
        await interaction.response.defer(ephemeral=True)
        panel = await self.load_rows(user_id=interaction.user.id)
        view = FeedbackPanelView(host=self, rows=panel.rows, total=panel.total)
        await interaction.edit_original_message(
            embed=build_panel_embed(rows=panel.rows, total=panel.total), view=view
        )
        view.bind_origin(interaction=interaction)

    @tasks.loop(minutes=RETRY_INTERVAL_MINUTES)
    async def retry_unfiled_reports(self) -> None:
        """Opens the issues that the submit path could not.

        This is the other half of writing locally first. Without it a report filed during
        a GitHub outage would sit in the store forever, which is indistinguishable from
        losing it as far as the reporter can tell.

        Wrapped whole, because an exception escaping a `tasks.loop` stops it for the rest
        of the process (nextcord only retries a short list of connection errors), and this
        loop is the entire mechanism behind the promise the reporter was given.

        Rows younger than `RETRY_MIN_AGE_SECONDS` are left alone: a submit in flight has
        already opened its issue and is about to record the number, and picking it up here
        would open a second one.
        """
        try:
            pending = await tickets_awaiting_issue(
                limit=RETRY_BATCH_SIZE, min_age_seconds=RETRY_MIN_AGE_SECONDS
            )
            if not pending:
                return
            if not self.config.available:
                # Worth saying: reports are queueing up behind a switch or a missing
                # token, and the reporters were told they would be filed.
                logfire.info(
                    "Reports are waiting for an issue but reporting is not configured",
                    pending=len(pending),
                )
                return
            stalled = [
                ticket
                for ticket in pending
                if _age_seconds(ticket=ticket) > RETRY_STALLED_AFTER_SECONDS
            ]
            if stalled:
                # Past this age it is no longer an outage waiting to clear, and the queue
                # is ordered, so one report nothing will ever accept blocks every later one.
                logfire.error(
                    "Reports have been waiting for an issue far too long",
                    stalled=len(stalled),
                    oldest_ticket_id=stalled[0].ticket_id,
                )
            for ticket in pending:
                number = await self._open_issue(ticket=ticket)
                if number is None:
                    # GitHub is still refusing; the rest of the batch would only repeat it.
                    return
                self._spawn(
                    self._apply_write_up(ticket=ticket.model_copy(update={"issue_number": number}))
                )
        # Broad on purpose, see the docstring: the alternative is a loop that dies once
        # and takes every queued report with it, silently, until the next deploy.
        except Exception as exc:
            logfire.error(
                "The report retry sweep failed; it will run again next interval",
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Starts the retry sweep once, on the first gateway ready."""
        if self._started:
            return
        self._started = True
        self.retry_unfiled_reports.start()

    def cog_unload(self) -> None:
        """Stops the retry sweep when the cog is unloaded or reloaded."""
        self.retry_unfiled_reports.cancel()


def setup(bot: commands.Bot) -> None:
    """Adds the FeedbackCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(FeedbackCogs(bot), override=True)
