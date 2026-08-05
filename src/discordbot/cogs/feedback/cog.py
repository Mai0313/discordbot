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
from discordbot.cogs.feedback.views import (
    MAX_PANEL_TICKETS,
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
    attach_issue_number,
    count_relayed_reply,
    tickets_awaiting_issue,
    seconds_since_last_ticket,
)

# How often the sweep tries again for reports whose issue was never opened, and how many
# it takes per pass. Small: this only ever runs after a GitHub outage.
RETRY_INTERVAL_MINUTES = 10
RETRY_BATCH_SIZE = 10


def _locale_text(*, interaction: Interaction[commands.Bot]) -> str:
    """The reporter's Discord locale as a plain string, or empty when unknown."""
    locale = interaction.locale
    if isinstance(locale, Locale):
        return str(locale.value)
    return str(locale or "")


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

    async def load_rows(self, *, user_id: int) -> list[TicketRow]:
        """Returns one person's reports with their current issue state."""
        tickets = await list_user_tickets(user_id=user_id, limit=MAX_PANEL_TICKETS)
        snapshots = await asyncio.gather(
            *(self._read_snapshot(ticket=ticket) for ticket in tickets)
        )
        return [
            TicketRow(ticket=ticket, snapshot=snapshot)
            for ticket, snapshot in zip(tickets, snapshots, strict=True)
        ]

    async def load_detail(self, *, ticket_id: int) -> TicketDetail | None:
        """Returns one report with its conversation, or None when it is gone."""
        ticket = await get_ticket(ticket_id=ticket_id)
        if ticket is None:
            return None
        snapshot = await self._read_snapshot(ticket=ticket)
        comments = []
        if ticket.issue_number is not None:
            try:
                comments = await self.issues.read_conversation(number=ticket.issue_number)
            except GitHubIssuesError as exc:
                logfire.warn(
                    "Could not read a report's replies; showing the report alone",
                    ticket_id=ticket.ticket_id,
                    issue_number=ticket.issue_number,
                    _exc_info=exc,
                )
        return TicketDetail(row=TicketRow(ticket=ticket, snapshot=snapshot), comments=comments)

    async def _open_issue(self, *, ticket: FeedbackTicket) -> int | None:
        """Opens the issue for a stored report, returning None when GitHub refuses.

        Returning None is safe precisely because the report is already stored: the sweep
        picks the row up again, and the reporter was told their words were kept.
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
        await attach_issue_number(ticket_id=ticket.ticket_id, issue_number=number)
        return number

    async def _apply_write_up(self, *, ticket: FeedbackTicket) -> None:
        """Writes the report up and rewrites its issue, best-effort.

        Every failure here leaves the issue exactly as the reporter wrote it, which is
        why none of them is escalated.
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
            await store_write_up(
                ticket_id=ticket.ticket_id,
                label=write_up.label,
                category=write_up.category,
                draft_title=write_up.title,
                draft_body=write_up.body,
            )
            if ticket.issue_number is None:
                return
            await self.issues.update_issue(
                number=ticket.issue_number,
                title=write_up.title,
                body=render_issue_body(ticket=ticket, write_up=write_up),
            )
            try:
                await self.issues.add_labels(
                    number=ticket.issue_number,
                    labels=label_for_category(category=write_up.category),
                )
            except GitHubIssuesError as exc:
                # Its own step so the log says what actually failed: the issue is
                # already rewritten by this point, and a label the repository does not
                # carry costs only the label.
                logfire.info(
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
        # Broad on purpose: a closed DM, a team-owned application, or a transport hiccup
        # all mean the same thing here, and none of them should touch the report itself.
        except Exception as exc:
            logfire.info(
                "Could not notify the owner about a new report",
                ticket_id=ticket.ticket_id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    async def _throttle_notice(self, *, user_id: int, outstanding: int) -> str:
        """Returns why this person may not file right now, or an empty string."""
        if outstanding >= self.config.max_open_reports:
            return f"你已經有 {outstanding} 張還沒處理完的單了, 先等其中一張有結果再回報新的吧。"
        elapsed = await seconds_since_last_ticket(user_id=user_id)
        if elapsed is not None and elapsed < self.config.submit_cooldown_seconds:
            wait_minutes = max(1, int((self.config.submit_cooldown_seconds - elapsed) // 60) + 1)
            return f"你剛剛才回報過, 大約 {wait_minutes} 分鐘後再送下一張。"
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
        number = await self._open_issue(ticket=ticket)
        if number is not None:
            ticket = ticket.model_copy(update={"issue_number": number})
        await interaction.followup.send(embed=build_submitted_embed(ticket=ticket), ephemeral=True)
        logfire.info(
            "A user report was filed",
            ticket_id=ticket.ticket_id,
            issue_number=ticket.issue_number,
            guild_id=ticket.guild_id,
        )
        self._spawn(self._notify_owner(ticket=ticket))
        if number is not None:
            self._spawn(self._apply_write_up(ticket=ticket))

    async def submit_reply(
        self, *, interaction: Interaction[commands.Bot], ticket_id: int, text: str
    ) -> None:
        """Relays one more line from the reporter onto their own report."""
        if interaction.user is None:
            return
        await interaction.response.defer(ephemeral=True)
        ticket = await get_ticket(ticket_id=ticket_id)
        # The panel is ephemeral, so only its owner can reach this; the check is here
        # because the call writes to a public issue and cheap certainty is worth more
        # than the assumption.
        if ticket is None or ticket.user_id != interaction.user.id or ticket.issue_number is None:
            await interaction.followup.send(
                embed=build_notice_embed(description="這張單現在沒辦法補充。"), ephemeral=True
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
                embed=build_notice_embed(description="送不出去, 等一下再試一次。"), ephemeral=True
            )
            return
        await count_relayed_reply(ticket_id=ticket.ticket_id)
        await interaction.followup.send(
            embed=build_notice_embed(
                description=f"補上去了, 開發者會在 #{ticket.issue_number} 看到。",
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
                "這台 bot 還沒接上回報系統, 我沒辦法幫你開單。\n"
                f"有問題的話直接找 **{self.config.contact}**, 他就是做這隻 bot 的人。"
            )
        else:
            description = "回報功能目前沒有開, 我沒辦法幫你開單。"
        return build_notice_embed(description=description, color=DISCORD_YELLOW)

    @nextcord.slash_command(
        name="feedback",
        description="Report a problem or ask for a feature, and read the developer's replies.",
        name_localizations={Locale.zh_TW: "回報", Locale.ja: "フィードバック"},
        description_localizations={
            Locale.zh_TW: "回報問題或許願, 也可以看開發者的回覆",
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
        rows = await self.load_rows(user_id=interaction.user.id)
        view = FeedbackPanelView(host=self, rows=rows)
        await interaction.edit_original_message(embed=build_panel_embed(rows=rows), view=view)
        view.bind_origin(interaction=interaction)

    @tasks.loop(minutes=RETRY_INTERVAL_MINUTES)
    async def retry_unfiled_reports(self) -> None:
        """Opens the issues that the submit path could not.

        This is the other half of writing locally first. Without it a report filed during
        a GitHub outage would sit in the store forever, which is indistinguishable from
        losing it as far as the reporter can tell.
        """
        if not self.config.available:
            return
        pending = await tickets_awaiting_issue(limit=RETRY_BATCH_SIZE)
        for ticket in pending:
            number = await self._open_issue(ticket=ticket)
            if number is None:
                # GitHub is still refusing; the rest of the batch would only repeat it.
                return
            self._spawn(
                self._apply_write_up(ticket=ticket.model_copy(update={"issue_number": number}))
            )

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Starts the retry sweep once, on the first gateway ready."""
        if self._started:
            return
        self._started = True
        self.retry_unfiled_reports.start()


def setup(bot: commands.Bot) -> None:
    """Adds the FeedbackCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(FeedbackCogs(bot), override=True)
