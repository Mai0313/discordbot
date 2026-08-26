"""The `/feedback` panel: the report list, one report's detail, and the two forms.

Every message here is ephemeral, so a view is only ever operated by the person who
opened it and no per-interaction author check is needed on top (same reasoning as
`cogs/memory/views.py`).

The panel reads which reports belong to whom from the local store and reads their state
from GitHub, so a report filed seconds ago is always listed even though GitHub search
would not have indexed it yet.
"""

from typing import Protocol
import contextlib

import nextcord
from nextcord import Embed, ButtonStyle, Interaction, SelectOption
from pydantic import Field, BaseModel
from nextcord.ui import View, Modal, Button, TextInput, StringSelect
from nextcord.ext import commands

from discordbot.typings.colors import NEUTRAL_BLUE, NEUTRAL_GREY, DISCORD_GREEN, DISCORD_YELLOW
from discordbot.cogs.feedback.github import (
    CloseOutcome,
    IssueComment,
    IssueSnapshot,
    close_outcome,
)
from discordbot.cogs.feedback.database import FeedbackTicket

FEEDBACK_VIEW_TIMEOUT_SECONDS = 180

# A form someone opened and walked away from. Longer than the panel's idle timeout, since
# writing a report takes as long as it takes, but bounded so the count it carries expires.
REPORT_FORM_TIMEOUT_SECONDS = 900

# The select caps at 25 options and the embed at 25 fields; ten is what stays readable
# without scrolling, and older reports are answered on the issue anyway.
MAX_PANEL_TICKETS = 10
MAX_DETAIL_REPLIES = 5

# Discord's real limits are 1024 per field value, 256 per title, 100 per select label and
# 4096 per description, under a message-wide 6000 across every text-bearing field of every
# embed. The numbers below sit under those with headroom, because a per-field guard alone
# still 400s once enough in-limit fields add up: the report form takes 2000 characters, so
# ten of those in one panel would be 20k without a cap here. Worst case with these values
# is roughly 5k on the detail screen and 2k on the panel.
_MAX_FIELD_CHARS = 700
_MAX_SELECT_LABEL_CHARS = 90
_MAX_EMBED_TITLE_CHARS = 240
_MAX_QUOTED_CHARS = 900
_MAX_SUMMARY_CHARS = 120

# Marks the body caps that run through `clipped`, so a trimmed report does not read as the bot
# having garbled it. The embed title and the select label are hard slices instead: both are one
# line of chrome the reader can open the full text from.
_TRUNCATED_SUFFIX = "…"

REPORT_TITLE = "🎫 回報中心"


class TicketStatus(BaseModel):
    """How one report reads on screen, and whether it still counts as outstanding."""

    text: str = Field(..., description="The status as shown to the reporter.")
    color: int = Field(..., description="Embed colour for the detail view.")
    outstanding: bool = Field(
        ..., description="Whether this report counts against the per-person open cap."
    )


# How each way of closing an issue reads in the panel. A table rather than a branch each,
# because `outstanding=False` is true of all three and stating it once is what says the cap
# is about the maintainer's inbox rather than about whether the work is finished.
_CLOSED_STATUS: dict[CloseOutcome, TicketStatus] = {
    "completed": TicketStatus(text="✅ 已處理", color=DISCORD_GREEN, outstanding=False),
    "not_planned": TicketStatus(text="⚪ 不處理", color=NEUTRAL_GREY, outstanding=False),
    "duplicate": TicketStatus(text="🔁 併入其他單", color=NEUTRAL_BLUE, outstanding=False),
}


class TicketRow(BaseModel):
    """One stored report together with whatever GitHub currently says about it."""

    ticket: FeedbackTicket = Field(..., description="The stored report.")
    snapshot: IssueSnapshot | None = Field(
        ..., description="The issue as GitHub reports it, or None when it is unreadable."
    )

    @property
    def status(self) -> TicketStatus:
        """Derives the displayed status; nothing about it is stored.

        The developer works on the issue, so the issue is the truth. Storing a copy
        would only give it something to drift from.

        The open-but-answered split reads the comment count against the replies this bot
        relayed on the reporter's behalf, which is exact for the two voices that matter.
        A comment from a passer-by (neither maintainer nor reporter) would read as an
        answer here; the detail view then shows the truth, which is that no maintainer
        has written anything.

        A report whose issue could not be read is NOT outstanding, deliberately. The
        alternative reads better on paper and is wrong in practice: GitHub having a bad
        minute would turn three long-closed reports into a cap that refuses to accept a
        new one, in the exact situation this whole design exists to keep working.

        A duplicate is not outstanding either, even though the work it describes is still
        somewhere in the queue. What the cap protects is the maintainer's inbox, and a
        report already merged into another issue costs that inbox nothing.
        """
        if self.snapshot is None:
            if self.ticket.issue_number is None:
                return TicketStatus(text="⏳ 建立中", color=DISCORD_YELLOW, outstanding=True)
            return TicketStatus(text="❔ 讀不到狀態", color=NEUTRAL_GREY, outstanding=False)
        if self.snapshot.state == "closed":
            return _CLOSED_STATUS[close_outcome(state_reason=self.snapshot.state_reason)]
        if self.snapshot.comment_count > self.ticket.relayed_replies:
            return TicketStatus(text="🟢 處理中", color=NEUTRAL_BLUE, outstanding=True)
        return TicketStatus(text="🟡 還沒回覆", color=DISCORD_YELLOW, outstanding=True)


class TicketDetail(BaseModel):
    """One report opened up: what was written, and everything said about it since."""

    row: TicketRow = Field(..., description="The report and its current issue state.")
    comments: list[IssueComment] | None = Field(
        ...,
        description=(
            "Maintainer replies and the reporter's own, oldest first; None when the "
            "conversation could not be read at all."
        ),
    )


class PanelRows(BaseModel):
    """What the panel lists, and how many reports there are behind it."""

    rows: list[TicketRow] = Field(..., description="The reports being listed, newest first.")
    total: int = Field(..., description="How many reports this person has filed in total.")


class FeedbackHost(Protocol):
    """What the views need from the cog, so neither module imports the other."""

    async def load_rows(self, *, user_id: int) -> PanelRows:
        """Returns one person's newest reports with their current issue state."""
        ...

    async def load_detail(self, *, ticket_id: int, viewer_id: int) -> TicketDetail | None:
        """Returns one report with its conversation, or None when the viewer may not see it.

        `viewer_id` is checked against the report's owner. The panel is ephemeral, so a
        stray id can only come from a doctored client, but the reply path already refuses
        to take someone else's word for it and the read path is where the other person's
        text actually is.
        """
        ...

    async def submit_report(
        self, *, interaction: Interaction[commands.Bot], text: str, outstanding: int
    ) -> None:
        """Files a new report and answers the person who wrote it.

        `outstanding` is how many unresolved reports the panel was showing when the form
        was opened, which is what the per-person cap is measured against — the panel had
        just read every one of them, so the check costs no extra request. It is advisory
        by construction: an unsent form holds its count until it expires, so the cooldown
        is the limit that actually binds.
        """
        ...

    async def submit_reply(
        self, *, interaction: Interaction[commands.Bot], ticket_id: int, text: str
    ) -> None:
        """Relays one more line from the reporter onto their own report."""
        ...


def _short_date(*, row: TicketRow) -> str:
    """The filing date as the panel shows it."""
    return row.ticket.created_at.strftime("%m/%d")


def clipped(*, text: str, limit: int) -> str:
    """Returns `text` within `limit`, marked when something was cut off.

    The marker is the whole point. A report is the one place where a person needs to
    recognise their own words, and text that just stops reads as the bot having garbled
    it rather than as more text existing.

    The cut is rstripped before the marker goes on, so a cut landing mid-space reads as
    `word…` rather than `word …`, which looks like the marker belongs to the next word.

    Shared with `notice.py`, which renders the same reporter-authored text into a DM.
    """
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATED_SUFFIX)].rstrip() + _TRUNCATED_SUFFIX


def build_panel_embed(*, rows: list[TicketRow], total: int | None = None) -> Embed:
    """Builds the report list shown by `/feedback`.

    `total` is how many reports the person actually has, when that is more than the
    listed rows; the footer then says how many are not on screen instead of leaving
    them to notice.
    """
    embed = Embed(title=REPORT_TITLE, color=NEUTRAL_BLUE)
    if not rows:
        embed.description = (
            "你還沒有回報過東西。\n"
            "遇到怪怪的地方，或是想要什麼功能，都可以按下面的按鈕告訴我。\n"
            "-# 回報會變成專案上一張公開的單，上面有你的 Discord 名稱，開發者才找得到你。"
        )
        return embed
    embed.description = (
        "你回報過的東西都在這裡，開發者回覆的時候也會出現在這。\n"
        "-# 回報會變成專案上一張公開的單，上面有你的 Discord 名稱，開發者才找得到你。"
    )
    for row in rows:
        status = row.status
        number = f"#{row.ticket.issue_number}" if row.ticket.issue_number else "尚未編號"
        # Only the date. The issue's comment count includes anyone who wandered past on a
        # public repository, so a number here would promise replies the detail view then
        # (correctly) refuses to show. The status already carries whether anyone answered.
        summary = clipped(text=row.ticket.summary_line, limit=_MAX_SUMMARY_CHARS)
        embed.add_field(
            name=f"{number} · {status.text}",
            value=f"{summary}\n-# {_short_date(row=row)} 送出",
            inline=False,
        )
    footer = "單號記著就好，之後問進度報這個號碼"
    hidden = (total or 0) - len(rows)
    if hidden > 0:
        footer = f"另外還有 {hidden} 張比較舊的單沒顯示 | {footer}"
    embed.set_footer(text=footer)
    return embed


def build_detail_embed(*, detail: TicketDetail) -> Embed:
    """Builds one report's detail: the original wording, then the conversation."""
    ticket = detail.row.ticket
    status = detail.row.status
    number = f"#{ticket.issue_number}" if ticket.issue_number else "尚未編號"
    embed = Embed(
        title=f"{number} {ticket.summary_line}"[:_MAX_EMBED_TITLE_CHARS], color=status.color
    )
    quoted = clipped(text=ticket.raw_text.strip(), limit=_MAX_QUOTED_CHARS) or "（沒有內容）"
    quoted_lines = "\n".join(f"> {line}" for line in quoted.splitlines())
    embed.description = f"{status.text} · {_short_date(row=detail.row)} 送出\n\n{quoted_lines}"
    if detail.comments is None:
        # Saying "nobody has replied" here would be asserting something we did not manage
        # to look up, and the person is only in this screen to find out whether anyone did.
        embed.add_field(
            name="現在讀不到回覆",
            value="跟伺服器要回覆的時候出了點問題，按下面的重新整理再試一次。",
            inline=False,
        )
        embed.set_footer(text="只有你看得到這個畫面")
        return embed
    if not detail.comments:
        embed.add_field(
            name="還沒有回覆",
            value="開發者看過之後會回在這裡，之後再用 `/feedback` 回來看就好。",
            inline=False,
        )
    hidden = len(detail.comments) - MAX_DETAIL_REPLIES
    if hidden > 0:
        embed.add_field(
            name=f"上面還有 {hidden} 則比較早的對話", value="這裡只放最近的幾則。", inline=False
        )
    for comment in detail.comments[-MAX_DETAIL_REPLIES:]:
        voice = "你" if comment.from_reporter else "開發者"
        stamp = comment.created_at[:10]
        embed.add_field(
            name=f"{voice} · {stamp}",
            value=clipped(text=comment.body.strip(), limit=_MAX_FIELD_CHARS) or "（空白）",
            inline=False,
        )
    embed.set_footer(text="只有你看得到這個畫面")
    return embed


def build_notice_embed(*, description: str, color: int = DISCORD_YELLOW) -> Embed:
    """Builds a one-line answer that is not a panel (unavailable, throttled, failed)."""
    return Embed(title=REPORT_TITLE, description=description, color=color)


def build_submitted_embed(*, ticket: FeedbackTicket) -> Embed:
    """Builds the answer to a submitted report, numbered or still waiting for a number."""
    if ticket.issue_number is None:
        return Embed(
            title="✅ 收到了，你寫的東西我存下來了",
            description=(
                "單號還在跟伺服器要，我會自己重試。\n"
                "等一下用 `/feedback` 就看得到，你寫的內容不會不見。"
            ),
            color=DISCORD_YELLOW,
        )
    return Embed(
        title=f"✅ 收到了, 你的單號是 #{ticket.issue_number}",
        description=(
            "我把你寫的原封不動送過去了，等一下會自己整理成比較好讀的樣子。\n"
            "開發者回覆的時候，再用 `/feedback` 就看得到。"
        ),
        color=DISCORD_GREEN,
    ).set_footer(text=f"記住 #{ticket.issue_number} 就好，不需要 GitHub 帳號")


class ReportModal(Modal):
    """The one-field form behind 我要回報.

    One field because a Discord modal cannot take a file upload, so an image could only
    be a link, and Discord's own attachment links are signed and expire within about a
    day; a screenshot pasted into an issue would be a broken image by the time anyone
    opened it. When one is genuinely needed the developer asks on the issue and the
    reporter answers through 回一句.
    """

    def __init__(self, *, host: FeedbackHost, outstanding: int) -> None:
        """Initializes the form bound to the cog that files the report.

        The timeout is what keeps the carried `outstanding` count from living forever:
        nextcord keeps a modal with no timeout in its store indefinitely, so an unsent
        form would hold a stale count for as long as the process runs.
        """
        super().__init__(title="我要回報", timeout=REPORT_FORM_TIMEOUT_SECONDS)
        self.host = host
        self.outstanding = outstanding
        self.report_text = TextInput(
            label="發生什麼事",
            placeholder="慢慢講沒關係。你做了什麼，看到什麼，本來以為會怎樣。",
            style=nextcord.TextInputStyle.paragraph,
            required=True,
            min_length=8,
            max_length=2000,
        )
        self.add_item(self.report_text)

    async def callback(self, interaction: Interaction[commands.Bot]) -> None:
        """Hands the written report to the cog."""
        await self.host.submit_report(
            interaction=interaction,
            text=self.report_text.value or "",
            outstanding=self.outstanding,
        )


class ReplyModal(Modal):
    """The form behind 回一句, which relays one more line onto an existing report."""

    def __init__(self, *, host: FeedbackHost, ticket_id: int, number: int | None) -> None:
        """Initializes the reply form for one report."""
        super().__init__(
            title=f"回覆 #{number}" if number else "回覆這張單",
            timeout=REPORT_FORM_TIMEOUT_SECONDS,
        )
        self.host = host
        self.ticket_id = ticket_id
        self.reply_text = TextInput(
            label="想補充什麼",
            placeholder="補一點細節，或是回答開發者問的問題。",
            style=nextcord.TextInputStyle.paragraph,
            required=True,
            min_length=2,
            max_length=1500,
        )
        self.add_item(self.reply_text)

    async def callback(self, interaction: Interaction[commands.Bot]) -> None:
        """Hands the written reply to the cog."""
        await self.host.submit_reply(
            interaction=interaction, ticket_id=self.ticket_id, text=self.reply_text.value or ""
        )


class _TicketSelect(StringSelect["FeedbackPanelView"]):
    """The report picker on the panel.

    A subclass rather than a decorated callback because the options are the caller's
    own reports, which only exist once the panel has been loaded.
    """

    def __init__(self, *, rows: list[TicketRow], host: FeedbackHost) -> None:
        """Builds one option per listed report."""
        options = [
            SelectOption(
                label=(
                    f"#{row.ticket.issue_number} {row.ticket.summary_line}"
                    if row.ticket.issue_number
                    else row.ticket.summary_line
                )[:_MAX_SELECT_LABEL_CHARS],
                value=str(row.ticket.ticket_id),
                description=row.status.text,
            )
            for row in rows
        ]
        super().__init__(placeholder="選一張單，看內容和開發者的回覆", options=options)
        self.host = host

    async def callback(self, interaction: Interaction[commands.Bot]) -> None:
        """Opens the selected report in place."""
        if interaction.user is None:
            return
        # Deferred first: reading this report is two GitHub calls, and Discord wants the
        # interaction acknowledged within three seconds.
        await interaction.response.defer()
        # The panel and the detail are the same ephemeral message, so the view being
        # replaced has to be stopped. Left running, its own timeout would later strip the
        # controls off whatever view is on that message by then.
        if self.view is not None:
            self.view.stop()
        detail = await self.host.load_detail(
            ticket_id=int(self.values[0]), viewer_id=interaction.user.id
        )
        if detail is None:
            await interaction.edit_original_message(
                embed=build_notice_embed(description="這張單找不到了。"), view=None
            )
            return
        view = TicketDetailView(host=self.host, detail=detail)
        await interaction.edit_original_message(embed=build_detail_embed(detail=detail), view=view)
        view.bind_origin(interaction=interaction)


class FeedbackPanelView(View):
    """The ephemeral report list, with a picker and the new-report button.

    Attributes:
        host: The cog that loads reports and files new ones.
        rows: The reports currently listed, newest first.
    """

    def __init__(self, *, host: FeedbackHost, rows: list[TicketRow]) -> None:
        """Initializes the panel, adding the picker only when there is something to pick."""
        super().__init__(timeout=FEEDBACK_VIEW_TIMEOUT_SECONDS)
        self.host = host
        self.rows = rows
        self._origin: Interaction[commands.Bot] | None = None
        if rows:
            self.add_item(_TicketSelect(rows=rows, host=host))

    def bind_origin(self, interaction: Interaction[commands.Bot]) -> None:
        """Records the originating interaction so timeout can drop the controls."""
        self._origin = interaction

    def outstanding_count(self) -> int:
        """How many listed reports are known to still be unresolved.

        Read straight off what the panel just fetched, so the submission cap costs no
        extra request: the person is pressing a button on a list whose state is current.
        A report whose issue could not be read does not count, so a bad minute at GitHub
        cannot turn into a refusal to accept anything.
        """
        return sum(1 for row in self.rows if row.status.outstanding)

    @nextcord.ui.button(label="✏️ 我要回報", style=ButtonStyle.primary)
    async def open_report_form(
        self, _button: Button["FeedbackPanelView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Opens the report form."""
        await interaction.response.send_modal(
            modal=ReportModal(host=self.host, outstanding=self.outstanding_count())
        )

    async def on_timeout(self) -> None:
        """Drops the controls once the panel goes idle."""
        if self._origin is None:
            return
        # Inert cleanup, broad on purpose: nextcord runs `on_timeout` in a bare
        # `create_task`, and the ephemeral panel may already be dismissed.
        with contextlib.suppress(Exception):
            await self._origin.edit_original_message(view=None)


class TicketDetailView(View):
    """One report opened from the panel: go back, refresh, or add a line.

    Attributes:
        host: The cog that loads reports and relays replies.
        detail: The report currently displayed.
    """

    def __init__(self, *, host: FeedbackHost, detail: TicketDetail) -> None:
        """Initializes the detail controls for one report."""
        super().__init__(timeout=FEEDBACK_VIEW_TIMEOUT_SECONDS)
        self.host = host
        self.detail = detail
        self._origin: Interaction[commands.Bot] | None = None

    def bind_origin(self, interaction: Interaction[commands.Bot]) -> None:
        """Records the interaction the detail was rendered onto."""
        self._origin = interaction

    @nextcord.ui.button(label="◀ 返回", style=ButtonStyle.secondary)
    async def back_to_panel(
        self, _button: Button["TicketDetailView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Rebuilds the list and shows it in place."""
        # Deferred and stopped for the same reasons as the picker: rebuilding the list is
        # one GitHub read per report, and this view is about to stop owning the message.
        await interaction.response.defer()
        self.stop()
        panel = await self.host.load_rows(user_id=self.detail.row.ticket.user_id)
        view = FeedbackPanelView(host=self.host, rows=panel.rows)
        await interaction.edit_original_message(
            embed=build_panel_embed(rows=panel.rows, total=panel.total), view=view
        )
        view.bind_origin(interaction=interaction)

    # Not named `refresh`: `View.__init__` rebinds a callback's name onto the item, and
    # `refresh` is a real method the gateway calls on every tracked message update.
    @nextcord.ui.button(label="🔄 重新整理", style=ButtonStyle.secondary)
    async def reload_ticket(
        self, _button: Button["TicketDetailView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Re-reads this report from GitHub and redraws it."""
        if interaction.user is None:
            return
        # Not stopped, unlike the other two: this view stays on the message, and stopping
        # it here would leave its own buttons inert.
        await interaction.response.defer()
        detail = await self.host.load_detail(
            ticket_id=self.detail.row.ticket.ticket_id, viewer_id=interaction.user.id
        )
        if detail is None:
            await interaction.edit_original_message(
                embed=build_notice_embed(description="這張單找不到了。"), view=None
            )
            return
        self.detail = detail
        await interaction.edit_original_message(embed=build_detail_embed(detail=detail), view=self)

    @nextcord.ui.button(label="💬 回一句", style=ButtonStyle.primary)
    async def open_reply_form(
        self, _button: Button["TicketDetailView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Opens the reply form for this report."""
        ticket = self.detail.row.ticket
        if ticket.issue_number is None:
            await interaction.response.send_message(
                embed=build_notice_embed(
                    description="這張單還在建立中，等它有單號之後就可以補充了。"
                ),
                ephemeral=True,
            )
            return
        await interaction.response.send_modal(
            modal=ReplyModal(
                host=self.host, ticket_id=ticket.ticket_id, number=ticket.issue_number
            )
        )

    async def on_timeout(self) -> None:
        """Drops the controls once the view goes idle."""
        if self._origin is None:
            return
        # Same inert cleanup as the panel's.
        with contextlib.suppress(Exception):
            await self._origin.edit_original_message(view=None)
