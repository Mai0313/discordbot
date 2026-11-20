"""Persistent thumbs-up / thumbs-down feedback buttons and ephemeral downvote reason picker.

`ReplyFeedbackView` is registered once on the bot at startup with fixed `custom_id`s so buttons on
replies keep working across restarts and in `/ask` contexts. All SQLite writes run only after the
interaction has been acknowledged.
"""

from typing import ClassVar
import contextlib

import logfire
from nextcord import ButtonStyle, Interaction, HTTPException, TextInputStyle
from nextcord.ui import View, Button, TextInput, button
from nextcord.ext import commands

from discordbot.utils.logged_ui import LoggedView, LoggedModal
from discordbot.cogs.gen_reply.feedback_store import (
    FeedbackVote,
    DownvoteReasonTag,
    set_feedback_reason,
    toggle_feedback_vote,
)

FEEDBACK_UP_CUSTOM_ID = "gen_reply:feedback:up"
FEEDBACK_DOWN_CUSTOM_ID = "gen_reply:feedback:down"
# Matches Discord's 15-minute ephemeral interaction token window.
FEEDBACK_REASON_VIEW_TIMEOUT_SECONDS = 900.0

DOWNVOTE_REASONS: tuple[tuple[DownvoteReasonTag, str], ...] = (
    ("off_topic", "答非所問"),
    ("inaccurate", "資訊錯誤"),
    ("tone", "語氣怪"),
    ("verbose", "太囉嗦"),
    ("other", "其他"),
)
REASON_LABELS: dict[DownvoteReasonTag, str] = dict(DOWNVOTE_REASONS)


class DownvoteReasonModal(LoggedModal):
    """Single-field modal opened only when the user explicitly clicks `其他`."""

    def __init__(self, parent_view: "DownvoteReasonView") -> None:
        """Initializes the modal pre-populated with any previously submitted text."""
        super().__init__(title="補充倒讚原因")
        self.parent_view = parent_view
        self.reason_input: TextInput[View] = TextInput(
            label="哪裡覺得不好?",
            style=TextInputStyle.paragraph,
            placeholder="例如：忘記前面的記憶、程式碼有錯、沒有看我附的圖...",
            default_value=parent_view.selected_text or None,
            required=True,
            max_length=300,
        )
        self.add_item(item=self.reason_input)

    async def callback(self, interaction: Interaction[commands.Bot]) -> None:
        """Acknowledges the submit, persists the free-text reason, and updates the ephemeral notice."""
        if interaction.user is None:
            return
        await interaction.response.defer()
        text = (self.reason_input.value or "").strip()
        try:
            result = await set_feedback_reason(
                reply_message_id=self.parent_view.reply_message_id,
                user_id=interaction.user.id,
                reason_tag="other",
                reason_text=text,
            )
            if not result.updated:
                await interaction.edit_original_message(
                    content="你已變更或取消這則評分", view=None
                )
                return
            self.parent_view.apply_selection(
                reason_tag=result.reason_tag, reason_text=result.reason_text
            )
            logfire.info(
                "gen_reply feedback reason",
                reply_message_id=self.parent_view.reply_message_id,
                source_message_id=result.source_message_id,
                model=result.model,
                effort=result.effort,
                route=result.route,
                user_id=interaction.user.id,
                reason_tag=result.reason_tag,
                reason_chars=len(result.reason_text or ""),
                is_asker=result.is_asker,
            )
            await interaction.edit_original_message(
                content=self.parent_view.status_text(), view=self.parent_view
            )
        except Exception as exc:
            # Broad on purpose: deferred modal submit must surface an error instead of stalling.
            logfire.error(
                "Failed to save downvote reason modal",
                reply_message_id=self.parent_view.reply_message_id,
                user_id=interaction.user.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            with contextlib.suppress(HTTPException):
                await interaction.edit_original_message(content="記錄原因時發生錯誤，請稍後再試")


class _ReasonButton(Button["DownvoteReasonView"]):
    """One quick-pick button inside `DownvoteReasonView`."""

    def __init__(self, reason_tag: DownvoteReasonTag, label: str) -> None:
        """Initializes the button for `reason_tag` in the single action row."""
        super().__init__(label=label, style=ButtonStyle.secondary, row=0)
        self.reason_tag: DownvoteReasonTag = reason_tag

    async def callback(self, interaction: Interaction[commands.Bot]) -> None:
        """Opens the modal for `other`, or saves and highlights a preset tag in place."""
        view = self.view
        if view is None or interaction.user is None:
            return
        if self.reason_tag == "other":
            await interaction.response.send_modal(modal=DownvoteReasonModal(parent_view=view))
            return
        await interaction.response.defer()
        try:
            result = await set_feedback_reason(
                reply_message_id=view.reply_message_id,
                user_id=interaction.user.id,
                reason_tag=self.reason_tag,
            )
            if not result.updated:
                await interaction.edit_original_message(
                    content="你已變更或取消這則評分", view=None
                )
                return
            view.apply_selection(reason_tag=result.reason_tag, reason_text=result.reason_text)
            logfire.info(
                "gen_reply feedback reason",
                reply_message_id=view.reply_message_id,
                source_message_id=result.source_message_id,
                model=result.model,
                effort=result.effort,
                route=result.route,
                user_id=interaction.user.id,
                reason_tag=result.reason_tag,
                reason_chars=0,
                is_asker=result.is_asker,
            )
            await interaction.edit_original_message(content=view.status_text(), view=view)
        except Exception as exc:
            # Broad on purpose: deferred button presses must surface a notice instead of hanging.
            logfire.error(
                "Failed to save downvote reason tag",
                reply_message_id=view.reply_message_id,
                user_id=interaction.user.id,
                reason_tag=self.reason_tag,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            with contextlib.suppress(HTTPException):
                await interaction.edit_original_message(content="記錄原因時發生錯誤，請稍後再試")


class DownvoteReasonView(LoggedView):
    """Ephemeral 5-button reason picker attached after a thumbs-down vote."""

    interaction_failure_log: ClassVar[str] = "Downvote reason button failed"

    def __init__(self, reply_message_id: int, changed: bool = False) -> None:
        """Initializes the single-row reason picker for `reply_message_id`.

        Args:
            reply_message_id (int): Discord message id the user downvoted.
            changed (bool): Whether this press switched an earlier thumbs-up to thumbs-down.
        """
        super().__init__(timeout=FEEDBACK_REASON_VIEW_TIMEOUT_SECONDS)
        self.reply_message_id = reply_message_id
        self.changed = changed
        self.selected_tag: DownvoteReasonTag | None = None
        self.selected_text: str | None = None
        self._reason_buttons: list[_ReasonButton] = []
        for reason_tag, label in DOWNVOTE_REASONS:
            btn = _ReasonButton(reason_tag=reason_tag, label=label)
            self._reason_buttons.append(btn)
            self.add_item(item=btn)

    def apply_selection(
        self, reason_tag: DownvoteReasonTag | None, reason_text: str | None
    ) -> None:
        """Updates internal selection state and highlights the active button."""
        self.selected_tag = reason_tag
        self.selected_text = reason_text
        for btn in self._reason_buttons:
            btn.style = (
                ButtonStyle.primary if btn.reason_tag == reason_tag else ButtonStyle.secondary
            )

    def status_text(self) -> str:
        """Returns the ephemeral message text reflecting the current downvote and reason."""
        prefix = "已改為 👎" if self.changed else "已記錄 👎"
        if self.selected_tag is None:
            return f"{prefix}，想順手點個原因嗎?（選填）"
        if self.selected_tag == "other" and self.selected_text:
            return f"{prefix}（其他：{self.selected_text}）"
        return f"{prefix}（{REASON_LABELS[self.selected_tag]}）"


class ReplyFeedbackView(LoggedView):
    """Persistent thumbs-up / thumbs-down action row attached to finalized replies."""

    interaction_failure_log: ClassVar[str] = "Reply feedback button failed"

    def __init__(self) -> None:
        """Initializes the persistent view (`timeout=None`) with stable `custom_id`s."""
        # `prevent_update=False` keeps nextcord from storing one immortal ViewStore entry per
        # delivered reply; `bot.add_view(ReplyFeedbackView())` registers the single global handler.
        super().__init__(timeout=None, prevent_update=False)

    async def _handle_vote(
        self, interaction: Interaction[commands.Bot], vote: FeedbackVote
    ) -> None:
        """Defers ephemerally, toggles the vote in `reply.db`, and edits the ephemeral notice."""
        if interaction.message is None or interaction.user is None:
            return
        reply_message_id = interaction.message.id
        # `with_message=True` creates a new ephemeral reply instead of editing the public answer.
        await interaction.response.defer(ephemeral=True, with_message=True)
        try:
            result = await toggle_feedback_vote(
                reply_message_id=reply_message_id,
                user_id=interaction.user.id,
                user_name=interaction.user.name,
                vote=vote,
            )
            if result is None:
                await interaction.edit_original_message(content="這則回覆已超過可評分範圍")
                return
            logfire.info(
                "gen_reply feedback",
                reply_message_id=reply_message_id,
                source_message_id=result.source_message_id,
                model=result.model,
                effort=result.effort,
                route=result.route,
                user_id=interaction.user.id,
                vote=vote,
                action=result.action,
                is_asker=result.is_asker,
            )
            if result.action == "clear":
                await interaction.edit_original_message(content="已取消評分")
                return
            if vote == "up":
                label = "已改為 👍" if result.action == "change" else "已記錄 👍"
                await interaction.edit_original_message(content=label)
                return
            reason_view = DownvoteReasonView(
                reply_message_id=reply_message_id, changed=result.action == "change"
            )
            await interaction.edit_original_message(
                content=reason_view.status_text(), view=reason_view
            )
        except Exception as exc:
            # Broad on purpose: deferred button callbacks must finish with a notice on failure.
            logfire.error(
                "Failed to record reply feedback",
                reply_message_id=reply_message_id,
                user_id=interaction.user.id,
                vote=vote,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            with contextlib.suppress(HTTPException):
                await interaction.edit_original_message(content="記錄評分時發生錯誤，請稍後再試")

    @button(emoji="👍", style=ButtonStyle.secondary, custom_id=FEEDBACK_UP_CUSTOM_ID, row=0)
    async def vote_up(
        self, _button: Button["ReplyFeedbackView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Records, switches to, or cancels a thumbs-up vote."""
        await self._handle_vote(interaction=interaction, vote="up")

    @button(emoji="👎", style=ButtonStyle.secondary, custom_id=FEEDBACK_DOWN_CUSTOM_ID, row=0)
    async def vote_down(
        self, _button: Button["ReplyFeedbackView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Records, switches to, or cancels a thumbs-down vote."""
        await self._handle_vote(interaction=interaction, vote="down")
