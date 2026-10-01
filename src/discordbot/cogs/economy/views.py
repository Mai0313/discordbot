"""Button views for deciding public loan requests."""

from typing import ClassVar

import logfire
import nextcord
from nextcord import Embed, Message, NotFound, Forbidden, ButtonStyle, Interaction
from nextcord.ui import View, Button
from nextcord.ext import commands

from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.economy import (
    LOAN_PROPOSAL_TIMEOUT_SECONDS,
    LoanProposalAcceptResult,
    LoanProposalExpiredError,
)
from discordbot.cogs.economy.embeds import (
    REPAY_COLOR,
    CENTRAL_BANK_COLOR,
    build_error_embed,
    build_simple_embed,
    build_credit_approved_embed,
    build_central_bank_approved_embed,
)
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import schedule_public_message_delete
from discordbot.services.economy.database import (
    accept_loan_proposal,
    cancel_loan_proposal,
    reject_loan_proposal,
    reject_expired_loan_proposal,
)
from discordbot.utils.interaction_responses import edit_response_embed, send_private_followup


def central_bank_exclude_user_ids(bot: commands.Bot) -> tuple[int, ...]:
    """Returns bot-owned account IDs excluded from central-bank capacity."""
    return (bot.user.id,) if bot.user is not None else ()


def is_guild_admin(interaction: Interaction[commands.Bot]) -> bool:
    """Returns whether the interacting user administers the server it happened in.

    `Interaction.permissions` is what Discord itself resolved for this member in
    this channel and rides in the payload, so it needs neither the members intent
    nor a cached guild — which matters because a user-installed invocation in a
    server the bot was never added to has no cached guild at all. Outside a guild
    it is empty, which is what refuses a DM.
    """
    return interaction.permissions.administrator


class LoanDecisionViewBase(View):
    """Shared terminal behavior for public loan-decision views.

    A subclass declares the wording and color of its own panels below, its approved panel, and
    who may decide its requests; expiry, approval, rejection and the creator-only cancel then
    behave the same for both.

    Every button that writes defers first, then answers by editing the panel or with a private
    followup. The token dies three seconds after the click while a SQLite writer waits out lock
    contention for longer, so deciding the response from the result lets an approval — the
    moment a lender is debited, or the central bank mints — commit and then fail to say so,
    leaving the buttons up over balances that have already moved. A component defers without a
    thinking message, so the panel is untouched until the edit lands.
    """

    PANEL_COLOR: ClassVar[int]
    TIMEOUT_TITLE: ClassVar[str]
    CANCEL_TITLE: ClassVar[str]
    REJECT_TITLE: ClassVar[str]
    CLOSED_HEADING: ClassVar[str]
    CANCEL_DENIED_NOTICE: ClassVar[str]
    APPROVE_FAILED_NOTICE: ClassVar[str]
    REJECT_FAILED_NOTICE: ClassVar[str]
    # Whether passing `_may_decide` means the clicker administers this server.
    APPROVER_IS_GUILD_ADMIN: ClassVar[bool]

    def __init__(self, proposal_id: int, creator_id: int) -> None:
        """Initializes a decision view for one proposal."""
        super().__init__(timeout=LOAN_PROPOSAL_TIMEOUT_SECONDS)
        self.proposal_id = proposal_id
        self.creator_id = creator_id
        self.message: Message | None = None

    async def _may_decide(self, interaction: Interaction[commands.Bot]) -> bool:
        """Returns whether the clicking user may approve or reject this request.

        Anyone else is answered with the permission notice before this returns `False`,
        so the caller must not reply again.
        """
        raise NotImplementedError

    def _approved_embed(
        self, result: LoanProposalAcceptResult, approver_mention: str, approver_avatar_url: str
    ) -> Embed:
        """Builds the panel an approved request is edited into."""
        raise NotImplementedError

    def _timeout_embed(self) -> Embed:
        """Builds the panel an expired request is edited into."""
        return build_simple_embed(
            title=self.TIMEOUT_TITLE,
            description="### 申請已逾時，自動拒絕",
            color=self.PANEL_COLOR,
        )

    def _schedule_cleanup(self, interaction: Interaction[commands.Bot] | None = None) -> None:
        """Schedules the public request message for cleanup after a terminal state."""
        message = self.message or getattr(interaction, "message", None)
        if message is None:
            return
        user_name = None
        if interaction is not None and interaction.user is not None:
            user_name = interaction.user.name
        schedule_public_message_delete(message=message, user_name=user_name)

    async def on_timeout(self) -> None:
        """Rejects a stale request and cleans up its message."""
        proposal = await reject_expired_loan_proposal(proposal_id=self.proposal_id)
        if proposal is None or self.message is None:
            return
        self.stop()
        embed = self._timeout_embed()
        try:
            await self.message.edit(
                embed=embed,
                view=None,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=self.message),
            )
        except NotFound:
            logfire.info(
                "Loan request message gone before its timeout edit",
                proposal_id=self.proposal_id,
                channel_id=self.message.channel.id,
                message_id=self.message.id,
            )
        except Forbidden:
            # The panel is the command's followup, so this edit rides the command's token rather
            # than the channel; a refusal's stack is identical every time, so the ids are the
            # whole finding.
            logfire.warn(
                "Discord refused the loan request's timeout edit",
                proposal_id=self.proposal_id,
                channel_id=self.message.channel.id,
                message_id=self.message.id,
            )
        # Broad on purpose: the proposal is already rejected, a raise here would only reach
        # nextcord's timeout task, and the cleanup below must still be scheduled.
        except Exception as exc:
            logfire.warn(
                "Loan request timeout edit failed",
                proposal_id=self.proposal_id,
                channel_id=self.message.channel.id,
                message_id=self.message.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        self._schedule_cleanup()

    async def _handle_cancel(self, interaction: Interaction[commands.Bot]) -> None:
        """Cancels the request for its creator, and answers anyone else privately."""
        if interaction.user is None:
            return
        await interaction.response.defer()
        if interaction.user.id != self.creator_id:
            embed = build_error_embed(title="權限不足", description=self.CANCEL_DENIED_NOTICE)
            await send_private_followup(interaction=interaction, embed=embed)
            return

        try:
            proposal = await cancel_loan_proposal(
                proposal_id=self.proposal_id, actor_id=interaction.user.id
            )
        except LoanProposalExpiredError:
            self.stop()
            await edit_response_embed(interaction=interaction, embed=self._timeout_embed())
            self._schedule_cleanup(interaction=interaction)
            return
        if proposal is None:
            embed = build_error_embed(
                title="取消失敗", description="### 申請不存在、已處理，或你不是發起者"
            )
            await send_private_followup(interaction=interaction, embed=embed)
            return

        embed = build_simple_embed(
            title=self.CANCEL_TITLE,
            description=f"{self.CLOSED_HEADING}\n發起者 {interaction.user.mention}",
            color=self.PANEL_COLOR,
        )
        self.stop()
        await edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)

    async def _handle_approve(
        self,
        interaction: Interaction[commands.Bot],
        guild_id: int | None = None,
        central_bank_exclude_user_ids: tuple[int, ...] = (),
        allow_central_bank_self_approval: bool = False,
    ) -> None:
        """Approves the request for whoever may decide it, and answers anyone else privately.

        The central-bank arguments pass straight through to `accept_loan_proposal`.
        """
        if interaction.user is None:
            return
        await interaction.response.defer()
        if not await self._may_decide(interaction=interaction):
            return

        actor_avatar_url = await guild_avatar_url(user=interaction.user, guild=interaction.guild)
        try:
            result = await accept_loan_proposal(
                proposal_id=self.proposal_id,
                actor_id=interaction.user.id,
                actor_name=interaction.user.name,
                actor_avatar_url=actor_avatar_url,
                approver_is_guild_admin=self.APPROVER_IS_GUILD_ADMIN,
                guild_id=guild_id,
                central_bank_exclude_user_ids=central_bank_exclude_user_ids,
                allow_central_bank_self_approval=allow_central_bank_self_approval,
            )
        except LoanProposalExpiredError:
            self.stop()
            await edit_response_embed(interaction=interaction, embed=self._timeout_embed())
            self._schedule_cleanup(interaction=interaction)
            return
        if result is None:
            embed = build_error_embed(title="批准失敗", description=self.APPROVE_FAILED_NOTICE)
            await send_private_followup(interaction=interaction, embed=embed)
            return

        embed = self._approved_embed(
            result=result,
            approver_mention=interaction.user.mention,
            approver_avatar_url=actor_avatar_url,
        )
        self.stop()
        await edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)

    async def _handle_reject(self, interaction: Interaction[commands.Bot]) -> None:
        """Rejects the request for whoever may decide it, and answers anyone else privately."""
        if interaction.user is None:
            return
        await interaction.response.defer()
        if not await self._may_decide(interaction=interaction):
            return

        try:
            proposal = await reject_loan_proposal(
                proposal_id=self.proposal_id,
                actor_id=interaction.user.id,
                approver_is_guild_admin=self.APPROVER_IS_GUILD_ADMIN,
            )
        except LoanProposalExpiredError:
            self.stop()
            await edit_response_embed(interaction=interaction, embed=self._timeout_embed())
            self._schedule_cleanup(interaction=interaction)
            return
        if proposal is None:
            embed = build_error_embed(title="拒絕失敗", description=self.REJECT_FAILED_NOTICE)
            await send_private_followup(interaction=interaction, embed=embed)
            return

        embed = build_simple_embed(
            title=self.REJECT_TITLE,
            description=f"{self.CLOSED_HEADING}\n處理人 {interaction.user.mention}",
            color=self.PANEL_COLOR,
        )
        self.stop()
        await edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)


class CentralBankLoanDecisionView(LoanDecisionViewBase):
    """Button controls for deciding a public central-bank loan request."""

    PANEL_COLOR = CENTRAL_BANK_COLOR
    TIMEOUT_TITLE = "🏛️ 央行申請已逾時"
    CANCEL_TITLE = "🏛️ 央行申請已取消"
    REJECT_TITLE = "🏛️ 央行申請已拒絕"
    CLOSED_HEADING = "### 央行借款申請已關閉"
    CANCEL_DENIED_NOTICE = "### 只有申請發起者可以取消央行借款申請"
    APPROVE_FAILED_NOTICE = (
        "### 申請不存在、已處理、自我批准未開放，或額度不足（本伺服器或申請人自己的上限）"
    )
    REJECT_FAILED_NOTICE = "### 申請不存在、已處理，或你沒有權限拒絕"
    APPROVER_IS_GUILD_ADMIN = True

    def __init__(
        self,
        bot: commands.Bot,
        proposal_id: int,
        creator_id: int,
        allow_self_approval: bool = False,
    ) -> None:
        """Initializes a decision view for one proposal."""
        super().__init__(proposal_id=proposal_id, creator_id=creator_id)
        self.bot = bot
        self.allow_self_approval = allow_self_approval

    async def _may_decide(self, interaction: Interaction[commands.Bot]) -> bool:
        """Returns whether the clicking user is a server administrator here.

        Anyone else is answered with the permission notice before this returns `False`,
        so the caller must not reply again.
        """
        if is_guild_admin(interaction=interaction):
            return True
        embed = build_error_embed(
            title="權限不足", description="### 只有這個伺服器的管理員可以處理央行借款申請"
        )
        await send_private_followup(interaction=interaction, embed=embed)
        return False

    def _approved_embed(
        self, result: LoanProposalAcceptResult, approver_mention: str, approver_avatar_url: str
    ) -> Embed:
        """Builds the central-bank approval panel; it shows no approver avatar."""
        return build_central_bank_approved_embed(result=result, approver_mention=approver_mention)

    @nextcord.ui.button(
        label="批准",
        emoji="✅",
        style=ButtonStyle.success,
        custom_id="central_bank:approve",
        row=0,
    )
    async def approve(
        self,
        _button: Button["CentralBankLoanDecisionView"],
        interaction: Interaction[commands.Bot],
    ) -> None:
        """Approves the central-bank request when clicked by a server administrator."""
        await self._handle_approve(
            interaction=interaction,
            guild_id=interaction.guild_id,
            central_bank_exclude_user_ids=central_bank_exclude_user_ids(bot=self.bot),
            allow_central_bank_self_approval=self.allow_self_approval,
        )

    @nextcord.ui.button(
        label="拒絕", emoji="✖️", style=ButtonStyle.danger, custom_id="central_bank:reject", row=0
    )
    async def reject(
        self,
        _button: Button["CentralBankLoanDecisionView"],
        interaction: Interaction[commands.Bot],
    ) -> None:
        """Rejects the central-bank request when clicked by a server administrator."""
        await self._handle_reject(interaction=interaction)

    @nextcord.ui.button(
        label="取消",
        emoji="🚫",
        style=ButtonStyle.secondary,
        custom_id="central_bank:cancel",
        row=0,
    )
    async def cancel(
        self,
        _button: Button["CentralBankLoanDecisionView"],
        interaction: Interaction[commands.Bot],
    ) -> None:
        """Cancels the central-bank request when clicked by its creator."""
        await self._handle_cancel(interaction=interaction)


class CreditLoanDecisionView(LoanDecisionViewBase):
    """Button controls for deciding a public personal credit request."""

    PANEL_COLOR = REPAY_COLOR
    TIMEOUT_TITLE = "信貸申請已逾時"
    CANCEL_TITLE = "信貸申請已取消"
    REJECT_TITLE = "信貸申請已拒絕"
    CLOSED_HEADING = "### 信貸申請已關閉"
    CANCEL_DENIED_NOTICE = "### 只有申請發起者可以取消這筆信貸申請"
    APPROVE_FAILED_NOTICE = "### 申請不存在、已處理、不是指定貸方，或貸方餘額不足"
    REJECT_FAILED_NOTICE = "### 申請不存在、已處理，或你不是指定貸方"
    APPROVER_IS_GUILD_ADMIN = False

    def __init__(self, proposal_id: int, lender_id: int, creator_id: int) -> None:
        """Initializes a decision view for one personal credit proposal."""
        super().__init__(proposal_id=proposal_id, creator_id=creator_id)
        self.lender_id = lender_id

    async def _may_decide(self, interaction: Interaction[commands.Bot]) -> bool:
        """Returns whether the clicking user is the requested lender.

        Someone other than the lender is answered with the permission notice
        before this returns `False`, so the caller must not reply again.
        """
        if interaction.user is None:
            return False
        if interaction.user.id == self.lender_id:
            return True
        embed = build_error_embed(
            title="權限不足", description="### 只有指定貸方可以處理這筆信貸申請"
        )
        await send_private_followup(interaction=interaction, embed=embed)
        return False

    def _approved_embed(
        self, result: LoanProposalAcceptResult, approver_mention: str, approver_avatar_url: str
    ) -> Embed:
        """Builds the personal credit approval panel; the approver is the lender."""
        return build_credit_approved_embed(
            result=result, approver_mention=approver_mention, lender_avatar_url=approver_avatar_url
        )

    @nextcord.ui.button(
        label="批准", emoji="✅", style=ButtonStyle.success, custom_id="credit:approve", row=0
    )
    async def approve(
        self, _button: Button["CreditLoanDecisionView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Approves the personal credit request when clicked by the lender."""
        await self._handle_approve(interaction=interaction)

    @nextcord.ui.button(
        label="拒絕", emoji="✖️", style=ButtonStyle.danger, custom_id="credit:reject", row=0
    )
    async def reject(
        self, _button: Button["CreditLoanDecisionView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Rejects the personal credit request when clicked by the lender."""
        await self._handle_reject(interaction=interaction)

    @nextcord.ui.button(
        label="取消", emoji="🚫", style=ButtonStyle.secondary, custom_id="credit:cancel", row=0
    )
    async def cancel(
        self, _button: Button["CreditLoanDecisionView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Cancels the personal credit request when clicked by its creator."""
        await self._handle_cancel(interaction=interaction)
