"""Button views for deciding public loan requests."""

from typing import ClassVar, cast

import nextcord
from nextcord import Embed, ButtonStyle, Interaction
from nextcord.ui import Button
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
from discordbot.utils.message_cleanup import PressBoundView, edit_or_report
from discordbot.services.economy.database import (
    accept_loan_proposal,
    cancel_loan_proposal,
    reject_loan_proposal,
    reject_expired_loan_proposal,
)
from discordbot.utils.interaction_responses import (
    edit_response_embed,
    send_ephemeral_notice,
    send_private_followup,
)


def is_guild_admin(interaction: Interaction[commands.Bot]) -> bool:
    """Returns whether the interacting user holds administrator permission in the guild.

    Reads `interaction.permissions` from the Discord payload (requires neither the members
    intent nor a cached guild); returns False in DMs where permissions are empty.

    Args:
        interaction (Interaction[commands.Bot]): Discord interaction to inspect.
    """
    return interaction.permissions.administrator


class LoanDecisionViewBase(PressBoundView):
    """Shared buttons and terminal behavior for public loan-decision views.

    Subclasses define `CUSTOM_ID_PREFIX`, panel wording/color, `_approved_embed`, and
    `_may_decide`. Write buttons defer before DB calls, then edit the panel or send a private
    followup.
    """

    CUSTOM_ID_PREFIX: ClassVar[str]
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
        """Initializes a decision view for one proposal and namespaces its button custom IDs.

        Args:
            proposal_id (int): Database ID of the pending loan proposal.
            creator_id (int): Discord user ID of the borrower who created the request.
        """
        super().__init__(timeout=LOAN_PROPOSAL_TIMEOUT_SECONDS)
        self.proposal_id = proposal_id
        self.creator_id = creator_id
        for action, button in (
            ("approve", self.approve),
            ("reject", self.reject),
            ("cancel", self.cancel),
        ):
            cast(
                'Button["LoanDecisionViewBase"]', button
            ).custom_id = f"{self.CUSTOM_ID_PREFIX}:{action}"

    async def _may_decide(self, interaction: Interaction[commands.Bot]) -> bool:
        """Returns whether the clicking user may approve or reject this request.

        Sends the permission notice before returning `False`; caller must not reply again.
        """
        raise NotImplementedError

    def _approved_embed(
        self, result: LoanProposalAcceptResult, approver_mention: str, approver_avatar_url: str
    ) -> Embed:
        """Builds the panel an approved request is edited into."""
        raise NotImplementedError

    def _approve_guild_id(self, interaction: Interaction[commands.Bot]) -> int | None:
        """Returns the guild ID passed to `accept_loan_proposal`, if guild-scoped."""
        del interaction
        return None

    def _allow_self_approval(self) -> bool:
        """Returns whether the borrower may approve their own request."""
        return False

    def _timeout_embed(self) -> Embed:
        """Builds the panel an expired request is edited into."""
        return build_simple_embed(
            title=self.TIMEOUT_TITLE,
            description="### 申請已逾時，自動拒絕",
            color=self.PANEL_COLOR,
        )

    async def on_timeout(self) -> None:
        """Rejects a stale request and cleans up its message."""
        proposal = await reject_expired_loan_proposal(proposal_id=self.proposal_id)
        if proposal is None or self.message is None:
            return
        self.stop()
        embed = self._timeout_embed()
        await edit_or_report(
            message=self.message,
            interaction=self.last_press,
            payload={
                "embed": embed,
                "view": None,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=self.message),
            },
            not_found_message="Loan request message gone before its timeout edit",
            forbidden_message="Discord refused the loan request's timeout edit",
            other_message="Loan request timeout edit failed",
            proposal_id=self.proposal_id,
            channel_id=self.message.channel.id,
            message_id=self.message.id,
        )
        self.schedule_delete()

    async def _notify_unexpected_error(
        self, interaction: Interaction[commands.Bot], content: str
    ) -> None:
        """Tells the clicker privately when a deferred button callback fails unexpectedly."""
        await send_ephemeral_notice(
            interaction=interaction,
            content=content,
            log_message="Failed to send loan decision error notice",
        )

    async def _finish_panel(
        self, interaction: Interaction[commands.Bot], embed: Embed, error_notice: str
    ) -> None:
        """Stops the view, replaces the panel with `embed`, and always schedules cleanup."""
        self.stop()
        try:
            await edit_response_embed(interaction=interaction, embed=embed)
        except Exception:
            # Broad on purpose: the panel outcome is settled; tell the clicker before LoggedView logs.
            await self._notify_unexpected_error(interaction=interaction, content=error_notice)
            raise
        finally:
            self.schedule_delete(interaction=interaction)

    async def _handle_proposal_write_error(
        self, interaction: Interaction[commands.Bot], error: Exception, error_notice: str
    ) -> bool:
        """Closes the panel on `LoanProposalExpiredError` (returning True), else notifies."""
        if isinstance(error, LoanProposalExpiredError):
            await self._finish_panel(
                interaction=interaction,
                embed=self._timeout_embed(),
                error_notice="申請已逾時，但更新面板失敗",
            )
            return True
        await self._notify_unexpected_error(interaction=interaction, content=error_notice)
        return False

    @nextcord.ui.button(
        label="批准", emoji="✅", style=ButtonStyle.success, custom_id="loan:approve", row=0
    )
    async def approve(
        self, _button: Button["LoanDecisionViewBase"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Approves the request when clicked by an authorized user.

        Args:
            _button (Button["LoanDecisionViewBase"]): Clicked approve button.
            interaction (Interaction[commands.Bot]): Component interaction from the click.
        """
        await self._handle_approve(interaction=interaction)

    @nextcord.ui.button(
        label="拒絕", emoji="✖️", style=ButtonStyle.danger, custom_id="loan:reject", row=0
    )
    async def reject(
        self, _button: Button["LoanDecisionViewBase"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Rejects the request when clicked by an authorized user.

        Args:
            _button (Button["LoanDecisionViewBase"]): Clicked reject button.
            interaction (Interaction[commands.Bot]): Component interaction from the click.
        """
        await self._handle_reject(interaction=interaction)

    @nextcord.ui.button(
        label="取消", emoji="🚫", style=ButtonStyle.secondary, custom_id="loan:cancel", row=0
    )
    async def cancel(
        self, _button: Button["LoanDecisionViewBase"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Cancels the request when clicked by its creator.

        Args:
            _button (Button["LoanDecisionViewBase"]): Clicked cancel button.
            interaction (Interaction[commands.Bot]): Component interaction from the click.
        """
        await self._handle_cancel(interaction=interaction)

    async def _handle_cancel(self, interaction: Interaction[commands.Bot]) -> None:
        """Cancels the request for its creator, and answers anyone else privately."""
        if interaction.user is None:
            return
        await interaction.response.defer()
        self.keep_press(interaction=interaction)
        if interaction.user.id != self.creator_id:
            embed = build_error_embed(title="權限不足", description=self.CANCEL_DENIED_NOTICE)
            await send_private_followup(interaction=interaction, embed=embed)
            return

        try:
            proposal = await cancel_loan_proposal(
                proposal_id=self.proposal_id, actor_id=interaction.user.id
            )
        # Broad on purpose: the deferred click needs a reply before LoggedView logs it.
        except Exception as exc:
            if await self._handle_proposal_write_error(
                interaction=interaction, error=exc, error_notice="取消失敗，請稍後再試"
            ):
                return
            raise
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
        await self._finish_panel(
            interaction=interaction, embed=embed, error_notice="申請已取消，但更新面板失敗"
        )

    async def _handle_approve(self, interaction: Interaction[commands.Bot]) -> None:
        """Approves the request if `_may_decide` passes, otherwise answers privately."""
        if interaction.user is None:
            return
        await interaction.response.defer()
        self.keep_press(interaction=interaction)
        if not await self._may_decide(interaction=interaction):
            return

        try:
            actor_avatar_url = await guild_avatar_url(
                user=interaction.user, guild=interaction.guild
            )
            result = await accept_loan_proposal(
                proposal_id=self.proposal_id,
                actor_id=interaction.user.id,
                actor_name=interaction.user.name,
                actor_avatar_url=actor_avatar_url,
                approver_is_guild_admin=self.APPROVER_IS_GUILD_ADMIN,
                guild_id=self._approve_guild_id(interaction=interaction),
                allow_central_bank_self_approval=self._allow_self_approval(),
            )
        # Broad on purpose: the deferred click needs a reply before LoggedView logs it.
        except Exception as exc:
            if await self._handle_proposal_write_error(
                interaction=interaction, error=exc, error_notice="批准失敗，請稍後再試"
            ):
                return
            raise
        if result is None:
            embed = build_error_embed(title="批准失敗", description=self.APPROVE_FAILED_NOTICE)
            await send_private_followup(interaction=interaction, embed=embed)
            return

        embed = self._approved_embed(
            result=result,
            approver_mention=interaction.user.mention,
            approver_avatar_url=actor_avatar_url,
        )
        await self._finish_panel(
            interaction=interaction, embed=embed, error_notice="申請已批准，但更新面板失敗"
        )

    async def _handle_reject(self, interaction: Interaction[commands.Bot]) -> None:
        """Rejects the request for whoever may decide it, and answers anyone else privately."""
        if interaction.user is None:
            return
        await interaction.response.defer()
        self.keep_press(interaction=interaction)
        if not await self._may_decide(interaction=interaction):
            return

        try:
            proposal = await reject_loan_proposal(
                proposal_id=self.proposal_id,
                actor_id=interaction.user.id,
                approver_is_guild_admin=self.APPROVER_IS_GUILD_ADMIN,
            )
        # Broad on purpose: the deferred click needs a reply before LoggedView logs it.
        except Exception as exc:
            if await self._handle_proposal_write_error(
                interaction=interaction, error=exc, error_notice="拒絕失敗，請稍後再試"
            ):
                return
            raise
        if proposal is None:
            embed = build_error_embed(title="拒絕失敗", description=self.REJECT_FAILED_NOTICE)
            await send_private_followup(interaction=interaction, embed=embed)
            return

        embed = build_simple_embed(
            title=self.REJECT_TITLE,
            description=f"{self.CLOSED_HEADING}\n處理人 {interaction.user.mention}",
            color=self.PANEL_COLOR,
        )
        await self._finish_panel(
            interaction=interaction, embed=embed, error_notice="申請已拒絕，但更新面板失敗"
        )


class CentralBankLoanDecisionView(LoanDecisionViewBase):
    """Button controls for deciding a public central-bank loan request."""

    CUSTOM_ID_PREFIX = "central_bank"
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
        self, proposal_id: int, creator_id: int, allow_self_approval: bool = False
    ) -> None:
        """Initializes a decision view for one proposal.

        Args:
            proposal_id (int): Database ID of the pending loan proposal.
            creator_id (int): Discord user ID of the borrower who created the request.
            allow_self_approval (bool): Whether the borrower may approve their own request.
        """
        super().__init__(proposal_id=proposal_id, creator_id=creator_id)
        self.allow_self_approval = allow_self_approval

    def _approve_guild_id(self, interaction: Interaction[commands.Bot]) -> int | None:
        """Returns the guild ID whose central-bank pool funds this request."""
        return interaction.guild_id

    def _allow_self_approval(self) -> bool:
        """Returns whether the borrower may approve their own central-bank request."""
        return self.allow_self_approval

    async def _may_decide(self, interaction: Interaction[commands.Bot]) -> bool:
        """Returns whether the clicking user is a server administrator here.

        Sends the permission notice before returning `False`; caller must not reply again.
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


class CreditLoanDecisionView(LoanDecisionViewBase):
    """Button controls for deciding a public personal credit request."""

    CUSTOM_ID_PREFIX = "credit"
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
        """Initializes a decision view for one personal credit proposal.

        Args:
            proposal_id (int): Database ID of the pending loan proposal.
            lender_id (int): Discord user ID of the target lender.
            creator_id (int): Discord user ID of the borrower who created the request.
        """
        super().__init__(proposal_id=proposal_id, creator_id=creator_id)
        self.lender_id = lender_id

    async def _may_decide(self, interaction: Interaction[commands.Bot]) -> bool:
        """Returns whether the clicking user is the requested lender.

        Sends the permission notice before returning `False`; caller must not reply again.
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
