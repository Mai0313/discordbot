"""Button views for deciding public loan requests."""

import contextlib

import nextcord
from nextcord import Message, ButtonStyle, Interaction
from nextcord.ui import View, Button
from nextcord.ext import commands

from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.economy import LOAN_PROPOSAL_TIMEOUT_SECONDS
from discordbot.cogs._economy.embeds import (
    REPAY_COLOR,
    CENTRAL_BANK_COLOR,
    build_error_embed,
    build_simple_embed,
    build_credit_approved_embed,
    build_central_bank_approved_embed,
)
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import schedule_public_message_delete
from discordbot.cogs._economy.database import (
    get_central_banker,
    accept_loan_proposal,
    cancel_loan_proposal,
    reject_loan_proposal,
    reject_expired_loan_proposal,
)
from discordbot.cogs._economy.interactions import edit_response_embed, send_ephemeral_response


class LoanDecisionViewBase(View):
    """Shared cleanup behavior for public loan-decision views."""

    message: Message | None

    def _schedule_cleanup(self, interaction: Interaction[commands.Bot] | None = None) -> None:
        """Schedules the public request message for cleanup after a terminal state."""
        message = self.message or getattr(interaction, "message", None)
        if message is None:
            return
        user_name = None
        if interaction is not None and interaction.user is not None:
            user_name = interaction.user.name
        schedule_public_message_delete(message=message, user_name=user_name)


class CentralBankLoanDecisionView(LoanDecisionViewBase):
    """Button controls for deciding a public central-bank loan request."""

    def __init__(
        self,
        bot: commands.Bot,
        proposal_id: int,
        creator_id: int,
        allow_self_approval: bool = False,
    ) -> None:
        """Initializes a decision view for one proposal."""
        super().__init__(timeout=LOAN_PROPOSAL_TIMEOUT_SECONDS)
        self.bot = bot
        self.proposal_id = proposal_id
        self.creator_id = creator_id
        self.allow_self_approval = allow_self_approval
        self.message: Message | None = None

    async def on_timeout(self) -> None:
        """Rejects a stale central-bank request and cleans up its message."""
        proposal = await reject_expired_loan_proposal(proposal_id=self.proposal_id)
        if proposal is None or self.message is None:
            return
        self.stop()
        embed = build_simple_embed(
            title="🏛️ 央行申請已逾時",
            description="### 申請已逾時，自動拒絕",
            color=CENTRAL_BANK_COLOR,
        )
        with contextlib.suppress(Exception):
            await self.message.edit(
                embed=embed,
                view=None,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=self.message),
            )
        self._schedule_cleanup()

    async def _send_permission_denied(self, interaction: Interaction[commands.Bot]) -> None:
        """Replies privately when a non-banker clicks a decision button."""
        embed = build_error_embed(
            title="權限不足", description="### 只有央行成員可以處理央行借款申請"
        )
        await send_ephemeral_response(interaction=interaction, embed=embed)

    async def _is_central_banker(self, interaction: Interaction[commands.Bot]) -> bool:
        """Returns whether the clicking user can decide central-bank proposals."""
        if interaction.user is None:
            return False
        return await get_central_banker(user_id=interaction.user.id)

    def _central_bank_exclude_user_ids(self) -> tuple[int, ...]:
        """Returns bot-owned account IDs excluded from central-bank capacity."""
        return (self.bot.user.id,) if self.bot.user is not None else ()

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
        """Approves the central-bank request when clicked by a central banker."""
        if interaction.user is None:
            return
        if not await self._is_central_banker(interaction=interaction):
            await self._send_permission_denied(interaction=interaction)
            return

        banker_avatar_url = await guild_avatar_url(
            user=interaction.user, guild=getattr(interaction, "guild", None)
        )
        result = await accept_loan_proposal(
            proposal_id=self.proposal_id,
            actor_id=interaction.user.id,
            actor_name=interaction.user.name,
            actor_avatar_url=banker_avatar_url,
            is_central_banker=True,
            central_bank_exclude_user_ids=self._central_bank_exclude_user_ids(),
            allow_central_bank_self_approval=self.allow_self_approval,
        )
        if result is None:
            embed = build_error_embed(
                title="批准失敗",
                description="### 申請不存在、已處理、自我批准未開放，或央行額度不足",
            )
            await send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = build_central_bank_approved_embed(
            result=result, approver_mention=interaction.user.mention
        )
        self.stop()
        await edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)

    @nextcord.ui.button(
        label="拒絕", emoji="✖️", style=ButtonStyle.danger, custom_id="central_bank:reject", row=0
    )
    async def reject(
        self,
        _button: Button["CentralBankLoanDecisionView"],
        interaction: Interaction[commands.Bot],
    ) -> None:
        """Rejects the central-bank request when clicked by a central banker."""
        if interaction.user is None:
            return
        if not await self._is_central_banker(interaction=interaction):
            await self._send_permission_denied(interaction=interaction)
            return

        proposal = await reject_loan_proposal(
            proposal_id=self.proposal_id, actor_id=interaction.user.id, is_central_banker=True
        )
        if proposal is None:
            embed = build_error_embed(
                title="拒絕失敗", description="### 申請不存在、已處理，或你沒有權限拒絕"
            )
            await send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = build_simple_embed(
            title="🏛️ 央行申請已拒絕",
            description=f"### 央行借款申請已關閉\n處理人 {interaction.user.mention}",
            color=CENTRAL_BANK_COLOR,
        )
        self.stop()
        await edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)

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
        if interaction.user is None:
            return
        if interaction.user.id != self.creator_id:
            embed = build_error_embed(
                title="權限不足", description="### 只有申請發起者可以取消央行借款申請"
            )
            await send_ephemeral_response(interaction=interaction, embed=embed)
            return

        proposal = await cancel_loan_proposal(
            proposal_id=self.proposal_id, actor_id=interaction.user.id
        )
        if proposal is None:
            embed = build_error_embed(
                title="取消失敗", description="### 申請不存在、已處理，或你不是發起者"
            )
            await send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = build_simple_embed(
            title="🏛️ 央行申請已取消",
            description=f"### 央行借款申請已關閉\n發起者 {interaction.user.mention}",
            color=CENTRAL_BANK_COLOR,
        )
        self.stop()
        await edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)


class CreditLoanDecisionView(LoanDecisionViewBase):
    """Button controls for deciding a public personal credit request."""

    def __init__(self, proposal_id: int, lender_id: int, creator_id: int) -> None:
        """Initializes a decision view for one personal credit proposal."""
        super().__init__(timeout=LOAN_PROPOSAL_TIMEOUT_SECONDS)
        self.proposal_id = proposal_id
        self.lender_id = lender_id
        self.creator_id = creator_id
        self.message: Message | None = None

    async def on_timeout(self) -> None:
        """Rejects a stale personal credit request and cleans up its message."""
        proposal = await reject_expired_loan_proposal(proposal_id=self.proposal_id)
        if proposal is None or self.message is None:
            return
        self.stop()
        embed = build_simple_embed(
            title="信貸申請已逾時", description="### 申請已逾時，自動拒絕", color=REPAY_COLOR
        )
        with contextlib.suppress(Exception):
            await self.message.edit(
                embed=embed,
                view=None,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=self.message),
            )
        self._schedule_cleanup()

    async def _send_permission_denied(
        self, interaction: Interaction[commands.Bot], description: str
    ) -> None:
        """Replies privately when a user clicks a button they cannot use."""
        embed = build_error_embed(title="權限不足", description=description)
        await send_ephemeral_response(interaction=interaction, embed=embed)

    async def _require_lender(self, interaction: Interaction[commands.Bot]) -> bool:
        """Returns whether the clicking user is the requested lender."""
        if interaction.user is None:
            return False
        if interaction.user.id == self.lender_id:
            return True
        await self._send_permission_denied(
            interaction=interaction, description="### 只有指定貸方可以處理這筆信貸申請"
        )
        return False

    @nextcord.ui.button(
        label="批准", emoji="✅", style=ButtonStyle.success, custom_id="credit:approve", row=0
    )
    async def approve(
        self, _button: Button["CreditLoanDecisionView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Approves the personal credit request when clicked by the lender."""
        if interaction.user is None or not await self._require_lender(interaction=interaction):
            return

        lender_avatar_url = await guild_avatar_url(
            user=interaction.user, guild=getattr(interaction, "guild", None)
        )
        result = await accept_loan_proposal(
            proposal_id=self.proposal_id,
            actor_id=interaction.user.id,
            actor_name=interaction.user.name,
            actor_avatar_url=lender_avatar_url,
        )
        if result is None:
            embed = build_error_embed(
                title="批准失敗",
                description="### 申請不存在、已處理、不是指定貸方，或貸方餘額不足",
            )
            await send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = build_credit_approved_embed(
            result=result,
            approver_mention=interaction.user.mention,
            lender_avatar_url=lender_avatar_url,
        )
        self.stop()
        await edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)

    @nextcord.ui.button(
        label="拒絕", emoji="✖️", style=ButtonStyle.danger, custom_id="credit:reject", row=0
    )
    async def reject(
        self, _button: Button["CreditLoanDecisionView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Rejects the personal credit request when clicked by the lender."""
        if interaction.user is None or not await self._require_lender(interaction=interaction):
            return

        proposal = await reject_loan_proposal(
            proposal_id=self.proposal_id, actor_id=interaction.user.id
        )
        if proposal is None:
            embed = build_error_embed(
                title="拒絕失敗", description="### 申請不存在、已處理，或你不是指定貸方"
            )
            await send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = build_simple_embed(
            title="信貸申請已拒絕",
            description=f"### 信貸申請已關閉\n處理人 {interaction.user.mention}",
            color=REPAY_COLOR,
        )
        self.stop()
        await edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)

    @nextcord.ui.button(
        label="取消", emoji="🚫", style=ButtonStyle.secondary, custom_id="credit:cancel", row=0
    )
    async def cancel(
        self, _button: Button["CreditLoanDecisionView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Cancels the personal credit request when clicked by its creator."""
        if interaction.user is None:
            return
        if interaction.user.id != self.creator_id:
            await self._send_permission_denied(
                interaction=interaction, description="### 只有申請發起者可以取消這筆信貸申請"
            )
            return

        proposal = await cancel_loan_proposal(
            proposal_id=self.proposal_id, actor_id=interaction.user.id
        )
        if proposal is None:
            embed = build_error_embed(
                title="取消失敗", description="### 申請不存在、已處理，或你不是發起者"
            )
            await send_ephemeral_response(interaction=interaction, embed=embed)
            return

        embed = build_simple_embed(
            title="信貸申請已取消",
            description=f"### 信貸申請已關閉\n發起者 {interaction.user.mention}",
            color=REPAY_COLOR,
        )
        self.stop()
        await edit_response_embed(interaction=interaction, embed=embed)
        self._schedule_cleanup(interaction=interaction)
