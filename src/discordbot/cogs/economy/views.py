"""Decision buttons that sit under a public loan request until it is settled.

`cog.py` posts the request embed and hands that message to one of these views, so everything
after the slash command returns happens here. `CentralBankLoanDecisionView` backs
`/central_bank borrow` and `CreditLoanDecisionView` backs `/credit borrow`; both carry the same
three buttons (批准 / 拒絕 / 取消) over a different permission model:

- Central bank: approve and reject are open to any central banker (the `is_central_banker` DB
  flag, set offline and unrelated to Discord admin). Approving one's own request additionally
  needs `ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL`, a local-testing switch that stays off in
  production, which is why the flag is carried in from the cog rather than read here.
- Personal credit: approve and reject belong to the single member the borrower named as lender.
- Cancel belongs to whoever opened the request, in both.

That gate is only the fast half. Every decision is settled by `services/economy/database.py`,
which re-checks the same permission inside its own transaction and also answers for what a view
cannot see: a proposal already decided, one whose window expired while the buttons sat there, a
lender who has since spent the money, central-bank capacity that no longer covers the amount.
All of those come back as None, so each callback answers with one ephemeral embed naming every
possible reason instead of claiming to know which one fired.

A settled or timed-out request is rewritten in place with its controls dropped, then handed to
`schedule_public_message_delete` so the channel does not collect dead request cards. The view
keeps its own handle on that message because `on_timeout` has no interaction to reach it
through; `send_loan_request_followup` writes the message onto the view for exactly that reason.
"""

import contextlib

import nextcord
from nextcord import Message, ButtonStyle, Interaction
from nextcord.ui import View, Button
from nextcord.ext import commands

from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.economy import LOAN_PROPOSAL_TIMEOUT_SECONDS
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
    get_central_banker,
    accept_loan_proposal,
    cancel_loan_proposal,
    reject_loan_proposal,
    reject_expired_loan_proposal,
)
from discordbot.utils.interaction_responses import edit_response_embed, send_ephemeral_response


class LoanDecisionViewBase(View):
    """Shared cleanup behavior for public loan-decision views.

    Both subclasses set `message` in `__init__` and `send_loan_request_followup` overwrites it
    with the posted request. The annotation lives here so `_schedule_cleanup` can read it off
    the base instead of being duplicated into each view.
    """

    message: Message | None

    def _schedule_cleanup(self, interaction: Interaction[commands.Bot] | None = None) -> None:
        """Hands the public request message to the delayed-deletion scheduler.

        Prefers the message recorded at send time, which is the only handle `on_timeout` has,
        and falls back to the message the button was clicked on. Does nothing when neither
        exists, so a request whose message was never recorded is simply left in the channel
        rather than raising into a component callback.

        Args:
            interaction (Interaction[commands.Bot] | None): The click that settled the request,
                whose user is recorded with the pending deletion. None on timeout, where there
                is no actor.
        """
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
        """Initializes a decision view for one central-bank proposal.

        The view's timeout and the database's expiry check both run on
        `LOAN_PROPOSAL_TIMEOUT_SECONDS`, so the buttons stop working around the moment a
        decision would be refused as stale anyway.

        Args:
            bot (commands.Bot): The running bot, whose own account is excluded from the
                central-bank lending pool on approval.
            proposal_id (int): The pending proposal these buttons decide.
            creator_id (int): The borrower who opened the request, and the only user allowed to
                cancel it.
            allow_self_approval (bool): Whether that borrower may also approve it. Carried in
                from the cog's config because it is a local-testing switch, unset in production.
        """
        super().__init__(timeout=LOAN_PROPOSAL_TIMEOUT_SECONDS)
        self.bot = bot
        self.proposal_id = proposal_id
        self.creator_id = creator_id
        self.allow_self_approval = allow_self_approval
        self.message: Message | None = None

    async def on_timeout(self) -> None:
        """Rejects the stale central-bank request and retires its message.

        Rejecting in the database comes first: a proposal that is no longer pending was already
        settled by a button, so this returns without touching the message that callback wrote.
        The rewrite is best-effort and the cleanup runs either way, so a request whose message a
        moderator deleted still leaves the proposal closed rather than pending forever.
        """
        proposal = await reject_expired_loan_proposal(proposal_id=self.proposal_id)
        if proposal is None or self.message is None:
            return
        self.stop()
        embed = build_simple_embed(
            title="🏛️ 央行申請已逾時",
            description="### 申請已逾時，自動拒絕",
            color=CENTRAL_BANK_COLOR,
        )
        # Broad: this runs in a detached timeout task, and every way the edit can fail (deleted
        # message, lost permissions, a 5xx) leaves the proposal correctly rejected already.
        with contextlib.suppress(Exception):
            await self.message.edit(
                embed=embed,
                view=None,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=self.message),
            )
        self._schedule_cleanup()

    async def _send_permission_denied(self, interaction: Interaction[commands.Bot]) -> None:
        """Tells a non-banker privately that the request is not theirs to decide.

        Ephemeral, so a bystander's failed click never adds noise under a request everyone in
        the channel can see.

        Args:
            interaction (Interaction[commands.Bot]): The rejected click, which has not been
                answered yet.
        """
        embed = build_error_embed(
            title="權限不足", description="### 只有央行成員可以處理央行借款申請"
        )
        await send_ephemeral_response(interaction=interaction, embed=embed)

    async def _is_central_banker(self, interaction: Interaction[commands.Bot]) -> bool:
        """Whether the clicking user carries the central-banker flag.

        Reads the flag per click rather than caching it on the view, so revoking someone's
        banker status takes effect on requests already sitting in the channel.

        Args:
            interaction (Interaction[commands.Bot]): The click to authorize.

        Returns:
            True when the user is a central banker, False for an unknown user.
        """
        if interaction.user is None:
            return False
        return await get_central_banker(user_id=interaction.user.id)

    def _central_bank_exclude_user_ids(self) -> tuple[int, ...]:
        """The accounts left out of the balance pool that backs central-bank capacity.

        Only the bot's own wallet. Capacity is the sum of every positive balance, so leaving the
        bot in would let its own winnings enlarge the pool it lends against.

        Returns:
            The bot's user id, or an empty tuple before the gateway has assigned one.
        """
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
        """Approves the central-bank request and opens the loan contract.

        A non-banker gets an ephemeral refusal and nothing is written. `accept_loan_proposal` is
        the authority and re-checks the same rules inside its own write lock, so its None covers
        four outcomes at once — proposal gone, already decided, self-approval closed, capacity
        below the amount — which is why the failure embed names them all instead of guessing
        which one fired. On success the view stops listening before the request message is
        rewritten into the approval record, and the message is queued for deletion.

        Args:
            _button (Button["CentralBankLoanDecisionView"]): The clicked button, unused.
            interaction (Interaction[commands.Bot]): The click to authorize and answer.
        """
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
        """Turns the central-bank request down on a banker's click.

        Rejection is a banker's call rather than the borrower's, so the same gate as approval
        applies. `reject_loan_proposal` runs its own permission check as well and answers None
        for a proposal that is gone, already decided, or expired between the click and the
        write.

        Args:
            _button (Button["CentralBankLoanDecisionView"]): The clicked button, unused.
            interaction (Interaction[commands.Bot]): The click to authorize and answer.
        """
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
        """Withdraws the central-bank request on its creator's click.

        The one button the borrower owns, so it carries its own refusal wording rather than the
        banker-flavored `_send_permission_denied`. `cancel_loan_proposal` re-checks the creator
        itself and answers None for a proposal that is gone, already decided, or expired between
        the click and the write, so a click that lost that race gets an ephemeral failure rather
        than a cancellation embed over an approved loan.

        Args:
            _button (Button["CentralBankLoanDecisionView"]): The clicked button, unused.
            interaction (Interaction[commands.Bot]): The click to authorize and answer.
        """
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
        """Initializes a decision view for one personal credit proposal.

        The view's timeout and the database's expiry check both run on
        `LOAN_PROPOSAL_TIMEOUT_SECONDS`, so the buttons stop working around the moment a
        decision would be refused as stale anyway. No bot handle is needed here: a personal loan
        moves money between two wallets and never reads the central-bank pool the bot's own
        balance is excluded from.

        Args:
            proposal_id (int): The pending proposal these buttons decide.
            lender_id (int): The member the borrower asked to lend, and the only user allowed to
                approve or reject.
            creator_id (int): The borrower who opened the request, and the only user allowed to
                cancel it.
        """
        super().__init__(timeout=LOAN_PROPOSAL_TIMEOUT_SECONDS)
        self.proposal_id = proposal_id
        self.lender_id = lender_id
        self.creator_id = creator_id
        self.message: Message | None = None

    async def on_timeout(self) -> None:
        """Rejects the stale personal credit request and retires its message.

        Rejecting in the database comes first: a proposal that is no longer pending was already
        settled by a button, so this returns without touching the message that callback wrote.
        The rewrite is best-effort and the cleanup runs either way, so a request whose message a
        moderator deleted still leaves the proposal closed rather than pending forever.
        """
        proposal = await reject_expired_loan_proposal(proposal_id=self.proposal_id)
        if proposal is None or self.message is None:
            return
        self.stop()
        embed = build_simple_embed(
            title="信貸申請已逾時", description="### 申請已逾時，自動拒絕", color=REPAY_COLOR
        )
        # Broad: this runs in a detached timeout task, and every way the edit can fail (deleted
        # message, lost permissions, a 5xx) leaves the proposal correctly rejected already.
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
        """Tells a user privately that the button they pressed is not theirs.

        Ephemeral, so a bystander's failed click never adds noise under a request everyone in
        the channel can see. The description is a parameter because approve/reject and cancel
        are gated on different people and each says who.

        Args:
            interaction (Interaction[commands.Bot]): The rejected click, which has not been
                answered yet.
            description (str): The refusal body, already carrying its Markdown heading.
        """
        embed = build_error_embed(title="權限不足", description=description)
        await send_ephemeral_response(interaction=interaction, embed=embed)

    async def _require_lender(self, interaction: Interaction[commands.Bot]) -> bool:
        """Whether the clicking user is the named lender, refusing them privately if not.

        Unlike the central-bank gate this answers the interaction on the way out, so a caller
        that gets False must return without responding again.

        Args:
            interaction (Interaction[commands.Bot]): The click to authorize.

        Returns:
            True when the clicker is the lender the borrower named.
        """
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
        """Approves the request, debiting the lender and crediting the borrower.

        The lender's guild avatar is resolved before the write because it is both stored as the
        contract's lender identity and shown as the approval embed's thumbnail.
        `accept_loan_proposal` moves both wallets in one transaction and re-checks everything,
        so its None covers proposal gone, already decided, a clicker who is not the lender, and
        a lender whose balance no longer covers the amount, which is why the failure embed names
        them all instead of guessing which one fired.

        Args:
            _button (Button["CreditLoanDecisionView"]): The clicked button, unused.
            interaction (Interaction[commands.Bot]): The click to authorize and answer.
        """
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
        """Turns the request down on the named lender's click.

        `reject_loan_proposal` re-checks the lender itself and answers None for a proposal that
        is gone, already decided, or expired between the click and the write, which is what the
        failure embed's three reasons correspond to.

        Args:
            _button (Button["CreditLoanDecisionView"]): The clicked button, unused.
            interaction (Interaction[commands.Bot]): The click to authorize and answer.
        """
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
        """Withdraws the request on its creator's click.

        Gated on the borrower rather than the lender, so it checks `creator_id` directly instead
        of going through `_require_lender`. `cancel_loan_proposal` re-checks the creator itself
        and answers None for a proposal that is gone, already decided, or expired between the
        click and the write, so a click that lost that race gets an ephemeral failure rather
        than a cancellation embed over an approved loan.

        Args:
            _button (Button["CreditLoanDecisionView"]): The clicked button, unused.
            interaction (Interaction[commands.Bot]): The click to authorize and answer.
        """
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
