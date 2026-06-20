"""Escalation and plan-approval button views for a deep-research thread.

Thin nextcord views: each button verifies the click came from the research owner,
then delegates to a `ResearchCogs` handler that does the deferring, UI disabling, and
background agent work. Every button carries an explicit `custom_id` (a missing one
trips Discord 50035 on nextcord 3.2.0).
"""

from typing import TYPE_CHECKING

import nextcord
from nextcord import ButtonStyle, Interaction
from nextcord.ui import View, Button

if TYPE_CHECKING:
    from discordbot.cogs.research import ResearchCogs

# The owner has a generous window to escalate a finished report or approve a plan.
ESCALATION_VIEW_TIMEOUT_SECONDS = 1800.0
PLAN_VIEW_TIMEOUT_SECONDS = 1800.0


async def _is_owner(*, interaction: Interaction, owner_id: int) -> bool:
    """Returns True for the owner's click; otherwise denies ephemerally and returns False."""
    if interaction.user is not None and interaction.user.id == owner_id:
        return True
    await interaction.response.send_message(
        content="這不是你開的研究,沒辦法幫你操作喔", ephemeral=True
    )
    return False


class ResultEscalationView(View):
    """Buttons under an Antigravity report to escalate to Deep Research / Max."""

    def __init__(self, *, cog: "ResearchCogs", owner_id: int, max_enabled: bool) -> None:
        super().__init__(timeout=ESCALATION_VIEW_TIMEOUT_SECONDS)
        self.cog = cog
        self.owner_id = owner_id
        # Presence-based: drop the Max button entirely when the tier is disabled.
        if not max_enabled:
            self.remove_item(self.escalate_max)

    @nextcord.ui.button(
        label="升級 Deep Research", style=ButtonStyle.primary, custom_id="research:escalate_dr"
    )
    async def escalate_dr(self, _button: Button, interaction: Interaction) -> None:
        if not await _is_owner(interaction=interaction, owner_id=self.owner_id):
            return
        await self.cog.on_escalate(interaction=interaction, view=self, max_tier=False)

    @nextcord.ui.button(
        label="Deep Research Max", style=ButtonStyle.danger, custom_id="research:escalate_max"
    )
    async def escalate_max(self, _button: Button, interaction: Interaction) -> None:
        if not await _is_owner(interaction=interaction, owner_id=self.owner_id):
            return
        await self.cog.on_escalate(interaction=interaction, view=self, max_tier=True)


class PlanApprovalView(View):
    """Buttons under a proposed Deep Research plan: accept and run, or modify."""

    def __init__(
        self, *, cog: "ResearchCogs", owner_id: int, plan_interaction_id: str, agent: str
    ) -> None:
        super().__init__(timeout=PLAN_VIEW_TIMEOUT_SECONDS)
        self.cog = cog
        self.owner_id = owner_id
        self.plan_interaction_id = plan_interaction_id
        self.agent = agent

    @nextcord.ui.button(
        label="接受並開始", style=ButtonStyle.success, custom_id="research:plan_accept"
    )
    async def accept(self, _button: Button, interaction: Interaction) -> None:
        if not await _is_owner(interaction=interaction, owner_id=self.owner_id):
            return
        await self.cog.on_accept_plan(interaction=interaction, view=self)

    @nextcord.ui.button(
        label="修改計畫", style=ButtonStyle.secondary, custom_id="research:plan_modify"
    )
    async def modify(self, _button: Button, interaction: Interaction) -> None:
        if not await _is_owner(interaction=interaction, owner_id=self.owner_id):
            return
        await self.cog.on_modify_plan(interaction=interaction, view=self)
