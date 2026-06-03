"""Interactive single-message views for the fishing mini-game.

One public message is edited in place across the panel, shop, cast reveal,
leaderboard, and stats. Only the user who opened the panel can operate it, and
the message deletes itself after the idle timeout, matching the stock panel UX.
"""

from typing import cast
import asyncio

import nextcord
from nextcord import User, Embed, Member, Message, ButtonStyle, Interaction, SelectOption
from nextcord.ui import View, Modal, Button, TextInput, StringSelect

from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.fishing import (
    FISHING_ACTION_TIMEOUT_SECONDS,
    GearView,
    CastStatus,
    PurchaseResult,
)
from discordbot.cogs._fishing.shop import (
    partition_gear,
    gear_option_label,
    parse_bait_quantity,
    gear_option_description,
)
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import (
    track_public_message,
    delete_public_message,
    forget_public_message,
)
from discordbot.cogs._fishing.database import (
    list_gear,
    settle_cast,
    purchase_gear,
    fetch_top_catches,
    get_fishing_panel,
    fetch_recent_catches,
    get_grade_config_map,
)
from discordbot.cogs._games.interactions import send_ephemeral_notice
from discordbot.cogs._fishing.presentation import (
    build_shop_embed,
    build_error_embed,
    build_panel_embed,
    build_stats_embed,
    build_reveal_embed,
    build_casting_embed,
    build_bait_select_embed,
    build_leaderboard_embed,
)

CAST_ANIMATION_SECONDS = 1.0


def require_fishing_user(interaction: Interaction) -> User | Member:
    """Returns the interaction user or fails before any fishing state can be written."""
    if interaction.user is None:
        raise RuntimeError("Fishing interaction is missing Discord user identity")
    return interaction.user


class FishingPublicView(View):
    """Base view for fishing states that own one Discord message."""

    def __init__(self, owner_id: int, delete_on_timeout: bool = True) -> None:
        """Initializes fishing controls with an idle timeout."""
        super().__init__(timeout=FISHING_ACTION_TIMEOUT_SECONDS)
        self.owner_id = owner_id
        self.delete_on_timeout = delete_on_timeout
        self.message: Message | None = None

    def bind_message(self, message: Message | None) -> None:
        """Records the message this view should update or delete."""
        self.message = message

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Allows only the user who opened this fishing panel to operate it."""
        user = require_fishing_user(interaction=interaction)
        if self.owner_id == user.id:
            return True
        await send_ephemeral_notice(
            interaction=interaction,
            content="這個釣魚面板只有發起者可以操作，請自己用 `/games fishing` 開一個新的",
            log_message="Failed to send fishing owner mismatch notice",
        )
        return False

    async def on_timeout(self) -> None:
        """Deletes the tracked public message after the idle timeout."""
        if self.message is None or not self.delete_on_timeout:
            return
        await delete_public_message(message=self.message)


async def edit_fishing_message(
    interaction: Interaction,
    embed: Embed,
    view: FishingPublicView | None,
    message: Message | None = None,
) -> None:
    """Edits the current fishing message for a component or modal interaction."""
    target_message = message or interaction.message
    if view is not None:
        view.bind_message(message=target_message)
    edit_payload = {
        "embed": embed,
        "view": view,
        **embed_spacer_payload(embeds=[embed], is_edit=True, target=target_message or interaction),
    }
    if not interaction.response.is_done():
        edited = await interaction.response.edit_message(**edit_payload)
        if isinstance(edited, Message) and view is not None:
            view.bind_message(message=edited)
        return
    if target_message is not None:
        try:
            await target_message.edit(**edit_payload)
            return
        except nextcord.NotFound:
            message_id = getattr(target_message, "id", None)
            if isinstance(message_id, int):
                await forget_public_message(message_id=message_id)
    sent_message = await interaction.followup.send(
        embed=embed,
        view=view,
        wait=True,
        **embed_spacer_payload(embeds=[embed], is_edit=False, target=interaction),
    )
    if view is not None:
        view.bind_message(message=sent_message)
    await track_public_message(
        message=sent_message,
        user_name=getattr(require_fishing_user(interaction=interaction), "name", None),
    )


class FishingPanelView(FishingPublicView):
    """Main fishing panel controls."""

    @nextcord.ui.button(
        label="拋竿", emoji="🎣", style=ButtonStyle.primary, custom_id="fishing:cast", row=0
    )
    async def cast(self, _button: Button, interaction: Interaction) -> None:
        """Starts a cast from the panel."""
        self.stop()
        await begin_cast(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="商店", emoji="🛒", style=ButtonStyle.secondary, custom_id="fishing:shop", row=0
    )
    async def shop(self, _button: Button, interaction: Interaction) -> None:
        """Opens the gear shop."""
        self.stop()
        await show_shop(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="排行榜", emoji="🏆", style=ButtonStyle.secondary, custom_id="fishing:board", row=1
    )
    async def leaderboard(self, _button: Button, interaction: Interaction) -> None:
        """Shows the top-catches leaderboard."""
        self.stop()
        await show_leaderboard(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="我的紀錄", emoji="📊", style=ButtonStyle.secondary, custom_id="fishing:stats", row=1
    )
    async def stats(self, _button: Button, interaction: Interaction) -> None:
        """Shows personal fishing stats and recent catches."""
        self.stop()
        await show_stats(interaction=interaction, owner_id=self.owner_id)


class FishingShopView(FishingPublicView):
    """Shop controls for buying rods and bait."""

    def __init__(
        self, owner_id: int, rods: tuple[GearView, ...], baits: tuple[GearView, ...]
    ) -> None:
        """Initializes shop selects from the gear catalog."""
        super().__init__(owner_id=owner_id)
        self.rods = rods
        self.baits = baits
        rod_select = cast("StringSelect", self.rod_select)
        rod_select.options = [
            SelectOption(
                label=gear_option_label(gear=rod),
                value=rod.gear_id,
                description=gear_option_description(gear=rod),
            )
            for rod in rods
        ] or [SelectOption(label="目前沒有釣竿", value="none")]
        bait_select = cast("StringSelect", self.bait_select)
        bait_select.options = [
            SelectOption(
                label=gear_option_label(gear=bait),
                value=bait.gear_id,
                description=gear_option_description(gear=bait),
            )
            for bait in baits
        ] or [SelectOption(label="目前沒有魚餌", value="none")]

    @nextcord.ui.string_select(
        placeholder="選擇要買的釣竿",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中", value="loading")],
        custom_id="fishing:shop:rod",
        row=0,
    )
    async def rod_select(self, select: StringSelect, interaction: Interaction) -> None:
        """Buys the selected rod and refreshes the shop."""
        value = select.values[0]
        if value in {"none", "loading"}:
            await send_ephemeral_notice(
                interaction=interaction,
                content="目前沒有可購買的釣竿",
                log_message="Failed to send fishing empty-rod notice",
            )
            return
        self.stop()
        await _purchase_and_refresh_shop(
            interaction=interaction, owner_id=self.owner_id, gear_id=value, quantity=1
        )

    @nextcord.ui.string_select(
        placeholder="選擇要買的魚餌",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中", value="loading")],
        custom_id="fishing:shop:bait",
        row=1,
    )
    async def bait_select(self, select: StringSelect, interaction: Interaction) -> None:
        """Opens the quantity modal for the selected bait."""
        value = select.values[0]
        if value in {"none", "loading"}:
            await send_ephemeral_notice(
                interaction=interaction,
                content="目前沒有可購買的魚餌",
                log_message="Failed to send fishing empty-bait notice",
            )
            return
        await interaction.response.send_modal(
            modal=FishingBaitQtyModal(
                bait_id=value, owner_id=self.owner_id, parent=self, message=interaction.message
            )
        )

    @nextcord.ui.button(
        label="返回", emoji="↩️", style=ButtonStyle.secondary, custom_id="fishing:shop:back", row=2
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the main panel."""
        self.stop()
        await show_panel(interaction=interaction, owner_id=self.owner_id)


class FishingBaitQtyModal(Modal):
    """Quantity modal for buying bait."""

    def __init__(
        self,
        bait_id: str,
        owner_id: int,
        parent: FishingPublicView | None = None,
        message: Message | None = None,
    ) -> None:
        """Initializes the modal with one quantity input."""
        super().__init__(title="購買魚餌")
        self.bait_id = bait_id
        self.owner_id = owner_id
        self.parent = parent
        self.message = message
        self.quantity: TextInput = TextInput(
            label="數量",
            placeholder="輸入要購買的數量，例如 10",
            min_length=1,
            max_length=8,
            required=True,
            row=0,
        )
        self.add_item(item=self.quantity)

    async def callback(self, interaction: Interaction) -> None:
        """Parses the quantity and buys the bait."""
        user = require_fishing_user(interaction=interaction)
        if self.owner_id != user.id:
            await send_ephemeral_notice(
                interaction=interaction,
                content="這個釣魚面板只有發起者可以操作，請自己用 `/games fishing` 開一個新的",
                log_message="Failed to send fishing modal owner mismatch notice",
            )
            return
        await interaction.response.defer()
        if self.parent is not None:
            self.parent.stop()
        quantity = parse_bait_quantity(raw_quantity=str(self.quantity.value or ""))
        if quantity is None:
            await show_shop(
                interaction=interaction,
                owner_id=self.owner_id,
                notice="❌ 數量格式錯誤，請輸入 1 以上的整數",
                message=self.message,
            )
            return
        await _purchase_and_refresh_shop(
            interaction=interaction,
            owner_id=self.owner_id,
            gear_id=self.bait_id,
            quantity=quantity,
            message=self.message,
        )


class FishingBaitSelectView(FishingPublicView):
    """Bait picker shown before a cast when the angler owns multiple baits."""

    def __init__(self, owner_id: int, bait_options: list[SelectOption]) -> None:
        """Initializes the bait picker from owned bait stacks."""
        super().__init__(owner_id=owner_id)
        bait_select = cast("StringSelect", self.bait_select)
        bait_select.options = bait_options

    @nextcord.ui.string_select(
        placeholder="選擇魚餌來拋竿",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中", value="loading")],
        custom_id="fishing:cast:bait",
        row=0,
    )
    async def bait_select(self, select: StringSelect, interaction: Interaction) -> None:
        """Casts with the selected bait."""
        self.stop()
        await run_cast(interaction=interaction, owner_id=self.owner_id, bait_id=select.values[0])

    @nextcord.ui.button(
        label="返回", emoji="↩️", style=ButtonStyle.secondary, custom_id="fishing:cast:back", row=1
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the main panel."""
        self.stop()
        await show_panel(interaction=interaction, owner_id=self.owner_id)


class FishingPostCastView(FishingPublicView):
    """Controls shown after a catch reveal."""

    @nextcord.ui.button(
        label="再拋一次", emoji="🎣", style=ButtonStyle.primary, custom_id="fishing:recast", row=0
    )
    async def recast(self, _button: Button, interaction: Interaction) -> None:
        """Starts another cast."""
        self.stop()
        await begin_cast(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="商店",
        emoji="🛒",
        style=ButtonStyle.secondary,
        custom_id="fishing:postcast:shop",
        row=0,
    )
    async def shop(self, _button: Button, interaction: Interaction) -> None:
        """Opens the gear shop."""
        self.stop()
        await show_shop(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="返回",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="fishing:postcast:back",
        row=0,
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the main panel."""
        self.stop()
        await show_panel(interaction=interaction, owner_id=self.owner_id)


class FishingNavView(FishingPublicView):
    """A single back-to-panel control for the leaderboard, stats, and error states."""

    @nextcord.ui.button(
        label="返回", emoji="↩️", style=ButtonStyle.secondary, custom_id="fishing:nav:back", row=0
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the main panel."""
        self.stop()
        await show_panel(interaction=interaction, owner_id=self.owner_id)


class FishingErrorView(FishingPublicView):
    """Controls shown on a no-rod, no-bait, or broken-rod state."""

    @nextcord.ui.button(
        label="去商店",
        emoji="🛒",
        style=ButtonStyle.primary,
        custom_id="fishing:error:shop",
        row=0,
    )
    async def shop(self, _button: Button, interaction: Interaction) -> None:
        """Opens the gear shop."""
        self.stop()
        await show_shop(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="返回", emoji="↩️", style=ButtonStyle.secondary, custom_id="fishing:error:back", row=0
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the main panel."""
        self.stop()
        await show_panel(interaction=interaction, owner_id=self.owner_id)


def _purchase_notice(result: PurchaseResult) -> str:
    """Returns a one-line shop notice describing a purchase outcome."""
    if result.success:
        return f"✅ 購買成功，花費 {result.total_cost:,}"
    reasons = {
        "insufficient": "❌ 餘額不足",
        "grant_failed": "❌ 購買失敗，已退款，請再試一次",
        "unknown_gear": "❌ 找不到這個道具",
        "invalid_quantity": "❌ 數量不正確",
    }
    return reasons.get(result.reason, "❌ 購買失敗")


def _cast_failure_message(status: CastStatus) -> str:
    """Returns a friendly message for a cast that could not produce a catch."""
    messages = {
        CastStatus.NO_ROD: "你還沒有可用的釣竿，先去商店買一支吧",
        CastStatus.BROKEN_ROD: "你的釣竿已損壞，去商店買一支新的",
        CastStatus.NO_BAIT: "你沒有這種魚餌了，先去商店補貨",
    }
    return messages.get(status, "這一竿沒有結果，請再試一次")


async def _purchase_and_refresh_shop(
    interaction: Interaction,
    owner_id: int,
    gear_id: str,
    quantity: int,
    message: Message | None = None,
) -> None:
    """Buys gear then re-renders the shop with a result notice."""
    user = require_fishing_user(interaction=interaction)
    avatar_url = await guild_avatar_url(user=user, guild=getattr(interaction, "guild", None))
    result = await purchase_gear(
        user_id=user.id, name=user.name, gear_id=gear_id, quantity=quantity, avatar_url=avatar_url
    )
    await show_shop(
        interaction=interaction,
        owner_id=owner_id,
        notice=_purchase_notice(result=result),
        message=message,
    )


async def show_panel(interaction: Interaction, owner_id: int) -> None:
    """Renders the main fishing panel into the public message."""
    user = require_fishing_user(interaction=interaction)
    panel = await get_fishing_panel(user_id=user.id)
    grade_map = await get_grade_config_map()
    await edit_fishing_message(
        interaction=interaction,
        embed=build_panel_embed(panel=panel, grade_map=grade_map),
        view=FishingPanelView(owner_id=owner_id),
    )


async def show_shop(
    interaction: Interaction, owner_id: int, notice: str = "", message: Message | None = None
) -> None:
    """Renders the gear shop into the public message."""
    user = require_fishing_user(interaction=interaction)
    panel = await get_fishing_panel(user_id=user.id)
    gear = await list_gear()
    rods, baits = partition_gear(gear=gear)
    await edit_fishing_message(
        interaction=interaction,
        embed=build_shop_embed(balance=panel.balance, rods=rods, baits=baits, notice=notice),
        view=FishingShopView(owner_id=owner_id, rods=rods, baits=baits),
        message=message,
    )


async def show_leaderboard(interaction: Interaction, owner_id: int) -> None:
    """Renders the top-catches leaderboard into the public message."""
    catches = await fetch_top_catches(limit=10)
    grade_map = await get_grade_config_map()
    await edit_fishing_message(
        interaction=interaction,
        embed=build_leaderboard_embed(catches=catches, grade_map=grade_map),
        view=FishingNavView(owner_id=owner_id),
    )


async def show_stats(interaction: Interaction, owner_id: int) -> None:
    """Renders personal fishing stats into the public message."""
    user = require_fishing_user(interaction=interaction)
    panel = await get_fishing_panel(user_id=user.id)
    recent = await fetch_recent_catches(user_id=user.id, limit=5)
    await edit_fishing_message(
        interaction=interaction,
        embed=build_stats_embed(panel=panel, recent=recent),
        view=FishingNavView(owner_id=owner_id),
    )


async def begin_cast(interaction: Interaction, owner_id: int) -> None:
    """Validates gear then casts directly or asks which bait to use."""
    user = require_fishing_user(interaction=interaction)
    panel = await get_fishing_panel(user_id=user.id)
    if panel.angler.rod is None or panel.angler.durability_remaining <= 0:
        message = (
            "你的釣竿已損壞，去商店買一支新的"
            if panel.angler.rod is not None
            else "你還沒有可用的釣竿，先去商店買一支吧"
        )
        await edit_fishing_message(
            interaction=interaction,
            embed=build_error_embed(message=message),
            view=FishingErrorView(owner_id=owner_id),
        )
        return
    if not panel.baits:
        await edit_fishing_message(
            interaction=interaction,
            embed=build_error_embed(message="你沒有魚餌了，先去商店補貨"),
            view=FishingErrorView(owner_id=owner_id),
        )
        return
    if len(panel.baits) == 1:
        await run_cast(interaction=interaction, owner_id=owner_id, bait_id=panel.baits[0].bait_id)
        return
    bait_options = [
        SelectOption(
            label=f"{stack.emoji} {stack.name}",
            value=stack.bait_id,
            description=f"剩 {stack.quantity}",
        )
        for stack in panel.baits
    ]
    await edit_fishing_message(
        interaction=interaction,
        embed=build_bait_select_embed(panel=panel),
        view=FishingBaitSelectView(owner_id=owner_id, bait_options=bait_options),
    )


async def run_cast(interaction: Interaction, owner_id: int, bait_id: str) -> None:
    """Runs the two-beat cast animation and settles the catch."""
    await edit_fishing_message(interaction=interaction, embed=build_casting_embed(), view=None)
    await asyncio.sleep(CAST_ANIMATION_SECONDS)
    user = require_fishing_user(interaction=interaction)
    avatar_url = await guild_avatar_url(user=user, guild=getattr(interaction, "guild", None))
    result = await settle_cast(
        user_id=user.id, name=user.name, bait_id=bait_id, avatar_url=avatar_url
    )
    if result.roll is None:
        await edit_fishing_message(
            interaction=interaction,
            embed=build_error_embed(message=_cast_failure_message(status=result.status)),
            view=FishingErrorView(owner_id=owner_id),
        )
        return
    panel = await get_fishing_panel(user_id=user.id)
    grade_map = await get_grade_config_map()
    await edit_fishing_message(
        interaction=interaction,
        embed=build_reveal_embed(result=result, panel=panel, grade_map=grade_map),
        view=FishingPostCastView(owner_id=owner_id),
    )


__all__ = [
    "FishingBaitQtyModal",
    "FishingBaitSelectView",
    "FishingErrorView",
    "FishingNavView",
    "FishingPanelView",
    "FishingPostCastView",
    "FishingPublicView",
    "FishingShopView",
    "begin_cast",
    "build_panel_embed",
    "edit_fishing_message",
    "require_fishing_user",
    "run_cast",
    "show_leaderboard",
    "show_panel",
    "show_shop",
    "show_stats",
]
