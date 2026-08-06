"""Interactive single-message views for the fishing mini-game: every screen `/games fishing` has.

`cogs/games/cog.py` posts the panel once and then leaves; from that press onward this module owns
the whole surface. One public message is edited in place across the panel, the shop, the bait
picker, the cast animation, the catch reveal, the leaderboard, and the stats, so a whole session
leaves one message in the channel rather than a trail of them.

A screen is a `FishingPublicView` subclass carrying nothing but its own controls, and a
transition is a module-level `show_*` / `*_cast` coroutine that re-reads state, renders an embed
from `presentation.py`, and hands a freshly built view to `edit_owned_public_message`. Keeping
the transitions at module level is what lets any screen reach any other without a view importing
its siblings, and it is why a transitioning callback calls `self.stop()` first: a stopped view
never runs `on_timeout`, so the screen being replaced cannot delete the message its replacement
now owns.

The rules half of the feature is never reimplemented here. `database.py` settles every purchase
and cast, `catch.py` rolls, `shop.py` parses quantities and partitions the catalog, and
`presentation.py` builds every embed. What is left is the Discord layer alone: which control
leads where, what has to be acknowledged before a slow write, and which message a modal submit
has to be pointed back at.

Ownership and lifetime come from `utils/owned_message_views.py`, shared with `/stock`: only the
user who opened the panel may operate it, and the message deletes itself after
`FISHING_ACTION_TIMEOUT_SECONDS` idle.
"""

from typing import cast
import asyncio

import nextcord
from nextcord import User, Member, Message, ButtonStyle, Interaction, SelectOption
from nextcord.ui import View, Modal, Button, TextInput, StringSelect
from nextcord.ext import commands

from discordbot.utils.avatars import guild_avatar_url
from discordbot.typings.fishing import (
    FISHING_ACTION_TIMEOUT_SECONDS,
    GearView,
    CastStatus,
    PurchaseResult,
)
from discordbot.cogs.games.fishing.shop import (
    partition_gear,
    gear_option_label,
    parse_bait_quantity,
    gear_option_description,
)
from discordbot.utils.owned_message_views import (
    OwnedPublicView,
    send_ephemeral_notice,
    edit_owned_public_message,
)
from discordbot.cogs.games.fishing.database import (
    list_gear,
    settle_cast,
    purchase_gear,
    fetch_top_catches,
    get_fishing_panel,
    fetch_recent_catches,
    get_grade_config_map,
)
from discordbot.cogs.games.fishing.presentation import (
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


def require_fishing_user(interaction: Interaction[commands.Bot]) -> User | Member:
    """Returns the interaction user or fails before any fishing state can be written.

    Every caller is about to key a wallet, an angler row or a catch log on this id, so an
    interaction that carries no user is refused here rather than defaulting to one and writing
    someone else's fishing state under it.

    Args:
        interaction (Interaction[commands.Bot]): The interaction being handled.

    Returns:
        The Discord user or member who triggered the interaction.

    Raises:
        RuntimeError: The interaction carries no Discord user identity.
    """
    if interaction.user is None:
        raise RuntimeError("Fishing interaction is missing Discord user identity")
    return interaction.user


class FishingPublicView(OwnedPublicView):
    """Base view for fishing states that own one Discord message.

    Holds the fishing idle timeout and the single refusal notice every screen answers a stranger
    with, so a subclass declares nothing but its own controls.
    """

    def __init__(self, owner_id: int, delete_on_timeout: bool = True) -> None:
        """Initializes fishing controls with an idle timeout.

        Args:
            owner_id (int): Discord id of the user allowed to operate this panel.
            delete_on_timeout (bool): Whether going idle deletes the panel message; every
                fishing screen takes the default.
        """
        super().__init__(
            owner_id=owner_id,
            timeout_seconds=FISHING_ACTION_TIMEOUT_SECONDS,
            owner_mismatch_notice="這個釣魚面板只有發起者可以操作，請自己用 `/games fishing` 開一個新的",
            delete_on_timeout=delete_on_timeout,
        )


class FishingPanelView(FishingPublicView):
    """Main fishing panel controls.

    The screen `/games fishing` opens on and the one every 返回 button leads back to.
    """

    @nextcord.ui.button(
        label="拋竿", emoji="🎣", style=ButtonStyle.primary, custom_id="fishing:cast", row=0
    )
    async def cast(
        self, _button: Button["FishingPanelView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Starts a cast from the panel.

        Args:
            _button (Button["FishingPanelView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await begin_cast(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="商店", emoji="🛒", style=ButtonStyle.secondary, custom_id="fishing:shop", row=0
    )
    async def shop(
        self, _button: Button["FishingPanelView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Opens the gear shop.

        Args:
            _button (Button["FishingPanelView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await show_shop(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="排行榜", emoji="🏆", style=ButtonStyle.secondary, custom_id="fishing:board", row=1
    )
    async def leaderboard(
        self, _button: Button["FishingPanelView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Shows the top-catches leaderboard.

        Args:
            _button (Button["FishingPanelView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await show_leaderboard(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="我的紀錄", emoji="📊", style=ButtonStyle.secondary, custom_id="fishing:stats", row=1
    )
    async def stats(
        self, _button: Button["FishingPanelView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Shows personal fishing stats and recent catches.

        Args:
            _button (Button["FishingPanelView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await show_stats(interaction=interaction, owner_id=self.owner_id)


class FishingShopView(FishingPublicView):
    """Shop controls for buying rods and bait.

    A rod is bought one at a time straight from its select; bait needs a quantity, so its select
    opens `FishingBaitQtyModal` instead of purchasing.
    """

    def __init__(
        self, owner_id: int, rods: tuple[GearView, ...], baits: tuple[GearView, ...]
    ) -> None:
        """Initializes shop selects from the gear catalog.

        The decorated callbacks are rebound to their `StringSelect` items by `View.__init__`, so
        the options can only be filled in after `super().__init__` and only through a cast: the
        declared type of `self.rod_select` is still the callback. Discord rejects a select with
        no options at all, hence the sentinel row when the catalog is empty — the callbacks
        refuse both it and the `loading` placeholder the decorator declares.

        Args:
            owner_id (int): Discord id of the user allowed to operate this panel.
            rods (tuple[GearView, ...]): Purchasable rods, in the order they are offered.
            baits (tuple[GearView, ...]): Purchasable baits, in the order they are offered.
        """
        super().__init__(owner_id=owner_id)
        self.rods = rods
        self.baits = baits
        rod_select = cast('StringSelect["FishingShopView"]', self.rod_select)
        rod_select.options = [
            SelectOption(
                label=gear_option_label(gear=rod),
                value=rod.gear_id,
                description=gear_option_description(gear=rod),
            )
            for rod in rods
        ] or [SelectOption(label="目前沒有釣竿", value="none")]
        bait_select = cast('StringSelect["FishingShopView"]', self.bait_select)
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
    async def rod_select(
        self, select: StringSelect["FishingShopView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Buys the selected rod and refreshes the shop.

        A rod replaces whatever is equipped and forfeits the casts left on it, at full price.
        That rule is the store's, and this select asks for no confirmation before spending.

        Args:
            select (StringSelect["FishingShopView"]): The select carrying the chosen gear id.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        value = select.values[0]
        if value in {"none", "loading"}:
            await send_ephemeral_notice(
                interaction=interaction,
                content="目前沒有可購買的釣竿",
                log_message="Failed to send fishing empty-rod notice",
            )
            return
        # Ack before the wallet debit and shop refresh so a slow DB write cannot
        # blow Discord's interaction window, mirroring the bait purchase path.
        await interaction.response.defer()
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
    async def bait_select(
        self, select: StringSelect["FishingShopView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Opens the quantity modal for the selected bait.

        Sending a modal has to be the interaction's first response, so this path cannot defer.
        The shop view is left running and is stopped by the modal on submit instead, which is
        what keeps the shop operable when the modal is dismissed without buying anything. The
        panel message is handed to the modal because a modal submit arrives on a fresh
        interaction that is attached to no message.

        Args:
            select (StringSelect["FishingShopView"]): The select carrying the chosen bait id.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
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
    async def back(
        self, _button: Button["FishingShopView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Returns to the main panel.

        Args:
            _button (Button["FishingShopView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await show_panel(interaction=interaction, owner_id=self.owner_id)


class FishingBaitQtyModal(Modal):
    """Quantity modal for buying bait.

    A `Modal` is not a `View`, so none of `FishingPublicView`'s scaffolding reaches it: nextcord
    gives a modal no `interaction_check` hook, which is why `callback` repeats the owner test
    itself rather than inheriting one.
    """

    def __init__(
        self,
        bait_id: str,
        owner_id: int,
        parent: FishingPublicView | None = None,
        message: Message | None = None,
    ) -> None:
        """Initializes the modal with one quantity input.

        Args:
            bait_id (str): Catalog id of the bait the quantity applies to.
            owner_id (int): Discord id of the user allowed to submit this modal.
            parent (FishingPublicView | None): The shop view to stop once the modal is submitted,
                or None when nothing should be stopped.
            message (Message | None): The panel message the refreshed shop must be written back
                to, since the submit interaction is attached to no message.
        """
        super().__init__(title="購買魚餌")
        self.bait_id = bait_id
        self.owner_id = owner_id
        self.parent = parent
        self.message = message
        self.quantity: TextInput[View] = TextInput(
            label="數量",
            placeholder="輸入要購買的數量，例如 10",
            min_length=1,
            max_length=8,
            required=True,
            row=0,
        )
        self.add_item(item=self.quantity)

    async def callback(self, interaction: Interaction[commands.Bot]) -> None:
        """Parses the quantity and buys the bait.

        A submit from anyone but the owner is refused before the parse, since a modal carries no
        owner gate of its own. A quantity outside `parse_bait_quantity`'s range re-renders the
        shop with a notice instead of buying anything; the store re-checks the cap regardless.

        Args:
            interaction (Interaction[commands.Bot]): The modal submission being answered.
        """
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
        """Initializes the bait picker from owned bait stacks.

        The options replace the decorator's `loading` placeholder, so they can only be set after
        `super().__init__` has rebound the callback to its item; `FishingShopView.__init__` has
        the mechanics. `begin_cast` only builds this view when there is more than one stack, so
        an empty option list cannot reach here.

        Args:
            owner_id (int): Discord id of the user allowed to operate this panel.
            bait_options (list[SelectOption]): One option per owned bait stack.
        """
        super().__init__(owner_id=owner_id)
        bait_select = cast('StringSelect["FishingBaitSelectView"]', self.bait_select)
        bait_select.options = bait_options

    @nextcord.ui.string_select(
        placeholder="選擇魚餌來拋竿",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中", value="loading")],
        custom_id="fishing:cast:bait",
        row=0,
    )
    async def bait_select(
        self, select: StringSelect["FishingBaitSelectView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Casts with the selected bait.

        Args:
            select (StringSelect["FishingBaitSelectView"]): The select carrying the chosen bait
                id.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await run_cast(interaction=interaction, owner_id=self.owner_id, bait_id=select.values[0])

    @nextcord.ui.button(
        label="返回", emoji="↩️", style=ButtonStyle.secondary, custom_id="fishing:cast:back", row=1
    )
    async def back(
        self, _button: Button["FishingBaitSelectView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Returns to the main panel.

        Args:
            _button (Button["FishingBaitSelectView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await show_panel(interaction=interaction, owner_id=self.owner_id)


class FishingPostCastView(FishingPublicView):
    """Controls shown after a catch reveal."""

    @nextcord.ui.button(
        label="再拋一次", emoji="🎣", style=ButtonStyle.primary, custom_id="fishing:recast", row=0
    )
    async def recast(
        self, _button: Button["FishingPostCastView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Starts another cast.

        Args:
            _button (Button["FishingPostCastView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await begin_cast(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="商店",
        emoji="🛒",
        style=ButtonStyle.secondary,
        custom_id="fishing:postcast:shop",
        row=0,
    )
    async def shop(
        self, _button: Button["FishingPostCastView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Opens the gear shop.

        Args:
            _button (Button["FishingPostCastView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await show_shop(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="返回",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="fishing:postcast:back",
        row=0,
    )
    async def back(
        self, _button: Button["FishingPostCastView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Returns to the main panel.

        Args:
            _button (Button["FishingPostCastView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await show_panel(interaction=interaction, owner_id=self.owner_id)


class FishingNavView(FishingPublicView):
    """A single back-to-panel control for the leaderboard, stats, and error states."""

    @nextcord.ui.button(
        label="返回", emoji="↩️", style=ButtonStyle.secondary, custom_id="fishing:nav:back", row=0
    )
    async def back(
        self, _button: Button["FishingNavView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Returns to the main panel.

        Args:
            _button (Button["FishingNavView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await show_panel(interaction=interaction, owner_id=self.owner_id)


class FishingErrorView(FishingPublicView):
    """Controls shown on a no-rod, no-bait, or broken-rod state.

    Every one of those is fixed by buying something, so this screen leads with the shop rather
    than only offering the way back.
    """

    @nextcord.ui.button(
        label="去商店",
        emoji="🛒",
        style=ButtonStyle.primary,
        custom_id="fishing:error:shop",
        row=0,
    )
    async def shop(
        self, _button: Button["FishingErrorView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Opens the gear shop.

        Args:
            _button (Button["FishingErrorView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await show_shop(interaction=interaction, owner_id=self.owner_id)

    @nextcord.ui.button(
        label="返回", emoji="↩️", style=ButtonStyle.secondary, custom_id="fishing:error:back", row=0
    )
    async def back(
        self, _button: Button["FishingErrorView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Returns to the main panel.

        Args:
            _button (Button["FishingErrorView"]): The pressed button, unused.
            interaction (Interaction[commands.Bot]): The press being answered.
        """
        self.stop()
        await show_panel(interaction=interaction, owner_id=self.owner_id)


def _purchase_notice(result: PurchaseResult) -> str:
    """Returns a one-line shop notice describing a purchase outcome.

    The four keys are every `reason` the store emits today; the trailing fallback covers a reason
    added there without a line here, so an unknown one reads as a failure rather than as success.

    Args:
        result (PurchaseResult): The settled outcome from `purchase_gear`.

    Returns:
        The notice line for the shop embed's header.
    """
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
    """Returns a friendly message for a cast that could not produce a catch.

    Only reached for a roll-less status, which today is exactly the three named here; the
    fallback is what keeps a future roll-less status from rendering as an empty line.

    Args:
        status (CastStatus): The status `settle_cast` refused the cast with.

    Returns:
        The line for the error embed, naming the purchase that fixes it where there is one.
    """
    messages = {
        CastStatus.NO_ROD: "你還沒有可用的釣竿，先去商店買一支吧",
        CastStatus.BROKEN_ROD: "你的釣竿已損壞，去商店買一支新的",
        CastStatus.NO_BAIT: "你沒有這種魚餌了，先去商店補貨",
    }
    return messages.get(status, "這一竿沒有結果，請再試一次")


async def _purchase_and_refresh_shop(
    interaction: Interaction[commands.Bot],
    owner_id: int,
    gear_id: str,
    quantity: int,
    message: Message | None = None,
) -> None:
    """Buys gear then re-renders the shop with a result notice.

    Both callers acknowledge the interaction first, so the wallet debit and the gear grant run
    outside Discord's three-second response window. The shop is re-rendered either way, since a
    refusal has to be shown next to the balance that caused it.

    Args:
        interaction (Interaction[commands.Bot]): The already-acknowledged interaction to answer.
        owner_id (int): Discord id of the user the refreshed shop stays locked to.
        gear_id (str): Catalog id of the rod or bait being bought.
        quantity (int): How many units to buy; the store forces a rod to one.
        message (Message | None): The panel message to write the refreshed shop back to, needed
            on the modal path whose interaction is attached to no message.
    """
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


async def show_panel(interaction: Interaction[commands.Bot], owner_id: int) -> None:
    """Renders the main fishing panel into the public message.

    State is re-read on every arrival rather than carried across the transition, so the balance,
    durability and bait counts a returning user sees are the ones a cast or a purchase just left
    behind.

    Args:
        interaction (Interaction[commands.Bot]): The interaction to answer.
        owner_id (int): Discord id of the user the new panel stays locked to.
    """
    user = require_fishing_user(interaction=interaction)
    panel = await get_fishing_panel(user_id=user.id)
    grade_map = await get_grade_config_map()
    await edit_owned_public_message(
        interaction=interaction,
        embed=build_panel_embed(panel=panel, grade_map=grade_map),
        view=FishingPanelView(owner_id=owner_id),
    )


async def show_shop(
    interaction: Interaction[commands.Bot],
    owner_id: int,
    notice: str = "",
    message: Message | None = None,
) -> None:
    """Renders the gear shop into the public message.

    Lists the whole catalog, not what the angler can afford, so the balance line and the prices
    are read together.

    Args:
        interaction (Interaction[commands.Bot]): The interaction to answer.
        owner_id (int): Discord id of the user the new shop view stays locked to.
        notice (str): One-line purchase outcome to show above the listing, empty for none.
        message (Message | None): The panel message to write the shop back to, needed on the
            modal path whose interaction is attached to no message.
    """
    user = require_fishing_user(interaction=interaction)
    panel = await get_fishing_panel(user_id=user.id)
    gear = await list_gear()
    rods, baits = partition_gear(gear=gear)
    await edit_owned_public_message(
        interaction=interaction,
        embed=build_shop_embed(balance=panel.balance, rods=rods, baits=baits, notice=notice),
        view=FishingShopView(owner_id=owner_id, rods=rods, baits=baits),
        message=message,
    )


async def show_leaderboard(interaction: Interaction[commands.Bot], owner_id: int) -> None:
    """Renders the top-catches leaderboard into the public message.

    Ranks single catches across every angler the bot knows, not the pressing user's, so it is
    the one transition here that reads no user identity.

    Args:
        interaction (Interaction[commands.Bot]): The interaction to answer.
        owner_id (int): Discord id of the user the back control stays locked to.
    """
    catches = await fetch_top_catches(limit=10)
    grade_map = await get_grade_config_map()
    await edit_owned_public_message(
        interaction=interaction,
        embed=build_leaderboard_embed(catches=catches, grade_map=grade_map),
        view=FishingNavView(owner_id=owner_id),
    )


async def show_stats(interaction: Interaction[commands.Bot], owner_id: int) -> None:
    """Renders personal fishing stats into the public message.

    Shows the pressing user's own lifetime totals and last five catches. The owner gate already
    guarantees those are the same person, so nothing here compares the two ids.

    Args:
        interaction (Interaction[commands.Bot]): The interaction to answer.
        owner_id (int): Discord id of the user the back control stays locked to.
    """
    user = require_fishing_user(interaction=interaction)
    panel = await get_fishing_panel(user_id=user.id)
    recent = await fetch_recent_catches(user_id=user.id, limit=5)
    await edit_owned_public_message(
        interaction=interaction,
        embed=build_stats_embed(panel=panel, recent=recent),
        view=FishingNavView(owner_id=owner_id),
    )


async def begin_cast(interaction: Interaction[commands.Bot], owner_id: int) -> None:
    """Validates gear then casts directly or asks which bait to use.

    The rod and bait checks here are for the user's benefit, not the store's: they turn a refusal
    into the error screen that offers the shop, instead of spending a cast on a state
    `settle_cast` would reject anyway. The store re-checks under its own lock, so a cast that
    races a purchase still lands on `run_cast`'s error path rather than on a wrong outcome.

    A single owned bait stack casts straight away; several open the picker, since the bait
    decides the value bonus and the luck shift and is not interchangeable.

    Args:
        interaction (Interaction[commands.Bot]): The interaction to answer.
        owner_id (int): Discord id of the user every screen this leads to stays locked to.
    """
    user = require_fishing_user(interaction=interaction)
    panel = await get_fishing_panel(user_id=user.id)
    if panel.angler.rod is None or panel.angler.durability_remaining <= 0:
        message = (
            "你的釣竿已損壞，去商店買一支新的"
            if panel.angler.rod is not None
            else "你還沒有可用的釣竿，先去商店買一支吧"
        )
        await edit_owned_public_message(
            interaction=interaction,
            embed=build_error_embed(message=message),
            view=FishingErrorView(owner_id=owner_id),
        )
        return
    if not panel.baits:
        await edit_owned_public_message(
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
    await edit_owned_public_message(
        interaction=interaction,
        embed=build_bait_select_embed(panel=panel),
        view=FishingBaitSelectView(owner_id=owner_id, bait_options=bait_options),
    )


async def run_cast(interaction: Interaction[commands.Bot], owner_id: int, bait_id: str) -> None:
    """Runs the two-beat cast animation and settles the catch.

    The first beat is posted with `view=None`, so the controls are gone for the whole pause and a
    second cast cannot be started on top of this one. Settlement runs after the sleep, and the
    panel is re-read afterwards so the reveal shows the balance, durability and bait the cast
    itself left behind.

    A roll-less result means the store refused the cast and it becomes the error screen. A
    deferred payout is not one of those: the catch is committed and carries a roll, so it reveals
    normally and the embed says the credit is late.

    Args:
        interaction (Interaction[commands.Bot]): The interaction to answer.
        owner_id (int): Discord id of the user the reveal's controls stay locked to.
        bait_id (str): Catalog id of the bait this cast consumes.
    """
    await edit_owned_public_message(
        interaction=interaction, embed=build_casting_embed(), view=None
    )
    await asyncio.sleep(CAST_ANIMATION_SECONDS)
    user = require_fishing_user(interaction=interaction)
    avatar_url = await guild_avatar_url(user=user, guild=getattr(interaction, "guild", None))
    result = await settle_cast(
        user_id=user.id, name=user.name, bait_id=bait_id, avatar_url=avatar_url
    )
    if result.roll is None:
        await edit_owned_public_message(
            interaction=interaction,
            embed=build_error_embed(message=_cast_failure_message(status=result.status)),
            view=FishingErrorView(owner_id=owner_id),
        )
        return
    panel = await get_fishing_panel(user_id=user.id)
    grade_map = await get_grade_config_map()
    await edit_owned_public_message(
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
    "require_fishing_user",
    "run_cast",
    "show_leaderboard",
    "show_panel",
    "show_shop",
    "show_stats",
]
