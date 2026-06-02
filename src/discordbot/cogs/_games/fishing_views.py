"""Single-message interactive UI for the fishing mini-game (`/games fishing`).

One public message is edited in place across every screen (釣魚台 station, 商店
shop, 拋竿 cast animation, 魚簍 inventory, 圖鑑 dex, 排行榜 leaderboard). Only the
opener may operate it; it auto-deletes after 180 idle seconds. Everything is
text/emoji only for now (no Pillow fish art). The optional narrator banter runs
as a background refresh and never blocks the cast result.
"""

from random import Random
from typing import Literal, cast
import asyncio
import contextlib
from collections.abc import Sequence, Coroutine

import nextcord
from nextcord import Embed, Message, ButtonStyle, Interaction, SelectOption
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.ui import View, Modal, Button, TextInput, StringSelect

from discordbot.typings.colors import IN_PROGRESS_COLOR
from discordbot.typings.fishing import (
    ROD_TIERS,
    BAIT_TYPES,
    ROD_BY_KEY,
    BAIT_BY_KEY,
    FISH_CATALOG,
    RARITY_COLOR,
    RARITY_EMOJI,
    RARITY_ORDER,
    SPECIES_BY_KEY,
    MAX_BAIT_PURCHASE_QUANTITY,
    RodTier,
    BaitType,
    DexEntry,
    CastResult,
    SellResult,
    LoadoutView,
    InventoryEntry,
    BiggestCatchRow,
    FishingLeaderboardRow,
)
from discordbot.cogs._games.dealer import SystemNarrator
from discordbot.cogs._games.prompts import fishing_catch_fallback_line
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import (
    track_public_message,
    delete_public_message,
    forget_public_message,
)
from discordbot.cogs._games.interactions import send_ephemeral_notice
from discordbot.cogs._games.presentation import WIN_COLOR, metadata_line, build_system_talk_embed
from discordbot.cogs._economy.presentation import CURRENCY_NAME, amount_code, bold_currency
from discordbot.cogs._games.fishing_database import (
    buy_rod,
    get_dex,
    buy_bait,
    sell_fish,
    get_loadout,
    execute_cast,
    list_inventory,
    leaderboard_total_earned,
    leaderboard_biggest_catch,
)

FISHING_TIMEOUT_SECONDS = 180
CAST_BEAT_DELAY_SECONDS = 0.8
CAST_REVEAL_DELAY_SECONDS = 1.0
FINAL_EDIT_TIMEOUT_SECONDS = 8.0
_DURABILITY_BAR_CELLS = 10
_INVENTORY_PAGE_SIZE = 25
_SELECT_LABEL_LIMIT = 100

STATION_COLOR = IN_PROGRESS_COLOR
SHOP_COLOR = 0xFEE75C
INVENTORY_COLOR = 0x3498DB
DEX_COLOR = 0x9B59B6
LEADERBOARD_COLOR = 0xF1C40F
MISS_COLOR = 0x95A5A6

LeaderboardMetric = Literal["earned", "biggest"]


class FishingContext(BaseModel):
    """Shared per-session state threaded through every fishing view."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    owner_id: int = Field(description="Discord user id of the angler who opened the panel.")
    owner_name: str = Field(description="Last-seen Discord name stored on fishing rows.")
    owner_avatar_url: str = Field(default="", description="Avatar URL stored on wallet writes.")
    rng: SkipValidation[Random] = Field(description="Randomness source for cast rolls.")
    narrator: SkipValidation[SystemNarrator | None] = Field(
        default=None, description="Optional narrator for background catch banter."
    )
    system_name: str = Field(description="Narrator display name for the talk embed.")
    system_avatar_url: str = Field(default="", description="Narrator avatar for the talk embed.")


# --- small helpers -----------------------------------------------------------


def _durability_bar(remaining: int, total: int) -> str:
    """Returns a fixed-width green/white durability bar."""
    if total <= 0:
        return ""
    filled = round(remaining / total * _DURABILITY_BAR_CELLS)
    filled = max(0, min(_DURABILITY_BAR_CELLS, filled))
    if remaining > 0 and filled == 0:
        filled = 1
    return "🟩" * filled + "⬜" * (_DURABILITY_BAR_CELLS - filled)


def _medal(rank: int) -> str:
    """Returns a medal emoji for the top three ranks, else a numbered label."""
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")


def _inventory_option_label(entry: InventoryEntry) -> str:
    """Returns a select label for one sellable fish, fitting Discord's limit."""
    species = SPECIES_BY_KEY.get(entry.species_key)
    name = species.name if species is not None else entry.species_key
    label = f"{entry.rarity} {name} {entry.size_mm}mm {entry.sell_value:,}豆"
    if len(label) <= _SELECT_LABEL_LIMIT:
        return label
    return f"{label[: _SELECT_LABEL_LIMIT - 3]}..."


# --- embed builders ----------------------------------------------------------


def build_station_embeds(loadout: LoadoutView, avatar_url: str = "") -> list[Embed]:
    """Builds the 釣魚台 station embed showing balance, rod, and bait."""
    lines = [f"💰 {bold_currency(amount=loadout.balance, compact=True)}"]
    rod = ROD_BY_KEY.get(loadout.rod_key)
    if rod is not None and loadout.rod_durability > 0:
        bar = _durability_bar(remaining=loadout.rod_durability, total=rod.durability)
        lines.append(
            f"🎣 釣竿　{rod.emoji} {rod.name}　{bar} {loadout.rod_durability}/{rod.durability}"
        )
    else:
        lines.append("🎣 釣竿　尚未持有，先到 🛒 商店買一支")
    owned = {key: qty for key, qty in loadout.baits.items() if qty > 0 and key in BAIT_BY_KEY}
    if owned:
        bait_text = " · ".join(
            f"{BAIT_BY_KEY[key].emoji}{BAIT_BY_KEY[key].name} x{qty}" for key, qty in owned.items()
        )
        lines.append(f"🪱 魚餌　{bait_text}")
    else:
        lines.append("🪱 魚餌　無，先到 🛒 商店補貨")
    embed = Embed(title="🎣 釣魚台", description="\n".join(lines), color=STATION_COLOR)
    if avatar_url:
        embed.set_thumbnail(url=avatar_url)
    embed.set_footer(text="拋一次竿消耗 1 個魚餌與 1 點耐久；釣到的魚可在魚簍賣出")
    return [embed]


def build_shop_embeds(loadout: LoadoutView) -> list[Embed]:
    """Builds the 商店 shop embed listing rods and baits with prices."""
    embed = Embed(
        title="🛒 釣具商店",
        description=(
            f"目前餘額 {bold_currency(amount=loadout.balance, compact=True)}\n"
            "用下方選單購買釣竿或魚餌"
        ),
        color=SHOP_COLOR,
    )
    rod_lines = [
        f"{rod.emoji} **{rod.name}**　{amount_code(amount=rod.cost, compact=True)}　"
        f"耐久 {rod.durability}　空竿率 {rod.miss_bps // 100}%"
        for rod in ROD_TIERS
    ]
    embed.add_field(name="🎣 釣竿（買了會換掉目前的竿）", value="\n".join(rod_lines), inline=False)
    bait_lines = [
        f"{bait.emoji} **{bait.name}**　{amount_code(amount=bait.cost, compact=True)} / 個"
        for bait in BAIT_TYPES
    ]
    embed.add_field(name="🪱 魚餌（每次拋竿消耗 1 個）", value="\n".join(bait_lines), inline=False)
    return [embed]


def build_cast_stage_embed(title: str, line: str) -> Embed:
    """Builds one in-progress cast animation beat."""
    return Embed(title=title, description=line, color=IN_PROGRESS_COLOR)


def build_bait_choice_embed() -> Embed:
    """Builds the prompt shown when the angler must pick which bait to use."""
    return Embed(title="🪱 選擇魚餌", description="挑一種魚餌來拋竿。", color=STATION_COLOR)


def build_cast_result_embeds(
    result: CastResult, rod: RodTier, talk_line: str, system_name: str, system_avatar_url: str = ""
) -> list[Embed]:
    """Builds the cast result embed (rarity-coloured) plus an optional talk embed."""
    outcome = result.outcome
    if outcome is None or outcome.miss or outcome.species is None:
        embed = Embed(
            title="💨 空竿", description="這一竿什麼都沒釣到，再接再厲。", color=MISS_COLOR
        )
    else:
        species = outcome.species
        embed = Embed(
            title=f"{RARITY_EMOJI[species.rarity]} {species.rarity} · {species.emoji} {species.name}",
            description=(
                f"尺寸 **{outcome.size_mm}** mm\n"
                f"可賣 {bold_currency(amount=outcome.sell_value, compact=True)}"
            ),
            color=RARITY_COLOR[species.rarity],
        )
    if result.rod_broke:
        status_line = f"💥 {rod.emoji}{rod.name} 斷了!快到 🛒 商店買新的"
    else:
        bar = _durability_bar(remaining=result.rod_durability_after, total=rod.durability)
        status_line = (
            f"🎣 {rod.emoji}{rod.name} {bar} {result.rod_durability_after}/{rod.durability}"
        )
    bait = BAIT_BY_KEY.get(result.bait_key)
    if bait is not None:
        status_line += "\n" + metadata_line(
            text=f"{bait.emoji}{bait.name} 剩 {result.bait_remaining}"
        )
    embed.add_field(name="狀態", value=status_line, inline=False)
    embeds = [embed]
    if talk_line:
        embeds.append(
            build_system_talk_embed(
                system_line=talk_line, system_name=system_name, system_avatar_url=system_avatar_url
            )
        )
    return embeds


def build_inventory_embeds(entries: Sequence[InventoryEntry]) -> list[Embed]:
    """Builds the 魚簍 inventory embed listing unsold catches."""
    if not entries:
        return [
            Embed(title="🪣 魚簍", description="魚簍空空，先去拋竿吧。", color=INVENTORY_COLOR)
        ]
    lines = []
    for entry in entries:
        species = SPECIES_BY_KEY.get(entry.species_key)
        name = f"{species.emoji}{species.name}" if species is not None else entry.species_key
        lines.append(
            f"{RARITY_EMOJI[entry.rarity]} {name} · {entry.size_mm}mm · "
            f"{bold_currency(amount=entry.sell_value, compact=True)}"
        )
    embed = Embed(title="🪣 魚簍", description="\n".join(lines), color=INVENTORY_COLOR)
    embed.set_footer(text=f"顯示最新 {len(entries)} 條；「全部賣出」會賣掉你所有未賣出的魚")
    return [embed]


def build_sell_result_embeds(result: SellResult) -> list[Embed]:
    """Builds the embed shown after selling fish."""
    if result.status == "nothing":
        return [Embed(title="🪣 魚簍", description="沒有可以賣的魚。", color=INVENTORY_COLOR)]
    return [
        Embed(
            title="💵 賣魚結算",
            description=(
                f"賣出 **{result.sold_count}** 條魚\n"
                f"收入 {bold_currency(amount=result.earned, signed=True, compact=True)}\n"
                f"餘額 {bold_currency(amount=result.new_balance, compact=True)}"
            ),
            color=WIN_COLOR,
        )
    ]


def build_dex_embeds(entries: Sequence[DexEntry]) -> list[Embed]:
    """Builds the 圖鑑 dex embed grouped by rarity with completion."""
    caught = sum(1 for entry in entries if entry.caught)
    total = len(entries)
    pct = round(caught / total * 100) if total else 0
    embed = Embed(
        title="📖 魚類圖鑑", description=f"完成度 **{caught}/{total}**（{pct}%）", color=DEX_COLOR
    )
    by_key = {entry.species_key: entry for entry in entries}
    for rarity in RARITY_ORDER:
        rows = []
        for species in FISH_CATALOG:
            if species.rarity != rarity:
                continue
            entry = by_key.get(species.key)
            if entry is not None and entry.caught:
                rows.append(
                    f"{species.emoji} {species.name} x{entry.count}（最大 {entry.biggest_mm}mm）"
                )
            else:
                rows.append("❓ ???")
        embed.add_field(
            name=f"{RARITY_EMOJI[rarity]} {rarity}", value="\n".join(rows) or "—", inline=False
        )
    return [embed]


def build_leaderboard_embeds(
    metric: LeaderboardMetric,
    earned_rows: Sequence[FishingLeaderboardRow] = (),
    biggest_rows: Sequence[BiggestCatchRow] = (),
) -> list[Embed]:
    """Builds the 排行榜 leaderboard embed for the selected metric."""
    if metric == "biggest":
        title = "🏆 最大魚排行榜"
        if not biggest_rows:
            description = "還沒有人釣到魚。"
        else:
            lines = []
            for rank, row in enumerate(biggest_rows, start=1):
                species = SPECIES_BY_KEY.get(row.species_key)
                name = f"{species.emoji}{species.name}" if species is not None else row.species_key
                lines.append(
                    f"{_medal(rank=rank)} **{row.user_name}** · {name} {row.size_mm}mm（{row.rarity}）"
                )
            description = "\n".join(lines)
    else:
        title = "🏆 漁獲收益排行榜"
        if not earned_rows:
            description = "還沒有人賣過魚。"
        else:
            description = "\n".join(
                f"{_medal(rank=rank)} **{row.user_name}** · {bold_currency(amount=row.value, compact=True)}"
                for rank, row in enumerate(earned_rows, start=1)
            )
    return [Embed(title=title, description=description, color=LEADERBOARD_COLOR)]


# --- message edit dispatcher -------------------------------------------------


async def edit_fishing_message(
    interaction: Interaction,
    embeds: list[Embed],
    view: "FishingPublicView | None",
    message: Message | None = None,
) -> None:
    """Edits the fishing panel for a component or modal interaction."""
    target = message or interaction.message
    if view is not None:
        view.bind_message(message=target)
    kwargs: dict[str, object] = {
        "embeds": embeds,
        "view": view,
        **embed_spacer_payload(embeds=embeds, is_edit=True, target=target or interaction),
    }
    if not interaction.response.is_done():
        edited = await interaction.response.edit_message(**kwargs)
        if isinstance(edited, Message) and view is not None:
            view.bind_message(message=edited)
        return
    if target is not None:
        try:
            await target.edit(**kwargs)
            return
        except nextcord.NotFound:
            message_id = getattr(target, "id", None)
            if isinstance(message_id, int):
                await forget_public_message(message_id=message_id)
    sent = await interaction.followup.send(
        embeds=embeds,
        view=view,
        wait=True,
        **embed_spacer_payload(embeds=embeds, is_edit=False, target=interaction),
    )
    if view is not None:
        view.bind_message(message=sent)
    user = interaction.user
    await track_public_message(message=sent, user_name=getattr(user, "name", None))


# --- navigation helpers ------------------------------------------------------


async def _show_station(
    interaction: Interaction, ctx: FishingContext, message: Message | None = None
) -> None:
    """Renders the station screen on the panel."""
    loadout = await get_loadout(user_id=ctx.owner_id)
    await edit_fishing_message(
        interaction=interaction,
        embeds=build_station_embeds(loadout=loadout, avatar_url=ctx.owner_avatar_url),
        view=FishingStationView(ctx=ctx),
        message=message,
    )


async def _show_shop(
    interaction: Interaction, ctx: FishingContext, message: Message | None = None
) -> None:
    """Renders the shop screen on the panel."""
    loadout = await get_loadout(user_id=ctx.owner_id)
    await edit_fishing_message(
        interaction=interaction,
        embeds=build_shop_embeds(loadout=loadout),
        view=FishingShopView(ctx=ctx),
        message=message,
    )


async def _show_inventory(
    interaction: Interaction, ctx: FishingContext, message: Message | None = None
) -> None:
    """Renders the inventory screen on the panel."""
    entries = await list_inventory(user_id=ctx.owner_id, limit=_INVENTORY_PAGE_SIZE)
    await edit_fishing_message(
        interaction=interaction,
        embeds=build_inventory_embeds(entries=entries),
        view=FishingInventoryView(ctx=ctx, entries=entries),
        message=message,
    )


async def _show_dex(
    interaction: Interaction, ctx: FishingContext, message: Message | None = None
) -> None:
    """Renders the dex screen on the panel."""
    entries = await get_dex(user_id=ctx.owner_id)
    await edit_fishing_message(
        interaction=interaction,
        embeds=build_dex_embeds(entries=entries),
        view=FishingDexView(ctx=ctx),
        message=message,
    )


async def _show_leaderboard(
    interaction: Interaction,
    ctx: FishingContext,
    metric: LeaderboardMetric,
    message: Message | None = None,
) -> None:
    """Renders a leaderboard screen on the panel."""
    if metric == "biggest":
        embeds = build_leaderboard_embeds(
            metric="biggest", biggest_rows=await leaderboard_biggest_catch()
        )
    else:
        embeds = build_leaderboard_embeds(
            metric="earned", earned_rows=await leaderboard_total_earned()
        )
    await edit_fishing_message(
        interaction=interaction,
        embeds=embeds,
        view=FishingLeaderboardView(ctx=ctx, metric=metric),
        message=message,
    )


# --- cast animation ----------------------------------------------------------


async def _animate(message: Message, embeds: list[Embed]) -> None:
    """Edits the panel to one animation beat, suppressing transient failures."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(
            message.edit(
                embeds=embeds,
                view=None,
                **embed_spacer_payload(embeds=embeds, is_edit=True, target=message),
            ),
            timeout=FINAL_EDIT_TIMEOUT_SECONDS,
        )


async def _run_cast(
    interaction: Interaction,
    ctx: FishingContext,
    rod: RodTier,
    bait: BaitType,
    source_view: "FishingPublicView",
) -> None:
    """Plays the cast animation, settles the cast, and shows the result."""
    source_view.stop()
    message = source_view.message or interaction.message
    if message is None:
        return
    await _animate(
        message=message,
        embeds=[build_cast_stage_embed(title="🎣 拋竿入水…", line="把釣線甩進水裡，靜待魚訊。")],
    )
    await asyncio.sleep(CAST_BEAT_DELAY_SECONDS)
    await _animate(
        message=message,
        embeds=[build_cast_stage_embed(title="🌊 等待中…", line="浮標在水面上輕輕晃動…")],
    )
    result = await execute_cast(
        user_id=ctx.owner_id, user_name=ctx.owner_name, bait_key=bait.key, rng=ctx.rng
    )
    if result.status != "ok" or result.outcome is None:
        loadout = await get_loadout(user_id=ctx.owner_id)
        station = FishingStationView(ctx=ctx)
        station.bind_message(message=message)
        embeds = build_station_embeds(loadout=loadout, avatar_url=ctx.owner_avatar_url)
        with contextlib.suppress(Exception):
            await message.edit(
                embeds=embeds,
                view=station,
                **embed_spacer_payload(embeds=embeds, is_edit=True, target=message),
            )
        await send_ephemeral_notice(
            interaction=interaction,
            content="這一竿沒辦法進行（魚餌或釣竿不足），請確認後再試。",
            log_message="fishing cast could not run",
        )
        return
    outcome = result.outcome
    reveal_title = "💨 空竿…" if outcome.miss else "❗ 有東西上鉤!"
    reveal_line = "線收回來，是空的。" if outcome.miss else "浮標被拉下去了，快收竿!"
    await asyncio.sleep(CAST_BEAT_DELAY_SECONDS)
    await _animate(
        message=message, embeds=[build_cast_stage_embed(title=reveal_title, line=reveal_line)]
    )
    await asyncio.sleep(CAST_REVEAL_DELAY_SECONDS)
    fallback_line = fishing_catch_fallback_line(rng=ctx.rng, rarity=outcome.rarity)
    embeds = build_cast_result_embeds(
        result=result,
        rod=rod,
        talk_line=fallback_line,
        system_name=ctx.system_name,
        system_avatar_url=ctx.system_avatar_url,
    )
    post_view = FishingPostCastView(ctx=ctx, last_bait_key=bait.key)
    post_view.bind_message(message=message)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(
            message.edit(
                embeds=embeds,
                view=post_view,
                **embed_spacer_payload(embeds=embeds, is_edit=True, target=message),
            ),
            timeout=FINAL_EDIT_TIMEOUT_SECONDS,
        )
    if ctx.narrator is not None and not outcome.miss and outcome.species is not None:
        post_view.spawn_background(
            coro=_refresh_catch_banter(
                message=message,
                ctx=ctx,
                result=result,
                rod=rod,
                post_view=post_view,
                fallback_line=fallback_line,
            )
        )


async def _refresh_catch_banter(  # noqa: PLR0913 -- background refresh needs full catch context
    message: Message,
    ctx: FishingContext,
    result: CastResult,
    rod: RodTier,
    post_view: "FishingPostCastView",
    fallback_line: str,
) -> None:
    """Upgrades the deterministic catch line with narrator banter in the background."""
    outcome = result.outcome
    if ctx.narrator is None or outcome is None or outcome.species is None:
        return
    line = await ctx.narrator.catch_fish(
        player_name=ctx.owner_name,
        species_name=outcome.species.name,
        rarity=outcome.species.rarity,
        size_mm=outcome.size_mm,
        sell_value=outcome.sell_value,
        fallback=fallback_line,
    )
    if line == fallback_line or post_view.is_finished():
        return
    embeds = build_cast_result_embeds(
        result=result,
        rod=rod,
        talk_line=line,
        system_name=ctx.system_name,
        system_avatar_url=ctx.system_avatar_url,
    )
    with contextlib.suppress(Exception):
        await asyncio.wait_for(
            message.edit(
                embeds=embeds,
                view=post_view,
                **embed_spacer_payload(embeds=embeds, is_edit=True, target=message),
            ),
            timeout=FINAL_EDIT_TIMEOUT_SECONDS,
        )


# --- views -------------------------------------------------------------------


class FishingPublicView(View):
    """Base view for fishing screens that own one public message."""

    def __init__(self, ctx: FishingContext, delete_on_timeout: bool = True) -> None:
        """Initializes a fishing screen with the shared session context."""
        super().__init__(timeout=FISHING_TIMEOUT_SECONDS)
        self.ctx = ctx
        self.delete_on_timeout = delete_on_timeout
        self.message: Message | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()

    def bind_message(self, message: Message | None) -> None:
        """Records the message this view should update or delete."""
        self.message = message

    def spawn_background(self, coro: Coroutine[object, object, None]) -> None:
        """Tracks a background coroutine so it is not garbage collected mid-flight."""
        task = asyncio.ensure_future(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Allows only the angler who opened this panel to operate it."""
        user = interaction.user
        if user is not None and user.id == self.ctx.owner_id:
            return True
        await send_ephemeral_notice(
            interaction=interaction,
            content="這個釣魚台只有發起者可以操作，請自己用 `/games fishing` 開一個新的。",
            log_message="Failed to send fishing owner mismatch notice",
        )
        return False

    async def on_timeout(self) -> None:
        """Deletes the tracked public message after 180 idle seconds."""
        if self.message is None or not self.delete_on_timeout:
            return
        await delete_public_message(message=self.message)


class FishingStationView(FishingPublicView):
    """Landing screen: cast, shop, inventory, dex, leaderboard."""

    @nextcord.ui.button(
        label="拋竿", emoji="🎣", style=ButtonStyle.primary, custom_id="fishing:cast", row=0
    )
    async def cast(self, _button: Button, interaction: Interaction) -> None:
        """Validates gear and starts a cast (choosing bait when several are owned)."""
        await interaction.response.defer()
        loadout = await get_loadout(user_id=self.ctx.owner_id)
        rod = ROD_BY_KEY.get(loadout.rod_key)
        if rod is None or loadout.rod_durability <= 0:
            await send_ephemeral_notice(
                interaction=interaction,
                content="你還沒有可用的釣竿，先到 🛒 商店買一支。",
                log_message="fishing cast without rod",
            )
            return
        owned = {key: qty for key, qty in loadout.baits.items() if qty > 0 and key in BAIT_BY_KEY}
        if not owned:
            await send_ephemeral_notice(
                interaction=interaction,
                content="你沒有魚餌了，先到 🛒 商店補貨。",
                log_message="fishing cast without bait",
            )
            return
        if len(owned) == 1:
            bait = BAIT_BY_KEY[next(iter(owned))]
            await _run_cast(
                interaction=interaction, ctx=self.ctx, rod=rod, bait=bait, source_view=self
            )
            return
        self.stop()
        await edit_fishing_message(
            interaction=interaction,
            embeds=[build_bait_choice_embed()],
            view=FishingBaitSelectView(ctx=self.ctx, rod=rod, owned=owned),
        )

    @nextcord.ui.button(
        label="商店", emoji="🛒", style=ButtonStyle.secondary, custom_id="fishing:shop", row=0
    )
    async def shop(self, _button: Button, interaction: Interaction) -> None:
        """Opens the shop screen."""
        await interaction.response.defer()
        self.stop()
        await _show_shop(interaction=interaction, ctx=self.ctx)

    @nextcord.ui.button(
        label="魚簍", emoji="🪣", style=ButtonStyle.secondary, custom_id="fishing:bag", row=1
    )
    async def inventory(self, _button: Button, interaction: Interaction) -> None:
        """Opens the inventory screen."""
        await interaction.response.defer()
        self.stop()
        await _show_inventory(interaction=interaction, ctx=self.ctx)

    @nextcord.ui.button(
        label="圖鑑", emoji="📖", style=ButtonStyle.secondary, custom_id="fishing:dex", row=1
    )
    async def dex(self, _button: Button, interaction: Interaction) -> None:
        """Opens the dex screen."""
        await interaction.response.defer()
        self.stop()
        await _show_dex(interaction=interaction, ctx=self.ctx)

    @nextcord.ui.button(
        label="排行榜", emoji="🏆", style=ButtonStyle.secondary, custom_id="fishing:board", row=2
    )
    async def leaderboard(self, _button: Button, interaction: Interaction) -> None:
        """Opens the leaderboard screen."""
        await interaction.response.defer()
        self.stop()
        await _show_leaderboard(interaction=interaction, ctx=self.ctx, metric="earned")

    @nextcord.ui.button(
        label="重新整理",
        emoji="🔄",
        style=ButtonStyle.secondary,
        custom_id="fishing:refresh",
        row=2,
    )
    async def refresh(self, _button: Button, interaction: Interaction) -> None:
        """Re-reads and re-renders the station."""
        await interaction.response.defer()
        self.stop()
        await _show_station(interaction=interaction, ctx=self.ctx)


class FishingShopView(FishingPublicView):
    """Shop screen with rod and bait purchase selects."""

    def __init__(self, ctx: FishingContext) -> None:
        """Populates the rod and bait select options."""
        super().__init__(ctx=ctx)
        rod_select = cast("StringSelect", self.rod_select)
        rod_select.options = [
            SelectOption(
                label=rod.name,
                value=rod.key,
                emoji=rod.emoji,
                description=f"{rod.cost:,} · 耐久{rod.durability} · 空竿{rod.miss_bps // 100}%",
            )
            for rod in ROD_TIERS
        ]
        bait_select = cast("StringSelect", self.bait_select)
        bait_select.options = [
            SelectOption(
                label=bait.name,
                value=bait.key,
                emoji=bait.emoji,
                description=f"{bait.cost:,} / 個",
            )
            for bait in BAIT_TYPES
        ]

    @nextcord.ui.string_select(
        placeholder="購買釣竿",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中", value="loading")],
        custom_id="fishing:shop:rod",
        row=0,
    )
    async def rod_select(self, select: StringSelect, interaction: Interaction) -> None:
        """Buys the selected rod, then returns to the station."""
        await interaction.response.defer()
        result = await buy_rod(
            user_id=self.ctx.owner_id,
            user_name=self.ctx.owner_name,
            rod_key=select.values[0],
            avatar_url=self.ctx.owner_avatar_url,
        )
        if result.status == "ok":
            self.stop()
            await _show_station(interaction=interaction, ctx=self.ctx)
            return
        if result.status == "insufficient":
            await send_ephemeral_notice(
                interaction=interaction,
                content=f"餘額不足，這支釣竿要 {result.cost:,} {CURRENCY_NAME}。",
                log_message="fishing rod insufficient",
            )
        else:
            await send_ephemeral_notice(
                interaction=interaction,
                content="購買失敗，請稍後再試。",
                log_message="fishing rod buy failed",
            )
        self.stop()
        await _show_shop(interaction=interaction, ctx=self.ctx)

    @nextcord.ui.string_select(
        placeholder="購買魚餌",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中", value="loading")],
        custom_id="fishing:shop:bait",
        row=1,
    )
    async def bait_select(self, select: StringSelect, interaction: Interaction) -> None:
        """Opens the bait quantity modal for the selected bait."""
        await interaction.response.send_modal(
            modal=BaitQuantityModal(
                ctx=self.ctx, bait_key=select.values[0], message=self.message, parent=self
            )
        )

    @nextcord.ui.button(
        label="返回釣魚台",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="fishing:shop:back",
        row=2,
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the station screen."""
        await interaction.response.defer()
        self.stop()
        await _show_station(interaction=interaction, ctx=self.ctx)


class BaitQuantityModal(Modal):
    """Quantity prompt for buying bait."""

    def __init__(
        self,
        ctx: FishingContext,
        bait_key: str,
        message: Message | None = None,
        parent: FishingShopView | None = None,
    ) -> None:
        """Initializes the modal with one quantity input."""
        bait = BAIT_BY_KEY.get(bait_key)
        super().__init__(title=f"購買 {bait.name if bait is not None else bait_key}")
        self.ctx = ctx
        self.bait_key = bait_key
        self.message = message
        self.parent = parent
        self.quantity: TextInput = TextInput(
            label="數量",
            placeholder=f"1 ~ {MAX_BAIT_PURCHASE_QUANTITY}",
            min_length=1,
            max_length=3,
            required=True,
            row=0,
        )
        self.add_item(item=self.quantity)

    async def callback(self, interaction: Interaction) -> None:
        """Validates the quantity, buys the bait, and returns to the station."""
        await interaction.response.defer()
        raw = (self.quantity.value or "").strip()
        if not raw.isdecimal():
            await send_ephemeral_notice(
                interaction=interaction,
                content="請輸入正整數數量。",
                log_message="fishing bait bad quantity",
            )
            return
        quantity = int(raw)
        if quantity <= 0 or quantity > MAX_BAIT_PURCHASE_QUANTITY:
            await send_ephemeral_notice(
                interaction=interaction,
                content=f"數量需在 1 ~ {MAX_BAIT_PURCHASE_QUANTITY} 之間。",
                log_message="fishing bait quantity out of range",
            )
            return
        result = await buy_bait(
            user_id=self.ctx.owner_id,
            user_name=self.ctx.owner_name,
            bait_key=self.bait_key,
            quantity=quantity,
            avatar_url=self.ctx.owner_avatar_url,
        )
        if self.parent is not None:
            self.parent.stop()
        if result.status == "insufficient":
            await send_ephemeral_notice(
                interaction=interaction,
                content=f"餘額不足，需要 {result.cost:,} {CURRENCY_NAME}。",
                log_message="fishing bait insufficient",
            )
            await _show_shop(interaction=interaction, ctx=self.ctx, message=self.message)
            return
        if result.status != "ok":
            await send_ephemeral_notice(
                interaction=interaction,
                content="購買失敗，請稍後再試。",
                log_message="fishing bait buy failed",
            )
            await _show_shop(interaction=interaction, ctx=self.ctx, message=self.message)
            return
        await _show_station(interaction=interaction, ctx=self.ctx, message=self.message)


class FishingBaitSelectView(FishingPublicView):
    """Prompt to choose which bait to cast with when several are owned."""

    def __init__(self, ctx: FishingContext, rod: RodTier, owned: dict[str, int]) -> None:
        """Populates the bait choices from the owned bait counts."""
        super().__init__(ctx=ctx)
        self.rod = rod
        select = cast("StringSelect", self.bait_pick)
        select.options = [
            SelectOption(
                label=f"{BAIT_BY_KEY[key].name} x{qty}", value=key, emoji=BAIT_BY_KEY[key].emoji
            )
            for key, qty in owned.items()
            if key in BAIT_BY_KEY
        ]

    @nextcord.ui.string_select(
        placeholder="選擇魚餌",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中", value="loading")],
        custom_id="fishing:cast:bait",
        row=0,
    )
    async def bait_pick(self, select: StringSelect, interaction: Interaction) -> None:
        """Runs a cast with the chosen bait."""
        await interaction.response.defer()
        bait = BAIT_BY_KEY.get(select.values[0])
        if bait is None:
            await _show_station(interaction=interaction, ctx=self.ctx)
            return
        await _run_cast(
            interaction=interaction, ctx=self.ctx, rod=self.rod, bait=bait, source_view=self
        )

    @nextcord.ui.button(
        label="返回釣魚台",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="fishing:cast:back",
        row=1,
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the station screen."""
        await interaction.response.defer()
        self.stop()
        await _show_station(interaction=interaction, ctx=self.ctx)


class FishingPostCastView(FishingPublicView):
    """Result screen after a cast: cast again, inventory, or back."""

    def __init__(self, ctx: FishingContext, last_bait_key: str) -> None:
        """Remembers the last bait so 再拋一次 can reuse it."""
        super().__init__(ctx=ctx)
        self.last_bait_key = last_bait_key

    @nextcord.ui.button(
        label="再拋一次",
        emoji="🎣",
        style=ButtonStyle.primary,
        custom_id="fishing:post:again",
        row=0,
    )
    async def again(self, _button: Button, interaction: Interaction) -> None:
        """Casts again, reusing the last bait when still available."""
        await interaction.response.defer()
        loadout = await get_loadout(user_id=self.ctx.owner_id)
        rod = ROD_BY_KEY.get(loadout.rod_key)
        if rod is None or loadout.rod_durability <= 0:
            await send_ephemeral_notice(
                interaction=interaction,
                content="你的釣竿斷了，先到 🛒 商店買一支。",
                log_message="fishing recast without rod",
            )
            return
        owned = {key: qty for key, qty in loadout.baits.items() if qty > 0 and key in BAIT_BY_KEY}
        if not owned:
            await send_ephemeral_notice(
                interaction=interaction,
                content="你沒有魚餌了，先到 🛒 商店補貨。",
                log_message="fishing recast without bait",
            )
            return
        bait_key = self.last_bait_key if self.last_bait_key in owned else next(iter(owned))
        await _run_cast(
            interaction=interaction,
            ctx=self.ctx,
            rod=rod,
            bait=BAIT_BY_KEY[bait_key],
            source_view=self,
        )

    @nextcord.ui.button(
        label="魚簍", emoji="🪣", style=ButtonStyle.secondary, custom_id="fishing:post:bag", row=0
    )
    async def inventory(self, _button: Button, interaction: Interaction) -> None:
        """Opens the inventory screen."""
        await interaction.response.defer()
        self.stop()
        await _show_inventory(interaction=interaction, ctx=self.ctx)

    @nextcord.ui.button(
        label="返回釣魚台",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="fishing:post:back",
        row=1,
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the station screen."""
        await interaction.response.defer()
        self.stop()
        await _show_station(interaction=interaction, ctx=self.ctx)


class FishingResultNavView(FishingPublicView):
    """Small nav shown after selling: inventory or back to station."""

    @nextcord.ui.button(
        label="魚簍", emoji="🪣", style=ButtonStyle.secondary, custom_id="fishing:sold:bag", row=0
    )
    async def inventory(self, _button: Button, interaction: Interaction) -> None:
        """Opens the inventory screen."""
        await interaction.response.defer()
        self.stop()
        await _show_inventory(interaction=interaction, ctx=self.ctx)

    @nextcord.ui.button(
        label="返回釣魚台",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="fishing:sold:back",
        row=0,
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the station screen."""
        await interaction.response.defer()
        self.stop()
        await _show_station(interaction=interaction, ctx=self.ctx)


class FishingInventoryView(FishingPublicView):
    """Inventory screen: sell all, sell one, or back."""

    def __init__(self, ctx: FishingContext, entries: Sequence[InventoryEntry]) -> None:
        """Populates the single-sell select, or removes selling controls when empty."""
        super().__init__(ctx=ctx)
        self.entries = entries
        if entries:
            select = cast("StringSelect", self.sell_one)
            select.options = [
                SelectOption(label=_inventory_option_label(entry=entry), value=str(entry.catch_id))
                for entry in entries[:_INVENTORY_PAGE_SIZE]
            ]
        else:
            self.remove_item(item=self.sell_all)
            self.remove_item(item=self.sell_one)

    @nextcord.ui.button(
        label="全部賣出",
        emoji="💵",
        style=ButtonStyle.primary,
        custom_id="fishing:bag:sellall",
        row=0,
    )
    async def sell_all(self, _button: Button, interaction: Interaction) -> None:
        """Sells every unsold fish."""
        await interaction.response.defer()
        result = await sell_fish(
            user_id=self.ctx.owner_id,
            user_name=self.ctx.owner_name,
            avatar_url=self.ctx.owner_avatar_url,
        )
        self.stop()
        await edit_fishing_message(
            interaction=interaction,
            embeds=build_sell_result_embeds(result=result),
            view=FishingResultNavView(ctx=self.ctx),
        )

    @nextcord.ui.string_select(
        placeholder="賣出單一漁獲",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中", value="loading")],
        custom_id="fishing:bag:sellone",
        row=1,
    )
    async def sell_one(self, select: StringSelect, interaction: Interaction) -> None:
        """Sells one selected fish and re-renders the inventory."""
        await interaction.response.defer()
        result = await sell_fish(
            user_id=self.ctx.owner_id,
            user_name=self.ctx.owner_name,
            catch_ids=[int(select.values[0])],
            avatar_url=self.ctx.owner_avatar_url,
        )
        if result.status == "ok":
            await send_ephemeral_notice(
                interaction=interaction,
                content=f"已賣出，得 {result.earned:,} {CURRENCY_NAME}。",
                log_message="fishing sell one",
            )
        self.stop()
        await _show_inventory(interaction=interaction, ctx=self.ctx)

    @nextcord.ui.button(
        label="返回釣魚台",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="fishing:bag:back",
        row=2,
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the station screen."""
        await interaction.response.defer()
        self.stop()
        await _show_station(interaction=interaction, ctx=self.ctx)


class FishingDexView(FishingPublicView):
    """Dex screen with a single back button."""

    @nextcord.ui.button(
        label="返回釣魚台",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="fishing:dex:back",
        row=0,
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the station screen."""
        await interaction.response.defer()
        self.stop()
        await _show_station(interaction=interaction, ctx=self.ctx)


class FishingLeaderboardView(FishingPublicView):
    """Leaderboard screen with metric switches and a back button."""

    def __init__(self, ctx: FishingContext, metric: LeaderboardMetric) -> None:
        """Remembers the active metric."""
        super().__init__(ctx=ctx)
        self.metric = metric

    @nextcord.ui.button(
        label="收益榜",
        emoji="💰",
        style=ButtonStyle.secondary,
        custom_id="fishing:lb:earned",
        row=0,
    )
    async def earned(self, _button: Button, interaction: Interaction) -> None:
        """Switches to the earnings leaderboard."""
        await interaction.response.defer()
        self.stop()
        await _show_leaderboard(interaction=interaction, ctx=self.ctx, metric="earned")

    @nextcord.ui.button(
        label="最大魚榜",
        emoji="📏",
        style=ButtonStyle.secondary,
        custom_id="fishing:lb:biggest",
        row=0,
    )
    async def biggest(self, _button: Button, interaction: Interaction) -> None:
        """Switches to the biggest-catch leaderboard."""
        await interaction.response.defer()
        self.stop()
        await _show_leaderboard(interaction=interaction, ctx=self.ctx, metric="biggest")

    @nextcord.ui.button(
        label="返回釣魚台",
        emoji="↩️",
        style=ButtonStyle.secondary,
        custom_id="fishing:lb:back",
        row=1,
    )
    async def back(self, _button: Button, interaction: Interaction) -> None:
        """Returns to the station screen."""
        await interaction.response.defer()
        self.stop()
        await _show_station(interaction=interaction, ctx=self.ctx)


async def open_fishing_panel(interaction: Interaction, ctx: FishingContext) -> None:
    """Sends the initial public fishing panel and tracks it for cleanup."""
    loadout = await get_loadout(user_id=ctx.owner_id)
    view = FishingStationView(ctx=ctx)
    embeds = build_station_embeds(loadout=loadout, avatar_url=ctx.owner_avatar_url)
    message = await interaction.followup.send(
        embeds=embeds,
        view=view,
        wait=True,
        **embed_spacer_payload(embeds=embeds, is_edit=False, target=interaction),
    )
    await track_public_message(message=message, user_name=ctx.owner_name)
    view.bind_message(message=message)


__all__ = ["FishingContext", "open_fishing_panel"]
