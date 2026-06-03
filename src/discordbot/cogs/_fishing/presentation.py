"""Embed builders and display helpers for the fishing mini-game.

Fish are shown as emoji. The big-emoji line in the reveal embed is the seam where
a rendered PNG (via `utils.pil_text`) could be attached later with
`attachment://...` without changing the surrounding layout.
"""

from nextcord import Embed

from discordbot.typings.fishing import (
    FISHING_BPS_DENOMINATOR,
    GearType,
    GearView,
    FishGrade,
    CastResult,
    CastStatus,
    CatchLogView,
    BaitStackView,
    FishingPanelData,
    FishGradeConfigView,
)
from discordbot.cogs._economy.presentation import CURRENCY_NAME, bold_currency, currency_text

FISHING_COLOR = 0x1ABC9C
FISHING_ERROR_COLOR = 0xE74C3C
_DURABILITY_BAR_SEGMENTS = 10
_BIG_CATCH_SIZE_BPS = 18_000
_MEDALS = ("🥇", "🥈", "🥉")


def _size_text(size_bps: int) -> str:
    """Formats a size multiplier from basis points, e.g. 1.42x."""
    return f"{size_bps / FISHING_BPS_DENOMINATOR:.2f}x"


def _grade_display(
    grade: FishGrade, grade_map: dict[FishGrade, FishGradeConfigView]
) -> tuple[int, str, str]:
    """Returns the (color, emoji, label) for a grade, with a neutral fallback."""
    config = grade_map.get(grade)
    if config is None:
        return FISHING_COLOR, "🐟", grade.value
    return config.color, config.emoji, config.label


def _durability_bar(remaining: int, total: int) -> str:
    """Renders a 10-segment durability bar."""
    if total <= 0:
        return ""
    filled = max(
        0, min(_DURABILITY_BAR_SEGMENTS, round(remaining / total * _DURABILITY_BAR_SEGMENTS))
    )
    return "▰" * filled + "▱" * (_DURABILITY_BAR_SEGMENTS - filled)


def _rod_line(panel: FishingPanelData) -> str:
    """Builds the equipped-rod line for the panel."""
    rod = panel.angler.rod
    if rod is None:
        return "🎣 釣竿：未持有（去商店買一支）"
    if panel.angler.durability_remaining <= 0:
        return f"🎣 釣竿：{rod.emoji} {rod.name}（已損壞，去商店買新的）"
    bar = _durability_bar(remaining=panel.angler.durability_remaining, total=rod.durability)
    return (
        f"🎣 釣竿：{rod.emoji} {rod.name}  {bar} "
        f"{panel.angler.durability_remaining}/{rod.durability}"
    )


def _bait_line(baits: tuple[BaitStackView, ...]) -> str:
    """Builds the owned-bait line for the panel."""
    if not baits:
        return "🎒 魚餌：（沒有魚餌，去商店補貨）"
    stacks = "   ".join(f"{stack.emoji} {stack.name} x{stack.quantity}" for stack in baits)
    return f"🎒 魚餌：{stacks}"


def _last_catch_line(
    last_catch: CatchLogView | None, grade_map: dict[FishGrade, FishGradeConfigView]
) -> str:
    """Builds the most-recent-catch line for the panel."""
    if last_catch is None:
        return "🐟 最近漁獲：尚未有漁獲，快去拋竿"
    _, _, label = _grade_display(grade=last_catch.grade, grade_map=grade_map)
    return (
        f"🐟 最近漁獲：{last_catch.emoji} {last_catch.species_name}"
        f"（{label}・{_size_text(size_bps=last_catch.size_bps)}）"
        f" {bold_currency(amount=last_catch.value, signed=True, compact=True)}"
    )


def build_panel_embed(
    panel: FishingPanelData, grade_map: dict[FishGrade, FishGradeConfigView]
) -> Embed:
    """Builds the main fishing panel embed."""
    description = "\n".join([
        f"💰 餘額：{bold_currency(amount=panel.balance, compact=True)}",
        "",
        _rod_line(panel=panel),
        _bait_line(baits=panel.baits),
        _last_catch_line(last_catch=panel.last_catch, grade_map=grade_map),
    ])
    embed = Embed(title="🎣 釣魚", description=description, color=FISHING_COLOR)
    embed.set_footer(text=f"以小搏大，把{CURRENCY_NAME}釣回來")
    return embed


def _gear_shop_line(gear: GearView) -> str:
    """Builds one shop line for a rod or bait."""
    price = currency_text(amount=gear.price, compact=True)
    rarity = f"稀有+{gear.rarity_shift_bps / 100:.1f}%"
    if gear.gear_type == GearType.ROD:
        return f"  {gear.emoji} {gear.name} · {price} · 耐久{gear.durability} {rarity}"
    value_bonus = f"價值+{gear.value_bonus_bps / 100:.1f}%"
    return f"  {gear.emoji} {gear.name} · {price} · {rarity} {value_bonus}"


def build_shop_embed(
    balance: int, rods: tuple[GearView, ...], baits: tuple[GearView, ...], notice: str = ""
) -> Embed:
    """Builds the fishing shop embed listing rods and baits."""
    lines = [f"💰 餘額：{bold_currency(amount=balance, compact=True)}"]
    if notice:
        lines.append(notice)
    lines.append("")
    lines.append("🎣 釣竿（買新竿會替換目前的竿子）")
    lines.extend(_gear_shop_line(gear=rod) for rod in rods)
    lines.append("")
    lines.append("🪱 魚餌（每個價格）")
    lines.extend(_gear_shop_line(gear=bait) for bait in baits)
    return Embed(title="🛒 釣具商店", description="\n".join(lines), color=FISHING_COLOR)


def build_casting_embed() -> Embed:
    """Builds the first-beat casting animation embed."""
    description = "🌊〰️〰️〰️🎣〰️〰️〰️\n浮標正在等待……"
    return Embed(title="🎣 拋竿中…", description=description, color=FISHING_COLOR)


def _reveal_status_note(result: CastResult) -> str:
    """Returns an optional trailing note line for the reveal embed."""
    notes: list[str] = []
    if result.rod_broke:
        notes.append("-# 🎣 你的竿子斷了! 去商店買一支新的")
    if result.status == CastStatus.PAYOUT_DEFERRED:
        notes.append("-# 入帳稍有延遲，稍後會補上")
    return "\n".join(notes)


def _reveal_bait_line(result: CastResult, panel: FishingPanelData) -> str:
    """Builds the remaining-bait line for the reveal embed."""
    for stack in panel.baits:
        if stack.bait_id == result.bait_id:
            return f"🪱 剩餘魚餌：{stack.name} x{stack.quantity}"
    return "🪱 剩餘魚餌：（這種魚餌用完了）"


def _reveal_rod_line(result: CastResult, panel: FishingPanelData) -> str:
    """Builds the rod durability line for the reveal embed."""
    rod = panel.angler.rod
    if result.rod_broke or rod is None:
        return "🎣 釣竿：已損壞"
    bar = _durability_bar(remaining=result.durability_remaining, total=rod.durability)
    return f"🎣 釣竿耐久：{bar} {result.durability_remaining}/{rod.durability}"


def build_reveal_embed(
    result: CastResult, panel: FishingPanelData, grade_map: dict[FishGrade, FishGradeConfigView]
) -> Embed:
    """Builds the second-beat catch reveal embed."""
    roll = result.roll
    if roll is None:
        return build_error_embed(message="這一竿沒有結果，請再試一次")
    color, grade_emoji, label = _grade_display(grade=roll.grade, grade_map=grade_map)
    big = "（大物!）" if roll.size_bps >= _BIG_CATCH_SIZE_BPS else ""
    lines = [
        f"# {roll.emoji}",
        f"## {roll.species_name}",
        f"{grade_emoji} {label}・{_size_text(size_bps=roll.size_bps)}{big}",
        "",
        f"💴 漁獲價值：{bold_currency(amount=result.payout, signed=True, compact=True)}",
        f"💰 目前餘額：{currency_text(amount=panel.balance, compact=True)}",
        _reveal_rod_line(result=result, panel=panel),
        _reveal_bait_line(result=result, panel=panel),
    ]
    note = _reveal_status_note(result=result)
    if note:
        lines.append(note)
    return Embed(title="✨ 上鉤了!", description="\n".join(lines), color=color)


def build_leaderboard_embed(
    catches: tuple[CatchLogView, ...], grade_map: dict[FishGrade, FishGradeConfigView]
) -> Embed:
    """Builds the top-single-catches leaderboard embed."""
    if not catches:
        return Embed(
            title="🏆 釣魚排行榜 · 最大單筆漁獲",
            description="還沒有人釣到魚，快去當第一個",
            color=FISHING_COLOR,
        )
    lines: list[str] = []
    for index, catch in enumerate(catches):
        rank = _MEDALS[index] if index < len(_MEDALS) else f"{index + 1}."
        _, _, label = _grade_display(grade=catch.grade, grade_map=grade_map)
        stamp = catch.created_at.strftime("%m/%d")
        lines.append(
            f"{rank} {catch.emoji} {catch.species_name} {label} "
            f"{currency_text(amount=catch.value, compact=True)} — {catch.user_name} {stamp}"
        )
    return Embed(
        title="🏆 釣魚排行榜 · 最大單筆漁獲", description="\n".join(lines), color=FISHING_COLOR
    )


def build_stats_embed(panel: FishingPanelData, recent: tuple[CatchLogView, ...]) -> Embed:
    """Builds the personal stats and recent-catch embed."""
    angler = panel.angler
    lines = [
        f"🎣 總拋竿數：{angler.total_casts:,}",
        f"🐟 漁獲總價值：{currency_text(amount=angler.total_catch_value, compact=True)}",
        f"🏅 最高單筆：{currency_text(amount=angler.best_catch_value, compact=True)}",
        f"🛒 購買支出：{currency_text(amount=angler.total_spent_on_gear, compact=True)}",
    ]
    if recent:
        lines.append("")
        lines.append("最近漁獲：")
        for catch in recent:
            lines.append(
                f"  {catch.emoji} {catch.species_name}"
                f"（{_size_text(size_bps=catch.size_bps)}）"
                f" {currency_text(amount=catch.value, signed=True, compact=True)}"
            )
    return Embed(title="📊 我的釣魚紀錄", description="\n".join(lines), color=FISHING_COLOR)


def build_error_embed(message: str) -> Embed:
    """Builds a generic fishing error or notice embed."""
    return Embed(title="🎣 釣魚", description=message, color=FISHING_ERROR_COLOR)


def build_bait_select_embed(panel: FishingPanelData) -> Embed:
    """Builds the prompt for choosing which bait to cast with."""
    return Embed(
        title="🎣 選擇魚餌", description="你有多種魚餌，選一個來拋竿", color=FISHING_COLOR
    )


__all__ = [
    "FISHING_COLOR",
    "FISHING_ERROR_COLOR",
    "build_bait_select_embed",
    "build_casting_embed",
    "build_error_embed",
    "build_leaderboard_embed",
    "build_panel_embed",
    "build_reveal_embed",
    "build_shop_embed",
    "build_stats_embed",
]
