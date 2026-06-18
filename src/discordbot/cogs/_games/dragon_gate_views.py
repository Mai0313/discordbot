"""Interactive components for multiplayer 射龍門 sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast
import asyncio
import contextlib

import logfire
import nextcord
from nextcord import Embed, Message, ButtonStyle, Interaction
from nextcord.ui import Item, View, Modal, Button, TextInput, StringSelect

from discordbot.typings.games import GameParticipant, DragonGatePlayerResult
from discordbot.cogs._games.lobby import (
    PrepareParticipant,
    RefreshParticipants,
    BaseJackpotLobbyView,
)
from discordbot.utils.number_text import compact_amount
from discordbot.cogs._games.wagers import parse_wager_amount
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import schedule_public_message_delete
from discordbot.cogs._economy.database import (
    get_balance,
    get_jackpot_snapshot,
    apply_jackpot_settlement,
)
from discordbot.cogs._games.dragon_gate import (
    ANTE,
    GAME_ID,
    DragonGateTurn,
    DragonGateError,
    DragonGateRound,
    DragonGateOutcome,
    DragonGateDirection,
    DragonGateTurnError,
    DragonGateTurnResult,
    DragonGateBetRangeError,
    DragonGateTableFinishedError,
    DragonGatePairChoiceRequiredError,
    DragonGateParticipantUnknownError,
    DragonGatePairChoiceUnavailableError,
)
from discordbot.cogs._games.interactions import set_view_item_visible, edit_message_with_retry
from discordbot.cogs._games.presentation import (
    WIN_COLOR,
    LOSE_COLOR,
    PUSH_COLOR,
    POT_FIELD_EMOJI,
    TURN_FIELD_EMOJI,
    WIN_RESULT_EMOJI,
    LAST_HAND_FIELD_EMOJI,
    FINISH_REASON_FIELD_EMOJI,
    LOBBY_PLAYERS_FIELD_EMOJI,
    metadata_line,
    lobby_participant_line,
)
from discordbot.utils.owned_message_views import send_ephemeral_notice
from discordbot.cogs._economy.presentation import amount_code, currency_text

if TYPE_CHECKING:
    from random import Random

    from discordbot.typings.economy import JackpotSnapshot

DRAGON_GATE_ACTION_TIMEOUT_SECONDS = 180
DRAGON_GATE_VISIBLE_PLAYER_LINES = 20
DRAGON_GATE_FINAL_EDIT_TIMEOUT_SECONDS: Final[float] = 8.0


def _dragon_gate_table_edit_kwargs(
    *, embeds: list[Embed], view: View | None, target: object | None = None
) -> dict[str, Any]:
    """Builds the shared edit payload for 射龍門 table renders."""
    return {
        "embeds": embeds,
        "view": view,
        **embed_spacer_payload(embeds=embeds, is_edit=True, target=target),
    }


def _participant_lines(participants: list[GameParticipant]) -> str:
    """Formats visible lobby participants and hidden overflow count."""
    lines: list[str] = []
    visible = participants[:DRAGON_GATE_VISIBLE_PLAYER_LINES]
    for index, participant in enumerate(visible, start=1):
        lines.append(lobby_participant_line(index=index, display_name=participant.display_name))
    hidden_count = len(participants) - len(visible)
    if hidden_count > 0:
        lines.append(f"-# 還有 {hidden_count} 位玩家")
    return "\n".join(lines)


def _direction_label(direction: DragonGateDirection | None) -> str:
    """Returns the display label for a pair-gate direction choice."""
    if direction == "higher":
        return "⬆️ 猜大"
    if direction == "lower":
        return "⬇️ 猜小"
    return "尚未選擇"


def _outcome_presentation(outcome: DragonGateOutcome) -> tuple[str, int]:
    """Returns the display label and embed color for a turn outcome."""
    values: dict[DragonGateOutcome, tuple[str, int]] = {
        "gate_win": ("✅ 射中", WIN_COLOR),
        "outside_lose": ("❌ 射偏", LOSE_COLOR),
        "pillar_hit": ("🧱 撞柱", LOSE_COLOR),
        "pair_win": ("✅ 猜中", WIN_COLOR),
        "pair_lose": ("❌ 猜錯", LOSE_COLOR),
        "pair_pillar_hit": ("🧱 同點撞柱", LOSE_COLOR),
    }
    return values[outcome]


def _result_line(result: DragonGateTurnResult) -> str:
    """Formats the latest resolved turn for the main embed."""
    outcome_label, _color = _outcome_presentation(outcome=result.outcome)
    direction = f" · {_direction_label(direction=result.direction)}" if result.direction else ""
    pillars = " ".join(str(card) for card in result.pillars)
    return (
        f"# {pillars}  →  {result.third_card}\n"
        f"**{result.participant.display_name}** (第 {result.turn_number} 手){direction}\n"
        f"### {outcome_label} {amount_code(amount=result.delta, signed=True, compact=True)}"
    )


def _history_code_lines(history: list[DragonGateTurnResult]) -> list[str]:
    """Builds monospace history lines for completed turns."""
    lines: list[str] = []
    for result in history:
        outcome_label, _color = _outcome_presentation(outcome=result.outcome)
        pillars = " ".join(str(card) for card in result.pillars)
        lines.append(
            f"第 {result.turn_number} 手 {result.participant.account_name}: "
            f"{pillars} → {result.third_card}  {outcome_label} "
            f"{compact_amount(amount=result.delta, signed=True)}"
        )
    return lines


def _scoreboard_code_lines(round_state: DragonGateRound) -> list[str]:
    """Builds monospace scoreboard lines from current table deltas."""
    lines: list[str] = []
    for participant in round_state.participants[:DRAGON_GATE_VISIBLE_PLAYER_LINES]:
        delta = round_state.player_delta(user_id=participant.user_id)
        suffix = " (已離桌)" if participant.user_id in round_state.withdrawn_user_ids else ""
        lines.append(
            f"{participant.account_name}{suffix}: {compact_amount(amount=delta, signed=True)}"
        )
    return lines


def _last_result_line(result: DragonGateTurnResult) -> str:
    """One-line summary of the previous turn for placement above the current state."""
    outcome_label, _color = _outcome_presentation(outcome=result.outcome)
    pillars = " ".join(str(card) for card in result.pillars)
    return (
        f"**{result.participant.display_name}**: "
        f"{pillars} → `{result.third_card}`  {outcome_label} "
        f"{amount_code(amount=result.delta, signed=True, compact=True)}"
    )


def _gate_description_block(turn: DragonGateTurn) -> str:
    """Formats the active gate and pair-choice hint for the main embed."""
    left_card, right_card = turn.pillars
    cards = f"# {left_card} ------- {right_card}"
    if turn.is_pair:
        if turn.direction is None:
            hint = "> 請先按「同點猜大」或「同點猜小」"
            return f"{cards}\n### ⚠️ 同點門柱\n{hint}"
        return f"{cards}\n### {_direction_label(direction=turn.direction)}"
    return cards


def _table_color(results: list[DragonGatePlayerResult]) -> int:
    """Returns the final embed color from the table's net result."""
    total_delta = sum(result.delta for result in results)
    if total_delta > 0:
        return WIN_COLOR
    if total_delta < 0:
        return LOSE_COLOR
    return PUSH_COLOR


def _settlement_result_heading(delta: int) -> str:
    """Formats one player's final net delta as an embed heading."""
    if delta > 0:
        return f"## {WIN_RESULT_EMOJI} {amount_code(amount=delta, signed=True, compact=True)}"
    if delta < 0:
        return f"## 💸 {amount_code(amount=delta, signed=True, compact=True)}"
    return "## 持平"


def _final_title(results: list[DragonGatePlayerResult]) -> str:
    """Builds the final 射龍門 title for single or multiplayer results."""
    if len(results) == 1:
        delta = results[0].delta
        if delta > 0:
            return f"♦️ 射龍門 · {WIN_RESULT_EMOJI} {amount_code(amount=delta, signed=True, compact=True)}"
        if delta < 0:
            return f"♦️ 射龍門 · 💸 {amount_code(amount=delta, compact=True)}"
        return "♦️ 射龍門 · 持平"
    total_delta = sum(result.delta for result in results)
    wins = sum(1 for result in results if result.delta > 0)
    losses = sum(1 for result in results if result.delta < 0)
    if total_delta > 0:
        prefix = f"{WIN_RESULT_EMOJI} "
    elif total_delta < 0:
        prefix = "💸 "
    else:
        prefix = ""
    return (
        f"♦️ 射龍門 · {prefix}{wins} 贏 {losses} 輸 · 淨 "
        f"{amount_code(amount=total_delta, signed=True, compact=True)}"
    )


def build_dragon_gate_lobby_embed(
    owner: GameParticipant,
    participants: list[GameParticipant],
    jackpot: int,
    status: str = "等待玩家加入",
) -> Embed:
    """Builds the lobby embed shown before a 射龍門 table starts."""
    embed = Embed(title="♦️ 射龍門 · 開桌準備", color=PUSH_COLOR)
    if status and status != "等待玩家加入":
        embed.description = status
    embed.add_field(
        name=f"{LOBBY_PLAYERS_FIELD_EMOJI} 桌上玩家 ({len(participants)})",
        value=_participant_lines(participants=participants),
        inline=False,
    )
    embed.add_field(
        name=f"{POT_FIELD_EMOJI} 彩金池 (跨桌累積)",
        value=amount_code(amount=jackpot, compact=True),
        inline=False,
    )
    if owner.avatar_url:
        embed.set_thumbnail(url=owner.avatar_url)
    embed.set_footer(text=f"入場費 {currency_text(amount=ANTE, compact=True)} 進彩金池")
    return embed


def build_dragon_gate_in_progress_embed(round_state: DragonGateRound, jackpot: int) -> Embed:
    """Builds the active 射龍門 table embed (current state only)."""
    active_turn = round_state.active_turn

    description_parts: list[str] = []
    if round_state.last_result is not None:
        description_parts.append(_last_result_line(result=round_state.last_result))
        description_parts.append("")
    description_parts.append(f"## {POT_FIELD_EMOJI} 彩金池 {compact_amount(amount=jackpot)}")
    if active_turn is not None:
        description_parts.append("")
        description_parts.append(_gate_description_block(turn=active_turn))
        description_parts.append("")
        description_parts.append(
            f"## {TURN_FIELD_EMOJI} 輪到 {active_turn.participant.display_name}"
        )

    embed = Embed(
        title=f"♦️ 射龍門 · 第 {round_state.turn_number} 手",
        description="\n".join(description_parts),
        color=PUSH_COLOR,
    )
    if round_state.participants and round_state.participants[0].avatar_url:
        embed.set_thumbnail(url=round_state.participants[0].avatar_url)
    embed.set_footer(text=f"{DRAGON_GATE_ACTION_TIMEOUT_SECONDS} 秒無互動會結束牌桌")
    return embed


def build_dragon_gate_history_embed(
    history: list[DragonGateTurnResult], round_state: DragonGateRound
) -> Embed | None:
    """Builds an auxiliary embed with each turn's history and cumulative scoreboard.

    Returns `None` when there is nothing to show (no history and zero deltas).
    """
    has_deltas = any(
        round_state.player_delta(user_id=participant.user_id) != 0
        for participant in round_state.participants
    )
    if not history and not has_deltas:
        return None

    lines: list[str] = []
    if history:
        lines.extend(_history_code_lines(history=history))
    if history and round_state.participants:
        lines.append("")
    if round_state.participants:
        lines.extend(_scoreboard_code_lines(round_state=round_state))

    code_block = "```\n" + "\n".join(lines) + "\n```"
    return Embed(description=f"**紀錄:**\n{code_block}", color=PUSH_COLOR)


def build_dragon_gate_final_embed(
    round_state: DragonGateRound, results: list[DragonGatePlayerResult], jackpot: int, reason: str
) -> Embed:
    """Builds the final embed for a settled 射龍門 table."""
    description_parts: list[str] = [f"### {FINISH_REASON_FIELD_EMOJI} 結束原因", reason, ""]
    description_parts.append(f"## {POT_FIELD_EMOJI} 彩金池 {compact_amount(amount=jackpot)}")
    description_parts.append("")
    if round_state.last_result is not None:
        description_parts.append(f"### {LAST_HAND_FIELD_EMOJI} 最後一手")
        description_parts.append(_result_line(result=round_state.last_result))
        description_parts.append("")
    description_parts.append("### 結算")
    for result in results[:DRAGON_GATE_VISIBLE_PLAYER_LINES]:
        balance_text = f"餘額 {amount_code(amount=result.final_balance, compact=True)}"
        if result.withdrawn:
            balance_text += " · 已離桌"
        if result.refunded_to_pool > 0:
            balance_text += (
                f" · 逆贏退回 {amount_code(amount=result.refunded_to_pool, compact=True)}"
            )
        description_parts.append(f"**{result.participant.display_name}**")
        description_parts.append(_settlement_result_heading(delta=result.delta))
        description_parts.append(metadata_line(text=balance_text))
    hidden_count = len(results) - DRAGON_GATE_VISIBLE_PLAYER_LINES
    if hidden_count > 0:
        description_parts.append(f"-# 還有 {hidden_count} 位玩家已結算")

    embed = Embed(
        title=_final_title(results=results),
        description="\n".join(description_parts),
        color=_table_color(results=results),
    )
    if round_state.participants and round_state.participants[0].avatar_url:
        embed.set_thumbnail(url=round_state.participants[0].avatar_url)
    return embed


class DragonGateLobbyView(BaseJackpotLobbyView):
    """Join / leave / start lobby for a 射龍門 game session."""

    game_id = GAME_ID
    ante = ANTE

    def __init__(  # noqa: PLR0913 -- lobby owns all table dependencies
        self,
        owner: GameParticipant,
        rng: Random,
        system_name: str,
        system_avatar_url: str,
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
        initial_jackpot: int,
        initial_jackpot_generation: int | None = None,
    ) -> None:
        """Initializes a 射龍門 lobby with the current jackpot snapshot."""
        super().__init__(
            owner=owner,
            rng=rng,
            system_name=system_name,
            system_avatar_url=system_avatar_url,
            prepare_participant=prepare_participant,
            refresh_participants=refresh_participants,
            initial_jackpot=initial_jackpot,
            timeout=DRAGON_GATE_ACTION_TIMEOUT_SECONDS,
            initial_jackpot_generation=initial_jackpot_generation,
        )

    def _build_lobby_embed(self, status: str = "等待玩家加入") -> Embed:
        """Builds the 射龍門 lobby embed from participants and jackpot state."""
        return build_dragon_gate_lobby_embed(
            owner=self.owner,
            participants=self.participants,
            jackpot=self._jackpot_snapshot,
            status=status,
        )

    async def _start_game_after_antes(
        self, message: Message, final_balances: dict[int, int]
    ) -> None:
        """Starts the active table after all lobby antes have been charged."""
        round_state = DragonGateRound.from_participants(
            rng=self.rng, participants=self.participants
        )
        view = DragonGateView(
            round_state=round_state,
            owner=self.owner,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
            jackpot_snapshot=self._jackpot_snapshot,
            jackpot_generation=self._jackpot_generation,
            final_balances=final_balances,
        )
        view.message = message
        view.sync_controls()
        embeds = view.in_progress_embeds()
        await edit_message_with_retry(
            message=message,
            kwargs_factory=lambda: _dragon_gate_table_edit_kwargs(
                embeds=embeds, view=view, target=message
            ),
        )


class DragonGateView(View):
    """High / low buttons, bet select, and leave button for an active 射龍門 table."""

    def __init__(  # noqa: PLR0913 -- view needs round, jackpot, and initial balances
        self,
        round_state: DragonGateRound,
        owner: GameParticipant,
        system_name: str,
        jackpot_snapshot: int,
        final_balances: dict[int, int],
        system_avatar_url: str = "",
        jackpot_generation: int | None = None,
    ) -> None:
        """Initializes the active 射龍門 table view."""
        super().__init__(timeout=DRAGON_GATE_ACTION_TIMEOUT_SECONDS)
        self.round_state = round_state
        self.owner = owner
        self.system_name = system_name
        self.system_avatar_url = system_avatar_url
        self.message: Message | None = None
        self._round_lock = asyncio.Lock()
        self._settled = False
        self._history: list[DragonGateTurnResult] = []
        self._jackpot_snapshot = jackpot_snapshot
        self._jackpot_generation = jackpot_generation
        self._final_balances: dict[int, int] = dict(final_balances)
        self._refunded_to_pool: dict[int, int] = {}
        self._buttons: dict[str, Button] = {
            "dg:higher": cast("Button", self.choose_higher),
            "dg:lower": cast("Button", self.choose_lower),
            "dg:leave": cast("Button", self.leave_table),
        }
        self._selects: dict[str, StringSelect] = {"dg:bet": cast("StringSelect", self.bet_select)}
        self.sync_controls()

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Restricts bet/direction to the active player; leave is open to all seated."""
        if self._settled or interaction.user is None:
            return False
        data = (
            cast("dict[str, Any]", interaction.data) if isinstance(interaction.data, dict) else {}
        )
        custom_id_value = data.get("custom_id", "")
        custom_id = custom_id_value if isinstance(custom_id_value, str) else ""
        user_id = interaction.user.id
        if custom_id == "dg:leave":
            if self.round_state.is_active(user_id=user_id):
                return True
            notice = "你不在這桌"
        else:
            active_turn = self.round_state.active_turn
            if active_turn is not None and user_id == active_turn.participant.user_id:
                return True
            notice = (
                f"現在輪到 {active_turn.participant.display_name}"
                if active_turn is not None
                else "這桌已經不能操作了"
            )
        await interaction.response.send_message(content=notice, ephemeral=True)
        return False

    async def on_timeout(self) -> None:
        """Finalises an abandoned table; refunds in-flight winnings into the pool."""
        if self.message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            await self._refund_remaining_winners_locked()
            await self._finalize_locked(message=self.message, reason="逾時未操作")

    @nextcord.ui.button(
        label="同點猜大", emoji="⬆️", style=ButtonStyle.secondary, custom_id="dg:higher", row=1
    )
    async def choose_higher(self, _button: Button, interaction: Interaction) -> None:
        """Chooses higher for a same-point gate."""
        await self._choose_direction(interaction=interaction, direction="higher")

    @nextcord.ui.button(
        label="同點猜小", emoji="⬇️", style=ButtonStyle.secondary, custom_id="dg:lower", row=1
    )
    async def choose_lower(self, _button: Button, interaction: Interaction) -> None:
        """Chooses lower for a same-point gate."""
        await self._choose_direction(interaction=interaction, direction="lower")

    @nextcord.ui.string_select(
        placeholder="🪙 選擇下注金額",
        custom_id="dg:bet",
        min_values=1,
        max_values=1,
        options=[
            nextcord.SelectOption(label="底注", value="min", emoji="🪙"),
            nextcord.SelectOption(label="全池", value="max", emoji="💰"),
            nextcord.SelectOption(label="自訂", value="custom", emoji="✏️"),
        ],
        row=2,
    )
    async def bet_select(self, select: StringSelect, interaction: Interaction) -> None:
        """Routes the bet select choice to a fixed amount or a custom modal."""
        await self._handle_bet_choice(choice=select.values[0], interaction=interaction)

    @nextcord.ui.button(
        label="離桌", emoji="🚪", style=ButtonStyle.danger, custom_id="dg:leave", row=0
    )
    async def leave_table(self, _button: Button, interaction: Interaction) -> None:
        """Lets any seated player withdraw mid-table without ending the round."""
        await self._handle_leave(interaction=interaction)

    async def _handle_bet_choice(self, choice: str, interaction: Interaction) -> None:
        """Routes a select-menu choice to a fixed bet or custom modal."""
        if choice == "custom":
            if self.round_state.needs_pair_choice():
                await interaction.response.send_message(
                    content="同點門柱要先猜大或猜小", ephemeral=True
                )
                return
            modal = DragonGateBetModal(
                view=self,
                minimum=self.round_state.current_min_bet(jackpot=self._jackpot_snapshot),
                maximum=self._active_max_bet(),
            )
            await interaction.response.send_modal(modal=modal)
            return
        if choice == "min":
            amount = self.round_state.current_min_bet(jackpot=self._jackpot_snapshot)
        else:
            amount = self._active_max_bet()
        await self._place_select_bet(interaction=interaction, amount=amount)

    async def submit_custom_bet(self, interaction: Interaction, raw_amount: str | None) -> None:
        """Handles the custom bet modal submission."""
        amount = parse_wager_amount(raw_amount=raw_amount)
        if amount is None:
            await interaction.response.send_message(content="下注金額要是整數", ephemeral=True)
            return
        await interaction.response.defer()
        await self._place_bet_locked_by_interaction(interaction=interaction, amount=amount)

    def sync_controls(self) -> None:
        """Updates button labels and select options from the current table state."""
        active = self.round_state.active_turn
        needs_pair_choice = not self._settled and self.round_state.needs_pair_choice()
        minimum = self.round_state.current_min_bet(jackpot=self._jackpot_snapshot)
        maximum = self._active_max_bet()
        can_bet = (
            not self._settled
            and active is not None
            and not needs_pair_choice
            and maximum >= minimum
        )

        higher_button = self._buttons["dg:higher"]
        lower_button = self._buttons["dg:lower"]
        higher_button.disabled = False
        lower_button.disabled = False

        if active is not None and active.is_pair and active.direction is not None:
            higher_button.label = "同點猜大 ✓" if active.direction == "higher" else "同點猜大"
            lower_button.label = "同點猜小 ✓" if active.direction == "lower" else "同點猜小"
        else:
            higher_button.label = "同點猜大"
            lower_button.label = "同點猜小"
        set_view_item_visible(view=self, item=higher_button, visible=needs_pair_choice)
        set_view_item_visible(view=self, item=lower_button, visible=needs_pair_choice)

        bet_select = self._selects["dg:bet"]
        bet_select.disabled = False
        if needs_pair_choice:
            bet_select.placeholder = "⚠️ 請先選擇猜大或猜小"
        else:
            bet_select.placeholder = "🪙 選擇下注金額"
        bet_select.options = [
            nextcord.SelectOption(
                label=f"底注 {compact_amount(amount=minimum)}",
                value="min",
                emoji="🪙",
                description="最低下注金額",
            ),
            nextcord.SelectOption(
                label=f"全池 {compact_amount(amount=maximum)}",
                value="max",
                emoji="💰",
                description="一把定生死, 清空彩金池",
            ),
            nextcord.SelectOption(
                label="自訂", value="custom", emoji="✏️", description="彈出視窗輸入精確金額"
            ),
        ]
        set_view_item_visible(view=self, item=bet_select, visible=can_bet)

        leave_button = self._buttons["dg:leave"]
        leave_button.disabled = False
        has_active_participant = bool(self.round_state.active_participants())
        set_view_item_visible(
            view=self, item=leave_button, visible=not self._settled and has_active_participant
        )

    async def _choose_direction(
        self, interaction: Interaction, direction: DragonGateDirection
    ) -> None:
        """Stores a high or low choice for the active pair gate."""
        await interaction.response.defer()
        if interaction.user is None or interaction.message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            try:
                self.round_state.choose_pair_direction(
                    user_id=interaction.user.id, direction=direction
                )
            except DragonGateTableFinishedError:
                await self._send_notice(interaction=interaction, content="這桌已經不能操作了")
                return
            except DragonGateTurnError:
                await self._send_notice(
                    interaction=interaction, content=self._current_turn_notice()
                )
                return
            except DragonGatePairChoiceUnavailableError:
                await self._send_notice(interaction=interaction, content="這手不需要猜大小")
                return
            self.sync_controls()
            await interaction.message.edit(
                **_dragon_gate_table_edit_kwargs(
                    embeds=self.in_progress_embeds(), view=self, target=interaction.message
                )
            )

    def _max_bet_for(self, user_id: int | None) -> int:
        """Returns the max legal bet bounded by the pool, the single-bet cap, and balance.

        Losses already clamp at the player's balance, so bounding the bet by the
        same balance closes the asymmetric free option where a low-balance player
        risks only their wallet yet could win the full pool. If the balance drops
        below the table minimum, the betting controls are hidden until the player
        leaves instead of flooring the maximum back above their wallet.
        """
        pool_max = self.round_state.current_max_bet(jackpot=self._jackpot_snapshot)
        if user_id is None:
            return pool_max
        balance = self._final_balances.get(user_id)
        if balance is None:
            return pool_max
        return min(pool_max, max(balance, 0))

    def _active_max_bet(self) -> int:
        """Returns the active player's balance-bounded maximum bet."""
        active = self.round_state.active_turn
        user_id = active.participant.user_id if active is not None else None
        return self._max_bet_for(user_id=user_id)

    async def _place_select_bet(self, interaction: Interaction, amount: int) -> None:
        """Defers a select interaction and places the chosen fixed bet."""
        await interaction.response.defer()
        await self._place_bet_locked_by_interaction(interaction=interaction, amount=amount)

    async def _place_bet_locked_by_interaction(
        self, interaction: Interaction, amount: int
    ) -> None:
        """Resolves a bet, settles it against the jackpot, and refreshes the table."""
        if interaction.user is None:
            return
        message = interaction.message or self.message
        if message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            jackpot_before = self._jackpot_snapshot
            participant = self._participant_for(user_id=interaction.user.id)
            try:
                # Refresh from the live wallet: a player may have spent or transferred
                # outside the table since the ante, so the in-table cache can be stale.
                self._final_balances[interaction.user.id] = await get_balance(
                    user_id=interaction.user.id
                )
                if amount > self._max_bet_for(user_id=interaction.user.id):
                    raise DragonGateBetRangeError("Bet exceeds the player's balance")
                turn_result = self.round_state.place_bet(
                    user_id=interaction.user.id, amount=amount, jackpot=self._jackpot_snapshot
                )
            except DragonGateError as error:
                await self._send_bet_error_notice(interaction=interaction, error=error)
                return
            was_loss = turn_result.delta < 0
            settlement = await apply_jackpot_settlement(
                player_id=interaction.user.id,
                player_account_name=participant.account_name if participant else "",
                player_avatar_url=participant.avatar_url if participant else "",
                player_delta=turn_result.delta,
                game_id=GAME_ID,
                expected_jackpot_generation=self._jackpot_generation,
            )
            player_balance = settlement.player_balance
            applied_delta = settlement.applied_player_delta
            if applied_delta != turn_result.delta:
                turn_result = self.round_state.replace_last_result_delta(
                    user_id=interaction.user.id, delta=applied_delta
                )
            self._history.append(turn_result)
            self._jackpot_snapshot = settlement.jackpot_balance
            self._jackpot_generation = settlement.jackpot_generation
            self._final_balances[interaction.user.id] = player_balance
            pool_was_cleared = settlement.jackpot_depleted or (
                applied_delta > 0 and applied_delta >= jackpot_before
            )
            if pool_was_cleared:
                reason = (
                    "彩金池清空，系統已自動補池" if settlement.jackpot_depleted else "彩金池清空"
                )
                await self._finalize_locked(message=message, reason=reason)
                return
            if (
                was_loss
                and player_balance <= 0
                and self.round_state.is_active(user_id=interaction.user.id)
            ):
                self.round_state.withdraw(user_id=interaction.user.id)
                if self.round_state.finished:
                    await self._finalize_locked(message=message, reason="所有玩家已離桌或餘額歸零")
                    return
            self.sync_controls()
            await message.edit(
                **_dragon_gate_table_edit_kwargs(
                    embeds=self.in_progress_embeds(), view=self, target=message
                )
            )

    async def _handle_leave(self, interaction: Interaction) -> None:
        """Withdraws a seated player and refunds positive table delta to the jackpot."""
        if interaction.user is None:
            return
        await interaction.response.defer()
        message = interaction.message or self.message
        if message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            try:
                delta = self.round_state.withdraw(user_id=interaction.user.id)
            except DragonGateParticipantUnknownError:
                await self._send_notice(interaction=interaction, content="你不在這桌")
                return
            participant = self._participant_for(user_id=interaction.user.id)
            if delta > 0:
                settlement = await apply_jackpot_settlement(
                    player_id=interaction.user.id,
                    player_account_name=participant.account_name if participant else "",
                    player_avatar_url=participant.avatar_url if participant else "",
                    player_delta=-delta,
                    game_id=GAME_ID,
                )
                self._jackpot_snapshot = settlement.jackpot_balance
                self._jackpot_generation = settlement.jackpot_generation
                self._final_balances[interaction.user.id] = settlement.player_balance
                refunded_to_pool = max(-settlement.applied_player_delta, 0)
                if refunded_to_pool > 0:
                    self._refunded_to_pool[interaction.user.id] = refunded_to_pool
            if self.round_state.finished:
                await self._finalize_locked(message=message, reason="所有玩家已離桌")
                return
            self.sync_controls()
            await message.edit(
                **_dragon_gate_table_edit_kwargs(
                    embeds=self.in_progress_embeds(), view=self, target=message
                )
            )

    def in_progress_embeds(self) -> list[Embed]:
        """Builds the current table and optional history embeds."""
        embeds: list[Embed] = [
            build_dragon_gate_in_progress_embed(
                round_state=self.round_state, jackpot=self._jackpot_snapshot
            )
        ]
        history_embed = build_dragon_gate_history_embed(
            history=self._history, round_state=self.round_state
        )
        if history_embed is not None:
            embeds.append(history_embed)
        return embeds

    async def _refund_remaining_winners_locked(self) -> None:
        """Returns positive in-flight deltas to the jackpot before table cleanup."""
        for participant in self.round_state.active_participants():
            delta = self.round_state.player_delta(user_id=participant.user_id)
            if delta <= 0:
                continue
            settlement = await apply_jackpot_settlement(
                player_id=participant.user_id,
                player_account_name=participant.account_name,
                player_avatar_url=participant.avatar_url,
                player_delta=-delta,
                game_id=GAME_ID,
            )
            self._jackpot_snapshot = settlement.jackpot_balance
            self._jackpot_generation = settlement.jackpot_generation
            self._final_balances[participant.user_id] = settlement.player_balance
            refunded_to_pool = max(-settlement.applied_player_delta, 0)
            if refunded_to_pool > 0:
                self._refunded_to_pool[participant.user_id] = refunded_to_pool

    async def _finalize_locked(self, message: Message, reason: str) -> None:
        """Builds final results, disables controls, and schedules cleanup."""
        if self._settled:
            return
        self._settled = True
        results: list[DragonGatePlayerResult] = []
        for participant in self.round_state.participants:
            user_id = participant.user_id
            final_balance = self._final_balances.get(user_id)
            if final_balance is None:
                final_balance = await get_balance(user_id=user_id)
            gross_delta = self.round_state.player_delta(user_id=user_id)
            refunded = self._refunded_to_pool.get(user_id, 0)
            results.append(
                DragonGatePlayerResult(
                    participant=participant,
                    delta=gross_delta - refunded,
                    final_balance=final_balance,
                    withdrawn=user_id in self.round_state.withdrawn_user_ids,
                    refunded_to_pool=refunded,
                )
            )

        final_embed = build_dragon_gate_final_embed(
            round_state=self.round_state,
            results=results,
            jackpot=self._jackpot_snapshot,
            reason=reason,
        )
        embeds: list[Embed] = [final_embed]
        history_embed = build_dragon_gate_history_embed(
            history=self._history, round_state=self.round_state
        )
        if history_embed is not None:
            embeds.append(history_embed)
        self.clear_items()
        self.stop()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                message.edit(
                    **_dragon_gate_table_edit_kwargs(embeds=embeds, view=None, target=message)
                ),
                timeout=DRAGON_GATE_FINAL_EDIT_TIMEOUT_SECONDS,
            )
        schedule_public_message_delete(message=message, user_name=self.owner.account_name)

    def _participant_for(self, user_id: int) -> GameParticipant | None:
        """Returns the participant matching a Discord user ID."""
        for participant in self.round_state.participants:
            if participant.user_id == user_id:
                return participant
        return None

    def _button(self, custom_id: str) -> Button:
        """Returns a button component by custom ID."""
        return self._buttons[custom_id]

    def _select(self, custom_id: str) -> StringSelect:
        """Returns a string select component by custom ID."""
        return self._selects[custom_id]

    def _current_turn_notice(self) -> str:
        """Returns the ephemeral notice for users acting out of turn."""
        active_turn = self.round_state.active_turn
        if active_turn is None:
            return "這桌已經不能操作了"
        return f"現在輪到 {active_turn.participant.display_name}"

    async def _send_bet_error_notice(
        self, interaction: Interaction, error: DragonGateError
    ) -> None:
        """Maps Dragon Gate rule errors to user-facing ephemeral notices."""
        if isinstance(error, DragonGatePairChoiceRequiredError):
            content = "同點門柱要先猜大或猜小"
        elif isinstance(error, DragonGateTableFinishedError):
            content = "這桌已經不能操作了"
        elif isinstance(error, DragonGateTurnError):
            content = self._current_turn_notice()
        elif isinstance(error, DragonGateBetRangeError):
            minimum = self.round_state.current_min_bet(jackpot=self._jackpot_snapshot)
            maximum = self._active_max_bet()
            if maximum < minimum:
                content = "餘額不足以下注，請先離桌"
            else:
                content = (
                    "下注金額需介於 "
                    f"{currency_text(amount=minimum, compact=True)} 到 "
                    f"{currency_text(amount=maximum, compact=True)}"
                )
        else:
            content = "這桌已經不能操作了"
        await self._send_notice(interaction=interaction, content=content)

    async def _send_notice(self, interaction: Interaction, content: str) -> None:
        """Sends a private action notice to the interacting user."""
        await send_ephemeral_notice(
            interaction=interaction,
            content=content,
            log_message="Failed to send Dragon Gate action notice",
        )

    async def on_error(self, error: Exception, item: Item, interaction: Interaction) -> None:
        """Logs active-table component failures instead of only printing to stderr."""
        logfire.error(
            "Dragon Gate action interaction failed",
            item_label=getattr(item, "label", None),
            user_id=getattr(interaction.user, "id", None),
            _exc_info=(type(error), error, error.__traceback__),
        )


class DragonGateBetModal(Modal):
    """Modal for entering an exact 射龍門 bet amount."""

    def __init__(self, view: DragonGateView, minimum: int, maximum: int) -> None:
        """Initializes the modal with a range-aware amount input."""
        super().__init__(title="自訂下注")
        self.view = view
        self.amount: TextInput = TextInput(
            label="下注金額",
            placeholder=f"{minimum:,} 到 {maximum:,}",
            min_length=1,
            max_length=max(len(f"{maximum:,}"), 1),
            required=True,
        )
        self.add_item(item=self.amount)

    async def callback(self, interaction: Interaction) -> None:
        """Submits the custom bet amount back to the active table view."""
        await self.view.submit_custom_bet(interaction=interaction, raw_amount=self.amount.value)


async def fetch_dragon_gate_jackpot_snapshot() -> JackpotSnapshot:
    """Reads the live 射龍門 jackpot pool balance and generation."""
    return await get_jackpot_snapshot(game_id=GAME_ID)


__all__ = [
    "DRAGON_GATE_ACTION_TIMEOUT_SECONDS",
    "DragonGateBetModal",
    "DragonGateLobbyView",
    "DragonGateView",
    "build_dragon_gate_final_embed",
    "build_dragon_gate_in_progress_embed",
    "build_dragon_gate_lobby_embed",
    "fetch_dragon_gate_jackpot_snapshot",
]
