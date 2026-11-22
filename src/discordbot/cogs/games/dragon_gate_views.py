"""Interactive components for multiplayer 射龍門 sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast
import asyncio

import logfire
import nextcord
from nextcord import Embed, Message, ButtonStyle, Interaction, SelectOption
from nextcord.ui import View, Button, TextInput, StringSelect

from discordbot.utils.logged_ui import LoggedModal
from discordbot.cogs.games.lobby import (
    PrepareParticipant,
    RefreshParticipants,
    BaseJackpotLobbyView,
)
from discordbot.cogs.games.seats import release_seats, hand_over_seats, update_seat_stake
from discordbot.utils.number_text import compact_amount
from discordbot.utils.amount_parsing import parse_decimal_amount
from discordbot.utils.discord_embeds import (
    DISCORD_EMBED_TOTAL_LIMIT,
    DISCORD_EMBED_DESCRIPTION_LIMIT,
    utf16_length,
    embed_text_length,
    clip_to_utf16_limit,
)
from discordbot.cogs.games.dragon_gate import (
    ANTE,
    GAME_ID,
    DragonGateTurn,
    DragonGateError,
    DragonGateRound,
    DragonGateOutcome,
    DragonGateDirection,
    DragonGateTurnError,
    DragonGateTurnResult,
    DragonGatePlayerResult,
    DragonGateBetRangeError,
    DragonGatePairChoiceRequiredError,
    DragonGatePairChoiceUnavailableError,
)
from discordbot.services.economy.holds import spending
from discordbot.cogs.games.interactions import (
    GameView,
    table_edit_kwargs,
    publish_final_table,
    set_view_item_visible,
)
from discordbot.cogs.games.presentation import (
    PUSH_COLOR,
    POT_FIELD_EMOJI,
    TURN_FIELD_EMOJI,
    WIN_RESULT_EMOJI,
    LAST_HAND_FIELD_EMOJI,
    FINISH_REASON_FIELD_EMOJI,
    LOBBY_PLAYERS_FIELD_EMOJI,
    delta_color,
    metadata_line,
    lobby_participant_line,
)
from discordbot.services.economy.database import get_balance, apply_jackpot_settlement
from discordbot.services.economy.presentation import amount_code, currency_text

if TYPE_CHECKING:
    from random import Random

    from nextcord.ext import commands

    from discordbot.typings.games import GameParticipant
    from discordbot.typings.economy import JackpotSettlementResult

DRAGON_GATE_ACTION_TIMEOUT_SECONDS: Final[int] = 180
DRAGON_GATE_VISIBLE_PLAYER_LINES: Final[int] = 20
# Turns kept in the history block. `_finalize_locked` sends the history embed beside the final
# embed and Discord caps a message at 6000 characters across embeds.
DRAGON_GATE_VISIBLE_HISTORY_LINES: Final[int] = 30


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


def _outcome_label(outcome: DragonGateOutcome) -> str:
    """Returns the display label for a turn outcome."""
    labels: dict[DragonGateOutcome, str] = {
        "gate_win": "✅ 射中",
        "outside_lose": "❌ 射偏",
        "pillar_hit": "🧱 撞柱",
        "pair_win": "✅ 猜中",
        "pair_lose": "❌ 猜錯",
        "pair_pillar_hit": "🧱 同點撞柱",
    }
    return labels[outcome]


def _result_line(result: DragonGateTurnResult) -> str:
    """Formats the last resolved turn for the final embed's last-hand block."""
    outcome_label = _outcome_label(outcome=result.outcome)
    direction = f" · {_direction_label(direction=result.direction)}" if result.direction else ""
    pillars = " ".join(str(card) for card in result.pillars)
    return (
        f"# {pillars}  →  {result.third_card}\n"
        f"**{result.participant.display_name}** (第 {result.turn_number} 手){direction}\n"
        f"### {outcome_label} {amount_code(amount=result.delta, signed=True, compact=True)}"
    )


def _history_code_lines(
    history: list[DragonGateTurnResult], visible_count: int = DRAGON_GATE_VISIBLE_HISTORY_LINES
) -> list[str]:
    """Builds monospace history lines for the most recent `visible_count` turns, counting older ones."""
    lines: list[str] = []
    hidden_count = len(history) - visible_count
    if hidden_count > 0:
        lines.append(f"(前 {hidden_count} 手省略)")
    if visible_count <= 0:
        return lines
    for result in history[-visible_count:]:
        outcome_label = _outcome_label(outcome=result.outcome)
        pillars = " ".join(str(card) for card in result.pillars)
        lines.append(
            f"第 {result.turn_number} 手 {result.participant.account_name}: "
            f"{pillars} → {result.third_card}  {outcome_label} "
            f"{compact_amount(amount=result.delta, signed=True)}"
        )
    return lines


def _scoreboard_code_lines(
    round_state: DragonGateRound, visible_count: int = DRAGON_GATE_VISIBLE_PLAYER_LINES
) -> list[str]:
    """Builds monospace scoreboard lines from current table deltas."""
    lines: list[str] = []
    shown = round_state.participants[:visible_count]
    for participant in shown:
        delta = round_state.player_delta(user_id=participant.user_id)
        suffix = " (已離桌)" if participant.user_id in round_state.withdrawn_user_ids else ""
        lines.append(
            f"{participant.account_name}{suffix}: {compact_amount(amount=delta, signed=True)}"
        )
    hidden_count = len(round_state.participants) - len(shown)
    if hidden_count > 0:
        lines.append(f"(還有 {hidden_count} 位玩家省略)")
    return lines


def _last_result_line(result: DragonGateTurnResult) -> str:
    """Returns a one-line summary of the previous turn for placement above the current state."""
    outcome_label = _outcome_label(outcome=result.outcome)
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
    status: str | None = None,
) -> Embed:
    """Returns the lobby embed shown before a 射龍門 table starts.

    Args:
        owner (GameParticipant): Table owner who opened the lobby.
        participants (list[GameParticipant]): Seated participants in join order.
        jackpot (int): Live jackpot pool balance.
        status (str | None): Optional status line placed in the embed description.
    """
    embed = Embed(title="♦️ 射龍門 · 開桌準備", color=PUSH_COLOR)
    if status:
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


def build_dragon_gate_in_progress_embed(
    round_state: DragonGateRound, jackpot: int, insufficient_balance: bool = False
) -> Embed:
    """Returns the active 射龍門 table embed.

    Args:
        round_state (DragonGateRound): Current 射龍門 round state.
        jackpot (int): Live jackpot pool balance.
        insufficient_balance (bool): Whether the active player's wallet is below the minimum bet.
    """
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
        if insufficient_balance:
            description_parts.append("> ⚠️ 餘額不足最低下注，請按「離桌」")

    embed = Embed(
        title=f"♦️ 射龍門 · 第 {round_state.turn_number} 手",
        description="\n".join(description_parts),
        color=PUSH_COLOR,
    )
    if round_state.participants and round_state.participants[0].avatar_url:
        embed.set_thumbnail(url=round_state.participants[0].avatar_url)
    embed.set_footer(text=f"{DRAGON_GATE_ACTION_TIMEOUT_SECONDS} 秒無互動會結束牌桌")
    return embed


def _render_history_description(history_lines: list[str], scoreboard_lines: list[str]) -> str:
    """Wraps history and scoreboard code lines inside the history embed description."""
    lines: list[str] = list(history_lines)
    if history_lines and scoreboard_lines:
        lines.append("")
    lines.extend(scoreboard_lines)
    return "**紀錄:**\n```\n" + "\n".join(lines) + "\n```"


def build_dragon_gate_history_embed(
    history: list[DragonGateTurnResult],
    round_state: DragonGateRound,
    max_utf16: int = DISCORD_EMBED_DESCRIPTION_LIMIT,
) -> Embed | None:
    """Returns an auxiliary embed with recent turns and cumulative scoreboard, or None when empty.

    Trims older turns, then trailing scoreboard rows, to fit within `max_utf16` UTF-16 code units.

    Args:
        history (list[DragonGateTurnResult]): Resolved turns in chronological order.
        round_state (DragonGateRound): Current 射龍門 round state.
        max_utf16 (int): Maximum UTF-16 code-unit budget for the embed description.
    """
    has_deltas = any(
        round_state.player_delta(user_id=participant.user_id) != 0
        for participant in round_state.participants
    )
    if not history and not has_deltas:
        return None
    budget = min(max_utf16, DISCORD_EMBED_DESCRIPTION_LIMIT)
    prefix = "**紀錄:**\n```\n"
    suffix = "\n```"
    wrapper_len = utf16_length(value=prefix) + utf16_length(value=suffix)
    if budget <= wrapper_len:
        return None

    visible_history = min(len(history), DRAGON_GATE_VISIBLE_HISTORY_LINES)
    visible_players = min(len(round_state.participants), DRAGON_GATE_VISIBLE_PLAYER_LINES)
    history_lines = (
        _history_code_lines(history=history, visible_count=visible_history) if history else []
    )
    scoreboard_lines = (
        _scoreboard_code_lines(round_state=round_state, visible_count=visible_players)
        if round_state.participants
        else []
    )
    description = _render_history_description(
        history_lines=history_lines, scoreboard_lines=scoreboard_lines
    )
    while visible_history > 0 and utf16_length(value=description) > budget:
        visible_history -= 1
        history_lines = _history_code_lines(history=history, visible_count=visible_history)
        description = _render_history_description(
            history_lines=history_lines, scoreboard_lines=scoreboard_lines
        )
    while visible_players > 0 and utf16_length(value=description) > budget:
        visible_players -= 1
        scoreboard_lines = _scoreboard_code_lines(
            round_state=round_state, visible_count=visible_players
        )
        description = _render_history_description(
            history_lines=history_lines, scoreboard_lines=scoreboard_lines
        )
    if utf16_length(value=description) > budget:
        inner = description.removeprefix(prefix).removesuffix(suffix)
        clipped = clip_to_utf16_limit(text=inner, limit=budget - wrapper_len, notice="…")
        description = f"{prefix}{clipped}{suffix}"
    return Embed(description=description, color=PUSH_COLOR)


def _final_result_blocks(results: list[DragonGatePlayerResult], visible_count: int) -> list[str]:
    """Formats the first `visible_count` settled player blocks plus any hidden-player notice."""
    parts: list[str] = []
    for result in results[:visible_count]:
        balance_text = f"餘額 {amount_code(amount=result.final_balance, compact=True)}"
        if result.withdrawn:
            balance_text += " · 已離桌"
        if result.refunded_to_pool > 0:
            balance_text += (
                f" · 逆贏退回 {amount_code(amount=result.refunded_to_pool, compact=True)}"
            )
        parts.append(f"**{result.participant.display_name}**")
        parts.append(_settlement_result_heading(delta=result.delta))
        parts.append(metadata_line(text=balance_text))
    hidden_count = len(results) - visible_count
    if hidden_count > 0:
        parts.append(f"-# 還有 {hidden_count} 位玩家已結算")
    return parts


def build_dragon_gate_final_embed(
    round_state: DragonGateRound, results: list[DragonGatePlayerResult], jackpot: int, reason: str
) -> Embed:
    """Returns the final embed for a settled 射龍門 table.

    Args:
        round_state (DragonGateRound): Settled 射龍門 round state.
        results (list[DragonGatePlayerResult]): Per-player final outcomes.
        jackpot (int): Final jackpot pool balance.
        reason (str): Human-readable reason the table closed.
    """
    header_parts: list[str] = [f"### {FINISH_REASON_FIELD_EMOJI} 結束原因", reason, ""]
    header_parts.append(f"## {POT_FIELD_EMOJI} 彩金池 {compact_amount(amount=jackpot)}")
    header_parts.append("")
    if round_state.last_result is not None:
        header_parts.append(f"### {LAST_HAND_FIELD_EMOJI} 最後一手")
        header_parts.append(_result_line(result=round_state.last_result))
        header_parts.append("")
    header_parts.append("### 結算")

    visible_players = min(len(results), DRAGON_GATE_VISIBLE_PLAYER_LINES)
    description = "\n".join([
        *header_parts,
        *_final_result_blocks(results=results, visible_count=visible_players),
    ])
    while (
        visible_players > 0 and utf16_length(value=description) > DISCORD_EMBED_DESCRIPTION_LIMIT
    ):
        visible_players -= 1
        description = "\n".join([
            *header_parts,
            *_final_result_blocks(results=results, visible_count=visible_players),
        ])
    description = clip_to_utf16_limit(
        text=description, limit=DISCORD_EMBED_DESCRIPTION_LIMIT, notice="\n…（已截斷）"
    )

    embed = Embed(
        title=_final_title(results=results),
        description=description,
        color=delta_color(delta=sum(result.delta for result in results)),
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
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
        initial_jackpot: int,
        initial_jackpot_generation: int | None = None,
    ) -> None:
        """Initializes a 射龍門 lobby with the current jackpot snapshot.

        Args:
            owner (GameParticipant): Table owner who opened the lobby.
            rng (Random): Random source passed to the started round.
            prepare_participant (PrepareParticipant): Validator for users pressing Join.
            refresh_participants (RefreshParticipants): Balance re-checker run at Start.
            initial_jackpot (int): Jackpot pool balance when the lobby opened.
            initial_jackpot_generation (int | None): Optimistic-concurrency generation
                of the jackpot row when the lobby opened.
        """
        super().__init__(
            owner=owner,
            rng=rng,
            prepare_participant=prepare_participant,
            refresh_participants=refresh_participants,
            initial_jackpot=initial_jackpot,
            timeout=DRAGON_GATE_ACTION_TIMEOUT_SECONDS,
            initial_jackpot_generation=initial_jackpot_generation,
        )

    def lobby_embed(self, status: str | None = None) -> Embed:
        """Returns the 射龍門 lobby embed from participants and jackpot state.

        Args:
            status (str | None): Optional status line placed in the embed description.
        """
        return build_dragon_gate_lobby_embed(
            owner=self.owner,
            participants=self.participants,
            jackpot=self._jackpot_snapshot,
            status=status,
        )

    async def _start_game_after_antes(
        self,
        interaction: Interaction[commands.Bot],
        message: Message,
        final_balances: dict[int, int],
    ) -> None:
        """Starts the active table after all lobby antes have been charged."""
        round_state = DragonGateRound.from_participants(
            rng=self.rng, participants=self.participants
        )
        view = DragonGateView(
            round_state=round_state,
            owner=self.owner,
            jackpot_snapshot=self._jackpot_snapshot,
            jackpot_generation=self._jackpot_generation,
            final_balances=final_balances,
        )
        view.message = message
        view.last_press = interaction
        embeds = view.in_progress_embeds()
        await self._show_table(
            interaction=interaction,
            payload=table_edit_kwargs(embeds=embeds, view=view, target=message),
        )
        # Each seat starts at a 0 hold and updates after each turn to max(net_delta, 0) so
        # mid-round winnings stay protected for 逆贏不拿 while keeping the player off other tables.
        hand_over_seats(
            stakes=dict.fromkeys(self._human_ids(), 0), from_holder=self, to_holder=view
        )


class DragonGateView(GameView):
    """High / low buttons, bet select, and leave button for an active 射龍門 table."""

    interaction_failure_log = "Dragon Gate action interaction failed"
    notice_failure_log = "Failed to send Dragon Gate action notice"

    def __init__(
        self,
        round_state: DragonGateRound,
        owner: GameParticipant,
        jackpot_snapshot: int,
        final_balances: dict[int, int],
        jackpot_generation: int | None = None,
    ) -> None:
        """Initializes the active 射龍門 table view.

        Args:
            round_state (DragonGateRound): Active 射龍門 round state.
            owner (GameParticipant): Table owner who opened the lobby.
            jackpot_snapshot (int): Jackpot pool balance after pre-game antes.
            final_balances (dict[int, int]): Per-player balances after pre-game antes.
            jackpot_generation (int | None): Optimistic-concurrency generation of the
                jackpot row after pre-game antes.
        """
        super().__init__(timeout=DRAGON_GATE_ACTION_TIMEOUT_SECONDS)
        self.round_state = round_state
        self.owner = owner
        self.message: Message | None = None
        # The last press acknowledged on the table; the timeout's edit and delete ride its token.
        self.last_press: Interaction[commands.Bot] | None = None
        self._round_lock = asyncio.Lock()
        self._settled = False
        self._history: list[DragonGateTurnResult] = []
        self._jackpot_snapshot = jackpot_snapshot
        self._jackpot_generation = jackpot_generation
        self._final_balances: dict[int, int] = dict(final_balances)
        self._refunded_to_pool: dict[int, int] = {}
        # Set once the table has published its result: until then its players stay seated.
        self._seats_released = False
        self.sync_controls()

    def holds_seats(self) -> bool:
        """Returns whether the table still holds its players' seats."""
        return not self._seats_released

    async def interaction_check(self, interaction: Interaction[commands.Bot]) -> bool:
        """Returns whether `interaction` came from a user allowed to use the pressed control.

        `dg:leave` is open to any seated player who has not withdrawn; every other
        control requires `interaction.user` to be the active turn holder.

        Args:
            interaction (Interaction[commands.Bot]): Component press interaction.
        """
        if self._settled:
            await self._send_notice(interaction=interaction, content="這桌已經結束, 等下一桌吧")
            return False
        if interaction.user is None:
            return False
        data = (
            cast("dict[str, Any]", interaction.data) if isinstance(interaction.data, dict) else {}
        )
        user_id = interaction.user.id
        if data.get("custom_id") == "dg:leave":
            if self.round_state.is_active(user_id=user_id):
                return True
            notice = "你不在這桌"
        else:
            active_turn = self.round_state.active_turn
            if active_turn is not None and user_id == active_turn.participant.user_id:
                return True
            notice = self._current_turn_notice()
        await self._send_notice(interaction=interaction, content=notice)
        return False

    async def on_timeout(self) -> None:
        """Finalizes an abandoned table and refunds in-flight winnings into the pool."""
        if self.message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            await self._refund_remaining_winners_locked(message=self.message)
            await self._finalize_locked(
                message=self.message, reason="逾時未操作", interaction=self.last_press
            )

    @nextcord.ui.button(
        label="同點猜大", emoji="⬆️", style=ButtonStyle.secondary, custom_id="dg:higher", row=1
    )
    async def choose_higher(
        self, _button: Button[DragonGateView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Chooses higher for a same-point gate.

        Args:
            _button (Button[DragonGateView]): Clicked Higher button.
            interaction (Interaction[commands.Bot]): Button press interaction.
        """
        await self._choose_direction(interaction=interaction, direction="higher")

    @nextcord.ui.button(
        label="同點猜小", emoji="⬇️", style=ButtonStyle.secondary, custom_id="dg:lower", row=1
    )
    async def choose_lower(
        self, _button: Button[DragonGateView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Chooses lower for a same-point gate.

        Args:
            _button (Button[DragonGateView]): Clicked Lower button.
            interaction (Interaction[commands.Bot]): Button press interaction.
        """
        await self._choose_direction(interaction=interaction, direction="lower")

    @nextcord.ui.string_select(
        placeholder="🪙 選擇下注金額",
        custom_id="dg:bet",
        min_values=1,
        max_values=1,
        options=[
            SelectOption(label="底注", value="min", emoji="🪙"),
            SelectOption(label="全池", value="max", emoji="💰"),
            SelectOption(label="自訂", value="custom", emoji="✏️"),
        ],
        row=2,
    )
    async def bet_select(
        self, select: StringSelect[DragonGateView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Routes the bet select choice to a fixed amount or a custom modal.

        Args:
            select (StringSelect[DragonGateView]): Bet amount select menu.
            interaction (Interaction[commands.Bot]): Select menu interaction.
        """
        await self._handle_bet_choice(choice=select.values[0], interaction=interaction)

    @nextcord.ui.button(
        label="離桌", emoji="🚪", style=ButtonStyle.danger, custom_id="dg:leave", row=0
    )
    async def leave_table(
        self, _button: Button[DragonGateView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Lets any seated player withdraw mid-table without ending the round.

        Args:
            _button (Button[DragonGateView]): Clicked Leave button.
            interaction (Interaction[commands.Bot]): Button press interaction.
        """
        await self._handle_leave(interaction=interaction)

    async def _handle_bet_choice(
        self, choice: str, interaction: Interaction[commands.Bot]
    ) -> None:
        """Routes a select-menu choice to a fixed bet or custom modal."""
        if choice == "custom":
            if self.round_state.needs_pair_choice():
                await self._send_notice(interaction=interaction, content="同點門柱要先猜大或猜小")
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

    async def submit_custom_bet(
        self, interaction: Interaction[commands.Bot], raw_amount: str | None
    ) -> None:
        """Handles the custom bet modal submission.

        Args:
            interaction (Interaction[commands.Bot]): Modal submission interaction.
            raw_amount (str | None): Raw bet string entered by the player.
        """
        amount = parse_decimal_amount(raw=raw_amount)
        if amount is None:
            await self._send_notice(interaction=interaction, content="下注金額要是整數")
            return
        await interaction.response.defer()
        await self._place_bet_locked_by_interaction(interaction=interaction, amount=amount)

    def sync_controls(self) -> None:
        """Rebuilds control labels, options, and visibility from the table state."""
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

        higher_button = cast('Button["DragonGateView"]', self.choose_higher)
        lower_button = cast('Button["DragonGateView"]', self.choose_lower)
        higher_button.disabled = False
        lower_button.disabled = False
        set_view_item_visible(view=self, item=higher_button, visible=needs_pair_choice)
        set_view_item_visible(view=self, item=lower_button, visible=needs_pair_choice)

        bet_select = cast('StringSelect["DragonGateView"]', self.bet_select)
        bet_select.disabled = False
        bet_select.placeholder = "🪙 選擇下注金額"
        bet_select.options = [
            SelectOption(
                label=f"底注 {compact_amount(amount=minimum)}",
                value="min",
                emoji="🪙",
                description="最低下注金額",
            ),
            SelectOption(
                label=f"全池 {compact_amount(amount=maximum)}",
                value="max",
                emoji="💰",
                description="一把定生死, 清空彩金池",
            ),
            SelectOption(
                label="自訂", value="custom", emoji="✏️", description="彈出視窗輸入精確金額"
            ),
        ]
        set_view_item_visible(view=self, item=bet_select, visible=can_bet)

        leave_button = cast('Button["DragonGateView"]', self.leave_table)
        leave_button.disabled = False
        has_active_participant = bool(self.round_state.active_participants())
        set_view_item_visible(
            view=self, item=leave_button, visible=not self._settled and has_active_participant
        )

    async def _choose_direction(
        self, interaction: Interaction[commands.Bot], direction: DragonGateDirection
    ) -> None:
        """Stores a high or low choice for the active pair gate."""
        await interaction.response.defer()
        if interaction.user is None or interaction.message is None:
            return
        async with self._round_lock:
            if self._settled:
                await self._send_notice(
                    interaction=interaction, content="這桌已經結束, 等下一桌吧"
                )
                return
            try:
                self.round_state.choose_pair_direction(
                    user_id=interaction.user.id, direction=direction
                )
            except DragonGateError as error:
                await self._send_notice(
                    interaction=interaction, content=self._rule_error_notice(error=error)
                )
                return
            self.sync_controls()
            self.last_press = interaction
            await interaction.edit_original_message(
                **table_edit_kwargs(
                    embeds=self.in_progress_embeds(), view=self, target=interaction.message
                )
            )

    def _max_bet_for(self, user_id: int | None) -> int:
        """Returns the max legal bet bounded by the pool, the single-bet cap, and player balance."""
        pool_max = self.round_state.current_max_bet(jackpot=self._jackpot_snapshot)
        if user_id is None:
            return pool_max
        return min(pool_max, max(self._final_balances[user_id], 0))

    def _active_max_bet(self) -> int:
        """Returns the active player's balance-bounded maximum bet."""
        active = self.round_state.active_turn
        user_id = active.participant.user_id if active is not None else None
        return self._max_bet_for(user_id=user_id)

    async def _place_select_bet(self, interaction: Interaction[commands.Bot], amount: int) -> None:
        """Defers a select interaction and places the chosen fixed bet."""
        await interaction.response.defer()
        await self._place_bet_locked_by_interaction(interaction=interaction, amount=amount)

    @staticmethod
    def _is_stale_generation_win(
        turn_result: DragonGateTurnResult,
        settlement: JackpotSettlementResult,
        prev_generation: int | None,
    ) -> bool:
        """Returns whether a winning bet paid +0 because another table cleared the pool."""
        return (
            turn_result.delta > 0
            and settlement.applied_player_delta == 0
            and prev_generation is not None
            and settlement.jackpot_generation != prev_generation
        )

    async def _refresh_table_locked(
        self, interaction: Interaction[commands.Bot], message: Message
    ) -> None:
        """Rebuilds controls and renders the current in-progress embeds onto `message`."""
        self.sync_controls()
        self.last_press = interaction
        await interaction.edit_original_message(
            **table_edit_kwargs(embeds=self.in_progress_embeds(), view=self, target=message)
        )

    async def _resolve_bet_locked(
        self, user_id: int, amount: int
    ) -> tuple[DragonGateTurnResult, JackpotSettlementResult, int | None]:
        """Refreshes wallet, resolves and settles one bet under `spending`, and updates the hold."""
        prev_generation = self._jackpot_generation
        async with spending(user_id=user_id):
            # Refresh from the live wallet: a player may have spent or transferred outside
            # the table since the ante, so the in-table cache can be stale.
            self._final_balances[user_id] = await get_balance(user_id=user_id)
            if amount > self._max_bet_for(user_id=user_id):
                raise DragonGateBetRangeError("Bet exceeds the player's balance")
            turn_result = self.round_state.resolve_bet(
                user_id=user_id, amount=amount, jackpot=self._jackpot_snapshot
            )
            settlement = await apply_jackpot_settlement(
                player_id=user_id,
                player_account_name=turn_result.participant.account_name,
                player_avatar_url=turn_result.participant.avatar_url,
                player_delta=turn_result.delta,
                game_id=GAME_ID,
                expected_jackpot_generation=prev_generation,
            )
            if not self._is_stale_generation_win(
                turn_result=turn_result, settlement=settlement, prev_generation=prev_generation
            ):
                applied_delta = settlement.applied_player_delta
                booked_result = (
                    turn_result
                    if applied_delta == turn_result.delta
                    else turn_result.model_copy(update={"delta": applied_delta})
                )
                self.round_state.record_result(result=booked_result)
                self._history.append(booked_result)
                # Update before releasing `spending` so concurrent debits cannot spend winnings
                # ahead of a 逆贏不拿 refund.
                update_seat_stake(
                    user_id=user_id,
                    holder=self,
                    amount=max(self.round_state.player_delta(user_id=user_id), 0),
                )
            return turn_result, settlement, prev_generation

    async def _withdraw_if_below_min_bet_locked(
        self,
        user_id: int,
        was_loss: bool,
        player_balance: int,
        interaction: Interaction[commands.Bot],
        message: Message,
    ) -> bool:
        """Auto-withdraws a losing player who cannot cover the minimum bet; True if table ended."""
        min_bet = self.round_state.current_min_bet(jackpot=self._jackpot_snapshot)
        if not (
            was_loss and player_balance < min_bet and self.round_state.is_active(user_id=user_id)
        ):
            return False
        delta = self.round_state.player_delta(user_id=user_id)
        if delta > 0:
            try:
                await self._refund_winnings_to_pool_locked(user_id=user_id, delta=delta)
            # Broad on purpose: a failed refund must not block auto-withdrawal or strand the table.
            except Exception as exc:
                logfire.error(
                    "Dragon Gate auto-withdraw refund failed; the player keeps the winnings",
                    user_id=user_id,
                    delta=delta,
                    channel_id=message.channel.id,
                    message_id=message.id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
        self.round_state.withdraw(user_id=user_id)
        # The table never touches a withdrawn wallet again, so the seat goes now.
        release_seats(user_ids=[user_id], holder=self)
        if player_balance > 0:
            await self._send_notice(
                interaction=interaction, content="餘額低於最低下注，已自動離桌"
            )
        if self.round_state.finished:
            await self._finalize_locked(
                message=message, reason="所有玩家已離桌或餘額不足", interaction=interaction
            )
            return True
        return False

    async def _record_settled_bet_locked(
        self,
        interaction: Interaction[commands.Bot],
        message: Message,
        turn_result: DragonGateTurnResult,
        settlement: JackpotSettlementResult,
        prev_generation: int | None,
    ) -> None:
        """Syncs snapshots from `settlement` and closes or refreshes the table."""
        user_id = turn_result.participant.user_id
        self._jackpot_snapshot = settlement.jackpot_balance
        self._jackpot_generation = settlement.jackpot_generation
        self._final_balances[user_id] = settlement.player_balance

        if self._is_stale_generation_win(
            turn_result=turn_result, settlement=settlement, prev_generation=prev_generation
        ):
            await self._send_notice(
                interaction=interaction,
                content="彩金池已被其他牌桌清空並補池，這手未結算，請依最新彩金池重新下注",
            )
            if settlement.jackpot_balance <= 0:
                await self._finalize_locked(
                    message=message, reason="彩金池清空", interaction=interaction
                )
                return
            await self._refresh_table_locked(interaction=interaction, message=message)
            return

        if settlement.jackpot_depleted or settlement.jackpot_balance <= 0:
            reason = "彩金池清空，系統已自動補池" if settlement.jackpot_depleted else "彩金池清空"
            await self._finalize_locked(message=message, reason=reason, interaction=interaction)
            return

        if await self._withdraw_if_below_min_bet_locked(
            user_id=user_id,
            was_loss=turn_result.delta < 0,
            player_balance=settlement.player_balance,
            interaction=interaction,
            message=message,
        ):
            return

        await self._refresh_table_locked(interaction=interaction, message=message)

    async def _place_bet_locked_by_interaction(
        self, interaction: Interaction[commands.Bot], amount: int
    ) -> None:
        """Resolves a bet, settles it against the jackpot, then books it and refreshes the table.

        A settlement that raises books nothing: the player is told, and the same gate takes the
        next bet.
        """
        if interaction.user is None:
            return
        message = interaction.message or self.message
        if message is None:
            return
        async with self._round_lock:
            if self._settled:
                await self._send_notice(
                    interaction=interaction, content="這桌已經結束, 等下一桌吧"
                )
                return
            try:
                turn_result, settlement, prev_generation = await self._resolve_bet_locked(
                    user_id=interaction.user.id, amount=amount
                )
            except DragonGateError as error:
                await self._send_notice(
                    interaction=interaction, content=self._rule_error_notice(error=error)
                )
                return
            except Exception:
                # Broad on purpose: whatever the write raised on, it rolled back and nothing was
                # booked, so the player only needs telling; the re-raise reaches on_error, which
                # logs it.
                await self._send_notice(interaction=interaction, content="下注失敗, 這注不算")
                raise
            await self._record_settled_bet_locked(
                interaction=interaction,
                message=message,
                turn_result=turn_result,
                settlement=settlement,
                prev_generation=prev_generation,
            )

    async def _handle_leave(self, interaction: Interaction[commands.Bot]) -> None:
        """Refunds a seated player's positive table delta to the jackpot, then withdraws them.

        A refund that raises leaves the player seated: they are told, and may press again.
        """
        if interaction.user is None:
            return
        await interaction.response.defer()
        message = interaction.message or self.message
        if message is None:
            return
        async with self._round_lock:
            if self._settled:
                await self._send_notice(
                    interaction=interaction, content="這桌已經結束, 等下一桌吧"
                )
                return
            if not self.round_state.is_active(user_id=interaction.user.id):
                await self._send_notice(interaction=interaction, content="你不在這桌")
                return
            delta = self.round_state.player_delta(user_id=interaction.user.id)
            if delta > 0:
                try:
                    await self._refund_winnings_to_pool_locked(
                        user_id=interaction.user.id, delta=delta
                    )
                except Exception:
                    # Broad on purpose: whatever the write raised on, it rolled back and the
                    # player is still seated, so they only need telling; the re-raise reaches
                    # on_error, which logs it.
                    await self._send_notice(
                        interaction=interaction, content="離桌失敗, 請再按一次離桌"
                    )
                    raise
            self.round_state.withdraw(user_id=interaction.user.id)
            # Their winnings went back to the pool above and no later bet reaches this wallet,
            # so the seat goes now rather than when the table ends.
            release_seats(user_ids=[interaction.user.id], holder=self)
            if self.round_state.finished:
                await self._finalize_locked(
                    message=message, reason="所有玩家已離桌", interaction=interaction
                )
                return
            self.sync_controls()
            self.last_press = interaction
            await interaction.edit_original_message(
                **table_edit_kwargs(embeds=self.in_progress_embeds(), view=self, target=message)
            )

    def in_progress_embeds(self) -> list[Embed]:
        """Returns the current table embed and optional history embed."""
        active_turn = self.round_state.active_turn
        minimum = self.round_state.current_min_bet(jackpot=self._jackpot_snapshot)
        insufficient_balance = (
            not self._settled and active_turn is not None and self._active_max_bet() < minimum
        )
        primary_embed = build_dragon_gate_in_progress_embed(
            round_state=self.round_state,
            jackpot=self._jackpot_snapshot,
            insufficient_balance=insufficient_balance,
        )
        embeds: list[Embed] = [primary_embed]
        remaining_utf16 = min(
            DISCORD_EMBED_DESCRIPTION_LIMIT,
            DISCORD_EMBED_TOTAL_LIMIT - embed_text_length(embed=primary_embed),
        )
        history_embed = build_dragon_gate_history_embed(
            history=self._history, round_state=self.round_state, max_utf16=remaining_utf16
        )
        if history_embed is not None:
            embeds.append(history_embed)
        return embeds

    async def _refund_winnings_to_pool_locked(self, user_id: int, delta: int) -> None:
        """Pushes one player's positive table delta back into the jackpot ("逆贏不拿")."""
        participant = next(
            participant
            for participant in self.round_state.participants
            if participant.user_id == user_id
        )
        async with spending(user_id=user_id):
            settlement = await apply_jackpot_settlement(
                player_id=user_id,
                player_account_name=participant.account_name,
                player_avatar_url=participant.avatar_url,
                player_delta=-delta,
                game_id=GAME_ID,
                refund=True,
            )
        self._jackpot_snapshot = settlement.jackpot_balance
        self._jackpot_generation = settlement.jackpot_generation
        self._final_balances[user_id] = settlement.player_balance
        refunded_to_pool = max(-settlement.applied_player_delta, 0)
        if refunded_to_pool > 0:
            self._refunded_to_pool[user_id] = refunded_to_pool

    async def _refund_remaining_winners_locked(self, message: Message) -> None:
        """Returns seated players' positive in-flight deltas to the jackpot on timeout."""
        for participant in self.round_state.active_participants():
            delta = self.round_state.player_delta(user_id=participant.user_id)
            if delta <= 0:
                continue
            try:
                await self._refund_winnings_to_pool_locked(
                    user_id=participant.user_id, delta=delta
                )
            except Exception as exc:
                # Broad on purpose: whatever the write raised on, it rolled back, and the table
                # must still settle, refund the other seats and be cleaned up.
                logfire.error(
                    "Dragon Gate timeout refund failed; the player keeps the winnings",
                    user_id=participant.user_id,
                    delta=delta,
                    channel_id=message.channel.id,
                    message_id=message.id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )

    async def _finalize_locked(
        self, message: Message, reason: str, interaction: Interaction[commands.Bot] | None
    ) -> None:
        """Builds final results, clears the controls, and schedules cleanup."""
        if self._settled:
            return
        self._settled = True
        try:
            await self._publish_result_locked(
                message=message, reason=reason, interaction=interaction
            )
        finally:
            self._seats_released = True
            release_seats(
                user_ids=[participant.user_id for participant in self.round_state.participants],
                holder=self,
            )

    async def _publish_result_locked(
        self, message: Message, reason: str, interaction: Interaction[commands.Bot] | None
    ) -> None:
        """Builds the final results and publishes them, scheduling the table's deletion."""
        results: list[DragonGatePlayerResult] = []
        for participant in self.round_state.participants:
            user_id = participant.user_id
            gross_delta = self.round_state.player_delta(user_id=user_id)
            refunded = self._refunded_to_pool.get(user_id, 0)
            results.append(
                DragonGatePlayerResult(
                    participant=participant,
                    delta=gross_delta - refunded,
                    final_balance=self._final_balances[user_id],
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
        remaining_utf16 = min(
            DISCORD_EMBED_DESCRIPTION_LIMIT,
            DISCORD_EMBED_TOTAL_LIMIT - embed_text_length(embed=final_embed),
        )
        history_embed = build_dragon_gate_history_embed(
            history=self._history, round_state=self.round_state, max_utf16=remaining_utf16
        )
        if history_embed is not None:
            embeds.append(history_embed)
        self.clear_items()
        self.stop()
        await publish_final_table(
            message=message,
            embeds=embeds,
            user_name=self.owner.account_name,
            game_name="Dragon Gate",
            interaction=interaction,
            channel_id=message.channel.id,
            message_id=message.id,
            reason=reason,
        )

    def _current_turn_notice(self) -> str:
        """Returns the ephemeral notice for users acting out of turn."""
        active_turn = self.round_state.active_turn
        if active_turn is None:
            return "這桌已經不能操作了"
        return f"現在輪到 {active_turn.participant.display_name}"

    def _rule_error_notice(self, error: DragonGateError) -> str:
        """Returns the ephemeral notice for a Dragon Gate rule error."""
        if isinstance(error, DragonGatePairChoiceRequiredError):
            return "同點門柱要先猜大或猜小"
        if isinstance(error, DragonGatePairChoiceUnavailableError):
            return "這手不需要猜大小"
        if isinstance(error, DragonGateTurnError):
            return self._current_turn_notice()
        if isinstance(error, DragonGateBetRangeError):
            minimum = self.round_state.current_min_bet(jackpot=self._jackpot_snapshot)
            maximum = self._active_max_bet()
            if maximum < minimum:
                return "餘額不足以下注，請先離桌"
            return (
                "下注金額需介於 "
                f"{currency_text(amount=minimum, compact=True)} 到 "
                f"{currency_text(amount=maximum, compact=True)}"
            )
        return "這桌已經不能操作了"


class DragonGateBetModal(LoggedModal):
    """Modal for entering an exact 射龍門 bet amount."""

    def __init__(self, view: DragonGateView, minimum: int, maximum: int) -> None:
        """Initializes the modal with a range-aware amount input.

        Args:
            view (DragonGateView): Active table view receiving the submission.
            minimum (int): Minimum legal bet displayed in the placeholder.
            maximum (int): Maximum legal bet displayed in the placeholder.
        """
        super().__init__(title="自訂下注")
        self.view = view
        self.amount: TextInput[View] = TextInput(
            label="下注金額",
            placeholder=f"{minimum:,} 到 {maximum:,}",
            min_length=1,
            max_length=max(len(f"{maximum:,}"), 1),
            required=True,
        )
        self.add_item(item=self.amount)

    async def callback(self, interaction: Interaction[commands.Bot]) -> None:
        """Submits the custom bet amount back to the active table view.

        Args:
            interaction (Interaction[commands.Bot]): Modal submission interaction.
        """
        await self.view.submit_custom_bet(interaction=interaction, raw_amount=self.amount.value)
