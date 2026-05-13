"""Interactive components for multiplayer 射龍門 sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
import asyncio
import contextlib

import logfire
from nextcord import Embed, Message, ButtonStyle, Interaction, ui

from discordbot.typings.games import GameParticipant, DragonGatePlayerResult
from discordbot.cogs._games.cleanup import schedule_game_message_delete
from discordbot.cogs._games.settlement import settle_dragon_gate_player
from discordbot.cogs._games.dragon_gate import (
    DragonGateError,
    DragonGateRound,
    DragonGateOutcome,
    DragonGateDirection,
    DragonGateTurnError,
    DragonGateTurnResult,
    DragonGateBetRangeError,
    DragonGateTableFinishedError,
    DragonGatePairChoiceRequiredError,
    DragonGatePairChoiceUnavailableError,
    render_cards,
)
from discordbot.cogs._games.presentation import (
    WIN_COLOR,
    LOSE_COLOR,
    PUSH_COLOR,
    ERROR_COLOR,
    dealer_quote,
    settlement_footer,
)
from discordbot.cogs._economy.presentation import currency_text

if TYPE_CHECKING:
    from random import Random

    from discordbot.cogs._games.dealer import DealerAI

DRAGON_GATE_ACTION_TIMEOUT_SECONDS = 180
DRAGON_GATE_VISIBLE_PLAYER_LINES = 20


class PrepareDragonGateParticipant(Protocol):
    """Callable used by 射龍門 lobby join buttons to validate a participant."""

    async def __call__(self, interaction: Interaction, ante: int) -> GameParticipant | None:
        """Returns a prepared participant or sends the interaction error."""


class RefreshDragonGateParticipants(Protocol):
    """Callable used by 射龍門 lobby start to re-check balances."""

    async def __call__(
        self, participants: list[GameParticipant], ante: int
    ) -> tuple[list[GameParticipant], list[str]]:
        """Returns refreshed participants and display names removed from the table."""


def _participant_lines(participants: list[GameParticipant], owner_id: int) -> str:
    lines: list[str] = []
    visible = participants[:DRAGON_GATE_VISIBLE_PLAYER_LINES]
    for index, participant in enumerate(visible, start=1):
        owner_label = " 房主" if participant.user_id == owner_id else ""
        lines.append(f"{index}. {participant.display_name}{owner_label}")
    hidden_count = len(participants) - len(visible)
    if hidden_count > 0:
        lines.append(f"還有 {hidden_count} 位玩家")
    return "\n".join(lines)


def _direction_label(direction: DragonGateDirection | None) -> str:
    if direction == "higher":
        return "猜大"
    if direction == "lower":
        return "猜小"
    return "尚未選擇"


def _outcome_presentation(outcome: DragonGateOutcome) -> tuple[str, int]:
    values: dict[DragonGateOutcome, tuple[str, int]] = {
        "gate_win": ("射進龍門", WIN_COLOR),
        "outside_lose": ("射偏", LOSE_COLOR),
        "pillar_hit": ("撞柱", LOSE_COLOR),
        "pair_win": ("猜中", WIN_COLOR),
        "pair_lose": ("猜錯", LOSE_COLOR),
        "pair_pillar_hit": ("同點撞柱", LOSE_COLOR),
    }
    return values[outcome]


def _result_line(result: DragonGateTurnResult) -> str:
    outcome_label, _color = _outcome_presentation(outcome=result.outcome)
    direction = f" | {_direction_label(direction=result.direction)}" if result.direction else ""
    return (
        f"第 {result.turn_number} 手 {result.participant.display_name}: "
        f"{render_cards(cards=result.pillars)} → {result.third_card}{direction} | "
        f"下注 {currency_text(amount=result.bet)} | **{outcome_label}** "
        f"{currency_text(amount=result.delta, signed=True)}"
    )


def _player_delta_lines(round_state: DragonGateRound) -> str:
    lines: list[str] = []
    for participant in round_state.participants[:DRAGON_GATE_VISIBLE_PLAYER_LINES]:
        delta = round_state.player_delta(user_id=participant.user_id)
        lines.append(f"{participant.display_name}: {currency_text(amount=delta, signed=True)}")
    hidden_count = len(round_state.participants) - DRAGON_GATE_VISIBLE_PLAYER_LINES
    if hidden_count > 0:
        lines.append(f"還有 {hidden_count} 位玩家")
    return "\n".join(lines)


def _table_result_detail(results: list[DragonGatePlayerResult]) -> str:
    lines: list[str] = []
    for result in results:
        delta = currency_text(amount=result.settlement.delta, signed=True)
        lines.append(f"{result.participant.display_name}: {delta}")
    return "；".join(lines)


def _table_color(results: list[DragonGatePlayerResult]) -> int:
    total_delta = sum(result.settlement.delta for result in results)
    if total_delta > 0:
        return WIN_COLOR
    if total_delta < 0:
        return ERROR_COLOR
    return PUSH_COLOR


def build_dragon_gate_lobby_embed(
    owner: GameParticipant,
    participants: list[GameParticipant],
    ante: int,
    status: str = "等待玩家加入",
) -> Embed:
    """Builds the lobby embed shown before a 射龍門 table starts."""
    embed = Embed(title="♦️ 射龍門 | Lobby", description=status, color=PUSH_COLOR)
    embed.add_field(
        name=f"玩家 {len(participants)}",
        value=_participant_lines(participants=participants, owner_id=owner.user_id),
        inline=False,
    )
    embed.set_footer(
        text=(f"每人先繳底注 {currency_text(amount=ante)} 進彩金池 | 房主開始後輪流下注")
    )
    return embed


def build_dragon_gate_in_progress_embed(round_state: DragonGateRound, dealer_line: str) -> Embed:
    """Builds the active 射龍門 table embed."""
    active_turn = round_state.active_turn
    embed = Embed(title="♦️ 射龍門", description=dealer_quote(text=dealer_line), color=PUSH_COLOR)
    embed.add_field(name="彩金池", value=currency_text(amount=round_state.pot), inline=False)
    if active_turn is not None:
        bet_range = (
            f"{currency_text(amount=round_state.current_min_bet())} 到 "
            f"{currency_text(amount=round_state.current_max_bet())}"
        )
        pair_note = ""
        if active_turn.is_pair:
            pair_note = f"\n同點門柱, {_direction_label(direction=active_turn.direction)}"
        embed.add_field(
            name=f"輪到 {active_turn.participant.display_name}",
            value=(
                f"門柱: **{render_cards(cards=active_turn.pillars)}**{pair_note}\n"
                f"可下注: {bet_range}"
            ),
            inline=False,
        )
    if round_state.last_result is not None:
        embed.add_field(
            name="上一手", value=_result_line(result=round_state.last_result), inline=False
        )
    embed.add_field(
        name="累積戰績", value=_player_delta_lines(round_state=round_state), inline=False
    )
    embed.set_footer(
        text=f"第 {round_state.turn_number} 手 | {DRAGON_GATE_ACTION_TIMEOUT_SECONDS} 秒未操作會結束牌桌"
    )
    return embed


def build_dragon_gate_final_embed(
    round_state: DragonGateRound,
    results: list[DragonGatePlayerResult],
    dealer_line: str,
    reason: str,
) -> Embed:
    """Builds the final embed for a settled 射龍門 table."""
    embed = Embed(
        title="♦️ 射龍門 | 結算",
        description=dealer_quote(text=dealer_line),
        color=_table_color(results=results),
    )
    embed.add_field(name="結束原因", value=reason, inline=False)
    if round_state.last_result is not None:
        embed.add_field(
            name="最後一手", value=_result_line(result=round_state.last_result), inline=False
        )
    for result in results[:DRAGON_GATE_VISIBLE_PLAYER_LINES]:
        embed.add_field(
            name=result.participant.display_name,
            value=settlement_footer(
                delta=result.settlement.delta,
                new_balance=result.settlement.new_balance,
                is_allin=result.participant.is_allin,
            ),
            inline=False,
        )
    hidden_count = len(results) - DRAGON_GATE_VISIBLE_PLAYER_LINES
    if hidden_count > 0:
        embed.add_field(name="其他玩家", value=f"還有 {hidden_count} 位玩家已結算", inline=False)
    return embed


class DragonGateLobbyView(ui.View):
    """Join / leave / start lobby for a 射龍門 game session."""

    def __init__(  # noqa: PLR0913 -- lobby owns all table dependencies
        self,
        owner: GameParticipant,
        ante: int,
        rng: Random,
        dealer: DealerAI,
        dealer_id: int,
        dealer_name: str,
        dealer_avatar_url: str,
        prepare_participant: PrepareDragonGateParticipant,
        refresh_participants: RefreshDragonGateParticipants,
    ) -> None:
        super().__init__(timeout=DRAGON_GATE_ACTION_TIMEOUT_SECONDS)
        self.owner = owner
        self.ante = ante
        self.rng = rng
        self.dealer = dealer
        self.dealer_id = dealer_id
        self.dealer_name = dealer_name
        self.dealer_avatar_url = dealer_avatar_url
        self.prepare_participant = prepare_participant
        self.refresh_participants = refresh_participants
        self.message: Message | None = None
        self._participants: dict[int, GameParticipant] = {owner.user_id: owner}
        self._lock = asyncio.Lock()
        self._started = False

    @property
    def participants(self) -> list[GameParticipant]:
        """Returns participants in join order."""
        return list(self._participants.values())

    async def on_timeout(self) -> None:
        """Cleans up a lobby that never started."""
        if self._started or self.message is None:
            return
        self._disable_buttons()
        self.stop()
        embed = build_dragon_gate_lobby_embed(
            owner=self.owner, participants=self.participants, ante=self.ante, status="Lobby 已逾時"
        )
        with contextlib.suppress(Exception):
            await self.message.edit(embed=embed, view=self)
        schedule_game_message_delete(message=self.message)

    @ui.button(label="加入", emoji="✅", style=ButtonStyle.success, custom_id="dg:lobby:join")
    async def join(self, _button: ui.Button, interaction: Interaction) -> None:
        """Adds the interacting user to the lobby."""
        if interaction.user is None:
            return
        async with self._lock:
            if self._started:
                await self._send_notice(interaction=interaction, content="這桌已經開始了")
                return
            if interaction.user.id in self._participants:
                await self._send_notice(interaction=interaction, content="你已經在這桌了")
                return
            await interaction.response.defer()
            participant = await self.prepare_participant(interaction=interaction, ante=self.ante)
            if participant is None:
                return
            self._participants[participant.user_id] = participant
            await self._refresh_message(
                message=interaction.message, status=f"{participant.display_name} 已加入"
            )

    @ui.button(label="離開", emoji="🚪", style=ButtonStyle.secondary, custom_id="dg:lobby:leave")
    async def leave(self, _button: ui.Button, interaction: Interaction) -> None:
        """Removes the interacting user from the lobby."""
        if interaction.user is None:
            return
        async with self._lock:
            if self._started:
                await self._send_notice(interaction=interaction, content="這桌已經開始了")
                return
            if interaction.user.id == self.owner.user_id:
                await self._send_notice(interaction=interaction, content="房主不能離開 lobby")
                return
            participant = self._participants.pop(interaction.user.id, None)
            if participant is None:
                await self._send_notice(interaction=interaction, content="你不在這桌")
                return
            await interaction.response.defer()
            await self._refresh_message(
                message=interaction.message, status=f"{participant.display_name} 已離開"
            )

    @ui.button(label="開始", emoji="▶️", style=ButtonStyle.primary, custom_id="dg:lobby:start")
    async def start(self, _button: ui.Button, interaction: Interaction) -> None:
        """Starts the game if the lobby owner pressed the button."""
        if interaction.user is None:
            return
        if interaction.user.id != self.owner.user_id:
            await self._send_notice(interaction=interaction, content="只有房主可以開始")
            return
        await interaction.response.defer()
        async with self._lock:
            if self._started:
                await self._send_notice(interaction=interaction, content="這桌已經開始了")
                return
            refreshed, dropped = await self.refresh_participants(
                participants=self.participants, ante=self.ante
            )
            self._participants = {participant.user_id: participant for participant in refreshed}
            if self.owner.user_id not in self._participants:
                await self._send_notice(interaction=interaction, content="你的餘額不足, 不能開始")
                await self._refresh_message(message=interaction.message, status="房主餘額不足")
                return
            self._started = True
        if dropped:
            names = "、".join(dropped)
            await self._send_notice(interaction=interaction, content=f"餘額不足已移出: {names}")
        self.stop()
        await self._start_dragon_gate(message=interaction.message)

    async def _start_dragon_gate(self, message: Message | None) -> None:
        if message is None:
            return
        round_state = DragonGateRound.from_participants(
            rng=self.rng, participants=self.participants, ante=self.ante
        )
        table_balance = sum(participant.balance_at_start for participant in self.participants)
        dealer_line = await self.dealer.taunt_bet(
            author_name=self.owner.account_name,
            player_name=f"{len(self.participants)} 位玩家",
            balance_at_start=table_balance,
            bet=round_state.pot,
            game="dragon_gate",
        )
        view = DragonGateView(
            dealer=self.dealer,
            round_state=round_state,
            owner=self.owner,
            dealer_id=self.dealer_id,
            dealer_name=self.dealer_name,
            dealer_avatar_url=self.dealer_avatar_url,
            dealer_line=dealer_line,
        )
        view.message = message
        view.sync_controls()
        await message.edit(
            embed=build_dragon_gate_in_progress_embed(
                round_state=round_state, dealer_line=dealer_line
            ),
            view=view,
        )

    async def _refresh_message(self, message: Message | None, status: str) -> None:
        if message is None:
            return
        self.message = message
        await message.edit(
            embed=build_dragon_gate_lobby_embed(
                owner=self.owner, participants=self.participants, ante=self.ante, status=status
            ),
            view=self,
        )

    async def _send_notice(self, interaction: Interaction, content: str) -> None:
        try:
            if interaction.response.is_done():
                await interaction.followup.send(content=content, ephemeral=True)
                return
            await interaction.response.send_message(content=content, ephemeral=True)
        except Exception:
            logfire.warn("Failed to send Dragon Gate lobby notice", _exc_info=True)

    async def on_error(self, error: Exception, item: ui.Item, interaction: Interaction) -> None:
        """Logs lobby component failures instead of only printing to stderr."""
        logfire.error(
            "Dragon Gate lobby interaction failed",
            item_label=getattr(item, "label", None),
            user_id=getattr(interaction.user, "id", None),
            _exc_info=(type(error), error, error.__traceback__),
        )

    def _disable_buttons(self) -> None:
        for child in self.children:
            if isinstance(child, ui.Button):
                child.disabled = True


class DragonGateView(ui.View):
    """High / low choice and betting buttons for an active 射龍門 table."""

    def __init__(  # noqa: PLR0913 -- view needs table, dealer, and ledger identity
        self,
        dealer: DealerAI,
        round_state: DragonGateRound,
        owner: GameParticipant,
        dealer_id: int,
        dealer_name: str,
        dealer_line: str,
        dealer_avatar_url: str = "",
    ) -> None:
        super().__init__(timeout=DRAGON_GATE_ACTION_TIMEOUT_SECONDS)
        self.dealer = dealer
        self.round_state = round_state
        self.owner = owner
        self.dealer_id = dealer_id
        self.dealer_name = dealer_name
        self.dealer_avatar_url = dealer_avatar_url
        self.message: Message | None = None
        self._round_lock = asyncio.Lock()
        self._settled = False
        self._dealer_line = dealer_line

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Restricts controls to the active player only."""
        if self._settled:
            return False
        active_turn = self.round_state.active_turn
        if interaction.user is not None and active_turn is not None:
            if interaction.user.id == active_turn.participant.user_id:
                return True
            await interaction.response.send_message(
                content=f"現在輪到 {active_turn.participant.display_name}", ephemeral=True
            )
            return False
        await interaction.response.send_message(content="這桌已經不能操作了", ephemeral=True)
        return False

    async def on_timeout(self) -> None:
        """Finalizes an abandoned table and leaves the remaining pot to the house."""
        if self.message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            await self._finalize_locked(message=self.message, reason="逾時未操作, 剩餘彩金歸莊家")

    @ui.button(label="同點猜大", emoji="⬆️", style=ButtonStyle.secondary, custom_id="dg:higher")
    async def choose_higher(self, _button: ui.Button, interaction: Interaction) -> None:
        """Chooses higher for a same-point gate."""
        await self._choose_direction(interaction=interaction, direction="higher")

    @ui.button(label="同點猜小", emoji="⬇️", style=ButtonStyle.secondary, custom_id="dg:lower")
    async def choose_lower(self, _button: ui.Button, interaction: Interaction) -> None:
        """Chooses lower for a same-point gate."""
        await self._choose_direction(interaction=interaction, direction="lower")

    @ui.button(label="底注", emoji="🪙", style=ButtonStyle.primary, custom_id="dg:min")
    async def bet_minimum(self, _button: ui.Button, interaction: Interaction) -> None:
        """Bets the current minimum legal amount."""
        await self._place_button_bet(
            interaction=interaction, amount=self.round_state.current_min_bet()
        )

    @ui.button(label="半池", emoji="🌓", style=ButtonStyle.primary, custom_id="dg:half")
    async def bet_half_pot(self, _button: ui.Button, interaction: Interaction) -> None:
        """Bets roughly half the current pot."""
        amount = max(self.round_state.current_min_bet(), self.round_state.current_max_bet() // 2)
        await self._place_button_bet(interaction=interaction, amount=amount)

    @ui.button(label="全池", emoji="💰", style=ButtonStyle.danger, custom_id="dg:all")
    async def bet_full_pot(self, _button: ui.Button, interaction: Interaction) -> None:
        """Bets the full current pot."""
        await self._place_button_bet(
            interaction=interaction, amount=self.round_state.current_max_bet()
        )

    @ui.button(label="自訂", emoji="✏️", style=ButtonStyle.secondary, custom_id="dg:custom")
    async def bet_custom(self, _button: ui.Button, interaction: Interaction) -> None:
        """Opens a modal for an exact bet amount."""
        if self.round_state.needs_pair_choice():
            await interaction.response.send_message(
                content="同點門柱要先猜大或猜小", ephemeral=True
            )
            return
        modal = DragonGateBetModal(
            view=self,
            minimum=self.round_state.current_min_bet(),
            maximum=self.round_state.current_max_bet(),
        )
        await interaction.response.send_modal(modal=modal)

    async def submit_custom_bet(self, interaction: Interaction, raw_amount: str | None) -> None:
        """Handles the custom bet modal submission."""
        try:
            amount = int((raw_amount or "").replace(",", "").strip())
        except ValueError:
            await interaction.response.send_message(content="下注金額要是整數", ephemeral=True)
            return
        await interaction.response.defer()
        await self._place_bet_locked_by_interaction(interaction=interaction, amount=amount)

    def sync_controls(self) -> None:
        """Updates button labels and disabled states from the current table state."""
        needs_pair_choice = self.round_state.needs_pair_choice()
        can_bet = (
            not self._settled
            and self.round_state.active_turn is not None
            and not needs_pair_choice
        )
        minimum = self.round_state.current_min_bet()
        maximum = self.round_state.current_max_bet()
        half = max(minimum, maximum // 2)

        self._button(custom_id="dg:higher").disabled = not needs_pair_choice
        self._button(custom_id="dg:lower").disabled = not needs_pair_choice
        min_button = self._button(custom_id="dg:min")
        half_button = self._button(custom_id="dg:half")
        all_button = self._button(custom_id="dg:all")
        custom_button = self._button(custom_id="dg:custom")
        min_button.disabled = not can_bet
        half_button.disabled = not can_bet
        all_button.disabled = not can_bet
        custom_button.disabled = not can_bet
        min_button.label = f"底注 {minimum:,}"
        half_button.label = f"半池 {half:,}"
        all_button.label = f"全池 {maximum:,}"

    async def _choose_direction(
        self, interaction: Interaction, direction: DragonGateDirection
    ) -> None:
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
                embed=build_dragon_gate_in_progress_embed(
                    round_state=self.round_state, dealer_line=self._dealer_line
                ),
                view=self,
            )

    async def _place_button_bet(self, interaction: Interaction, amount: int) -> None:
        await interaction.response.defer()
        await self._place_bet_locked_by_interaction(interaction=interaction, amount=amount)

    async def _place_bet_locked_by_interaction(
        self, interaction: Interaction, amount: int
    ) -> None:
        if interaction.user is None:
            return
        message = interaction.message or self.message
        if message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            try:
                self.round_state.place_bet(user_id=interaction.user.id, amount=amount)
            except DragonGateError as error:
                await self._send_bet_error_notice(interaction=interaction, error=error)
                return
            if self.round_state.finished:
                await self._finalize_locked(message=message, reason="彩金池清空")
                return
            self.sync_controls()
            await message.edit(
                embed=build_dragon_gate_in_progress_embed(
                    round_state=self.round_state, dealer_line=self._dealer_line
                ),
                view=self,
            )

    async def _finalize_locked(self, message: Message, reason: str) -> None:
        if self._settled:
            return
        self._settled = True
        results: list[DragonGatePlayerResult] = []
        for participant in self.round_state.participants:
            delta = self.round_state.player_delta(user_id=participant.user_id)
            settlement = await settle_dragon_gate_player(
                player_id=participant.user_id,
                player_account_name=participant.account_name,
                player_avatar_url=participant.avatar_url,
                dealer_id=self.dealer_id,
                dealer_name=self.dealer_name,
                dealer_avatar_url=self.dealer_avatar_url,
                delta=delta,
            )
            results.append(DragonGatePlayerResult(participant=participant, settlement=settlement))

        dealer_line = await self.dealer.table_settle(
            author_name=self.owner.account_name,
            table_name="射龍門",
            player_count=len(results),
            net_delta=sum(result.settlement.delta for result in results),
            game="dragon_gate",
            detail=_table_result_detail(results=results),
        )
        embed = build_dragon_gate_final_embed(
            round_state=self.round_state, results=results, dealer_line=dealer_line, reason=reason
        )
        self._disable_buttons()
        self.stop()
        with contextlib.suppress(Exception):
            await message.edit(embed=embed, view=self)
        schedule_game_message_delete(message=message)

    def _button(self, custom_id: str) -> ui.Button:
        for child in self.children:
            if isinstance(child, ui.Button) and child.custom_id == custom_id:
                return child
        raise RuntimeError(f"Missing button: {custom_id}")

    def _current_turn_notice(self) -> str:
        active_turn = self.round_state.active_turn
        if active_turn is None:
            return "這桌已經不能操作了"
        return f"現在輪到 {active_turn.participant.display_name}"

    async def _send_bet_error_notice(
        self, interaction: Interaction, error: DragonGateError
    ) -> None:
        if isinstance(error, DragonGatePairChoiceRequiredError):
            content = "同點門柱要先猜大或猜小"
        elif isinstance(error, DragonGateTableFinishedError):
            content = "這桌已經不能操作了"
        elif isinstance(error, DragonGateTurnError):
            content = self._current_turn_notice()
        elif isinstance(error, DragonGateBetRangeError):
            content = (
                "下注金額需介於 "
                f"{currency_text(amount=self.round_state.current_min_bet())} 到 "
                f"{currency_text(amount=self.round_state.current_max_bet())}"
            )
        else:
            content = "這桌已經不能操作了"
        await self._send_notice(interaction=interaction, content=content)

    async def _send_notice(self, interaction: Interaction, content: str) -> None:
        try:
            if interaction.response.is_done():
                await interaction.followup.send(content=content, ephemeral=True)
                return
            await interaction.response.send_message(content=content, ephemeral=True)
        except Exception:
            logfire.warn("Failed to send Dragon Gate action notice", _exc_info=True)

    async def on_error(self, error: Exception, item: ui.Item, interaction: Interaction) -> None:
        """Logs active-table component failures instead of only printing to stderr."""
        logfire.error(
            "Dragon Gate action interaction failed",
            item_label=getattr(item, "label", None),
            user_id=getattr(interaction.user, "id", None),
            _exc_info=(type(error), error, error.__traceback__),
        )

    def _disable_buttons(self) -> None:
        for child in self.children:
            if isinstance(child, ui.Button):
                child.disabled = True


class DragonGateBetModal(ui.Modal):
    """Modal for entering an exact 射龍門 bet amount."""

    def __init__(self, view: DragonGateView, minimum: int, maximum: int) -> None:
        super().__init__(title="自訂下注")
        self.view = view
        self.amount = ui.TextInput(
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


__all__ = [
    "DRAGON_GATE_ACTION_TIMEOUT_SECONDS",
    "DragonGateBetModal",
    "DragonGateLobbyView",
    "DragonGateView",
    "build_dragon_gate_final_embed",
    "build_dragon_gate_in_progress_embed",
    "build_dragon_gate_lobby_embed",
]
