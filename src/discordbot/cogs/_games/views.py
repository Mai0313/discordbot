"""Interactive components for multiplayer casino game sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol
import asyncio
import contextlib

from nextcord import Embed, Message, ButtonStyle, Interaction, ui

from discordbot.typings.games import GameParticipant, BlackjackPlayerResult
from discordbot.cogs._games.cleanup import schedule_game_message_delete
from discordbot.cogs._games.blackjack import BlackjackRound, BlackjackPlayerHand, render_hand
from discordbot.cogs._games.settlement import settle_blackjack_round, blackjack_early_finish_note
from discordbot.cogs._games.presentation import (
    PUSH_COLOR,
    ERROR_COLOR,
    allin_note,
    dealer_quote,
    settlement_footer,
    blackjack_outcome_presentation,
)
from discordbot.cogs._economy.presentation import currency_text

if TYPE_CHECKING:
    from random import Random

    from discordbot.cogs._games.dealer import DealerAI

MAX_BLACKJACK_PLAYERS: Final[int] = 5
BLACKJACK_ACTION_TIMEOUT_SECONDS: Final[int] = 180


class PrepareParticipant(Protocol):
    """Callable used by lobby join buttons to validate a participant."""

    async def __call__(
        self, *, interaction: Interaction, requested_bet: int
    ) -> GameParticipant | None:
        """Returns a prepared participant or sends the interaction error."""


class RefreshParticipants(Protocol):
    """Callable used by lobby start to re-check balances."""

    async def __call__(
        self, *, participants: list[GameParticipant], requested_bet: int
    ) -> tuple[list[GameParticipant], list[str]]:
        """Returns refreshed participants and display names removed from the table."""


def _format_player_line(*, player: BlackjackPlayerHand) -> str:
    return f"{render_hand(cards=player.cards)}  **{player.total()}**"


def _format_dealer_line(*, round_state: BlackjackRound, hide_hole: bool) -> str:
    if hide_hole:
        return f"{render_hand(cards=round_state.dealer, hide_first=True)}  **?**"
    return f"{render_hand(cards=round_state.dealer)}  **{round_state.dealer_total()}**"


def _status_for_player(
    *, round_state: BlackjackRound, player: BlackjackPlayerHand, active_user_id: int | None
) -> str:
    if player.is_blackjack():
        return "Blackjack"
    if player.is_bust():
        return "爆牌"
    if active_user_id == player.participant.user_id and not round_state.finished:
        return "輪到你"
    if player.finished:
        return "stand"
    return "等待"


def _participant_lines(*, participants: list[GameParticipant], owner_id: int) -> str:
    lines: list[str] = []
    for index, participant in enumerate(participants, start=1):
        owner_label = " 發起者" if participant.user_id == owner_id else ""
        wager = currency_text(amount=participant.bet)
        lines.append(f"{index}. {participant.display_name}{owner_label} | 下注 {wager}")
    return "\n".join(lines)


def build_blackjack_lobby_embed(
    *,
    owner: GameParticipant,
    participants: list[GameParticipant],
    requested_bet: int,
    max_players: int,
    status: str = "等待玩家加入",
) -> Embed:
    """Builds the lobby embed shown before a Blackjack table starts."""
    embed = Embed(title="♠️ 21 點 | Lobby", description=status, color=PUSH_COLOR)
    embed.add_field(
        name=f"玩家 {len(participants)}/{max_players}",
        value=_participant_lines(participants=participants, owner_id=owner.user_id),
        inline=False,
    )
    embed.set_footer(
        text=(f"每人最高下注 {currency_text(amount=requested_bet)} | 發起者開始後進入同一張桌")
    )
    return embed


def build_in_progress_embed(
    *, dealer_name: str, round_state: BlackjackRound, dealer_line: str
) -> Embed:
    """Builds the shared Blackjack table embed while players are acting."""
    active = round_state.active_player()
    if active is None:
        status = "準備結算"
        active_user_id = None
    else:
        status = f"輪到 {active.participant.display_name}"
        active_user_id = active.participant.user_id

    embed = Embed(title="♠️ 21 點", description=dealer_quote(text=dealer_line), color=PUSH_COLOR)
    embed.add_field(
        name=dealer_name,
        value=_format_dealer_line(round_state=round_state, hide_hole=True),
        inline=False,
    )
    for player in round_state.players:
        participant = player.participant
        value = (
            f"{_format_player_line(player=player)}\n"
            f"{_status_for_player(round_state=round_state, player=player, active_user_id=active_user_id)}"
            f" | 下注 {currency_text(amount=participant.bet)}"
            f"{allin_note(is_allin=participant.is_allin)}"
        )
        embed.add_field(name=participant.display_name, value=value, inline=False)
    embed.set_footer(text=f"{status} | 不操作 {BLACKJACK_ACTION_TIMEOUT_SECONDS} 秒會自動 stand")
    return embed


def _table_result_detail(*, results: list[BlackjackPlayerResult]) -> str:
    lines: list[str] = []
    for result in results:
        outcome_label, _color = blackjack_outcome_presentation(outcome=result.settlement.outcome)
        delta = currency_text(amount=result.settlement.delta, signed=True)
        lines.append(f"{result.participant.display_name}: {outcome_label} {delta}")
    return "；".join(lines)


def _table_color(*, results: list[BlackjackPlayerResult]) -> int:
    total_delta = sum(result.settlement.delta for result in results)
    if total_delta > 0:
        return 0x57F287
    if total_delta < 0:
        return ERROR_COLOR
    return PUSH_COLOR


def build_final_embed(
    *,
    dealer_name: str,
    round_state: BlackjackRound,
    results: list[BlackjackPlayerResult],
    dealer_line: str,
) -> Embed:
    """Builds the final embed for a settled Blackjack table."""
    embed = Embed(
        title="♠️ 21 點 | 結算",
        description=dealer_quote(text=dealer_line),
        color=_table_color(results=results),
    )
    embed.add_field(
        name=dealer_name,
        value=_format_dealer_line(round_state=round_state, hide_hole=False),
        inline=False,
    )
    for result in results:
        participant = result.participant
        player = next(
            player
            for player in round_state.players
            if player.participant.user_id == participant.user_id
        )
        outcome_label, _color = blackjack_outcome_presentation(outcome=result.settlement.outcome)
        note = blackjack_early_finish_note(hand=round_state.settlement_hand(player=player))
        note_line = f"\n{note}" if note else ""
        value = (
            f"{_format_player_line(player=player)}\n"
            f"**{outcome_label}**{note_line}\n"
            + settlement_footer(
                delta=result.settlement.delta,
                new_balance=result.settlement.new_balance,
                is_allin=participant.is_allin,
            )
        )
        embed.add_field(name=participant.display_name, value=value, inline=False)
    return embed


class BlackjackLobbyView(ui.View):
    """Join / leave / start lobby for a Blackjack game session."""

    def __init__(  # noqa: PLR0913 -- lobby owns all table dependencies
        self,
        *,
        owner: GameParticipant,
        requested_bet: int,
        rng: Random,
        dealer: DealerAI,
        dealer_id: int,
        dealer_name: str,
        dealer_avatar_url: str,
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
    ) -> None:
        super().__init__(timeout=BLACKJACK_ACTION_TIMEOUT_SECONDS)
        self.owner = owner
        self.requested_bet = requested_bet
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
        embed = build_blackjack_lobby_embed(
            owner=self.owner,
            participants=self.participants,
            requested_bet=self.requested_bet,
            max_players=MAX_BLACKJACK_PLAYERS,
            status="Lobby 已逾時",
        )
        with contextlib.suppress(Exception):
            await self.message.edit(embed=embed, view=self)
        schedule_game_message_delete(message=self.message)

    @ui.button(label="加入", emoji="✅", style=ButtonStyle.success)
    async def join(self, _button: ui.Button, interaction: Interaction) -> None:
        """Adds the interacting user to the lobby."""
        await interaction.response.defer()
        if interaction.user is None:
            return
        async with self._lock:
            if self._started:
                await self._send_notice(interaction=interaction, content="這桌已經開始了")
                return
            if interaction.user.id in self._participants:
                await self._send_notice(interaction=interaction, content="你已經在這桌了")
                return
            if len(self._participants) >= MAX_BLACKJACK_PLAYERS:
                await self._send_notice(interaction=interaction, content="這桌已經滿了")
                return
            participant = await self.prepare_participant(
                interaction=interaction, requested_bet=self.requested_bet
            )
            if participant is None:
                return
            self._participants[participant.user_id] = participant
            await self._refresh_message(
                message=interaction.message, status=f"{participant.display_name} 已加入"
            )

    @ui.button(label="離開", emoji="🚪", style=ButtonStyle.secondary)
    async def leave(self, _button: ui.Button, interaction: Interaction) -> None:
        """Removes the interacting user from the lobby."""
        await interaction.response.defer()
        if interaction.user is None:
            return
        async with self._lock:
            if self._started:
                await self._send_notice(interaction=interaction, content="這桌已經開始了")
                return
            if interaction.user.id == self.owner.user_id:
                await self._send_notice(interaction=interaction, content="發起者不能離開 lobby")
                return
            participant = self._participants.pop(interaction.user.id, None)
            if participant is None:
                await self._send_notice(interaction=interaction, content="你不在這桌")
                return
            await self._refresh_message(
                message=interaction.message, status=f"{participant.display_name} 已離開"
            )

    @ui.button(label="開始", emoji="▶️", style=ButtonStyle.primary)
    async def start(self, _button: ui.Button, interaction: Interaction) -> None:
        """Starts the game if the lobby owner pressed the button."""
        await interaction.response.defer()
        if interaction.user is None:
            return
        if interaction.user.id != self.owner.user_id:
            await self._send_notice(interaction=interaction, content="只有發起者可以開始")
            return
        async with self._lock:
            if self._started:
                await self._send_notice(interaction=interaction, content="這桌已經開始了")
                return
            refreshed, dropped = await self.refresh_participants(
                participants=self.participants, requested_bet=self.requested_bet
            )
            self._participants = {participant.user_id: participant for participant in refreshed}
            if self.owner.user_id not in self._participants:
                await self._send_notice(interaction=interaction, content="你的餘額不足, 不能開始")
                await self._refresh_message(message=interaction.message, status="發起者餘額不足")
                return
            self._started = True
            if dropped:
                names = "、".join(dropped)
                await self._send_notice(
                    interaction=interaction, content=f"餘額不足已移出: {names}"
                )
            self.stop()
            await self._start_blackjack(message=interaction.message)

    async def _start_blackjack(self, *, message: Message | None) -> None:
        if message is None:
            return
        round_state = BlackjackRound.from_participants(
            rng=self.rng, participants=self.participants
        )
        round_state.deal_initial()
        table_bet = sum(participant.bet for participant in self.participants)
        table_balance = sum(participant.balance_at_start for participant in self.participants)
        dealer_line = await self.dealer.taunt_bet(
            author_name=self.owner.account_name,
            player_name=f"{len(self.participants)} 位玩家",
            balance_at_start=table_balance,
            bet=table_bet,
            game="blackjack",
        )
        view = BlackjackView(
            dealer=self.dealer,
            round_state=round_state,
            starter_id=self.owner.user_id,
            author_name=self.owner.account_name,
            dealer_id=self.dealer_id,
            dealer_name=self.dealer_name,
            dealer_avatar_url=self.dealer_avatar_url,
        )
        view.message = message
        if round_state.finished:
            await view.finalize(message=message)
            return
        await message.edit(
            embed=build_in_progress_embed(
                dealer_name=self.dealer_name, round_state=round_state, dealer_line=dealer_line
            ),
            view=view,
        )

    async def _refresh_message(self, *, message: Message | None, status: str) -> None:
        if message is None:
            return
        self.message = message
        await message.edit(
            embed=build_blackjack_lobby_embed(
                owner=self.owner,
                participants=self.participants,
                requested_bet=self.requested_bet,
                max_players=MAX_BLACKJACK_PLAYERS,
                status=status,
            ),
            view=self,
        )

    async def _send_notice(self, *, interaction: Interaction, content: str) -> None:
        with contextlib.suppress(Exception):
            await interaction.followup.send(content=content, ephemeral=True)

    def _disable_buttons(self) -> None:
        for child in self.children:
            if isinstance(child, ui.Button):
                child.disabled = True


class BlackjackView(ui.View):
    """Hit / Stand buttons for an active multiplayer Blackjack table."""

    def __init__(  # noqa: PLR0913 -- view needs table, dealer, and ledger identity
        self,
        *,
        dealer: DealerAI,
        round_state: BlackjackRound,
        starter_id: int,
        author_name: str,
        dealer_id: int,
        dealer_name: str,
        dealer_avatar_url: str = "",
    ) -> None:
        super().__init__(timeout=BLACKJACK_ACTION_TIMEOUT_SECONDS)
        self.dealer = dealer
        self.round_state = round_state
        self.starter_id = starter_id
        self.author_name = author_name
        self.dealer_id = dealer_id
        self.dealer_name = dealer_name
        self.dealer_avatar_url = dealer_avatar_url
        self.message: Message | None = None
        self._round_lock = asyncio.Lock()
        self._settled = False
        self._dealer_line = "下好離手, 不要等下哭"

    async def interaction_check(self, interaction: Interaction) -> bool:
        """Restricts Hit / Stand to the active player only."""
        if self._settled:
            return False
        active = self.round_state.active_player()
        if interaction.user is not None and active is not None:
            if interaction.user.id == active.participant.user_id:
                return True
            await interaction.response.send_message(
                content=f"現在輪到 {active.participant.display_name}", ephemeral=True
            )
            return False
        await interaction.response.send_message(content="這局已經不能操作了", ephemeral=True)
        return False

    async def on_timeout(self) -> None:
        """Auto-stands unresolved players and settles the table."""
        if self.message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            self.round_state.stand_all_remaining()
            await self._finalize_locked(message=self.message)

    @ui.button(label="再要一張", emoji="🃏", style=ButtonStyle.primary)
    async def hit(self, _button: ui.Button, interaction: Interaction) -> None:
        """Handles the active player's Hit button."""
        await interaction.response.defer()
        if interaction.message is None:
            return
        async with self._round_lock:
            if self._settled or self.round_state.finished:
                return
            active = self.round_state.active_player()
            if active is None:
                await self._finalize_locked(message=interaction.message)
                return
            self.round_state.hit(user_id=active.participant.user_id)
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            active_after_hit = self.round_state.active_player()
            if (
                active_after_hit is not None
                and active_after_hit.participant.user_id == active.participant.user_id
            ):
                self._dealer_line = await self.dealer.hint(
                    author_name=active.participant.account_name,
                    player_name=active.participant.display_name,
                    player_total=active_after_hit.total(),
                    dealer_visible=self.round_state.dealer_visible_value(),
                )
            await self._edit_in_progress_locked(message=interaction.message)

    @ui.button(label="停手", emoji="✋", style=ButtonStyle.secondary)
    async def stand(self, _button: ui.Button, interaction: Interaction) -> None:
        """Handles the active player's Stand button."""
        await interaction.response.defer()
        if interaction.message is None:
            return
        async with self._round_lock:
            if self._settled or self.round_state.finished:
                return
            active = self.round_state.active_player()
            if active is None:
                await self._finalize_locked(message=interaction.message)
                return
            self.round_state.stand(user_id=active.participant.user_id)
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._edit_in_progress_locked(message=interaction.message)

    async def finalize(self, *, message: Message) -> None:
        """Settles every player exactly once."""
        async with self._round_lock:
            await self._finalize_locked(message=message)

    async def _finalize(self, *, message: Message) -> None:
        """Backward-compatible wrapper for tests and older internal callers."""
        await self.finalize(message=message)

    async def _edit_in_progress_locked(self, *, message: Message) -> None:
        embed = build_in_progress_embed(
            dealer_name=self.dealer_name,
            round_state=self.round_state,
            dealer_line=self._dealer_line,
        )
        await message.edit(embed=embed, view=self)

    async def _finalize_locked(self, *, message: Message) -> None:
        if self._settled:
            return
        self._settled = True
        if not self.round_state.finished:
            self.round_state.stand_all_remaining()

        results: list[BlackjackPlayerResult] = []
        for player in self.round_state.players:
            participant = player.participant
            settlement = await settle_blackjack_round(
                hand=self.round_state.settlement_hand(player=player),
                player_id=participant.user_id,
                player_account_name=participant.account_name,
                player_avatar_url=participant.avatar_url,
                dealer_id=self.dealer_id,
                dealer_name=self.dealer_name,
                dealer_avatar_url=self.dealer_avatar_url,
            )
            results.append(BlackjackPlayerResult(participant=participant, settlement=settlement))

        dealer_line = await self._settlement_line(results=results)
        embed = build_final_embed(
            dealer_name=self.dealer_name,
            round_state=self.round_state,
            results=results,
            dealer_line=dealer_line,
        )
        self._disable_buttons()
        self.stop()
        with contextlib.suppress(Exception):
            await message.edit(embed=embed, view=self)
        schedule_game_message_delete(message=message)

    async def _settlement_line(self, *, results: list[BlackjackPlayerResult]) -> str:
        if len(results) == 1:
            result = results[0]
            return await self.dealer.settle(
                author_name=result.participant.account_name,
                player_name=result.participant.display_name,
                outcome=result.settlement.outcome,
                bet=result.participant.bet,
                delta=result.settlement.delta,
                new_balance=result.settlement.new_balance,
                game="blackjack",
                detail=result.settlement.detail,
            )
        return await self.dealer.table_settle(
            author_name=self.author_name,
            table_name="Blackjack table",
            player_count=len(results),
            net_delta=sum(result.settlement.delta for result in results),
            game="blackjack",
            detail=_table_result_detail(results=results),
        )

    def _disable_buttons(self) -> None:
        for child in self.children:
            if isinstance(child, ui.Button):
                child.disabled = True


__all__: list[str] = [
    "MAX_BLACKJACK_PLAYERS",
    "BlackjackLobbyView",
    "BlackjackView",
    "blackjack_outcome_presentation",
    "build_blackjack_lobby_embed",
    "build_final_embed",
    "build_in_progress_embed",
]
