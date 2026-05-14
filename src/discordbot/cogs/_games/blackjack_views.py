"""Interactive components for multiplayer casino game sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
import asyncio
import contextlib

from nextcord import Embed, Message, ButtonStyle, Interaction, ui

from discordbot.typings.games import Card, GameParticipant, BlackjackPlayerResult
from discordbot.cogs._games.lobby import BaseGameLobbyView, PrepareParticipant, RefreshParticipants
from discordbot.cogs._games.cleanup import schedule_game_message_delete
from discordbot.cogs._games.blackjack import (
    BlackjackRound,
    BlackjackPlayerHand,
    hand_value,
    render_hand,
)
from discordbot.cogs._games.settlement import settle_blackjack_round, blackjack_early_finish_note
from discordbot.cogs._games.interactions import disable_view_components
from discordbot.cogs._games.presentation import (
    WIN_COLOR,
    LOSE_COLOR,
    PUSH_COLOR,
    WIN_RESULT_EMOJI,
    BUST_RESULT_EMOJI,
    LOSE_RESULT_EMOJI,
    NATURAL_RESULT_EMOJI,
    LOBBY_PLAYERS_FIELD_EMOJI,
    card_line,
    metadata_line,
    player_result_title,
    settlement_metadata,
    player_result_inline,
    lobby_participant_line,
    build_dealer_talk_embed,
    blackjack_outcome_presentation,
)
from discordbot.cogs._economy.presentation import currency_text

if TYPE_CHECKING:
    from random import Random

    from discordbot.cogs._games.dealer import DealerAI

MAX_BLACKJACK_PLAYERS: Final[int] = 5
BLACKJACK_ACTION_TIMEOUT_SECONDS: Final[int] = 180


def _hand_summary_line(cards: list[Card], suffix: str = "") -> str:
    """H1 heading combining the hand and its total, e.g. ``# 10♠  5♥ = 15``."""
    if not cards:
        return ""
    spaced = render_hand(cards=cards).replace(" ", "  ")
    return f"# {spaced} = {hand_value(cards=cards)}{suffix}"


def _format_dealer_block(round_state: BlackjackRound, hide_hole: bool) -> str:
    """Formats dealer cards for an in-progress or final table embed."""
    if hide_hole:
        if len(round_state.dealer) >= 2:
            up_card = round_state.dealer[1]
            return card_line(cards_text=str(up_card))
        return ""
    return _hand_summary_line(cards=round_state.dealer)


def _player_status_suffix(player: BlackjackPlayerHand) -> str:
    """Returns the inline status label appended to a player's hand total."""
    if player.is_blackjack():
        return f" {NATURAL_RESULT_EMOJI} BLACKJACK"
    if player.is_bust():
        return f" {BUST_RESULT_EMOJI} 爆牌"
    if player.finished:
        return " ✋ stand"
    return ""


def _format_player_block(player: BlackjackPlayerHand) -> str:
    """Formats one player's hand and wager metadata for the table embed."""
    summary = _hand_summary_line(cards=player.cards, suffix=_player_status_suffix(player=player))
    participant = player.participant
    bet_text = f"下注 `{participant.bet:,}`"
    if participant.is_allin:
        bet_text += " · all-in"
    return f"{summary}\n{metadata_line(text=bet_text)}"


def _participant_lines(participants: list[GameParticipant]) -> str:
    """Formats lobby participants in join order."""
    lines: list[str] = []
    for index, participant in enumerate(participants, start=1):
        lines.append(
            lobby_participant_line(
                index=index,
                display_name=participant.display_name,
                bet=participant.bet,
                is_allin=participant.is_allin,
            )
        )
    return "\n".join(lines)


def build_blackjack_lobby_embed(
    owner: GameParticipant,
    participants: list[GameParticipant],
    requested_bet: int,
    max_players: int,
    status: str = "等待玩家加入",
) -> Embed:
    """Builds the lobby embed shown before a Blackjack table starts."""
    embed = Embed(title="♠️ 二十一點 · 開桌準備", color=PUSH_COLOR)
    if status and status != "等待玩家加入":
        embed.description = status
    embed.add_field(
        name=f"{LOBBY_PLAYERS_FIELD_EMOJI} 桌上玩家 ({len(participants)}/{max_players})",
        value=_participant_lines(participants=participants),
        inline=False,
    )
    if owner.avatar_url:
        embed.set_thumbnail(url=owner.avatar_url)
    embed.set_footer(text=f"基本下注 {currency_text(amount=requested_bet)}")
    return embed


def build_in_progress_embed(dealer_name: str, round_state: BlackjackRound) -> Embed:
    """Builds the shared Blackjack table embed while players are acting."""
    active = round_state.active_player()
    footer_status = "準備結算" if active is None else f"輪到 {active.participant.display_name}"

    description_parts: list[str] = [
        f"### {dealer_name}",
        _format_dealer_block(round_state=round_state, hide_hole=True),
    ]
    for player in round_state.players:
        participant = player.participant
        description_parts.append("")
        description_parts.append(f"### {participant.display_name}")
        description_parts.append(_format_player_block(player=player))

    embed = Embed(title="♠️ 二十一點", description="\n".join(description_parts), color=PUSH_COLOR)
    if round_state.players and round_state.players[0].participant.avatar_url:
        embed.set_thumbnail(url=round_state.players[0].participant.avatar_url)
    embed.set_footer(
        text=f"{footer_status} · 不操作 {BLACKJACK_ACTION_TIMEOUT_SECONDS} 秒會自動 stand"
    )
    return embed


def _table_result_detail(results: list[BlackjackPlayerResult]) -> str:
    """Formats compact per-player settlement details for dealer banter."""
    lines: list[str] = []
    for result in results:
        outcome_label, _color = blackjack_outcome_presentation(outcome=result.settlement.outcome)
        delta = currency_text(amount=result.settlement.delta, signed=True)
        lines.append(f"{result.participant.display_name}: {outcome_label} {delta}")
    return "；".join(lines)


def _table_color(results: list[BlackjackPlayerResult]) -> int:
    """Returns the final embed color from the table's net player result."""
    total_delta = sum(result.settlement.delta for result in results)
    if total_delta > 0:
        return WIN_COLOR
    if total_delta < 0:
        return LOSE_COLOR
    return PUSH_COLOR


def _final_title(
    results: list[BlackjackPlayerResult], dealer_total: int, round_state: BlackjackRound
) -> str:
    """Builds the final Blackjack title for single or multiplayer results."""
    if len(results) == 1:
        result = results[0]
        player = next(
            player
            for player in round_state.players
            if player.participant.user_id == result.participant.user_id
        )
        return "♠️ 二十一點 · " + player_result_inline(
            outcome=result.settlement.outcome,
            player_total=player.total(),
            dealer_total=dealer_total,
        )
    wins = sum(1 for result in results if result.settlement.delta > 0)
    losses = sum(1 for result in results if result.settlement.delta < 0)
    pushes = len(results) - wins - losses
    parts: list[str] = []
    if wins:
        parts.append(f"{WIN_RESULT_EMOJI} {wins} 勝")
    if losses:
        parts.append(f"{LOSE_RESULT_EMOJI} {losses} 負")
    if pushes:
        parts.append(f"{pushes} 平")
    return "♠️ 二十一點 · " + " ".join(parts)


def build_final_embed(
    dealer_name: str, round_state: BlackjackRound, results: list[BlackjackPlayerResult]
) -> Embed:
    """Builds the final embed for a settled Blackjack table."""
    dealer_total = round_state.dealer_total()
    description_parts: list[str] = [
        f"### {dealer_name}",
        _format_dealer_block(round_state=round_state, hide_hole=False),
    ]
    for result in results:
        participant = result.participant
        player = next(
            player
            for player in round_state.players
            if player.participant.user_id == participant.user_id
        )
        summary = _hand_summary_line(cards=player.cards)
        title = player_result_title(
            outcome=result.settlement.outcome,
            player_total=player.total(),
            dealer_total=dealer_total,
        )
        metadata = settlement_metadata(
            delta=result.settlement.delta,
            new_balance=result.settlement.new_balance,
            is_allin=participant.is_allin,
        )
        note = blackjack_early_finish_note(hand=round_state.settlement_hand(player=player))
        note_segment = f"\n{metadata_line(text=note)}" if note else ""
        description_parts.append("")
        description_parts.append(f"### {participant.display_name}")
        description_parts.append(f"{summary}\n{title}\n{metadata}{note_segment}")

    embed = Embed(
        title=_final_title(results=results, dealer_total=dealer_total, round_state=round_state),
        description="\n".join(description_parts),
        color=_table_color(results=results),
    )
    if round_state.players and round_state.players[0].participant.avatar_url:
        embed.set_thumbnail(url=round_state.players[0].participant.avatar_url)
    return embed


class BlackjackLobbyView(BaseGameLobbyView):
    """Join / leave / start lobby for a Blackjack game session."""

    max_players = MAX_BLACKJACK_PLAYERS

    def __init__(  # noqa: PLR0913 -- lobby owns all table dependencies
        self,
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
        """Initializes a Blackjack lobby with wager and dealer identity."""
        super().__init__(
            owner=owner,
            rng=rng,
            dealer=dealer,
            dealer_name=dealer_name,
            dealer_avatar_url=dealer_avatar_url,
            prepare_participant=prepare_participant,
            refresh_participants=refresh_participants,
            timeout=BLACKJACK_ACTION_TIMEOUT_SECONDS,
        )
        self.requested_bet = requested_bet
        self.dealer_id = dealer_id

    def _build_lobby_embed(self, status: str = "等待玩家加入") -> Embed:
        """Builds the Blackjack lobby embed from current participants."""
        return build_blackjack_lobby_embed(
            owner=self.owner,
            participants=self.participants,
            requested_bet=self.requested_bet,
            max_players=MAX_BLACKJACK_PLAYERS,
            status=status,
        )

    async def _start_game(self, message: Message | None) -> None:
        """Deals the table and replaces the lobby message with the game view."""
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
            dealer_line=dealer_line,
        )
        view.message = message
        if round_state.finished:
            await view.finalize(message=message)
            return
        await message.edit(
            embeds=[
                build_dealer_talk_embed(
                    dealer_line=dealer_line,
                    dealer_name=self.dealer_name,
                    dealer_avatar_url=self.dealer_avatar_url,
                ),
                build_in_progress_embed(dealer_name=self.dealer_name, round_state=round_state),
            ],
            view=view,
        )


class BlackjackView(ui.View):
    """Hit / Stand buttons for an active multiplayer Blackjack table."""

    def __init__(  # noqa: PLR0913 -- view needs table, dealer, and ledger identity
        self,
        dealer: DealerAI,
        round_state: BlackjackRound,
        starter_id: int,
        author_name: str,
        dealer_id: int,
        dealer_name: str,
        dealer_avatar_url: str = "",
        dealer_line: str = "下好離手, 不要等下哭",
    ) -> None:
        """Initializes the active Blackjack table view."""
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
        self._dealer_line = dealer_line

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

    async def finalize(self, message: Message) -> None:
        """Settles every player exactly once."""
        async with self._round_lock:
            await self._finalize_locked(message=message)

    async def _finalize(self, message: Message) -> None:
        """Backward-compatible wrapper for tests and older internal callers."""
        await self.finalize(message=message)

    async def _edit_in_progress_locked(self, message: Message) -> None:
        """Refreshes dealer talk and table embeds while holding the round lock."""
        talk_embed = build_dealer_talk_embed(
            dealer_line=self._dealer_line,
            dealer_name=self.dealer_name,
            dealer_avatar_url=self.dealer_avatar_url,
        )
        main_embed = build_in_progress_embed(
            dealer_name=self.dealer_name, round_state=self.round_state
        )
        await message.edit(embeds=[talk_embed, main_embed], view=self)

    async def _finalize_locked(self, message: Message) -> None:
        """Applies settlements and publishes the final table embeds once."""
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
        talk_embed = build_dealer_talk_embed(
            dealer_line=dealer_line,
            dealer_name=self.dealer_name,
            dealer_avatar_url=self.dealer_avatar_url,
        )
        final_embed = build_final_embed(
            dealer_name=self.dealer_name, round_state=self.round_state, results=results
        )
        self._disable_buttons()
        self.stop()
        with contextlib.suppress(Exception):
            await message.edit(embeds=[talk_embed, final_embed], view=self)
        schedule_game_message_delete(message=message)

    async def _settlement_line(self, results: list[BlackjackPlayerResult]) -> str:
        """Builds single-player or table-level dealer settlement banter."""
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
        """Disables Hit and Stand controls after settlement."""
        disable_view_components(children=self.children, component_types=(ui.Button,))


__all__: list[str] = [
    "BLACKJACK_ACTION_TIMEOUT_SECONDS",
    "MAX_BLACKJACK_PLAYERS",
    "BlackjackLobbyView",
    "BlackjackView",
    "PrepareParticipant",
    "RefreshParticipants",
    "blackjack_outcome_presentation",
    "build_blackjack_lobby_embed",
    "build_final_embed",
    "build_in_progress_embed",
]
