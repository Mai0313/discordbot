"""Interactive components for multiplayer casino game sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast
import asyncio
import contextlib

import logfire
import nextcord
from nextcord import Embed, Message, ButtonStyle, Interaction
from nextcord.ui import View, Button

from discordbot.typings.games import (
    Card,
    GameParticipant,
    BlackjackDealerStep,
    BlackjackDealerAction,
    BlackjackPlayerResult,
)
from discordbot.cogs._games.lobby import BaseGameLobbyView, PrepareParticipant, RefreshParticipants
from discordbot.cogs._games.cleanup import schedule_game_message_delete
from discordbot.cogs._games.blackjack import (
    BlackjackRound,
    BlackjackHandState,
    BlackjackPlayerHand,
    can_split,
    can_double,
    hand_value,
    render_hand,
    can_surrender,
    dealer_up_card,
    committed_wagers,
)
from discordbot.cogs._games.settlement import (
    settle_blackjack_player,
    blackjack_player_early_finish_note,
)
from discordbot.cogs._games.interactions import (
    send_ephemeral_notice,
    disable_view_components,
    edit_message_with_retry,
)
from discordbot.cogs._games.presentation import (
    WIN_COLOR,
    LOSE_COLOR,
    PUSH_COLOR,
    WIN_RESULT_EMOJI,
    BUST_RESULT_EMOJI,
    IN_PROGRESS_COLOR,
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
MAX_DEALER_DECISION_STEPS: Final[int] = 8
FINAL_EDIT_TIMEOUT_SECONDS: Final[float] = 8.0
PEEK_REVEAL_DELAY_SECONDS: Final[float] = 1.6


def _hand_summary_line(cards: list[Card], suffix: str = "") -> str:
    """H1 heading combining the hand and its total, e.g. ``# 10♠  5♥ = 15``."""
    if not cards:
        return ""
    spaced = render_hand(cards=cards).replace(" ", "  ")
    return f"# {spaced} = {hand_value(cards=cards)}{suffix}"


def _format_dealer_block(round_state: BlackjackRound, hide_hole: bool) -> str:
    """Formats dealer cards for an in-progress or final table embed."""
    if hide_hole:
        if not round_state.dealer:
            return ""
        if len(round_state.dealer) == 1:
            return card_line(cards_text=str(round_state.dealer[0]))
        return card_line(cards_text=render_hand(cards=round_state.dealer, hide_first=True))
    return _hand_summary_line(cards=round_state.dealer)


def _format_dealer_decision_path(steps: list[BlackjackDealerStep]) -> str:
    """Formats compact AI dealer decisions for the final embed."""
    if not steps:
        return ""
    parts: list[str] = []
    for step in steps:
        part = f"{step.total_before} {step.action}"
        if step.drawn_card is not None:
            part += f" 抽 {step.drawn_card}"
            if step.total_after is not None:
                part += f" → {step.total_after}"
        if step.fallback:
            part += " (fallback basic rule)"
        elif step.forced:
            part += " (guard)"
        parts.append(part)
    return "；".join(parts)


def _hand_status_suffix(hand: BlackjackHandState, is_active: bool) -> str:  # noqa: PLR0911 -- ladder of mutually exclusive hand states; flattening hurts clarity
    """Returns the inline status label appended to one sub-hand's total."""
    if hand.surrendered:
        return " 🏳️ 投降"
    if hand.is_blackjack():
        return f" {NATURAL_RESULT_EMOJI} BLACKJACK"
    if hand.is_bust():
        return f" {BUST_RESULT_EMOJI} 爆牌"
    if hand.doubled and hand.finished:
        return " 💰 doubled"
    if hand.finished:
        return " ✋ stand"
    if is_active:
        return " ▶ 進行中"
    return ""


def _hand_metadata_text(hand: BlackjackHandState, participant: GameParticipant) -> str:
    """Returns the small-text metadata for one sub-hand."""
    parts: list[str] = [f"下注 `{hand.bet:,}`"]
    if hand.is_split_hand:
        parts.append("split A" if hand.is_split_aces else "split")
    if hand.doubled:
        parts.append("doubled")
    if participant.is_allin and not hand.is_split_hand and not hand.doubled:
        parts.append("all-in")
    return " · ".join(parts)


def _insurance_phase_status(player: BlackjackPlayerHand) -> str:
    """Returns the per-player status text shown during the insurance phase."""
    if player.insurance_bet > 0:
        return f"保險 `{player.insurance_bet:,}`"
    if player.insurance_resolved:
        return "已拒絕保險"
    return "保險待決定"


def _format_player_block(
    player: BlackjackPlayerHand, active_hand_index: int | None, insurance_status: str | None
) -> str:
    """Formats one player's hands and wager metadata for the table embed."""
    lines: list[str] = []
    for index, hand in enumerate(player.hands):
        is_active = active_hand_index == index
        summary = _hand_summary_line(
            cards=hand.cards, suffix=_hand_status_suffix(hand=hand, is_active=is_active)
        )
        meta = _hand_metadata_text(hand=hand, participant=player.participant)
        lines.append(summary)
        lines.append(metadata_line(text=meta))
    if insurance_status:
        lines.append(metadata_line(text=insurance_status))
    return "\n".join(lines)


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


def _footer_status(round_state: BlackjackRound) -> str:
    """Returns the in-progress footer status text."""
    if round_state.phase == "insurance":
        undecided = sum(1 for player in round_state.players if not player.insurance_resolved)
        return f"保險決定中 · 莊家明牌 A · 等待 {undecided} 位玩家決定"
    active = round_state.active_player()
    if active is None:
        return "準備結算"
    if len(active.hands) > 1:
        return f"輪到 {active.participant.display_name} 第 {round_state.current_hand_index + 1} 手"
    return f"輪到 {active.participant.display_name}"


def build_in_progress_embed(
    dealer_name: str, round_state: BlackjackRound, force_show_hole: bool = False
) -> Embed:
    """Builds the shared Blackjack table embed while players are acting.

    Pass ``force_show_hole=True`` during peek-reveal animations so the dealer
    hole card is uncovered even before the round reaches the dealer phase.
    """
    description_parts: list[str] = [
        f"### {dealer_name}",
        _format_dealer_block(round_state=round_state, hide_hole=not force_show_hole),
    ]
    for player_index, player in enumerate(round_state.players):
        participant = player.participant
        active_hand_index: int | None = None
        if (
            round_state.current_player_index == player_index
            and round_state.phase == "player_actions"
        ):
            active_hand_index = round_state.current_hand_index
        insurance_status: str | None = None
        if round_state.phase == "insurance":
            insurance_status = _insurance_phase_status(player=player)
        elif player.insurance_bet > 0:
            insurance_status = f"保險 `{player.insurance_bet:,}`"
        description_parts.append("")
        description_parts.append(f"### {participant.display_name}")
        description_parts.append(
            _format_player_block(
                player=player,
                active_hand_index=active_hand_index,
                insurance_status=insurance_status,
            )
        )

    color = IN_PROGRESS_COLOR if round_state.phase == "insurance" else PUSH_COLOR
    embed = Embed(title="♠️ 二十一點", description="\n".join(description_parts), color=color)
    if round_state.players and round_state.players[0].participant.avatar_url:
        embed.set_thumbnail(url=round_state.players[0].participant.avatar_url)
    embed.set_footer(
        text=(
            f"{_footer_status(round_state=round_state)} · "
            f"不操作 {BLACKJACK_ACTION_TIMEOUT_SECONDS} 秒會自動 stand"
        )
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


def _dealer_hand_status(hand: BlackjackHandState) -> str:
    """Returns a compact status label for one sub-hand."""
    if hand.surrendered:
        return "surrender"
    if hand.is_blackjack():
        return "blackjack"
    if hand.is_bust():
        return "bust"
    if hand.doubled:
        return "doubled"
    if hand.finished:
        return "stand"
    return "acting"


def _dealer_decision_table_state(round_state: BlackjackRound) -> str:
    """Builds the full table state sent to the AI dealer."""
    soft, total = round_state.dealer_is_soft_total()
    lines: list[str] = [
        "遊戲: 21 點",
        f"莊家手牌: {render_hand(cards=round_state.dealer)}",
        f"莊家總點數: {total}",
        f"莊家是否 soft total: {'是' if soft else '否'}",
        f"莊家是否 soft 17: {'是' if round_state.dealer_is_soft_17() else '否'}",
        "玩家:",
    ]
    for player in round_state.players:
        participant = player.participant
        for index, hand in enumerate(player.hands):
            label = (
                participant.display_name
                if len(player.hands) == 1
                else f"{participant.display_name} (手{index + 1})"
            )
            lines.append(
                f"- {label}: "
                f"{render_hand(cards=hand.cards)} = {hand.total()}, "
                f"bet={hand.bet}, status={_dealer_hand_status(hand=hand)}"
            )
    return "\n".join(lines)


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
        if len(player.hands) == 1 and result.settlement.insurance is None:
            return "♠️ 二十一點 · " + player_result_inline(
                outcome=result.settlement.outcome,
                player_total=player.hands[0].total(),
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
    dealer_name: str,
    round_state: BlackjackRound,
    results: list[BlackjackPlayerResult],
    dealer_steps: list[BlackjackDealerStep] | None = None,
) -> Embed:
    """Builds the final embed for a settled Blackjack table."""
    dealer_total = round_state.dealer_total()
    dealer_decision_path = _format_dealer_decision_path(steps=dealer_steps or [])
    description_parts: list[str] = [
        f"### {dealer_name}",
        _format_dealer_block(round_state=round_state, hide_hole=False),
    ]
    if dealer_decision_path:
        description_parts.append(metadata_line(text=f"AI決策: {dealer_decision_path}"))
    for result in results:
        participant = result.participant
        player = next(
            player
            for player in round_state.players
            if player.participant.user_id == participant.user_id
        )
        description_parts.append("")
        description_parts.append(f"### {participant.display_name}")
        for hand_index, hand_settlement in enumerate(result.settlement.hands):
            cards = hand_settlement.cards
            summary = _hand_summary_line(cards=cards)
            hand_total = hand_value(cards=cards)
            title = player_result_title(
                outcome=hand_settlement.outcome, player_total=hand_total, dealer_total=dealer_total
            )
            if len(result.settlement.hands) > 1:
                description_parts.append(metadata_line(text=f"手{hand_index + 1}"))
            description_parts.append(f"{summary}\n{title}")
        if result.settlement.insurance is not None:
            ins = result.settlement.insurance
            label = (
                f"保險 `{ins.bet:,}` → 中獎 (+{ins.delta:,})"
                if ins.won
                else f"保險 `{ins.bet:,}` → 莊家無 BJ ({ins.delta:+,})"
            )
            description_parts.append(metadata_line(text=label))
        metadata = settlement_metadata(
            delta=result.settlement.delta,
            new_balance=result.settlement.new_balance,
            is_allin=participant.is_allin,
            base_delta=result.settlement.base_delta,
            vip_bonus=result.settlement.vip_bonus,
        )
        description_parts.append(metadata)
        note = blackjack_player_early_finish_note(
            player=player, dealer=round_state.dealer, peeked_blackjack=round_state.peeked_blackjack
        )
        if note:
            description_parts.append(metadata_line(text=note))

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

    async def _start_game(self, message: Message | None) -> bool:
        """Deals the table and replaces the lobby message with the game view."""
        if message is None:
            return False
        round_state = BlackjackRound.from_participants(
            rng=self.rng, participants=self.participants, auto_play_dealer=False
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
            return True
        view.sync_buttons()
        await edit_message_with_retry(
            message=message,
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
        return True


class BlackjackView(View):
    """Hit / Stand / Double / Split / Surrender / Insurance controls."""

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
        self.round_state.auto_play_dealer = False
        self._dealer_steps: list[BlackjackDealerStep] = []
        self._peek_animated = False
        self._insurance_buttons: tuple[Button, Button] = (
            cast("Button", self.insure_yes),
            cast("Button", self.insure_no),
        )

    async def interaction_check(self, interaction: Interaction) -> bool:  # noqa: PLR0911 -- phase + identity gating naturally fans out into early returns
        """Restricts buttons to the active player (or any undecided insurance player)."""
        if self._settled:
            await send_ephemeral_notice(
                interaction=interaction,
                content="這局已經結束, 等下一局吧",
                log_message="Failed to send Blackjack settled notice",
            )
            return False
        if interaction.user is None:
            return False
        if self.round_state.phase == "insurance":
            player = next(
                (
                    candidate
                    for candidate in self.round_state.players
                    if candidate.participant.user_id == interaction.user.id
                ),
                None,
            )
            if player is None:
                await interaction.response.send_message(content="你不在這個牌桌", ephemeral=True)
                return False
            if player.insurance_resolved:
                await interaction.response.send_message(content="你已決定過保險", ephemeral=True)
                return False
            return True
        active = self.round_state.active_player()
        if active is not None and interaction.user.id == active.participant.user_id:
            return True
        if active is not None:
            await interaction.response.send_message(
                content=f"現在輪到 {active.participant.display_name}", ephemeral=True
            )
            return False
        await interaction.response.send_message(content="這局已經不能操作了", ephemeral=True)
        return False

    async def on_timeout(self) -> None:
        """Auto-resolves the round when nobody clicked in time."""
        if self.message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            if self.round_state.phase == "insurance":
                self.round_state.decline_insurance_for_all_unresolved()
                if self.round_state.finished:
                    await self._finalize_locked(message=self.message)
                    return
            self.round_state.stand_all_remaining()
            await self._finalize_locked(message=self.message)

    @nextcord.ui.button(
        label="再要一張", emoji="🃏", style=ButtonStyle.primary, custom_id="bj:hit", row=0
    )
    async def hit(self, _button: Button, interaction: Interaction) -> None:
        """Handles the active player's Hit button."""
        await interaction.response.defer()
        if interaction.message is None or interaction.user is None:
            return
        async with self._round_lock:
            if self._settled or self.round_state.finished:
                return
            active = self.round_state.active_player()
            if active is None:
                await self._finalize_locked(message=interaction.message)
                return
            try:
                self.round_state.hit(user_id=interaction.user.id)
            except ValueError:
                await self._reject_stale_action_locked(
                    interaction=interaction, message=interaction.message
                )
                return
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            active_after_hit = self.round_state.active_player()
            active_hand = self.round_state.active_hand()
            if (
                active_after_hit is not None
                and active_after_hit.participant.user_id == interaction.user.id
                and active_hand is not None
            ):
                self._dealer_line = await self.dealer.hint(
                    author_name=active_after_hit.participant.account_name,
                    player_name=active_after_hit.participant.display_name,
                    player_total=active_hand.total(),
                    dealer_visible=self.round_state.dealer_visible_value(),
                )
            await self._edit_in_progress_locked(message=interaction.message)

    @nextcord.ui.button(
        label="停手", emoji="✋", style=ButtonStyle.secondary, custom_id="bj:stand", row=0
    )
    async def stand(self, _button: Button, interaction: Interaction) -> None:
        """Handles the active player's Stand button."""
        await interaction.response.defer()
        if interaction.message is None or interaction.user is None:
            return
        async with self._round_lock:
            if self._settled or self.round_state.finished:
                return
            active = self.round_state.active_player()
            if active is None:
                await self._finalize_locked(message=interaction.message)
                return
            try:
                self.round_state.stand(user_id=interaction.user.id)
            except ValueError:
                await self._reject_stale_action_locked(
                    interaction=interaction, message=interaction.message
                )
                return
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._edit_in_progress_locked(message=interaction.message)

    @nextcord.ui.button(
        label="加倍", emoji="💰", style=ButtonStyle.success, custom_id="bj:double", row=0
    )
    async def double(self, _button: Button, interaction: Interaction) -> None:
        """Doubles the active hand's bet and finishes it after one draw."""
        await interaction.response.defer()
        if interaction.message is None or interaction.user is None:
            return
        async with self._round_lock:
            if self._settled or self.round_state.finished:
                return
            active = self.round_state.active_player()
            if active is None:
                await self._finalize_locked(message=interaction.message)
                return
            try:
                self.round_state.double_down(user_id=interaction.user.id)
            except ValueError:
                await self._reject_stale_action_locked(
                    interaction=interaction, message=interaction.message
                )
                return
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._edit_in_progress_locked(message=interaction.message)

    @nextcord.ui.button(
        label="分牌", emoji="🪓", style=ButtonStyle.success, custom_id="bj:split", row=0
    )
    async def split(self, _button: Button, interaction: Interaction) -> None:
        """Splits the active pair into two sibling sub-hands."""
        await interaction.response.defer()
        if interaction.message is None or interaction.user is None:
            return
        async with self._round_lock:
            if self._settled or self.round_state.finished:
                return
            active = self.round_state.active_player()
            if active is None:
                await self._finalize_locked(message=interaction.message)
                return
            try:
                self.round_state.split(user_id=interaction.user.id)
            except ValueError:
                await self._reject_stale_action_locked(
                    interaction=interaction, message=interaction.message
                )
                return
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._edit_in_progress_locked(message=interaction.message)

    @nextcord.ui.button(
        label="投降", emoji="🏳️", style=ButtonStyle.danger, custom_id="bj:surrender", row=0
    )
    async def surrender(self, _button: Button, interaction: Interaction) -> None:
        """Surrenders the active hand for a half-bet refund."""
        await interaction.response.defer()
        if interaction.message is None or interaction.user is None:
            return
        async with self._round_lock:
            if self._settled or self.round_state.finished:
                return
            active = self.round_state.active_player()
            if active is None:
                await self._finalize_locked(message=interaction.message)
                return
            try:
                self.round_state.surrender(user_id=interaction.user.id)
            except ValueError:
                await self._reject_stale_action_locked(
                    interaction=interaction, message=interaction.message
                )
                return
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._edit_in_progress_locked(message=interaction.message)

    @nextcord.ui.button(
        label="保險 ½", emoji="🛡️", style=ButtonStyle.success, custom_id="bj:insure_yes", row=1
    )
    async def insure_yes(self, _button: Button, interaction: Interaction) -> None:
        """Takes insurance for the calling player."""
        await interaction.response.defer()
        if interaction.message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            if interaction.user is None:
                return
            player = next(
                (
                    candidate
                    for candidate in self.round_state.players
                    if candidate.participant.user_id == interaction.user.id
                ),
                None,
            )
            if player is None:
                return
            try:
                self.round_state.take_insurance(
                    user_id=interaction.user.id, amount=player.participant.bet // 2
                )
            except ValueError as error:
                content = (
                    "餘額不足，不能買保險"
                    if "balance" in str(error).lower()
                    else "現在不能買保險，請看最新牌桌"
                )
                await send_ephemeral_notice(
                    interaction=interaction,
                    content=content,
                    log_message="Failed to send Blackjack insurance rejection notice",
                )
                await self._edit_in_progress_locked(message=interaction.message)
                return
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._maybe_animate_insurance_close_locked(message=interaction.message)
            await self._edit_in_progress_locked(message=interaction.message)

    @nextcord.ui.button(
        label="不保險", emoji="❌", style=ButtonStyle.secondary, custom_id="bj:insure_no", row=1
    )
    async def insure_no(self, _button: Button, interaction: Interaction) -> None:
        """Declines insurance for the calling player."""
        await interaction.response.defer()
        if interaction.message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            if interaction.user is None:
                return
            try:
                self.round_state.decline_insurance(user_id=interaction.user.id)
            except ValueError:
                await self._edit_in_progress_locked(message=interaction.message)
                return
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._maybe_animate_insurance_close_locked(message=interaction.message)
            await self._edit_in_progress_locked(message=interaction.message)

    async def finalize(self, message: Message) -> None:
        """Settles every player exactly once."""
        async with self._round_lock:
            await self._finalize_locked(message=message)

    def sync_buttons(self) -> None:  # noqa: C901 -- single switch over six button kinds; splitting hurts readability
        """Recomputes ``disabled`` on every action / insurance button."""
        in_insurance = self.round_state.phase == "insurance"
        in_actions = self.round_state.phase == "player_actions"
        self._sync_insurance_button_visibility(in_insurance=in_insurance)
        active_player = self.round_state.active_player() if in_actions else None
        active_hand = self.round_state.active_hand() if in_actions else None
        balance_remaining = 0
        if active_player is not None:
            balance_remaining = active_player.participant.balance_at_start - committed_wagers(
                player=active_player
            )
        for child in self.children:
            if not isinstance(child, Button):
                continue
            cid = child.custom_id or ""
            if cid in ("bj:insure_yes", "bj:insure_no"):
                child.disabled = not in_insurance
                continue
            if not in_actions or active_hand is None:
                child.disabled = True
                continue
            if cid == "bj:hit":
                child.disabled = active_hand.finished or active_hand.is_split_aces
            elif cid == "bj:stand":
                child.disabled = active_hand.finished
            elif cid == "bj:double":
                child.disabled = not can_double(
                    hand=active_hand, balance_remaining=balance_remaining
                )
            elif cid == "bj:split":
                child.disabled = not can_split(
                    hand=active_hand, balance_remaining=balance_remaining
                )
            elif cid == "bj:surrender":
                child.disabled = not can_surrender(
                    hand=active_hand, peeked_blackjack=self.round_state.peeked_blackjack
                )

    def _sync_insurance_button_visibility(self, in_insurance: bool) -> None:
        """Adds insurance controls only while the table is in insurance phase."""
        for button in self._insurance_buttons:
            if in_insurance and button not in self.children:
                self.add_item(item=button)
            elif not in_insurance and button in self.children:
                self.remove_item(item=button)

    async def _edit_in_progress_locked(self, message: Message) -> None:
        """Refreshes dealer talk and table embeds while holding the round lock."""
        self.sync_buttons()
        talk_embed = build_dealer_talk_embed(
            dealer_line=self._dealer_line,
            dealer_name=self.dealer_name,
            dealer_avatar_url=self.dealer_avatar_url,
        )
        main_embed = build_in_progress_embed(
            dealer_name=self.dealer_name, round_state=self.round_state
        )
        await message.edit(embeds=[talk_embed, main_embed], view=self)

    async def _reject_stale_action_locked(
        self, interaction: Interaction, message: Message
    ) -> None:
        """Sends a private stale-action notice and refreshes the table."""
        await send_ephemeral_notice(
            interaction=interaction,
            content="這個操作已經失效，請看最新牌桌",
            log_message="Failed to send Blackjack stale action notice",
        )
        await self._edit_in_progress_locked(message=message)

    async def _finalize_locked(self, message: Message) -> None:
        """Applies settlements and publishes the final table embeds once.

        The view is disabled and stopped before any long-running ``await``
        (dealer LLM, DB settlement, message edit) so a stalled coroutine
        cannot leave players staring at live-looking buttons. ``interaction_check``
        then replies with an ephemeral notice instead of falling through to
        Discord's "interaction failed" timeout.
        """
        if self._settled:
            return
        self._settled = True
        if self.round_state.phase == "insurance":
            self.round_state.decline_insurance_for_all_unresolved()
        if not self.round_state.finished:
            self.round_state.stand_all_remaining()
        self._disable_buttons()
        self.stop()
        await self._safe_edit_view_locked(message=message)
        logfire.info("Blackjack finalize started", players=len(self.round_state.players))

        if self.round_state.peeked_blackjack and not self._peek_animated:
            self._peek_animated = True
            await self._animate_peek_reveal_bj_locked(message=message)

        await self._edit_dealer_thinking_locked(message=message)
        await self._play_dealer_locked()
        logfire.info("Blackjack dealer phase done", dealer_total=self.round_state.dealer_total())

        results: list[BlackjackPlayerResult] = []
        for player in self.round_state.players:
            participant = player.participant
            settlement = await settle_blackjack_player(
                round_state=self.round_state,
                player=player,
                player_id=participant.user_id,
                player_account_name=participant.account_name,
                player_avatar_url=participant.avatar_url,
                dealer_id=self.dealer_id,
                dealer_name=self.dealer_name,
                dealer_avatar_url=self.dealer_avatar_url,
            )
            results.append(BlackjackPlayerResult(participant=participant, settlement=settlement))
        logfire.info("Blackjack settlement done", results=len(results))

        dealer_line = await self._settlement_line(results=results)
        talk_embed = build_dealer_talk_embed(
            dealer_line=dealer_line,
            dealer_name=self.dealer_name,
            dealer_avatar_url=self.dealer_avatar_url,
        )
        final_embed = build_final_embed(
            dealer_name=self.dealer_name,
            round_state=self.round_state,
            results=results,
            dealer_steps=self._dealer_steps,
        )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                message.edit(embeds=[talk_embed, final_embed], view=self),
                timeout=FINAL_EDIT_TIMEOUT_SECONDS,
            )
        logfire.info("Blackjack final edit done")
        schedule_game_message_delete(message=message)

    async def _safe_edit_view_locked(self, message: Message) -> None:
        """Refreshes only the view so disabled buttons are visible immediately."""
        with contextlib.suppress(Exception):
            await asyncio.wait_for(message.edit(view=self), timeout=FINAL_EDIT_TIMEOUT_SECONDS)

    async def _animate_peek_locked(
        self, message: Message, *, intro_line: str, reveal_line: str
    ) -> None:
        """Renders the dealer hole-card peek as a 2-stage reveal.

        Stage 1 keeps the hole card hidden while the dealer "peeks", stage 2
        flips it face-up. Buttons stay disabled throughout so the caller can
        safely chain finalize / further edits after the animation returns.
        """
        self._disable_buttons()
        intro_talk = build_dealer_talk_embed(
            dealer_line=intro_line,
            dealer_name=self.dealer_name,
            dealer_avatar_url=self.dealer_avatar_url,
        )
        body_hidden = build_in_progress_embed(
            dealer_name=self.dealer_name, round_state=self.round_state
        )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                message.edit(embeds=[intro_talk, body_hidden], view=self),
                timeout=FINAL_EDIT_TIMEOUT_SECONDS,
            )
        await asyncio.sleep(PEEK_REVEAL_DELAY_SECONDS)

        reveal_talk = build_dealer_talk_embed(
            dealer_line=reveal_line,
            dealer_name=self.dealer_name,
            dealer_avatar_url=self.dealer_avatar_url,
        )
        reveal_body = build_in_progress_embed(
            dealer_name=self.dealer_name, round_state=self.round_state, force_show_hole=True
        )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                message.edit(embeds=[reveal_talk, reveal_body], view=self),
                timeout=FINAL_EDIT_TIMEOUT_SECONDS,
            )
        await asyncio.sleep(PEEK_REVEAL_DELAY_SECONDS)

    async def _animate_peek_reveal_bj_locked(self, message: Message) -> None:
        """Peek animation when the hole card revealed a Blackjack."""
        up = dealer_up_card(dealer=self.round_state.dealer)
        up_label = str(up) if up is not None else "明牌"
        await self._animate_peek_locked(
            message=message,
            intro_line=f"莊家明牌 {up_label}, 慢慢翻開 hole card 看一眼...",
            reveal_line="Boom, 莊家 21 點, 本局直接結算",
        )

    async def _animate_peek_no_bj_locked(self, message: Message) -> None:
        """Peek animation after insurance closes without a dealer Blackjack."""
        up = dealer_up_card(dealer=self.round_state.dealer)
        up_label = str(up) if up is not None else "明牌"
        await self._animate_peek_locked(
            message=message,
            intro_line=f"莊家明牌 {up_label}, 翻開 hole card 看一眼...",
            reveal_line="嘖, hole card 不到位, 遊戲繼續",
        )

    async def _maybe_animate_insurance_close_locked(self, message: Message) -> None:
        """Plays the no-BJ peek reveal once when insurance phase ends without BJ."""
        if self._peek_animated:
            return
        if not self.round_state.insurance_offered:
            return
        if self.round_state.peeked_blackjack:
            return
        if self.round_state.phase != "player_actions":
            return
        self._peek_animated = True
        await self._animate_peek_no_bj_locked(message=message)

    async def _edit_dealer_thinking_locked(self, message: Message) -> None:
        """Shows that the dealer is choosing hit / stand before the AI call."""
        if self.round_state.dealer_played or not self.round_state.needs_dealer_play():
            return
        self.sync_buttons()
        self._disable_buttons()
        talk_embed = build_dealer_talk_embed(
            dealer_line="莊家正在思考 hit / stand, 請稍等",
            dealer_name=self.dealer_name,
            dealer_avatar_url=self.dealer_avatar_url,
        )
        main_embed = build_in_progress_embed(
            dealer_name=self.dealer_name, round_state=self.round_state
        )
        main_embed.set_footer(text="莊家正在思考 hit / stand, 請稍等")
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                message.edit(embeds=[talk_embed, main_embed], view=self),
                timeout=FINAL_EDIT_TIMEOUT_SECONDS,
            )

    async def _play_dealer_locked(self) -> None:
        """Runs the AI dealer phase before settlement."""
        if self.round_state.dealer_played or not self.round_state.needs_dealer_play():
            return

        use_basic_rule = False
        for _step_index in range(MAX_DEALER_DECISION_STEPS):
            total_before = self.round_state.dealer_total()
            if total_before > 21:
                self.round_state.mark_dealer_played()
                return
            if total_before == 21:
                self._dealer_steps.append(
                    BlackjackDealerStep(
                        total_before=total_before,
                        action="stand",
                        reason="guard: 21 點必停",
                        forced=True,
                    )
                )
                self.round_state.mark_dealer_played()
                return

            action: BlackjackDealerAction
            if total_before <= 16:
                action = "hit"
                reason = "guard: 16 點以下必 hit"
                fallback = False
                forced = True
            elif use_basic_rule:
                action = "hit" if total_before < 17 else "stand"
                reason = "fallback basic rule"
                fallback = True
                forced = False
            else:
                decision = await self.dealer.decide_blackjack_action(
                    author_name=self.author_name,
                    table_state=_dealer_decision_table_state(round_state=self.round_state),
                    dealer_total=total_before,
                )
                action = decision.action
                reason = decision.reason
                fallback = reason.startswith("basic rule:")
                forced = False
                if fallback:
                    use_basic_rule = True
                if total_before >= 18 and action == "hit":
                    logfire.warn(
                        "Dealer LLM suggested hit at >=18; overriding to stand",
                        dealer_total=total_before,
                        original_reason=reason,
                    )
                    action = "stand"
                    reason = "override: 18+ 必停 (原 LLM 回 hit)"
                    forced = True

            drawn_card: Card | None = None
            total_after: int | None = None
            if action == "stand":
                self._dealer_steps.append(
                    BlackjackDealerStep(
                        total_before=total_before,
                        action="stand",
                        reason=reason,
                        fallback=fallback,
                        forced=forced,
                    )
                )
                self.round_state.mark_dealer_played()
                return

            drawn_card = self.round_state.draw_dealer_card()
            total_after = self.round_state.dealer_total()
            self._dealer_steps.append(
                BlackjackDealerStep(
                    total_before=total_before,
                    action="hit",
                    reason=reason,
                    drawn_card=drawn_card,
                    total_after=total_after,
                    fallback=fallback,
                    forced=forced,
                )
            )
        logfire.warn(
            "Dealer Blackjack decision loop reached maximum steps; forcing stand",
            max_steps=MAX_DEALER_DECISION_STEPS,
        )
        self._dealer_steps.append(
            BlackjackDealerStep(
                total_before=self.round_state.dealer_total(),
                action="stand",
                reason="guard: decision limit",
                forced=True,
            )
        )
        self.round_state.mark_dealer_played()

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
        """Disables every action / insurance control after settlement."""
        self._sync_insurance_button_visibility(in_insurance=self.round_state.phase == "insurance")
        disable_view_components(children=self.children, component_types=(Button,))


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
