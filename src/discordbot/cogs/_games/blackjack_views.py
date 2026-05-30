"""Interactive components for multiplayer casino game sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast
import asyncio
import contextlib

import logfire
import nextcord
from nextcord import Embed, Message, ButtonStyle, Interaction
from nextcord.ui import View, Button

from discordbot.typings.games import (
    Card,
    BotAction,
    SettleOutcome,
    GameParticipant,
    OtherPlayerView,
    BlackjackDealerStep,
    BotFinancialContext,
    BlackjackPlayerResult,
    BlackjackDealerStepSource,
    BlackjackPlayerSettlement,
)
from discordbot.cogs._games.lobby import BaseGameLobbyView, PrepareParticipant, RefreshParticipants
from discordbot.utils.discord_embeds import embed_spacer_payload
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
    is_five_card_win,
    is_five_card_twenty_one,
)
from discordbot.utils.message_cleanup import schedule_public_message_delete
from discordbot.cogs._economy.database import get_account, get_casino_daily_stats
from discordbot.cogs._games.bot_player import build_bot_action_context, build_bot_insurance_context
from discordbot.cogs._games.settlement import (
    settle_blackjack_player,
    blackjack_player_early_finish_note,
)
from discordbot.cogs._games.interactions import (
    send_ephemeral_notice,
    set_view_item_visible,
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
    NATURAL_RESULT_EMOJI,
    LOBBY_PLAYERS_FIELD_EMOJI,
    card_line,
    metadata_line,
    player_result_title,
    settlement_metadata,
    lobby_participant_line,
    build_system_talk_embed,
    blackjack_outcome_presentation,
)
from discordbot.cogs._economy.presentation import amount_code, currency_text

if TYPE_CHECKING:
    from random import Random
    from collections.abc import Coroutine

    from discordbot.cogs._games.dealer import SystemNarrator
    from discordbot.cogs._games.bot_player import BotPlayerAI

MAX_BLACKJACK_PLAYERS: Final[int] = 6
BLACKJACK_ACTION_TIMEOUT_SECONDS: Final[int] = 180
MAX_DEALER_DECISION_STEPS: Final[int] = 8
MAX_BOT_TURN_STEPS: Final[int] = 16
FINAL_EDIT_TIMEOUT_SECONDS: Final[float] = 8.0
PEEK_REVEAL_DELAY_SECONDS: Final[float] = 1.6
BOT_TURN_EDIT_DELAY_SECONDS: Final[float] = 0.4
HintRefreshContext = tuple[int, str, int, int]
BLACKJACK_SETTLEMENT_FALLBACK_LINES: Final[dict[SettleOutcome, str]] = {
    "win": "本局玩家獲勝, 賭場已支付賠付",
    "lose": "本局玩家未過關, 籌碼歸入賭場",
    "push": "本局雙方點數一致, 押注全額退回",
    "blackjack": "Blackjack 達成, 賭場依規則支付 1.5 倍賠付",
    "five_card_win": "過五關未爆, 玩家獲得本局勝利",
    "five_card_twenty_one": "過五關 21 點, 額外加碼支付",
    "player_bust": "玩家點數超過 21, 本局結算為輸",
    "dealer_bust": "莊家點數超過 21, 本局玩家獲勝",
    "surrender": "玩家投降, 退回一半本金",
}


def _blackjack_table_edit_kwargs(
    *, embeds: list[Embed], view: View | None, target: object | None = None
) -> dict[str, Any]:
    """Builds the shared edit payload for Blackjack table renders."""
    return {
        "embeds": embeds,
        "view": view,
        **embed_spacer_payload(embeds=embeds, is_edit=True, target=target),
    }


def _hand_summary_line(cards: list[Card], suffix: str = "") -> str:
    """H1 heading combining the hand and its total, e.g. `# 10♠  5♥ = 15`."""
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
    """Formats compact dealer actions for the final embed."""
    if not steps:
        return ""
    source_labels: dict[BlackjackDealerStepSource, str] = {"auto": "規則", "guard": "防呆"}
    parts: list[str] = []
    for step in steps:
        part = f"{source_labels[step.source]}: {step.total_before} {step.action}"
        if step.drawn_card is not None:
            part += f" 抽 {step.drawn_card}"
            if step.total_after is not None:
                part += f" → {step.total_after}"
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
    if is_five_card_twenty_one(cards=hand.cards):
        return f" {NATURAL_RESULT_EMOJI} 過五關 21"
    if is_five_card_win(cards=hand.cards):
        return f" {WIN_RESULT_EMOJI} 過五關"
    if hand.doubled and hand.finished:
        return " 💰 doubled"
    if hand.finished:
        return " ✋ stand"
    if is_active:
        return " ▶ 進行中"
    return ""


def _hand_metadata_text(hand: BlackjackHandState, participant: GameParticipant) -> str:
    """Returns the small-text metadata for one sub-hand."""
    parts: list[str] = [f"下注 {amount_code(amount=hand.bet, compact=True)}"]
    if hand.is_split_hand:
        parts.append("分牌 A" if hand.is_split_aces else "分牌")
    if hand.doubled:
        parts.append("加倍")
    if participant.is_allin and not hand.is_split_hand and not hand.doubled:
        parts.append("all-in")
    return " · ".join(parts)


def _split_hand_header(index: int, total: int) -> str:
    """Returns the heading line announcing one sub-hand of a split player."""
    return f"### 🪓 分牌 · 手 {index + 1} / 共 {total}"


def _insurance_phase_status(player: BlackjackPlayerHand) -> str:
    """Returns the per-player status text shown during the insurance phase."""
    if player.insurance_bet > 0:
        return f"保險 {amount_code(amount=player.insurance_bet, compact=True)}"
    if player.insurance_resolved:
        return "已拒絕保險"
    return "保險待決定"


def _format_player_block(
    player: BlackjackPlayerHand, active_hand_index: int | None, insurance_status: str | None
) -> str:
    """Formats one player's hands and wager metadata for the table embed."""
    lines: list[str] = []
    is_split = len(player.hands) > 1
    for index, hand in enumerate(player.hands):
        is_active = active_hand_index == index
        if is_split:
            lines.append(_split_hand_header(index=index, total=len(player.hands)))
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
    embed.set_footer(text=f"基本下注 {currency_text(amount=requested_bet, compact=True)}")
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


def _dealer_in_progress_color(round_state: BlackjackRound) -> int:
    """Returns the in-progress dealer seat color."""
    if round_state.phase == "insurance":
        return IN_PROGRESS_COLOR
    return PUSH_COLOR


def _player_seat_color(
    player: BlackjackPlayerHand,
    *,
    settlement: BlackjackPlayerSettlement | None,
    is_active: bool,
    insurance_phase: bool,
) -> int:
    """Picks a player seat embed color from settlement or in-progress state."""
    if settlement is not None:
        if settlement.delta > 0:
            return WIN_COLOR
        if settlement.delta < 0:
            return LOSE_COLOR
        return PUSH_COLOR
    if insurance_phase:
        return IN_PROGRESS_COLOR
    if is_active:
        return IN_PROGRESS_COLOR
    return PUSH_COLOR


def _dealer_settlement_color(results: list[BlackjackPlayerResult]) -> int:
    """Picks the dealer seat color from how the casino fared this round.

    The dealer is one-vs-many: lose to a single player and the casino owes a
    payout, so we surface red as soon as any player has a positive delta. All
    losses (no player won anything) means the casino held the line; all
    pushes means a neutral round.
    """
    any_player_won = any(result.settlement.delta > 0 for result in results)
    if any_player_won:
        return LOSE_COLOR
    any_player_lost = any(result.settlement.delta < 0 for result in results)
    if any_player_lost:
        return WIN_COLOR
    return PUSH_COLOR


def build_dealer_seat_embed(  # noqa: PLR0913 -- dealer seat needs round + identity + render flags
    *,
    round_state: BlackjackRound,
    system_name: str,
    system_avatar_url: str,
    hide_hole: bool,
    dealer_steps: list[BlackjackDealerStep] | None = None,
    is_settled: bool = False,
    results: list[BlackjackPlayerResult] | None = None,
) -> Embed:
    """Builds the dealer seat embed shown alongside player seats.

    Pass `hide_hole=False` during the dealer phase or peek-reveal animations so
    the hole card is visible; `dealer_steps` populates the rule-driven action
    log once dealer play starts. When `is_settled=True`, `results` drives the
    color from the casino-vs-table outcome rather than the dealer hand total.
    """
    description_parts: list[str] = [
        _format_dealer_block(round_state=round_state, hide_hole=hide_hole)
    ]
    decision_path = _format_dealer_decision_path(steps=dealer_steps or [])
    if decision_path:
        description_parts.append(metadata_line(text=f"動作: {decision_path}"))
    if not is_settled and not hide_hole:
        # We are still showing the table while the dealer plays.
        description_parts.append(metadata_line(text="莊家正在依規則出牌"))
    elif not is_settled:
        description_parts.append(metadata_line(text="莊家暗牌待揭示"))
    if is_settled:
        color = _dealer_settlement_color(results=results or [])
    else:
        color = _dealer_in_progress_color(round_state=round_state)
    embed = Embed(
        title="♠️ 莊家",
        description="\n".join(part for part in description_parts if part),
        color=color,
    )
    # `system_avatar_url` is intentionally not surfaced as a thumbnail: the bot
    # is now a player at the table, so reusing its avatar for the dealer seat
    # would conflict with the bot's own player seat.
    embed.set_author(name=system_name)
    embed.set_footer(text="莊家規則: <=16 必補, soft 17 補, hard 17+ 停")
    return embed


def _player_seat_status_footer(
    *, round_state: BlackjackRound, is_active: bool, insurance_phase: bool
) -> str:
    """Returns the per-player seat footer."""
    if insurance_phase:
        return "保險決定中"
    if is_active:
        return f"進行中 · 不操作 {BLACKJACK_ACTION_TIMEOUT_SECONDS} 秒會自動 stand"
    if round_state.phase == "settled" or round_state.finished:
        return "已結算"
    return "待輪到"


def _format_settlement_insurance_line(settlement: BlackjackPlayerSettlement) -> str | None:
    """Returns the small-text insurance settlement line, if any."""
    ins = settlement.insurance
    if ins is None:
        return None
    if ins.won:
        return (
            f"保險 {amount_code(amount=ins.bet, compact=True)} → 中獎 "
            f"{amount_code(amount=ins.delta, signed=True, compact=True)}"
        )
    return (
        f"保險 {amount_code(amount=ins.bet, compact=True)} → 莊家無 BJ "
        f"{amount_code(amount=ins.delta, signed=True, compact=True)}"
    )


def build_player_seat_embed(  # noqa: PLR0913, C901 -- seat needs round, player, optional settlement, bot reason
    *,
    player: BlackjackPlayerHand,
    round_state: BlackjackRound,
    active_hand_index: int | None,
    insurance_status: str | None,
    settlement: BlackjackPlayerSettlement | None = None,
    dealer_total: int | None = None,
    bot_reason: str | None = None,
) -> Embed:
    """Builds one player's seat embed.

    Same shape for human and bot players. `bot_reason`, when provided, surfaces
    the AI's last decision reasoning as a `💭` metadata line — humans pass
    `None`.
    """
    is_active = active_hand_index is not None
    insurance_phase = round_state.phase == "insurance"
    color = _player_seat_color(
        player=player, settlement=settlement, is_active=is_active, insurance_phase=insurance_phase
    )
    description_parts: list[str] = []
    bot_reason_line = metadata_line(text=f"💭 {bot_reason}") if bot_reason else ""
    hand_count = len(player.hands)
    # In-progress vs settled hand rendering: settled uses settlement.hands so
    # the result label + outcome surface lines up with the actually-applied
    # delta. In-progress reads hands directly off `player.hands`.
    if settlement is not None:
        for hand_index, hand_settlement in enumerate(settlement.hands):
            if hand_count > 1:
                description_parts.append(_split_hand_header(index=hand_index, total=hand_count))
            summary = _hand_summary_line(cards=hand_settlement.cards)
            title = player_result_title(
                outcome=hand_settlement.outcome,
                player_total=hand_value(cards=hand_settlement.cards),
                dealer_total=dealer_total or 0,
            )
            description_parts.append(f"{summary}\n{title}")
    else:
        for index, hand in enumerate(player.hands):
            this_active = active_hand_index == index
            if hand_count > 1:
                description_parts.append(_split_hand_header(index=index, total=hand_count))
            suffix = _hand_status_suffix(hand=hand, is_active=this_active)
            description_parts.append(_hand_summary_line(cards=hand.cards, suffix=suffix))
            description_parts.append(
                metadata_line(text=_hand_metadata_text(hand=hand, participant=player.participant))
            )
    if insurance_status:
        description_parts.append(metadata_line(text=insurance_status))
    if settlement is not None:
        ins_line = _format_settlement_insurance_line(settlement=settlement)
        if ins_line:
            description_parts.append(metadata_line(text=ins_line))
        description_parts.append(bot_reason_line)
        description_parts.append(
            settlement_metadata(
                delta=settlement.delta,
                new_balance=settlement.new_balance,
                is_allin=player.participant.is_allin,
                base_delta=settlement.base_delta,
                vip_bonus=settlement.vip_bonus,
                five_card_bonus=settlement.five_card_bonus,
            )
        )
        note = blackjack_player_early_finish_note(
            player=player, dealer=round_state.dealer, peeked_blackjack=round_state.peeked_blackjack
        )
        if note:
            description_parts.append(metadata_line(text=note))
    if settlement is None:
        description_parts.append(bot_reason_line)
    embed = Embed(description="\n".join(part for part in description_parts if part), color=color)
    embed.set_author(name=player.participant.display_name)
    if player.participant.avatar_url:
        embed.set_thumbnail(url=player.participant.avatar_url)
    embed.set_footer(
        text=_player_seat_status_footer(
            round_state=round_state, is_active=is_active, insurance_phase=insurance_phase
        )
    )
    return embed


def build_in_progress_embeds(  # noqa: PLR0913 -- table render needs identity, optional dealer steps, and bot context
    *,
    round_state: BlackjackRound,
    system_name: str,
    system_avatar_url: str,
    dealer_steps: list[BlackjackDealerStep] | None = None,
    force_show_hole: bool = False,
    bot_user_id: int | None = None,
    bot_reasons: dict[int, str] | None = None,
) -> list[Embed]:
    """Builds dealer + per-player seat embeds for the in-progress table.

    Pass `force_show_hole=True` for peek-reveal animations to expose the
    dealer hole card before settlement.
    """
    bot_reasons = bot_reasons or {}
    embeds: list[Embed] = [
        build_dealer_seat_embed(
            round_state=round_state,
            system_name=system_name,
            system_avatar_url=system_avatar_url,
            hide_hole=not force_show_hole,
            dealer_steps=dealer_steps,
            is_settled=False,
        )
    ]
    insurance_phase = round_state.phase == "insurance"
    for player_index, player in enumerate(round_state.players):
        active_hand_index: int | None = None
        if (
            round_state.current_player_index == player_index
            and round_state.phase == "player_actions"
        ):
            active_hand_index = round_state.current_hand_index
        insurance_status: str | None = None
        if insurance_phase:
            insurance_status = _insurance_phase_status(player=player)
        elif player.insurance_bet > 0:
            insurance_status = f"保險 {amount_code(amount=player.insurance_bet, compact=True)}"
        reason = (
            bot_reasons.get(player.participant.user_id)
            if player.participant.user_id == bot_user_id
            else None
        )
        embeds.append(
            build_player_seat_embed(
                player=player,
                round_state=round_state,
                active_hand_index=active_hand_index,
                insurance_status=insurance_status,
                bot_reason=reason,
            )
        )
    return embeds


def _table_result_detail(results: list[BlackjackPlayerResult]) -> str:
    """Formats compact per-player settlement details for narrator banter."""
    lines: list[str] = []
    for result in results:
        outcome_label, _color = blackjack_outcome_presentation(outcome=result.settlement.outcome)
        delta = currency_text(amount=result.settlement.delta, signed=True, compact=True)
        lines.append(f"{result.participant.display_name}: {outcome_label} {delta}")
    return "；".join(lines)


def build_final_embeds(  # noqa: PLR0913 -- final render mirrors in-progress signature
    *,
    round_state: BlackjackRound,
    results: list[BlackjackPlayerResult],
    system_name: str = "賭場系統",
    system_avatar_url: str = "",
    dealer_steps: list[BlackjackDealerStep] | None = None,
    bot_user_id: int | None = None,
    bot_reasons: dict[int, str] | None = None,
) -> list[Embed]:
    """Builds dealer + per-player seat embeds for the settled table."""
    bot_reasons = bot_reasons or {}
    dealer_total = round_state.dealer_total()
    embeds: list[Embed] = [
        build_dealer_seat_embed(
            round_state=round_state,
            system_name=system_name,
            system_avatar_url=system_avatar_url,
            hide_hole=False,
            dealer_steps=dealer_steps,
            is_settled=True,
            results=results,
        )
    ]
    results_by_user: dict[int, BlackjackPlayerResult] = {
        result.participant.user_id: result for result in results
    }
    for player in round_state.players:
        result = results_by_user.get(player.participant.user_id)
        if result is None:
            logfire.error(
                "Blackjack player has no settlement result at final embed build",
                user_id=player.participant.user_id,
            )
        settlement = result.settlement if result is not None else None
        reason = (
            bot_reasons.get(player.participant.user_id)
            if player.participant.user_id == bot_user_id
            else None
        )
        embeds.append(
            build_player_seat_embed(
                player=player,
                round_state=round_state,
                active_hand_index=None,
                insurance_status=None,
                settlement=settlement,
                dealer_total=dealer_total,
                bot_reason=reason,
            )
        )
    return embeds


class BlackjackLobbyView(BaseGameLobbyView):
    """Join / leave / start lobby for a Blackjack game session."""

    max_players = MAX_BLACKJACK_PLAYERS

    def __init__(  # noqa: PLR0913 -- lobby owns all table dependencies
        self,
        owner: GameParticipant,
        requested_bet: int,
        rng: Random,
        narrator: SystemNarrator,
        system_name: str,
        system_avatar_url: str,
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
        bot_player_ai: BotPlayerAI | None = None,
        bot_user_id: int | None = None,
        extra_initial_participants: list[GameParticipant] | None = None,
    ) -> None:
        """Initializes a Blackjack lobby with wager and system identity."""
        super().__init__(
            owner=owner,
            rng=rng,
            narrator=narrator,
            system_name=system_name,
            system_avatar_url=system_avatar_url,
            prepare_participant=prepare_participant,
            refresh_participants=refresh_participants,
            timeout=BLACKJACK_ACTION_TIMEOUT_SECONDS,
            extra_initial_participants=extra_initial_participants,
        )
        self.requested_bet = requested_bet
        self.bot_player_ai = bot_player_ai
        self.bot_user_id = bot_user_id

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
        system_line = await self.narrator.taunt_bet(
            player_name=f"{len(self.participants)} 位玩家",
            balance_at_start=table_balance,
            bet=table_bet,
            game="blackjack",
        )
        view = BlackjackView(
            narrator=self.narrator,
            round_state=round_state,
            starter_id=self.owner.user_id,
            author_name=self.owner.account_name,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
            system_line=system_line,
            bot_player_ai=self.bot_player_ai,
            bot_user_id=self.bot_user_id,
        )
        view.message = message
        if round_state.finished:
            await view.finalize(message=message)
            return True
        view.sync_buttons()
        talk_embed = build_system_talk_embed(
            system_line=system_line,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
        )
        seat_embeds = build_in_progress_embeds(
            round_state=round_state,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
            bot_user_id=self.bot_user_id,
            bot_reasons={},
        )
        embeds = [talk_embed, *seat_embeds]
        await edit_message_with_retry(
            message=message,
            kwargs_factory=lambda: _blackjack_table_edit_kwargs(
                embeds=embeds, view=view, target=message
            ),
        )
        await view.maybe_play_bot_turn(message=message)
        return True


class BlackjackView(View):
    """Hit / Stand / Double / Split / Surrender / Insurance controls."""

    def __init__(  # noqa: PLR0913 -- view needs table, narrator, and bot context
        self,
        narrator: SystemNarrator,
        round_state: BlackjackRound,
        starter_id: int,
        author_name: str,
        system_name: str = "賭場系統",
        system_avatar_url: str = "",
        system_line: str = "賭場已收到下注, 牌桌即將發牌",
        bot_player_ai: BotPlayerAI | None = None,
        bot_user_id: int | None = None,
    ) -> None:
        """Initializes the active Blackjack table view."""
        super().__init__(timeout=BLACKJACK_ACTION_TIMEOUT_SECONDS)
        self.narrator = narrator
        self.round_state = round_state
        self.starter_id = starter_id
        self.author_name = author_name
        self.system_name = system_name
        self.system_avatar_url = system_avatar_url
        self.bot_player_ai = bot_player_ai
        self.bot_user_id = bot_user_id
        self.message: Message | None = None
        self._round_lock = asyncio.Lock()
        self._settled = False
        self._system_line = system_line
        self.round_state.auto_play_dealer = False
        self._dealer_steps: list[BlackjackDealerStep] = []
        self._peek_animated = False
        self._state_revision = 0
        self._background_tasks: set[asyncio.Task[None]] = set()
        # Last LLM-decided reason per user_id (bot only). Surfaced in the seat
        # embed so observers can see *why* the bot picked an action / insurance.
        self._bot_reasons: dict[int, str] = {}
        # The bot wallet is stable until settlement, so its finance snapshot is
        # built once per round and reused across the bot's turns.
        self._bot_finance_context: BotFinancialContext | None = None
        self._action_buttons: dict[str, Button] = {
            "bj:hit": cast("Button", self.hit),
            "bj:stand": cast("Button", self.stand),
            "bj:double": cast("Button", self.double),
            "bj:split": cast("Button", self.split),
            "bj:surrender": cast("Button", self.surrender),
        }
        self._insurance_buttons: tuple[Button, Button] = (
            cast("Button", self.insure_yes),
            cast("Button", self.insure_no),
        )
        self.sync_buttons()

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
            hand_before_hit = self.round_state.active_hand()
            try:
                self.round_state.hit(user_id=interaction.user.id)
            except ValueError:
                await self._reject_stale_action_locked(
                    interaction=interaction, message=interaction.message
                )
                return
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            active_after_hit = self.round_state.active_player()
            active_hand = self.round_state.active_hand()
            hint_context: HintRefreshContext | None = None
            if (
                active_after_hit is not None
                and active_after_hit.participant.user_id == interaction.user.id
                and active_hand is not None
                and active_hand is hand_before_hit
            ):
                hint_context = (
                    self._state_revision,
                    active_after_hit.participant.display_name,
                    active_hand.total(),
                    self.round_state.dealer_visible_value(),
                )
            await self._edit_in_progress_locked(message=interaction.message)
            await self._maybe_play_bot_turn_locked(message=interaction.message)
            if hint_context is not None:
                self._track_background_task(
                    self._refresh_hint_later(
                        message=interaction.message, hint_context=hint_context
                    )
                )

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
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._edit_in_progress_locked(message=interaction.message)
            await self._maybe_play_bot_turn_locked(message=interaction.message)

    @nextcord.ui.button(
        label="加倍", emoji="💰", style=ButtonStyle.success, custom_id="bj:double", row=1
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
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._edit_in_progress_locked(message=interaction.message)
            await self._maybe_play_bot_turn_locked(message=interaction.message)

    @nextcord.ui.button(
        label="分牌", emoji="🪓", style=ButtonStyle.success, custom_id="bj:split", row=1
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
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._edit_in_progress_locked(message=interaction.message)
            await self._maybe_play_bot_turn_locked(message=interaction.message)

    @nextcord.ui.button(
        label="投降", emoji="🏳️", style=ButtonStyle.danger, custom_id="bj:surrender", row=1
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
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._edit_in_progress_locked(message=interaction.message)
            await self._maybe_play_bot_turn_locked(message=interaction.message)

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
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._maybe_animate_insurance_close_locked(message=interaction.message)
            await self._edit_in_progress_locked(message=interaction.message)
            await self._maybe_play_bot_turn_locked(message=interaction.message)

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
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message)
                return
            await self._maybe_animate_insurance_close_locked(message=interaction.message)
            await self._edit_in_progress_locked(message=interaction.message)
            await self._maybe_play_bot_turn_locked(message=interaction.message)

    async def finalize(self, message: Message) -> None:
        """Settles every player exactly once."""
        async with self._round_lock:
            await self._finalize_locked(message=message)

    async def maybe_play_bot_turn(self, message: Message) -> None:
        """Public entry point that drives the bot's turn(s) under the round lock."""
        async with self._round_lock:
            await self._maybe_play_bot_turn_locked(message=message)

    async def _maybe_play_bot_turn_locked(self, message: Message) -> None:
        """Plays consecutive bot moves until the active seat is non-bot or finished."""
        if self.bot_player_ai is None or self.bot_user_id is None:
            return
        bot_user_id = self.bot_user_id
        bot_ai = self.bot_player_ai
        steps = 0
        while not self._settled and not self.round_state.finished:
            if self._bot_turn_step_limit_reached(steps=steps, bot_user_id=bot_user_id):
                return
            before_revision = self._state_revision
            if self.round_state.phase == "insurance":
                bot_player = self._find_player_by_user_id(user_id=bot_user_id)
                if (
                    bot_player is None
                    or bot_player.insurance_resolved
                    or not self.round_state.insurance_offered
                ):
                    return
                await self._dispatch_bot_insurance_locked(
                    message=message, bot_player=bot_player, bot_ai=bot_ai
                )
                steps += 1
                if self._bot_turn_dispatch_stalled(
                    before_revision=before_revision,
                    bot_user_id=bot_user_id,
                    action_label="insurance",
                ):
                    return
                await self._pace_next_bot_turn_if_pending(bot_user_id=bot_user_id)
                continue
            active = self.round_state.active_player()
            if active is None or active.participant.user_id != bot_user_id:
                return
            await self._dispatch_bot_action_locked(message=message, active=active, bot_ai=bot_ai)
            steps += 1
            if self._bot_turn_dispatch_stalled(
                before_revision=before_revision, bot_user_id=bot_user_id, action_label="action"
            ):
                return
            await self._pace_next_bot_turn_if_pending(bot_user_id=bot_user_id)

    def _bot_turn_step_limit_reached(self, *, steps: int, bot_user_id: int) -> bool:
        """Returns whether the bot loop exceeded its safety step limit."""
        if steps < MAX_BOT_TURN_STEPS:
            return False
        logfire.error(
            "Bot turn loop exceeded step limit; breaking to prevent hang",
            bot_user_id=bot_user_id,
            state_revision=self._state_revision,
        )
        return True

    def _bot_turn_dispatch_stalled(
        self, *, before_revision: int, bot_user_id: int, action_label: str
    ) -> bool:
        """Returns whether a bot dispatch failed to advance round state."""
        if self._state_revision != before_revision:
            return False
        logfire.error(
            "Bot {action_label} dispatch did not advance state; breaking",
            action_label=action_label,
            bot_user_id=bot_user_id,
            state_revision=self._state_revision,
        )
        return True

    async def _pace_next_bot_turn_if_pending(self, *, bot_user_id: int) -> None:
        """Waits briefly before another immediate bot-owned table decision."""
        if self._bot_turn_pending(bot_user_id=bot_user_id):
            await asyncio.sleep(delay=BOT_TURN_EDIT_DELAY_SECONDS)

    def _bot_turn_pending(self, *, bot_user_id: int) -> bool:
        """Returns whether the bot still owns the next immediate table decision."""
        if self._settled or self.round_state.finished:
            return False
        if self.round_state.phase == "insurance":
            bot_player = self._find_player_by_user_id(user_id=bot_user_id)
            return (
                bot_player is not None
                and not bot_player.insurance_resolved
                and self.round_state.insurance_offered
            )
        if self.round_state.phase != "player_actions":
            return False
        active = self.round_state.active_player()
        return active is not None and active.participant.user_id == bot_user_id

    def _find_player_by_user_id(self, *, user_id: int) -> BlackjackPlayerHand | None:
        """Returns the player hand container matching a user_id, if any."""
        for candidate in self.round_state.players:
            if candidate.participant.user_id == user_id:
                return candidate
        return None

    async def _build_bot_finance_context(self, *, user_id: int) -> BotFinancialContext:
        """Snapshots the bot's wallet + daily casino counters for AI context.

        Cached for the round: the bot's wallet only changes at settlement, so
        the snapshot is reused across the bot's turns instead of re-reading the
        economy DB on every action.
        """
        if self._bot_finance_context is not None:
            return self._bot_finance_context
        account, daily_stats = await asyncio.gather(
            get_account(user_id=user_id), get_casino_daily_stats(user_id=user_id)
        )
        self._bot_finance_context = BotFinancialContext(
            balance=account.balance if account is not None else 0,
            total_earned=account.total_earned if account is not None else 0,
            total_spent=account.total_spent if account is not None else 0,
            daily_loss=daily_stats.daily_loss,
            daily_win=daily_stats.daily_win,
            daily_net=daily_stats.daily_net,
        )
        return self._bot_finance_context

    def _build_other_players_views(self, *, exclude_user_id: int) -> list[OtherPlayerView]:
        """Returns visible state for every non-bot player at the table."""
        views: list[OtherPlayerView] = []
        for candidate in self.round_state.players:
            if candidate.participant.user_id == exclude_user_id:
                continue
            hand_reprs: list[str] = []
            for hand in candidate.hands:
                if hand.cards:
                    hand_reprs.append(f"{render_hand(cards=hand.cards)} = {hand.total()}")
            is_finished = (
                all(hand.finished for hand in candidate.hands) if candidate.hands else False
            )
            views.append(
                OtherPlayerView(
                    display_name=candidate.participant.display_name,
                    bet=candidate.participant.bet,
                    hands=hand_reprs,
                    is_finished=is_finished,
                )
            )
        return views

    def _own_other_hand_reprs(self, *, bot_player: BlackjackPlayerHand) -> list[str]:
        """Returns labels for the bot's non-active split hands."""
        active_hand = self.round_state.active_hand()
        labels: list[str] = []
        for hand in bot_player.hands:
            if hand is active_hand:
                continue
            if not hand.cards:
                continue
            status = "已停" if hand.finished else "進行中"
            labels.append(f"{render_hand(cards=hand.cards)} = {hand.total()} ({status})")
        return labels

    async def _dispatch_bot_insurance_locked(
        self, *, message: Message, bot_player: BlackjackPlayerHand, bot_ai: BotPlayerAI
    ) -> None:
        """Asks the bot AI whether to take insurance, then applies the decision."""
        first_hand = bot_player.hands[0] if bot_player.hands else None
        if first_hand is None:
            return
        dealer_up = dealer_up_card(dealer=self.round_state.dealer)
        finance = await self._build_bot_finance_context(user_id=bot_player.participant.user_id)
        other_players = self._build_other_players_views(
            exclude_user_id=bot_player.participant.user_id
        )
        insurance_context = build_bot_insurance_context(
            dealer_cards=list(self.round_state.dealer),
            dealer_up=dealer_up,
            shoe=list(self.round_state.shoe),
            insurance_cost=bot_player.participant.bet // 2,
        )
        decision = await bot_ai.decide_bot_insurance(
            dealer_cards=list(self.round_state.dealer),
            dealer_up=dealer_up,
            hand_repr=f"{render_hand(cards=first_hand.cards)} = {first_hand.total()}",
            bet=bot_player.participant.bet,
            finance=finance,
            other_players=other_players,
            insurance_context=insurance_context,
        )
        self._bot_reasons[bot_player.participant.user_id] = (
            f"{'買保險' if decision.take_insurance else '不買保險'}: {decision.reason}"
        )
        try:
            if decision.take_insurance:
                self.round_state.take_insurance(
                    user_id=bot_player.participant.user_id, amount=bot_player.participant.bet // 2
                )
            else:
                self.round_state.decline_insurance(user_id=bot_player.participant.user_id)
        except ValueError:
            logfire.warn(
                "Bot insurance action rejected; declining as fallback",
                user_id=bot_player.participant.user_id,
            )
            with contextlib.suppress(ValueError):
                self.round_state.decline_insurance(user_id=bot_player.participant.user_id)
        self._state_revision += 1
        if self.round_state.finished:
            await self._finalize_locked(message=message)
            return
        await self._maybe_animate_insurance_close_locked(message=message)
        await self._edit_in_progress_locked(message=message)

    async def _dispatch_bot_action_locked(
        self, *, message: Message, active: BlackjackPlayerHand, bot_ai: BotPlayerAI
    ) -> None:
        """Asks the bot AI for an action on its active hand, then applies it."""
        hand = self.round_state.active_hand()
        if hand is None:
            return
        balance_remaining = active.participant.balance_at_start - committed_wagers(player=active)
        allowed: list[BotAction] = []
        if not hand.finished and not hand.is_split_aces:
            allowed.append("hit")
            allowed.append("stand")
        if can_double(hand=hand, balance_remaining=balance_remaining):
            allowed.append("double")
        if can_split(hand=hand, balance_remaining=balance_remaining):
            allowed.append("split")
        if can_surrender(hand=hand, peeked_blackjack=self.round_state.peeked_blackjack):
            allowed.append("surrender")
        if not allowed:
            with contextlib.suppress(ValueError):
                self.round_state.stand(user_id=active.participant.user_id)
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=message)
            else:
                await self._edit_in_progress_locked(message=message)
            return
        dealer_up = dealer_up_card(dealer=self.round_state.dealer)
        finance = await self._build_bot_finance_context(user_id=active.participant.user_id)
        other_players = self._build_other_players_views(exclude_user_id=active.participant.user_id)
        own_other_hands = self._own_other_hand_reprs(bot_player=active)
        is_pair_hand = len(hand.cards) == 2 and not hand.is_split_hand and "split" in allowed
        action_context = build_bot_action_context(
            hand_cards=list(hand.cards),
            dealer_cards=list(self.round_state.dealer),
            dealer_up=dealer_up,
            shoe=list(self.round_state.shoe),
            allowed_actions=tuple(allowed),
            is_pair_hand=is_pair_hand,
            bet=hand.bet,
            balance_remaining=balance_remaining,
            doubled=hand.doubled,
        )
        decision = await bot_ai.decide_bot_action(
            hand_cards=hand.cards,
            hand_total=hand.total(),
            hand_repr=render_hand(cards=hand.cards),
            dealer_up=dealer_up,
            is_pair_hand=is_pair_hand,
            allowed_actions=tuple(allowed),
            bet=hand.bet,
            balance_remaining=balance_remaining,
            finance=finance,
            other_players=other_players,
            own_other_hands=own_other_hands,
            action_context=action_context,
        )
        self._bot_reasons[active.participant.user_id] = f"{decision.action}: {decision.reason}"
        applied = self._apply_bot_action(
            user_id=active.participant.user_id, action=decision.action, allowed=tuple(allowed)
        )
        if not applied:
            with contextlib.suppress(ValueError):
                self.round_state.stand(user_id=active.participant.user_id)
        self._state_revision += 1
        if self.round_state.finished:
            await self._finalize_locked(message=message)
            return
        await self._edit_in_progress_locked(message=message)

    def _apply_bot_action(
        self, *, user_id: int, action: BotAction, allowed: tuple[BotAction, ...]
    ) -> bool:
        """Routes the bot AI decision through the BlackjackRound API, returning success."""
        if action not in allowed:
            return False
        try:
            if action == "hit":
                self.round_state.hit(user_id=user_id)
            elif action == "stand":
                self.round_state.stand(user_id=user_id)
            elif action == "double":
                self.round_state.double_down(user_id=user_id)
            elif action == "split":
                self.round_state.split(user_id=user_id)
            elif action == "surrender":
                self.round_state.surrender(user_id=user_id)
        except ValueError:
            logfire.warn(
                "Bot action raised on BlackjackRound; falling back", user_id=user_id, action=action
            )
            return False
        return True

    def sync_buttons(self) -> None:
        """Shows only the controls that are currently actionable."""
        for button in self._action_buttons.values():
            set_view_item_visible(view=self, item=button, visible=False)
        for button in self._insurance_buttons:
            set_view_item_visible(view=self, item=button, visible=False)

        if self._settled or self.round_state.finished:
            return
        if self.round_state.phase == "insurance":
            for button in self._insurance_buttons:
                button.disabled = False
                set_view_item_visible(view=self, item=button, visible=True)
            return
        if self.round_state.phase != "player_actions":
            return

        active_player = self.round_state.active_player()
        active_hand = self.round_state.active_hand()
        if active_player is None or active_hand is None:
            return

        balance_remaining = active_player.participant.balance_at_start - committed_wagers(
            player=active_player
        )
        visible: dict[str, bool] = {
            "bj:hit": not active_hand.finished and not active_hand.is_split_aces,
            "bj:stand": not active_hand.finished and not active_hand.is_split_aces,
            "bj:double": can_double(hand=active_hand, balance_remaining=balance_remaining),
            "bj:split": can_split(hand=active_hand, balance_remaining=balance_remaining),
            "bj:surrender": can_surrender(
                hand=active_hand, peeked_blackjack=self.round_state.peeked_blackjack
            ),
        }
        for custom_id, button in self._action_buttons.items():
            button.disabled = False
            set_view_item_visible(view=self, item=button, visible=visible[custom_id])

    async def _edit_in_progress_locked(self, message: Message) -> None:
        """Refreshes system narrator and per-seat embeds while holding the round lock."""
        self.sync_buttons()
        talk_embed = build_system_talk_embed(
            system_line=self._system_line,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
        )
        seat_embeds = build_in_progress_embeds(
            round_state=self.round_state,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
            dealer_steps=self._dealer_steps,
            bot_user_id=self.bot_user_id,
            bot_reasons=self._bot_reasons,
        )
        await message.edit(
            **_blackjack_table_edit_kwargs(
                embeds=[talk_embed, *seat_embeds], view=self, target=message
            )
        )

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

        The visible controls are disabled and stopped before settlement work,
        then the final table is sent with a deterministic dealer line. LLM
        settlement banter runs as a background refresh so players see the
        result without waiting on model latency.
        """
        if self._settled:
            return
        self._settled = True
        self._state_revision += 1
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
            )
            results.append(BlackjackPlayerResult(participant=participant, settlement=settlement))
        logfire.info("Blackjack settlement done", results=len(results))

        system_line = self._fallback_settlement_line(results=results)
        talk_embed = build_system_talk_embed(
            system_line=system_line,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
        )
        seat_embeds = build_final_embeds(
            round_state=self.round_state,
            results=results,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
            dealer_steps=self._dealer_steps,
            bot_user_id=self.bot_user_id,
            bot_reasons=self._bot_reasons,
        )
        self.clear_items()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                message.edit(
                    **_blackjack_table_edit_kwargs(
                        embeds=[talk_embed, *seat_embeds], view=None, target=message
                    )
                ),
                timeout=FINAL_EDIT_TIMEOUT_SECONDS,
            )
        logfire.info("Blackjack final edit done")
        self._track_background_task(
            self._refresh_settlement_line_later(
                message=message, results=results, seat_embeds=seat_embeds
            )
        )
        schedule_public_message_delete(message=message, user_name=self.author_name)

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
        intro_talk = build_system_talk_embed(
            system_line=intro_line,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
        )
        body_hidden = build_in_progress_embeds(
            round_state=self.round_state,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
            dealer_steps=self._dealer_steps,
            bot_user_id=self.bot_user_id,
            bot_reasons=self._bot_reasons,
        )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                message.edit(
                    **_blackjack_table_edit_kwargs(
                        embeds=[intro_talk, *body_hidden], view=self, target=message
                    )
                ),
                timeout=FINAL_EDIT_TIMEOUT_SECONDS,
            )
        await asyncio.sleep(PEEK_REVEAL_DELAY_SECONDS)

        reveal_talk = build_system_talk_embed(
            system_line=reveal_line,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
        )
        reveal_body = build_in_progress_embeds(
            round_state=self.round_state,
            system_name=self.system_name,
            system_avatar_url=self.system_avatar_url,
            dealer_steps=self._dealer_steps,
            force_show_hole=True,
            bot_user_id=self.bot_user_id,
            bot_reasons=self._bot_reasons,
        )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                message.edit(
                    **_blackjack_table_edit_kwargs(
                        embeds=[reveal_talk, *reveal_body], view=self, target=message
                    )
                ),
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

    async def _play_dealer_locked(self) -> None:
        """Runs the dealer phase using H17 rules (no AI involved)."""
        if self.round_state.dealer_played or not self.round_state.needs_dealer_play():
            return

        for _step_index in range(MAX_DEALER_DECISION_STEPS):
            total_before = self.round_state.dealer_total()
            if total_before > 21:
                self.round_state.mark_dealer_played()
                return
            soft_17 = self.round_state.dealer_is_soft_17()
            should_hit = total_before < 17 or (total_before == 17 and soft_17)
            if not should_hit:
                self._dealer_steps.append(
                    BlackjackDealerStep(
                        total_before=total_before,
                        action="stand",
                        reason="規則: 已達 17 點且非 soft 17",
                        source="auto",
                        forced=True,
                    )
                )
                self.round_state.mark_dealer_played()
                return
            reason = "規則: soft 17 追牌" if soft_17 else "規則: 未滿 17 點"
            drawn_card = self.round_state.draw_dealer_card()
            total_after = self.round_state.dealer_total()
            self._dealer_steps.append(
                BlackjackDealerStep(
                    total_before=total_before,
                    action="hit",
                    reason=reason,
                    source="auto",
                    drawn_card=drawn_card,
                    total_after=total_after,
                    forced=True,
                )
            )
        logfire.warn(
            "Dealer Blackjack play loop reached maximum steps; forcing stand",
            max_steps=MAX_DEALER_DECISION_STEPS,
        )
        self._dealer_steps.append(
            BlackjackDealerStep(
                total_before=self.round_state.dealer_total(),
                action="stand",
                reason="guard: decision limit",
                source="guard",
                forced=True,
            )
        )
        self.round_state.mark_dealer_played()

    def _fallback_settlement_line(self, results: list[BlackjackPlayerResult]) -> str:
        """Returns the immediate non-LLM narrator line for a final table."""
        if len(results) == 1:
            return BLACKJACK_SETTLEMENT_FALLBACK_LINES[results[0].settlement.outcome]
        net_delta = sum(result.settlement.delta for result in results)
        if net_delta > 0:
            return "本桌整體玩家略勝, 賭場結算後支付差額"
        if net_delta < 0:
            return "本桌整體玩家未過關, 籌碼流向賭場"
        return "本桌全部結算後雙方持平"

    async def _refresh_hint_later(
        self, *, message: Message, hint_context: HintRefreshContext
    ) -> None:
        """Refreshes the in-progress narrator line if the table did not advance."""
        revision, player_name, player_total, dealer_visible = hint_context
        try:
            system_line = await self.narrator.hint(
                player_name=player_name, player_total=player_total, dealer_visible=dealer_visible
            )
        except Exception:
            logfire.warn("Blackjack narrator hint refresh failed", _exc_info=True)
            return
        async with self._round_lock:
            if self._settled or self._state_revision != revision:
                return
            self._system_line = system_line
            await self._edit_in_progress_locked(message=message)

    async def _refresh_settlement_line_later(
        self, *, message: Message, results: list[BlackjackPlayerResult], seat_embeds: list[Embed]
    ) -> None:
        """Refreshes the final narrator line with LLM banter after results are visible."""
        try:
            system_line = await self._settlement_line(results=results)
        except Exception:
            logfire.warn("Blackjack settlement banter refresh failed", _exc_info=True)
            return
        async with self._round_lock:
            if not self._settled:
                return
            talk_embed = build_system_talk_embed(
                system_line=system_line,
                system_name=self.system_name,
                system_avatar_url=self.system_avatar_url,
            )
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    message.edit(
                        **_blackjack_table_edit_kwargs(
                            embeds=[talk_embed, *seat_embeds], view=None, target=message
                        )
                    ),
                    timeout=FINAL_EDIT_TIMEOUT_SECONDS,
                )

    async def _settlement_line(self, results: list[BlackjackPlayerResult]) -> str:
        """Builds single-player or table-level narrator settlement broadcast."""
        if len(results) == 1:
            result = results[0]
            return await self.narrator.settle(
                player_name=result.participant.display_name,
                outcome=result.settlement.outcome,
                bet=result.participant.bet,
                delta=result.settlement.delta,
                new_balance=result.settlement.new_balance,
                game="blackjack",
                detail=result.settlement.detail,
            )
        return await self.narrator.table_settle(
            table_name="Blackjack table",
            player_count=len(results),
            net_delta=sum(result.settlement.delta for result in results),
            game="blackjack",
            detail=_table_result_detail(results=results),
        )

    def _track_background_task(self, coroutine: Coroutine[Any, Any, None]) -> None:
        """Tracks a background UI refresh task until it finishes."""
        task = asyncio.create_task(coro=coroutine)
        self._background_tasks.add(task)

        def _discard_task(done_task: asyncio.Task[None]) -> None:
            self._background_tasks.discard(done_task)

        task.add_done_callback(_discard_task)

    async def wait_for_background_tasks(self) -> None:
        """Waits for currently scheduled background UI refreshes."""
        while self._background_tasks:
            await asyncio.gather(*tuple(self._background_tasks))

    def _disable_buttons(self) -> None:
        """Disables every currently visible action / insurance control."""
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
    "build_dealer_seat_embed",
    "build_final_embeds",
    "build_in_progress_embeds",
    "build_player_seat_embed",
]
