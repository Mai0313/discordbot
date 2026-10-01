"""Blackjack lobby and table views, and the seat embeds they render."""

from __future__ import annotations

from uuid import uuid4
from typing import TYPE_CHECKING, Any, Final, cast
import asyncio
import contextlib

import logfire
import nextcord
from nextcord import Embed, Message, NotFound, Forbidden, ButtonStyle, Interaction

from discordbot.typings.games import (
    Card,
    BotAction,
    GameParticipant,
    BlackjackDealerStep,
    BlackjackPlayerResult,
    BlackjackPlayerSettlement,
)
from discordbot.cogs.games.lobby import BaseGameLobbyView, PrepareParticipant, RefreshParticipants
from discordbot.typings.timeouts import GAME_FINAL_EDIT_TIMEOUT_SECONDS
from discordbot.cogs.games.database import record_blackjack_history
from discordbot.utils.asyncio_locks import spawn_tracked
from discordbot.cogs.games.blackjack import (
    BlackjackRound,
    BlackjackHandState,
    BlackjackPlayerHand,
    InsuranceBetTooSmallError,
    InsuranceBeyondBalanceError,
    hand_value,
    is_five_card_win,
    is_five_card_twenty_one,
)
from discordbot.cogs.games.bot_player import choose_bot_action, bot_takes_insurance
from discordbot.cogs.games.settlement import settle_blackjack_player
from discordbot.cogs.games.interactions import (
    GameView,
    edit_game_message,
    table_edit_kwargs,
    publish_final_table,
    set_view_item_visible,
)
from discordbot.cogs.games.presentation import (
    WIN_COLOR,
    LOSE_COLOR,
    PUSH_COLOR,
    WIN_RESULT_EMOJI,
    BUST_RESULT_EMOJI,
    IN_PROGRESS_COLOR,
    NATURAL_RESULT_EMOJI,
    SYSTEM_NARRATOR_NAME,
    LOBBY_PLAYERS_FIELD_EMOJI,
    card_line,
    delta_color,
    render_hand,
    metadata_line,
    player_result_title,
    settlement_metadata,
    lobby_participant_line,
    blackjack_player_early_finish_note,
)
from discordbot.services.economy.presentation import amount_code, currency_text

if TYPE_CHECKING:
    from random import Random
    from collections.abc import Callable, Coroutine

    from nextcord.ui import Button
    from nextcord.ext import commands

    from discordbot.cogs.games.shoe import BlackjackShoeStore

MAX_BLACKJACK_PLAYERS: Final[int] = 6
BLACKJACK_ACTION_TIMEOUT_SECONDS: Final[int] = 180
MAX_BOT_TURN_STEPS: Final[int] = 16
PEEK_REVEAL_DELAY_SECONDS: Final[float] = 1.6
BOT_TURN_EDIT_DELAY_SECONDS: Final[float] = 0.4


def _hand_summary_line(cards: list[Card], suffix: str = "") -> str:
    """H1 heading combining the hand and its total, e.g. `# 10♠  5♥ = 15`."""
    if not cards:
        return ""
    return f"{card_line(cards_text=render_hand(cards=cards))} = {hand_value(cards=cards)}{suffix}"


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
    """Formats the dealer's decision steps into one compact line."""
    if not steps:
        return ""
    parts: list[str] = []
    for step in steps:
        part = f"規則: {step.total_before} {step.action}"
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


def _insurance_refusal_notice(error: ValueError) -> str:
    """Returns what to tell a seat whose insurance the round would not take.

    Read off the exception's class, never its message: the message is English, the class is what
    the rules layer decided, and only one of these three is worth acting on. Sending the seat to
    refresh a table that will never offer it insurance is the worst of the three to get wrong.
    """
    if isinstance(error, InsuranceBetTooSmallError):
        return "你的下注太小，一半不到 1 點，這局沒有保險可買"
    if isinstance(error, InsuranceBeyondBalanceError):
        return "餘額不足，不能買保險"
    # `InsuranceClosedError` plus anything the rules raise that is not about this seat's money: the
    # table has moved and the newest one is the answer either way.
    return "現在不能買保險，請看最新牌桌"


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
    status: str | None = None,
) -> Embed:
    """Builds the lobby embed shown before a Blackjack table starts."""
    embed = Embed(title="♠️ 二十一點 · 開桌準備", color=PUSH_COLOR)
    if status:
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


def _dealer_in_progress_color(round_state: BlackjackRound) -> int:
    """Returns the in-progress dealer seat color."""
    if round_state.phase == "insurance":
        return IN_PROGRESS_COLOR
    return PUSH_COLOR


def _player_seat_color(
    *, settlement: BlackjackPlayerSettlement | None, is_active: bool, insurance_phase: bool
) -> int:
    """Picks a player seat embed color from settlement or in-progress state."""
    if settlement is not None:
        return delta_color(delta=settlement.delta)
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


def build_dealer_seat_embed(
    *,
    round_state: BlackjackRound,
    hide_hole: bool,
    dealer_steps: list[BlackjackDealerStep] | None = None,
    is_settled: bool = False,
    results: list[BlackjackPlayerResult] | None = None,
) -> Embed:
    """Builds the dealer seat embed shown alongside player seats.

    `dealer_steps` populates the rule-driven action log once dealer play starts.
    """
    description_parts: list[str] = [
        _format_dealer_block(round_state=round_state, hide_hole=hide_hole)
    ]
    decision_path = _format_dealer_decision_path(steps=dealer_steps or [])
    if decision_path:
        description_parts.append(metadata_line(text=f"動作: {decision_path}"))
    if not is_settled and not hide_hole:
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
    # No thumbnail: the bot plays at this table, so its avatar on the dealer seat would
    # collide with its own player seat.
    embed.set_author(name=SYSTEM_NARRATOR_NAME)
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
    if round_state.finished:
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


def build_player_seat_embed(  # noqa: PLR0913, C901 -- seat needs round, player, and optional settlement
    *,
    player: BlackjackPlayerHand,
    round_state: BlackjackRound,
    active_hand_index: int | None,
    insurance_status: str | None,
    settlement: BlackjackPlayerSettlement | None = None,
    dealer_total: int = 0,
) -> Embed:
    """Builds one player's seat embed. Same shape for human and bot players."""
    is_active = active_hand_index is not None
    insurance_phase = round_state.phase == "insurance"
    color = _player_seat_color(
        settlement=settlement, is_active=is_active, insurance_phase=insurance_phase
    )
    description_parts: list[str] = []
    hand_count = len(player.hands)
    # A settled seat renders `settlement.hands` rather than `player.hands` so the
    # result label lines up with the delta that was actually applied.
    if settlement is not None:
        for hand_index, hand_settlement in enumerate(settlement.hands):
            if hand_count > 1:
                description_parts.append(_split_hand_header(index=hand_index, total=hand_count))
            summary = _hand_summary_line(cards=hand_settlement.cards)
            title = player_result_title(
                outcome=hand_settlement.outcome,
                player_total=hand_value(cards=hand_settlement.cards),
                dealer_total=dealer_total,
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
        description_parts.append(
            settlement_metadata(
                delta=settlement.delta,
                new_balance=settlement.new_balance,
                is_allin=player.participant.is_allin,
                vip_bonus=settlement.vip_bonus,
                five_card_bonus=settlement.five_card_bonus,
            )
        )
        note = blackjack_player_early_finish_note(
            player=player, dealer=round_state.dealer, peeked_blackjack=round_state.peeked_blackjack
        )
        if note:
            description_parts.append(metadata_line(text=note))
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


def build_in_progress_embeds(
    *, round_state: BlackjackRound, force_show_hole: bool = False
) -> list[Embed]:
    """Builds dealer + per-player seat embeds for the in-progress table."""
    embeds: list[Embed] = [
        build_dealer_seat_embed(
            round_state=round_state, hide_hole=not force_show_hole, is_settled=False
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
        embeds.append(
            build_player_seat_embed(
                player=player,
                round_state=round_state,
                active_hand_index=active_hand_index,
                insurance_status=insurance_status,
            )
        )
    return embeds


def build_final_embeds(
    *,
    round_state: BlackjackRound,
    results: list[BlackjackPlayerResult],
    dealer_steps: list[BlackjackDealerStep] | None = None,
) -> list[Embed]:
    """Builds dealer + per-player seat embeds for the settled table."""
    dealer_total = round_state.dealer_total()
    embeds: list[Embed] = [
        build_dealer_seat_embed(
            round_state=round_state,
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
        embeds.append(
            build_player_seat_embed(
                player=player,
                round_state=round_state,
                active_hand_index=None,
                insurance_status=None,
                settlement=settlement,
                dealer_total=dealer_total,
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
        prepare_participant: PrepareParticipant,
        refresh_participants: RefreshParticipants,
        bot_user_id: int | None = None,
        extra_initial_participants: list[GameParticipant] | None = None,
        shoe_store: BlackjackShoeStore | None = None,
        channel_id: int = 0,
    ) -> None:
        """Initializes a Blackjack lobby with its table wager."""
        super().__init__(
            owner=owner,
            rng=rng,
            prepare_participant=prepare_participant,
            refresh_participants=refresh_participants,
            timeout=BLACKJACK_ACTION_TIMEOUT_SECONDS,
            extra_initial_participants=extra_initial_participants,
        )
        self.requested_bet = requested_bet
        self.bot_user_id = bot_user_id
        self._shoe_store = shoe_store
        self._channel_id = channel_id

    def _build_lobby_embed(self, status: str) -> Embed:
        """Builds the Blackjack lobby embed from current participants."""
        return build_blackjack_lobby_embed(
            owner=self.owner,
            participants=self.participants,
            requested_bet=self.requested_bet,
            max_players=MAX_BLACKJACK_PLAYERS,
            status=status,
        )

    async def _start_game(self, interaction: Interaction[commands.Bot]) -> bool:
        """Deals the table and replaces the lobby message with the game view."""
        message = interaction.message
        if message is None:
            return False
        shoe: list[Card] | None = None
        shoe_generation = 0
        if self._shoe_store is not None:
            shoe, shoe_generation = self._shoe_store.take_shoe(
                channel_id=self._channel_id, rng=self.rng
            )
        round_state = BlackjackRound.from_participants(
            rng=self.rng, participants=self.participants, shoe=shoe
        )
        round_state.deal_initial()
        view = BlackjackView(
            round_state=round_state,
            owner=self.owner,
            bot_user_id=self.bot_user_id,
            shoe_store=self._shoe_store,
            channel_id=self._channel_id,
            shoe_generation=shoe_generation,
        )
        view.message = message
        view.last_press = interaction
        if round_state.finished:
            await view.finalize(message=message, interaction=interaction)
            return True
        view.sync_buttons()
        seat_embeds = build_in_progress_embeds(round_state=round_state)
        await self._show_table(
            interaction=interaction,
            payload=table_edit_kwargs(embeds=seat_embeds, view=view, target=message),
        )
        await view.maybe_play_bot_turn(message=message, interaction=interaction)
        return True


class BlackjackView(GameView):
    """Hit / Stand / Double / Split / Surrender / Insurance controls."""

    interaction_failure_log = "Blackjack action interaction failed"
    notice_failure_log = "Failed to send Blackjack action notice"

    def __init__(  # noqa: PLR0913 -- view needs table identity and bot/shoe context
        self,
        round_state: BlackjackRound,
        owner: GameParticipant,
        bot_user_id: int | None = None,
        shoe_store: BlackjackShoeStore | None = None,
        channel_id: int = 0,
        shoe_generation: int = 0,
    ) -> None:
        """Initializes the active Blackjack table view."""
        super().__init__(timeout=BLACKJACK_ACTION_TIMEOUT_SECONDS)
        self.round_state = round_state
        self.owner = owner
        self.bot_user_id = bot_user_id
        self._shoe_store = shoe_store
        self._channel_id = channel_id
        self._shoe_generation = shoe_generation
        self.message: Message | None = None
        # The last press that edited the table; the timeout's edits and delete ride its token.
        self.last_press: Interaction[commands.Bot] | None = None
        self._round_lock = asyncio.Lock()
        self._settled = False
        self._peek_animated = False
        self._state_revision = 0
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._action_buttons: dict[BotAction, Button[BlackjackView]] = {
            "hit": cast('Button["BlackjackView"]', self.hit),
            "stand": cast('Button["BlackjackView"]', self.stand),
            "double": cast('Button["BlackjackView"]', self.double),
            "split": cast('Button["BlackjackView"]', self.split),
            "surrender": cast('Button["BlackjackView"]', self.surrender),
        }
        self._insurance_buttons: tuple[Button[BlackjackView], Button[BlackjackView]] = (
            cast('Button["BlackjackView"]', self.insure_yes),
            cast('Button["BlackjackView"]', self.insure_no),
        )
        self.sync_buttons()

    async def interaction_check(self, interaction: Interaction[commands.Bot]) -> bool:  # noqa: PLR0911 -- phase + identity gating naturally fans out into early returns
        """Restricts buttons to the active player (or any undecided insurance player)."""
        if self._settled:
            await self._send_notice(interaction=interaction, content="這局已經結束, 等下一局吧")
            return False
        if interaction.user is None:
            return False
        if self.round_state.phase == "insurance":
            player = self.round_state.find_player(user_id=interaction.user.id)
            if player is None:
                await self._send_notice(interaction=interaction, content="你不在這個牌桌")
                return False
            if player.insurance_resolved:
                await self._send_notice(interaction=interaction, content="你已決定過保險")
                return False
            return True
        active = self.round_state.active_player()
        if active is not None and interaction.user.id == active.participant.user_id:
            return True
        if active is not None:
            await self._send_notice(
                interaction=interaction, content=f"現在輪到 {active.participant.display_name}"
            )
            return False
        await self._send_notice(interaction=interaction, content="這局已經不能操作了")
        return False

    async def on_timeout(self) -> None:
        """Auto-resolves the round when nobody clicked in time."""
        if self.message is None:
            return
        await self.finalize(message=self.message, interaction=self.last_press)

    async def _run_player_action(
        self, *, interaction: Interaction[commands.Bot], apply: Callable[..., object]
    ) -> None:
        """Runs one active-player action under the round lock, then refreshes the table.

        `apply` is the `BlackjackRound` method the pressed button performs; the defer,
        the lock, the stale-action rejection and the re-render around it are the same
        for every action button.
        """
        await interaction.response.defer()
        if interaction.message is None or interaction.user is None:
            return
        async with self._round_lock:
            if self._settled or self.round_state.finished:
                return
            active = self.round_state.active_player()
            if active is None:
                await self._finalize_locked(message=interaction.message, interaction=interaction)
                return
            try:
                apply(user_id=interaction.user.id)
            except ValueError:
                await self._reject_stale_action_locked(
                    interaction=interaction, message=interaction.message
                )
                return
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message, interaction=interaction)
                return
            await self._edit_in_progress_locked(
                message=interaction.message, interaction=interaction
            )
            await self._maybe_play_bot_turn_locked(
                message=interaction.message, interaction=interaction
            )

    @nextcord.ui.button(
        label="再要一張", emoji="🃏", style=ButtonStyle.primary, custom_id="bj:hit", row=0
    )
    async def hit(
        self, _button: Button[BlackjackView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Handles the active player's Hit button."""
        await self._run_player_action(interaction=interaction, apply=self.round_state.hit)

    @nextcord.ui.button(
        label="停手", emoji="✋", style=ButtonStyle.secondary, custom_id="bj:stand", row=0
    )
    async def stand(
        self, _button: Button[BlackjackView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Handles the active player's Stand button."""
        await self._run_player_action(interaction=interaction, apply=self.round_state.stand)

    @nextcord.ui.button(
        label="加倍", emoji="💰", style=ButtonStyle.success, custom_id="bj:double", row=1
    )
    async def double(
        self, _button: Button[BlackjackView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Doubles the active hand's bet and finishes it after one draw."""
        await self._run_player_action(interaction=interaction, apply=self.round_state.double_down)

    @nextcord.ui.button(
        label="分牌", emoji="🪓", style=ButtonStyle.success, custom_id="bj:split", row=1
    )
    async def split(
        self, _button: Button[BlackjackView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Splits the active pair into two sibling sub-hands."""
        await self._run_player_action(interaction=interaction, apply=self.round_state.split)

    @nextcord.ui.button(
        label="投降", emoji="🏳️", style=ButtonStyle.danger, custom_id="bj:surrender", row=1
    )
    async def surrender(
        self, _button: Button[BlackjackView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Surrenders the active hand for a half-bet refund."""
        await self._run_player_action(interaction=interaction, apply=self.round_state.surrender)

    async def _run_insurance_action(
        self,
        *,
        interaction: Interaction[commands.Bot],
        decide: Callable[..., Coroutine[Any, Any, bool]],
    ) -> None:
        """Runs one insurance decision under the round lock, then refreshes the table.

        `decide` applies the pressed button's choice and returns False when the round
        refused it, having already reported that; the defer, the lock, the stale-round
        guards and the re-render around it are the same for both insurance buttons.
        """
        await interaction.response.defer()
        if interaction.message is None:
            return
        async with self._round_lock:
            if self._settled:
                return
            if interaction.user is None:
                return
            decided = await decide(
                interaction=interaction, message=interaction.message, user_id=interaction.user.id
            )
            if not decided:
                return
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=interaction.message, interaction=interaction)
                return
            await self._maybe_animate_insurance_close_locked(
                message=interaction.message, interaction=interaction
            )
            await self._edit_in_progress_locked(
                message=interaction.message, interaction=interaction
            )
            await self._maybe_play_bot_turn_locked(
                message=interaction.message, interaction=interaction
            )

    async def _take_insurance_locked(
        self, *, interaction: Interaction[commands.Bot], message: Message, user_id: int
    ) -> bool:
        """Buys half-bet insurance for one seat; False when the round refused it."""
        if self.round_state.find_player(user_id=user_id) is None:
            return False
        try:
            self.round_state.take_insurance(user_id=user_id)
        except ValueError as error:
            await self._send_notice(
                interaction=interaction, content=_insurance_refusal_notice(error=error)
            )
            await self._edit_in_progress_locked(message=message, interaction=interaction)
            return False
        return True

    async def _decline_insurance_locked(
        self, *, interaction: Interaction[commands.Bot], message: Message, user_id: int
    ) -> bool:
        """Declines insurance for one seat; False when the round refused it."""
        try:
            self.round_state.decline_insurance(user_id=user_id)
        except ValueError:
            await self._edit_in_progress_locked(message=message, interaction=interaction)
            return False
        return True

    @nextcord.ui.button(
        label="保險 ½", emoji="🛡️", style=ButtonStyle.success, custom_id="bj:insure_yes", row=1
    )
    async def insure_yes(
        self, _button: Button[BlackjackView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Takes insurance for the calling player."""
        await self._run_insurance_action(
            interaction=interaction, decide=self._take_insurance_locked
        )

    @nextcord.ui.button(
        label="不保險", emoji="❌", style=ButtonStyle.secondary, custom_id="bj:insure_no", row=1
    )
    async def insure_no(
        self, _button: Button[BlackjackView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Declines insurance for the calling player."""
        await self._run_insurance_action(
            interaction=interaction, decide=self._decline_insurance_locked
        )

    async def finalize(
        self, message: Message, interaction: Interaction[commands.Bot] | None
    ) -> None:
        """Settles every player exactly once."""
        async with self._round_lock:
            await self._finalize_locked(message=message, interaction=interaction)

    async def maybe_play_bot_turn(
        self, message: Message, interaction: Interaction[commands.Bot]
    ) -> None:
        """Public entry point that drives the bot's turn(s) under the round lock."""
        async with self._round_lock:
            await self._maybe_play_bot_turn_locked(message=message, interaction=interaction)

    async def _maybe_play_bot_turn_locked(
        self, message: Message, interaction: Interaction[commands.Bot]
    ) -> None:
        """Plays consecutive bot moves until the bot no longer owns the next table decision.

        The bot moves inside the human press that handed it the turn, so its edits go through
        that press too.
        """
        if self.bot_user_id is None:
            return
        bot_user_id = self.bot_user_id
        steps = 0
        while not self._settled and not self.round_state.finished:
            if steps >= MAX_BOT_TURN_STEPS:
                logfire.error(
                    "Bot turn loop exceeded step limit; breaking to prevent hang",
                    bot_user_id=bot_user_id,
                    state_revision=self._state_revision,
                )
                return
            bot_seat = self._pending_bot_seat(bot_user_id=bot_user_id)
            if bot_seat is None:
                return
            before_revision = self._state_revision
            if self.round_state.phase == "insurance":
                action_label = "insurance"
                await self._dispatch_bot_insurance_locked(
                    message=message, bot_player=bot_seat, interaction=interaction
                )
            else:
                action_label = "action"
                await self._dispatch_bot_action_locked(
                    message=message, active=bot_seat, interaction=interaction
                )
            steps += 1
            if self._state_revision == before_revision:
                logfire.error(
                    "Bot {action_label} dispatch did not advance state; breaking",
                    action_label=action_label,
                    bot_user_id=bot_user_id,
                    state_revision=self._state_revision,
                )
                return
            if self._pending_bot_seat(bot_user_id=bot_user_id) is not None:
                await asyncio.sleep(delay=BOT_TURN_EDIT_DELAY_SECONDS)

    def _pending_bot_seat(self, *, bot_user_id: int) -> BlackjackPlayerHand | None:
        """Returns the bot's seat while the bot owns the next immediate table decision."""
        if self._settled or self.round_state.finished:
            return None
        if self.round_state.phase == "insurance":
            bot_player = self.round_state.find_player(user_id=bot_user_id)
            if bot_player is None or bot_player.insurance_resolved:
                return None
            return bot_player
        if self.round_state.phase != "player_actions":
            return None
        active = self.round_state.active_player()
        if active is None or active.participant.user_id != bot_user_id:
            return None
        return active

    async def _dispatch_bot_insurance_locked(
        self,
        *,
        message: Message,
        bot_player: BlackjackPlayerHand,
        interaction: Interaction[commands.Bot],
    ) -> None:
        """Applies the bot's deterministic count-based insurance decision."""
        user_id = bot_player.participant.user_id
        take_insurance = bot_takes_insurance(shoe=self.round_state.shoe)
        try:
            if take_insurance:
                self.round_state.take_insurance(user_id=user_id)
            else:
                self.round_state.decline_insurance(user_id=user_id)
        except InsuranceBetTooSmallError:
            # Expected on a 1-point bet, and the seat is still open, so the decline cannot fail.
            logfire.info(
                "Bot bet too small to insure; declining",
                user_id=user_id,
                bet=bot_player.participant.bet,
            )
            self.round_state.decline_insurance(user_id=user_id)
        except ValueError as exc:
            logfire.warn(
                "Bot insurance action rejected; declining as fallback",
                user_id=user_id,
                take_insurance=take_insurance,
                _exc_info=exc,
            )
            try:
                self.round_state.decline_insurance(user_id=user_id)
            except ValueError as decline_exc:
                # Bot seat stays unresolved; the loop ends at MAX_BOT_TURN_STEPS or the
                # timeout path closes the phase via decline_insurance_for_all_unresolved.
                logfire.warn(
                    "Bot insurance fallback decline also rejected; seat left unresolved",
                    user_id=user_id,
                    phase=self.round_state.phase,
                    _exc_info=decline_exc,
                )
        self._state_revision += 1
        if self.round_state.finished:
            await self._finalize_locked(message=message, interaction=interaction)
            return
        await self._maybe_animate_insurance_close_locked(message=message, interaction=interaction)
        await self._edit_in_progress_locked(message=message, interaction=interaction)

    async def _dispatch_bot_action_locked(
        self,
        *,
        message: Message,
        active: BlackjackPlayerHand,
        interaction: Interaction[commands.Bot],
    ) -> None:
        """Computes the bot's deterministic action on its active hand, then applies it."""
        hand = self.round_state.active_hand()
        if hand is None:
            return
        allowed = self.round_state.allowed_actions()
        if not allowed:
            with contextlib.suppress(ValueError):
                self.round_state.stand(user_id=active.participant.user_id)
            self._state_revision += 1
            if self.round_state.finished:
                await self._finalize_locked(message=message, interaction=interaction)
            else:
                await self._edit_in_progress_locked(message=message, interaction=interaction)
            return
        chosen_action = choose_bot_action(
            hand_cards=list(hand.cards),
            dealer_cards=list(self.round_state.dealer),
            shoe=list(self.round_state.shoe),
            allowed_actions=allowed,
            bet=hand.bet,
        )
        applied = self._apply_bot_action(
            user_id=active.participant.user_id, action=chosen_action, allowed=allowed
        )
        if not applied:
            with contextlib.suppress(ValueError):
                self.round_state.stand(user_id=active.participant.user_id)
        self._state_revision += 1
        if self.round_state.finished:
            await self._finalize_locked(message=message, interaction=interaction)
            return
        await self._edit_in_progress_locked(message=message, interaction=interaction)

    def _apply_bot_action(
        self, *, user_id: int, action: BotAction, allowed: tuple[BotAction, ...]
    ) -> bool:
        """Routes the bot's chosen action through the BlackjackRound API, returning success."""
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
        except ValueError as exc:
            logfire.warn(
                "Bot action raised on BlackjackRound; falling back",
                user_id=user_id,
                action=action,
                _exc_info=exc,
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

        allowed = self.round_state.allowed_actions()
        for action, button in self._action_buttons.items():
            button.disabled = False
            set_view_item_visible(view=self, item=button, visible=action in allowed)

    async def _edit_in_progress_locked(
        self, message: Message, interaction: Interaction[commands.Bot]
    ) -> None:
        """Refreshes the per-seat embeds while holding the round lock."""
        self.last_press = interaction
        self.sync_buttons()
        seat_embeds = build_in_progress_embeds(round_state=self.round_state)
        await interaction.edit_original_message(
            **table_edit_kwargs(embeds=seat_embeds, view=self, target=message)
        )

    async def _reject_stale_action_locked(
        self, interaction: Interaction[commands.Bot], message: Message
    ) -> None:
        """Sends a private stale-action notice and refreshes the table."""
        await self._send_notice(interaction=interaction, content="這個操作已經失效，請看最新牌桌")
        await self._edit_in_progress_locked(message=message, interaction=interaction)

    async def _finalize_locked(
        self, message: Message, interaction: Interaction[commands.Bot] | None
    ) -> None:
        """Applies settlements and publishes the final table embeds once."""
        if self._settled:
            return
        self._settled = True
        self._state_revision += 1
        if not self.round_state.finished:
            self.round_state.stand_all_remaining()
        self._disable_buttons()
        self.stop()
        await self._safe_edit_view_locked(message=message, interaction=interaction)
        logfire.debug(
            "Blackjack finalize started",
            players=len(self.round_state.players),
            channel_id=self._channel_id,
        )

        if self.round_state.peeked_blackjack and not self._peek_animated:
            self._peek_animated = True
            await self._animate_peek_locked(message=message, interaction=interaction)

        dealer_steps = self.round_state.play_dealer()
        logfire.debug(
            "Blackjack dealer phase done",
            dealer_total=self.round_state.dealer_total(),
            channel_id=self._channel_id,
        )

        if self._shoe_store is not None:
            self._shoe_store.save_shoe(
                channel_id=self._channel_id,
                cards=self.round_state.shoe,
                generation=self._shoe_generation,
            )

        results: list[BlackjackPlayerResult] = []
        for player in self.round_state.players:
            settlement = await settle_blackjack_player(round_state=self.round_state, player=player)
            results.append(
                BlackjackPlayerResult(participant=player.participant, settlement=settlement)
            )
        logfire.debug(
            "Blackjack settlement done", results=len(results), channel_id=self._channel_id
        )
        dealer_cards = list(self.round_state.dealer)
        dealer_total = self.round_state.dealer_total()
        spawn_tracked(
            coro=self._record_history_later(
                message=message,
                results=results,
                dealer_cards=dealer_cards,
                dealer_total=dealer_total,
            ),
            tasks=self._background_tasks,
            name="blackjack-round-history",
        )

        seat_embeds = build_final_embeds(
            round_state=self.round_state, results=results, dealer_steps=dealer_steps
        )
        self.clear_items()
        landed = await publish_final_table(
            message=message,
            embeds=seat_embeds,
            user_name=self.owner.account_name,
            game_name="Blackjack",
            interaction=interaction,
            channel_id=self._channel_id,
            message_id=message.id,
            players=len(self.round_state.players),
        )
        if landed:
            logfire.debug(
                "Blackjack final edit done", channel_id=self._channel_id, message_id=message.id
            )

    async def _safe_edit_locked(
        self,
        message: Message,
        interaction: Interaction[commands.Bot] | None,
        payload: dict[str, Any],
        step: str,
    ) -> None:
        """Edits the table on the way to settling; a failed edit is logged, never raised."""
        try:
            await asyncio.wait_for(
                edit_game_message(message=message, interaction=interaction, payload=payload),
                timeout=GAME_FINAL_EDIT_TIMEOUT_SECONDS,
            )
        except NotFound:
            logfire.info(
                "Blackjack table message gone before an edit",
                step=step,
                channel_id=self._channel_id,
                message_id=message.id,
            )
        except Forbidden:
            # Only an edit with no working press behind it goes through the channel, which
            # the server can shut the bot out of; the ids are the whole finding.
            logfire.warn(
                "Discord refused a Blackjack table edit",
                step=step,
                channel_id=self._channel_id,
                message_id=message.id,
            )
        # Broad on purpose: these edits only show the round moving, so no failure of theirs may
        # stop it from settling.
        except Exception as exc:
            logfire.warn(
                "Blackjack table edit failed",
                step=step,
                channel_id=self._channel_id,
                message_id=message.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    async def _safe_edit_view_locked(
        self, message: Message, interaction: Interaction[commands.Bot] | None
    ) -> None:
        """Refreshes only the view so disabled buttons are visible immediately."""
        await self._safe_edit_locked(
            message=message, interaction=interaction, payload={"view": self}, step="view"
        )

    async def _animate_peek_locked(
        self, message: Message, interaction: Interaction[commands.Bot] | None
    ) -> None:
        """Renders the dealer hole-card peek as a 2-stage reveal.

        Buttons stay disabled throughout so the caller can safely chain finalize /
        further edits after the animation returns.
        """
        self._disable_buttons()
        body_hidden = build_in_progress_embeds(round_state=self.round_state)
        await self._safe_edit_locked(
            message=message,
            interaction=interaction,
            payload=table_edit_kwargs(embeds=body_hidden, view=self, target=message),
            step="peek hidden",
        )
        await asyncio.sleep(PEEK_REVEAL_DELAY_SECONDS)

        reveal_body = build_in_progress_embeds(round_state=self.round_state, force_show_hole=True)
        await self._safe_edit_locked(
            message=message,
            interaction=interaction,
            payload=table_edit_kwargs(embeds=reveal_body, view=self, target=message),
            step="peek reveal",
        )
        await asyncio.sleep(PEEK_REVEAL_DELAY_SECONDS)

    async def _maybe_animate_insurance_close_locked(
        self, message: Message, interaction: Interaction[commands.Bot]
    ) -> None:
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
        await self._animate_peek_locked(message=message, interaction=interaction)

    async def _record_history_later(
        self,
        *,
        message: Message,
        results: list[BlackjackPlayerResult],
        dealer_cards: list[Card],
        dealer_total: int,
    ) -> None:
        """Persists the settled round to the games-history store off the critical path."""
        try:
            await record_blackjack_history(
                round_id=uuid4().hex,
                channel_id=self._channel_id,
                guild_id=message.guild.id if message.guild is not None else 0,
                message_id=message.id,
                bot_user_id=self.bot_user_id,
                results=results,
                dealer_cards=dealer_cards,
                dealer_total=dealer_total,
            )
        # Broad on purpose: history is off the critical path and runs as a background
        # task, so it must never disturb an already-settled round.
        except Exception as exc:
            logfire.warn(
                "Blackjack round history persistence failed",
                channel_id=self._channel_id,
                message_id=message.id,
                players=len(results),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    async def wait_for_background_tasks(self) -> None:
        """Waits for the round's off-critical-path tasks (round-history persistence).

        The round never blocks on them, so this drain exists only for a caller that
        needs those writes to have landed.
        """
        while self._background_tasks:
            await asyncio.gather(*tuple(self._background_tasks))
