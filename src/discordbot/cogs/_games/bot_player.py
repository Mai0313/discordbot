"""AI that decides the bot player's bet, action, and insurance choices.

The bot is a regular Blackjack player; the casino system is the dealer. This
module mirrors `SystemNarrator` shape (BaseModel + AsyncOpenAI client +
ModelSettings) and provides deterministic fallbacks so a slow or failing LLM
never blocks the bot's turn at the table.

Decision-time context is verbose by design: the LLM sees its lifetime balance,
today's casino loss/win/net, this round's bet and remaining wallet, every
other player's hands plus bets, and its own other split hands. The goal is to
let the model reason from the actual table state rather than fall back on a
prescriptive script.
"""

from typing import Final, cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import BaseModel, ConfigDict, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.games import (
    Card,
    BotAction,
    OtherPlayerView,
    BotFinancialContext,
    BotPlayerBetDecision,
    BotPlayerActionDecision,
    BotPlayerInsuranceDecision,
)
from discordbot.typings.models import ModelSettings
from discordbot.cogs._games.prompts import (
    BOT_PLAYER_BET_PROMPT,
    BOT_PLAYER_ACTION_PROMPT,
    BOT_PLAYER_INSURANCE_PROMPT,
)
from discordbot.cogs._economy.presentation import CURRENCY_NAME

BOT_BET_AI_TIMEOUT_SECONDS = 30.0
BOT_ACTION_AI_TIMEOUT_SECONDS = 30.0
BOT_INSURANCE_AI_TIMEOUT_SECONDS = 30.0
# Bot decisions are system-side LLM calls. ASCII labels per method let LiteLLM
# telemetry split bet / action / insurance traffic, mirroring the
# `auto_unmute.py` / `_stock/news.py` / `prompt_dev.py` pattern.
_BET_END_USER_ID: Final[str] = "bot_player_bet"
_ACTION_END_USER_ID: Final[str] = "bot_player_action"
_INSURANCE_END_USER_ID: Final[str] = "bot_player_insurance"


def _dealer_up_value(*, up_card: Card | None) -> int:
    """Returns the Blackjack value of the dealer's up-card (A counts as 11)."""
    if up_card is None:
        return 0
    if up_card.rank == "A":
        return 11
    if up_card.rank in ("J", "Q", "K"):
        return 10
    return int(up_card.rank)


def fallback_bet(*, balance: int, table_bet: int) -> int:
    """Deterministic bet fallback when the LLM is slow or fails.

    Matches the table bet, clamped into [1, balance]. Returns 1 when balance
    is non-positive; callers guard auto-join against that already.
    """
    if balance <= 0:
        return 1
    return max(1, min(balance, table_bet))


def fallback_action(
    *,
    hand_total: int,
    dealer_up: Card | None,
    is_pair_hand: bool,
    allowed_actions: tuple[BotAction, ...],
) -> BotAction:
    """Deterministic basic-strategy fallback that only emits allowed actions."""
    dealer_value = _dealer_up_value(up_card=dealer_up)
    if is_pair_hand and "split" in allowed_actions and dealer_value <= 7:
        return "split"
    if hand_total >= 17 and "stand" in allowed_actions:
        return "stand"
    if hand_total <= 11 and "hit" in allowed_actions:
        return "hit"
    if 12 <= hand_total <= 16 and dealer_value <= 6 and "stand" in allowed_actions:
        return "stand"
    if "hit" in allowed_actions:
        return "hit"
    return allowed_actions[0]


def fallback_insurance() -> bool:
    """Deterministic insurance fallback: never take (negative EV)."""
    return False


def _format_finance_block(finance: BotFinancialContext) -> str:
    """Renders the bot's lifetime + daily financial state as a prompt block."""
    return (
        f"自身財務狀態:\n"
        f"- 目前餘額 ({CURRENCY_NAME}): {finance.balance}\n"
        f"- 終身贏得 ({CURRENCY_NAME}): {finance.total_earned}\n"
        f"- 終身輸掉 ({CURRENCY_NAME}): {finance.total_spent}\n"
        f"- 今日累計贏 ({CURRENCY_NAME}): {finance.daily_win}\n"
        f"- 今日累計輸 ({CURRENCY_NAME}): {finance.daily_loss}\n"
        f"- 今日淨值 ({CURRENCY_NAME}): {finance.daily_net:+d}"
    )


def _format_other_players_block(other_players: list[OtherPlayerView]) -> str:
    """Renders other players' visible table state, or a placeholder when empty."""
    if not other_players:
        return "桌上其他玩家: 無 (只有你和賭場)"
    lines: list[str] = ["桌上其他玩家:"]
    for other in other_players:
        status = "已完成" if other.is_finished else "進行中"
        hands_repr = " | ".join(other.hands) if other.hands else "尚未發牌"
        lines.append(
            f"- {other.display_name} (下注 {other.bet} {CURRENCY_NAME}, {status}): {hands_repr}"
        )
    return "\n".join(lines)


def _format_other_player_bets_block(other_player_bets: list[tuple[str, int]]) -> str:
    """Renders the per-player bet list visible during the bet phase."""
    if not other_player_bets:
        return "桌上其他玩家的下注: 無"
    lines: list[str] = ["桌上其他玩家的下注:"]
    for display_name, bet in other_player_bets:
        lines.append(f"- {display_name}: {bet} {CURRENCY_NAME}")
    return "\n".join(lines)


class BotPlayerAI(BaseModel):
    """Wraps slow-model calls for the bot's player-side decisions.

    Attributes:
        client: The shared AsyncOpenAI client.
        model: Slow-model settings for strategic Blackjack reasoning.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: AsyncOpenAI
    model: ModelSettings

    async def decide_bot_bet(
        self,
        *,
        finance: BotFinancialContext,
        table_bet: int,
        other_player_bets: list[tuple[str, int]],
    ) -> int:
        """Returns the bot's bet for the upcoming round, falling back on error."""
        fallback = fallback_bet(balance=finance.balance, table_bet=table_bet)
        user_text = (
            f"{_format_finance_block(finance=finance)}\n\n"
            f"開桌者的下注 ({CURRENCY_NAME}): {table_bet}\n"
            f"{_format_other_player_bets_block(other_player_bets=other_player_bets)}"
        )
        try:
            async with asyncio.timeout(delay=BOT_BET_AI_TIMEOUT_SECONDS):
                responses = await self.client.responses.parse(
                    model=self.model.name,
                    instructions=BOT_PLAYER_BET_PROMPT,
                    input=cast(
                        "ResponseInputParam",
                        [EasyInputMessageParam(role="user", content=user_text)],
                    ),
                    text_format=BotPlayerBetDecision,
                    reasoning=self.model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": _BET_END_USER_ID},
                    extra_body={"mock_testing_fallbacks": False},
                )
        except TimeoutError:
            logfire.warn(
                "Bot bet decision timed out; using deterministic fallback",
                timeout_seconds=BOT_BET_AI_TIMEOUT_SECONDS,
            )
            return fallback
        except (ValidationError, Exception):
            logfire.warn("Bot bet decision failed; using deterministic fallback", _exc_info=True)
            return fallback
        if responses.output_parsed is None:
            return fallback
        candidate = responses.output_parsed.bet_amount
        return max(1, min(finance.balance, candidate)) if finance.balance > 0 else fallback

    async def decide_bot_action(  # noqa: PLR0913 -- bot action decision needs full table context
        self,
        *,
        hand_total: int,
        hand_repr: str,
        dealer_up: Card | None,
        is_pair_hand: bool,
        allowed_actions: tuple[BotAction, ...],
        bet: int,
        balance_remaining: int,
        finance: BotFinancialContext,
        other_players: list[OtherPlayerView],
        own_other_hands: list[str],
    ) -> BotPlayerActionDecision:
        """Returns the bot's next action with reasoning, falling back to basic strategy."""
        fallback_decision = BotPlayerActionDecision(
            action=fallback_action(
                hand_total=hand_total,
                dealer_up=dealer_up,
                is_pair_hand=is_pair_hand,
                allowed_actions=allowed_actions,
            ),
            reason="基本策略 fallback",
        )
        dealer_label = str(dealer_up) if dealer_up else "未知"
        allowed_text = ", ".join(allowed_actions)
        own_other = (
            "你自己其他分牌手: " + " | ".join(own_other_hands)
            if own_other_hands
            else "你自己其他分牌手: 無"
        )
        user_text = (
            f"{_format_finance_block(finance=finance)}\n\n"
            f"本手下注 ({CURRENCY_NAME}): {bet}\n"
            f"本局尚未投入的剩餘籌碼 ({CURRENCY_NAME}): {balance_remaining}\n\n"
            f"你的當前手牌: {hand_repr}\n"
            f"你的當前手牌總點數: {hand_total}\n"
            f"是否為對子 (可 split): {'是' if is_pair_hand else '否'}\n"
            f"{own_other}\n\n"
            f"莊家明牌: {dealer_label}\n"
            f"{_format_other_players_block(other_players=other_players)}\n\n"
            f"allowed_actions (你只能選其中之一): [{allowed_text}]"
        )
        try:
            async with asyncio.timeout(delay=BOT_ACTION_AI_TIMEOUT_SECONDS):
                responses = await self.client.responses.parse(
                    model=self.model.name,
                    instructions=BOT_PLAYER_ACTION_PROMPT,
                    input=cast(
                        "ResponseInputParam",
                        [EasyInputMessageParam(role="user", content=user_text)],
                    ),
                    text_format=BotPlayerActionDecision,
                    reasoning=self.model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": _ACTION_END_USER_ID},
                    extra_body={"mock_testing_fallbacks": False},
                )
        except TimeoutError:
            logfire.warn(
                "Bot action decision timed out; using basic-strategy fallback",
                timeout_seconds=BOT_ACTION_AI_TIMEOUT_SECONDS,
            )
            return fallback_decision
        except (ValidationError, Exception):
            logfire.warn(
                "Bot action decision failed; using basic-strategy fallback", _exc_info=True
            )
            return fallback_decision
        if responses.output_parsed is None:
            return fallback_decision
        candidate = responses.output_parsed
        if candidate.action in allowed_actions:
            return candidate
        return fallback_decision

    async def decide_bot_insurance(
        self,
        *,
        dealer_up: Card | None,
        hand_repr: str,
        bet: int,
        finance: BotFinancialContext,
        other_players: list[OtherPlayerView],
    ) -> BotPlayerInsuranceDecision:
        """Returns whether the bot takes insurance with reasoning, falling back to False."""
        fallback_decision = BotPlayerInsuranceDecision(
            take_insurance=fallback_insurance(), reason="保險長期 EV 為負, 直接拒絕"
        )
        dealer_label = str(dealer_up) if dealer_up else "未知"
        insurance_cost = bet // 2
        user_text = (
            f"{_format_finance_block(finance=finance)}\n\n"
            f"本手下注 ({CURRENCY_NAME}): {bet}\n"
            f"買保險要再下 ({CURRENCY_NAME}): {insurance_cost} "
            f"(賠率 2:1, 莊家若湊出 Blackjack 賠 {insurance_cost * 2})\n\n"
            f"你的起手牌: {hand_repr}\n"
            f"莊家明牌: {dealer_label}\n"
            f"{_format_other_players_block(other_players=other_players)}"
        )
        try:
            async with asyncio.timeout(delay=BOT_INSURANCE_AI_TIMEOUT_SECONDS):
                responses = await self.client.responses.parse(
                    model=self.model.name,
                    instructions=BOT_PLAYER_INSURANCE_PROMPT,
                    input=cast(
                        "ResponseInputParam",
                        [EasyInputMessageParam(role="user", content=user_text)],
                    ),
                    text_format=BotPlayerInsuranceDecision,
                    reasoning=self.model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": _INSURANCE_END_USER_ID},
                    extra_body={"mock_testing_fallbacks": False},
                )
        except TimeoutError:
            logfire.warn(
                "Bot insurance decision timed out; declining insurance",
                timeout_seconds=BOT_INSURANCE_AI_TIMEOUT_SECONDS,
            )
            return fallback_decision
        except (ValidationError, Exception):
            logfire.warn("Bot insurance decision failed; declining insurance", _exc_info=True)
            return fallback_decision
        if responses.output_parsed is None:
            return fallback_decision
        return responses.output_parsed
