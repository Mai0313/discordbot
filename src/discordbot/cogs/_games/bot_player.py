"""AI that decides the bot player's bet, action, and insurance choices.

The bot is a regular Blackjack player; the casino system is the dealer. This
module mirrors `SystemNarrator` shape (BaseModel + AsyncOpenAI client +
ModelSettings) and provides deterministic fallbacks so a slow or failing LLM
never blocks the bot's turn at the table.
"""

from typing import Final, Literal, cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.games import Card
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

BotAction = Literal["hit", "stand", "double", "split", "surrender"]


class BotPlayerBetDecision(BaseModel):
    """Structured bet decision returned by the bot player AI."""

    model_config = ConfigDict(frozen=True)

    bet_amount: int = Field(ge=1)
    reason: str


class BotPlayerActionDecision(BaseModel):
    """Structured hit / stand / double / split / surrender decision."""

    model_config = ConfigDict(frozen=True)

    action: BotAction
    reason: str


class BotPlayerInsuranceDecision(BaseModel):
    """Structured insurance-take / decline decision."""

    model_config = ConfigDict(frozen=True)

    take_insurance: bool
    reason: str


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


class BotPlayerAI(BaseModel):
    """Wraps fast-model calls for the bot's player-side decisions.

    Attributes:
        client: The shared AsyncOpenAI client.
        model: Fast-model settings used for every bot decision.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI]
    model: ModelSettings

    async def decide_bot_bet(self, *, balance: int, table_bet: int) -> int:
        """Returns the bot's bet for the upcoming round, falling back on error."""
        fallback = fallback_bet(balance=balance, table_bet=table_bet)
        user_text = (
            f"目前餘額 ({CURRENCY_NAME}): {balance}\n"
            f"桌上其他玩家的下注 ({CURRENCY_NAME}): {table_bet}"
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
        return max(1, min(balance, candidate)) if balance > 0 else fallback

    async def decide_bot_action(
        self,
        *,
        hand_total: int,
        hand_repr: str,
        dealer_up: Card | None,
        is_pair_hand: bool,
        allowed_actions: tuple[BotAction, ...],
    ) -> BotAction:
        """Returns the bot's next action, falling back to basic strategy on error."""
        fallback = fallback_action(
            hand_total=hand_total,
            dealer_up=dealer_up,
            is_pair_hand=is_pair_hand,
            allowed_actions=allowed_actions,
        )
        dealer_label = str(dealer_up) if dealer_up else "未知"
        allowed_text = ", ".join(allowed_actions)
        user_text = (
            f"玩家手牌: {hand_repr}\n"
            f"玩家手牌總點數: {hand_total}\n"
            f"是否為對子: {'是' if is_pair_hand else '否'}\n"
            f"莊家明牌: {dealer_label}\n"
            f"allowed_actions: [{allowed_text}]"
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
            return fallback
        except (ValidationError, Exception):
            logfire.warn(
                "Bot action decision failed; using basic-strategy fallback", _exc_info=True
            )
            return fallback
        if responses.output_parsed is None:
            return fallback
        candidate = responses.output_parsed.action
        if candidate in allowed_actions:
            return candidate
        return fallback

    async def decide_bot_insurance(self, *, dealer_up: Card | None, hand_repr: str) -> bool:
        """Returns whether the bot takes insurance, falling back to False on error."""
        dealer_label = str(dealer_up) if dealer_up else "未知"
        user_text = f"莊家明牌: {dealer_label}\n玩家手牌: {hand_repr}"
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
            return fallback_insurance()
        except (ValidationError, Exception):
            logfire.warn("Bot insurance decision failed; declining insurance", _exc_info=True)
            return fallback_insurance()
        if responses.output_parsed is None:
            return fallback_insurance()
        return responses.output_parsed.take_insurance
