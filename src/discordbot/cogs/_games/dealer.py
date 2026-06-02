"""Casino system narrator that produces short neutral lines for the games."""

from typing import Final, cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import BaseModel, ConfigDict, SkipValidation
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.games import GameKind, SettleOutcome
from discordbot.typings.models import ModelSettings
from discordbot.typings.fishing import Rarity
from discordbot.cogs._games.prompts import (
    SYSTEM_HINT_PROMPT,
    SYSTEM_SETTLE_PROMPT,
    SYSTEM_TAUNT_BET_PROMPT,
    SYSTEM_FISH_CATCH_PROMPT,
)
from discordbot.cogs._games.presentation import SETTLEMENT_FALLBACK_LINES
from discordbot.cogs._economy.presentation import CURRENCY_NAME

NARRATOR_AI_TIMEOUT_SECONDS = 5.0
# System-side LLM calls. ASCII labels per method let LiteLLM telemetry split
# bet / settle / table_settle / hint traffic, mirroring the `auto_unmute.py` /
# `_stock/news.py` / `prompt_dev.py` pattern.
_TAUNT_BET_END_USER_ID: Final[str] = "casino_taunt_bet"
_SETTLE_END_USER_ID: Final[str] = "casino_settle"
_TABLE_SETTLE_END_USER_ID: Final[str] = "casino_table_settle"
_HINT_END_USER_ID: Final[str] = "casino_hint"
_FISH_CATCH_END_USER_ID: Final[str] = "casino_fish_catch"

_FISH_RARITY_LABELS: Final[dict[Rarity, str]] = {
    "N": "普通",
    "R": "稀有",
    "SR": "高度稀有",
    "SSR": "非常稀有",
    "UR": "傳說等級",
}


class SystemNarrator(BaseModel):
    """Wraps fast-model calls for the casino system's neutral broadcast lines.

    Attributes:
        client: The shared AsyncOpenAI client.
        model: Fast-model settings used for every narrator line.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI]
    model: ModelSettings

    async def _ask(
        self, instructions: str, user_text: str, fallback: str, end_user_id: str
    ) -> str:
        """Calls the LLM and returns the trimmed text, falling back on any error."""
        try:
            async with asyncio.timeout(delay=NARRATOR_AI_TIMEOUT_SECONDS):
                responses = await self.client.responses.create(
                    model=self.model.name,
                    instructions=instructions,
                    input=cast(
                        "ResponseInputParam",
                        [EasyInputMessageParam(role="user", content=user_text)],
                    ),
                    reasoning=self.model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": end_user_id},
                    extra_body={"mock_testing_fallbacks": False},
                )
        except TimeoutError:
            logfire.warn(
                "System narrator request timed out; using fallback line",
                timeout_seconds=NARRATOR_AI_TIMEOUT_SECONDS,
            )
            return fallback
        except Exception:
            logfire.warn("System narrator request failed; using fallback line", _exc_info=True)
            return fallback
        text = (responses.output_text or "").strip()
        return text or fallback

    async def taunt_bet(
        self, player_name: str, balance_at_start: int, bet: int, game: GameKind
    ) -> str:
        """Returns a neutral narrator line for a newly placed bet."""
        game_labels: dict[GameKind, str] = {"blackjack": "21 點", "dragon_gate": "射龍門"}
        user_text = (
            f"遊戲: {game_labels[game]}\n"
            f"玩家: {player_name}\n"
            f"下注金額 ({CURRENCY_NAME}): {bet}\n"
            f"開局餘額 ({CURRENCY_NAME}): {balance_at_start}"
        )
        return await self._ask(
            instructions=SYSTEM_TAUNT_BET_PROMPT,
            user_text=user_text,
            fallback="賭場已收到下注, 牌桌即將發牌",
            end_user_id=_TAUNT_BET_END_USER_ID,
        )

    async def settle(  # noqa: PLR0913 -- the round summary needs every field for the prompt
        self,
        player_name: str,
        outcome: SettleOutcome,
        bet: int,
        delta: int,
        new_balance: int,
        game: GameKind,
        detail: str,
    ) -> str:
        """Returns a neutral narrator line for a finished round."""
        game_labels: dict[GameKind, str] = {"blackjack": "21 點", "dragon_gate": "射龍門"}
        outcome_labels: dict[SettleOutcome, str] = {
            "win": "玩家贏",
            "lose": "玩家輸",
            "push": "平手",
            "blackjack": "玩家 Blackjack 21 點 (賠 1.5x)",
            "five_card_win": "玩家過五關未爆獲勝",
            "five_card_twenty_one": "玩家過五關 21 點",
            "player_bust": "玩家爆牌",
            "dealer_bust": "莊家爆牌",
            "surrender": "玩家投降, 退回一半本金",
        }
        user_text = (
            f"遊戲: {game_labels[game]}\n"
            f"玩家: {player_name}\n"
            f"下注金額 ({CURRENCY_NAME}): {bet}\n"
            f"結果: {outcome_labels[outcome]}\n"
            f"玩家本局淨變動 ({CURRENCY_NAME}): {delta:+d} (正為贏, 負為輸)\n"
            f"玩家結算後餘額 ({CURRENCY_NAME}): {new_balance}\n"
            f"局面細節: {detail}"
        )
        return await self._ask(
            instructions=SYSTEM_SETTLE_PROMPT,
            user_text=user_text,
            fallback=SETTLEMENT_FALLBACK_LINES[outcome],
            end_user_id=_SETTLE_END_USER_ID,
        )

    async def table_settle(
        self, table_name: str, player_count: int, net_delta: int, game: GameKind, detail: str
    ) -> str:
        """Returns one narrator line for a multiplayer table settlement."""
        game_labels: dict[GameKind, str] = {"blackjack": "21 點", "dragon_gate": "射龍門"}
        if net_delta > 0:
            fallback = "本桌整體玩家略勝, 賭場結算後支付差額"
        elif net_delta < 0:
            fallback = "本桌整體玩家未過關, 籌碼流向賭場"
        else:
            fallback = "本桌全部結算後雙方持平"
        user_text = (
            f"遊戲: {game_labels[game]}\n"
            f"桌名: {table_name}\n"
            f"玩家數: {player_count}\n"
            f"全桌玩家淨變動總和 ({CURRENCY_NAME}): {net_delta:+d}\n"
            f"局面細節: {detail}"
        )
        return await self._ask(
            instructions=SYSTEM_SETTLE_PROMPT,
            user_text=user_text,
            fallback=fallback,
            end_user_id=_TABLE_SETTLE_END_USER_ID,
        )

    async def catch_fish(  # noqa: PLR0913 -- the catch summary needs every field for the prompt
        self,
        player_name: str,
        species_name: str,
        rarity: Rarity,
        size_mm: int,
        sell_value: int,
        fallback: str,
    ) -> str:
        """Returns a neutral narrator line for a fishing catch, off the critical path."""
        user_text = (
            f"玩家: {player_name}\n"
            f"釣到魚種: {species_name}\n"
            f"稀有度: {_FISH_RARITY_LABELS[rarity]}\n"
            f"尺寸 (mm): {size_mm}\n"
            f"可賣金額 ({CURRENCY_NAME}): {sell_value}"
        )
        return await self._ask(
            instructions=SYSTEM_FISH_CATCH_PROMPT,
            user_text=user_text,
            fallback=fallback,
            end_user_id=_FISH_CATCH_END_USER_ID,
        )

    async def hint(self, player_name: str, player_total: int, dealer_visible: int) -> str:
        """Returns a neutral narrator line summarizing the current Blackjack state."""
        user_text = (
            f"玩家: {player_name}\n"
            f"玩家當前手牌總點數: {player_total}\n"
            f"莊家明牌點數: {dealer_visible}"
        )
        return await self._ask(
            instructions=SYSTEM_HINT_PROMPT,
            user_text=user_text,
            fallback="現場觀察: 玩家點數與莊家明牌已揭示, 等待玩家決策",
            end_user_id=_HINT_END_USER_ID,
        )
