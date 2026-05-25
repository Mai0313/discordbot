"""AI dealer that produces short banter lines for the casino games."""

from typing import cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import BaseModel, ConfigDict, SkipValidation, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.games import GameKind, SettleOutcome, BlackjackDealerDecision
from discordbot.typings.models import ModelSettings
from discordbot.cogs._games.prompts import (
    DEALER_HINT_PROMPT,
    DEALER_SETTLE_PROMPT,
    DEALER_TAUNT_BET_PROMPT,
    DEALER_BLACKJACK_DECISION_PROMPT,
)
from discordbot.cogs._economy.presentation import CURRENCY_NAME

DEALER_AI_TIMEOUT_SECONDS = 5.0
DEALER_BLACKJACK_DECISION_TIMEOUT_SECONDS = 20.0


def _fallback_blackjack_decision(dealer_total: int) -> BlackjackDealerDecision:
    """Returns the deterministic basic-rule dealer decision."""
    if dealer_total < 17:
        return BlackjackDealerDecision(action="hit", reason="basic rule: 未滿 17 點")
    return BlackjackDealerDecision(action="stand", reason="basic rule: 已達 17 點")


class DealerAI(BaseModel):
    """Wraps fast-model calls for game banter.

    Attributes:
        client: The shared AsyncOpenAI client.
        model: Fast-model settings used for every dealer line.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SkipValidation[AsyncOpenAI]
    model: ModelSettings

    async def _ask(
        self, instructions: str, user_text: str, fallback: str, end_user_id: str
    ) -> str:
        """Calls the LLM and returns the trimmed text, falling back on any error."""
        try:
            async with asyncio.timeout(delay=DEALER_AI_TIMEOUT_SECONDS):
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
                "Dealer AI request timed out; using fallback line",
                timeout_seconds=DEALER_AI_TIMEOUT_SECONDS,
            )
            return fallback
        except Exception:
            logfire.warn("Dealer AI request failed; using fallback line", _exc_info=True)
            return fallback
        text = (responses.output_text or "").strip()
        return text or fallback

    async def taunt_bet(
        self, author_name: str, player_name: str, balance_at_start: int, bet: int, game: GameKind
    ) -> str:
        """Returns a dealer line for a newly placed bet.

        Args:
            author_name: Discord username used as the LiteLLM end-user ID.
            player_name: Display name to include in the prompt.
            balance_at_start: Player balance observed when the round started.
            bet: Effective bet amount in points.
            game: Game type for the prompt.

        Returns:
            A trimmed model-generated line, or the fallback line on request
            failure or empty output.
        """
        game_labels: dict[GameKind, str] = {"blackjack": "21 點", "dragon_gate": "射龍門"}
        user_text = (
            f"遊戲: {game_labels[game]}\n"
            f"玩家: {player_name}\n"
            f"下注金額 ({CURRENCY_NAME}): {bet}\n"
            f"開局餘額 ({CURRENCY_NAME}): {balance_at_start}"
        )
        return await self._ask(
            instructions=DEALER_TAUNT_BET_PROMPT,
            user_text=user_text,
            fallback="下好離手, 不要等下哭",
            end_user_id=author_name,
        )

    async def settle(  # noqa: PLR0913 -- the round summary needs every field for the prompt
        self,
        author_name: str,
        player_name: str,
        outcome: SettleOutcome,
        bet: int,
        delta: int,
        new_balance: int,
        game: GameKind,
        detail: str,
    ) -> str:
        """Returns a dealer line for a finished round.

        Args:
            author_name: Discord username used as the LiteLLM end-user ID.
            player_name: Display name to include in the prompt.
            outcome: Player-facing outcome label.
            bet: Effective bet amount in points.
            delta: Player net point change for the round.
            new_balance: Player balance after settlement.
            game: Game type for the prompt.
            detail: Game-state detail text to include in the prompt.

        Returns:
            A trimmed model-generated line, or an outcome-specific fallback line
            on request failure or empty output.
        """
        game_labels: dict[GameKind, str] = {"blackjack": "21 點", "dragon_gate": "射龍門"}
        outcome_labels: dict[SettleOutcome, str] = {
            "win": "玩家贏",
            "lose": "玩家輸",
            "push": "平手",
            "blackjack": "玩家直接 Blackjack 21 點 (賠 1.5x)",
            "five_card_win": "玩家過五關未爆直接贏",
            "five_card_twenty_one": "玩家過五關 21 點",
            "player_bust": "玩家爆牌, 莊家自動贏",
            "dealer_bust": "莊家爆牌, 玩家贏",
            "surrender": "玩家投降, 退回一半本金",
        }
        fallback_lines: dict[SettleOutcome, str] = {
            "win": "算你今天運氣好, 下一把不會這麼順",
            "lose": "下次再來送錢吧",
            "push": "白忙一場, 賭場最開心的就是這種局",
            "blackjack": "Blackjack? 算你會玩, 下一把見真章",
            "five_card_win": "過五關沒爆, 這把讓你過",
            "five_card_twenty_one": "過五關也給你摸到, 這把算你有耐心",
            "player_bust": "爆了爆了, 沒事多算算數字好嗎",
            "dealer_bust": "靠杯, 這把莊家自爆, 你撿到便宜了",
            "surrender": "投降也算會止血, 下一把再說",
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
            instructions=DEALER_SETTLE_PROMPT,
            user_text=user_text,
            fallback=fallback_lines[outcome],
            end_user_id=author_name,
        )

    async def table_settle(  # noqa: PLR0913 -- table summary needs every field for the prompt
        self,
        author_name: str,
        table_name: str,
        player_count: int,
        net_delta: int,
        game: GameKind,
        detail: str,
    ) -> str:
        """Returns one dealer line for a multiplayer table settlement.

        Args:
            author_name: Discord username used as the LiteLLM end-user ID.
            table_name: Display label for the table.
            player_count: Number of settled players.
            net_delta: Sum of all player deltas from the table.
            game: Game type for the prompt.
            detail: Compact per-player result summary.

        Returns:
            A trimmed model-generated line, or a fallback line on request
            failure or empty output.
        """
        game_labels: dict[GameKind, str] = {"blackjack": "21 點", "dragon_gate": "射龍門"}
        if net_delta > 0:
            fallback = "今天這桌有點旺, 但賭場不會天天讓你們舒服"
        elif net_delta < 0:
            fallback = "一桌人一起送, 我收得都不好意思了"
        else:
            fallback = "忙了半天打平, 這桌也算會拖時間"
        user_text = (
            f"遊戲: {game_labels[game]}\n"
            f"桌名: {table_name}\n"
            f"玩家數: {player_count}\n"
            f"全桌玩家淨變動總和 ({CURRENCY_NAME}): {net_delta:+d}\n"
            f"局面細節: {detail}"
        )
        return await self._ask(
            instructions=DEALER_SETTLE_PROMPT,
            user_text=user_text,
            fallback=fallback,
            end_user_id=author_name,
        )

    async def hint(
        self, author_name: str, player_name: str, player_total: int, dealer_visible: int
    ) -> str:
        """Returns a dealer hint for a Blackjack hit-or-stand decision.

        Args:
            author_name: Discord username used as the LiteLLM end-user ID.
            player_name: Display name to include in the prompt.
            player_total: Current player hand total.
            dealer_visible: Visible dealer card value.

        Returns:
            A trimmed model-generated line, or the fallback hint on request
            failure or empty output.
        """
        user_text = (
            f"玩家: {player_name}\n"
            f"玩家當前手牌總點數: {player_total}\n"
            f"莊家明牌點數: {dealer_visible}"
        )
        return await self._ask(
            instructions=DEALER_HINT_PROMPT,
            user_text=user_text,
            fallback="看你自己的, 我可不會手下留情",
            end_user_id=author_name,
        )

    async def decide_blackjack_action(
        self, author_name: str, table_state: str, dealer_total: int
    ) -> BlackjackDealerDecision:
        """Returns the AI dealer's next Blackjack hit / stand decision.

        Args:
            author_name: Discord username used as the LiteLLM end-user ID.
            table_state: Full table state available to the dealer.
            dealer_total: Current dealer hand total, used for fallback rules.

        Returns:
            Parsed dealer decision, or the basic-rule fallback on timeout,
            parse failure, empty output, or SDK failure.
        """
        fallback = _fallback_blackjack_decision(dealer_total=dealer_total)
        try:
            async with asyncio.timeout(delay=DEALER_BLACKJACK_DECISION_TIMEOUT_SECONDS):
                responses = await self.client.responses.parse(
                    model=self.model.name,
                    instructions=DEALER_BLACKJACK_DECISION_PROMPT,
                    input=cast(
                        "ResponseInputParam",
                        [EasyInputMessageParam(role="user", content=table_state)],
                    ),
                    text_format=BlackjackDealerDecision,
                    reasoning=self.model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": author_name},
                    extra_body={"mock_testing_fallbacks": False},
                )
        except TimeoutError:
            logfire.warn(
                "Dealer Blackjack decision timed out; using basic-rule fallback",
                timeout_seconds=DEALER_BLACKJACK_DECISION_TIMEOUT_SECONDS,
            )
            return fallback
        except ValidationError:
            logfire.warn(
                "Dealer Blackjack decision parse failed; using basic-rule fallback", _exc_info=True
            )
            return fallback
        except Exception:
            logfire.warn(
                "Dealer Blackjack decision request failed; using basic-rule fallback",
                _exc_info=True,
            )
            return fallback
        if responses.output_parsed is None:
            logfire.warn("Dealer Blackjack decision was empty; using basic-rule fallback")
            return fallback
        return responses.output_parsed
