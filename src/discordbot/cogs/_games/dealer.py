"""AI dealer that produces short banter lines for the casino games."""

from typing import Literal, cast

from openai import AsyncOpenAI
import logfire
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.models import ModelSettings
from discordbot.cogs._games.prompts import (
    DEALER_HINT_PROMPT,
    DEALER_SETTLE_PROMPT,
    DEALER_TAUNT_BET_PROMPT,
)

GameKind = Literal["dice", "blackjack"]
SettleOutcome = Literal["win", "lose", "push", "blackjack", "player_bust", "dealer_bust"]

_FALLBACK_BET = "下好離手, 不要等下哭。"
_FALLBACK_WIN = "算你今天運氣好, 下一把不會這麼順。"
_FALLBACK_LOSE = "下次再來送錢吧。"
_FALLBACK_PUSH = "白忙一場, 賭場最開心的就是這種局。"
_FALLBACK_HINT = "看你自己的, 我可不會手下留情。"


class DealerAI:
    """Wraps fast-model calls for game banter.

    Attributes:
        client: The shared AsyncOpenAI client.
        model: Fast-model settings used for every dealer line.
    """

    def __init__(self, client: AsyncOpenAI, model: ModelSettings) -> None:
        """Initialises the dealer with a pre-built client and model.

        Args:
            client: The AsyncOpenAI client to reuse.
            model: ``ModelSettings`` for the chat model that produces lines.
        """
        self.client = client
        self.model = model

    async def _ask(
        self, *, instructions: str, user_text: str, fallback: str, end_user_id: str
    ) -> str:
        """Calls the LLM and returns the trimmed text, falling back on any error."""
        try:
            responses = await self.client.responses.create(
                model=self.model.name,
                instructions=instructions,
                input=cast(
                    "ResponseInputParam", [EasyInputMessageParam(role="user", content=user_text)]
                ),
                reasoning=self.model.reasoning,
                service_tier="auto",
                extra_headers={"x-litellm-end-user-id": end_user_id},
                extra_body={"mock_testing_fallbacks": False},
            )
        except Exception:
            logfire.warn("Dealer AI request failed; using fallback line", _exc_info=True)
            return fallback
        text = (responses.output_text or "").strip()
        return text or fallback

    async def taunt_bet(
        self,
        *,
        author_name: str,
        player_name: str,
        balance_after_bet: int,
        bet: int,
        game: GameKind,
    ) -> str:
        """Returns a dealer line for a newly placed bet.

        Args:
            author_name: Discord username used as the LiteLLM end-user ID.
            player_name: Display name to include in the prompt.
            balance_after_bet: Player balance after the wager was withdrawn.
            bet: Effective bet amount in points.
            game: Game type for the prompt.

        Returns:
            A trimmed model-generated line, or the fallback line on request
            failure or empty output.
        """
        game_label = "比大小骰子" if game == "dice" else "21 點"
        user_text = (
            f"遊戲: {game_label}\n"
            f"玩家: {player_name}\n"
            f"下注金額: {bet}\n"
            f"下注後剩餘餘額: {balance_after_bet}"
        )
        return await self._ask(
            instructions=DEALER_TAUNT_BET_PROMPT,
            user_text=user_text,
            fallback=_FALLBACK_BET,
            end_user_id=author_name,
        )

    async def settle(  # noqa: PLR0913 -- the round summary needs every field for the prompt
        self,
        *,
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
        game_label = "比大小骰子" if game == "dice" else "21 點"
        outcome_label = {
            "win": "玩家贏",
            "lose": "玩家輸",
            "push": "平手",
            "blackjack": "玩家直接 Blackjack 21 點 (賠 1.5x)",
            "player_bust": "玩家爆牌, 莊家自動贏",
            "dealer_bust": "莊家爆牌, 玩家贏",
        }[outcome]
        fallback = {
            "win": _FALLBACK_WIN,
            "lose": _FALLBACK_LOSE,
            "push": _FALLBACK_PUSH,
            "blackjack": "Blackjack? 算你會玩, 下一把見真章。",
            "player_bust": "爆了爆了, 沒事多算算數字好嗎。",
            "dealer_bust": "靠杯, 這把莊家自爆, 你撿到便宜了。",
        }[outcome]
        user_text = (
            f"遊戲: {game_label}\n"
            f"玩家: {player_name}\n"
            f"下注金額: {bet}\n"
            f"結果: {outcome_label}\n"
            f"玩家本局淨變動: {delta:+d} (正為贏, 負為輸)\n"
            f"玩家結算後餘額: {new_balance}\n"
            f"局面細節: {detail}"
        )
        return await self._ask(
            instructions=DEALER_SETTLE_PROMPT,
            user_text=user_text,
            fallback=fallback,
            end_user_id=author_name,
        )

    async def hint(
        self, *, author_name: str, player_name: str, player_total: int, dealer_visible: int
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
            fallback=_FALLBACK_HINT,
            end_user_id=author_name,
        )
