"""AI-backed news generation for the simulated stock market."""

from typing import cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.stock import StockGeneratedNews, StockNewsGenerationContext
from discordbot.typings.models import ModelSettings
from discordbot.cogs._stock.prompts import STOCK_NEWS_PROMPT

STOCK_NEWS_AI_TIMEOUT_SECONDS = 4.0


class StockNewsDraft(BaseModel):
    """Structured LLM output for one generated stock news item."""

    model_config = ConfigDict(frozen=True)

    headline: str = Field(description="One short fictional Traditional Chinese headline")
    sentiment_bps: int = Field(
        description="Simulated market sentiment impact in basis points, from -180 to 180",
        ge=-180,
        le=180,
    )


class StockNewsAI(BaseModel):
    """Generates one bounded fictional stock news item with the runtime LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: AsyncOpenAI
    model: ModelSettings

    async def generate(self, context: StockNewsGenerationContext) -> StockGeneratedNews | None:
        """Returns one generated news item, or `None` when the LLM path fails."""
        profile = context.profile
        user_text = (
            f"Symbol: {profile.symbol}\n"
            f"Company: {profile.name}\n"
            f"Category: {profile.category}\n"
            f"Price: {profile.price_cents / 100:.2f}\n"
            f"Daily price change: {_signed_percent(bps=context.change_bps)} "
            f"({context.change_cents / 100:+.2f})\n"
            f"Recent order flow window: {context.lookback_hours} hours\n"
            f"Buy-side shares: {context.buy_side_shares:,}\n"
            f"Sell-side shares: {context.sell_side_shares:,}\n"
            f"Net order shares: {context.net_order_shares:+,}\n"
            f"Order pressure: {_signed_percent(bps=context.pressure_bps)} "
            f"({_pressure_label(pressure_bps=context.pressure_bps)})\n"
            f"Existing news sentiment now: {_signed_percent(bps=context.recent_news_sentiment_bps)}\n"
            f"Latest previous headline: {context.latest_news_headline or 'None'}\n"
            "Write a plausible fictional event that fits this context."
        )
        try:
            async with asyncio.timeout(delay=STOCK_NEWS_AI_TIMEOUT_SECONDS):
                responses = await self.client.responses.parse(
                    model=self.model.name,
                    instructions=STOCK_NEWS_PROMPT,
                    input=cast(
                        "ResponseInputParam",
                        [EasyInputMessageParam(role="user", content=user_text)],
                    ),
                    text_format=StockNewsDraft,
                    reasoning=self.model.reasoning,
                    service_tier="auto",
                    extra_headers={"x-litellm-end-user-id": "stock_news"},
                    extra_body={"mock_testing_fallbacks": False},
                )
        except TimeoutError:
            logfire.warn(
                "Stock news AI request timed out; using deterministic fallback",
                timeout_seconds=STOCK_NEWS_AI_TIMEOUT_SECONDS,
                symbol=profile.symbol,
            )
            return None
        except ValidationError:
            logfire.warn("Stock news AI parse failed; using fallback", symbol=profile.symbol)
            return None
        except Exception:
            logfire.warn(
                "Stock news AI request failed; using deterministic fallback",
                symbol=profile.symbol,
                _exc_info=True,
            )
            return None
        if responses.output_parsed is None:
            logfire.warn("Stock news AI returned no parsed output", symbol=profile.symbol)
            return None
        return StockGeneratedNews(
            headline=responses.output_parsed.headline.strip(),
            sentiment_bps=responses.output_parsed.sentiment_bps,
            source="ai",
            model=self.model.name,
        )


def _signed_percent(bps: int) -> str:
    """Formats basis points as a signed percent."""
    return f"{bps / 100:+.2f}%"


def _pressure_label(pressure_bps: int) -> str:
    """Returns a compact order-flow label for the AI prompt."""
    if pressure_bps >= 60:
        return "strong buy pressure"
    if pressure_bps >= 20:
        return "buy pressure"
    if pressure_bps <= -60:
        return "strong sell pressure"
    if pressure_bps <= -20:
        return "sell pressure"
    return "balanced"


__all__ = ["STOCK_NEWS_AI_TIMEOUT_SECONDS", "STOCK_NEWS_PROMPT", "StockNewsAI", "StockNewsDraft"]
