"""AI-backed news generation for the simulated stock market."""

from openai import AsyncOpenAI
from pydantic import Field, BaseModel, ConfigDict

from discordbot.utils.llm import parse_responses_or_none
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

    client: AsyncOpenAI = Field(description="Async OpenAI client for the news generation call.")
    model: ModelSettings = Field(description="Model settings for the news generation call.")

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
        draft = await parse_responses_or_none(
            client=self.client,
            model=self.model,
            instructions=STOCK_NEWS_PROMPT,
            user_text=user_text,
            end_user_id="stock_news",
            text_format=StockNewsDraft,
            timeout_seconds=STOCK_NEWS_AI_TIMEOUT_SECONDS,
        )
        if draft is None:
            return None
        return StockGeneratedNews(
            headline=draft.headline.strip(),
            sentiment_bps=draft.sentiment_bps,
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


__all__ = ["STOCK_NEWS_AI_TIMEOUT_SECONDS", "StockNewsAI", "StockNewsDraft"]
