"""AI-backed news generation for the simulated stock market."""

from typing import cast
import asyncio

from openai import AsyncOpenAI
import logfire
from pydantic import Field, BaseModel, ConfigDict, SkipValidation, ValidationError
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.stock import StockProfileView, StockGeneratedNews
from discordbot.typings.models import ModelSettings

STOCK_NEWS_AI_TIMEOUT_SECONDS = 4.0
STOCK_NEWS_PROMPT = """
You write one short fictional news headline for a Discord bot's simulated stock market.
The company is virtual. Do not claim this is real financial news, real investment advice, or a real exchange event.
Return one concise Traditional Chinese headline and a market sentiment value in basis points from -180 to 180.
The headline should fit naturally in a Discord embed and should not include markdown.
""".strip()


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

    client: SkipValidation[AsyncOpenAI]
    model: ModelSettings

    async def generate(self, profile: StockProfileView) -> StockGeneratedNews | None:
        """Returns one generated news item, or `None` when the LLM path fails."""
        user_text = (
            f"Symbol: {profile.symbol}\n"
            f"Company: {profile.name}\n"
            f"Category: {profile.category}\n"
            f"Price: {profile.price_cents / 100:.2f}\n"
            f"Daily change bps anchor is not available; write a plausible fictional event."
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


__all__ = ["STOCK_NEWS_AI_TIMEOUT_SECONDS", "STOCK_NEWS_PROMPT", "StockNewsAI", "StockNewsDraft"]
