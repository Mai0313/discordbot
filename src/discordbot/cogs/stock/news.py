"""The AI half of the simulated market's news feed: one LLM call per symbol that has come due.

`StockNewsAI.generate` is one of the two interchangeable producers behind
`services/stock/database.py::ensure_due_stock_news`. It flattens a `StockNewsGenerationContext`
into a single user-role message, sends it under `STOCK_NEWS_PROMPT`, and hands back the same
`StockGeneratedNews` shape the deterministic templates return — stamped `source="ai"` plus the
model string, which is what lets a stored `stock_news` row say which producer wrote it and lets
the store upgrade a template bucket in place without ever downgrading it back.

It sits in the cog rather than beside the engine because a market tick may never depend on an LLM
call, and `services/` may not import a cog. So the dependency runs the other way: `StockCogs`
builds this only when the proxy is configured and passes `generate` in as the sweep's optional
`news_provider`, while the fallback copy stays in `services/stock/prompts.py` next to the engine
that has to reach it. A deployment with no credentials is the same market with template headlines.

Nothing here handles failure, which is the arrangement rather than an omission.
`parse_responses_or_none` already turns a timeout, an empty or off-schema body and any transport
error into None; `generate` passes that None on, and the sweep degrades that one symbol to a
template built from the very same context. `STOCK_NEWS_AI_TIMEOUT_SECONDS` bounds one call, and it
is short because the sweep holds a process-wide lock while its provider calls run, so a stalled
call would keep the next `/stock` refresh from starting; the cost of overrunning it is one
template headline.

The two module-private formatters are here so the rendered context reads in the vocabulary the
instructions were written against: signed percentages rather than bare basis points, and one
coarse order-flow label rather than a number the model would have to threshold for itself.
"""

from openai import AsyncOpenAI
from pydantic import Field, BaseModel, ConfigDict

from discordbot.utils.llm import parse_responses_or_none
from discordbot.typings.stock import StockGeneratedNews, StockNewsGenerationContext
from discordbot.typings.models import ModelSettings
from discordbot.services.stock.prompts import STOCK_NEWS_PROMPT

STOCK_NEWS_AI_TIMEOUT_SECONDS = 4.0


class StockNewsDraft(BaseModel):
    """Structured LLM output for one generated stock news item.

    Only what the model authors, which is why it is not `StockGeneratedNews`: the origin and the
    model string are stamped by code afterwards rather than asked for and trusted. Both field
    descriptions are sent to the model as part of the parse schema, so they are prompt text and
    changing one changes what comes back.

    The ±180 bps bound mirrors the fallback templates instead of the market's own clamp
    (`NEWS_SENTIMENT_LIMIT_BPS`, 300), so the two producers stay comparable; a draft outside it
    fails validation and the whole call degrades to None rather than being clipped.
    """

    model_config = ConfigDict(frozen=True)

    headline: str = Field(..., description="One short fictional Traditional Chinese headline")
    sentiment_bps: int = Field(
        ...,
        description="Simulated market sentiment impact in basis points, from -180 to 180",
        ge=-180,
        le=180,
    )


class StockNewsAI(BaseModel):
    """Generates one bounded fictional stock news item with the runtime LLM.

    Holds nothing but the call surface, so the one instance the cog caches is safe to call for
    several due symbols at once; the sweep is what bounds that concurrency.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: AsyncOpenAI = Field(
        ..., description="Async OpenAI client for the news generation call."
    )
    model: ModelSettings = Field(..., description="Model settings for the news generation call.")

    async def generate(self, context: StockNewsGenerationContext) -> StockGeneratedNews | None:
        """Renders the due symbol's market state into one prompt and returns what the model wrote.

        Best-effort by contract: every failure is already absorbed by `parse_responses_or_none` and
        arrives here as None, which is passed straight on because the caller's template fallback is
        what owns the degrade. The previous headline is rendered too, since the prompt's
        do-not-repeat rule has nothing to compare against otherwise; a symbol with no stored
        headline sends the literal "None".

        Args:
            context (StockNewsGenerationContext): The due symbol's market picture, rendered whole
                into the single user message.

        Returns:
            The headline stamped `source="ai"` and the model string, or None when the call produced
            no draft. A draft whose headline is only whitespace still comes back as a headline, and
            the caller reads that blank as a failed provider.
        """
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
    """Formats basis points as a signed percent.

    The sign is always written out, so a rise and a fall are told apart by the figure itself rather
    than by the label around it.

    Args:
        bps (int): The value to render, in basis points.

    Returns:
        The value as a percentage carrying an explicit sign, such as "+1.25%".
    """
    return f"{bps / 100:+.2f}%"


def _pressure_label(pressure_bps: int) -> str:
    """Returns a compact order-flow label for the AI prompt.

    The prompt asks the model to tie its joke to buy or sell pressure, so the tape is handed over
    already bucketed rather than as a number the model would have to threshold itself; the ±20 bps
    dead band is what stops a near-flat tape reading as a direction.

    Args:
        pressure_bps (int): Recent decayed order-flow pressure in basis points.

    Returns:
        One of "strong buy pressure", "buy pressure", "balanced", "sell pressure" or "strong sell
        pressure".
    """
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
