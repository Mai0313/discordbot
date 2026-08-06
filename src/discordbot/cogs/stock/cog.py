"""Discord surface of the simulated stock market: the single `/stock` slash command.

`/stock` answers publicly with the market board — an embed plus a rendered PNG of every listed
company — and attaches `StockMarketView`, the opener-only panel every later screen is reached
through (stock select, pagination, tutorial, per-stock detail, news, and the buy / short flow).
Those screens all live in `views.py`; this module opens the first one, binds the view to the
message it just sent, and records that message with `utils.message_cleanup`, so the panel is
still cleaned up after a restart that loses the view's own idle timeout. An interaction carrying
no Discord user raises instead of posting a panel nobody would be allowed to operate.

No kill-switch and no permission gate: the command is open to everyone, and the market itself
(ticks, pricing, settlement, storage) belongs to `services/stock/`, which this module only reads
through. The one optional piece is `news_ai`, the LLM headline writer, which self-disables when
the runtime proxy credentials are missing; the market then runs on the service's deterministic
news templates instead.

Headline generation never sits on the command's critical path. When the writer exists the command
schedules the due-news sweep as a background task — de-duped process-wide by the module globals
below, so overlapping `/stock` calls share one sweep — and renders the board without waiting, so
a slow LLM call costs the NEXT panel its freshest headline rather than this one its latency. Only
the writer-free path sweeps inline, where the templates are cheap enough to await.
"""

import asyncio
from functools import cached_property

from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import Locale, Interaction
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs.stock.news import StockNewsAI
from discordbot.cogs.stock.views import (
    StockMarketView,
    require_stock_user,
    build_market_message_payload,
)
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.message_cleanup import track_public_message
from discordbot.services.stock.database import list_market_quotes, ensure_due_stock_news

_stock_news_refresh_task: asyncio.Task[None] | None = None
_stock_news_refresh_task_loop: asyncio.AbstractEventLoop | None = None


class StockCogs(commands.Cog):
    """Registers `/stock` and owns the optional LLM news writer behind it."""

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the stock cog.

        Args:
            bot (commands.Bot): The bot instance this cog is added to.
        """
        self.bot = bot
        self.runtime_models = RuntimeModelCatalog()

    @cached_property
    def news_ai(self) -> StockNewsAI | None:
        """The LLM headline writer, or None when the runtime proxy is not configured.

        Cached so one client is built per cog instance rather than per `/stock`.

        Returns:
            A writer bound to the proxy on `fast_model`, or None when the market has to fall
            back to the service's deterministic news templates.
        """
        config = LLMConfig()
        # Credentials default to empty rather than raising, so a missing proxy shows up as the
        # empty value here; falling back to deterministic news beats building a client that
        # would only fail on its first request.
        if not config.base_url or not config.api_key:
            return None
        return StockNewsAI(
            client=AsyncOpenAI(base_url=config.base_url, api_key=config.api_key),
            model=self.runtime_models.fast_model,
        )

    @nextcord.slash_command(
        name="stock",
        description="Open the simulated stock market.",
        name_localizations={Locale.zh_TW: "股票", Locale.ja: "株式"},
        description_localizations={
            Locale.zh_TW: "開啟模擬股票市場",
            Locale.ja: "シミュレーション株式市場を開きます。",
        },
        nsfw=False,
    )
    async def stock(self, interaction: Interaction[commands.Bot]) -> None:
        """Opens the public market board and binds its panel to the invoking user.

        Resolves the user before any market read, so an interaction with no Discord identity
        fails before a panel nobody owns is posted. The due-news sweep is left out of the board
        read whenever the writer exists, because the background task started just above already
        runs it with that writer; without one the sweep runs inline on the cheap templates. The
        sent message is tracked for cleanup, so the panel is deleted even across a restart.

        Args:
            interaction (Interaction[commands.Bot]): The `/stock` invocation.
        """
        await interaction.response.defer()
        user = require_stock_user(interaction=interaction)
        news_ai = self.news_ai
        if news_ai is not None:
            _schedule_stock_news_refresh(news_ai=news_ai)
        quotes = await list_market_quotes(refresh_news=news_ai is None)
        view = StockMarketView(quotes=quotes, owner_id=user.id)
        embed, file = build_market_message_payload(quotes=quotes)
        message = await interaction.followup.send(
            embed=embed,
            view=view,
            wait=True,
            **embed_spacer_payload(
                embeds=[embed], is_edit=False, target=interaction, extra_files=[file]
            ),
        )
        view.bind_message(message=message)
        await track_public_message(message=message, user_name=user.name)


def _schedule_stock_news_refresh(news_ai: StockNewsAI) -> None:
    """Starts the due-news sweep in the background so the market UI never waits on the LLM.

    One sweep at a time per process: a `/stock` arriving while another is in flight is dropped
    rather than queued, since the running sweep already covers every symbol that is due. The slot
    is reset whenever the running loop changes, because a task left behind by a closed loop can
    never report `done()` and would block every later refresh (each test runs on a fresh loop).

    Args:
        news_ai (StockNewsAI): Writer whose `generate` becomes the sweep's headline provider.
    """
    global _stock_news_refresh_task, _stock_news_refresh_task_loop  # noqa: PLW0603 -- process task de-dupe
    loop = asyncio.get_running_loop()
    if _stock_news_refresh_task_loop is not loop:
        _stock_news_refresh_task = None
        _stock_news_refresh_task_loop = loop
    if _stock_news_refresh_task is not None and not _stock_news_refresh_task.done():
        return
    task = asyncio.create_task(ensure_due_stock_news(news_provider=news_ai.generate))
    _stock_news_refresh_task = task
    task.add_done_callback(_finish_stock_news_refresh)


def _finish_stock_news_refresh(task: asyncio.Task[None]) -> None:
    """Frees the background refresh slot and logs a sweep that did not succeed.

    Clears the slot only while it still holds this task: the callback runs after the task is
    already `done()`, so a newer sweep may have claimed the slot in between and must not lose its
    de-dupe to a late callback.

    Args:
        task (asyncio.Task[None]): The refresh task that just finished.
    """
    global _stock_news_refresh_task  # noqa: PLW0603 -- process task de-dupe
    if _stock_news_refresh_task is task:
        _stock_news_refresh_task = None
    _log_stock_news_refresh_failure(task=task)


def _log_stock_news_refresh_failure(task: asyncio.Task[None]) -> None:
    """Reads the finished sweep's outcome and logs anything other than success.

    Reading the result is also what keeps asyncio from reporting the exception as never retrieved
    when the task is collected, since nothing awaits this fire-and-forget task.

    Args:
        task (asyncio.Task[None]): The refresh task that just finished.
    """
    try:
        task.result()
    except asyncio.CancelledError:
        # CancelledError is a BaseException, so it must be handled here or it escapes
        # this done-callback into the loop's exception handler on shutdown.
        logfire.info("Background stock news refresh cancelled")
    # Broad on purpose: fire-and-forget refresh whose failure must only leave the news
    # stale (deterministic fallback templates), never surface to a caller.
    except Exception as exc:
        logfire.warn(
            "Background stock news refresh failed", error_type=type(exc).__name__, _exc_info=exc
        )


def setup(bot: commands.Bot) -> None:
    """Adds the StockCogs to the bot.

    Args:
        bot (commands.Bot): The bot instance to register the cog on.
    """
    bot.add_cog(StockCogs(bot), override=True)
