# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tradingagents @ git+https://github.com/TauricResearch/TradingAgents",
#     "rich",
# ]
# ///
"""Runs one TradingAgents analysis with every setting pinned in this file.

`uv run <path to this file>` installs TradingAgents from the default branch of https://github.com/TauricResearch/TradingAgents on every run, so the file works from any folder and needs no checkout. Imported from other code, it needs the `tradingagents` package installed there.

Where to change a setting:
- Ticker, trade date and results folder: the `__main__` block, or the arguments of get_trading_decision().
- Models, rounds, language, news limits and the rest: config.update() in get_trading_decision(). It overrides every TRADINGAGENTS_* environment variable.
- Thinking level: google_thinking_level in config.update() is the deep tier's; _create_tier_client() sets the quick tier's.
- API keys (GOOGLE_API_KEY, FRED_API_KEY): the shell, or the nearest .env at or above the folder the command runs in. TradingAgents loads it on import.

Where a run writes:
- Reports: <results_dir>/reports/<TICKER>_<YYYYmmdd_HHMMSS>/, with complete_report.md and complete_report.html.
- Final state as JSON: <results_dir>/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<trade_date>.json.
- Price and SEC EDGAR cache: data_cache_dir; checkpoints in its checkpoints/.
- Decision and reflection log: memory_log_path, a markdown file each finished run appends its decision to.

Upstream code this file depends on:
- https://github.com/TauricResearch/TradingAgents/blob/main/tradingagents/default_config.py
  _ENV_OVERRIDES maps environment variables to config keys. config.update() must set every key it lists, plus results_dir, data_cache_dir and memory_log_path; a key left out takes its environment value. The other keys set here come from build_default_config() in the same file.
- https://github.com/TauricResearch/TradingAgents/blob/main/tradingagents/graph/trading_graph.py
  TradingAgentsGraph.__init__ builds both tiers through create_tier_client (defined in tradingagents/llm_clients/factory.py), which applies one google_thinking_level to both. _create_tier_client() replaces it there while the graph is built; drop it once TradingAgents takes a level per tier.
  propagate() also writes the state log and the memory log entry and clears the checkpoint, so get_trading_decision() adds only save_reports().

To check a change without a paid run, build TradingAgentsGraph and stop before propagate().
"""

import os
from datetime import date
from unittest import mock

from tradingagents.llm_clients import BaseLLMClient, create_tier_client
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _create_tier_client(config: dict, tier: str, **extra: object) -> BaseLLMClient:
    """Creates a tier's client, with the quick tier at Gemini thinking level low."""
    if tier == "quick":
        config = {**config, "google_thinking_level": "low"}
    return create_tier_client(config, tier, **extra)


def get_trading_decision(ticker: str, trade_date: str, results_dir: str) -> str:
    """Returns the rating for `ticker` as of `trade_date` and saves the reports.

    The rating is one of Buy, Overweight, Hold, Underweight or Sell, or REVIEW when the final decision carries no parseable rating.

    Args:
        ticker (str): Yahoo Finance symbol, such as `2330.TW` or `NVDA`.
        trade_date (str): Analysis date as YYYY-MM-DD.
        results_dir (str): Folder for reports and state logs; a relative path resolves against the working directory.

    Raises:
        ValueError: `trade_date` is not YYYY-MM-DD or is after today.
    """
    # Every setting is pinned here; only API keys come from the environment.
    config = DEFAULT_CONFIG.copy()
    config.update({
        "llm_provider": "google",
        "deep_think_llm": "gemini-4-argon",
        "quick_think_llm": "gemini-4-argon",
        "deep_think_provider": None,
        "quick_think_provider": None,
        "backend_url": None,
        "deep_think_backend_url": None,
        "quick_think_backend_url": None,
        "google_thinking_level": "high",  # deep tier; the quick tier's is in _create_tier_client()
        "openai_reasoning_effort": None,
        "anthropic_effort": None,
        "temperature": 0.7,
        "max_tokens": 32768,
        "llm_max_retries": 3,
        "output_language": "zh-TW",
        "max_debate_rounds": 5,
        "max_risk_discuss_rounds": 5,
        "max_tool_rounds": 20,
        "checkpoint_enabled": True,
        "benchmark_ticker": None,
        "results_dir": results_dir,
        "data_cache_dir": os.path.expanduser("~/.tradingagents/cache"),
        "memory_log_path": os.path.expanduser("~/.tradingagents/memory/trading_memory.md"),
        "news_article_limit": 20,
        "global_news_article_limit": 10,
        "global_news_lookback_days": 7,
        "global_news_queries": [
            "Federal Reserve interest rates inflation",
            "S&P 500 earnings GDP economic outlook",
            "geopolitical risk trade war sanctions",
            "ECB Bank of England BOJ central bank policy",
            "oil commodities supply chain energy",
        ],
        "holding_period_days": 5,
    })

    with mock.patch("tradingagents.graph.trading_graph.create_tier_client", _create_tier_client):
        ta = TradingAgentsGraph(config=config)
    final_state, decision = ta.propagate(ticker, trade_date)
    ta.save_reports(final_state=final_state, ticker=ticker)
    return decision


if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    trade_date = date.today().isoformat()
    result = get_trading_decision(ticker="GOOG", trade_date=trade_date, results_dir="./reports")
    console.print(result)
