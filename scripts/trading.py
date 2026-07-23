"""Standalone (non-TUI) TradingAgents runner for a basket of tickers.

A scripted alternative to the TUI, meant to be runnable from any directory:
every config value the `.env` + `DEFAULT_CONFIG` flow would resolve is set explicitly below, so the script does not depend on a discoverable `.env` (which is only found by walking up from the current working directory).
Only the provider API key must be present in the environment (e.g. `GOOGLE_API_KEY`); results/cache/memory paths default to `~/.tradingagents` and are cwd-independent.

The interactive TUI selections are hardcoded instead: all four analysts, today's date, and the ticker list at the bottom.
The commented provider blocks are ready-to-swap examples — uncomment one and comment out the active block.
"""

import datetime

# Not a pyproject dependency; kept for a future feature and runs outside this env.
from tradingagents.default_config import DEFAULT_CONFIG  # ty: ignore[unresolved-import]
from tradingagents.graph.trading_graph import TradingAgentsGraph  # ty: ignore[unresolved-import]


def run_trading_agents(stock: str) -> str:
    config = DEFAULT_CONFIG.copy()

    # Use Gemini
    config["llm_provider"] = "google"
    config["deep_think_llm"] = "gemini-3.1-pro-preview"
    config["quick_think_llm"] = "gemini-3.5-flash"
    config["backend_url"] = None
    config["google_thinking_level"] = "high"

    # # Use OpenAI
    # config["llm_provider"] = "openai"
    # config["deep_think_llm"] = "gpt-5.5"
    # config["quick_think_llm"] = "gpt-5.5"
    # config["backend_url"] = "https://api.openai.com/v1"
    # config["openai_reasoning_effort"] = "xhigh"

    # # Use Anthropic (needs ANTHROPIC_API_KEY)
    # config["llm_provider"] = "anthropic"
    # config["deep_think_llm"] = "claude-opus-4-8"
    # config["quick_think_llm"] = "claude-sonnet-5"
    # config["backend_url"] = "https://api.anthropic.com/v1"
    # config["anthropic_effort"] = "high"

    # Common Settings
    config["temperature"] = 0.7
    config["checkpoint_enabled"] = True
    config["output_language"] = "zh-TW"
    config["max_debate_rounds"] = 5
    config["max_risk_discuss_rounds"] = 5
    config["max_recur_limit"] = 100
    config["news_article_limit"] = 20
    config["global_news_article_limit"] = 10
    config["global_news_lookback_days"] = 7
    config["llm_max_retries"] = 3

    selected_analysts = ["market", "social", "news", "fundamentals"]

    # Initialize with custom config
    ta = TradingAgentsGraph(selected_analysts=selected_analysts, config=config)

    # forward propagate
    today = datetime.date.today().strftime("%Y-%m-%d")
    _, decision = ta.propagate(company_name=stock, trade_date=today)

    # Memorize mistakes and reflect
    # ta.reflect_and_remember(1000) # parameter is the position returns
    return decision


if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    stocks = ["GOOG", "2330.TW", "NVDA", "SPCX"]
    for stock in stocks:
        decision = run_trading_agents(stock=stock)
        console.print(f"Decision for {stock}:\n{decision}")
