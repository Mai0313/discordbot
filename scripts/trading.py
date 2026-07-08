"""Standalone TradingAgents run for GOOG with a self-contained config.

Every run setting that used to live in .env is inlined here, so .env only needs
to hold API keys (e.g. GOOGLE_API_KEY). That makes this script portable: copy
it, tweak it into other states, or call it from elsewhere without depending on
a particular .env. Provider is Gemini by default; switch by swapping the active
block below for one of the commented ones.

These values reproduce the interactive `uv run tradingagents` run for
GOOG + today + all analysts, with the model/language/thinking choices your .env
was pinning moved in here.
"""

import datetime

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


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
    config["global_news_article_limit"] = 100
    config["global_news_lookback_days"] = 30
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
