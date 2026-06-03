"""Per-token pricing lookup, replacing the runtime dependency on `litellm`.

Fetches the LiteLLM upstream price table
(https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)
on first use and memoizes it for the rest of the process. Returns
`(0.0, 0.0)` for unknown models — the reply footer then shows
`$0.00000000` instead of an estimate.
"""

from collections.abc import Mapping

import logfire
from pydantic import Field, BaseModel, ConfigDict, ValidationError
import requests


class ModelPriceEntry(BaseModel):
    """Subset of one LiteLLM price-table entry used by this bot."""

    model_config = ConfigDict(extra="ignore")

    input_cost_per_token: float = Field(default=0.0)
    output_cost_per_token: float = Field(default=0.0)
    supported_modalities: list[str] = Field(default=["text", "image"])


def _fetch_upstream() -> dict[str, object]:
    """Fetches the upstream price table; returns `{}` on network or parse error."""
    try:
        response = requests.get(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logfire.warn(f"Skipping model price fetch: {exc!s}")
        return {}
    return data if isinstance(data, dict) else {}


_price_table_cache: dict[str, dict[str, ModelPriceEntry]] = {}


def _load_model_prices() -> dict[str, ModelPriceEntry]:
    """Returns the validated price table, fetched once per process.

    A successful (non-empty) fetch is memoized for the rest of the process. A
    transient fetch failure yields an empty table that is deliberately NOT
    memoized, so a later lookup retries upstream instead of permanently
    degrading every model to the `{"text", "image"}` modality fallback.
    """
    cached = _price_table_cache.get("table")
    if cached:
        return cached
    prices: dict[str, ModelPriceEntry] = {}
    for name, entry in _fetch_upstream().items():
        if not isinstance(entry, Mapping):
            continue
        try:
            prices[name] = ModelPriceEntry.model_validate(obj=entry)
        except ValidationError as exc:
            logfire.warn(f"Skipping malformed model price entry {name}: {exc!s}")
    if prices:
        _price_table_cache["table"] = prices
    return prices


def get_token_rates(model_name: str) -> tuple[float, float]:
    """Returns `(input_cost_per_token, output_cost_per_token)` for `model_name`.

    Returns `(0.0, 0.0)` for unknown models so the reply footer shows
    `$0.00000000` instead of an estimate.

    Args:
        model_name: Model identifier to look up in the cached price table.

    Returns:
        Input and output token rates for the model.
    """
    model_info = _load_model_prices()
    info = model_info.get(model_name, ModelPriceEntry())
    return info.input_cost_per_token, info.output_cost_per_token


def warm_pricing_cache() -> None:
    """Populates the price-table cache eagerly.

    Calling this once at startup (off the event loop) keeps the first runtime
    `get_token_rates` / `get_supported_modalities` lookup from blocking on the
    upstream HTTP fetch during a live interaction.
    """
    if not _load_model_prices():
        logfire.warn(
            "Model pricing table is empty after the warm fetch; token-cost display and "
            "slow-model attachment modality detection fall back to defaults until a later "
            "lookup successfully refetches upstream."
        )


def get_supported_modalities(model_name: str) -> set[str]:
    """Returns the input modalities accepted by `model_name`.

    Reads `supported_modalities` from the cached LiteLLM price table. The
    field is unevenly populated upstream (Claude entries omit it entirely),
    so missing entries default to `{"text", "image"}` — the safe baseline
    that virtually every modern multimodal LLM accepts.

    Args:
        model_name: Model identifier to look up in the cached price table.

    Returns:
        Set of modality strings (e.g. `{"text", "image", "audio", "video"}`).
    """
    model_info = _load_model_prices()
    info = model_info.get(model_name, ModelPriceEntry())
    return set(info.supported_modalities)
