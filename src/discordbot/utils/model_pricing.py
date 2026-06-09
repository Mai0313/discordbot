"""LiteLLM model info lookup, replacing the runtime dependency on `litellm`.

Fetches the LiteLLM upstream price table
(https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)
on first use and memoizes it for the rest of the process. Returns
`(0.0, 0.0)` for unknown models, and the reply footer then shows
`$0.00000000` instead of an estimate.
"""

from functools import cache
from collections.abc import Mapping

import logfire
from pydantic import Field, BaseModel, ConfigDict, ValidationError
import requests

MODEL_INFO_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)


class ModelPriceEntry(BaseModel):
    """Subset of one LiteLLM price table entry used by this bot."""

    model_config = ConfigDict(extra="ignore")

    input_cost_per_token: float = Field(default=0.0)
    output_cost_per_token: float = Field(default=0.0)
    supported_modalities: list[str] = Field(default=["text", "image"])


@cache
def load_model_info() -> dict[str, ModelPriceEntry]:
    """Returns the validated LiteLLM model info table, fetched once per process."""
    try:
        response = requests.get(url=MODEL_INFO_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logfire.warn(f"Skipping model price fetch: {exc!s}")
        return {}

    prices: dict[str, ModelPriceEntry] = {}
    if not isinstance(data, dict):
        return prices

    for name, entry in data.items():
        if not isinstance(entry, Mapping):
            continue
        try:
            prices[name] = ModelPriceEntry.model_validate(obj=entry)
        except ValidationError as exc:
            logfire.warn(f"Skipping malformed model price entry {name}: {exc!s}")
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
    model_info = load_model_info()
    info = model_info.get(model_name, ModelPriceEntry())
    return info.input_cost_per_token, info.output_cost_per_token


def get_supported_modalities(model_name: str) -> set[str]:
    """Returns the input modalities accepted by `model_name`.

    Reads `supported_modalities` from the cached LiteLLM price table. The
    field is unevenly populated upstream (Claude entries omit it entirely),
    so missing entries default to `{"text", "image"}`, the safe baseline
    that virtually every modern multimodal LLM accepts.

    Args:
        model_name: Model identifier to look up in the cached price table.

    Returns:
        Set of modality strings (e.g. `{"text", "image", "audio", "video"}`).
    """
    model_info = load_model_info()
    info = model_info.get(model_name, ModelPriceEntry())
    return set(info.supported_modalities)


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    model_name = "gemini-pro-latest"
    model_info = load_model_info()
    console.print(model_info)
    supported_modalities = get_supported_modalities(model_name=model_name)
    console.print(supported_modalities)
    token_rates = get_token_rates(model_name=model_name)
    console.print(token_rates)
