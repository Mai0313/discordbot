"""Per-token prices and accepted input modalities for a model name, read from LiteLLM's table.

The runtime backend is LiteLLM Proxy, but nothing under `src/` imports the `litellm` package;
the only thing this bot wants from that project is its published price table. So the table is
fetched straight from the upstream repository on first use and memoized for the rest of the
process, and rates are never hardcoded anywhere else.

Two unrelated consumers read it. The usage footer prices a finished reply, and the attachment
gate asks which modalities the answer model accepts before anything is uploaded
(`gen_reply/input.py`). The footer is priced from two different cogs (`gen_reply/streaming.py`
and `research/cog.py`), and no cog may import another, which is why the lookup sits in `utils/`
rather than inside one of them.

What it promises is an estimate that degrades quietly on a *missing* model: an entry the table
does not carry answers `(0.0, 0.0)`, so the footer shows `$0.00000000` instead of a price
borrowed from some other model, and an entry with no `supported_modalities` answers a baseline
set. What it deliberately does not do is hide a broken fetch. The first lookup in a process is a
blocking HTTP call whose failure reaches the caller, which is why `cli.py` warms the cache off
the event loop at startup and `gen_reply/input.py` handles a cold-start failure of its own.
"""

from typing import Any
from functools import cache

from pydantic import Field, BaseModel, ConfigDict
import requests

MODEL_INFO_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)


class ModelPriceEntry(BaseModel):
    """The prices and accepted modalities this bot reads out of one LiteLLM price table entry.

    `extra="ignore"` is already pydantic's default; it is pinned here so that a later switch to
    `extra="forbid"` has to be a deliberate decision rather than a tidy-up, because an upstream
    entry carries dozens of other fields and grows new ones, and none of them should be able to
    fail a load. Every field defaults, so a bare `ModelPriceEntry()` doubles as the answer for a
    model the table does not list.
    """

    model_config = ConfigDict(extra="ignore")

    input_cost_per_token: float = Field(
        default=0.0, description="Per-token input price in USD; 0.0 when the model is unknown."
    )
    output_cost_per_token: float = Field(
        default=0.0, description="Per-token output price in USD; 0.0 when the model is unknown."
    )
    supported_modalities: list[str] = Field(
        default=["text", "image"],
        description="Input modalities the model accepts; defaults to text and image.",
    )


@cache
def load_model_info() -> dict[str, ModelPriceEntry]:
    """Fetches and validates the whole LiteLLM price table, once per process.

    A blocking HTTP call on the calling thread, so `cli.py` warms it through
    `asyncio.to_thread` at startup rather than letting the first AI reply pay for it on the
    event loop. `@cache` memoizes only a successful load, so an unreachable table (or one entry
    that will not validate, which loses the whole load) is retried by the next caller instead
    of the failure being memoized for the life of the process.

    Both failures leave here uncaught: a table that cannot be fetched, answers a non-2xx status
    or does not decode as JSON surfaces as a `requests.RequestException` (a `Timeout` at the 5s
    bound being the routine one), and an entry this model cannot parse as a
    `pydantic.ValidationError`. Neither gets a `Raises:` section only because DOC502 reserves
    that for an exception raised in this body.

    Returns:
        Every model name in the table mapped to its parsed entry.
    """
    prices: dict[str, ModelPriceEntry] = {}
    response = requests.get(url=MODEL_INFO_URL, timeout=5)
    response.raise_for_status()
    data_dict: dict[str, dict[str, Any]] = response.json()

    for name, entry in data_dict.items():
        prices[name] = ModelPriceEntry(**entry)
    return prices


def get_token_rates(model_name: str) -> tuple[float, float]:
    """Looks up the per-token input and output prices for one model.

    A model the table does not list answers `(0.0, 0.0)`, so the reply footer shows
    `$0.00000000` rather than a price borrowed from another entry. The first call in a process
    pays for the table fetch and propagates its failure; see `load_model_info`.

    Args:
        model_name (str): Model identifier as LiteLLM spells it in the table.

    Returns:
        `(input_cost_per_token, output_cost_per_token)`, in USD.
    """
    model_info = load_model_info()
    info = model_info.get(model_name, ModelPriceEntry())
    return info.input_cost_per_token, info.output_cost_per_token


def get_supported_modalities(model_name: str) -> set[str]:
    """Looks up the input modalities one model accepts.

    Upstream populates `supported_modalities` unevenly (Claude entries omit it entirely), so an
    absent field and an unlisted model both answer `{"text", "image"}`, the baseline virtually
    every modern multimodal LLM accepts. The fallback has to be that set rather than an empty
    one because `gen_reply/input.py` drops an attachment whose modality is missing from it, and
    an empty answer would silently drop every attachment for a model the table has not caught
    up with. The first call in a process pays for the table fetch and propagates its failure;
    see `load_model_info`.

    Args:
        model_name (str): Model identifier as LiteLLM spells it in the table.

    Returns:
        Modality strings such as `{"text", "image", "audio", "video"}`.
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
