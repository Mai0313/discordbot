"""Per-token pricing lookup, replacing the runtime dependency on `litellm`.

Mirrors what `litellm.model_cost` was doing for us: pulls the same upstream
JSON (https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)
once per process, caches the parsed dict in memory, and stashes a copy at
`data/model_prices.json` so a subsequent restart can degrade gracefully
when the network is down. Returns `(0.0, 0.0)` for unknown models — the
reply footer then shows `$0.00000000` instead of an estimate.
"""

import json
from typing import cast
from pathlib import Path
from functools import cache
import contextlib
from collections.abc import Mapping

import logfire
from pydantic import Field, BaseModel, ConfigDict, ValidationError, field_validator
import requests

_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
_CACHE_PATH = Path("./data/model_prices.json")

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonRecord = dict[str, JsonValue]
type JsonMapping = Mapping[str, JsonValue]
type SupportedModalitiesInput = JsonValue | tuple[JsonValue, ...] | set[JsonScalar]


class ModelPriceEntry(BaseModel):
    """Subset of one LiteLLM price-table entry used by this bot."""

    model_config = ConfigDict(extra="ignore")

    input_cost_per_token: float = 0.0
    input_cost_per_token_priority: float | None = None
    output_cost_per_token: float = 0.0
    output_cost_per_token_priority: float | None = None
    supported_modalities: list[str] = Field(default_factory=list)

    @field_validator("supported_modalities", mode="before")
    @classmethod
    def _coerce_supported_modalities(cls, value: SupportedModalitiesInput) -> list[str]:
        """Normalizes uneven upstream modality metadata into strings."""
        if not value:
            return []
        if isinstance(value, list | tuple | set):
            return [item for item in value if isinstance(item, str)]
        return []


def _fetch_upstream(timeout: int = 5) -> JsonRecord:
    """Fetches the upstream price table; raises on network or parse errors."""
    response = requests.get(url=_UPSTREAM_URL, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not data:
        raise ValueError("Upstream price table is empty or malformed")
    return cast("JsonRecord", data)


def _load_disk_cache() -> JsonRecord:
    """Loads the previous-run snapshot, or returns `{}` if unavailable."""
    if not _CACHE_PATH.is_file():
        return {}
    try:
        with _CACHE_PATH.open(encoding="utf-8") as f:
            data = json.load(fp=f)
    except (OSError, json.JSONDecodeError):
        return {}
    return cast("JsonRecord", data) if isinstance(data, dict) else {}


def _save_disk_cache(data: JsonMapping) -> None:
    """Best-effort write of the freshly fetched table to the on-disk cache."""
    with contextlib.suppress(OSError):
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(data=json.dumps(obj=data), encoding="utf-8")


def _parse_model_prices(data: JsonMapping) -> dict[str, ModelPriceEntry]:
    """Validates raw price-table data into the subset this bot reads."""
    prices: dict[str, ModelPriceEntry] = {}
    for model_name, raw_entry in data.items():
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            prices[model_name] = ModelPriceEntry.model_validate(obj=raw_entry)
        except ValidationError as exc:
            logfire.warn(f"Skipping malformed model price entry {model_name}: {exc!s}")
    return prices


@cache
def _load_model_prices() -> dict[str, ModelPriceEntry]:
    """Returns the price table, fetching once and falling back to disk cache."""
    try:
        raw_data = _fetch_upstream()
        data = _parse_model_prices(data=raw_data)
        if not data:
            raise ValueError("Upstream price table has no usable entries")
    except (requests.RequestException, ValueError) as exc:
        logfire.warn(f"Falling back to disk cache for model prices: {exc!s}")
        return _parse_model_prices(data=_load_disk_cache())
    _save_disk_cache(data=raw_data)
    return data


def get_token_rates(model_name: str) -> tuple[float, float]:
    """Returns `(input_cost_per_token, output_cost_per_token)` for `model_name`.

    Prefers the priority-tier rates when present (e.g. Gemini's burst pricing
    via the `*_priority` suffix). Returns `(0.0, 0.0)` for unknown models
    so the reply footer shows `$0.00000000` instead of an estimate.

    Args:
        model_name: Model identifier to look up in the cached price table.

    Returns:
        Input and output token rates for the model.
    """
    info = _load_model_prices().get(model_name)
    if info is None:
        return 0.0, 0.0
    default_input = info.input_cost_per_token
    input_rate = info.input_cost_per_token_priority
    if input_rate is None:
        input_rate = default_input
    default_output = info.output_cost_per_token
    output_rate = info.output_cost_per_token_priority
    if output_rate is None:
        output_rate = default_output
    return float(input_rate), float(output_rate)


def get_supported_modalities(model_name: str) -> set[str]:
    """Returns the input modalities accepted by `model_name`.

    Reads `supported_modalities` from the cached LiteLLM price table. The
    field is unevenly populated upstream (Claude entries omit it entirely,
    some Gemini variants only set the per-modality booleans), so when it is
    missing or empty we default to `{"text", "image"}` — the safe baseline
    that virtually every modern multimodal LLM accepts.

    Args:
        model_name: Model identifier to look up in the cached price table.

    Returns:
        Set of modality strings (e.g. `{"text", "image", "audio", "video"}`).
    """
    info = _load_model_prices().get(model_name)
    if info is None:
        return {"text", "image"}
    modalities = info.supported_modalities
    if not modalities:
        return {"text", "image"}
    return set(modalities)
