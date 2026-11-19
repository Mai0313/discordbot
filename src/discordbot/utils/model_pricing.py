"""Cached LiteLLM model pricing and input-modality lookup.

Fetches LiteLLM's upstream price table on first use, mirrors usable tables to
`MODEL_INFO_CACHE_PATH`, and refreshes via `refresh_model_info` while degraded or past
`MODEL_INFO_MAX_AGE_HOURS`. Lookups never raise: an unavailable upstream degrades to the mirror
and then to an empty table (unknown rates and default modalities).
"""

import json
from time import monotonic
from typing import Any
from pathlib import Path
import secrets
import threading

import logfire
from pydantic import Field, BaseModel, ConfigDict, ValidationError
import requests

from discordbot.typings.timeouts import PRICE_TABLE_FETCH_TIMEOUT_SECONDS

MODEL_INFO_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
MODEL_INFO_CACHE_PATH = Path("./data/model_prices_and_context_window.json")
MODEL_INFO_REFRESH_MINUTES = 30.0
MODEL_INFO_MAX_AGE_HOURS = 24.0


class ModelPriceEntry(BaseModel):
    """Subset of one LiteLLM price table entry used by this bot."""

    model_config = ConfigDict(extra="ignore")

    input_cost_per_token: float = Field(
        default=0.0, description="Per-token input price in USD; 0.0 when the entry lists none."
    )
    output_cost_per_token: float = Field(
        default=0.0, description="Per-token output price in USD; 0.0 when the entry lists none."
    )
    supported_modalities: list[str] | None = Field(
        default=None,
        description="Input modalities the model accepts; None when the entry lists none.",
    )


class _LoadedTable(BaseModel):
    """The price table this process holds, and when upstream served it."""

    prices: dict[str, ModelPriceEntry] = Field(
        ..., description="Model name to price entry; empty when no table could be loaded."
    )
    served_at: float | None = Field(
        ...,
        description=(
            "`monotonic()` reading when upstream served this table; None when it did not, "
            "which is when recovery is still owed."
        ),
    )


# Guards table swaps only; network fetches run outside the lock.
_LOADED_TABLE_LOCK = threading.Lock()
_LOADED_TABLE: _LoadedTable | None = None


def _decode_table(payload: str, source: str) -> dict[str, ModelPriceEntry] | None:
    """Returns the price table `payload` holds, or None when it cannot be read as one."""
    try:
        data_dict: dict[str, dict[str, Any]] = json.loads(s=payload)
        entries = data_dict.items()
    except Exception as exc:
        # Broad on purpose: this payload is whatever upstream served or whatever the mirror
        # holds, and every way it can fail to be a table at all degrades the same way.
        logfire.warn(
            "model price table is unreadable",
            source=source,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        return None

    prices: dict[str, ModelPriceEntry] = {}
    dropped: list[str] = []
    for name, entry in entries:
        try:
            prices[name] = ModelPriceEntry(**entry)
        except (TypeError, ValidationError):
            # Drop malformed rows individually so one bad upstream entry leaves the rest intact.
            dropped.append(name)
    if dropped:
        logfire.warn(
            "model price table has unreadable entries",
            source=source,
            dropped=len(dropped),
            kept=len(prices),
        )
    return prices


def _fetch_table() -> str | None:
    """Returns the upstream price table as raw text, or None when the fetch fails."""
    try:
        response = requests.get(url=MODEL_INFO_URL, timeout=PRICE_TABLE_FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        logfire.warn(
            "model price table fetch failed",
            url=MODEL_INFO_URL,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
        return None


def _read_mirror() -> str | None:
    """Returns the mirrored price table as raw text, or None when there is no readable one."""
    try:
        # `errors="replace"` avoids UnicodeDecodeError bypassing except OSError.
        return MODEL_INFO_CACHE_PATH.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logfire.warn(
            "model price mirror is unreadable", path=str(MODEL_INFO_CACHE_PATH), _exc_info=exc
        )
        return None


def _write_mirror(payload: str) -> None:
    """Atomically mirrors `payload` to `MODEL_INFO_CACHE_PATH` via a unique temp file."""
    # Unique temp name prevents concurrent unlocked cold-start fetches from clobbering each other.
    tmp_path = MODEL_INFO_CACHE_PATH.with_name(
        f"{MODEL_INFO_CACHE_PATH.name}.{secrets.token_urlsafe(8)}.tmp"
    )
    try:
        MODEL_INFO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(data=payload, encoding="utf-8")
        tmp_path.replace(target=MODEL_INFO_CACHE_PATH)
    except OSError as exc:
        logfire.warn(
            "model price mirror write failed", path=str(MODEL_INFO_CACHE_PATH), _exc_info=exc
        )


def _load_upstream() -> tuple[dict[str, ModelPriceEntry] | None, str]:
    """Returns `(prices, "served")` from upstream (mirroring non-empty), or `(None, reason)`."""
    payload = _fetch_table()
    if payload is None:
        return None, "unreachable"
    prices = _decode_table(payload=payload, source=MODEL_INFO_URL)
    if not prices:
        return None, "unusable"
    _write_mirror(payload=payload)
    return prices, "served"


def _load_table() -> _LoadedTable:
    """Returns the best table there is: upstream, else the mirror, else an empty one."""
    prices, upstream = _load_upstream()
    if prices:
        return _LoadedTable(prices=prices, served_at=monotonic())

    mirrored = _read_mirror()
    mirror = "absent"
    if mirrored is not None:
        mirror = "unusable"
        prices = _decode_table(payload=mirrored, source=str(MODEL_INFO_CACHE_PATH))
        if prices:
            logfire.info(
                "serving the mirrored model price table",
                path=str(MODEL_INFO_CACHE_PATH),
                entries=len(prices),
            )
            return _LoadedTable(prices=prices, served_at=None)

    # Log at error since replies still succeed without a price table.
    logfire.error(
        "no model price table available; every model reads as unknown",
        upstream=upstream,
        mirror=mirror,
        url=MODEL_INFO_URL,
        path=str(MODEL_INFO_CACHE_PATH),
    )
    return _LoadedTable(prices={}, served_at=None)


def load_model_info() -> dict[str, ModelPriceEntry]:
    """Returns the cached LiteLLM model info table, or an empty dict if none could be loaded.

    Loads once on first call (upstream, mirror, then empty) and serves the held table until
    `refresh_model_info` replaces it.
    """
    global _LOADED_TABLE  # noqa: PLW0603 -- the process-wide price table

    with _LOADED_TABLE_LOCK:
        held = _LOADED_TABLE
    if held is not None:
        return held.prices

    loaded = _load_table()
    with _LOADED_TABLE_LOCK:
        # Keep an upstream table installed by a concurrent refresh instead of overwriting it.
        if _LOADED_TABLE is None:
            _LOADED_TABLE = loaded
        return _LOADED_TABLE.prices


def refresh_model_info() -> None:
    """Loads the price table if unset, or re-fetches upstream when degraded or past max age.

    Retains the currently held table if a refresh attempt fails.
    """
    global _LOADED_TABLE  # noqa: PLW0603 -- the process-wide price table

    with _LOADED_TABLE_LOCK:
        held = _LOADED_TABLE
    if held is None:
        load_model_info()
        return
    if (
        held.served_at is not None
        and monotonic() - held.served_at < MODEL_INFO_MAX_AGE_HOURS * 3600
    ):
        return

    prices, _ = _load_upstream()
    if not prices:
        return
    with _LOADED_TABLE_LOCK:
        _LOADED_TABLE = _LoadedTable(prices=prices, served_at=monotonic())
    if held.served_at is None:
        logfire.info("recovered the upstream model price table", entries=len(prices))
    else:
        logfire.info("refreshed the upstream model price table", entries=len(prices))


def _price_entry(model_name: str) -> ModelPriceEntry | None:
    """Returns the table entry for `model_name`, checking `gemini/` before the bare name."""
    model_info = load_model_info()
    # Prefer gemini/ (Developer API) over bare name (Vertex); never try reseller prefixes.
    return model_info.get(f"gemini/{model_name}") or model_info.get(model_name)


def get_token_rates(model_name: str) -> tuple[float, float] | None:
    """Returns `(input_cost_per_token, output_cost_per_token)` for `model_name`, or None if unknown.

    Returns None rather than `(0.0, 0.0)` for unknown models to distinguish them from free models.

    Args:
        model_name (str): Model identifier to look up in the cached price table.
    """
    info = _price_entry(model_name=model_name)
    if info is None:
        return None
    return info.input_cost_per_token, info.output_cost_per_token


def get_supported_modalities(model_name: str) -> set[str]:
    """Returns the input modality strings accepted by `model_name`.

    Reads `supported_modalities` from the cached LiteLLM price table. Where the table does
    not say, a Gemini model defaults to `{"text", "image", "audio", "video"}` and any other
    model defaults to `{"text", "image"}`.

    Args:
        model_name (str): Model identifier to look up in the cached price table.
    """
    info = _price_entry(model_name=model_name)
    if info is not None and info.supported_modalities is not None:
        return set(info.supported_modalities)
    if "gemini" in model_name:
        return {"text", "image", "audio", "video"}
    return {"text", "image"}
