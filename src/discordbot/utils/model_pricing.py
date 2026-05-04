"""Per-token pricing lookup, replacing the runtime dependency on ``litellm``.

Mirrors what ``litellm.model_cost`` was doing for us: pulls the same upstream
JSON (https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)
once per process, caches the parsed dict in memory, and stashes a copy at
``data/model_prices.json`` so a subsequent restart can degrade gracefully
when the network is down. Returns ``(0.0, 0.0)`` for unknown models — the
reply footer then shows ``$0.00000000`` instead of an estimate.
"""

import json
from typing import Any
from pathlib import Path
from functools import cache
import contextlib

import logfire
import requests

_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
_CACHE_PATH = Path("./data/model_prices.json")


def _fetch_upstream(timeout: int = 5) -> dict[str, dict[str, Any]]:
    """Fetches the upstream price table; raises on network or parse errors."""
    response = requests.get(url=_UPSTREAM_URL, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not data:
        raise ValueError("Upstream price table is empty or malformed")
    return data


def _load_disk_cache() -> dict[str, dict[str, Any]]:
    """Loads the previous-run snapshot, or returns ``{}`` if unavailable."""
    if not _CACHE_PATH.is_file():
        return {}
    try:
        with _CACHE_PATH.open(encoding="utf-8") as f:
            data = json.load(fp=f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_disk_cache(data: dict[str, dict[str, Any]]) -> None:
    """Best-effort write of the freshly fetched table to the on-disk cache."""
    with contextlib.suppress(OSError):
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(data=json.dumps(obj=data), encoding="utf-8")


@cache
def _load_model_prices() -> dict[str, dict[str, Any]]:
    """Returns the price table, fetching once and falling back to disk cache."""
    try:
        data = _fetch_upstream()
    except (requests.RequestException, ValueError) as exc:
        logfire.warn(f"Falling back to disk cache for model prices: {exc!s}")
        return _load_disk_cache()
    _save_disk_cache(data=data)
    return data


def get_token_rates(model_name: str) -> tuple[float, float]:
    """Returns ``(input_cost_per_token, output_cost_per_token)`` for ``model_name``.

    Prefers the priority-tier rates when present (e.g. Gemini's burst pricing
    via the ``*_priority`` suffix). Returns ``(0.0, 0.0)`` for unknown models
    so the reply footer shows ``$0.00000000`` instead of an estimate.
    """
    info = _load_model_prices().get(model_name) or {}
    default_input = info.get("input_cost_per_token", 0)
    input_rate = info.get("input_cost_per_token_priority", default_input)
    default_output = info.get("output_cost_per_token", 0)
    output_rate = info.get("output_cost_per_token_priority", default_output)
    return float(input_rate), float(output_rate)
