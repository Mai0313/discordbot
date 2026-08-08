"""LiteLLM model info lookup, replacing the runtime dependency on `litellm`.

Fetches the LiteLLM upstream price table
(https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)
on first use, mirrors what it parsed to `MODEL_INFO_CACHE_PATH`, and memoizes it for the
rest of the process. Fetching live is the point: the rates track a maintained upstream
instead of a hardcoded table that rots.

A lookup never raises. The table only feeds a cosmetic cost estimate and the attachment
modality gate, so an unreachable upstream degrades to the mirror and then to an empty
table, where every model is unknown: `(0.0, 0.0)` rates and a `$0.00000000` footer, and
the `{"text", "image"}` modality baseline, which keeps text, images and documents flowing
and drops only audio and video instead of the whole message's attachments.
"""

import json
from typing import Any
from pathlib import Path
import secrets
from functools import cache

import logfire
from pydantic import Field, BaseModel, ConfigDict, ValidationError
import requests

MODEL_INFO_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
MODEL_INFO_CACHE_PATH = Path("./data/model_prices_and_context_window.json")


class ModelPriceEntry(BaseModel):
    """Subset of one LiteLLM price table entry used by this bot."""

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
            # Per entry, not per table: upstream is a 3000-row community file, and one
            # retyped row must not cost the other 2999 their rates and modalities.
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
        response = requests.get(url=MODEL_INFO_URL, timeout=5)
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
        # `errors="replace"` rather than a wider except: a corrupt mirror raises
        # UnicodeDecodeError, which is a ValueError and would sail past `except OSError`
        # into the caller. Replaced bytes just make `_decode_table` reject the payload.
        return MODEL_INFO_CACHE_PATH.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        # Nothing has been mirrored yet; the caller reports the outcome once it knows it.
        return None
    except OSError as exc:
        logfire.warn(
            "model price mirror is unreadable", path=str(MODEL_INFO_CACHE_PATH), _exc_info=exc
        )
        return None


def _write_mirror(payload: str) -> None:
    """Mirrors a usable price table so a later start without upstream still has rates.

    The temp name is unique because `cache` does not serialize the body it memoizes: two
    threads that miss together (`cli.py`'s warm-up in `asyncio.to_thread` and a reply
    finalizing its footer on the loop) both run this, and a shared temp path lets one
    truncate the other's bytes into the mirror the next outage depends on.
    """
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


@cache
def load_model_info() -> dict[str, ModelPriceEntry]:
    """Returns the validated LiteLLM model info table, loaded once per process.

    Upstream is fetched first so the rates stay current, and only a payload that yielded
    at least one entry is mirrored, so neither an upstream shape change nor an empty
    answer can overwrite the last good table. A fetch or a payload that fails that bar
    serves the mirror instead; with neither the table is empty and every model is unknown.

    `cache` memoizes the degrade too, so an outage costs one timeout for the life of
    the process rather than one per call. A restart is the retry.

    Returns:
        Model name to price entry, empty when no table could be loaded at all.
    """
    payload = _fetch_table()
    upstream = "unreachable"
    if payload is not None:
        upstream = "unusable"
        prices = _decode_table(payload=payload, source=MODEL_INFO_URL)
        if prices:
            _write_mirror(payload=payload)
            return prices

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
            return prices

    # The one line saying this happened, and the only one an operator can act on: a reply
    # still goes out, so nothing else about the process looks wrong from the outside.
    logfire.error(
        "no model price table available; every model reads as unknown",
        upstream=upstream,
        mirror=mirror,
        url=MODEL_INFO_URL,
        path=str(MODEL_INFO_CACHE_PATH),
    )
    return {}


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
