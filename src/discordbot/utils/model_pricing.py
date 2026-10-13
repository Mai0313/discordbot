"""LiteLLM model info lookup, replacing the runtime dependency on `litellm`.

Fetches the LiteLLM upstream price table
(https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)
on first use, mirrors what it parsed to `MODEL_INFO_CACHE_PATH`, and holds it for a day
before fetching it again. Fetching live is the point: the rates track a maintained upstream
instead of a hardcoded table that rots.

A lookup never raises. The table only feeds a cosmetic cost estimate and the attachment
modality gate, so an unreachable upstream degrades to the mirror and then to an empty
table, where every model is unknown: no rates, so a usage footer with no cost, and the
modality defaults in `get_supported_modalities`, which keep a Gemini model's audio and
video and drop only those on any other model instead of the whole message's attachments.

A degrade is not permanent either: the process remembers when upstream served the table
it holds, and `refresh_model_info` re-checks upstream while it never has, and again once
that table is a day old. It is driven off the event loop every `MODEL_INFO_REFRESH_MINUTES`,
so neither recovery nor the daily refetch costs a lookup anything or needs a restart.
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
# Half an hour is far inside the weeks a process can outlive an outage by, and it keeps the
# fetch warning a total outage emits per attempt readable.
MODEL_INFO_REFRESH_MINUTES = 30.0
# A day is how long a model upstream has just listed goes without its rates and modalities.
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


# Only the swap of the held table happens under this lock, never a fetch: a load runs
# outside it so two callers that miss together do not wait out each other's timeout.
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
            # Per entry, not per table: upstream is a community-maintained file, and one
            # retyped row must not cost the rest their rates and modalities.
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

    The temp name is unique because a load runs outside the lock that swaps the held table
    in: two threads that miss together both run this, and a shared temp path lets one
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


def _load_upstream() -> tuple[dict[str, ModelPriceEntry] | None, str]:
    """Returns the upstream price table, and how upstream answered.

    Only a payload that yielded at least one entry is mirrored, so neither an upstream
    shape change nor an empty answer can overwrite the last good table.

    Returns:
        The parsed table and `served`, or None and the reason upstream did not serve one.
    """
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

    # The one line saying this happened, and the only one an operator can act on: a reply
    # still goes out, so nothing else about the process looks wrong from the outside.
    logfire.error(
        "no model price table available; every model reads as unknown",
        upstream=upstream,
        mirror=mirror,
        url=MODEL_INFO_URL,
        path=str(MODEL_INFO_CACHE_PATH),
    )
    return _LoadedTable(prices={}, served_at=None)


def load_model_info() -> dict[str, ModelPriceEntry]:
    """Returns the validated LiteLLM model info table, held between refreshes.

    The first call takes the whole chain and holds whatever it got; every later one reads
    that back, so an outage costs one timeout for the life of the process rather than one
    per call. `refresh_model_info` replaces a held table upstream did not serve once
    upstream is reachable again, and one it did serve once a day, so neither a degrade nor
    a newly listed model waits for a restart.

    Returns:
        Model name to price entry, empty when no table could be loaded at all.
    """
    global _LOADED_TABLE  # noqa: PLW0603 -- the process-wide price table

    with _LOADED_TABLE_LOCK:
        held = _LOADED_TABLE
    if held is not None:
        return held.prices

    loaded = _load_table()
    with _LOADED_TABLE_LOCK:
        # A refresh may have installed an upstream table while this call was loading, and
        # the mirror it fell back to must not replace it.
        if _LOADED_TABLE is None:
            _LOADED_TABLE = loaded
        return _LOADED_TABLE.prices


def refresh_model_info() -> None:
    """Loads the price table, or re-checks upstream while what is held is missing or stale.

    A caller outside this module runs this off the event loop, so no RETRY is ever paid on
    a lookup's path. The first load still falls to whichever caller reaches it first, which
    is why `_write_mirror` needs a unique temp name. A table upstream did not serve is
    re-checked on every call, and one it did serve once it is `MODEL_INFO_MAX_AGE_HOURS`
    old, so a model upstream lists after this process started gains its rates and
    modalities without a restart.

    A retry that fails keeps what is held: the table a process is serving is worth more
    than the empty one re-running the whole chain would install over it once the mirror
    file goes away.
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


def get_token_rates(model_name: str) -> tuple[float, float] | None:
    """Returns `(input_cost_per_token, output_cost_per_token)` for `model_name`.

    An unknown model has no rates rather than zero ones: a listed rate of zero is a free
    model, and folding the two together would price every unknown model as free.

    Args:
        model_name: Model identifier to look up in the cached price table.

    Returns:
        Input and output token rates for the model, or None when the table has no entry for it.
    """
    model_info = load_model_info()
    info = model_info.get(model_name)
    if info is None:
        return None
    return info.input_cost_per_token, info.output_cost_per_token


def get_supported_modalities(model_name: str) -> set[str]:
    """Returns the input modalities accepted by `model_name`.

    Reads `supported_modalities` from the cached LiteLLM price table. The
    field is unevenly populated upstream and a new model can be missing from
    the table for days, so where the table does not say, a Gemini model
    defaults to text, image, audio and video, which every Gemini answer
    model accepts, and any other to `{"text", "image"}`, the safe baseline
    that virtually every modern multimodal LLM accepts.

    Args:
        model_name: Model identifier to look up in the cached price table.

    Returns:
        Set of modality strings (e.g. `{"text", "image", "audio", "video"}`).
    """
    info = load_model_info().get(model_name)
    if info is not None and info.supported_modalities is not None:
        return set(info.supported_modalities)
    if "gemini" in model_name:
        return {"text", "image", "audio", "video"}
    return {"text", "image"}
