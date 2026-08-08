"""Tests for the LiteLLM price-table load: the disk mirror and the never-raises degrade."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from pathlib import Path

import pytest
import requests

from discordbot.utils.model_pricing import (
    MODEL_INFO_URL,
    _write_mirror,
    get_token_rates,
    load_model_info,
    get_supported_modalities,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_MODEL = "gemini-pro-latest"
_TABLE = {
    _MODEL: {
        "input_cost_per_token": 1.0,
        "output_cost_per_token": 2.0,
        "supported_modalities": ["text", "image", "video"],
    }
}


@pytest.fixture(autouse=True)
def price_cache_isolated() -> Iterator[None]:
    """Keeps this module's fake tables out of the process-wide loader cache."""
    load_model_info.cache_clear()
    yield
    load_model_info.cache_clear()


class FakeResponse:
    """The two members `_fetch_table` reads off a `requests` response."""

    def __init__(self, text: str) -> None:
        """Stores the body upstream answered with."""
        self.text = text

    def raise_for_status(self) -> None:
        """Upstream answered 200, so there is nothing to raise."""


def _serve(monkeypatch: pytest.MonkeyPatch, payload: str) -> list[str]:
    """Answers every fetch with `payload`; returns the log of requested urls."""
    calls: list[str] = []

    def fake_get(url: str, timeout: int) -> FakeResponse:
        """Stands in for `requests.get` against a healthy upstream."""
        del timeout
        calls.append(url)
        return FakeResponse(text=payload)

    monkeypatch.setattr("discordbot.utils.model_pricing.requests.get", fake_get)
    return calls


def _refuse(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fails every fetch the way an unreachable upstream does; returns the url log."""
    calls: list[str] = []

    def fake_get(url: str, timeout: int) -> FakeResponse:
        """Stands in for `requests.get` with no route to the host."""
        del timeout
        calls.append(url)
        raise requests.ConnectionError("name or service not known")

    monkeypatch.setattr("discordbot.utils.model_pricing.requests.get", fake_get)
    return calls


def test_a_fetched_price_table_is_mirrored_to_disk(
    monkeypatch: pytest.MonkeyPatch, model_price_mirror_isolated: Path
) -> None:
    """The mirror is written from the fetch, so a later start without upstream has rates."""
    _serve(monkeypatch=monkeypatch, payload=json.dumps(obj=_TABLE))

    assert get_token_rates(model_name=_MODEL) == (1.0, 2.0)
    assert json.loads(s=model_price_mirror_isolated.read_text(encoding="utf-8")) == _TABLE


def test_a_failed_fetch_serves_the_mirrored_table(
    monkeypatch: pytest.MonkeyPatch, model_price_mirror_isolated: Path
) -> None:
    """An unreachable upstream costs nothing while a mirror is on disk."""
    model_price_mirror_isolated.write_text(json.dumps(obj=_TABLE), encoding="utf-8")
    _refuse(monkeypatch=monkeypatch)

    assert get_token_rates(model_name=_MODEL) == (1.0, 2.0)
    assert get_supported_modalities(model_name=_MODEL) == {"text", "image", "video"}


def test_an_unreadable_upstream_payload_keeps_the_mirror(
    monkeypatch: pytest.MonkeyPatch, model_price_mirror_isolated: Path
) -> None:
    """An upstream that answers 200 with something else must not become the new mirror."""
    model_price_mirror_isolated.write_text(json.dumps(obj=_TABLE), encoding="utf-8")
    _serve(monkeypatch=monkeypatch, payload="<html>404: Not Found</html>")

    assert get_token_rates(model_name=_MODEL) == (1.0, 2.0)
    assert json.loads(s=model_price_mirror_isolated.read_text(encoding="utf-8")) == _TABLE


def test_an_empty_upstream_answer_keeps_the_mirror(
    monkeypatch: pytest.MonkeyPatch, model_price_mirror_isolated: Path
) -> None:
    """`{}` parses, so only the yielded-an-entry bar stops it destroying the last good table."""
    model_price_mirror_isolated.write_text(json.dumps(obj=_TABLE), encoding="utf-8")
    _serve(monkeypatch=monkeypatch, payload="{}")

    assert get_token_rates(model_name=_MODEL) == (1.0, 2.0)
    assert json.loads(s=model_price_mirror_isolated.read_text(encoding="utf-8")) == _TABLE


def test_a_non_utf8_mirror_degrades_like_an_absent_one(
    monkeypatch: pytest.MonkeyPatch, model_price_mirror_isolated: Path
) -> None:
    """A corrupt mirror raises UnicodeDecodeError, a ValueError that `except OSError` misses."""
    model_price_mirror_isolated.write_bytes(b'\xff\xfe{"m": {}}')
    _refuse(monkeypatch=monkeypatch)

    assert load_model_info() == {}
    assert get_token_rates(model_name=_MODEL) == (0.0, 0.0)


def test_one_unreadable_entry_does_not_cost_the_whole_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream is a 3000-row community file; one retyped row must not blank the rest."""
    _serve(
        monkeypatch=monkeypatch,
        payload=json.dumps(obj={**_TABLE, "broken": {"input_cost_per_token": None}}),
    )

    assert get_token_rates(model_name=_MODEL) == (1.0, 2.0)
    assert set(load_model_info()) == {_MODEL}


def test_no_table_anywhere_degrades_to_the_documented_defaults(
    monkeypatch: pytest.MonkeyPatch, model_price_mirror_isolated: Path
) -> None:
    """The empty table keeps text and images flowing and only costs the estimate."""
    _refuse(monkeypatch=monkeypatch)

    assert load_model_info() == {}
    assert get_token_rates(model_name=_MODEL) == (0.0, 0.0)
    assert get_supported_modalities(model_name=_MODEL) == {"text", "image"}
    assert not model_price_mirror_isolated.exists()


def test_a_total_outage_is_logged_as_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing reaches the user when there is no table, so this line is the whole signal."""
    recorded: list[tuple[str, dict[str, object]]] = []

    def record(message: str, **fields: object) -> None:
        """Collects the module's error lines instead of emitting them."""
        recorded.append((message, fields))

    monkeypatch.setattr("discordbot.utils.model_pricing.logfire.error", record)
    _refuse(monkeypatch=monkeypatch)

    assert load_model_info() == {}
    assert [message for message, _ in recorded] == [
        "no model price table available; every model reads as unknown"
    ]
    assert recorded[0][1]["upstream"] == "unreachable"
    assert recorded[0][1]["mirror"] == "absent"


def test_a_failed_fetch_is_paid_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The degrade is memoized too, so an outage is not re-paid at 5s per call."""
    calls = _refuse(monkeypatch=monkeypatch)

    for _ in range(3):
        assert get_token_rates(model_name=_MODEL) == (0.0, 0.0)

    assert calls == [MODEL_INFO_URL]


def test_each_mirror_write_uses_its_own_temp_file(
    monkeypatch: pytest.MonkeyPatch, model_price_mirror_isolated: Path
) -> None:
    """`cache` does not serialize its body, so a shared temp path is two writers on one file."""
    del model_price_mirror_isolated
    staged: list[Path] = []
    real_replace = Path.replace

    def record(self: Path, target: Path) -> Path:
        """Notes which temp file each write promoted into the mirror."""
        staged.append(self)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", record)
    _write_mirror(payload=json.dumps(obj=_TABLE))
    _write_mirror(payload=json.dumps(obj=_TABLE))

    assert len(set(staged)) == 2


def test_a_mirror_write_failure_never_costs_the_fetched_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A serve directory that cannot be written loses the mirror, never the loaded table."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the mirror wants a directory", encoding="utf-8")
    monkeypatch.setattr(
        "discordbot.utils.model_pricing.MODEL_INFO_CACHE_PATH",
        blocker / "model_prices_and_context_window.json",
    )
    _serve(monkeypatch=monkeypatch, payload=json.dumps(obj=_TABLE))

    assert get_token_rates(model_name=_MODEL) == (1.0, 2.0)
