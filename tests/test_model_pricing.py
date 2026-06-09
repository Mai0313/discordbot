"""Tests for LiteLLM model pricing cache parsing."""

from collections.abc import Iterator

import pytest

from discordbot.utils import model_pricing


class FakeResponse:
    """Minimal `requests.Response` stand in for model info fetch tests."""

    def __init__(self, data: object) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        """Simulates a successful HTTP response."""

    def json(self) -> object:
        """Returns the configured JSON payload."""
        return self.data


def stub_model_info_response(monkeypatch: pytest.MonkeyPatch, raw_table: object) -> None:
    """Stubs the LiteLLM model info HTTP response."""

    def fake_get(*, url: str, timeout: int) -> FakeResponse:
        assert url == model_pricing.MODEL_INFO_URL
        assert timeout == 5
        return FakeResponse(data=raw_table)

    monkeypatch.setattr(target=model_pricing.requests, name="get", value=fake_get)


@pytest.fixture(autouse=True)
def clear_model_price_cache() -> Iterator[None]:
    """Clears the process level model price cache around each test."""
    model_pricing.load_model_info.cache_clear()
    yield
    model_pricing.load_model_info.cache_clear()


def test_model_pricing_parses_typed_rates_and_modalities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid raw price entries are parsed into typed lookup models."""
    raw_table: dict[str, object] = {
        "test-model": {
            "input_cost_per_token": 0.1,
            "output_cost_per_token": 0.3,
            "supported_modalities": ["text", "video"],
            "ignored_upstream_field": "kept out of the typed model",
        }
    }
    stub_model_info_response(monkeypatch=monkeypatch, raw_table=raw_table)

    assert model_pricing.get_token_rates(model_name="test-model") == (0.1, 0.3)
    assert model_pricing.get_supported_modalities(model_name="test-model") == {"text", "video"}


def test_model_pricing_defaults_unknown_and_missing_modalities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown models and incomplete modality data keep the previous fallbacks."""
    stub_model_info_response(
        monkeypatch=monkeypatch,
        raw_table={"text-model": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}},
    )

    assert model_pricing.get_token_rates(model_name="missing-model") == (0.0, 0.0)
    assert model_pricing.get_supported_modalities(model_name="missing-model") == {"text", "image"}
    assert model_pricing.get_supported_modalities(model_name="text-model") == {"text", "image"}


def test_model_pricing_skips_malformed_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed upstream entries: valid ones parse, invalid rows are skipped."""
    stub_model_info_response(
        monkeypatch=monkeypatch,
        raw_table={
            "bad-shape": "not a mapping",
            "bad-rate": {"input_cost_per_token": "not-a-number"},
            "cached-model": {
                "input_cost_per_token": "0.000001",
                "output_cost_per_token": 0.000002,
                "supported_modalities": ["text", "image"],
            },
        },
    )

    assert model_pricing.get_token_rates(model_name="cached-model") == (0.000001, 0.000002)
    assert model_pricing.get_supported_modalities(model_name="cached-model") == {"text", "image"}
    assert model_pricing.get_token_rates(model_name="bad-rate") == (0.0, 0.0)


def test_model_pricing_handles_empty_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the upstream fetch yields nothing, every lookup falls back to defaults."""
    stub_model_info_response(monkeypatch=monkeypatch, raw_table={})

    assert model_pricing.get_token_rates(model_name="any-model") == (0.0, 0.0)
    assert model_pricing.get_supported_modalities(model_name="any-model") == {"text", "image"}
