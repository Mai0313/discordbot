"""Tests for LiteLLM model pricing cache parsing."""

from collections.abc import Iterator

import pytest

from discordbot.utils import model_pricing


@pytest.fixture(autouse=True)
def clear_model_price_cache() -> Iterator[None]:
    """Clears the process-level model price cache around each test."""
    model_pricing._load_model_prices.cache_clear()
    yield
    model_pricing._load_model_prices.cache_clear()


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
    monkeypatch.setattr(target=model_pricing, name="_fetch_upstream", value=lambda: raw_table)

    assert model_pricing.get_token_rates(model_name="test-model") == (0.1, 0.3)
    assert model_pricing.get_supported_modalities(model_name="test-model") == {"text", "video"}


def test_model_pricing_defaults_unknown_and_missing_modalities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown models and incomplete modality data keep the previous fallbacks."""
    monkeypatch.setattr(
        target=model_pricing,
        name="_fetch_upstream",
        value=lambda: {"text-model": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}},
    )

    assert model_pricing.get_token_rates(model_name="missing-model") == (0.0, 0.0)
    assert model_pricing.get_supported_modalities(model_name="missing-model") == {"text", "image"}
    assert model_pricing.get_supported_modalities(model_name="text-model") == {"text", "image"}


def test_model_pricing_skips_malformed_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed upstream entries: valid ones parse, invalid rows are skipped."""
    monkeypatch.setattr(
        target=model_pricing,
        name="_fetch_upstream",
        value=lambda: {
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
    monkeypatch.setattr(target=model_pricing, name="_fetch_upstream", value=lambda: {})

    assert model_pricing.get_token_rates(model_name="any-model") == (0.0, 0.0)
    assert model_pricing.get_supported_modalities(model_name="any-model") == {"text", "image"}
