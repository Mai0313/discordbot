"""Tests for LiteLLM model pricing cache parsing."""

from collections.abc import Mapping, Iterator

import pytest

from discordbot.utils import model_pricing


@pytest.fixture(autouse=True)
def clear_model_price_cache() -> Iterator[None]:
    """Clears the process-level model price cache around each test."""
    model_pricing._load_model_prices.cache_clear()
    yield
    model_pricing._load_model_prices.cache_clear()


def test_model_pricing_uses_typed_priority_rates_and_modalities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid raw price entries are parsed into typed lookup models."""
    raw_table: dict[str, object] = {
        "test-model": {
            "input_cost_per_token": 0.1,
            "input_cost_per_token_priority": 0.2,
            "output_cost_per_token": 0.3,
            "output_cost_per_token_priority": 0.4,
            "supported_modalities": ["text", "video"],
            "ignored_upstream_field": "kept out of the typed model",
        }
    }
    saved_tables: list[Mapping[str, object]] = []

    def fake_fetch_upstream(timeout: int = 5) -> dict[str, object]:
        """Returns a deterministic upstream table."""
        assert timeout == 5
        return raw_table

    def fake_save_disk_cache(data: Mapping[str, object]) -> None:
        """Records the raw data that would be cached on disk."""
        saved_tables.append(data)

    monkeypatch.setattr(target=model_pricing, name="_fetch_upstream", value=fake_fetch_upstream)
    monkeypatch.setattr(target=model_pricing, name="_save_disk_cache", value=fake_save_disk_cache)

    assert model_pricing.get_token_rates(model_name="test-model") == (0.2, 0.4)
    assert model_pricing.get_supported_modalities(model_name="test-model") == {"text", "video"}
    assert saved_tables == [raw_table]


def test_model_pricing_defaults_unknown_and_missing_modalities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown models and incomplete modality data keep the previous fallbacks."""

    def fake_fetch_upstream(timeout: int = 5) -> dict[str, object]:
        """Returns one entry without modality metadata."""
        assert timeout == 5
        return {"text-model": {"input_cost_per_token": 0.1, "output_cost_per_token": 0.2}}

    def fake_save_disk_cache(data: Mapping[str, object]) -> None:
        """Avoids writing test data into the repo data directory."""
        assert data

    monkeypatch.setattr(target=model_pricing, name="_fetch_upstream", value=fake_fetch_upstream)
    monkeypatch.setattr(target=model_pricing, name="_save_disk_cache", value=fake_save_disk_cache)

    assert model_pricing.get_token_rates(model_name="missing-model") == (0.0, 0.0)
    assert model_pricing.get_supported_modalities(model_name="missing-model") == {"text", "image"}
    assert model_pricing.get_supported_modalities(model_name="text-model") == {"text", "image"}


def test_model_pricing_falls_back_to_disk_cache_and_skips_malformed_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetch failures use validated disk-cache entries and skip malformed rows."""

    def fake_fetch_upstream(timeout: int = 5) -> dict[str, object]:
        """Simulates a malformed upstream table."""
        assert timeout == 5
        raise ValueError("bad upstream")

    def fake_load_disk_cache() -> dict[str, object]:
        """Returns mixed valid and invalid cached rows."""
        return {
            "bad-shape": "not a mapping",
            "bad-rate": {"input_cost_per_token": "not-a-number"},
            "cached-model": {
                "input_cost_per_token": "0.000001",
                "output_cost_per_token": 0.000002,
                "supported_modalities": ["text", 123, "image"],
            },
        }

    monkeypatch.setattr(target=model_pricing, name="_fetch_upstream", value=fake_fetch_upstream)
    monkeypatch.setattr(target=model_pricing, name="_load_disk_cache", value=fake_load_disk_cache)

    assert model_pricing.get_token_rates(model_name="cached-model") == (0.000001, 0.000002)
    assert model_pricing.get_supported_modalities(model_name="cached-model") == {"text", "image"}
    assert model_pricing.get_token_rates(model_name="bad-rate") == (0.0, 0.0)
