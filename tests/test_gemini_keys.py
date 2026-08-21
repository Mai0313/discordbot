"""Tests for Gemini key discovery and the per-reply key balancer.

The autouse `gemini_key_set_isolated` fixture strips every `GEMINI_API_KEY_<n>` before each
test here, so a checkout with three real keys configured cannot decide any outcome below.
Each test sets the exact key set it is about.
"""

import pytest

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings


def _keys(config: LLMConfig) -> list[tuple[int, str]]:
    """Flattens the configured slots to (index, key) pairs for comparison."""
    return [(slot.index, slot.api_key) for slot in config.gemini_keys]


def test_the_numbered_keys_follow_the_unsuffixed_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Key 1 is `GEMINI_API_KEY`; `_2` and `_3` follow it in number order."""
    monkeypatch.setenv(name="GEMINI_API_KEY", value="first")
    monkeypatch.setenv(name="GEMINI_API_KEY_3", value="third")
    monkeypatch.setenv(name="GEMINI_API_KEY_2", value="second")

    assert _keys(config=LLMConfig()) == [(1, "first"), (2, "second"), (3, "third")]


def test_a_gap_in_the_numbering_keeps_every_key_on_its_own_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing `_2` must not renumber `_3` down onto the proxy's `-key2` deployment.

    The number is shared with LiteLLM and with the Google project that will accept the
    reply's uploaded files, so shifting it would send a file to one project and the request
    that names it to another.
    """
    monkeypatch.setenv(name="GEMINI_API_KEY", value="first")
    monkeypatch.setenv(name="GEMINI_API_KEY_3", value="third")

    assert _keys(config=LLMConfig()) == [(1, "first"), (3, "third")]


def test_a_blank_key_is_dropped_rather_than_occupying_its_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emptying a variable retires that key without renumbering the ones after it."""
    monkeypatch.setenv(name="GEMINI_API_KEY", value="first")
    monkeypatch.setenv(name="GEMINI_API_KEY_2", value="   ")
    monkeypatch.setenv(name="GEMINI_API_KEY_3", value="third")

    assert _keys(config=LLMConfig()) == [(1, "first"), (3, "third")]


def test_key_one_comes_from_the_unsuffixed_variable_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GEMINI_API_KEY_1` is not a second spelling of key 1, so it is ignored.

    Two spellings of one number could only ever disagree, and the proxy has no `-key1`
    credential of its own to break the tie.
    """
    monkeypatch.setenv(name="GEMINI_API_KEY", value="first")
    monkeypatch.setenv(name="GEMINI_API_KEY_1", value="impostor")

    assert _keys(config=LLMConfig()) == [(1, "first")]


def test_a_non_numeric_suffix_is_not_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only `GEMINI_API_KEY_<digits>` counts, so a neighbouring variable is not swept in."""
    monkeypatch.setenv(name="GEMINI_API_KEY", value="first")
    monkeypatch.setenv(name="GEMINI_API_KEY_BACKUP", value="not-a-slot")

    assert _keys(config=LLMConfig()) == [(1, "first")]


def test_a_double_digit_key_sorts_after_the_single_digit_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering is numeric, not lexicographic, so key 10 does not land between 1 and 2."""
    monkeypatch.setenv(name="GEMINI_API_KEY", value="first")
    monkeypatch.setenv(name="GEMINI_API_KEY_10", value="tenth")
    monkeypatch.setenv(name="GEMINI_API_KEY_2", value="second")

    assert _keys(config=LLMConfig()) == [(1, "first"), (2, "second"), (10, "tenth")]


def test_an_unconfigured_deployment_has_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key configured is an empty set, never a slot holding an empty string."""
    monkeypatch.setenv(name="GEMINI_API_KEY", value="")

    assert LLMConfig().gemini_keys == []


def test_key_one_follows_the_field_rather_than_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An overridden `gemini_api_key` is honoured, which is how most tests set the key."""
    monkeypatch.setenv(name="GEMINI_API_KEY", value="from-env")
    config = LLMConfig()
    config.gemini_api_key = "overridden"

    assert _keys(config=config) == [(1, "overridden")]


def test_a_pinned_tier_dispatches_on_the_key_deployment() -> None:
    """The pin appends the proxy's `-key<n>` suffix and leaves `name` alone."""
    pinned = ModelSettings(name="gemini-3.7-flash", effort="high", key_index=2)

    assert pinned.deployment_name == "gemini-3.7-flash-key2"
    assert pinned.name == "gemini-3.7-flash"


def test_an_unpinned_tier_dispatches_on_its_own_name() -> None:
    """No pin means no suffix, so an unbalanced path reaches the deployment it always did."""
    unpinned = ModelSettings(name="gemini-3.7-flash", effort="high")

    assert unpinned.deployment_name == "gemini-3.7-flash"


def test_the_pin_never_reaches_the_provider_test() -> None:
    """`tools` still reads `name`, so a pinned Gemini tier keeps its grounding tools.

    The provider tests are substring matches that happen to survive a suffix, but the ones
    that do not survive it (`get_token_rates`, `get_supported_modalities`) fail silently, so
    this pins the rule rather than the one case that would have been noticed.
    """
    pinned = ModelSettings(name="gemini-3.7-flash", effort="high", key_index=3)

    assert pinned.tools == [{"googleSearch": {}}, {"urlContext": {}}]
