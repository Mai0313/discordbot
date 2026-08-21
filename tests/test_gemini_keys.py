"""Tests for Gemini key discovery and the per-reply key balancer.

The autouse `gemini_key_set_isolated` fixture strips every `GEMINI_API_KEY_<n>` before each
test here, so a checkout with three real keys configured cannot decide any outcome below.
Each test sets the exact key set it is about.
"""

from datetime import datetime

import pytest

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.services.gemini_keys.balancer import pick_gemini_key, reset_balancer_state
from discordbot.services.gemini_keys.database import read_day_counts


def _keys(config: LLMConfig) -> list[tuple[int, str]]:
    """Flattens the configured slots to (index, key) pairs for comparison."""
    return [(slot.index, slot.api_key) for slot in config.gemini_keys]


def _configure(monkeypatch: pytest.MonkeyPatch, count: int) -> LLMConfig:
    """Configures `count` Gemini keys and returns the config that sees them."""
    monkeypatch.setenv(name="GEMINI_API_KEY", value="key-1")
    for number in range(2, count + 1):
        monkeypatch.setenv(name=f"GEMINI_API_KEY_{number}", value=f"key-{number}")
    return LLMConfig()


def _pin_day(monkeypatch: pytest.MonkeyPatch, day: str) -> None:
    """Pins the balancer's day window to `day` (a `YYYY-MM-DD` date)."""
    stamped = datetime.fromisoformat(day).replace(tzinfo=TAIWAN_TIMEZONE)
    monkeypatch.setattr("discordbot.services.gemini_keys.balancer.database_now", lambda: stamped)


async def _pick_indexes(config: LLMConfig, times: int) -> list[int]:
    """Picks `times` keys in a row and returns the numbers handed out."""
    picked: list[int] = []
    for _ in range(times):
        slot = await pick_gemini_key(config=config)
        assert slot is not None
        picked.append(slot.index)
    return picked


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


async def test_three_keys_take_an_equal_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nine replies split three ways, which is the whole point of the feature."""
    config = _configure(monkeypatch=monkeypatch, count=3)

    picked = await _pick_indexes(config=config, times=9)

    assert sorted(picked) == [1, 1, 1, 2, 2, 2, 3, 3, 3]


async def test_a_tie_goes_to_the_lowest_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """From zero every key ties, so the order is deterministic rather than arbitrary."""
    config = _configure(monkeypatch=monkeypatch, count=3)

    assert await _pick_indexes(config=config, times=3) == [1, 2, 3]


async def test_a_key_added_later_catches_up_rather_than_alternating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new key absorbs replies until it is level, which round-robin would never do.

    Round-robin would resume alternating and leave the existing gap in place for good, so
    the key added at noon would stay permanently behind the ones that ran all morning.
    """
    config = _configure(monkeypatch=monkeypatch, count=2)
    await _pick_indexes(config=config, times=4)

    widened = _configure(monkeypatch=monkeypatch, count=3)
    caught_up = await _pick_indexes(config=widened, times=2)

    assert caught_up == [3, 3]


async def test_a_new_day_starts_every_key_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """The window is what bounds how far behind a newly added key can start.

    Without it a lifetime counter would hand a fourth key every reply for as long as it took
    to catch up with months of history.
    """
    config = _configure(monkeypatch=monkeypatch, count=2)
    _pin_day(monkeypatch=monkeypatch, day="2026-08-21")
    await _pick_indexes(config=config, times=3)

    _pin_day(monkeypatch=monkeypatch, day="2026-08-22")

    assert await _pick_indexes(config=config, times=2) == [1, 2]


async def test_an_unconfigured_deployment_picks_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key is a supported state: the caller runs unpinned, as the bot did before."""
    monkeypatch.setenv(name="GEMINI_API_KEY", value="")

    assert await pick_gemini_key(config=LLMConfig()) is None


async def test_the_counts_survive_a_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """A restart resumes the day's split instead of putting every key back at zero."""
    config = _configure(monkeypatch=monkeypatch, count=3)
    _pin_day(monkeypatch=monkeypatch, day="2026-08-22")
    await _pick_indexes(config=config, times=2)

    reset_balancer_state()

    assert await _pick_indexes(config=config, times=1) == [3]
    assert await read_day_counts(day="2026-08-22") == {1: 1, 2: 1, 3: 1}


async def test_an_unreachable_database_still_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-memory counts are the authoritative side, so a dead database costs history only.

    A balancer that refused to hand out a key when its bookkeeping was unavailable would turn
    a cosmetic outage into a total one, on the path every single reply goes through.
    """
    config = _configure(monkeypatch=monkeypatch, count=3)

    async def _explode(day: str) -> dict[int, int]:
        """Stands in for a database that cannot be read."""
        raise RuntimeError(f"llm_keys.db unavailable for {day}")

    monkeypatch.setattr("discordbot.services.gemini_keys.balancer.read_day_counts", _explode)

    assert sorted(await _pick_indexes(config=config, times=6)) == [1, 1, 2, 2, 3, 3]
