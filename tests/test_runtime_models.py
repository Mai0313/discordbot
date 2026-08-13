"""Guards the reasoning-effort values the runtime model catalog ships.

Gemini 3 cannot switch thinking off, so its `thinking_level` vocabulary starts at `minimal`.
`none` still round-trips through LiteLLM, but only for a model it recognises as Gemini 3 by
the literal substring `gemini-3`; the `*-latest` aliases this project dispatches on do not
carry it, so `none` falls through to the pre-3 branch and sends `thinkingBudget: 0`, which a
Gemini 3.x model rejects. The failure is invisible in tests because it only shows up against
the live API, which is why it is pinned here.

`slow_model`'s no-alias rule is the same trap through the other door, so it is guarded here too:
an alias narrows the `thinking_level` vocabulary the YouTube answer turn may hand to
`interactions.create`, which is how #459 lost whole replies.
"""

from types import SimpleNamespace
from datetime import UTC, datetime

import pytest

from discordbot.typings.models import ModelSettings, RuntimeModelCatalog

# Everything below `minimal` is unrepresentable on Gemini 3; `disable` and `none` both mean
# "no thinking", which the model cannot honor.
_REFUSED_EFFORTS = frozenset({"none", "disable"})


def _catalog_models() -> dict[str, ModelSettings]:
    """Every ModelSettings the catalog exposes, keyed by its property name."""
    catalog = RuntimeModelCatalog()
    names = [
        name
        for name in dir(type(catalog))
        if not name.startswith("_") and isinstance(getattr(type(catalog), name, None), property)
    ]
    found = {name: getattr(catalog, name) for name in names}
    return {name: value for name, value in found.items() if isinstance(value, ModelSettings)}


def _slow_model_at(*, monkeypatch: pytest.MonkeyPatch, now: datetime) -> ModelSettings:
    """The slow-model branch the catalog dispatches with its clock pinned to `now`."""

    def fixed_now(tz: object) -> datetime:
        """Returns the pinned timestamp."""
        assert tz is UTC
        return now

    monkeypatch.setattr("discordbot.typings.models.datetime", SimpleNamespace(now=fixed_now))
    return RuntimeModelCatalog().slow_model


def test_no_tier_asks_for_an_effort_gemini_cannot_honor() -> None:
    """No model tier ships `none`/`disable`, which Gemini 3 turns into a rejected request."""
    offenders = {
        name: settings.effort
        for name, settings in _catalog_models().items()
        if settings.effort in _REFUSED_EFFORTS
    }
    assert offenders == {}, (
        "Gemini 3 has no way to disable thinking; use 'minimal' as the floor. Offenders: "
        f"{offenders}"
    )


def test_the_default_effort_is_the_gemini_floor() -> None:
    """A tier that names no effort still gets one the model can honor."""
    assert ModelSettings(name="gemini-flash-latest").effort == "minimal"


def test_neither_slow_model_branch_dispatches_an_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both slow-model branches stay on a pinned snapshot at high effort.

    The one tier whose effort is decided at runtime is also the one dispatched direct to Google,
    where the accepted `thinking_level` set is per-model and an alias's is the narrowest measured.
    Which snapshot each branch names is free to change; that neither is an alias is not, and
    nothing else in the tree reports it. The two branches may legitimately be equal.
    """
    branches = {
        "peak": _slow_model_at(
            monkeypatch=monkeypatch, now=datetime(year=2026, month=5, day=18, hour=12, tzinfo=UTC)
        ),
        "off_peak": _slow_model_at(
            monkeypatch=monkeypatch, now=datetime(year=2026, month=5, day=23, hour=12, tzinfo=UTC)
        ),
    }

    aliases = {
        branch: settings.name for branch, settings in branches.items() if "latest" in settings.name
    }
    assert aliases == {}, (
        "A `*-latest` alias accepts only low / high as a thinking level, so a graded effort "
        f"outside that pair loses the whole reply. Pin a snapshot instead. Offenders: {aliases}"
    )
    assert {branch: settings.effort for branch, settings in branches.items()} == {
        "peak": "high",
        "off_peak": "high",
    }


def test_the_catalog_exposes_the_tiers_under_test() -> None:
    """Guards the sweep itself: a catalog that stopped exposing tiers would pass vacuously."""
    models = _catalog_models()
    assert {"fast_model", "tool_model", "media_reply_model", "slow_model"} <= set(models)
