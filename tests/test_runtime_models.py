"""Guards the model strings and reasoning-effort values the runtime model catalog ships.

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
from datetime import UTC, datetime, timedelta

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


def test_no_slow_model_branch_dispatches_an_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every slow-model branch stays on a pinned snapshot at high effort.

    The one tier whose effort is decided at runtime is also the one dispatched direct to Google,
    where the accepted `thinking_level` set is per-model and an alias's is the narrowest measured.
    Which snapshot a branch names is free to change and two branches may legitimately be equal;
    that none of them is an alias is not free, and nothing else in the tree reports it.

    Every hour of a week is swept rather than one instant per branch the catalog has today: the
    dispatch condition is the catalog's own to change, so a branch added on a second condition
    would be invisible to a fixed pair of timestamps and the guard would report nothing.
    """
    monday = datetime(year=2026, month=5, day=18, tzinfo=UTC)
    aliases: dict[str, str] = {}
    wrong_effort: dict[str, str] = {}
    for offset in range(7 * 24):
        now = monday + timedelta(hours=offset)
        settings = _slow_model_at(monkeypatch=monkeypatch, now=now)
        when = f"{now:%a %H:00} UTC"
        if "latest" in settings.name:
            aliases.setdefault(settings.name, when)
        if settings.effort != "high":
            wrong_effort.setdefault(str(settings.effort), when)

    assert aliases == {}, (
        "A `*-latest` alias narrows the thinking levels this tier can be dispatched with to "
        "low / high. `EffortGrade` emits only those two today, so this guards the next "
        f"vocabulary change rather than today's traffic, which #459 was. Aliases: {aliases}"
    )
    assert wrong_effort == {}, f"Every slow-model branch ships `high`. Offenders: {wrong_effort}"


def test_the_catalog_exposes_the_tiers_under_test() -> None:
    """Guards the sweep itself: a catalog that stopped exposing tiers would pass vacuously."""
    models = _catalog_models()
    assert {"fast_model", "tool_model", "media_reply_model", "slow_model"} <= set(models)
