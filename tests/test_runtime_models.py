"""Guards the model strings and reasoning-effort values the runtime model catalog ships.

Gemini 3 cannot switch thinking off, so its `thinking_level` vocabulary starts at `minimal`.
`none` still round-trips through LiteLLM, but what it turns into depends on the branch the model
string picks. On the `thinking_level` branch every name here now takes, it is rewritten to
minimal / low with `includeThoughts: False`, which ends the reasoning summary the streaming
preview reads without failing anything; on a `*-latest` alias, which carries no `gemini-3`
substring for LiteLLM to match, it falls through to the pre-3 branch and sends
`thinkingBudget: 0`, which a Gemini 3.x model rejects. Neither shows up in tests because both
only surface against the live API, which is why the floor is pinned here.

The no-alias rule is the same trap through the other door, so it is guarded here too, at two
strengths. No tier may name an alias at all, because the substring is what decides the whole
translation. `slow_model` additionally may not drift off `high`, since it is the one tier whose
effort is chosen at runtime and handed to `interactions.create`, where an alias narrows the
accepted `thinking_level` vocabulary and #459 lost whole replies to it.
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


def test_no_tier_dispatches_an_alias() -> None:
    """Every tier names an explicit snapshot rather than a `*-latest` alias.

    `slow_model` has a hard reason of its own, guarded below. Every other tier has a quieter
    one: an alias carries no `gemini-3` substring, so LiteLLM translates its effort into a
    `thinkingBudget` on the pre-3 branch instead of a `thinking_level`, and picks `v1beta`
    over `v1alpha` while it is there. Nothing fails, the model is simply asked for something
    other than what the tier says. Pinning keeps that decision in the string written here
    instead of in LiteLLM's name matching.
    """
    offenders = {
        name: settings.name
        for name, settings in _catalog_models().items()
        if "latest" in settings.name
    }
    assert offenders == {}, (
        "A `*-latest` alias moves under the deployment and switches LiteLLM onto its pre-3 "
        f"translation branch. Name the snapshot instead. Offenders: {offenders}"
    )


def test_the_default_effort_is_the_gemini_floor() -> None:
    """A tier that names no effort still gets one the model can honor."""
    assert ModelSettings(name="gemini-3.7-flash").effort == "minimal"


def test_no_slow_model_branch_dispatches_an_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every slow-model branch stays on a pinned snapshot at high effort.

    The one tier whose effort is decided at runtime is also the one dispatched direct to Google,
    where the accepted `thinking_level` set is per-model and an alias's is the narrowest measured.
    Which snapshot a branch names is free to change and two branches may legitimately be equal;
    that none of them is an alias is not free, and nothing else in the tree reports it.

    Every hour of a week is swept rather than one instant per branch the catalog has today: the
    dispatch condition is the catalog's own to change, so a branch added on a second condition
    would be invisible to a fixed pair of timestamps and the guard would report nothing. That
    sweep is also why the catalog-wide alias guard above does not replace this one: reading the
    property once shows only the branch the clock happened to select.
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
    assert {"triage_model", "fast_model", "slow_model"} <= set(models)
