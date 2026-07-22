"""Guards the reasoning-effort values the runtime model catalog ships.

Gemini 3 cannot switch thinking off, so its `thinking_level` vocabulary starts at `minimal`.
`none` still round-trips through LiteLLM, but only for a model it recognises as Gemini 3 by
the literal substring `gemini-3`; the `*-latest` aliases this project dispatches on do not
carry it, so `none` falls through to the pre-3 branch and sends `thinkingBudget: 0`, which a
Gemini 3.x model rejects. The failure is invisible in tests because it only shows up against
the live API, which is why it is pinned here.
"""

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


def test_the_catalog_exposes_the_tiers_under_test() -> None:
    """Guards the sweep itself: a catalog that stopped exposing tiers would pass vacuously."""
    models = _catalog_models()
    assert {"fast_model", "tool_model", "media_reply_model", "slow_model"} <= set(models)
