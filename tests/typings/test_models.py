"""Guards the model strings and reasoning-effort values the runtime model catalog ships.

What this file can check and what it cannot is the thing to hold on to. Whether a given model
accepts a given effort is a fact about that model, it changes the moment a tier is repointed, and
no test here can know it — look it up in openrouter's list instead (`ModelSettings.effort` carries
the lookup). So nothing below asserts that a pairing is legal.

What is checkable is the shape of the values the catalog ships, and both rules exist because their
failure only surfaces against the live API. No tier may ask for `none` / `disable`: that is not a
universal rule (openrouter lists 38 models that do accept `none`, and OpenAI's are among them) but
it holds for every tier here, because all of them are Gemini, where thinking cannot be switched
off at all. LiteLLM does not reject the ask but rewrites it to minimal / low with
`includeThoughts: False`, ending the reasoning summary the streaming preview reads without failing
anything. Point a tier at a non-Gemini model and this guard is the one to revisit rather than
route around.

And no tier may name a `*-latest` alias, because an alias moves under the deployment, so the
effort set it accepts can change with nothing in this repo changing. That matters most for
`slow_model`, the one tier whose effort is chosen at runtime and handed to `interactions.create`,
where #459 lost whole replies to it.
"""

import json

import pytest

from discordbot.typings.models import (
    ModelSettings,
    RouteClassification,
    RuntimeModelCatalog,
    RecallRouteClassification,
)
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES

# `disable` and `none` both mean "no thinking", which no Gemini model can honor. Every tier the
# catalog ships is Gemini, which is what makes this a guard rather than a preference; it is not a
# property of the field, and other providers do accept `none` (see the module docstring).
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


def test_no_tier_dispatches_an_alias() -> None:
    """Every tier names an explicit snapshot rather than a `*-latest` alias.

    `slow_model` has a hard reason of its own. Every other tier has a quieter
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


def test_model_settings_and_catalog_helpers() -> None:
    """Verifies model properties and provider-specific tool dispatch."""
    catalog = RuntimeModelCatalog()
    # LiteLLM sends an image model whose name lacks `gemini` to Imagen's `:predict` endpoint.
    assert catalog.image_model.is_gemini
    # Code execution is omitted on purpose: it 400s the request on file attachments.
    assert ModelSettings(name="gemini-test").tools == [{"googleSearch": {}}, {"urlContext": {}}]
    assert ModelSettings(name="claude-test").tools == [
        {"type": "web_search_20260209", "name": "web_search"},
        {"type": "web_fetch_20260209", "name": "web_fetch"},
    ]
    assert ModelSettings(name="openai-test").tools == [{"type": "web_search"}]


def test_the_catalog_exposes_the_tiers_under_test() -> None:
    """Guards the sweep itself: a catalog that stopped exposing tiers would pass vacuously."""
    models = _catalog_models()
    assert {"triage_model", "fast_model", "slow_model"} <= set(models)


@pytest.mark.parametrize("route_shape", [RouteClassification, RecallRouteClassification])
def test_the_route_schema_names_every_registered_link_source_inline(
    route_shape: type[RouteClassification],
) -> None:
    """The triage call's schema spells out exactly the registry's source names, in place.

    A source the enum lacks can never be selected, so its builder never starts; a name the
    registry lacks is one the model can pick with nothing behind it. The enum stays inline
    because the schema reaches the model as-is and both shapes were measured that way (#725):
    a named alias hoisted into `$defs` behind a `$ref` changes the bytes the route call sends.
    """
    schema = route_shape.model_json_schema()
    field = schema["properties"]["link_context_sources"]
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(obj=field)
    assert set(field["items"]["enum"]) == {source.name for source in LINK_CONTEXT_SOURCES}


@pytest.mark.parametrize("route_shape", [RouteClassification, RecallRouteClassification])
def test_the_route_schema_exposes_all_three_efforts_inline(
    route_shape: type[RouteClassification],
) -> None:
    """`RouteClassification.effort` stays inline with all three grades and no `xhigh`."""
    schema = route_shape.model_json_schema()
    effort_field = schema["properties"]["effort"]
    assert "$defs" not in schema
    assert "$ref" not in json.dumps(obj=effort_field)
    assert effort_field["enum"] == ["low", "medium", "high"]
    for grade in ("low", "medium", "high"):
        assert route_shape.model_validate(obj={"decision": "QA", "effort": grade}).effort == grade
