"""Guards view button and select callbacks against shadowing the base `View` API."""

from __future__ import annotations

from pkgutil import walk_packages
from importlib import import_module

from nextcord.ui import View

# Namespace import: the package object itself is the input, for its `__path__`.
import discordbot


def _import_every_module() -> None:
    """Imports the whole package so every `View` subclass is registered."""
    for module in walk_packages(path=discordbot.__path__, prefix=f"{discordbot.__name__}."):
        import_module(name=module.name)


def test_the_module_walk_reaches_a_nested_subpackage() -> None:
    """`walk_packages` skips a directory with no `__init__.py`, and does so silently.

    Views live inside cog directories now, some of them a level down (`games/fishing/`,
    `gen_reply/link_sources/`). A subpackage missing its `__init__.py` still imports
    fine by name and the cog still loads, so nothing else would notice that every
    `View` inside it dropped out of the shadowing check below.
    """
    _import_every_module()
    walked = {
        module.name
        for module in walk_packages(path=discordbot.__path__, prefix=f"{discordbot.__name__}.")
    }
    assert "discordbot.cogs.games.fishing.views" in walked
    assert "discordbot.cogs.gen_reply.link_sources.threads" in walked


def _view_subclasses() -> set[type[View]]:
    """Collects every `View` subclass reachable from the imported package."""
    found: set[type[View]] = set()
    pending: list[type[View]] = list(View.__subclasses__())
    while pending:
        cls = pending.pop()
        if cls in found:
            continue
        found.add(cls)
        pending.extend(cls.__subclasses__())
    return found


def test_no_view_callback_shadows_the_base_view_api() -> None:
    """A callback named after a `View` attribute silently breaks that attribute.

    `View.__init__` runs `setattr(self, func.__name__, item)` for every decorated
    callback, so naming one `refresh` rebinds the instance's `refresh` to a `Button`.
    The gateway calls `View.refresh(components)` on `MESSAGE_UPDATE` for any view
    attached to a tracked message, which then raises `TypeError`.
    """
    _import_every_module()
    reserved = set(dir(View))
    offenders = sorted(
        f"{cls.__module__}.{cls.__qualname__}.{callback.__name__}"
        for cls in _view_subclasses()
        if cls.__module__.startswith("discordbot.")
        for callback in getattr(cls, "__view_children_items__", ())
        if callback.__name__ in reserved
    )
    assert not offenders, f"view callbacks shadowing the base View API: {offenders}"
