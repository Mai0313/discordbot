"""Pins that no `nextcord.ui.View` in this package names a callback after the base `View` API.

`View.__init_subclass__` collects every decorated button and select callback of the class and its
bases into `__view_children_items__`, and `View.__init__` then runs
`setattr(self, func.__name__, item)` over that list. A callback named `refresh`, `stop` or
`children` therefore rebinds the instance attribute of that name to the `Button`, and the base
method is gone. Nothing else reports it: the decorator retypes the callback to a `Callable` alias,
which suppresses the type checker's override rule, and the loss only surfaces on the gateway's
`MESSAGE_UPDATE` path, where `ViewStore.update_from_message` calls `view.refresh(components)` on
every message-tracked view and gets `TypeError: 'Button' object is not callable`. A stock view
shipped that way once, so the guard is a name check across the whole package rather than a test of
one view.

The check only sees a subclass whose module has been imported, so both tests import the package
first and then read `View.__subclasses__()` transitively. That makes the module walk load-bearing:
`walk_packages` skips a directory with no `__init__.py` and says nothing about it, while such a
subpackage still imports fine by name and its cog still loads, so every view inside it would drop
out of the check with nothing else noticing. The second test pins the walk against the two nested
cog subpackages in the tree, `games/fishing/` being the one holding views today.
"""

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
    """The walk feeding the check below descends into cog subpackages, not just cog roots."""
    _import_every_module()
    walked = {
        module.name
        for module in walk_packages(path=discordbot.__path__, prefix=f"{discordbot.__name__}.")
    }
    assert "discordbot.cogs.games.fishing.views" in walked
    assert "discordbot.cogs.gen_reply.link_sources.threads" in walked


def _view_subclasses() -> set[type[View]]:
    """Collects every `View` subclass reachable from the imported package.

    Returns:
        Every loaded `View` subclass, transitively, this package's and the library's alike; the
        caller narrows it by module.
    """
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
    """No button or select callback in the package is named after an attribute of `View`."""
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
