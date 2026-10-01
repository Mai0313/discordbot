"""Guards every `View` and `Modal` subclass against three nextcord traps that fail silently."""

from __future__ import annotations

from pkgutil import walk_packages
from importlib import import_module

from nextcord.ui import View, Modal

# Namespace import: the package object itself is the input, for its `__path__`.
import discordbot


def _import_every_module() -> None:
    """Imports the whole package so every `View` subclass is registered."""
    for module in walk_packages(path=discordbot.__path__, prefix=f"{discordbot.__name__}."):
        import_module(name=module.name)


def test_the_module_walk_reaches_a_nested_subpackage() -> None:
    """`walk_packages` skips a directory with no `__init__.py`, and does so silently.

    Views live inside cog directories, and a cog may nest a subpackage a level down
    (`gen_reply/link_sources/`). A subpackage missing its `__init__.py` still imports fine
    by name and the cog still loads, so nothing else would notice that every `View` inside
    it dropped out of the shadowing check below.
    """
    _import_every_module()
    walked = {
        module.name
        for module in walk_packages(path=discordbot.__path__, prefix=f"{discordbot.__name__}.")
    }
    assert "discordbot.cogs.gen_reply.link_sources.threads" in walked


def _subclasses[T](base: type[T]) -> set[type[T]]:
    """Collects every subclass of `base` reachable from the imported package."""
    found: set[type[T]] = set()
    pending: list[type[T]] = list(base.__subclasses__())
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
    callbacks = [
        (cls, callback)
        for cls in _subclasses(base=View)
        if cls.__module__.startswith("discordbot.")
        for callback in getattr(cls, "__view_children_items__", ())
    ]
    offenders = sorted(
        f"{cls.__module__}.{cls.__qualname__}.{callback.__name__}"
        for cls, callback in callbacks
        if callback.__name__ in reserved
    )

    # The callbacks are read off a private nextcord attribute, so a rename there would empty the
    # sweep and pass it.
    assert callbacks, "the sweep found no view callbacks"
    assert not offenders, f"view callbacks shadowing the base View API: {offenders}"


def test_no_view_or_modal_leaves_a_failure_to_nextcords_stderr_print() -> None:
    """A raising callback reaches only its own view's or modal's `on_error`.

    nextcord dispatches a press or a submit outside `Client._run_event`, so `DiscordBot.on_error`
    never sees it, and the default `on_error` prints to `sys.stderr`, which `./data/logs` does
    not tee: a view or modal still on that default fails without a trace.
    """
    _import_every_module()
    swept = [
        (base, cls)
        for base in (View, Modal)
        for cls in _subclasses(base=base)
        if cls.__module__.startswith("discordbot.")
    ]
    offenders = sorted(
        f"{cls.__module__}.{cls.__qualname__}"
        for base, cls in swept
        if cls.on_error is base.on_error
    )

    assert {base for base, _cls in swept} == {View, Modal}, "the sweep missed a base"
    assert not offenders, (
        "views or modals on nextcord's stderr on_error, subclass "
        f"utils/logged_ui.py's LoggedView or LoggedModal instead: {offenders}"
    )


def test_no_view_timeout_leaves_a_failure_to_asyncios_stderr_print() -> None:
    """A raising `on_timeout` reaches no `on_error`: nothing awaits the task nextcord runs it in.

    Only the logging wrapper `LoggedView` puts around an override keeps such a failure out of
    asyncio's stderr print, at every depth below it.
    """
    _import_every_module()
    overriding = [
        cls
        for cls in _subclasses(base=View)
        if cls.__module__.startswith("discordbot.") and cls.on_timeout is not View.on_timeout
    ]
    offenders = sorted(
        f"{cls.__module__}.{cls.__qualname__}"
        for cls in overriding
        if not hasattr(cls.on_timeout, "__wrapped__")
    )

    assert overriding, "the sweep found no on_timeout override"
    assert not offenders, (
        "views whose on_timeout failure reaches only stderr, subclass "
        f"utils/logged_ui.py's LoggedView instead: {offenders}"
    )
