"""Pins the import direction between `cogs/`, `services/`, `utils/` and `typings/`.

The layering is what the cog-per-directory layout buys: a cog directory holds only what
that cog uses, so "where does this feature live" has one answer. Nothing else enforces it
— an import reaching sideways into a peer cog's directory runs perfectly well and only
shows up as a tangle months later, which is the state this replaced.
"""

import ast
from pathlib import Path
from functools import cache

import pytest

_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "discordbot"
_COGS = _PACKAGE / "cogs"


def _relative_import_base(module: Path) -> str:
    """The package a `from . ...` inside this file resolves against.

    For `pkg/__init__.py` that is `pkg` itself, not `pkg`'s parent: a package's `__init__`
    has `__package__ == "pkg"`, so `from .x import y` there means `pkg.x`. Reading it one
    level too high makes every relative import in an `__init__.py` look like it points at a
    sibling package — which both hides a real `from ..peer.mod import X` and invents a
    violation out of an ordinary `from .own_mod import X`.
    """
    return ".".join(module.relative_to(_PACKAGE.parent).with_suffix("").parts[:-1])


def _imported_modules(module: Path) -> set[str]:
    """Returns every `discordbot.*` module name a file imports, relative imports resolved."""
    return _imports_in(
        source=module.read_text(encoding="utf-8"), parent=_relative_import_base(module)
    )


def _imports_in(source: str, parent: str, scope: str = "discordbot.") -> set[str]:
    """Returns every `discordbot.*` module name a source file imports, relative ones resolved.

    Reads `TYPE_CHECKING` and function-local imports too: they are still edges in the
    dependency graph, and the one cog-to-cog import this repo ever had was a
    `TYPE_CHECKING` one that never executes and so no test could otherwise see.

    Takes the source rather than the path so the relative forms can be asserted directly;
    no module in the package writes one today.

    `scope` is what the result is narrowed to. It defaults to this package, which is all the
    layering rules below need, and is widened to everything by the Discord-free guard — the one
    question here that turns on a third-party name.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source=source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if not node.level:
            if node.module:
                found.add(node.module)
                found.update(f"{node.module}.{alias.name}" for alias in node.names)
            continue
        base = parent
        for _ in range(node.level - 1):
            base = base.rsplit(".", maxsplit=1)[0]
        prefix = f"{base}.{node.module}" if node.module else base
        found.add(prefix)
        found.update(f"{prefix}.{alias.name}" for alias in node.names)
    return {name for name in found if name.startswith(scope)}


def _modules(root: Path) -> list[Path]:
    """Every Python module under a directory, ignoring bytecode caches."""
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _cog_of(module: Path) -> str:
    """The cog directory a module under `cogs/` belongs to, empty for `cogs/__init__.py`."""
    relative = module.relative_to(_COGS)
    return relative.parts[0] if len(relative.parts) > 1 else ""


def test_a_cog_never_imports_a_peer_cog() -> None:
    """A cog directory holds one cog's code; reaching into another one is what services are for.

    The rule covers helper modules too, not just `cog.py`. Before the split it was only the
    cog modules that stayed apart while their helper packages imported each other freely,
    which is how "where does the economy live" stopped having an answer.
    """
    offenders: list[str] = []
    for module in _modules(_COGS):
        owner = _cog_of(module)
        if not owner:
            continue
        for imported in _imported_modules(module):
            parts = imported.split(".")
            if len(parts) < 3 or parts[1] != "cogs" or parts[2] == owner:
                continue
            offenders.append(f"{module.relative_to(_COGS).as_posix()} -> {imported}")
    assert not offenders, f"cogs importing a peer cog: {sorted(offenders)}"


@pytest.mark.parametrize(
    ("layer", "forbidden"),
    [
        ("services", ("discordbot.cogs.",)),
        ("utils", ("discordbot.cogs.", "discordbot.services.")),
        ("typings", ("discordbot.cogs.", "discordbot.services.")),
    ],
)
def test_a_lower_layer_never_imports_a_higher_one(layer: str, forbidden: tuple[str, ...]) -> None:
    """`services` is Discord-free domain code, and `utils` / `typings` sit below even that.

    An edge the other way is what turns a shared engine back into one cog's private helper
    that a second cog happens to reach into.
    """
    offenders: list[str] = []
    for module in _modules(_PACKAGE / layer):
        for imported in _imported_modules(module):
            if imported.startswith(forbidden):
                offenders.append(f"{module.relative_to(_PACKAGE).as_posix()} -> {imported}")
    assert not offenders, f"{layer} importing a higher layer: {sorted(offenders)}"


def _module_file(name: str) -> Path | None:
    """The file a dotted `discordbot.*` name refers to, or None when it names nothing on disk.

    An import yields both `discordbot.x.y` and `discordbot.x.y.SomeName`, so a name that resolves
    to no file is a member of its parent rather than a miss — hence walking up rather than failing.
    """
    parts = name.split(".")[1:]
    while parts:
        candidate = _PACKAGE.joinpath(*parts)
        if candidate.with_suffix(".py").is_file():
            return candidate.with_suffix(".py")
        if (candidate / "__init__.py").is_file():
            return candidate / "__init__.py"
        parts.pop()
    return None


@cache
def _import_roots(module: Path) -> frozenset[str]:
    """The top-level package of every import in a file, this one's own included."""
    names = _imports_in(
        source=module.read_text(encoding="utf-8"), parent=_relative_import_base(module), scope=""
    )
    return frozenset(name.split(".", maxsplit=1)[0] for name in names)


def _reachable_within_package(module: Path) -> dict[Path, str]:
    """Every module in this package reachable from one, mapped to the path that got there.

    Following the graph to a fixed point rather than reading one file is the whole point: an
    import two hops away pulls its dependencies in just as surely as a direct one, and the
    violation this guards against is exactly the hop nobody looked at.

    The package root is seeded rather than discovered. Python executes `discordbot/__init__.py` for
    every import in the tree, so it is an unconditional dependency of all of them, but no import
    statement names it in a form the walk can resolve — `_module_file("discordbot")` has no path
    segments left to try, and a bare `import discordbot` does not survive the scanner's prefix
    filter either. A dynamic import is the one edge that stays invisible; that is inherent to
    reading the AST and is not worth machinery.
    """
    start = module.relative_to(_PACKAGE).as_posix()
    root = _PACKAGE / "__init__.py"
    seen = {module: start, root: f"{start} -> __init__.py"}
    queue = [module, root]
    while queue:
        current = queue.pop()
        for name in _imported_modules(current):
            found = _module_file(name=name)
            if found is None or found in seen:
                continue
            seen[found] = f"{seen[current]} -> {found.relative_to(_PACKAGE).as_posix()}"
            queue.append(found)
    return seen


def test_services_never_reaches_discord() -> None:
    """`services/` is the Discord-free layer, and until now nothing but prose said so.

    The layering scan above reads `discordbot.*` edges only, so `import nextcord` inside a service
    — or inside anything a service imports — was invisible to every test in the suite. That was
    affordable while `services/` held a ledger and a memory store, neither of which has a Discord
    surface to be tempted by. `services/platforms/` is what changes it: its job is to turn a link
    into something a channel shows, the send sits one import away, and `utils/douyin_delivery.py`
    exists precisely because that one import was there.

    Transitive on purpose. A direct-import check is satisfied by moving the offending line one
    module over, which is the same edge wearing a hat.
    """
    modules = _modules(_PACKAGE / "services")

    # `rglob` on a directory that is not there yields nothing, so a renamed or mistyped start path
    # would leave this scanning zero modules and passing. The other two discovery sweeps in this
    # change carry the same tripwire for the same reason. Anchored on the package this guard exists
    # for rather than on a module inside it, so nothing here depends on which files that package
    # happens to hold.
    assert _PACKAGE / "services" / "platforms" / "__init__.py" in modules, "scan found no services"

    offenders: list[str] = []
    for module in modules:
        for reached, path in sorted(_reachable_within_package(module=module).items()):
            if "nextcord" in _import_roots(reached):
                offenders.append(path)
    assert not offenders, f"services reaching nextcord: {sorted(set(offenders))}"


def test_the_layering_scan_reads_relative_and_type_checking_imports() -> None:
    """The scan is only worth its assertions if it sees the forms a violation can be written in.

    No module in the package writes a relative import any more, so that half is asserted on
    source of its own: a resolver that quietly stopped reading `from ..peer.mod import X`
    would pass the tests above while seeing nothing, and that form is the one they exist to
    catch. Pinning it is new rather than preserved — the cog this used to read wrote only the
    single-dot form, which walks up no levels at all, so `..` had never been exercised.
    `cogs/games/blackjack_views.py` still supplies the other half, importing a cog module
    under `TYPE_CHECKING`.
    """
    relative = _imports_in(
        source="from .own_mod import A\nfrom ..peer.mod import B", parent="discordbot.cogs.own"
    )
    assert relative == {
        "discordbot.cogs.own.own_mod",
        "discordbot.cogs.own.own_mod.A",
        "discordbot.cogs.peer.mod",
        "discordbot.cogs.peer.mod.B",
    }

    type_checking = _imported_modules(_COGS / "games" / "blackjack_views.py")
    assert "discordbot.cogs.games.shoe" in type_checking


def test_a_package_init_resolves_relative_imports_against_its_own_package() -> None:
    """`pkg/__init__.py` is inside `pkg`, not beside it.

    No `__init__.py` in the tree uses a relative import today, so nothing else would notice
    this being off by one — and off by one is exactly the direction that hides a peer-cog
    import written as `from ..peer.mod import X`.
    """
    assert _relative_import_base(_COGS / "economy" / "cog.py") == "discordbot.cogs.economy"
    assert _relative_import_base(_COGS / "economy" / "__init__.py") == "discordbot.cogs.economy"
    assert _relative_import_base(_COGS / "gen_reply" / "link_sources" / "__init__.py") == (
        "discordbot.cogs.gen_reply.link_sources"
    )


def test_every_cog_directory_is_shaped_like_a_cog() -> None:
    """The loader's rule, asserted on the tree so a half-finished move fails here first."""
    for entry in sorted(_COGS.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        assert (entry / "__init__.py").is_file(), (
            f"{entry.name}: a cog directory needs __init__.py"
        )
        assert (entry / "cog.py").is_file(), f"{entry.name}: a cog directory needs cog.py"
