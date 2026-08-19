"""Pins the import direction between `cogs/`, `services/`, `utils/` and `typings/`.

The layering is what the cog-per-directory layout buys: a cog directory holds only what
that cog uses, so "where does this feature live" has one answer. Nothing else enforces it
— an import reaching sideways into a peer cog's directory runs perfectly well and only
shows up as a tangle months later, which is the state this replaced.
"""

import ast
from pathlib import Path

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
    """Returns every `discordbot.*` module name a file imports, relative imports resolved.

    Reads `TYPE_CHECKING` and function-local imports too: they are still edges in the
    dependency graph, and the one cog-to-cog import this repo ever had was a
    `TYPE_CHECKING` one that never executes and so no test could otherwise see.
    """
    parent = _relative_import_base(module)

    found: set[str] = set()
    for node in ast.walk(ast.parse(source=module.read_text(encoding="utf-8"))):
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
    return {name for name in found if name.startswith("discordbot.")}


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


def test_the_layering_scan_reads_relative_and_type_checking_imports() -> None:
    """The scan is only worth its assertions if it sees the forms the tree actually uses.

    `cogs/maplestory/` is the one cog using relative imports, and
    `cogs/games/blackjack_views.py` imports a cog module under `TYPE_CHECKING`. A scan that
    silently skipped either would pass the tests above while seeing nothing.
    """
    relative = _imported_modules(_COGS / "maplestory" / "cog.py")
    assert "discordbot.cogs.maplestory.views" in relative

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
