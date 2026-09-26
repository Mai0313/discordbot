"""Keeps `utils/scratch_dir.py::scratch_directory` the only way the package opens a scratch dir.

The rule is not style. `tempfile.TemporaryDirectory` hands a failed removal to whoever owned the
`with` block, and every site that wants one here has already told the user what happened by the
time the teardown runs, so that exception relabels a delivered file as undelivered, replaces a
timeout's own report with a generic failure, or discards a result the block had already finished
computing. #558 collected four sites onto the helper and deliberately left the three link-source
builders on the plain class; #561 measured what the carve-out cost and moved those too. A rule
that has already decayed once earns a scan rather than a sentence.

The scan is blunt on purpose. Any attribute named `TemporaryDirectory` or `mkdtemp` is an offence
whatever it is read off, because nothing else here owns those names and a scan that first proved
the object was `tempfile` would miss an aliased import. `mkdtemp` is named on its own account
rather than for completeness: it is the same directory with the removal written by hand, so a rule
naming only the class would be naming the way around itself. And the scan reads `src/discordbot`
alone — `scripts/` and `tests/` open temp directories whose teardown they do own.
"""

import ast
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "discordbot"
_SCRATCH_MODULE = _PACKAGE / "utils" / "scratch_dir.py"

# The two ways of saying "a temp directory whose teardown is mine to write".
_SCRATCH_FACTORIES = frozenset({"TemporaryDirectory", "mkdtemp"})


def _modules() -> list[Path]:
    """Every module under the package except the one allowed to open a scratch directory."""
    return sorted(
        path
        for path in _PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts and path != _SCRATCH_MODULE
    )


def _offences_in(module: Path) -> list[str]:
    """Every scratch-directory factory one module reaches, as `path:line what` strings."""
    found: list[str] = []
    relative = module.relative_to(_PACKAGE.parent)
    for node in ast.walk(ast.parse(source=module.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute) and node.attr in _SCRATCH_FACTORIES:
            found.append(f"{relative}:{node.lineno} .{node.attr}")
        elif isinstance(node, ast.ImportFrom) and node.module == "tempfile":
            found.extend(
                f"{relative}:{node.lineno} from tempfile import {alias.name}"
                for alias in node.names
                if alias.name in _SCRATCH_FACTORIES
            )
    return found


def test_a_scratch_directory_is_only_ever_opened_through_the_shared_helper() -> None:
    """A scratch directory whose teardown is not the helper's speaks for work it never did.

    Seven paths here abandon a worker `asyncio.to_thread` cannot cancel, so their removal walks
    a tree something is still writing into and can raise `ENOTEMPTY`. The helper reports that
    instead of raising it; the plain class raises it at a caller who has already answered.
    """
    offences = [offence for module in _modules() for offence in _offences_in(module)]
    assert offences == [], (
        "a scratch directory must come from discordbot/utils/scratch_dir.py::scratch_directory: "
        + ", ".join(offences)
    )
