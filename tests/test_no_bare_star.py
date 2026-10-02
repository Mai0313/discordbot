"""No signature declares a bare `*` unless an external API needs it.

Calling with keyword arguments is the convention every call site follows, so a `*` that only
forces it adds a rule the code already keeps. A signature an external API holds to keyword-only
parameters is the one exception, and each goes in `_ALLOWED` with its reason.
"""

import ast

from tests.helpers.source_tree import REPO_ROOT, python_modules

_ROOTS = ("src", "scripts", "tests")
# `path::qualified.name` -> the external API that needs its keyword-only marker.
_ALLOWED: dict[str, str] = {}


def _bare_stars(node: ast.AST, prefix: str) -> list[tuple[str, int]]:
    """Every function under `node` declaring a bare `*`, as `(qualified name, line)` pairs."""
    found: list[tuple[str, int]] = []
    for child in ast.iter_child_nodes(node):
        inner = prefix
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            inner = f"{prefix}{child.name}."
        elif isinstance(child, ast.Lambda):
            inner = f"{prefix}<lambda>."
        if (
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            and child.args.kwonlyargs
            and child.args.vararg is None
        ):
            found.append((inner.removesuffix("."), child.lineno))
        found += _bare_stars(node=child, prefix=inner)
    return found


def test_no_signature_declares_a_bare_star() -> None:
    """A bare `*` outside `_ALLOWED` fails, named by its path, line and qualified name."""
    offences: list[str] = []
    for root in _ROOTS:
        modules = python_modules(root=REPO_ROOT / root)
        assert modules, f"no modules under {root}/"
        for path in modules:
            relative = path.relative_to(REPO_ROOT).as_posix()
            tree = ast.parse(source=path.read_text(encoding="utf-8"))
            offences += [
                f"{relative}:{line} {name}"
                for name, line in _bare_stars(node=tree, prefix="")
                if f"{relative}::{name}" not in _ALLOWED
            ]

    assert offences == [], "bare `*` in a signature:\n" + "\n".join(offences)
