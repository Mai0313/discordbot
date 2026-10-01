"""Keeps a model field's description in one place.

A pydantic field can be documented twice — in `Field(description=...)` and again in a Google-style
`Attributes:` entry in the class docstring — and nothing keeps the two in step. They drifted: when
this scan was written 73 models carried both, and 77 of those field pairs no longer agreed, in
both directions. Some entries had kept a reason the `Field` dropped, some `Field` descriptions had
grown one the entry never got, and nothing marked either copy as the stale one, so a reader who
opened the docstring and a reader who opened the field came away believing different things.

`Field(description=...)` is the copy that survives: it reaches the JSON schema a structured call
sends, and for the handful of prompt-bearing models it is what the answer model actually reads.

The scan is narrow on purpose. An `Attributes:` entry naming something that is NOT a `Field` — a
`ClassVar`, a property, a plain annotation pydantic ignores — has no second copy to disagree with
and is left alone. Only a name that is both is an offence, in either Google entry form, `name:`
or `name (type):`, anywhere in the package, the scripts or the tests except the template-owned
`scripts/gen_docs.py`.
"""

import re
import ast
from pathlib import Path

from tests.helpers.source_tree import PACKAGE, REPO_ROOT, python_modules

_MODEL_BASES = frozenset({"BaseModel", "BaseSettings"})
_ENTRY = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)(?:\s*\([^)]*\))?:\s")
_ROOTS = (PACKAGE, REPO_ROOT / "scripts", REPO_ROOT / "tests")
# Its `Attributes:` block comes from the repository template, which still carries the duplicate, so
# the next template sync would undo a local fix; fix it in the template, then drop this exclusion.
_TEMPLATE_OWNED = frozenset({REPO_ROOT / "scripts" / "gen_docs.py"})


def _classes(*, root: Path) -> list[tuple[Path, ast.ClassDef]]:
    """Every class under `root`, one entry each.

    A list rather than a name-keyed map because two modules can define the same name, and a map
    would scan whichever came last and skip the rest. `_by_name` below is still a map, but only to
    walk bases with — a name that resolves to the wrong one of two same-named models still
    resolves to a model, and nothing here reads a base's fields.
    """
    found: list[tuple[Path, ast.ClassDef]] = []
    for path in python_modules(root=root):
        for node in ast.walk(ast.parse(source=path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                found.append((path, node))
    return found


def _base_names(node: ast.ClassDef) -> list[str]:
    """The bases of one class, unwrapping a generic subscript to the class it parameterises."""
    names: list[str] = []
    for declared in node.bases:
        base = declared.value if isinstance(declared, ast.Subscript) else declared
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _is_model(name: str, by_name: dict[str, ast.ClassDef], seen: set[str]) -> bool:
    """Whether a class reaches `BaseModel` or `BaseSettings` through any chain of bases."""
    if name in _MODEL_BASES:
        return True
    if name in seen or name not in by_name:
        return False
    seen.add(name)
    return any(_is_model(base, by_name, seen) for base in _base_names(by_name[name]))


def _described_fields(node: ast.ClassDef) -> set[str]:
    """The fields of one class that carry a `Field(description=...)`."""
    found: set[str] = set()
    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        value = statement.value
        if not isinstance(value, ast.Call):
            continue
        if getattr(value.func, "id", getattr(value.func, "attr", "")) != "Field":
            continue
        if any(keyword.arg == "description" for keyword in value.keywords):
            found.add(statement.target.id)
    return found


def _documented_attributes(node: ast.ClassDef) -> set[str]:
    """The names an `Attributes:` block in this class's docstring gives an entry to."""
    lines = (ast.get_docstring(node) or "").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Attributes:")
    except StopIteration:
        return set()
    names: set[str] = set()
    indent = None
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        here = len(line) - len(line.lstrip())
        if indent is None:
            indent = here
        if here < indent:
            break
        entry = _ENTRY.match(line)
        if entry and here == indent:
            names.add(entry.group(1))
    return names


def test_no_field_is_described_twice() -> None:
    """A field with a `Field(description=)` must not also have an `Attributes:` entry."""
    classes = [
        (path, node)
        for root in _ROOTS
        for path, node in _classes(root=root)
        if path not in _TEMPLATE_OWNED
    ]
    by_name = {node.name: node for _, node in classes}
    scanned: list[str] = []
    offenders: list[str] = []
    for path, node in classes:
        if not any(_is_model(base, by_name, set()) for base in _base_names(node)):
            continue
        scanned.append(f"{path.name}::{node.name}")
        duplicated = _documented_attributes(node) & _described_fields(node)
        if duplicated:
            offenders.append(f"{path.relative_to(REPO_ROOT)}::{node.name} — {sorted(duplicated)}")

    assert "models.py::RouteClassification" in scanned, "the walk found no models"
    assert not offenders, (
        "these fields are documented twice and the two copies will drift; keep the "
        f"`Field(description=)` and drop the `Attributes:` entry: {sorted(offenders)}"
    )


def test_the_walk_keeps_every_class_that_shares_a_name(tmp_path: Path) -> None:
    """Two modules defining one name both reach the scan, so neither can hide an offence."""
    for module in ("first", "second"):
        (tmp_path / f"{module}.py").write_text(
            data="class Twin(BaseModel):\n    pass\n", encoding="utf-8"
        )

    walked = [(path.name, node.name) for path, node in _classes(root=tmp_path)]

    assert walked == [("first.py", "Twin"), ("second.py", "Twin")]
