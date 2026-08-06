"""Pins the docstring convention `.github/CONTRIBUTING.md` states, across `src/` and `tests/`.

Ruff already owns half the convention: `convention = "google"` shapes the sections, and the
`DOC*` selections in `pyproject.toml` require `Returns:`, `Yields:` and `Raises:` wherever the
body needs one. What ruff has no rule for is the other half, and it is the half that decays
first — a module with no docstring at all, a function with none, and an `Args:` block that has
drifted from the signature it describes. `D417` only fires once a docstring already has an
`Args:` section, so a function that simply never grew one is invisible to it.

That gap is why this file exists rather than more ruff configuration. The convention is only
worth writing down if the first read of an unfamiliar file can be relied on, and #404 wants to
defer per-subsystem detail to exactly these docstrings, so a convention nothing enforces would
be a promise that quietly stops being true.

The `Args:` rule is deliberately narrower in `tests/`: a test's parameters are pytest fixtures
resolved by name, so there is no caller to document for. `tests/helpers/` and `conftest.py` are
the exception inside that exception, being the two places a test file calls into.
"""

import re
import ast
from pathlib import Path
from collections.abc import Iterator

_ROOT = Path(__file__).resolve().parents[1]
_SCANNED_ROOTS = (_ROOT / "src" / "discordbot", _ROOT / "tests")

# Where a complete typed `Args:` block is owed. Everything under `src/`, plus the two places in
# `tests/` a test file calls into rather than receives fixtures from.
_ARGS_REQUIRED = ("src/discordbot/", "tests/helpers/", "tests/conftest.py")

# `name (Type): description`, with the type optional so a missing one is reported as a missing
# type rather than as an unparsable entry. `*args` / `**kwargs` keep their stars.
_ARGS_ENTRY_RE = re.compile(r"^(?P<name>\*{0,2}\w+)\s*(?:\((?P<type>.+)\))?\s*:\s*\S")

# Enough of the tree to prove the scan is not passing because it found nothing. Both numbers sit
# well under the real counts (197 files, 4151 functions when this landed), so ordinary growth or
# a deleted module never trips them, while a broken glob or a walk that stops at the first
# directory does.
_MIN_MODULES = 150
_MIN_FUNCTIONS = 3000

type Function = ast.FunctionDef | ast.AsyncFunctionDef


def _modules() -> list[Path]:
    """Every Python module in scope, skipping bytecode caches and the empty `__init__.py`.

    Returns:
        Each module's path, `src/` before `tests/` and sorted within each.
    """
    found: list[Path] = []
    for root in _SCANNED_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if not path.read_text(encoding="utf-8").strip():
                continue
            found.append(path)
    return found


def _parse(path: Path) -> ast.Module:
    """Parses the module at a path.

    Returns:
        Its syntax tree.
    """
    return ast.parse(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    """Renders a path for offender messages and for the `Args:` scope test.

    Returns:
        The path relative to the repository root, with forward slashes.
    """
    return path.relative_to(_ROOT).as_posix()


def _owes_typed_args(path: Path) -> bool:
    """Decides whether this file's functions owe a complete typed `Args:` block.

    Returns:
        True for everything under `src/`, plus `tests/helpers/` and `tests/conftest.py`.
    """
    return _rel(path).startswith(_ARGS_REQUIRED)


def _documentable(tree: ast.Module) -> Iterator[Function]:
    """Yields every module-level function and method, skipping closures.

    A function defined inside another function body is implementation of its enclosing
    function, and the contract belongs in that function's docstring, so requiring one of its
    own would document the same thing twice. Class bodies are walked at any nesting depth: a
    method on a class declared inside another class is still a method.
    """

    def walk(node: ast.AST, *, inside_function: bool) -> Iterator[Function]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if not inside_function:
                    yield child
                yield from walk(child, inside_function=True)
            else:
                yield from walk(child, inside_function=inside_function)

    yield from walk(tree, inside_function=False)


def _parameters(node: Function) -> list[tuple[str, str | None]]:
    """The `(display_name, annotation)` of every parameter a docstring owes an entry for.

    Ordered the way `Args:` has to list them, which is the signature's own order:
    positional-only, positional-or-keyword, `*args`, keyword-only, then `**kwargs`. `self` and
    `cls` are dropped; the annotation is None where the signature carries none.

    Returns:
        One `(display_name, annotation)` pair per documentable parameter.
    """
    arguments = node.args
    found: list[tuple[str, str | None]] = []

    def annotation_of(argument: ast.arg) -> str | None:
        return ast.unparse(argument.annotation) if argument.annotation else None

    for argument in [*arguments.posonlyargs, *arguments.args]:
        found.append((argument.arg, annotation_of(argument)))
    if arguments.vararg:
        found.append((f"*{arguments.vararg.arg}", annotation_of(arguments.vararg)))
    for argument in arguments.kwonlyargs:
        found.append((argument.arg, annotation_of(argument)))
    if arguments.kwarg:
        found.append((f"**{arguments.kwarg.arg}", annotation_of(arguments.kwarg)))
    return [(name, annotation) for name, annotation in found if name not in ("self", "cls")]


def _args_entries(docstring: str) -> list[tuple[str, str | None]] | None:
    """The `(name, declared_type)` pairs in an `Args:` section, or None when there is none.

    `ast.get_docstring` has already run `inspect.cleandoc`, so the section header sits at
    column 0 and its entries one level in. A line indented further is a wrapped description
    and is skipped, which is what keeps a continuation containing a colon from being read as
    another parameter.

    Returns:
        One `(name, declared_type)` pair per entry, `declared_type` None where the entry carries
        no type, or None when the docstring has no `Args:` section at all.
    """
    lines = docstring.splitlines()
    header = next((i for i, line in enumerate(lines) if line.strip() == "Args:"), None)
    if header is None:
        return None
    base = len(lines[header]) - len(lines[header].lstrip())
    entries: list[tuple[str, str | None]] = []
    for line in lines[header + 1 :]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base:
            break
        if indent != base + 4:
            continue
        match = _ARGS_ENTRY_RE.match(line.strip())
        if match:
            entries.append((match.group("name"), match.group("type")))
    return entries


def _normalised(annotation: str) -> str:
    """An annotation reduced to what the comparison should care about.

    `ast.unparse` re-spells the signature's source: it single-quotes a string annotation and
    normalises the spacing inside a union. Neither is a difference the convention is about, so
    quote style and whitespace are flattened and a whole-annotation string form is unwrapped.

    Returns:
        The annotation with quotes normalised, an outer string form unwrapped, and whitespace
        removed.
    """
    text = annotation.strip().replace("'", '"')
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return re.sub(r"\s+", "", text)


def _report(offenders: list[str], *, limit: int = 25) -> str:
    """Builds a failure message listing offenders, stating the real total when it truncates.

    Returns:
        The offender lines, capped at `limit`, with the dropped count named rather than hidden.
    """
    shown = "\n".join(f"  {line}" for line in offenders[:limit])
    if len(offenders) <= limit:
        return f"{len(offenders)} found:\n{shown}"
    return f"{len(offenders)} found, first {limit} shown:\n{shown}\n  ... and {len(offenders) - limit} more"


def test_every_module_carries_a_docstring() -> None:
    """A module docstring is the cheapest orientation an unfamiliar file can offer."""
    offenders = [_rel(path) for path in _modules() if not ast.get_docstring(_parse(path))]
    assert not offenders, f"modules with no docstring: {_report(offenders)}"


def test_every_function_and_method_carries_a_docstring() -> None:
    """Signatures do not carry side effects, failure behavior or ordering requirements."""
    offenders: list[str] = []
    for path in _modules():
        for node in _documentable(_parse(path)):
            if not ast.get_docstring(node):
                offenders.append(f"{_rel(path)}:{node.lineno}:{node.name}")
    assert not offenders, f"functions with no docstring: {_report(offenders)}"


def _documented_functions_owing_args() -> Iterator[tuple[str, Function, str]]:
    """Yields `(location, node, docstring)` for every documented function that owes typed `Args:`.

    A function with no docstring at all is left to the test above, so a single missing docstring
    is reported once rather than as three separate failures.
    """
    for path in _modules():
        if not _owes_typed_args(path):
            continue
        for node in _documentable(_parse(path)):
            docstring = ast.get_docstring(node)
            if docstring:
                yield f"{_rel(path)}:{node.lineno}:{node.name}", node, docstring


def _mistyped_entry(
    *, where: str, name: str, annotation: str | None, declared: str | None
) -> str | None:
    """The complaint about one entry's type, or None when it matches the signature.

    Args:
        where (str): The `path:line:name` this entry belongs to.
        name (str): The parameter name, `*args` and `**kwargs` keeping their stars.
        annotation (str | None): The signature's annotation, None where it carries none.
        declared (str | None): The type in the docstring entry, None where it declares none.

    Returns:
        The offender line, or None when the entry is in step with the signature.
    """
    if annotation is None:
        return f"{where}: {name} has no annotation to copy" if declared else None
    if declared is None:
        return f"{where}: {name} has no type"
    if _normalised(declared) != _normalised(annotation):
        return f"{where}: {name} says ({declared}), signature has {annotation}"
    return None


def test_every_args_block_names_the_signature_it_documents() -> None:
    """An `Args:` block is owed for every parameter, in the order the signature has them.

    Order matters as much as membership: two blocks in the tree listed the right names in an
    order the signature never had, which reads as maintained while pointing the caller at the
    wrong argument.
    """
    offenders: list[str] = []
    for where, node, docstring in _documented_functions_owing_args():
        expected = [name for name, _ in _parameters(node)]
        if not expected:
            continue
        entries = _args_entries(docstring)
        if entries is None:
            offenders.append(f"{where}: no Args: section")
            continue
        documented = [name for name, _ in entries]
        if documented != expected:
            offenders.append(f"{where}: documents {documented}, signature has {expected}")
    assert not offenders, f"Args: blocks out of step with their signature: {_report(offenders)}"


def test_every_args_entry_carries_the_annotation_from_the_signature() -> None:
    """The type is transcribed, not paraphrased.

    A description can say what a value means; only the annotation says what may be passed, and
    the entries that drift do so by dropping `| None` or collapsing a precise generic to a bare
    container. Copying it verbatim is what makes the block answerable without the signature.
    """
    offenders: list[str] = []
    for where, node, docstring in _documented_functions_owing_args():
        parameters = _parameters(node)
        entries = _args_entries(docstring)
        if entries is None or [name for name, _ in entries] != [name for name, _ in parameters]:
            continue  # reported by the test above
        for (name, annotation), (_, declared) in zip(parameters, entries, strict=True):
            offence = _mistyped_entry(
                where=where, name=name, annotation=annotation, declared=declared
            )
            if offence:
                offenders.append(offence)
    assert not offenders, f"Args: entries out of step with their annotation: {_report(offenders)}"


def test_the_scan_covers_the_tree_it_claims_to() -> None:
    """Every assertion above passes on an empty scan, so the scan's own reach is asserted."""
    modules = _modules()
    assert len(modules) >= _MIN_MODULES, f"only {len(modules)} modules scanned"

    functions = sum(len(list(_documentable(_parse(path)))) for path in modules)
    assert functions >= _MIN_FUNCTIONS, f"only {functions} functions scanned"

    roots = {_rel(path).split("/")[0] for path in modules}
    assert roots == {"src", "tests"}, f"unexpected scan roots: {sorted(roots)}"


def test_the_scan_reads_the_signature_forms_the_tree_uses() -> None:
    """The parameter walk and the `Args:` reader, checked against the shapes they have to handle.

    Asserted against source written here rather than against a file in the tree, so moving the
    one module that happens to use `*args` today cannot quietly disarm the checks above.
    """
    source = '''
class Holder:
    async def run(self, first, /, second: "Later", *rest: int, flag: bool = False, **extra: str):
        """Summary.

        Args:
            first: No annotation to copy.
            second (Later): A string annotation in the signature.
            *rest (int): Collected positionals.
            flag (bool): Keyword-only, and its description
                wraps onto a second line: with a colon in it.
            **extra (str): Collected keywords.

        Returns:
            Nothing in particular.
        """

        def helper(value: int) -> int:
            return value
'''
    tree = ast.parse(source)
    nodes = list(_documentable(tree))
    assert [node.name for node in nodes] == ["run"], "closures must not be documentable"

    assert _parameters(nodes[0]) == [
        ("first", None),
        ("second", "'Later'"),
        ("*rest", "int"),
        ("flag", "bool"),
        ("**extra", "str"),
    ]

    docstring = ast.get_docstring(nodes[0])
    assert docstring is not None
    assert _args_entries(docstring) == [
        ("first", None),
        ("second", "Later"),
        ("*rest", "int"),
        ("flag", "bool"),
        ("**extra", "str"),
    ]

    assert _normalised("'Later'") == _normalised("Later")
    assert _normalised("int|None") == _normalised("int | None")
    assert _normalised("Literal['a', 'b']") == _normalised('Literal["a", "b"]')
    assert _normalised("Sequence[bytes | str]") != _normalised("list")


def test_the_args_reader_finds_no_section_when_there_is_none() -> None:
    """A docstring that only mentions the word must not be read as declaring parameters."""
    assert _args_entries("Summary.\n\nPassed straight to Args: further down the stack.") is None
    assert _args_entries("Summary.\n\nArgs:\n    value (int): A parameter.") == [("value", "int")]
