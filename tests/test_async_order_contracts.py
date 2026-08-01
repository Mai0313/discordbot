"""Rejects undocumented call-order assertions in async tests.

An async test double that appends calls to a list observes whichever coroutine or worker reaches
it first. Comparing that recorder directly with an ordered sequence silently turns scheduler
timing into a contract. Tests should compare a stable invariant instead, or document the real
production ordering contract beside the assertion.
"""

from __future__ import annotations

from io import StringIO
import ast
from typing import TYPE_CHECKING
from pathlib import Path
import tokenize

from pydantic import Field, BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Iterator

_TESTS = Path(__file__).resolve().parent
_MUTATING_METHODS = frozenset({"__iadd__", "append", "extend", "insert"})
_ORDER_INDEPENDENT_CALLS = frozenset({
    "Counter",
    "all",
    "any",
    "dict",
    "frozenset",
    "len",
    "set",
    "sorted",
})
_ORDER_CONTRACT_MARKER = "# order-contract:"
_ORDER_PRESERVING_TRANSFORMS = frozenset({
    "copy",
    "filter",
    "iter",
    "list",
    "map",
    "reversed",
    "tuple",
})


class OrderAssertion(BaseModel):
    """One exact recorder-order assertion without a documented contract.

    Attributes:
        test_name: Name of the test function holding the assertion.
        lineno: 1-indexed line the assertion starts on.
        recorders: Recorder names the assertion compares in order.
    """

    model_config = ConfigDict(frozen=True)

    test_name: str = Field(
        ...,
        description="Name of the test function holding the assertion.",
        examples=["test_bad_order"],
    )
    lineno: int = Field(..., description="1-indexed line the assertion starts on.", examples=[13])
    recorders: tuple[str, ...] = Field(
        ..., description="Recorder names the assertion compares in order.", examples=[("calls",)]
    )


def _called_name(node: ast.expr) -> str:
    """Returns the bare name of a call target, or an empty string for another expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_list_initializer(node: ast.expr | None) -> bool:
    """Whether an assignment creates a list that can record test-double calls."""
    if isinstance(node, (ast.List, ast.ListComp)):
        return True
    if isinstance(node, ast.IfExp):
        return _is_list_initializer(node.body) and _is_list_initializer(node.orelse)
    return isinstance(node, ast.Call) and _called_name(node.func) == "list"


def _list_assignment_names(target: ast.expr, value: ast.expr | None) -> set[str]:
    """Returns names assigned list initializers, including unpacked assignments."""
    if isinstance(target, ast.Name):
        return {target.id} if _is_list_initializer(value) else set()
    if not (
        isinstance(target, (ast.List, ast.Tuple))
        and isinstance(value, (ast.List, ast.Tuple))
        and len(target.elts) == len(value.elts)
    ):
        return set()
    names: set[str] = set()
    for child_target, child_value in zip(target.elts, value.elts, strict=True):
        names.update(_list_assignment_names(child_target, child_value))
    return names


class _LocalListRecorderVisitor(ast.NodeVisitor):
    """Finds list initializers in a test body without entering nested scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Records one annotated list assignment."""
        self.names.update(_list_assignment_names(node.target, node.value))

    def visit_Assign(self, node: ast.Assign) -> None:
        """Records direct, chained, and unpacked list assignments."""
        for target in node.targets:
            self.names.update(_list_assignment_names(target, node.value))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Does not enter nested async test doubles."""
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Does not enter nested fake classes."""
        del node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Does not enter nested sync test doubles."""
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Does not enter lambda test doubles."""
        del node


def _local_list_recorders(test: ast.AsyncFunctionDef) -> set[str]:
    """Returns lists declared anywhere in an async test's own scope."""
    visitor = _LocalListRecorderVisitor()
    for statement in test.body:
        visitor.visit(statement)
    return visitor.names


class _CallableScopeVisitor(ast.NodeVisitor):
    """Collects bindings and child scopes without entering nested callables."""

    def __init__(self) -> None:
        self.bound_names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()
        self.callables: list[ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda] = []
        self.classes: list[ast.ClassDef] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Records a nested async function and its binding."""
        self.bound_names.add(node.name)
        self.callables.append(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Records a nested class and its binding."""
        self.bound_names.add(node.name)
        self.classes.append(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Records an exception target as a local binding."""
        if node.name:
            self.bound_names.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Records a nested sync function and its binding."""
        self.bound_names.add(node.name)
        self.callables.append(node)

    def visit_Global(self, node: ast.Global) -> None:
        """Records names explicitly resolved at module scope."""
        self.global_names.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        """Records imported names as local bindings."""
        self.bound_names.update(
            alias.asname or alias.name.partition(".")[0] for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Records imported names as local bindings."""
        self.bound_names.update(alias.asname or alias.name for alias in node.names)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Records a nested lambda without entering its body."""
        self.callables.append(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Records assignment targets in the current callable scope."""
        if isinstance(node.ctx, ast.Store):
            self.bound_names.add(node.id)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Records names explicitly resolved in an enclosing callable scope."""
        self.nonlocal_names.update(node.names)

    def _visit_comprehension(
        self, generators: list[ast.comprehension], *expressions: ast.expr
    ) -> None:
        """Visits comprehension expressions without leaking their iteration targets."""
        for generator in generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for expression in expressions:
            self.visit(expression)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Keeps comprehension iteration variables out of the containing scope."""
        self._visit_comprehension(node.generators, node.key, node.value)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Keeps generator iteration variables out of the containing scope."""
        self._visit_comprehension(node.generators, node.elt)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Keeps list-comprehension iteration variables out of the containing scope."""
        self._visit_comprehension(node.generators, node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Keeps set-comprehension iteration variables out of the containing scope."""
        self._visit_comprehension(node.generators, node.elt)


def _argument_names(arguments: ast.arguments) -> set[str]:
    """Returns every local name introduced by a callable signature."""
    names = {argument.arg for argument in (*arguments.posonlyargs, *arguments.args)}
    names.update(argument.arg for argument in arguments.kwonlyargs)
    if arguments.vararg:
        names.add(arguments.vararg.arg)
    if arguments.kwarg:
        names.add(arguments.kwarg.arg)
    return names


def _default_recorder_aliases(
    function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    *,
    outer_environment: dict[str, str],
) -> dict[str, str]:
    """Maps parameters whose defaults capture an outer recorder."""
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    aliases: dict[str, str] = {}
    if arguments.defaults:
        for argument, default in zip(
            positional[-len(arguments.defaults) :], arguments.defaults, strict=True
        ):
            if isinstance(default, ast.Name) and default.id in outer_environment:
                aliases[argument.arg] = outer_environment[default.id]
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        if isinstance(default, ast.Name) and default.id in outer_environment:
            aliases[argument.arg] = outer_environment[default.id]
    return aliases


class _RecorderWriteVisitor(ast.NodeVisitor):
    """Finds mutations that resolve to test-scope recorder bindings."""

    def __init__(self, *, environment: dict[str, str], nonlocal_names: set[str]) -> None:
        self.environment = environment
        self.nonlocal_names = nonlocal_names
        self.found: set[str] = set()

    def _record_subscripts(self, target: ast.expr) -> None:
        for child in ast.walk(target):
            if isinstance(child, ast.Subscript):
                recorder = self.environment.get(_root_subscript_name(child))
                if recorder:
                    self.found.add(recorder)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Records subscript mutation and a genuine nonlocal rebinding."""
        self._record_subscripts(node.target)
        self._record_nonlocal_rebinding(node.target, node.value)
        if node.value:
            self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Records subscript mutation and a genuine nonlocal rebinding."""
        for target in node.targets:
            self._record_subscripts(target)
            self._record_nonlocal_rebinding(target, node.value)
        self.visit(node.value)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Does not enter a child callable with a different lexical scope."""
        del node

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Records in-place addition to a captured recorder."""
        if isinstance(node.op, ast.Add):
            if isinstance(node.target, ast.Name):
                recorder = self.environment.get(node.target.id)
                if recorder:
                    self.found.add(recorder)
            else:
                self._record_subscripts(node.target)
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        """Records a mutating method call on a captured recorder."""
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATING_METHODS
            and isinstance(node.func.value, ast.Name)
        ):
            recorder = self.environment.get(node.func.value.id)
            if recorder:
                self.found.add(recorder)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Does not enter a child class containing separate callable scopes."""
        del node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Does not enter a child callable with a different lexical scope."""
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Does not enter a child lambda with a different lexical scope."""
        del node

    def _record_nonlocal_rebinding(self, target: ast.expr, value: ast.expr | None) -> None:
        if not (
            isinstance(target, ast.Name)
            and target.id in self.nonlocal_names
            and target.id in self.environment
            and value is not None
        ):
            return
        referenced = _referenced_recorders(value, recorders=set(self.environment))
        if self.environment[target.id] in {self.environment[name] for name in referenced}:
            self.found.add(self.environment[target.id])


def _scope_details(
    function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> _CallableScopeVisitor:
    """Returns bindings and child scopes owned by one callable."""
    visitor = _CallableScopeVisitor()
    visitor.bound_names.update(_argument_names(function.args))
    if isinstance(function, ast.Lambda):
        visitor.visit(function.body)
    else:
        for statement in function.body:
            visitor.visit(statement)
    return visitor


class _NestedRecorderWriteFinder:
    """Resolves recorder mutations through nested lexical scopes."""

    def __init__(self, *, recorders: set[str]) -> None:
        self.initial_environment = {recorder: recorder for recorder in recorders}
        self.found: set[str] = set()

    def visit_callable(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
        outer_environment: dict[str, str],
    ) -> None:
        """Records writes in one callable and recursively visits its children."""
        scope = _scope_details(function)
        local_names = scope.bound_names - scope.nonlocal_names
        environment = {
            name: recorder
            for name, recorder in outer_environment.items()
            if name not in local_names and name not in scope.global_names
        }
        environment.update(
            _default_recorder_aliases(function, outer_environment=outer_environment)
        )
        writes = _RecorderWriteVisitor(
            environment=environment, nonlocal_names=scope.nonlocal_names
        )
        if isinstance(function, ast.Lambda):
            writes.visit(function.body)
        else:
            for statement in function.body:
                writes.visit(statement)
        self.found.update(writes.found)
        for child in scope.callables:
            self.visit_callable(child, environment)
        for child_class in scope.classes:
            self.visit_class(child_class, environment)

    def visit_class(self, node: ast.ClassDef, outer_environment: dict[str, str]) -> None:
        """Visits methods without treating a class namespace as a closure."""
        scope = _CallableScopeVisitor()
        for statement in node.body:
            scope.visit(statement)
        for child in scope.callables:
            self.visit_callable(child, outer_environment)
        for child_class in scope.classes:
            self.visit_class(child_class, outer_environment)

    def find(self, test: ast.AsyncFunctionDef) -> set[str]:
        """Returns recorder mutations reachable below the async test scope."""
        test_scope = _scope_details(test)
        for function in test_scope.callables:
            self.visit_callable(function, self.initial_environment)
        for nested_class in test_scope.classes:
            self.visit_class(nested_class, self.initial_environment)
        return self.found


def _nested_function_writes(test: ast.AsyncFunctionDef, *, recorders: set[str]) -> set[str]:
    """Returns test-scope recorders mutated from lexically nested callables."""
    return _NestedRecorderWriteFinder(recorders=recorders).find(test)


def _root_subscript_name(node: ast.Subscript) -> str:
    """Returns the name at the root of a chained subscription."""
    value: ast.expr = node
    while isinstance(value, ast.Subscript):
        value = value.value
    return value.id if isinstance(value, ast.Name) else ""


def _referenced_recorders(node: ast.AST, *, recorders: set[str]) -> set[str]:
    """Returns recorder names referenced anywhere under an expression."""
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and child.id in recorders
    }


def _normalizer_encodes_positions(node: ast.AST, *, recorders: set[str]) -> bool:
    """Whether a nested transform preserves recorder positions as output values."""
    safe_nested_calls = _ORDER_INDEPENDENT_CALLS | _ORDER_PRESERVING_TRANSFORMS
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, ast.Subscript) and _root_subscript_name(child) in recorders:
            return True
        if (
            isinstance(child, ast.Call)
            and _called_name(child.func) not in safe_nested_calls
            and _referenced_recorders(child, recorders=recorders)
        ):
            return True
    return False


class _OrderedRecorderVisitor(ast.NodeVisitor):
    """Finds recorder uses whose observed sequence remains part of an expression."""

    def __init__(self, *, recorders: set[str]) -> None:
        self.recorders = recorders
        self.found: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        """Skips calls that deliberately remove or ignore order."""
        if _called_name(
            node.func
        ) in _ORDER_INDEPENDENT_CALLS and not _normalizer_encodes_positions(
            node, recorders=self.recorders
        ):
            return
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Skips one recorded item while retaining slices, which still encode sequence."""
        if _root_subscript_name(node) in self.recorders and not isinstance(node.slice, ast.Slice):
            try:
                index = ast.literal_eval(node.slice)
            except (ValueError, TypeError):
                index = None
            if isinstance(index, int):
                return
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Skips mapping projections unless they retain recorder positions."""
        if not _normalizer_encodes_positions(node, recorders=self.recorders):
            return
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Skips set projections unless they retain recorder positions."""
        if not _normalizer_encodes_positions(node, recorders=self.recorders):
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Records a sequence-preserving reference to a test-double recorder."""
        if node.id in self.recorders:
            self.found.add(node.id)


class _IndexedRecorderVisitor(ast.NodeVisitor):
    """Finds explicit positions read from a recorder inside one assertion."""

    def __init__(self, *, recorders: set[str]) -> None:
        self.recorders = recorders
        self.found: set[tuple[str, int]] = set()

    def visit_Call(self, node: ast.Call) -> None:
        """Skips indexes applied after a deliberate order-independent transform."""
        if _called_name(
            node.func
        ) in _ORDER_INDEPENDENT_CALLS and not _normalizer_encodes_positions(
            node, recorders=self.recorders
        ):
            return
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Records direct integer indexes while ignoring fields inside a recorded item."""
        if isinstance(node.value, ast.Name) and node.value.id in self.recorders:
            try:
                index = ast.literal_eval(node.slice)
            except (ValueError, TypeError):
                index = None
            if isinstance(index, int):
                self.found.add((node.value.id, index))
                return
        self.generic_visit(node)


class _TestAssertionVisitor(ast.NodeVisitor):
    """Collects assertions in a test body without entering its nested test doubles."""

    def __init__(self) -> None:
        self.found: list[ast.Assert] = []

    def visit_Assert(self, node: ast.Assert) -> None:
        """Records one test-body assertion."""
        self.found.append(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Does not treat a nested async test double's assertions as the test's own."""
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Does not enter nested fake classes."""
        del node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Does not treat a nested sync test double's assertions as the test's own."""
        del node


def _ordered_recorders(node: ast.expr, *, recorders: set[str]) -> set[str]:
    """Returns recorder names whose order remains observable in an expression."""
    visitor = _OrderedRecorderVisitor(recorders=recorders)
    visitor.visit(node)
    return visitor.found


def _indexed_recorders(node: ast.expr, *, recorders: set[str]) -> set[tuple[str, int]]:
    """Returns explicit recorder positions inspected by an assertion expression."""
    visitor = _IndexedRecorderVisitor(recorders=recorders)
    visitor.visit(node)
    return visitor.found


def _test_assertions(test: ast.AsyncFunctionDef) -> list[ast.Assert]:
    """Returns assertions owned by the test rather than by a nested test double."""
    visitor = _TestAssertionVisitor()
    for statement in test.body:
        visitor.visit(statement)
    return visitor.found


def _can_encode_multiple_positions(node: ast.expr) -> bool:
    """Whether an expected expression can distinguish the order of two or more records."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts) >= 2 or any(
            isinstance(element, ast.Starred) for element in node.elts
        )
    if isinstance(node, ast.Set):
        return len(node.elts) >= 2 or any(
            isinstance(element, ast.Starred) for element in node.elts
        )
    if isinstance(node, ast.Dict):
        return len(node.keys) >= 2 or any(key is None for key in node.keys)
    if isinstance(node, ast.Constant):
        return False
    return isinstance(
        node,
        (
            ast.Attribute,
            ast.BinOp,
            ast.Call,
            ast.GeneratorExp,
            ast.IfExp,
            ast.DictComp,
            ast.ListComp,
            ast.Name,
            ast.SetComp,
            ast.Subscript,
        ),
    )


def _comments_by_line(source: str) -> dict[int, list[str]]:
    """Returns real Python comments without mistaking marker-like string content for one."""
    comments: dict[int, list[str]] = {}
    for token in tokenize.generate_tokens(StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comments.setdefault(token.start[0], []).append(token.string)
    return comments


def _contract_reason(comments: dict[int, list[str]], *, lineno: int, end_lineno: int) -> str:
    """Returns an inline or immediately preceding order-contract reason."""
    for candidate_lineno in (lineno - 1, *range(lineno, end_lineno + 1)):
        for comment in comments.get(candidate_lineno, []):
            _before, marker, reason = comment.partition(_ORDER_CONTRACT_MARKER)
            if marker and reason.strip():
                return reason.strip()
    return ""


def _async_tests(nodes: list[ast.stmt]) -> Iterator[ast.AsyncFunctionDef]:
    """Yields module-level async tests and async test methods nested in classes."""
    for node in nodes:
        if isinstance(node, ast.AsyncFunctionDef) and node.name.startswith("test_"):
            yield node
        elif isinstance(node, ast.ClassDef):
            yield from _async_tests(node.body)


def _ordered_assertion_recorders(
    assertion: ast.Assert,
    *,
    recorders: set[str],
    indexed: set[tuple[str, int]],
    multi_index_recorders: set[str],
) -> set[str]:
    """Returns recorder names whose positions one assertion distinguishes."""
    ordered = {recorder for recorder, _index in indexed if recorder in multi_index_recorders}
    comparison = assertion.test
    if _normalizer_encodes_positions(comparison, recorders=recorders):
        ordered.update(_ordered_recorders(comparison, recorders=recorders))
    if not (
        isinstance(comparison, ast.Compare)
        and comparison.ops
        and all(isinstance(operator, ast.Eq) for operator in comparison.ops)
    ):
        return ordered
    operands = [comparison.left, *comparison.comparators]
    for index, operand in enumerate(operands):
        if any(
            _can_encode_multiple_positions(other)
            for other_index, other in enumerate(operands)
            if other_index != index
        ):
            ordered.update(_ordered_recorders(operand, recorders=recorders))
    return ordered


def _find_undocumented_order_assertions(source: str) -> list[OrderAssertion]:
    """Finds exact sequence assertions on nested test-double recorders in async tests."""
    tree = ast.parse(source=source)
    comments = _comments_by_line(source)
    findings: list[OrderAssertion] = []
    for test in _async_tests(tree.body):
        local_recorders = _local_list_recorders(test)
        recorders = _nested_function_writes(test, recorders=local_recorders)
        if not recorders:
            continue
        assertions = _test_assertions(test)
        indexed = {
            assertion: _indexed_recorders(assertion.test, recorders=recorders)
            for assertion in assertions
        }
        indexes_by_recorder: dict[str, set[int]] = {recorder: set() for recorder in recorders}
        for uses in indexed.values():
            for recorder, index in uses:
                indexes_by_recorder[recorder].add(index)
        multi_index_recorders = {
            recorder for recorder, indexes in indexes_by_recorder.items() if len(indexes) >= 2
        }
        for assertion in assertions:
            ordered = _ordered_assertion_recorders(
                assertion,
                recorders=recorders,
                indexed=indexed[assertion],
                multi_index_recorders=multi_index_recorders,
            )
            if not ordered or _contract_reason(
                comments,
                lineno=assertion.lineno,
                end_lineno=assertion.end_lineno or assertion.lineno,
            ):
                continue
            findings.append(
                OrderAssertion(
                    test_name=test.name, lineno=assertion.lineno, recorders=tuple(sorted(ordered))
                )
            )
    return findings


def test_the_original_issue_414_assertions_are_rejected() -> None:
    """The old Threads recorder checks fail deterministically without scheduler timing."""
    source = """
async def test_bad_order():
    written: list[str] = []
    uploaded: list[tuple[str, bytes]] = []

    def fake_download(name: str) -> None:
        written.append(name)

    async def fake_upload(name: str, data: bytes) -> None:
        uploaded.append((name, data))

    await run_concurrently(fake_download, fake_upload)
    assert written == ["threads_video_0.mp4", "threads_quoted_video_0.mp4"]
    assert uploaded == [("threads_video_0.mp4", b"target"), ("threads_quoted_video_0.mp4", b"quoted")]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_bad_order", lineno=13, recorders=("written",)),
        OrderAssertion(test_name="test_bad_order", lineno=14, recorders=("uploaded",)),
    ]


def test_order_independent_recorder_assertions_are_accepted() -> None:
    """Sorting and mapping recorder output state invariants rather than completion order."""
    source = """
async def test_stable_invariants():
    written: list[str] = []
    uploaded: list[tuple[str, bytes]] = []

    def fake_download(name: str) -> None:
        written.append(name)

    async def fake_upload(name: str, data: bytes) -> None:
        uploaded.append((name, data))

    await run_concurrently(fake_download, fake_upload)
    assert sorted(written) == ["quoted.mp4", "target.mp4"]
    assert dict(uploaded) == {"target.mp4": b"target", "quoted.mp4": b"quoted"}
"""

    assert _find_undocumented_order_assertions(source) == []


def test_set_and_mapping_projections_are_accepted() -> None:
    """Comprehensions that discard sequence order are stable invariants."""
    source = """
async def test_stable_projections():
    uploaded: list[tuple[str, bytes]] = []

    async def fake_upload(name: str, data: bytes) -> None:
        uploaded.append((name, data))

    await run_concurrently(fake_upload)
    assert {name for name, _data in uploaded} == {"target.mp4", "quoted.mp4"}
    assert {name: data for name, data in uploaded} == {
        "target.mp4": b"target",
        "quoted.mp4": b"quoted",
    }
    assert set(uploaded.copy()) == {
        ("target.mp4", b"target"),
        ("quoted.mp4", b"quoted"),
    }
    assert dict(uploaded.copy()) == {
        "target.mp4": b"target",
        "quoted.mp4": b"quoted",
    }
"""

    assert _find_undocumented_order_assertions(source) == []


def test_a_starred_expected_sequence_is_rejected() -> None:
    """Expanding one expected name can still describe the whole recorded sequence."""
    source = """
async def test_starred_expected():
    calls: list[str] = []
    expected = ["first", "second"]

    async def record(value: str) -> None:
        calls.append(value)

    await run_concurrently(record)
    assert calls == [*expected]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_starred_expected", lineno=10, recorders=("calls",))
    ]


def test_dynamic_index_projection_is_rejected() -> None:
    """Rebuilding a recorder by dynamic positions preserves its sequence."""
    source = """
async def test_dynamic_indexes():
    calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await run_concurrently(record)
    assert [calls[i] for i in range(2)] == ["first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_dynamic_indexes", lineno=9, recorders=("calls",))
    ]


def test_positional_boolean_aggregate_is_rejected() -> None:
    """A boolean reduction does not erase position pairings created by zip."""
    source = """
async def test_all_zipped_positions():
    calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await run_concurrently(record)
    assert all(actual == expected for actual, expected in zip(calls, ("first", "second")))
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_all_zipped_positions", lineno=9, recorders=("calls",))
    ]


def test_lambda_test_double_writes_are_scanned() -> None:
    """A lambda can record concurrent calls just like a nested function does."""
    source = """
async def test_lambda_recorder():
    calls: list[str] = []
    record = lambda value: calls.append(value)

    await run_concurrently(record)
    assert calls == ["first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_lambda_recorder", lineno=7, recorders=("calls",))
    ]


def test_chained_sequence_equality_is_rejected() -> None:
    """Every operand in a chained equality can expose the recorder sequence."""
    source = """
async def test_chained_equality():
    calls: list[str] = []
    expected = ["first", "second"]

    async def record(value: str) -> None:
        calls.append(value)

    await run_concurrently(record)
    assert expected == calls == ["first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_chained_equality", lineno=10, recorders=("calls",))
    ]


def test_nonlocal_recorder_rebinding_is_scanned() -> None:
    """Replacing a nonlocal list with its extended value still records call order."""
    source = """
async def test_nonlocal_rebind():
    calls: list[str] = []

    async def record(value: str) -> None:
        nonlocal calls
        calls = [*calls, value]

    await run_concurrently(record)
    assert calls == ["first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_nonlocal_rebind", lineno=10, recorders=("calls",))
    ]


def test_shadowed_nested_bindings_are_not_recorders() -> None:
    """A nested local with the same name is not the test-scope recorder."""
    source = """
async def test_shadowed_bindings():
    calls = ["first", "second"]

    async def consume(calls):
        calls.append("parameter")

    async def replace_locally():
        calls = []
        calls.append("local")

    await consume([])
    await replace_locally()
    assert calls == ["first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == []


def test_default_and_nonlocal_recorder_captures_are_scanned() -> None:
    """Explicit captures still resolve to the test-scope recorder."""
    source = """
async def test_explicit_captures():
    calls: list[str] = []

    async def default_capture(alias=calls):
        alias.append("default")

    async def outer():
        async def closure_capture(value: str) -> None:
            calls.append(value)

        await closure_capture("closure")

    await run_concurrently(default_capture, outer)
    assert calls == ["default", "closure"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_explicit_captures", lineno=15, recorders=("calls",))
    ]


def test_recorder_initialized_under_control_flow_is_scanned() -> None:
    """A test-scope list remains a recorder when setup uses a conditional block."""
    source = """
async def test_conditional_initializer():
    if enabled:
        calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await run_concurrently(record)
    assert calls == ["first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_conditional_initializer", lineno=10, recorders=("calls",))
    ]


def test_a_normalizer_that_preserves_positions_is_rejected() -> None:
    """Sorting indexed records does not erase their original positions."""
    source = """
async def test_indexed_sort():
    calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await run_concurrently(record)
    assert sorted(enumerate(calls)) == [(0, "first"), (1, "second")]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_indexed_sort", lineno=9, recorders=("calls",))
    ]


def test_async_test_methods_inside_classes_are_scanned() -> None:
    """Pytest class methods receive the same recorder-order guard as module tests."""
    source = """
class TestQueue:
    async def test_class_method_order(self):
        calls: list[str] = []

        async def record(value: str) -> None:
            calls.append(value)

        await run_concurrently(record)
        assert calls == ["first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_class_method_order", lineno=10, recorders=("calls",))
    ]


def test_seeded_test_double_recorders_are_scanned() -> None:
    """A setup sentinel does not make later concurrent records ordered."""
    source = """
async def test_seeded_recorder():
    calls = ["setup"]

    async def record(value: str) -> None:
        calls.append(value)

    await run_concurrently(record)
    assert calls == ["setup", "first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_seeded_recorder", lineno=9, recorders=("calls",))
    ]


def test_unordered_containers_that_retain_positions_are_rejected() -> None:
    """A set or mapping cannot erase indexes embedded in its values."""
    source = """
async def test_positional_containers():
    calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await run_concurrently(record)
    assert set(enumerate(calls)) == {(0, "first"), (1, "second")}
    assert dict(enumerate(calls)) == {0: "first", 1: "second"}
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_positional_containers", lineno=9, recorders=("calls",)),
        OrderAssertion(test_name="test_positional_containers", lineno=10, recorders=("calls",)),
    ]


def test_all_does_not_hide_multiple_recorder_indexes() -> None:
    """A boolean reduction still encodes order when its inputs name positions."""
    source = """
async def test_all_positions():
    calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await run_concurrently(record)
    assert all((calls[0] == "first", calls[1] == "second"))
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_all_positions", lineno=9, recorders=("calls",))
    ]


def test_augmented_assignment_is_a_nested_recorder_write() -> None:
    """In-place list concatenation records calls just like extend does."""
    source = """
async def test_augmented_recorder():
    calls: list[str] = []

    async def record(value: str) -> None:
        nonlocal calls
        calls += [value]

    await run_concurrently(record)
    assert calls == ["first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_augmented_recorder", lineno=10, recorders=("calls",))
    ]


def test_a_documented_production_order_contract_is_accepted() -> None:
    """A concise adjacent reason keeps real FIFO behavior testable."""
    source = """
async def test_fifo_contract():
    calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await queue.flush(record)
    # order-contract: queue.flush promises FIFO delivery.
    assert calls == ["first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == []


def test_a_closing_line_order_contract_is_accepted() -> None:
    """A multi-line assertion can carry its rationale on the closing line."""
    source = """
async def test_multiline_fifo_contract():
    calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await queue.flush(record)
    assert calls == [
        "first",
        "second",
    ]  # order-contract: queue.flush promises FIFO delivery.
"""

    assert _find_undocumented_order_assertions(source) == []


def test_an_empty_order_contract_marker_does_not_bypass_the_guard() -> None:
    """The escape hatch requires a reviewable explanation, not a magic suppression token."""
    source = """
async def test_unexplained_order():
    calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await queue.flush(record)
    # order-contract:
    assert calls == ["first", "second"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_unexplained_order", lineno=10, recorders=("calls",))
    ]


def test_marker_text_inside_an_expected_value_does_not_bypass_the_guard() -> None:
    """Only a Python comment can explain a contract; test data containing the marker cannot."""
    source = """
async def test_marker_as_data():
    calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await queue.flush(record)
    assert calls == ["# order-contract: not a comment", "second"]
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_marker_as_data", lineno=9, recorders=("calls",))
    ]


def test_split_position_assertions_are_rejected() -> None:
    """Checking two indexes separately cannot bypass the exact-sequence guard."""
    source = """
async def test_split_order():
    calls: list[str] = []

    async def record(value: str) -> None:
        calls.append(value)

    await queue.flush(record)
    assert calls[0] == "first"
    assert calls[1] == "second"
"""

    assert _find_undocumented_order_assertions(source) == [
        OrderAssertion(test_name="test_split_order", lineno=9, recorders=("calls",)),
        OrderAssertion(test_name="test_split_order", lineno=10, recorders=("calls",)),
    ]


def test_async_test_double_recorders_do_not_assume_undocumented_order() -> None:
    """Every exact call-order assertion either has a contract or uses a stable invariant."""
    findings: list[str] = []
    for path in sorted(_TESTS.rglob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        findings.extend(
            f"{path.relative_to(_TESTS)}:{finding.lineno} "
            f"{', '.join(finding.recorders)} in {finding.test_name}"
            for finding in _find_undocumented_order_assertions(source)
        )
    assert findings == [], (
        "nested async test doubles record completion order, not a contract; compare a mapping, "
        "set, Counter, or sorted value, or add an adjacent '# order-contract: <reason>': "
        f"{findings}"
    )
