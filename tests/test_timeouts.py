"""Keeps `typings/timeouts.py` the single place a wall-clock bound is written down.

Collecting the bounds once is easy; the state this replaced is what a collection decays back
into. A bound spelled at its call site as a bare `timeout=15` is invisible to anyone who does
not already know it is there, which is how three download paths ended up with none and how the
same 8.0 ended up in two files describing the same Discord edit.

So the scan reads the shapes a bound is actually written in here, not just the obvious one.
`asyncio.timeout(delay=...)` is the dominant idiom in this package and its keyword is `delay`,
not `timeout`; five of the bounds this collected were pydantic `Field(default=...)` values and
one more a parameter default; a deadline is often a local `<name>_timeout_seconds = 15.0`
rather than a module constant; and yt-dlp's is a string key in a params dict. A rule that only
knew `timeout=` would pass every one of them.

The allowlist is the scope rule made mechanical rather than an escape hatch. Both entries are
bounds whose expiry is designed behaviour rather than a failure — a Discord panel closing
after nobody touched it, a just-published file becoming reapable — which is the line the
collection drew, and each is named here so the exclusion stays a decision someone made.
"""

import ast
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "discordbot"
_TIMEOUTS_MODULE = _PACKAGE / "typings" / "timeouts.py"

# Where the bound sits positionally in the `asyncio` helpers that take one. `asyncio.wait` is
# absent because its `timeout` is keyword-only, which the keyword rule already reads.
_DEADLINE_POSITIONS = {"timeout": 0, "wait_for": 1}

# Keyword arguments that carry a bound wherever they appear.
_TIMEOUT_KEYWORDS = frozenset({"timeout", "timeout_seconds"})

# `delay` names a bound only in `asyncio.timeout`. Elsewhere it is pacing (`asyncio.sleep`),
# which is out of scope.
_DELAY_CALLS = frozenset({"timeout"})

# Names that carry a bound. Read case-insensitively so a module constant, a function-local
# variable and a model field are held to the same rule. `_timeout` is here on its own account
# rather than as a prefix of the first entry: `DouyinDownloader.download_timeout` was one of
# the bounds this collected, and it matches nothing longer.
_BOUND_NAME_SUFFIXES = ("_timeout", "_timeout_seconds", "_deadline_seconds", "_grace_seconds")

# Discord view and modal idle expiry: the timer that closes an untouched panel. Out of scope by
# the collection's own rule — expiry is the design, not a failure — and deliberately left beside
# the view it belongs to, which is also why every repeated copy of 180 here stays a copy.
_VIEW_IDLE_EXPIRY = frozenset({
    "BLACKJACK_ACTION_TIMEOUT_SECONDS",
    "DRAGON_GATE_ACTION_TIMEOUT_SECONDS",
    "FEEDBACK_VIEW_TIMEOUT_SECONDS",
    "LOAN_PROPOSAL_TIMEOUT_SECONDS",
    "MEMORY_VIEW_TIMEOUT_SECONDS",
    "REPORT_FORM_TIMEOUT_SECONDS",
})

# How long the hosted-media reaper leaves a just-published file alone. Out of scope for the
# same reason `MEDIA_HOSTING_RETENTION_HOURS` is: nothing fails when it elapses, the file
# simply stops being protected. It reads as a bound only because of how it is named.
_SWEEP_PROTECTION = frozenset({"_EVICTION_GRACE_SECONDS"})

# Names the scan sees but the scope rule excludes. Listing them is what keeps each exclusion a
# decision someone made rather than a shape the scan happens not to see.
_OUT_OF_SCOPE = _VIEW_IDLE_EXPIRY | _SWEEP_PROTECTION


def _modules() -> list[Path]:
    """Every module under the package except the one the bounds are allowed to live in."""
    return sorted(
        path
        for path in _PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts and path != _TIMEOUTS_MODULE
    )


def _is_number(node: ast.expr) -> bool:
    """Whether an expression is a bare numeric literal, negatives included.

    `bool` is excluded explicitly because it subclasses `int`, so a flag named for a timeout
    (`delete_on_timeout=True`) would otherwise read as a bound.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_number(node.operand)
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    )


def _carries_a_bound(name: str) -> bool:
    """Whether a keyword or assignment target names a wall-clock bound."""
    lowered = name.lower()
    return lowered in _TIMEOUT_KEYWORDS or lowered.endswith(_BOUND_NAME_SUFFIXES)


def _called_name(node: ast.Call) -> str:
    """The bare function name of a call, `asyncio.timeout(...)` reading as `timeout`."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _call_offences(node: ast.Call) -> list[tuple[int, str]]:
    """Bounds passed to a call, as `(line, what)` pairs."""
    called = _called_name(node)
    position = _DEADLINE_POSITIONS.get(called)
    found: list[tuple[int, str]] = []
    if position is not None and len(node.args) > position and _is_number(node.args[position]):
        found.append((node.args[position].lineno, "positional deadline"))
    found.extend(
        (keyword.value.lineno, f"{keyword.arg}=")
        for keyword in node.keywords
        if keyword.arg is not None
        and _is_number(keyword.value)
        and (_carries_a_bound(keyword.arg) or (keyword.arg == "delay" and called in _DELAY_CALLS))
    )
    return found


def _parameter_offences(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[int, str]]:
    """Bounds written as a parameter's default value, as `(line, what)` pairs.

    A default is a bound like any other -- it decides what every caller that stays quiet gets.
    """
    arguments = node.args
    positional = arguments.posonlyargs + arguments.args
    paired = list(
        zip(
            positional[len(positional) - len(arguments.defaults) :],
            arguments.defaults,
            strict=True,
        )
    )
    paired += [
        (argument, default)
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
        if default is not None
    ]
    return [
        (default.lineno, f"{argument.arg}= default")
        for argument, default in paired
        if _carries_a_bound(argument.arg) and _is_number(default)
    ]


def _dict_offences(node: ast.Dict) -> list[tuple[int, str]]:
    """Bounds written as a `"...timeout..."` key in a params dict, as `(line, what)` pairs."""
    return [
        (value.lineno, f"{key.value!r} key")
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and "timeout" in key.value.lower()
        and _is_number(value)
    ]


def _assignment_offences(node: ast.Assign | ast.AnnAssign) -> list[tuple[int, str]]:
    """Bounds bound to a name the scope rule does not excuse, as `(line, what)` pairs."""
    literal = _literal_value(node=node.value)
    if literal is None:
        return []
    return [
        (literal.lineno, f"{target} =")
        for target in _assignment_targets(node)
        if _carries_a_bound(target) and target not in _OUT_OF_SCOPE
    ]


def _offences_in(module: Path) -> list[str]:
    """Every numeric literal in one module that spells a bound instead of naming one."""
    found: list[str] = []
    relative = module.relative_to(_PACKAGE.parent)
    for node in ast.walk(ast.parse(source=module.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            offences = _call_offences(node)
        elif isinstance(node, ast.Dict):
            offences = _dict_offences(node)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            offences = _assignment_offences(node)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            offences = _parameter_offences(node)
        else:
            continue
        found.extend(f"{relative}:{line} {what}" for line, what in offences)
    return found


def _literal_value(node: ast.expr | None) -> ast.expr | None:
    """The numeric literal an assigned value carries, or None.

    Reads through a pydantic `Field(default=...)`, because that is where five of the bounds
    this collected were written and a scan blind to it would let every one of them back.
    """
    if node is None:
        return None
    if _is_number(node):
        return node
    if isinstance(node, ast.Call):
        return next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "default" and _is_number(keyword.value)
            ),
            None,
        )
    return None


def _assignment_targets(node: ast.Assign | ast.AnnAssign) -> list[str]:
    """The names a bound-valued assignment binds, empty when it carries no numeric literal.

    Reads `x = 1.0`, `x: Final[float] = 1.0` and `x: float = Field(default=1.0)` alike: a
    bound written as an annotated constant, a plain local, or a model field default is the
    same thing to a reader looking for the bounds.
    """
    if _literal_value(node=node.value) is None:
        return []
    if isinstance(node, ast.AnnAssign):
        return [node.target.id] if isinstance(node.target, ast.Name) else []
    return [target.id for target in node.targets if isinstance(target, ast.Name)]


def test_a_wall_clock_bound_is_never_a_literal_outside_the_timeouts_module() -> None:
    """A bound has to be named in `typings/timeouts.py`, not spelled where it is used.

    The point is not tidiness. Two of the bounds this collected were coupled across files and
    could only describe each other in prose, and three download paths had no bound at all —
    both because a number written at its call site is invisible to anyone reading for the
    bounds rather than for that call.
    """
    offences = [offence for module in _modules() for offence in _offences_in(module)]
    assert offences == [], (
        "wall-clock bounds must be named in discordbot/typings/timeouts.py: " + ", ".join(offences)
    )


def test_the_out_of_scope_list_names_only_constants_that_still_exist() -> None:
    """An allowlist entry that names nothing is an exclusion nobody can see is stale.

    The list is the scope rule written as code, so an excluded constant that gets renamed or
    deleted has to be answered here rather than leaving a dead name that quietly excuses the
    next constant to take it.
    """
    declared = {
        target
        for module in _modules()
        for node in ast.walk(ast.parse(source=module.read_text(encoding="utf-8")))
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for target in _assignment_targets(node)
    }
    assert declared >= _OUT_OF_SCOPE, (
        f"out-of-scope names that no longer exist: {sorted(_OUT_OF_SCOPE - declared)}"
    )
