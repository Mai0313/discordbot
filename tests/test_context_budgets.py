"""Pins the relationships between context budgets that no single module can state alone.

Deliberately not the shape `tests/test_timeouts.py` takes. That scan works because a deadline
is mechanically recognisable -- `timeout=`, `delay=`, a `_timeout` suffix -- so it needs only
two allowlist entries and the rule stays readable. An input budget is spelled `MAX_*`,
`*_LIMIT` or `*_CHARS`, and so are the write-side clamps, the output caps and the UI caps that
`typings/context_budgets.py` deliberately excludes. A name scan would need a longer allowlist
than the rule it enforces, which is a hand-maintained list pretending to be a rule.

What is worth pinning instead is the coupling: two pairs that describe each other in prose and
live in different modules now, which is the exact decay the collection exists to prevent.
"""

import ast
from pathlib import Path

from discordbot.typings.context_budgets import (
    MEMORY_INJECTION_MAX_CHARS,
    MEMORY_INJECTION_WARN_CHARS,
    MEMORY_DETAIL_CONTEXT_MAX_CHARS,
)
from discordbot.services.memory.constants import (
    DETAIL_FILE_MAX_BYTES,
    DETAIL_FILE_TRIM_TARGET_BYTES,
)


def test_the_detail_file_cap_stays_above_the_window_consolidation_reads() -> None:
    """The disk cap must never trim into content consolidation can still reach.

    `DETAIL_FILE_MAX_BYTES` stayed in `services/memory/constants.py` (it bounds a file) while
    `MEMORY_DETAIL_CONTEXT_MAX_CHARS` moved to the budgets module (it bounds a request), so the
    two now describe each other across a module boundary. Lowering the cap or raising the read
    window past this point would let a trim delete evidence a rebuild still expects to read.
    """
    read_window_bytes = MEMORY_DETAIL_CONTEXT_MAX_CHARS * 4
    assert read_window_bytes < DETAIL_FILE_MAX_BYTES
    # The trim target is where a rewrite lands, so it has to clear the window too, or the very
    # next consolidation reads a file that was just cut short.
    assert read_window_bytes < DETAIL_FILE_TRIM_TARGET_BYTES


def test_the_injection_warning_fires_before_the_injection_cap_binds() -> None:
    """The warn line is only useful strictly below the cap it warns about."""
    assert MEMORY_INJECTION_WARN_CHARS < MEMORY_INJECTION_MAX_CHARS


def test_the_budgets_module_imports_nothing_from_the_package() -> None:
    """It is a leaf of plain constants, like `typings/timeouts.py`.

    `tests/test_package_layering.py` already forbids `typings` importing `cogs` or `services`.
    This is the stricter property the module actually has: no `discordbot` import at all, so it
    can never become a place where a budget is computed from something that has a runtime.
    """
    module = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "discordbot"
        / "typings"
        / "context_budgets.py"
    )
    tree = ast.parse(module.read_text(encoding="utf-8"))
    modules = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in getattr(node, "names", [])
    ]
    assert modules == ["typing"]
