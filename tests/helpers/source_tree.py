"""Where this repository's own files are, and how to read them, for tests that read the source."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "src" / "discordbot"


def python_modules(root: Path) -> list[Path]:
    """Every `.py` file under `root`, in path order, bytecode caches skipped.

    A directory that is not there yields an empty list rather than raising, so a scan that must
    find something asserts on what it found.
    """
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def called_name(node: ast.expr) -> str:
    """The bare name of a call target, `asyncio.timeout` reading as `timeout`; empty otherwise."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""
