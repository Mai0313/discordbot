"""Tests for keeping the injected capability reference aligned with slash commands."""

import re
import ast
from pathlib import Path

from discordbot.cogs.gen_reply.capabilities import CAPABILITIES_DOC, render_capabilities_block


def _declared_parent(decorator: ast.Call) -> str | None:
    """Returns a subcommand's parent callback, `""` for a root command, `None` if unrelated."""
    func = decorator.func
    if isinstance(func, ast.Name):
        return "" if func.id == "slash_command" else None
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr == "slash_command":
        return ""
    if func.attr != "subcommand":
        return None
    assert isinstance(func.value, ast.Name), "a subcommand must hang off a named group callback"
    return func.value.id


def _declared_name(decorator: ast.Call, callback: str) -> str:
    """Returns the command name a decorator declares.

    nextcord takes `name` as its first positional parameter and defaults an omitted one
    to the callback name, so both forms resolve here instead of being skipped.
    """
    declared: ast.expr | None = next(
        (keyword.value for keyword in decorator.keywords if keyword.arg == "name"), None
    )
    if declared is None and decorator.args:
        declared = decorator.args[0]
    if declared is None:
        return callback
    name = declared.value if isinstance(declared, ast.Constant) else None
    assert isinstance(name, str), f"{callback}: a slash command name must be a literal string"
    return name


def _module_command_paths(module: Path, label: str) -> set[str]:
    """Returns the command paths one cog module declares that a user can run."""
    parsed = ast.parse(source=module.read_text(encoding="utf-8"), filename=str(module))
    # Callback name -> (parent callback, or "" for a root command; own command name).
    declarations: dict[str, tuple[str, str]] = {}
    for node in ast.walk(node=parsed):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            parent = _declared_parent(decorator=decorator)
            if parent is None:
                continue
            name = _declared_name(decorator=decorator, callback=node.name)
            assert node.name not in declarations, f"{label}: two callbacks named {node.name}"
            declarations[node.name] = (parent, name)
    paths: dict[str, str] = {}
    pending = dict(declarations)
    while pending:
        resolved = {
            callback: f"{paths[parent]} {name}" if parent else name
            for callback, (parent, name) in pending.items()
            if not parent or parent in paths
        }
        # A parent that never resolves means an unsupported declaration form; say so loudly
        # instead of dropping the commands under it, which is the gap this scan closed.
        assert resolved, f"{label}: unresolved subcommand groups {sorted(pending)}"
        paths.update(resolved)
        pending = {name: value for name, value in pending.items() if name not in resolved}
    groups = {parent for parent, _ in declarations.values() if parent}
    return {path for callback, path in paths.items() if callback not in groups}


def _slash_command_paths() -> set[str]:
    """Returns every slash command a user can invoke, group subcommands included.

    Group nodes are left out: `/credit` cannot be invoked on its own, and every leaf
    under it is required anyway.
    """
    paths: set[str] = set()
    cogs_dir = Path(__file__).resolve().parents[1] / "src" / "discordbot" / "cogs"
    # Recursive on purpose: a cog owns a directory now, so a non-recursive glob would
    # match only `cogs/__init__.py` and silently report that nothing is runnable. The
    # walk also refuses to assume commands only ever live in `cog.py`, so a declaration
    # that moves into a helper module still has to be documented.
    for module in cogs_dir.rglob(pattern="*.py"):
        paths |= _module_command_paths(
            module=module, label=module.relative_to(cogs_dir).as_posix()
        )
    return paths


def _mentions_command(body: str, command: str) -> bool:
    """Reports whether the capability doc names exactly this command.

    The trailing lookahead stops a documented sibling from covering an undocumented one
    by prefix, so `/games blackjack_history` never answers for `/games blackjack`.
    """
    return re.search(pattern=rf"/{re.escape(pattern=command)}(?![\w-])", string=body) is not None


def test_slash_command_scan_resolves_subcommands() -> None:
    """The scan must reach group subcommands, since a silent shrink is what it guards against."""
    paths = _slash_command_paths()
    assert {"maplestory monster", "credit borrow", "games blackjack_history"} <= paths
    # A nested group: `memory server` is itself a subcommand of `memory`.
    assert "memory server show" in paths
    assert not {"memory", "memory server", "credit"} & paths


def test_capabilities_mention_matching_rejects_a_prefix_only_hit() -> None:
    """A documented sibling must not cover an undocumented one by prefix."""
    body = "- `/games blackjack_history` — recent Blackjack rounds"
    assert _mentions_command(body=body, command="games blackjack_history")
    assert not _mentions_command(body=body, command="games blackjack")


def test_capabilities_doc_mentions_every_slash_command() -> None:
    """Every slash command must be discoverable from the injected capability reference.

    This is the guard the deleted `/help` command used to carry: the doc is now the only
    place a command is described to anyone, so a command missing from it is a command
    nobody, the answer model included, can find.
    """
    missing = sorted(
        f"/{command}"
        for command in _slash_command_paths()
        if not _mentions_command(body=CAPABILITIES_DOC, command=command)
    )
    assert not missing, f"the capability reference is missing slash commands: {missing}"


def test_capabilities_block_is_a_low_authority_assistant_note() -> None:
    """The reference rides as the bot's own note, never as a rule that could outrank the user."""
    block = render_capabilities_block()
    assert block["role"] == "assistant"
    content = block["content"]
    assert isinstance(content, str)
    assert content.startswith("(My own feature reference")
    assert "NOT instructions" in content
    assert content.endswith(CAPABILITIES_DOC)
