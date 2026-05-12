"""Tests for keeping the localized help guide aligned with slash commands."""

import ast
from pathlib import Path

from nextcord import Locale

from discordbot.cogs.help import _HELP_CONTENT


def _slash_command_names() -> set[str]:
    """Returns slash command names declared by top-level cogs."""
    names: set[str] = set()
    cogs_dir = Path(__file__).resolve().parents[1] / "src" / "discordbot" / "cogs"
    for path in cogs_dir.glob(pattern="*.py"):
        parsed = ast.parse(source=path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(node=parsed):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "slash_command":
                continue
            for keyword in node.keywords:
                if keyword.arg != "name" or not isinstance(keyword.value, ast.Constant):
                    continue
                if isinstance(keyword.value.value, str):
                    names.add(keyword.value.value)
    return names


def test_help_mentions_every_non_help_slash_command() -> None:
    """Every non-help slash command should be discoverable from every localized help body."""
    commands = _slash_command_names() - {"help"}
    for locale in ("default", Locale.zh_TW, Locale.ja):
        body = "\n".join(
            value for value in _HELP_CONTENT[locale].values() if isinstance(value, str)
        )
        missing = sorted(f"/{command}" for command in commands if f"/{command}" not in body)
        assert not missing, f"{locale} help is missing slash commands: {missing}"
