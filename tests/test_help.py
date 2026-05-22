"""Tests for keeping the localized help guide aligned with slash commands."""

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from pathlib import Path

from nextcord import Locale, Interaction

from discordbot.cogs.help import (
    _SECTIONS,
    _HELP_CONTENT,
    _EMBED_FIELD_COUNT_LIMIT,
    _EMBED_FIELD_VALUE_LIMIT,
    _EMBED_TOTAL_LENGTH_LIMIT,
    _MESSAGE_EMBED_COUNT_LIMIT,
    HelpCogs,
    _build_help_embeds,
    _split_field_value,
)

if TYPE_CHECKING:
    from nextcord.ext import commands


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


def test_long_help_sections_are_split_without_hiding_content() -> None:
    """Long help sections are split into Discord-safe fields without truncation."""
    for locale in ("default", Locale.zh_TW, Locale.ja):
        for section in _SECTIONS:
            value = _HELP_CONTENT[locale][section]
            chunks = _split_field_value(value=value)

            assert "".join(chunks) == value
            assert all(len(chunk) <= _EMBED_FIELD_VALUE_LIMIT for chunk in chunks)


def test_help_embeds_fit_discord_limits() -> None:
    """Generated help embeds stay within Discord's embed validation limits."""
    for locale in ("default", Locale.zh_TW, Locale.ja):
        embeds = _build_help_embeds(
            locale=locale,
            requester_name="tester",
            requester_avatar_url="https://example.com/avatar.png",
        )

        assert len(embeds) <= _MESSAGE_EMBED_COUNT_LIMIT
        for embed in embeds:
            assert len(embed.fields) <= _EMBED_FIELD_COUNT_LIMIT
            assert len(embed) <= _EMBED_TOTAL_LENGTH_LIMIT
            assert all(len(field.value) <= _EMBED_FIELD_VALUE_LIMIT for field in embed.fields)


async def test_help_followups_stay_ephemeral() -> None:
    """The help command keeps every response private."""

    class ResponseStub:
        """Records the initial deferred response."""

        def __init__(self) -> None:
            self.deferred_ephemeral = False

        async def defer(self, ephemeral: bool = False) -> None:
            self.deferred_ephemeral = ephemeral

    class FollowupStub:
        """Records followup payloads."""

        def __init__(self) -> None:
            self.sent: list[dict[str, object]] = []

        async def send(self, **kwargs: object) -> None:
            self.sent.append(kwargs)

    interaction = SimpleNamespace(
        locale=Locale.zh_TW,
        user=SimpleNamespace(
            display_name="tester",
            display_avatar=SimpleNamespace(url="https://example.com/avatar.png"),
        ),
        response=ResponseStub(),
        followup=FollowupStub(),
    )

    await HelpCogs.help.callback(
        HelpCogs(bot=cast("commands.Bot", SimpleNamespace())), cast("Interaction", interaction)
    )

    assert interaction.response.deferred_ephemeral is True
    assert interaction.followup.sent
    assert all(payload["ephemeral"] is True for payload in interaction.followup.sent)
