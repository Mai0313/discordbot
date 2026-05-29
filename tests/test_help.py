"""Tests for keeping the localized help guide aligned with slash commands."""

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from pathlib import Path

from nextcord import Embed, Locale, Interaction
from nextcord.ui import View

from discordbot.cogs.help import HelpCogs
from discordbot.cogs._help.views import HelpView
from discordbot.cogs._help.content import HELP_CONTENT, CATEGORY_ORDER, OVERVIEW_VALUE

if TYPE_CHECKING:
    from nextcord.ext import commands

_LOCALES = ("default", Locale.zh_TW, Locale.ja)

# Discord embed validation limits the help view must respect.
_EMBED_TITLE_LIMIT = 256
_EMBED_DESCRIPTION_LIMIT = 4096
_EMBED_TOTAL_LENGTH_LIMIT = 6000
# Select option label / description hard limits.
_SELECT_LABEL_LIMIT = 100
_SELECT_DESCRIPTION_LIMIT = 100


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


def _guide_text(locale: "Locale | str") -> str:
    """Joins every user-visible string of a locale's help guide."""
    guide = HELP_CONTENT[locale]
    parts = [guide.intro]
    for section in guide.sections.values():
        parts.extend([section.summary, section.detail])
    return "\n".join(parts)


def test_help_mentions_every_non_help_slash_command() -> None:
    """Every non-help slash command should be discoverable from every localized help body."""
    commands = _slash_command_names() - {"help"}
    for locale in _LOCALES:
        body = _guide_text(locale=locale)
        missing = sorted(f"/{command}" for command in commands if f"/{command}" not in body)
        assert not missing, f"{locale} help is missing slash commands: {missing}"


def test_help_sections_cover_the_category_order() -> None:
    """Every locale defines exactly the categories declared in `CATEGORY_ORDER`."""
    for locale in _LOCALES:
        assert set(HELP_CONTENT[locale].sections) == set(CATEGORY_ORDER)


async def test_help_embeds_fit_discord_limits() -> None:
    """Overview and category detail embeds stay within Discord's embed limits."""
    for locale in _LOCALES:
        view = HelpView(
            locale=locale,
            requester_name="tester",
            requester_avatar_url="https://example.com/avatar.png",
        )
        embeds = [view.initial_embed()] + [view._embed_for(key=key) for key in CATEGORY_ORDER]
        for embed in embeds:
            assert len(embed.title or "") <= _EMBED_TITLE_LIMIT
            assert len(embed.description or "") <= _EMBED_DESCRIPTION_LIMIT
            assert len(embed) <= _EMBED_TOTAL_LENGTH_LIMIT


async def test_help_select_options_fit_discord_limits() -> None:
    """Select options (overview + every category) stay within Discord's limits."""
    for locale in _LOCALES:
        view = HelpView(
            locale=locale,
            requester_name="tester",
            requester_avatar_url="https://example.com/avatar.png",
        )
        values = {option.value for option in view._select.options}
        assert values == {OVERVIEW_VALUE, *CATEGORY_ORDER}
        for option in view._select.options:
            assert len(option.label) <= _SELECT_LABEL_LIMIT
            assert len(option.description or "") <= _SELECT_DESCRIPTION_LIMIT


async def test_help_select_marks_active_category() -> None:
    """Selecting a category rebuilds options with that category as the default."""
    view = HelpView(
        locale=Locale.zh_TW,
        requester_name="tester",
        requester_avatar_url="https://example.com/avatar.png",
    )
    view._active = "economy"
    view._sync_options()
    defaults = [option.value for option in view._select.options if option.default]
    assert defaults == ["economy"]


async def test_help_response_is_ephemeral_with_a_view() -> None:
    """The help command replies privately and ships an interactive view."""

    class ResponseStub:
        """Records the initial response payload."""

        def __init__(self) -> None:
            self.sent: dict[str, object] = {}

        async def send_message(self, **kwargs: object) -> None:
            self.sent = kwargs

    interaction = SimpleNamespace(
        locale=Locale.zh_TW,
        user=SimpleNamespace(
            display_name="tester",
            display_avatar=SimpleNamespace(url="https://example.com/avatar.png"),
        ),
        response=ResponseStub(),
    )

    await HelpCogs.help.callback(
        HelpCogs(bot=cast("commands.Bot", SimpleNamespace())), cast("Interaction", interaction)
    )

    assert interaction.response.sent["ephemeral"] is True
    assert isinstance(interaction.response.sent["view"], View)
    assert isinstance(interaction.response.sent["embed"], Embed)
