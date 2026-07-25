"""Tests for keeping the localized help guide aligned with slash commands."""

import re
import ast
from types import SimpleNamespace
from pathlib import Path

from nextcord import Embed, Locale
from nextcord.ui import View

from discordbot.cogs.help import HelpCogs
from discordbot.cogs._help.views import HelpView
from discordbot.cogs._help.content import HELP_CONTENT, CATEGORY_ORDER, OVERVIEW_VALUE

from tests.helpers.casting import as_bot, as_interaction

_LOCALES = ("default", Locale.zh_TW, Locale.ja)

# Discord embed validation limits the help view must respect.
_EMBED_TITLE_LIMIT = 256
_EMBED_DESCRIPTION_LIMIT = 4096
_EMBED_TOTAL_LENGTH_LIMIT = 6000
# Select option label / description hard limits.
_SELECT_LABEL_LIMIT = 100
_SELECT_DESCRIPTION_LIMIT = 100


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


def _module_command_paths(module: Path) -> set[str]:
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
        assert resolved, f"{module.name}: unresolved subcommand groups {sorted(pending)}"
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
    for module in cogs_dir.glob(pattern="*.py"):
        paths |= _module_command_paths(module=module)
    return paths


def _mentions_command(body: str, command: str) -> bool:
    """Reports whether a help body names exactly this command.

    The trailing lookahead stops a documented sibling from covering an undocumented one
    by prefix, so `/games blackjack_history` never answers for `/games blackjack`.
    """
    return re.search(pattern=rf"/{re.escape(pattern=command)}(?![\w-])", string=body) is not None


def _guide_text(locale: "Locale | str") -> str:
    """Joins every user-visible string of a locale's help guide."""
    guide = HELP_CONTENT[locale]
    parts = [guide.intro]
    for section in guide.sections.values():
        parts.extend([section.summary, section.detail])
    return "\n".join(parts)


def test_slash_command_scan_resolves_subcommands() -> None:
    """The scan must reach group subcommands, since a silent shrink is what it guards against."""
    paths = _slash_command_paths()
    assert {"maplestory monster", "credit borrow", "games blackjack_history"} <= paths
    # A nested group: `memory server` is itself a subcommand of `memory`.
    assert "memory server show" in paths
    assert not {"memory", "memory server", "credit"} & paths


def test_help_mentions_every_non_help_slash_command() -> None:
    """Every non-help slash command should be discoverable from every localized help body."""
    commands = _slash_command_paths() - {"help"}
    for locale in _LOCALES:
        body = _guide_text(locale=locale)
        missing = sorted(
            f"/{command}"
            for command in commands
            if not _mentions_command(body=body, command=command)
        )
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


async def test_help_select_carries_a_custom_id() -> None:
    """The category select must serialize a non-empty custom_id or Discord rejects the send (50035)."""
    view = HelpView(
        locale=Locale.zh_TW,
        requester_name="tester",
        requester_avatar_url="https://example.com/avatar.png",
    )
    assert isinstance(view._select.custom_id, str)
    assert view._select.custom_id
    component = view.to_components()[0]["components"][0]
    assert component.get("custom_id"), "select component is missing custom_id"


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
        HelpCogs(bot=as_bot(fake=SimpleNamespace())), as_interaction(fake=interaction)
    )

    assert interaction.response.sent["ephemeral"] is True
    assert isinstance(interaction.response.sent["view"], View)
    assert isinstance(interaction.response.sent["embed"], Embed)
