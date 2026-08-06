"""The select menu that turns a multi-hit `/maplestory` lookup back into a single embed.

Every `/maplestory` subcommand posts the embed directly when its query matched exactly one
entry. This file owns the other branch: it shows the matches as a string select and, once the
user picks one, produces the same embed the single-hit path would have. It is its own file
because the pick is a second, later interaction — the command callback has already returned by
then, so the resolution cannot sit inside it.

Each option carries the entry's own canonical name as its value rather than an index, so nothing
about the original result list has to survive the gap between the command and the callback.
`_RESOLVERS` maps the `search_type` string the command passed in (`monster` / `item` /
`equipment` / `scroll` / `npc` / `quest` / `map`, which are `cog.py`'s own spellings and have to
stay in step with it) onto the resolver that re-queries `MapleStoryService` for that exact name
and hands the hit to the matching builder in `embeds.py`. Nothing here reads the JSON or
formats a field itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import nextcord
from nextcord import Embed, Interaction, SelectOption
from nextcord.ui import View, StringSelect

from discordbot.utils.discord_embeds import embed_spacer_payload

from .embeds import (
    TranslateFn,
    create_map_embed,
    create_npc_embed,
    create_quest_embed,
    create_scroll_embed,
    create_monster_embed,
    create_equipment_embed,
    create_item_source_embed,
)

if TYPE_CHECKING:
    from nextcord.ext import commands

    from .service import MapleStoryService


def _resolve_monster(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Looks the picked monster up by exact name and renders its embed.

    Args:
        service (MapleStoryService): Lookup engine over the local JSON.
        name (str): Exact name carried by the chosen option.
        tr (TranslateFn): Translator handed to the embed builder.

    Returns:
        The monster embed, or None when the name matches nothing.
    """
    monster = service.get_monster(name=name)
    return create_monster_embed(monster=monster, translate=tr) if monster else None


def _resolve_item(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Renders which monsters drop the picked item.

    An item has no entry of its own in the data — it exists only as a monster's drop — so the
    resolution is a reverse lookup over every monster's drop table, and an item nothing drops
    resolves to None.

    Args:
        service (MapleStoryService): Lookup engine over the local JSON.
        name (str): Exact item name carried by the chosen option.
        tr (TranslateFn): Translator handed to the embed builder.

    Returns:
        The drop-source embed, or None when no monster drops the item.
    """
    monsters = service.get_monsters_by_drop(item_name=name)
    return (
        create_item_source_embed(item_name=name, monsters=monsters, translate=tr)
        if monsters
        else None
    )


def _resolve_equipment(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Looks the picked equipment up by exact name and renders its embed.

    Args:
        service (MapleStoryService): Lookup engine over the local JSON.
        name (str): Exact name carried by the chosen option.
        tr (TranslateFn): Translator handed to the embed builder.

    Returns:
        The equipment embed, or None when the name matches nothing.
    """
    equip = service.get_equipment(name=name)
    return create_equipment_embed(equip=equip, translate=tr) if equip else None


def _resolve_scroll(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Looks the picked scroll up by exact name and renders its embed.

    Args:
        service (MapleStoryService): Lookup engine over the local JSON.
        name (str): Exact name carried by the chosen option.
        tr (TranslateFn): Translator handed to the embed builder.

    Returns:
        The scroll embed, or None when the name matches nothing.
    """
    match = service.get_scroll(name=name)
    return create_scroll_embed(scroll=match, translate=tr) if match else None


def _resolve_npc(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Looks the picked NPC up by exact name and renders its embed.

    Args:
        service (MapleStoryService): Lookup engine over the local JSON.
        name (str): Exact name carried by the chosen option.
        tr (TranslateFn): Translator handed to the embed builder.

    Returns:
        The NPC embed, or None when the name matches nothing.
    """
    match = service.get_npc(name=name)
    return create_npc_embed(npc=match, translate=tr) if match else None


def _resolve_quest(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Looks the picked quest up by exact name and renders its embed.

    Args:
        service (MapleStoryService): Lookup engine over the local JSON.
        name (str): Exact name carried by the chosen option.
        tr (TranslateFn): Translator handed to the embed builder.

    Returns:
        The quest embed, or None when the name matches nothing.
    """
    match = service.get_quest(name=name)
    return create_quest_embed(quest=match, translate=tr) if match else None


def _resolve_map(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Looks the picked map up by exact name and renders its embed.

    Args:
        service (MapleStoryService): Lookup engine over the local JSON.
        name (str): Exact name carried by the chosen option.
        tr (TranslateFn): Translator handed to the embed builder.

    Returns:
        The map embed, or None when the name matches nothing.
    """
    match = service.get_map(name=name)
    return create_map_embed(map_entry=match, translate=tr) if match else None


class _ResolverFn(Protocol):
    """The one shape every `_RESOLVERS` entry has, so the dict is typed rather than `Callable`."""

    def __call__(self, service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
        """Resolves one picked result name into its embed.

        Args:
            service (MapleStoryService): Lookup engine over the local JSON.
            name (str): Exact name carried by the chosen option.
            tr (TranslateFn): Translator handed to the embed builder.

        Returns:
            The embed for the picked result, or None when the name matches nothing.
        """
        ...


_RESOLVERS: dict[str, _ResolverFn] = {
    "monster": _resolve_monster,
    "item": _resolve_item,
    "equipment": _resolve_equipment,
    "scroll": _resolve_scroll,
    "npc": _resolve_npc,
    "quest": _resolve_quest,
    "map": _resolve_map,
}


class MapleDropSearchView(View):
    """One-shot select menu over the results of a `/maplestory` lookup that matched several.

    The pick replaces the whole message with the chosen entry's embed and drops the menu, so the
    view is used at most once. There is no owner check, so anyone who can see the message can be
    the one who spends it.
    """

    def __init__(
        self, service: MapleStoryService, search_type: str, query: str, timeout: float | None = 300
    ) -> None:
        """Initializes the search view.

        The options are not passed here: the caller fills them through `set_options` before the
        message is sent, so the menu ships with the decorator's placeholder until then.

        Args:
            service (MapleStoryService): Lookup engine the resolver re-queries on the pick.
            search_type (str): Which `_RESOLVERS` entry the pick goes through.
            query (str): The text the user searched for. Stored only; nothing reads it back.
            timeout (float | None): Seconds of inactivity before nextcord stops the view.
        """
        super().__init__(timeout=timeout)
        self.service = service
        self.search_type = search_type
        self.query = query

    @nextcord.ui.select(
        placeholder="選擇要查看的結果...",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="載入中...", value="loading")],
    )
    async def select_result(
        self, select: StringSelect[MapleDropSearchView], interaction: Interaction[commands.Bot]
    ) -> None:
        """Replaces the result list with the embed of the entry the user picked.

        The edit passes `view=None`, so the menu goes away with the first pick rather than
        letting a second one edit an already-answered message. A `search_type` no resolver
        covers, or a name the service no longer knows, leaves the message exactly as it was
        instead of raising into nextcord's error path.

        Args:
            select (StringSelect[MapleDropSearchView]): Menu holding the chosen option's value.
            interaction (Interaction[commands.Bot]): The interaction the pick raised.
        """
        await interaction.response.defer()

        selected = select.values[0]
        # The decorator has to declare a non-empty option list at class-definition time, so this
        # sentinel is only reachable on a menu whose `set_options` never ran.
        if selected == "loading":
            await interaction.followup.send("請先選擇有效的結果", ephemeral=True)
            return

        resolver = _RESOLVERS.get(self.search_type)
        embed = (
            resolver(service=self.service, name=selected, tr=self.service.translate)
            if resolver
            else None
        )

        message = interaction.message
        if embed and message is not None:
            await interaction.followup.edit_message(
                message_id=message.id,
                embed=embed,
                view=None,
                **embed_spacer_payload(embeds=[embed], is_edit=True, target=interaction),
            )

    def set_options(self, options: list[SelectOption]) -> None:
        """Replaces the menu's placeholder option with the search results.

        Discord rejects a string select carrying more than 25 options, so anything past the 25th
        result is dropped here and is simply unreachable; the caller is expected to have narrowed
        the query rather than to be told about it.

        Args:
            options (list[SelectOption]): One option per result, each valued with the exact name
                its resolver will look up.
        """
        for child in self.children:
            if isinstance(child, StringSelect):
                child.options = options[:25]
