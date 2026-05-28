"""Views for MapleStory cog.

This module provides Discord UI views and helper resolvers for interactive
search results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import nextcord
from nextcord import Embed, Interaction, SelectOption
from nextcord.ui import View, Select

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
    from .service import MapleStoryService


def _resolve_monster(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Resolves a monster name to an Embed."""
    monster = service.get_monster(name)
    return create_monster_embed(monster=monster, translate=tr) if monster else None


def _resolve_item(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Resolves an item name to an Embed showing its sources."""
    monsters = service.get_monsters_by_drop(name)
    return (
        create_item_source_embed(item_name=name, monsters=monsters, translate=tr)
        if monsters
        else None
    )


def _resolve_equipment(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Resolves an equipment name to an Embed."""
    equip = service.get_equipment(name)
    return create_equipment_embed(equip=equip, translate=tr) if equip else None


def _resolve_scroll(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Resolves a scroll name to an Embed."""
    match = next((s for s in service.search_scrolls_by_name(name) if s.name == name), None)
    return create_scroll_embed(scroll=match, translate=tr) if match else None


def _resolve_npc(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Resolves an NPC name to an Embed."""
    match = next((n for n in service.search_npcs_by_name(name) if n.name == name), None)
    return create_npc_embed(npc=match, translate=tr) if match else None


def _resolve_quest(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Resolves a quest name to an Embed."""
    match = next((q for q in service.search_quests_by_name(name) if q.name == name), None)
    return create_quest_embed(quest=match, translate=tr) if match else None


def _resolve_map(service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
    """Resolves a map name to an Embed."""
    match = next((m for m in service.search_maps_by_name(name) if m.name == name), None)
    return create_map_embed(map_entry=match, translate=tr) if match else None


class _ResolverFn(Protocol):
    """Protocol for resolver functions."""

    def __call__(self, service: MapleStoryService, name: str, tr: TranslateFn) -> Embed | None:
        """Resolves a search result name into a Discord embed.

        Args:
            service: Service used to look up MapleStory data.
            name: Selected result name to resolve.
            tr: Translation function passed to embed builders.

        Returns:
            The embed for the selected result, or None when no match is found.
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
    """Interactive select menu for Artale search results."""

    def __init__(
        self, service: MapleStoryService, search_type: str, query: str, timeout: float | None = 300
    ) -> None:
        """Initializes the search view.

        Args:
            service: The MapleStoryService instance.
            search_type: The type of search (e.g., 'monster', 'item').
            query: The original search query.
            timeout: Timeout in seconds.
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
    async def select_result(self, select: Select, interaction: Interaction) -> None:
        """Handles the selection of a search result.

        Args:
            select: Select menu that contains the chosen result value.
            interaction: Discord interaction that triggered the callback.
        """
        await interaction.response.defer()

        selected = select.values[0]
        if selected == "loading":
            await interaction.followup.send("請先選擇有效的結果", ephemeral=True)
            return

        resolver = _RESOLVERS.get(self.search_type)
        embed = (
            resolver(service=self.service, name=selected, tr=self.service.translate)
            if resolver
            else None
        )

        if embed:
            await interaction.followup.edit_message(
                message_id=interaction.message.id,
                embed=embed,
                view=None,
                **embed_spacer_payload(embeds=[embed], is_edit=True),
            )

    def set_options(self, options: list[SelectOption]) -> None:
        """Sets the options for the select menu, capped at 25.

        Args:
            options: A list of SelectOption objects.
        """
        self.select_result.options = options[:25]
