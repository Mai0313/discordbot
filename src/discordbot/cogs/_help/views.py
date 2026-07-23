"""Interactive view that swaps the help embed between categories."""

from typing import cast
import contextlib

import nextcord
from nextcord import Embed, Locale, Interaction, SelectOption
from nextcord.ui import View, StringSelect
from nextcord.ext import commands

from discordbot.cogs._help.content import CATEGORY_ORDER, OVERVIEW_VALUE, resolve_guide
from discordbot.cogs._help.presentation import build_section_embed, build_overview_embed

HELP_TIMEOUT_SECONDS = 180


class HelpView(View):
    """Ephemeral help view that swaps a single embed between categories.

    Attributes:
        guide: Localized help content driving the embeds and select options.
    """

    def __init__(
        self, locale: Locale | str, requester_name: str, requester_avatar_url: str
    ) -> None:
        """Initializes the view for one requester and locale."""
        super().__init__(timeout=HELP_TIMEOUT_SECONDS)
        self.guide = resolve_guide(locale=locale)
        self._requester_name = requester_name
        self._requester_avatar_url = requester_avatar_url
        self._active = OVERVIEW_VALUE
        self._origin: Interaction[commands.Bot] | None = None
        self._select = cast("StringSelect[HelpView]", self.category_select)
        self._select.placeholder = self.guide.select_placeholder
        self._sync_options()

    def initial_embed(self) -> Embed:
        """Returns the overview embed shown when the command is first run."""
        return build_overview_embed(
            guide=self.guide,
            requester_name=self._requester_name,
            requester_avatar_url=self._requester_avatar_url,
        )

    def bind_origin(self, interaction: Interaction[commands.Bot]) -> None:
        """Records the originating interaction so timeout can disable the menu."""
        self._origin = interaction

    def _sync_options(self) -> None:
        """Rebuilds select options and marks the active category as default."""
        options = [
            SelectOption(
                label=self.guide.overview_label,
                value=OVERVIEW_VALUE,
                emoji="🏠",
                default=self._active == OVERVIEW_VALUE,
            )
        ]
        for key in CATEGORY_ORDER:
            section = self.guide.sections[key]
            options.append(
                SelectOption(
                    label=section.label,
                    value=key,
                    description=section.summary,
                    emoji=section.emoji,
                    default=self._active == key,
                )
            )
        self._select.options = options

    def _embed_for(self, key: str) -> Embed:
        """Returns the overview or a category detail embed for the active key."""
        if key == OVERVIEW_VALUE:
            return self.initial_embed()
        return build_section_embed(
            guide=self.guide,
            key=key,
            requester_name=self._requester_name,
            requester_avatar_url=self._requester_avatar_url,
        )

    @nextcord.ui.string_select(
        custom_id="help_category",
        placeholder="…",
        min_values=1,
        max_values=1,
        options=[SelectOption(label="loading", value="loading")],
    )
    async def category_select(
        self, select: StringSelect["HelpView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Swaps the embed in place to the chosen category."""
        self._active = select.values[0]
        self._sync_options()
        await interaction.response.edit_message(embed=self._embed_for(key=self._active), view=self)

    async def on_timeout(self) -> None:
        """Disables the menu once the view goes idle."""
        if self._origin is None:
            return
        self._select.disabled = True
        with contextlib.suppress(Exception):
            await self._origin.edit_original_message(view=self)
