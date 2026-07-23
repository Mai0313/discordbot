"""Pagination view and embed builders for the /memory show command."""

from typing import cast
import contextlib

import nextcord
from nextcord import Embed, ButtonStyle, Interaction
from nextcord.ui import View, Button
from nextcord.ext import commands

MEMORY_VIEW_TIMEOUT_SECONDS = 180

MEMORY_EMBED_COLOR = 0x5865F2

# Embed descriptions cap at 4,096 chars; pages stay below that with headroom
# so the page indicator and footer never push the embed near Discord's
# 6,000-char total.
MEMORY_PAGE_MAX_CHARS = 4_000


def paginate_on_lines(text: str, limit: int) -> list[str]:
    """Splits text into pages at line boundaries, never tearing a line.

    A single line longer than the limit is hard-split as a fallback so every
    page honors the limit.

    Raises:
        ValueError: The limit is not positive (the hard-split fallback would
            otherwise never shrink an oversized line).
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    pages: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line
        while len(line) > limit:
            if current:
                pages.append(current)
                current = ""
            pages.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            pages.append(current)
            current = line
        else:
            current = candidate
    if current:
        pages.append(current)
    return pages or [""]


def memory_footer_text(pending_count: int) -> str:
    """Returns the footer line describing pending background observations."""
    if pending_count:
        return f"另有 {pending_count} 筆新觀察待整理，會在背景慢慢併入"
    return "記憶會在你與我對話後於背景慢慢更新"


def build_memory_embed(
    page_text: str, page_index: int, page_count: int, footer_text: str, title: str
) -> Embed:
    """Builds one /memory show embed page with the shared footer."""
    embed = Embed(title=title, description=page_text, color=MEMORY_EMBED_COLOR)
    footer = footer_text
    if page_count > 1:
        footer = f"第 {page_index + 1}/{page_count} 頁 | {footer}"
    embed.set_footer(text=footer)
    return embed


class MemoryPagesView(View):
    """Ephemeral pagination view for an oversized personal memory embed.

    Attributes:
        pages: Pre-split page texts, each within one embed description.
        footer_text: Footer line shared by every page.
        title: Embed title shared by every page.
        page_index: The currently displayed page.
    """

    def __init__(self, pages: list[str], footer_text: str, title: str) -> None:
        """Initializes the view on the first page."""
        super().__init__(timeout=MEMORY_VIEW_TIMEOUT_SECONDS)
        self.pages = pages
        self.footer_text = footer_text
        self.title = title
        self.page_index = 0
        self._origin: Interaction[commands.Bot] | None = None
        self._sync_buttons()

    def bind_origin(self, interaction: Interaction[commands.Bot]) -> None:
        """Records the originating interaction so timeout can disable the buttons."""
        self._origin = interaction

    def current_embed(self) -> Embed:
        """Returns the embed for the currently displayed page."""
        return build_memory_embed(
            page_text=self.pages[self.page_index],
            page_index=self.page_index,
            page_count=len(self.pages),
            footer_text=self.footer_text,
            title=self.title,
        )

    def _sync_buttons(self) -> None:
        """Disables the boundary buttons at the first and last page."""
        cast("Button[MemoryPagesView]", self.previous_page).disabled = self.page_index <= 0
        cast("Button[MemoryPagesView]", self.next_page).disabled = (
            self.page_index >= len(self.pages) - 1
        )

    @nextcord.ui.button(label="◀ 上一頁", style=ButtonStyle.secondary)
    async def previous_page(
        self, _button: Button["MemoryPagesView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Shows the previous page in place."""
        self.page_index = max(self.page_index - 1, 0)
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.current_embed(), view=self)

    @nextcord.ui.button(label="下一頁 ▶", style=ButtonStyle.secondary)
    async def next_page(
        self, _button: Button["MemoryPagesView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Shows the next page in place."""
        self.page_index = min(self.page_index + 1, len(self.pages) - 1)
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.current_embed(), view=self)

    async def on_timeout(self) -> None:
        """Disables the buttons once the view goes idle."""
        if self._origin is None:
            return
        for child in self.children:
            if isinstance(child, Button):
                child.disabled = True
        # Inert cleanup: the ephemeral response may already be dismissed or gone,
        # and there is nothing left to degrade. Broad on purpose: nextcord runs
        # `on_timeout` in a bare `create_task`, so a narrower filter would let an
        # aiohttp transport error escape into a task that cannot handle it.
        with contextlib.suppress(Exception):
            await self._origin.edit_original_message(view=self)
