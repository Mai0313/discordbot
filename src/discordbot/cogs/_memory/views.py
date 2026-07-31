"""Views and embed builders for the /memory show and /memory clear commands."""

from typing import cast
import contextlib

import logfire
import nextcord
from nextcord import Embed, ButtonStyle, Interaction
from nextcord.ui import View, Button
from nextcord.ext import commands

from discordbot.typings.colors import DISCORD_RED, NEUTRAL_BLUE, DISCORD_GREEN, DISCORD_YELLOW
from discordbot.cogs._memory.store import DM_COMPARTMENT, GLOBAL_COMPARTMENT
from discordbot.cogs._memory.pipeline import clear_scope_memory

MEMORY_VIEW_TIMEOUT_SECONDS = 180

MEMORY_EMBED_COLOR = NEUTRAL_BLUE

MEMORY_CLEAR_TITLE = "🧠 清除記憶"


def compartment_label(compartment: str, bot: commands.Bot) -> str:
    """Names a memory compartment for the owner's own `/memory show`.

    The guild name is resolved when the bot still shares that server, and falls back to
    the id when it does not — a user can have memory from a server the bot has since
    left, and hiding that would misrepresent what is stored.
    """
    if compartment == GLOBAL_COMPARTMENT:
        return "全部聊天都看得到"
    if compartment == DM_COMPARTMENT:
        return "只有我們的私訊看得到"
    guild_id = compartment.removeprefix("g/")
    guild = bot.get_guild(int(guild_id)) if guild_id.isdigit() else None
    return f"只有 {guild.name} 看得到" if guild is not None else f"只有伺服器 {guild_id} 看得到"


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


def build_clear_confirm_embed() -> Embed:
    """Builds the warning shown above the clear confirmation buttons.

    Names every tier that goes, since the user only ever saw the consolidated
    half through `/memory show` and would not otherwise know the observation
    log and the tone note are part of the wipe.
    """
    return Embed(
        title=MEMORY_CLEAR_TITLE,
        description=(
            "這會刪掉我對你的所有長期記憶：整理好的記憶、還沒整理的觀察、觀察記錄，"
            "還有語氣偏好，而且沒辦法復原。\n"
            "伺服器記憶不受影響。清掉之後我會從下一次聊天重新開始認識你。\n"
            "想先看看我記得什麼，可以先取消，用 `/memory show` 看過再回來。"
        ),
        color=DISCORD_YELLOW,
    )


def build_clear_result_embed(removed: bool) -> Embed:
    """Builds the outcome embed for a completed clear."""
    if removed:
        return Embed(
            title=MEMORY_CLEAR_TITLE,
            description="已經把我對你的記憶都清掉了，從下一次聊天開始重新認識你。",
            color=DISCORD_GREEN,
        )
    return Embed(
        title=MEMORY_CLEAR_TITLE,
        description="我目前沒有留下任何對你的記憶，所以沒有東西需要清除。",
        color=DISCORD_YELLOW,
    )


def build_clear_failed_embed() -> Embed:
    """Builds the outcome embed for a clear that could not complete.

    Deliberately does not promise the memory is untouched: only the reply.db half
    is guaranteed to have changed nothing, while a filesystem error can land after
    some tiers are already gone. Pointing at a retry is the honest advice, since
    the clear is idempotent and a second run finishes whatever the first left.
    """
    return Embed(
        title=MEMORY_CLEAR_TITLE,
        description="清除沒有完成，可能還有一部分沒清掉。等一下再試一次，重複清除不會有問題。",
        color=DISCORD_RED,
    )


def build_clear_cancelled_embed() -> Embed:
    """Builds the outcome embed for a cancelled clear."""
    return Embed(
        title=MEMORY_CLEAR_TITLE,
        description="已取消，沒有清掉任何東西。",
        color=MEMORY_EMBED_COLOR,
    )


class MemoryClearConfirmView(View):
    """Confirmation buttons guarding an irreversible personal-memory clear.

    The prompt is ephemeral, so only its invoker can see or press it and no
    author check is needed on top (same as `MemoryPagesView`).

    Attributes:
        scope: The memory scope erased once the clear is confirmed.
    """

    def __init__(self, scope: str) -> None:
        """Initializes the confirmation prompt for one scope's pending clear."""
        super().__init__(timeout=MEMORY_VIEW_TIMEOUT_SECONDS)
        self.scope = scope
        self._origin: Interaction[commands.Bot] | None = None

    def bind_origin(self, interaction: Interaction[commands.Bot]) -> None:
        """Records the originating interaction so timeout can disable the buttons."""
        self._origin = interaction

    @nextcord.ui.button(label="確認清除", style=ButtonStyle.danger)
    async def confirm_clear(
        self, _button: Button["MemoryClearConfirmView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Erases the scope's memory and replaces the prompt with the outcome."""
        if self.is_finished():
            # A second click lands while the first press is still on its way to
            # removing the buttons. Re-running the clear is harmless (it is
            # idempotent) but its "nothing to clear" result would overwrite the
            # real outcome, so this press is acked and dropped.
            await interaction.response.defer()
            return
        self.stop()
        # Acked first: the clear writes reply.db and the filesystem, and a reply
        # that misses Discord's 3s window would report a failure for a wipe that
        # already happened.
        await interaction.response.defer()
        try:
            removed = await clear_scope_memory(scope=self.scope)
        except Exception as exc:
            # Broad on purpose: this is the button-callback boundary, so anything
            # escaping here surfaces as Discord's bare "This interaction failed",
            # which never tells the user what happened to their memory. A failed
            # delete of the reply.db row leaves every file in place; a filesystem
            # error can leave the scope half cleared, so the embed points at a
            # retry rather than claiming either outcome.
            logfire.error(
                "Personal memory clear failed",
                scope=self.scope,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await interaction.edit_original_message(embed=build_clear_failed_embed(), view=None)
            return
        await interaction.edit_original_message(
            embed=build_clear_result_embed(removed=removed), view=None
        )

    @nextcord.ui.button(label="取消", style=ButtonStyle.secondary)
    async def cancel_clear(
        self, _button: Button["MemoryClearConfirmView"], interaction: Interaction[commands.Bot]
    ) -> None:
        """Dismisses the prompt without touching any memory."""
        self.stop()
        await interaction.response.edit_message(embed=build_clear_cancelled_embed(), view=None)

    async def on_timeout(self) -> None:
        """Disables the buttons once the prompt goes idle; nothing is cleared."""
        if self._origin is None:
            return
        for child in self.children:
            if isinstance(child, Button):
                child.disabled = True
        # Inert cleanup, broad for the same reason as `MemoryPagesView.on_timeout`:
        # nextcord runs this in a bare `create_task`, and the ephemeral prompt may
        # already be dismissed.
        with contextlib.suppress(Exception):
            await self._origin.edit_original_message(view=self)


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
