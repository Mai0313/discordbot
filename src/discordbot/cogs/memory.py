"""Slash commands for viewing and clearing per-user long-term memory."""

import nextcord
from nextcord import Embed, Locale, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.typings.config import MemoryConfig
from discordbot.cogs._memory.store import (
    read_detail_tail,
    read_main_memory,
    clear_user_memory,
    count_raw_entries,
)
from discordbot.cogs._memory.views import (
    MEMORY_EMBED_COLOR,
    MEMORY_PAGE_MAX_CHARS,
    MemoryPagesView,
    paginate_on_lines,
    build_memory_embed,
    memory_footer_text,
)
from discordbot.cogs._memory.constants import MEMORY_DETAIL_VIEW_MAX_CHARS

_CLEAR_EMBED_COLOR = 0x57F287
_CLEAR_DISABLED_EMBED_COLOR = 0xFEE75C

_MEMORY_TITLE = "🧠 我對你的記憶"
_DETAIL_TITLE = "🧠 詳細記憶記錄"


class MemoryCogs(commands.Cog):
    """Provides the personal long-term memory management commands.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        memory_config: Env-backed memory settings, including the clear kill switch.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the memory cog.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.memory_config = MemoryConfig()

    @nextcord.slash_command(
        name="memory",
        description="Manage what the bot remembers about you.",
        name_localizations={Locale.zh_TW: "記憶", Locale.ja: "メモリー"},
        description_localizations={
            Locale.zh_TW: "管理 bot 對你的長期記憶",
            Locale.ja: "ボットがあなたについて記憶している内容を管理します。",
        },
        nsfw=False,
    )
    async def memory(self, interaction: Interaction) -> None:
        """Slash command group for per-user memory management."""

    @memory.subcommand(
        name="show",
        description="Show what the bot remembers about you.",
        name_localizations={Locale.zh_TW: "查看", Locale.ja: "表示"},
        description_localizations={
            Locale.zh_TW: "查看 bot 對你的長期記憶",
            Locale.ja: "ボットがあなたについて記憶している内容を表示します。",
        },
    )
    async def memory_show(
        self,
        interaction: Interaction,
        detail: bool = SlashOption(
            name="detail",
            description="Show the recent fine-grained detail log instead of the consolidated memory.",
            name_localizations={Locale.zh_TW: "詳細", Locale.ja: "詳細"},
            description_localizations={
                Locale.zh_TW: "改為顯示最近的詳細觀察記錄，而非整理後的記憶",
                Locale.ja: "統合メモリーの代わりに最近の詳細ログを表示します。",
            },
            required=False,
            default=False,
        ),
    ) -> None:
        """Shows the caller's consolidated memory or its detail log, paginated."""
        if interaction.user is None:
            return
        if detail:
            await self._show_detail(interaction=interaction)
            return
        memory_text = read_main_memory(user_id=interaction.user.id)
        pending_count = count_raw_entries(user_id=interaction.user.id)
        if memory_text:
            # Strip only the exact `v1` header line, never a `v1`-prefixed first
            # token of a malformed/hand-edited file (e.g. `v10...`, `v1: ...`).
            display_text = memory_text.removeprefix("v1\n").strip()
            await self._send_memory_pages(
                interaction=interaction,
                text=display_text,
                footer_text=memory_footer_text(pending_count=pending_count),
                title=_MEMORY_TITLE,
            )
            return
        if pending_count:
            # Extraction has produced raw observations but the first
            # consolidation has not run yet; saying "no memory" here would
            # contradict what the user just experienced in chat.
            embed = Embed(
                title=_MEMORY_TITLE,
                description=(
                    f"我已經記下 {pending_count} 筆對你的觀察，正在整理成長期記憶，"
                    "再多聊幾次就會在這裡看到完整內容。"
                ),
                color=MEMORY_EMBED_COLOR,
            )
        else:
            embed = Embed(
                title=_MEMORY_TITLE,
                description="目前還沒有任何記憶，多跟我聊聊，我會慢慢認識你。",
                color=MEMORY_EMBED_COLOR,
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _show_detail(self, interaction: Interaction) -> None:
        """Shows the newest window of the caller's cold-tier detail log."""
        if interaction.user is None:
            return
        detail_text = read_detail_tail(
            user_id=interaction.user.id, max_chars=MEMORY_DETAIL_VIEW_MAX_CHARS
        )
        if not detail_text:
            embed = Embed(
                title=_DETAIL_TITLE,
                description="目前還沒有任何詳細記錄，等我整理過幾輪記憶後就會出現。",
                color=MEMORY_EMBED_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        await self._send_memory_pages(
            interaction=interaction,
            text=detail_text,
            footer_text="已整理過的詳細觀察記錄，僅顯示最近的視窗",
            title=_DETAIL_TITLE,
        )

    async def _send_memory_pages(
        self, interaction: Interaction, text: str, footer_text: str, title: str
    ) -> None:
        """Sends paginated memory pages, attaching the pager only when needed."""
        pages = paginate_on_lines(text=text, limit=MEMORY_PAGE_MAX_CHARS)
        embed = build_memory_embed(
            page_text=pages[0],
            page_index=0,
            page_count=len(pages),
            footer_text=footer_text,
            title=title,
        )
        if len(pages) == 1:
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        view = MemoryPagesView(pages=pages, footer_text=footer_text, title=title)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        view.bind_origin(interaction=interaction)

    @memory.subcommand(
        name="clear",
        description="Clear everything the bot remembers about you.",
        name_localizations={Locale.zh_TW: "清除", Locale.ja: "消去"},
        description_localizations={
            Locale.zh_TW: "清除 bot 對你的所有長期記憶",
            Locale.ja: "ボットがあなたについて記憶している内容をすべて消去します。",
        },
    )
    async def memory_clear(self, interaction: Interaction) -> None:
        """Deletes the caller's memory files and aborts in-flight updates.

        Deliberately lock-free: a background memory update can hold the
        per-user lock across LLM calls far past Discord's interaction ack
        window. `clear_user_memory` flags `mark_cleared`, and in-flight
        updates re-check `cleared_since` before every write, so the clear
        cannot be resurrected by a slower background task.
        """
        if interaction.user is None:
            return
        if not self.memory_config.clear_enabled:
            embed = Embed(
                title="🧹 記憶清除",
                description="記憶清除功能暫時停用，你仍然可以用 `/memory show` 查看我對你的記憶。",
                color=_CLEAR_DISABLED_EMBED_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        removed = clear_user_memory(user_id=interaction.user.id)
        description = "已清除我對你的所有記憶。" if removed else "本來就沒有任何記憶，無事發生。"
        embed = Embed(title="🧹 記憶清除", description=description, color=_CLEAR_EMBED_COLOR)
        await interaction.response.send_message(embed=embed, ephemeral=True)


def setup(bot: commands.Bot) -> None:
    """Adds the MemoryCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(MemoryCogs(bot), override=True)
