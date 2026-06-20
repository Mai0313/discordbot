"""Slash commands for viewing and regenerating long-term memory.

`/memory show` and `/memory regenerate` operate on the caller's own per-user
memory; `/memory server show` views the bot's per-server (community) memory for
the current guild. All three are read-only or rebuild-only by design: there is
no user-facing clear, so memory is never deleted from chat.
"""

from functools import cached_property

from openai import AsyncOpenAI
import nextcord
from nextcord import Embed, Locale, Interaction
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs._memory.store import (
    user_scope,
    server_scope,
    read_main_memory,
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
from discordbot.cogs._gen_reply.input import render_author_identity
from discordbot.cogs._memory.pipeline import (
    regeneration_on_cooldown,
    regeneration_has_evidence,
    schedule_memory_regeneration,
)
from discordbot.cogs._memory.extraction import MemoryExtractorAI

_SUCCESS_EMBED_COLOR = 0x57F287
_WARN_EMBED_COLOR = 0xFEE75C

_MEMORY_TITLE = "🧠 我對你的記憶"
_SERVER_MEMORY_TITLE = "🧠 我對這個伺服器的記憶"
_REGEN_TITLE = "🔄 記憶重建"
_REGEN_COOLDOWN_DESCRIPTION = "記憶重建剛執行過，請稍後再試。"


class MemoryCogs(commands.Cog):
    """Provides the long-term memory viewing and regeneration commands.

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: The LLM client configuration used for memory regeneration.
        runtime_models: Catalog providing the memory model settings.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the memory cog.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()
        self.runtime_models = RuntimeModelCatalog()

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The cached AsyncOpenAI client instance.

        Returns:
            A configured AsyncOpenAI client reused across regeneration requests.
        """
        return AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    @cached_property
    def memory_extractor(self) -> MemoryExtractorAI:
        """The cached memory extraction service used for regeneration.

        Returns:
            An extractor bound to this cog's client and the memory models.
        """
        return MemoryExtractorAI(
            client=self.client,
            extract_model=self.runtime_models.extract_model,
            evaluate_model=self.runtime_models.memory_evaluator_model,
            consolidate_model=self.runtime_models.memories_model,
        )

    @nextcord.slash_command(
        name="memory",
        description="Manage what the bot remembers.",
        name_localizations={Locale.zh_TW: "記憶", Locale.ja: "メモリー"},
        description_localizations={
            Locale.zh_TW: "管理 bot 的長期記憶",
            Locale.ja: "ボットの長期記憶を管理します。",
        },
        nsfw=False,
    )
    async def memory(self, interaction: Interaction) -> None:
        """Slash command group for memory management."""

    @memory.subcommand(
        name="show",
        description="Show what the bot remembers about you.",
        name_localizations={Locale.zh_TW: "查看", Locale.ja: "表示"},
        description_localizations={
            Locale.zh_TW: "查看 bot 對你的長期記憶",
            Locale.ja: "ボットがあなたについて記憶している内容を表示します。",
        },
    )
    async def memory_show(self, interaction: Interaction) -> None:
        """Shows the caller's consolidated memory, paginated."""
        if interaction.user is None:
            return
        scope = user_scope(user_id=interaction.user.id)
        await self._show_memory(
            interaction=interaction,
            scope=scope,
            title=_MEMORY_TITLE,
            empty_description="目前還沒有任何記憶，多跟我聊聊，我會慢慢認識你。",
            pending_template=(
                "我已經記下 {count} 筆對你的觀察，正在整理成長期記憶，"
                "再多聊幾次就會在這裡看到完整內容。"
            ),
        )

    @memory.subcommand(
        name="server",
        description="View the bot's memory of this server.",
        name_localizations={Locale.zh_TW: "伺服器", Locale.ja: "サーバー"},
        description_localizations={
            Locale.zh_TW: "查看 bot 對這個伺服器的記憶",
            Locale.ja: "このサーバーについてボットが記憶している内容を確認します。",
        },
    )
    async def memory_server(self, interaction: Interaction) -> None:
        """Subcommand group for per-server memory viewing."""

    @memory_server.subcommand(
        name="show",
        description="Show what the bot remembers about this server's community.",
        name_localizations={Locale.zh_TW: "查看", Locale.ja: "表示"},
        description_localizations={
            Locale.zh_TW: "查看 bot 對這個伺服器社群的長期記憶",
            Locale.ja: "このサーバーのコミュニティについてボットが記憶している内容を表示します。",
        },
    )
    async def memory_server_show(self, interaction: Interaction) -> None:
        """Shows the bot's consolidated memory of the current server, paginated."""
        if interaction.guild is None or self.bot.user is None:
            # Per-server memory only exists inside a guild; there is no scope in DMs.
            embed = Embed(
                title=_SERVER_MEMORY_TITLE,
                description="這個指令只能在伺服器裡使用。",
                color=_WARN_EMBED_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        scope = server_scope(bot_id=self.bot.user.id, server_id=interaction.guild.id)
        await self._show_memory(
            interaction=interaction,
            scope=scope,
            title=_SERVER_MEMORY_TITLE,
            empty_description="我還沒有對這個伺服器的記憶，多在這裡聊聊，我會慢慢認識這個社群。",
            pending_template=(
                "我已經記下 {count} 筆對這個伺服器的觀察，正在整理成長期記憶，"
                "再多聊幾次就會在這裡看到完整內容。"
            ),
        )

    async def _show_memory(
        self,
        interaction: Interaction,
        scope: str,
        title: str,
        empty_description: str,
        pending_template: str,
    ) -> None:
        """Shows a scope's consolidated memory, or a friendly placeholder when empty."""
        memory_text = read_main_memory(scope=scope)
        pending_count = count_raw_entries(scope=scope)
        if memory_text:
            # Strip only the exact `v1` header line, never a `v1`-prefixed first
            # token of a malformed/hand-edited file (e.g. `v10...`, `v1: ...`).
            display_text = memory_text.removeprefix("v1\n").strip()
            await self._send_memory_pages(
                interaction=interaction,
                text=display_text,
                footer_text=memory_footer_text(pending_count=pending_count),
                title=title,
            )
            return
        # Extraction may have produced raw observations before the first
        # consolidation ran; saying "no memory" then would contradict chat.
        description = (
            pending_template.format(count=pending_count) if pending_count else empty_description
        )
        embed = Embed(title=title, description=description, color=MEMORY_EMBED_COLOR)
        await interaction.response.send_message(embed=embed, ephemeral=True)

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
        name="regenerate",
        description="Rebuild what the bot remembers about you from its observation log.",
        name_localizations={Locale.zh_TW: "重建", Locale.ja: "再生成"},
        description_localizations={
            Locale.zh_TW: "只根據觀察記錄，從頭重建 bot 對你的長期記憶",
            Locale.ja: "観察ログだけを使って、あなたに関する記憶を一から作り直します。",
        },
    )
    async def memory_regenerate(self, interaction: Interaction) -> None:
        """Schedules a background rebuild of the caller's memory from evidence alone."""
        if interaction.user is None:
            return
        scope = user_scope(user_id=interaction.user.id)
        if regeneration_on_cooldown(scope=scope):
            embed = Embed(
                title=_REGEN_TITLE,
                description=_REGEN_COOLDOWN_DESCRIPTION,
                color=_WARN_EMBED_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        if not regeneration_has_evidence(scope=scope):
            # A from-scratch rebuild needs cold-tier evidence; without any, the
            # background task would silently no-op, so say so up front instead
            # of claiming a rebuild was scheduled.
            embed = Embed(
                title=_REGEN_TITLE,
                description="目前還沒有足夠的觀察記錄可以重建記憶，多跟我聊聊吧。",
                color=_WARN_EMBED_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        # The rebuild is one whole-file LLM rewrite that runs far past Discord's
        # ack window, so it is dispatched to the background task queue and the
        # command replies immediately; the user checks back with `/memory show`.
        scheduled = schedule_memory_regeneration(
            scope=scope,
            extractor=self.memory_extractor,
            identity=render_author_identity(
                display_name=interaction.user.display_name,
                username=interaction.user.name,
                user_id=interaction.user.id,
            ),
        )
        if scheduled:
            description = "已排程重建記憶，整理完成後可以用 `/memory show` 查看。"
            color = _SUCCESS_EMBED_COLOR
        else:
            description = "記憶正在重建中，完成後可以用 `/memory show` 查看。"
            color = _WARN_EMBED_COLOR
        embed = Embed(title=_REGEN_TITLE, description=description, color=color)
        await interaction.response.send_message(embed=embed, ephemeral=True)


def setup(bot: commands.Bot) -> None:
    """Adds the MemoryCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(MemoryCogs(bot), override=True)
