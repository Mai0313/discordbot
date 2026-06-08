"""Slash commands for viewing, clearing, and regenerating per-user long-term memory."""

from functools import cached_property

from openai import AsyncOpenAI
import nextcord
from nextcord import Embed, Locale, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.utils.llm import create_litellm_client
from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import RuntimeModelCatalog
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
from discordbot.cogs._gen_reply.input import render_author_identity
from discordbot.cogs._memory.pipeline import regenerate_main_memory, regeneration_on_cooldown
from discordbot.cogs._memory.constants import MEMORY_DETAIL_VIEW_MAX_CHARS
from discordbot.cogs._memory.extraction import MemoryExtractorAI

_SUCCESS_EMBED_COLOR = 0x57F287
_WARN_EMBED_COLOR = 0xFEE75C
_ERROR_EMBED_COLOR = 0xED4245

_MEMORY_TITLE = "🧠 我對你的記憶"
_DETAIL_TITLE = "🧠 詳細記憶記錄"
_REGEN_TITLE = "🔄 記憶重建"
_REGEN_COOLDOWN_DESCRIPTION = "記憶重建剛執行過，請稍後再試。"


class MemoryCogs(commands.Cog):
    """Provides the personal long-term memory management commands.

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
        return create_litellm_client(config=self.config)

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
        updates re-check `cleared_since` before rewriting memory state, so
        a slower background task cannot resurrect the cleared memory.
        """
        if interaction.user is None:
            return
        removed = clear_user_memory(user_id=interaction.user.id)
        description = "已清除我對你的所有記憶。" if removed else "本來就沒有任何記憶，無事發生。"
        embed = Embed(title="🧹 記憶清除", description=description, color=_SUCCESS_EMBED_COLOR)
        await interaction.response.send_message(embed=embed, ephemeral=True)

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
        """Rebuilds the caller's consolidated memory from cold-tier evidence alone."""
        if interaction.user is None:
            return
        if regeneration_on_cooldown(user_id=interaction.user.id):
            embed = Embed(
                title=_REGEN_TITLE,
                description=_REGEN_COOLDOWN_DESCRIPTION,
                color=_WARN_EMBED_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        # The rebuild is one whole-file LLM rewrite that can run far past
        # Discord's 3-second ack window; defer keeps the interaction alive.
        await interaction.response.defer(ephemeral=True)
        result = await regenerate_main_memory(
            user_id=interaction.user.id,
            extractor=self.memory_extractor,
            identity=render_author_identity(
                display_name=interaction.user.display_name,
                username=interaction.user.name,
                user_id=interaction.user.id,
            ),
        )
        outcomes = {
            "regenerated": (
                "已根據觀察記錄重新整理我對你的長期記憶，可以用 `/memory show` 查看。",
                _SUCCESS_EMBED_COLOR,
            ),
            "no_evidence": (
                "目前還沒有足夠的觀察記錄可以重建記憶，多跟我聊聊吧。",
                _WARN_EMBED_COLOR,
            ),
            "failed": ("重建失敗，已保留原本的記憶，請稍後再試。", _ERROR_EMBED_COLOR),
            "cooldown": (_REGEN_COOLDOWN_DESCRIPTION, _WARN_EMBED_COLOR),
        }
        description, color = outcomes[result]
        embed = Embed(title=_REGEN_TITLE, description=description, color=color)
        await interaction.followup.send(embed=embed, ephemeral=True)


def setup(bot: commands.Bot) -> None:
    """Adds the MemoryCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(MemoryCogs(bot), override=True)
