"""Slash commands for viewing, regenerating, and clearing long-term memory.

`/memory show`, `/memory regenerate` and `/memory clear` operate on the caller's
own per-user memory; `/memory server show` views the bot's per-server
(community) memory for the current guild. Only the personal scope is erasable
from chat, and only behind a confirmation: server memory stays
operator-maintained.
"""

from functools import cached_property

from openai import AsyncOpenAI
import nextcord
from nextcord import Embed, Locale, Interaction
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.colors import DISCORD_GREEN, DISCORD_YELLOW
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs.memory.views import (
    MEMORY_EMBED_COLOR,
    MEMORY_PAGE_MAX_CHARS,
    MemoryPagesView,
    MemoryClearConfirmView,
    compartment_label,
    paginate_on_lines,
    build_memory_embed,
    memory_footer_text,
    build_clear_confirm_embed,
)
from discordbot.utils.llm_transcript import render_author_identity
from discordbot.services.memory.store import (
    read_tone,
    user_scope,
    server_scope,
    count_raw_entries,
    list_compartments,
    read_memory_document,
)
from discordbot.services.memory.pipeline import (
    flavor_of,
    regeneration_on_cooldown,
    regeneration_has_evidence,
    schedule_memory_regeneration,
)
from discordbot.services.memory.extraction import MemoryExtractorAI

_SUCCESS_EMBED_COLOR = DISCORD_GREEN
_WARN_EMBED_COLOR = DISCORD_YELLOW

_MEMORY_TITLE = "🧠 我對你的記憶"
_SERVER_MEMORY_TITLE = "🧠 我對這個伺服器的記憶"
_REGEN_TITLE = "🔄 記憶重建"
_REGEN_COOLDOWN_DESCRIPTION = "記憶重建剛執行過，請稍後再試。"

# `/memory show` is the owner reading their own store, so it is not bound by what a
# reply prompt can carry; the pager splits whatever comes back. Kept finite only so a
# corrupted tree cannot build an unbounded string.
_SHOW_MAX_CHARS = 200_000


class MemoryCogs(commands.Cog):
    """Provides the long-term memory viewing, regeneration, and clearing commands.

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
            extract_model=self.runtime_models.memory_extractor_model,
            evaluate_model=self.runtime_models.memory_writer_model,
            consolidate_model=self.runtime_models.memory_writer_model,
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
    async def memory(self, interaction: Interaction[commands.Bot]) -> None:
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
    async def memory_show(self, interaction: Interaction[commands.Bot]) -> None:
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
            tone_text=read_tone(scope=scope),
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
    async def memory_server(self, interaction: Interaction[commands.Bot]) -> None:
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
    async def memory_server_show(self, interaction: Interaction[commands.Bot]) -> None:
        """Shows the bot's consolidated memory of the current server, paginated."""
        if interaction.guild is None:
            # Per-server memory only exists inside a guild; there is no scope in DMs.
            embed = Embed(
                title=_SERVER_MEMORY_TITLE,
                description="這個指令只能在伺服器裡使用。",
                color=_WARN_EMBED_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        scope = server_scope(server_id=interaction.guild.id)
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

    async def _show_memory(  # noqa: PLR0913 -- display strings plus the optional tone section
        self,
        interaction: Interaction[commands.Bot],
        scope: str,
        title: str,
        empty_description: str,
        pending_template: str,
        tone_text: str = "",
    ) -> None:
        """Shows a scope's stored memory, or a friendly placeholder when empty.

        `tone_text` is the per-user tone note (empty for the per-server view); when
        present it leads the display as its own section, and it counts as content so
        a user with only a tone note still sees it instead of the empty placeholder.

        The caller's own memory is shown compartment by compartment, each under a
        heading naming who can see it. Provenance used to be a per-bullet tag the model
        wrote and the reply path had to strip; now it is which directory a fact lives
        in, so showing it costs nothing and tells the owner exactly where each thing
        they told the bot can come back up.
        """
        pending_count = count_raw_entries(scope=scope)
        sections: list[str] = []
        if tone_text:
            sections.append(tone_text)
        sections.extend(self._memory_sections(scope=scope))
        if sections:
            await self._send_memory_pages(
                interaction=interaction,
                text="\n\n".join(sections),
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

    def _memory_sections(self, scope: str) -> list[str]:
        """Renders one scope's compartments as labelled display sections.

        A server scope has exactly one compartment and no boundary to explain, so it is
        rendered bare; a user scope gets one heading per compartment. The cap is lifted
        far above the injection ceiling here on purpose: this view is for the owner, so
        it should show everything stored rather than what a reply would fit.
        """
        flavor = flavor_of(scope=scope)
        compartments = list_compartments(scope=scope)
        if flavor == "server":
            document = read_memory_document(
                scope=scope, compartments=compartments, flavor=flavor, max_chars=_SHOW_MAX_CHARS
            )
            return [document] if document else []
        sections: list[str] = []
        for compartment in compartments:
            document = read_memory_document(
                scope=scope, compartments=[compartment], flavor=flavor, max_chars=_SHOW_MAX_CHARS
            )
            if document:
                label = compartment_label(compartment=compartment, bot=self.bot)
                sections.append(f"# {label}\n{document}")
        return sections

    async def _send_memory_pages(
        self, interaction: Interaction[commands.Bot], text: str, footer_text: str, title: str
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
    async def memory_regenerate(self, interaction: Interaction[commands.Bot]) -> None:
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
        # The rebuild is one LLM call per non-empty compartment plus one for the tone
        # note, and runs far past Discord's
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

    @memory.subcommand(
        name="clear",
        description="Erase everything the bot remembers about you.",
        name_localizations={Locale.zh_TW: "清除", Locale.ja: "削除"},
        description_localizations={
            Locale.zh_TW: "清除 bot 對你的所有長期記憶",
            Locale.ja: "あなたについてボットが記憶している内容をすべて削除します。",
        },
    )
    async def memory_clear(self, interaction: Interaction[commands.Bot]) -> None:
        """Asks for confirmation before erasing the caller's own memory."""
        if interaction.user is None:
            return
        # The wipe is irreversible and covers tiers `/memory show` never displays,
        # so the command only opens the prompt; the view owns the clear itself.
        view = MemoryClearConfirmView(scope=user_scope(user_id=interaction.user.id))
        await interaction.response.send_message(
            embed=build_clear_confirm_embed(), view=view, ephemeral=True
        )
        view.bind_origin(interaction=interaction)


def setup(bot: commands.Bot) -> None:
    """Adds the MemoryCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(MemoryCogs(bot), override=True)
