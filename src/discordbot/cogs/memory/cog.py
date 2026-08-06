"""The long-term memory Discord surface: viewing, rebuilding and erasing what the bot remembers.

Registers one cog, `MemoryCogs`, and no listeners. `/memory show`, `/memory regenerate` and
`/memory clear` act on the caller's OWN per-user scope, which is keyed by Discord id and so covers
every server plus their DMs; `/memory server show` reads the bot's per-server (community) scope and
is refused outside a guild, because that scope does not exist in a DM. Every reply is ephemeral —
this is where a user sees their own stored memory, and it is nobody else's business.

Nothing here stores, renders or erases anything itself. Scopes, compartments and the rendered
document come from `services/memory/store.py`; the background rebuild and the clear from
`services/memory/pipeline.py`; the embeds, the pager and the confirmation buttons from `views.py`.
What is left is the Discord half: resolve the scope, decide whether there is anything to show, pick
the placeholder that matches when there is not, and page what there is.

Three asymmetries are deliberate. There is no server-side rebuild or clear, because community
memory stays operator-maintained. `/memory clear` only OPENS its confirmation and
`MemoryClearConfirmView` owns the wipe, which is irreversible and reaches tiers `/memory show`
never displays. And `/memory regenerate` answers before its work starts: a from-scratch rebuild is
minutes of LLM work, far past Discord's acknowledgement window, so it is dispatched to the
pipeline's background queue and the user checks back with `/memory show`.

No kill-switch and no permission gate. The commands are self-scoped instead, so a caller can only
ever reach their own memory and the one server-scoped command is read-only. The proxy client and
the extractor behind the rebuild are `cached_property`, so a deployment where nobody runs
`/memory regenerate` never builds either.
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
    """Registers the `/memory` group, its `/memory server` subgroup, and no listeners.

    Attributes:
        bot: The Discord bot instance that owns this cog, and the only way a compartment
            heading can name a guild.
        config: LLM proxy settings, read once at load and used only by the rebuild path.
        runtime_models: Catalog supplying the memory model tiers handed to the extractor.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the cog and reads the settings the rebuild path will need.

        Args:
            bot (commands.Bot): The Discord bot instance that owns this cog.
        """
        self.bot = bot
        self.config = LLMConfig()
        self.runtime_models = RuntimeModelCatalog()

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The LiteLLM-proxy client, opened on first use.

        Only `/memory regenerate` reaches it, so a deployment where nobody runs that command
        never opens a client at all.

        Returns:
            An `AsyncOpenAI` bound to the proxy, shared by every rebuild this process runs.
        """
        return AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    @cached_property
    def memory_extractor(self) -> MemoryExtractorAI:
        """The extraction service handed to a background rebuild, built on first use.

        A rebuild only ever runs the consolidation half, so `extract_model` and `evaluate_model`
        are filled in to complete the service rather than because this path calls them. The
        per-user prompts are left at their defaults, which is correct here since only the
        personal scope is rebuildable from chat.

        Returns:
            A `MemoryExtractorAI` on this cog's client and the catalog's memory model tiers.
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
        """Registration anchor for the `/memory` group; Discord only dispatches its subcommands.

        Args:
            interaction (Interaction[commands.Bot]): Unused; nextcord requires the parameter on
                a group callback.
        """

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
        """Shows the caller their own consolidated memory, tone note first, paginated.

        The only command that reads the tone note, since it is a per-user tier the per-server
        view has no counterpart for. An interaction carrying no user names no scope to read, so
        it is dropped without a reply.

        Args:
            interaction (Interaction[commands.Bot]): The invoking interaction, not yet answered.
        """
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
        """Registration anchor for the `/memory server` subgroup; only its leaf is dispatched.

        Args:
            interaction (Interaction[commands.Bot]): Unused; nextcord requires the parameter on
                a group callback.
        """

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
        """Shows the bot's consolidated memory of the current server, paginated.

        Read-only, and the whole subgroup: community memory is operator-maintained, so there is
        no server-side rebuild or clear to pair with it. The scope has one compartment and no
        tone note, so the document is rendered bare rather than under visibility headings.

        Args:
            interaction (Interaction[commands.Bot]): The invoking interaction, not yet answered.
        """
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
        """Shows a scope's stored memory, or a friendly placeholder when there is none.

        The tone note leads the display as its own section when there is one, and it counts as
        content, so a user with only a tone note sees it instead of the empty placeholder.

        The caller's own memory is shown compartment by compartment, each under a heading naming
        who can see it. Provenance used to be a per-bullet tag the model wrote and the reply path
        had to strip; now it is which directory a fact lives in, so showing it costs nothing and
        tells the owner exactly where each thing they told the bot can come back up.

        Args:
            interaction (Interaction[commands.Bot]): The invoking interaction, answered here.
            scope (str): The memory scope to read, personal or per-server.
            title (str): Embed title shared by every page.
            empty_description (str): Shown when the scope holds nothing at all.
            pending_template (str): Shown instead when only unconsolidated observations exist;
                formatted with a `count` field.
            tone_text (str): The per-user tone note, empty for the per-server view.
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

        A server scope has exactly one compartment and no boundary to explain, so it is rendered
        bare; a user scope gets one heading per compartment. The cap is lifted far above the
        injection ceiling on purpose: this view is for the owner, so it should show everything
        stored rather than what a reply would fit.

        Args:
            scope (str): The memory scope to read.

        Returns:
            One section per non-empty compartment in `list_compartments` order, empty when the
            scope has nothing stored.
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
        """Sends the first page, attaching the pager view only when there is more than one.

        The view is bound to the interaction after the send, since its timeout needs the
        originating interaction to disable its own buttons.

        Args:
            interaction (Interaction[commands.Bot]): The invoking interaction, answered here.
            text (str): The whole document to split across pages.
            footer_text (str): Footer line shared by every page.
            title (str): Embed title shared by every page.
        """
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
        """Schedules a background rebuild of the caller's memory from evidence alone.

        Both refusals are checked here rather than left to the background task, which reports
        only to the log: a cooldown and a scope with no cold-tier evidence would each be a
        silent no-op the user was told had been scheduled. The pipeline re-checks the cooldown
        under the scope lock, so this one is the up-front answer and not the authority.

        Args:
            interaction (Interaction[commands.Bot]): The invoking interaction, answered here.
        """
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
        """Opens the confirmation prompt guarding an erase of the caller's own memory.

        Erases nothing itself. The view is bound to the interaction after the send so its
        timeout can disable its own buttons.

        Args:
            interaction (Interaction[commands.Bot]): The invoking interaction, answered here.
        """
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
    """Registers `MemoryCogs` on the bot.

    Sync on purpose: the loader schedules an `async def setup` without awaiting it, which breaks
    the first command sync.

    Args:
        bot (commands.Bot): The bot to add the cog to.
    """
    bot.add_cog(MemoryCogs(bot), override=True)
