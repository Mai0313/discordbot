"""Slash commands for viewing, regenerating, and clearing long-term memory.

`/memory show`, `/memory regenerate` and `/memory clear` operate on the caller's
own per-user memory; `/memory server show` views the bot's per-server
(community) memory for the current guild, and `/memory server catchup` reads the
channel it is used in into that memory. Only the personal scope is erasable
from chat, and only behind a confirmation: server memory stays
operator-maintained.
"""

import asyncio
from functools import cached_property

from openai import AsyncOpenAI, OpenAIError
import logfire
import nextcord
from nextcord import Embed, Locale, Message, Forbidden, Interaction, HTTPException
from pydantic import Field, BaseModel, ConfigDict, SkipValidation
from nextcord.abc import Messageable
from nextcord.ext import commands
from nextcord.utils import utcnow
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.colors import DISCORD_RED, DISCORD_GREEN, DISCORD_YELLOW
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.typings.timeouts import INTERACTION_DELIVERY_MARGIN_SECONDS
from discordbot.utils.llm_errors import extract_friendly_error
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
from discordbot.utils.asyncio_locks import LoopLocalRegistry
from discordbot.utils.discord_embeds import DISCORD_EMBED_DESCRIPTION_LIMIT, clip_to_utf16_limit
from discordbot.utils.llm_transcript import render_author_identity, render_server_identity
from discordbot.services.memory.store import (
    flavor_of,
    read_tone,
    user_scope,
    server_scope,
    count_raw_entries,
    list_compartments,
    read_memory_document,
)
from discordbot.services.memory.writer import (
    MemoryWriterAI,
    MemoryObservation,
    server_subject,
    transcript_from_messages,
)
from discordbot.services.memory.catchup import (
    catchup_on_cooldown,
    review_catchup_notes,
    start_catchup_cooldown,
    release_catchup_cooldown,
)
from discordbot.typings.context_budgets import HISTORY_MESSAGE_LIMIT
from discordbot.utils.channel_visibility import channel_is_public
from discordbot.services.memory.regeneration import (
    regeneration_on_cooldown,
    regeneration_has_evidence,
    schedule_memory_regeneration,
)
from discordbot.services.memory.consolidation import consolidate_if_needed

_SERVER_MEMORY_TITLE = "🧠 我對這個伺服器的記憶"
_REGEN_TITLE = "🔄 記憶重建"
_CATCHUP_TITLE = "📚 跟上進度"

# A catchup's result colours; the next catchup in that channel starts reading after the newest
# result in one of them. A failure is red and the start message is the memory colour, so
# neither counts, and a failed run is read again.
_CATCHUP_CHECKPOINT_COLORS = (DISCORD_GREEN, DISCORD_YELLOW)

# The running catchup per guild, so a second one there is refused rather than started.
_catchup_tasks: LoopLocalRegistry[int, asyncio.Task[None]] = LoopLocalRegistry()

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
    def memory_writer(self) -> MemoryWriterAI:
        """The cached memory writing service used for regeneration.

        Returns:
            A writer bound to this cog's client and the memory model.
        """
        return MemoryWriterAI(client=self.client, model=self.runtime_models.memory_writer_model)

    @nextcord.slash_command(
        name="memory",
        description="Manage what the bot remembers.",
        name_localizations={Locale.zh_TW: "記憶", Locale.ja: "メモリー"},
        description_localizations={
            Locale.zh_TW: "管理 bot 的長期記憶",
            Locale.ja: "ボットの長期記憶を管理します。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
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
            title="🧠 我對你的記憶",
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
            # Server memory is shown only where the bot is a member. A user install also runs this
            # in servers the bot was never added to, which carry a guild id but resolve no guild.
            description = (
                "我沒有被加進這個伺服器，這個指令只能在我所在的伺服器裡使用。"
                if interaction.guild_id is not None
                else "這個指令只能在伺服器裡使用。"
            )
            embed = Embed(
                title=_SERVER_MEMORY_TITLE, description=description, color=DISCORD_YELLOW
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
        heading naming who can see it: the directory a fact lives in IS the privacy
        boundary, so showing it costs nothing and tells the owner exactly where each
        thing they told the bot can come back up.
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
        # A review may have staged raw observations before the first
        # consolidation ran; saying "no memory" then would contradict chat.
        description = (
            pending_template.format(count=pending_count) if pending_count else empty_description
        )
        embed = Embed(title=title, description=description, color=MEMORY_EMBED_COLOR)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    def _memory_sections(self, scope: str) -> list[str]:
        """Renders one scope's compartments as labelled display sections.

        A server scope has no compartment boundary to explain, so it is rendered bare;
        a user scope gets one heading per compartment.
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

    @memory_server.subcommand(
        name="catchup",
        description="Read this channel's recent conversation into server memory.",
        name_localizations={Locale.zh_TW: "跟上進度", Locale.ja: "キャッチアップ"},
        description_localizations={
            Locale.zh_TW: "讀這個頻道最近的對話，整理進伺服器記憶",
            Locale.ja: "このチャンネルの最近の会話をサーバーの記憶に取り込みます。",
        },
    )
    async def memory_server_catchup(self, interaction: Interaction[commands.Bot]) -> None:
        """Reads the channel since the last catchup and folds it into the server's memory.

        Every refusal reaches only the caller. A run posts a public start message, then
        replaces it with a public result, which is also where the next run in that channel
        stops reading. A missing proxy key is left to fail like any other slash command error:
        the writer is built before anything is posted, so no start message is left behind.
        """
        guild = interaction.guild
        if guild is None:
            # A user install also runs this in servers the bot was never added to, which carry
            # a guild id but resolve no guild.
            await self._refuse_catchup(
                interaction=interaction,
                text=(
                    "把老子加進來伺服器"
                    if interaction.guild_id is not None
                    else "這個指令只能在伺服器裡使用。"
                ),
            )
            return
        channel = interaction.channel
        if not isinstance(channel, Messageable) or not channel_is_public(
            guild=guild, channel=channel
        ):
            await self._refuse_catchup(
                interaction=interaction, text="這裡是私人頻道, 不要偷偷幹壞事"
            )
            return
        running = _catchup_tasks.get(key=guild.id)
        if running is not None and not running.done():
            await self._refuse_catchup(
                interaction=interaction,
                text="已經有人在整理這個伺服器的對話，完成後可以用 `/memory server show` 查看。",
            )
            return
        scope = server_scope(server_id=guild.id)
        if catchup_on_cooldown(scope=scope):
            await self._refuse_catchup(interaction=interaction, text="記過了啦 吵啥")
            return
        writer = self.memory_writer
        # Claimed before the first await, so two callers racing through the checks above
        # cannot both start a run, and given back on every way out that starts none.
        start_catchup_cooldown(scope=scope)
        started = False
        try:
            started = await self._start_catchup(
                interaction=interaction, channel=channel, writer=writer
            )
        finally:
            if not started:
                release_catchup_cooldown(scope=scope)

    async def _start_catchup(
        self, interaction: Interaction[commands.Bot], channel: Messageable, writer: MemoryWriterAI
    ) -> bool:
        """Reads the channel and, when there is something to read, starts the run.

        Returns whether a run started; the caller gives the cooldown back when none did.
        """
        history = await _read_since_checkpoint(interaction=interaction, channel=channel)
        if history is None:
            await self._refuse_catchup(interaction=interaction, text="看不到啦 不給權限我是要看啥")
            return False
        if not history.messages:
            await self._refuse_catchup(interaction=interaction, text="你們啥也沒聊我是要記個鬼")
            return False
        count = len(history.messages)
        intro = (
            f"讀了上次跟上進度之後的 {count} 則新對話"
            if history.since_checkpoint
            else f"讀了這個頻道最近的 {count} 則對話"
        )
        start = Embed(
            title=_CATCHUP_TITLE,
            description=f"{intro}，正在查看近期對話計入小本本⋯",
            color=MEMORY_EMBED_COLOR,
        )
        await interaction.response.send_message(embed=start)
        if interaction.guild_id is not None:
            _catchup_tasks.set(
                key=interaction.guild_id,
                value=asyncio.create_task(
                    self._run_catchup(
                        interaction=interaction,
                        messages=history.messages,
                        intro=intro,
                        read_until=history.read_until,
                        writer=writer,
                    )
                ),
            )
        return True

    async def _run_catchup(
        self,
        interaction: Interaction[commands.Bot],
        messages: list[Message],
        intro: str,
        read_until: str,
        writer: MemoryWriterAI,
    ) -> None:
        """Reviews the read messages, replaces the start message with the result, consolidates.

        The review is bounded by what the interaction token has left less the margin to
        deliver, because the start message can be replaced only through that token: past it,
        the start message would stand for good.
        """
        guild = interaction.guild
        user = interaction.user
        if guild is None or user is None:
            return
        scope = server_scope(server_id=guild.id)
        transcript = transcript_from_messages(
            message_list=[_catchup_input(message=message) for message in messages], full_reply=""
        )
        budget = (
            interaction.expires_at - utcnow()
        ).total_seconds() - INTERACTION_DELIVERY_MARGIN_SECONDS
        observations: tuple[MemoryObservation, ...] | None
        try:
            async with asyncio.timeout(budget):
                observations = await review_catchup_notes(
                    scope=scope,
                    subject=server_subject(server_id=guild.id),
                    transcript=transcript,
                    writer=writer,
                )
        except TimeoutError:
            logfire.warn("Memory catchup ran out of the interaction's time", guild_id=guild.id)
            observations = None
        except Exception as exc:
            # Broad on purpose: this is a detached task, and the start message must still be
            # replaced with a result whatever broke underneath it.
            logfire.error(
                "Memory catchup failed",
                guild_id=guild.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            observations = None
        await _post_catchup_result(
            interaction=interaction,
            result=_catchup_result(
                intro=intro,
                observations=observations,
                invoker=user.display_name,
                read_until=read_until,
            ),
        )
        if observations:
            await consolidate_if_needed(
                scope=scope,
                writer=writer,
                identity=render_server_identity(server_name=guild.name, server_id=guild.id),
            )

    async def _refuse_catchup(self, interaction: Interaction[commands.Bot], text: str) -> None:
        """Tells only the caller why no catchup runs."""
        embed = Embed(title=_CATCHUP_TITLE, description=text, color=DISCORD_YELLOW)
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
    async def memory_regenerate(self, interaction: Interaction[commands.Bot]) -> None:
        """Schedules a background rebuild of the caller's memory from evidence alone."""
        if interaction.user is None:
            return
        scope = user_scope(user_id=interaction.user.id)
        if regeneration_on_cooldown(scope=scope):
            embed = Embed(
                title=_REGEN_TITLE,
                description="記憶重建剛執行過，請稍後再試。",
                color=DISCORD_YELLOW,
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
                color=DISCORD_YELLOW,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        try:
            writer = self.memory_writer
        except OpenAIError as exc:
            # The SDK refuses to build the client when no proxy key is configured.
            logfire.info(
                "memory regeneration needs the proxy key; none is configured", scope=scope
            )
            embed = Embed(
                title=_REGEN_TITLE,
                description=f"```\n{extract_friendly_error(exc=exc)}\n```",
                color=DISCORD_RED,
            )
            embed.set_footer(text=type(exc).__name__)
            await interaction.response.send_message(embed=embed, ephemeral=True)
            return
        # The rebuild runs far past Discord's ack window, so it is dispatched to the
        # background task queue and the command replies immediately; the user checks
        # back with `/memory show`.
        scheduled = schedule_memory_regeneration(
            scope=scope,
            writer=writer,
            identity=render_author_identity(
                display_name=interaction.user.display_name,
                username=interaction.user.name,
                user_id=interaction.user.id,
            ),
        )
        if scheduled:
            description = "已排程重建記憶，整理完成後可以用 `/memory show` 查看。"
            color = DISCORD_GREEN
        else:
            description = "記憶正在重建中，完成後可以用 `/memory show` 查看。"
            color = DISCORD_YELLOW
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


class _CatchupHistory(BaseModel):
    """What one catchup read off its channel."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: SkipValidation[list[Message]] = Field(
        ..., description="Members' messages to review, oldest first."
    )
    since_checkpoint: bool = Field(
        ..., description="Whether reading stopped at an earlier catchup rather than the limit."
    )
    read_until: str = Field(
        ...,
        description=(
            "Link to the newest message read, which the result carries as the next run's stop."
        ),
    )


async def _read_since_checkpoint(
    interaction: Interaction[commands.Bot], channel: Messageable
) -> _CatchupHistory | None:
    """Reads members' messages since the last catchup, or None when the bot may not read.

    A result is posted only when its run ends, below whatever members said while it ran, so
    reading stops not at the result but at the message its title links to, the newest one
    that run read. Without read-history permission Discord answers an empty page rather than
    a 403, which would read as a channel nobody talked in, so the permission is checked first.
    """
    permissions = interaction.app_permissions
    if not (permissions.view_channel and permissions.read_message_history):
        logfire.warn(
            "Memory catchup has no permission to read the channel's history",
            guild_id=interaction.guild_id,
            channel_id=interaction.channel_id,
        )
        return None
    messages: list[Message] = []
    newest: Message | None = None
    boundary: int | None = None
    try:
        async for message in channel.history(limit=HISTORY_MESSAGE_LIMIT):
            newest = newest or message
            if boundary is None:
                boundary = _catchup_boundary(message=message, bot=interaction.client)
            if boundary is not None and message.id <= boundary:
                break
            if not message.author.bot and message.content.strip():
                messages.append(message)
    except Forbidden:
        logfire.warn(
            "Memory catchup was refused the channel's history",
            guild_id=interaction.guild_id,
            channel_id=interaction.channel_id,
        )
        return None
    messages.reverse()
    return _CatchupHistory(
        messages=messages,
        since_checkpoint=boundary is not None,
        read_until=newest.jump_url if newest is not None else "",
    )


def _catchup_boundary(message: Message, bot: commands.Bot) -> int | None:
    """The id of the newest message a finished catchup read, when `message` is its result.

    A result's title links to that message. One without a usable link stops reading at the
    result itself, which is the older behaviour's boundary rather than a guess.
    """
    if bot.user is None or message.author.id != bot.user.id:
        return None
    for embed in message.embeds:
        if (
            embed.title != _CATCHUP_TITLE
            or embed.colour is None
            or embed.colour.value not in _CATCHUP_CHECKPOINT_COLORS
        ):
            continue
        tail = (embed.url or "").rsplit("/", 1)[-1]
        return int(tail) if tail.isdigit() else message.id
    return None


def _catchup_input(message: Message) -> EasyInputMessageParam:
    """Renders one read message the way the reply path renders a member's text."""
    identity = render_author_identity(
        display_name=message.author.display_name,
        username=message.author.name,
        user_id=message.author.id,
    )
    return EasyInputMessageParam(role="user", content=f"{identity}: {message.content.strip()}")


def _catchup_result(
    intro: str, observations: tuple[MemoryObservation, ...] | None, invoker: str, read_until: str
) -> Embed:
    """Builds the public result that replaces a catchup's start message.

    Its title links to the newest message the run read, which is where the next run stops.
    """
    footer = f"由 {invoker} 發起"
    if observations is None:
        embed = Embed(
            title=_CATCHUP_TITLE,
            description=f"{intro}，但整理時出錯，這次沒有記下什麼。請稍後再試。",
            color=DISCORD_RED,
        )
    elif not observations:
        embed = Embed(
            title=_CATCHUP_TITLE,
            description=f"{intro}。你們聊這什麼屌 廢話一堆",
            color=DISCORD_YELLOW,
        )
    else:
        recorded = "\n".join(f"- {observation.summary_zh}" for observation in observations)
        embed = Embed(
            title=_CATCHUP_TITLE,
            description=clip_to_utf16_limit(
                text=f"{intro}，✏️ 已記入小本本：\n{recorded}",
                limit=DISCORD_EMBED_DESCRIPTION_LIMIT,
                notice="\n⋯ 太長了，其餘的用 `/memory server show` 查看",
            ),
            color=DISCORD_GREEN,
        )
        footer = f"{footer} · 記下的內容會在背景併入伺服器記憶，過程中可能被合併或捨棄"
    embed.url = read_until
    embed.set_footer(text=footer)
    return embed


async def _post_catchup_result(interaction: Interaction[commands.Bot], result: Embed) -> None:
    """Posts the result, then removes the start message, both through the interaction.

    In that order, so a failed delete leaves both messages up rather than neither.
    """
    try:
        await interaction.followup.send(embed=result)
    except HTTPException as exc:
        logfire.warn(
            "Could not post a memory catchup's result",
            guild_id=interaction.guild_id,
            channel_id=interaction.channel_id,
            _exc_info=exc,
        )
        return
    try:
        await interaction.delete_original_message()
    except HTTPException as exc:
        logfire.warn(
            "Could not remove a memory catchup's start message",
            guild_id=interaction.guild_id,
            channel_id=interaction.channel_id,
            _exc_info=exc,
        )


def setup(bot: commands.Bot) -> None:
    """Adds the MemoryCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(MemoryCogs(bot), override=True)
