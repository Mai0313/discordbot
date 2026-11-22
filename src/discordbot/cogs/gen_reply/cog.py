"""Cog that routes Discord messages and `/ask` invocations through the AI reply pipeline.

Holds process-wide state (gateway listeners, cached clients and toolkit, restart memory resume,
and last-resort failure reporting), while per-turn execution lives in `pipeline.py`. `/ask`
enters the same pipeline through an interaction-backed `TurnSurface` where gateway message events
never arrive (unjoined servers, group DMs, and other users' DMs).
"""

from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, cast, runtime_checkable
from functools import cached_property
import contextlib

from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import (
    File,
    User,
    Embed,
    Locale,
    Message,
    NotFound,
    Forbidden,
    Attachment,
    Interaction,
    SlashOption,
    HTTPException,
    PartialMessageable,
)
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.colors import DISCORD_RED
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.timezone import database_now
from discordbot.utils.reactions import ReactionStatusChain
from discordbot.utils.usage_log import UsageRecorder
from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.utils.llm_errors import extract_friendly_error
from discordbot.utils.asyncio_locks import KeyedLockManager, spawn_tracked
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.discord_errors import is_reply_target_gone
from discordbot.utils.media_delivery import MediaDeliveryPlanner, build_media_delivery_planner
from discordbot.cogs.gen_reply.answer import RegenOptions, max_regens_for
from discordbot.services.memory.facts import render_owner_identity
from discordbot.services.memory.store import read_owner, iter_scopes
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.pipeline import ReplyPipeline
from discordbot.services.memory.pipeline import safe_list_resumable, resume_memory_update
from discordbot.cogs.gen_reply.turn_state import dispatched_model, current_answer_streamer
from discordbot.cogs.gen_reply.status_marks import FAILED_EMOJI, EMPTY_PROMPT_EMOJI
from discordbot.services.memory.git_history import memory_git
from discordbot.cogs.gen_reply.feedback_store import (
    RegenChainState,
    load_reply_turn,
    record_regen_feedback,
    load_regen_chain_state,
    parse_reply_message_ids,
)
from discordbot.cogs.gen_reply.feedback_views import ReplyFeedbackView
from discordbot.services.memory.consolidation import needs_consolidation, consolidate_if_needed
from discordbot.cogs.gen_reply.research_bridge import in_active_research_thread

if TYPE_CHECKING:
    import asyncio


@runtime_checkable
class _MessageFetchable(Protocol):
    """Minimal channel surface that can fetch a message by id."""

    async def fetch_message(self, message_id: int, /) -> Message:
        """Fetches one Discord message by id."""
        ...


@runtime_checkable
class _DeletableMessage(Protocol):
    """Minimal message handle that can be deleted."""

    async def delete(self) -> None:
        """Deletes the message."""
        ...


@runtime_checkable
class _PartialMessageChannel(Protocol):
    """Channel surface exposing `get_partial_message` for one-call deletion."""

    def get_partial_message(self, message_id: int, /) -> _DeletableMessage:
        """Returns a partial message handle without an extra REST fetch."""
        ...


class _MessageLogFields(TypedDict):
    """Statically typed non-underscored log fields for `**`-spreading into `logfire` calls."""

    user_id: int
    user_name: str
    display_name: str
    message_id: int
    channel_id: int
    guild_id: int | None
    guild_name: str | None


def _message_log_fields(surface: TurnSurface) -> _MessageLogFields:
    """Returns the standard Discord identifying fields for correlating one reply's logs."""
    message = surface.message
    guild = message.guild
    return {
        "user_id": message.author.id,
        "user_name": message.author.name,
        "display_name": message.author.display_name,
        "message_id": message.id,
        "channel_id": message.channel.id,
        "guild_id": surface.guild_id,
        "guild_name": guild.name if guild else None,
    }


async def _delete_dangling_reply() -> None:
    """Best-effort deletes any partial reply chunks left behind by a failed regen stream."""
    streamer = current_answer_streamer.get()
    if streamer is None:
        return
    dangling = list(streamer.reply_messages)
    if not dangling and streamer.reply is not None:
        dangling = [streamer.reply]
    for chunk in reversed(dangling):
        with contextlib.suppress(HTTPException):
            await chunk.delete()


class ReplyGeneratorCogs(commands.Cog):
    """Generates AI replies for Discord messages.

    Attributes:
        bot (commands.Bot): The Discord bot instance that owns this cog.
        config (LLMConfig): The LLM client configuration loaded for reply generation.
        usage_recorder (UsageRecorder): The per-reply usage-record writer.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the ReplyGeneratorCogs instance.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()
        self.usage_recorder = UsageRecorder()
        # Tracked background tasks for the one-shot restart memory resume.
        self._tasks: set[asyncio.Task[None]] = set()
        self._regen_locks: KeyedLockManager[int] = KeyedLockManager()
        self._resume_started = False
        # Taken at load, before the gateway connects, as `DiscordBot._started_at` is: the restart
        # sweep must not re-run a review this process stages ahead of `on_ready`.
        self._started_at = database_now()

    @cached_property
    def openai_client(self) -> AsyncOpenAI:
        """Returns the cached `AsyncOpenAI` client for LiteLLM-proxy calls."""
        return AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    @cached_property
    def toolkit(self) -> ReplyToolkit:
        """Returns the process-wide `ReplyToolkit` shared across turns."""
        return ReplyToolkit(
            bot=self.bot,
            openai_client=self.openai_client,
            gemini_api_key=self.config.gemini_api_key,
        )

    @cached_property
    def media_delivery(self) -> MediaDeliveryPlanner:
        """Returns the process-wide `MediaDeliveryPlanner` for attach-vs-host routing."""
        return build_media_delivery_planner()

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Registers persistent feedback views and spawns one-shot post-restart memory resume."""
        if self._resume_started:
            return
        # Register once regardless of `feedback_enabled` so buttons on earlier replies keep working.
        self.bot.add_view(ReplyFeedbackView(allow_regen=True, on_regen=self._handle_regen))
        # Start on the running loop; an unstarted service drops commits instead of binding early.
        memory_git.start()
        self._resume_started = True
        spawn_tracked(coro=self._resume_memory(), tasks=self._tasks, name="memory-resume")

    async def _resume_memory(self) -> None:
        """Re-enqueues persisted review jobs and consolidates over-threshold scopes."""
        jobs = await safe_list_resumable(updated_before=self._started_at)
        for job in jobs:
            if job.transcript is None:
                continue
            resume_memory_update(
                scope=job.scope,
                subject=job.subject,
                transcript=job.transcript,
                writer=self.toolkit.memory_writer,
                identity=job.identity,
                token=job.token,
                status=job.status,
            )
        if jobs:
            logfire.info("resumed persisted memory jobs", count=len(jobs))
        swept = 0
        for scope in iter_scopes():
            if not needs_consolidation(scope=scope):
                continue
            spawn_tracked(
                coro=consolidate_if_needed(
                    scope=scope,
                    writer=self.toolkit.memory_writer,
                    identity=render_owner_identity(owner=read_owner(scope=scope)),
                ),
                tasks=self._tasks,
                name=f"memory-consolidation-{scope}",
            )
            swept += 1
        if swept:
            logfire.info("scheduled memory consolidation sweep", count=swept)

    async def _deliver_failure_notice(self, surface: TurnSurface, error_embed: Embed) -> None:
        """Delivers `error_embed` onto the in-progress answer message or as a fresh surface reply."""
        streamer = current_answer_streamer.get()
        if streamer is not None and await streamer.land_failure(embed=error_embed):
            return
        message = surface.message

        def _spacer_files() -> list[File] | None:
            return embed_spacer_payload(embeds=[error_embed], is_edit=False, target=message).get(
                "files"
            )

        await surface.send_or_unparented(
            embed=error_embed,
            files=_spacer_files,
            unparented_log="Source message deleted before failure notice; sending unparented",
        )

    async def _run_turn(self, surface: TurnSurface, content: str) -> None:
        """Runs one turn and delivers a last-resort failure notice if an unhandled error escapes."""
        message = surface.message
        reactions = surface.status_chain(bot_user=self.bot.user)
        dispatched_model.set(None)
        current_answer_streamer.set(None)
        try:
            await self._start_turn(surface=surface, content=content, reactions=reactions)
        except Exception as e:
            logfire.error(
                "gen_reply failed",
                **_message_log_fields(surface=surface),
                model=dispatched_model.get(),
                error_type=type(e).__name__,
                _exc_info=True,
            )
            try:
                reactions.advance(emoji=FAILED_EMOJI)
                error_embed = Embed(
                    title="Something went wrong",
                    description=f"```\n{extract_friendly_error(exc=e)}\n```",
                    color=DISCORD_RED,
                )
                error_embed.set_footer(text=type(e).__name__)
                await self._deliver_failure_notice(surface=surface, error_embed=error_embed)
            except Exception as report_error:
                # Broad on purpose: this is the last-resort user notice; nothing above it can
                # recover, and it must not displace the original failure.
                logfire.warn(
                    "failed to deliver the pipeline failure notice",
                    message_id=message.id,
                    error_type=type(report_error).__name__,
                    _exc_info=report_error,
                )
        finally:
            dispatched_model.set(None)
            current_answer_streamer.set(None)
            await reactions.flush()

    async def _start_turn(
        self,
        surface: TurnSurface,
        content: str,
        reactions: ReactionStatusChain,
        regen: RegenOptions | None = None,
    ) -> int | None:
        """Dispatches `ReplyPipeline` on the turn's prompt, or replies with `?` when empty."""
        message = surface.message
        user_prompt = self.toolkit.input_builder.get_user_prompt(content=content)
        has_attachment = bool(message.attachments or message.stickers)
        is_forward = bool(message.snapshots)
        # Merge rather than fallback-only: guild forwards carry a `<@bot>` trigger comment that
        # may survive mention-stripping, and media routes still need the forwarded request text.
        if is_forward and (
            forwarded := self.toolkit.input_builder.forwarded_request_text(message=message)
        ):
            user_prompt = f"{user_prompt}\n{forwarded}".strip() if user_prompt else forwarded

        if not user_prompt and not has_attachment and not is_forward:
            if regen is not None:
                return None
            logfire.debug(
                "gen_reply empty prompt; replied with ?", **_message_log_fields(surface=surface)
            )
            await surface.mark(emoji=EMPTY_PROMPT_EMOJI)
            try:
                await surface.send(content="?")
            except HTTPException as exc:
                # A `?` means nothing without its message, so it is not sent unparented.
                if not is_reply_target_gone(error=exc):
                    raise
                logfire.info("gen_reply empty prompt's message is gone", message_id=message.id)
            return None

        logfire.info(
            "gen_reply received",
            **_message_log_fields(surface=surface),
            prompt_chars=len(user_prompt),
            has_attachment=has_attachment,
            attachment_count=len(message.attachments),
            sticker_count=len(message.stickers),
            is_dm=surface.guild_id is None,
        )
        return await ReplyPipeline(
            config=self.config,
            media_delivery=self.media_delivery,
            usage_recorder=self.usage_recorder,
            toolkit=self.toolkit,
            surface=surface,
            user_prompt=user_prompt,
            reactions=reactions,
            regen=regen,
        ).run()

    async def _resolve_regen_channel(
        self, interaction: Interaction[commands.Bot], channel_id: int, reply_message_id: int
    ) -> _MessageFetchable | None:
        """Returns the fetchable channel for `channel_id`, or edits the ephemeral notice and returns None."""
        channel: object | None = interaction.channel
        if channel is None or isinstance(channel, PartialMessageable):
            channel = self.bot.get_channel(channel_id)
        if channel is None or isinstance(channel, PartialMessageable):
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except Forbidden:
                logfire.warn(
                    "Channel refused lookup for reply regen",
                    channel_id=channel_id,
                    reply_message_id=reply_message_id,
                )
                await interaction.edit_original_message(content="沒有權限讀取原頻道，無法重新生成")
                return None
            except NotFound:
                await interaction.edit_original_message(content="找不到原頻道，無法重新生成")
                return None
            except HTTPException as exc:
                logfire.warn(
                    "Failed to fetch channel for reply regen",
                    channel_id=channel_id,
                    reply_message_id=reply_message_id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
                await interaction.edit_original_message(content="讀取原頻道時發生錯誤，請稍後再試")
                return None
        if not isinstance(channel, _MessageFetchable):
            await interaction.edit_original_message(content="這個頻道不支援重新生成")
            return None
        return channel

    async def _fetch_regen_source(
        self,
        interaction: Interaction[commands.Bot],
        channel: _MessageFetchable,
        state: RegenChainState,
    ) -> Message | None:
        """Fetches the original asking message for regen, reporting failures ephemerally."""
        try:
            return await channel.fetch_message(state.source_message_id)
        except NotFound:
            await interaction.edit_original_message(content="找不到原始發問訊息，無法重新生成")
            return None
        except Forbidden:
            logfire.warn(
                "Channel refused source message fetch for reply regen",
                channel_id=state.channel_id,
                message_id=state.source_message_id,
                reply_message_id=state.reply_message_id,
            )
            await interaction.edit_original_message(content="沒有權限讀取原始訊息，無法重新生成")
            return None
        except HTTPException as exc:
            logfire.warn(
                "Failed to fetch source message for reply regen",
                channel_id=state.channel_id,
                message_id=state.source_message_id,
                reply_message_id=state.reply_message_id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await interaction.edit_original_message(content="讀取原始訊息時發生錯誤，請稍後再試")
            return None

    async def _delete_old_reply_chunks(
        self, channel: _MessageFetchable, state: RegenChainState
    ) -> None:
        """Deletes old reply chunk messages in reverse order after a replacement lands."""
        old_ids = parse_reply_message_ids(
            raw=state.reply_message_ids, fallback_id=state.reply_message_id
        )
        for old_id in reversed(old_ids):
            try:
                if isinstance(channel, _PartialMessageChannel):
                    await channel.get_partial_message(old_id).delete()
                else:
                    old_message = await channel.fetch_message(old_id)
                    await old_message.delete()
            except NotFound:
                continue
            except Forbidden:
                logfire.warn(
                    "Channel refused old reply deletion after regen",
                    channel_id=state.channel_id,
                    message_id=old_id,
                    reply_message_id=state.reply_message_id,
                )
                break
            except HTTPException as exc:
                logfire.warn(
                    "Failed to delete old reply chunk after regen",
                    channel_id=state.channel_id,
                    message_id=old_id,
                    reply_message_id=state.reply_message_id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )

    async def _execute_regen_turn(
        self,
        interaction: Interaction[commands.Bot],
        source_message: Message,
        state: RegenChainState,
        max_regens: int,
    ) -> int | None:
        """Runs the replacement QA turn, cleaning up dangling previews on failure."""
        remaining = max_regens - (state.chain_depth + 1)
        if self.config.slow_model_blind_test_enabled:
            progress = (
                f"正在換另一個模型重新生成中⋯（本次後剩餘 {remaining} 次重抽）"
                if remaining > 0
                else "正在換另一個模型重新生成中⋯（最後一次重抽）"
            )
        else:
            progress = "正在重新生成回覆中⋯"
        with contextlib.suppress(HTTPException):
            await interaction.edit_original_message(content=progress)

        surface = TurnSurface.for_message(message=source_message)
        reactions = surface.status_chain(bot_user=self.bot.user)
        forced_effort: Literal["low", "medium", "high"] | None = (
            cast("Literal['low', 'medium', 'high']", state.effort)
            if state.effort in ("low", "medium", "high")
            else None
        )
        regen = RegenOptions(
            regen_of_reply_message_id=state.reply_message_id,
            regen_count=state.chain_depth + 1,
            excluded_models=frozenset(state.tried_models),
            forced_effort=forced_effort,
        )
        dispatched_model.set(None)
        current_answer_streamer.set(None)
        try:
            new_reply_message_id = await self._start_turn(
                surface=surface, content=source_message.content, reactions=reactions, regen=regen
            )
        except Exception as exc:
            logfire.error(
                "gen_reply regen failed",
                **_message_log_fields(surface=surface),
                reply_message_id=state.reply_message_id,
                model=dispatched_model.get(),
                error_type=type(exc).__name__,
                _exc_info=True,
            )
            await _delete_dangling_reply()
            reactions.advance(emoji=FAILED_EMOJI)
            with contextlib.suppress(HTTPException):
                await interaction.edit_original_message(
                    content=f"重新生成失敗：{extract_friendly_error(exc=exc)}"
                )
            return None
        finally:
            dispatched_model.set(None)
            current_answer_streamer.set(None)
            await reactions.flush()
        if new_reply_message_id is None:
            with contextlib.suppress(HTTPException):
                await interaction.edit_original_message(content="原始訊息內容為空，無法重新生成")
            return None
        return new_reply_message_id

    async def _validate_regen_access(
        self, interaction: Interaction[commands.Bot], reply_message_id: int
    ) -> str | None:
        """Returns a user-facing refusal reason before locking, or None when allowed."""
        if not self.config.feedback_enabled:
            return "目前已停用重新生成功能"
        turn = await load_reply_turn(reply_message_id=reply_message_id)
        if turn is None:
            return "這則回覆已超過可重新生成範圍"
        if turn.surface != "gateway" or turn.route != "QA":
            return "這則回覆不支援重新生成"
        user = interaction.user
        if user is None:
            return "只有原發問者可以重新生成這則回覆"
        is_allowed = user.id == turn.author_id or await self.bot.is_owner(user=cast("User", user))
        if not is_allowed:
            return "只有原發問者可以重新生成這則回覆"
        return None

    async def _run_locked_regen(
        self, interaction: Interaction[commands.Bot], reply_message_id: int
    ) -> None:
        """Re-checks regen state under the per-reply lock, runs replacement QA, and cleans old chunks."""
        user = interaction.user
        if user is None:
            return
        state = await load_regen_chain_state(reply_message_id=reply_message_id)
        max_regens = max_regens_for(
            config=self.config,
            candidates=self.toolkit.runtime_models.slow_model_blind_test_candidates,
        )
        refusal = (
            "這則回覆已超過可重新生成範圍"
            if state is None
            else "這則回覆已經重新生成過了"
            if state.already_regenerated
            else "已達重新生成次數上限"
            if state.chain_depth >= max_regens
            else None
        )
        if state is None or refusal is not None:
            await interaction.edit_original_message(
                content=refusal or "這則回覆已超過可重新生成範圍"
            )
            return

        channel = await self._resolve_regen_channel(
            interaction=interaction, channel_id=state.channel_id, reply_message_id=reply_message_id
        )
        if channel is None:
            return
        source_message = await self._fetch_regen_source(
            interaction=interaction, channel=channel, state=state
        )
        if source_message is None:
            return

        new_reply_message_id = await self._execute_regen_turn(
            interaction=interaction,
            source_message=source_message,
            state=state,
            max_regens=max_regens,
        )
        if new_reply_message_id is None:
            return

        try:
            await record_regen_feedback(
                reply_message_id=reply_message_id,
                user_id=user.id,
                user_name=user.name,
                user_display_name=user.display_name,
            )
        except Exception as exc:
            # Broad on purpose: best-effort implicit feedback must not block old cleanup.
            logfire.warn(
                "Failed to record implicit regen feedback",
                reply_message_id=reply_message_id,
                new_reply_message_id=new_reply_message_id,
                user_id=user.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        await self._delete_old_reply_chunks(channel=channel, state=state)
        with contextlib.suppress(HTTPException):
            await interaction.edit_original_message(content="已重新生成回覆")

    async def _handle_regen(self, interaction: Interaction[commands.Bot]) -> None:
        """Validates permissions/limits, regenerates a QA reply, and removes the old reply after delivery."""
        if interaction.message is None or interaction.user is None:
            return
        reply_message_id = interaction.message.id
        await interaction.response.defer(ephemeral=True, with_message=True)
        try:
            refusal = await self._validate_regen_access(
                interaction=interaction, reply_message_id=reply_message_id
            )
            if refusal is not None:
                await interaction.edit_original_message(content=refusal)
                return
            async with self._regen_locks.hold(key=reply_message_id):
                await self._run_locked_regen(
                    interaction=interaction, reply_message_id=reply_message_id
                )
        except Exception as exc:
            # Broad on purpose: deferred button callbacks must finish with a notice on failure.
            logfire.error(
                "Failed to handle reply regen",
                reply_message_id=reply_message_id,
                user_id=interaction.user.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            with contextlib.suppress(HTTPException):
                await interaction.edit_original_message(content="重新生成時發生錯誤，請稍後再試")

    @nextcord.slash_command(
        name="ask",
        description="Talk to me anywhere, even where I cannot see the conversation.",
        name_localizations={Locale.zh_TW: "問", Locale.ja: "質問"},
        description_localizations={
            Locale.zh_TW: "在我讀不到對話的地方跟我說話,例如我沒被加進去的伺服器或群組私訊",
            Locale.ja: "会話を読めない場所でも話しかけられます（未参加のサーバーやグループDMなど）。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def ask(
        self,
        interaction: Interaction[commands.Bot],
        question: str = SlashOption(
            name="question",
            description="What you want to say or ask.",
            name_localizations={Locale.zh_TW: "訊息", Locale.ja: "メッセージ"},
            description_localizations={
                Locale.zh_TW: "想跟我說或問的事",
                Locale.ja: "話しかけたい内容や質問。",
            },
            required=True,
            min_length=1,
        ),
        attachment: Attachment | None = SlashOption(
            name="attachment",
            description="An image, video, audio clip or document to send with it.",
            name_localizations={Locale.zh_TW: "附件", Locale.ja: "添付"},
            description_localizations={
                Locale.zh_TW: "要一起傳給我的圖片、影片、音檔或文件",
                Locale.ja: "一緒に送る画像・動画・音声・ファイル。",
            },
            required=False,
        ),
    ) -> None:
        """Answers one message through the interaction, where no gateway event ever arrives.

        Defers non-ephemerally up front so the interaction token stays valid across the turn and
        group-DM replies remain visible to the conversation.

        Args:
            interaction (Interaction[commands.Bot]): The invocation to answer through.
            question (str): What the user wants to say.
            attachment (Attachment | None): Optional file attached to the invocation.
        """
        await interaction.response.defer()
        surface = TurnSurface.for_interaction(interaction=interaction, question=question)
        await self._run_turn(surface=surface, content=question)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and handles AI reply generation.

        Args:
            message (Message): The message that was sent.
        """
        if message.author.bot:
            return

        if not is_addressed_to_bot(message=message, bot_user=self.bot.user):
            return

        # Yield active research threads to `ResearchCogs` until the report lands.
        surface = TurnSurface.for_message(message=message)
        if in_active_research_thread(bot=self.bot, channel_id=message.channel.id):
            logfire.debug(
                "gen_reply skipped: the research cog is still writing into this thread",
                **_message_log_fields(surface=surface),
            )
            return

        await self._run_turn(surface=surface, content=message.content)


def setup(bot: commands.Bot) -> None:
    """Adds the ReplyGeneratorCogs to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    bot.add_cog(cog=ReplyGeneratorCogs(bot=bot), override=True)
