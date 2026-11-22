"""Cog that routes Discord messages and `/ask` invocations through the AI reply pipeline.

Holds process-wide state (gateway listeners, cached clients and toolkit, restart memory resume,
and last-resort failure reporting), while per-turn execution lives in `pipeline.py`. `/ask`
enters the same pipeline through an interaction-backed `TurnSurface` where gateway message events
never arrive (unjoined servers, group DMs, and other users' DMs).
"""

from typing import TypedDict
import asyncio
from functools import cached_property

from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import (
    File,
    Embed,
    Locale,
    Message,
    Attachment,
    Interaction,
    SlashOption,
    HTTPException,
)
from nextcord.ext import tasks, commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.colors import DISCORD_RED
from discordbot.utils.mentions import is_addressed_to_bot
from discordbot.utils.timezone import database_now
from discordbot.utils.reactions import ReactionStatusChain
from discordbot.utils.usage_log import UsageRecorder
from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.utils.llm_errors import extract_friendly_error
from discordbot.utils.asyncio_locks import spawn_tracked
from discordbot.utils.model_pricing import MODEL_INFO_REFRESH_MINUTES, refresh_model_info
from discordbot.utils.sqlite_config import ensure_database_upgraded
from discordbot.cogs.gen_reply.regen import ReplyRegenerator
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.discord_errors import is_reply_target_gone
from discordbot.utils.media_delivery import MediaDeliveryPlanner, build_media_delivery_planner
from discordbot.cogs.gen_reply.answer import RegenOptions
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.pipeline import ReplyPipeline
from discordbot.services.memory.pipeline import resume_after_restart
from discordbot.cogs.gen_reply.turn_state import dispatched_model, current_answer_streamer
from discordbot.cogs.gen_reply.status_marks import FAILED_EMOJI, EMPTY_PROMPT_EMOJI
from discordbot.services.memory.git_history import memory_git
from discordbot.cogs.gen_reply.feedback_views import ReplyFeedbackView
from discordbot.cogs.gen_reply.research_bridge import in_active_research_thread


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
        return ReplyToolkit(bot=self.bot, openai_client=self.openai_client, config=self.config)

    @cached_property
    def media_delivery(self) -> MediaDeliveryPlanner:
        """Returns the process-wide `MediaDeliveryPlanner` for attach-vs-host routing."""
        return build_media_delivery_planner()

    @cached_property
    def regenerator(self) -> ReplyRegenerator:
        """Returns the `ReplyRegenerator` handling `🔄` feedback-button clicks."""
        return ReplyRegenerator(
            bot=self.bot,
            config=self.config,
            blind_test_candidates=(
                lambda: self.toolkit.runtime_models.slow_model_blind_test_candidates
            ),
            start_turn=self._start_turn,
        )

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Registers persistent feedback views, starts price-table refreshes, and resumes memory."""
        if self._resume_started:
            return
        # Register once regardless of `feedback_enabled` so buttons on earlier replies keep working.
        self.bot.add_view(ReplyFeedbackView(allow_regen=True, on_regen=self.regenerator.handle))
        memory_git.start()
        if not self.price_table_task.is_running():
            self.price_table_task.start()
        self._resume_started = True
        spawn_tracked(
            coro=resume_after_restart(
                writer=lambda: self.toolkit.memory_writer,
                tasks=self._tasks,
                updated_before=self._started_at,
            ),
            tasks=self._tasks,
            name="memory-resume",
        )

    def cog_unload(self) -> None:
        """Stops the price-table refresh loop and stops git history if no memory cog remains."""
        self.price_table_task.cancel()
        get_cog = getattr(self.bot, "get_cog", None)
        if memory_git.is_running and (get_cog is None or get_cog("MemoryCogs") is None):
            spawn_tracked(coro=memory_git.stop(), tasks=self._tasks, name="memory-git-stop")

    @tasks.loop(minutes=MODEL_INFO_REFRESH_MINUTES)
    async def price_table_task(self) -> None:
        """Refreshes the LiteLLM model price table off the event loop.

        Runs immediately on start to warm the cache, retries while degraded, and refreshes daily
        once upstream has succeeded.
        """
        try:
            await asyncio.to_thread(refresh_model_info)
        except Exception as exc:
            # Broad on purpose: an uncaught exception stops the tasks.loop permanently, and
            # nextcord's bound `_error` crashes silently on class methods (`_injected` arg).
            logfire.error(
                "model price table refresh failed; retrying next pass",
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

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
    """Upgrades `reply.db` and adds `ReplyGeneratorCogs` to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    ensure_database_upgraded(section="reply")
    bot.add_cog(cog=ReplyGeneratorCogs(bot=bot), override=True)
