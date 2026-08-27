"""Cog that routes Discord messages through the AI reply pipeline.

What is left here is what belongs to the PROCESS rather than to one turn: the gateway listeners,
the clients and the toolkit every reply composes itself from, the restart memory resume, and the
last-resort failure notice. One turn's own work lives in `pipeline.py` and the phase modules
beside it.

`/ask` is the second entry point into that same pipeline, and it exists because the first one
cannot reach three places a user-installed app can: a server the bot was never added to, a group
DM, and a DM between two other people. There is no gateway message event in any of them, so
mentioning the bot, replying to it and the link expansions all do nothing; the command hands the
conversation back, running the identical turn over a `TurnSurface` that answers through the
interaction instead of the channel.
"""

from typing import TYPE_CHECKING, Any, TypedDict
import asyncio
from functools import cached_property

from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import (
    Embed,
    Locale,
    Message,
    NotFound,
    Attachment,
    Interaction,
    SlashOption,
    HTTPException,
)
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.colors import DISCORD_RED
from discordbot.utils.mentions import has_bot_mention
from discordbot.utils.reactions import ReactionStatusChain, update_reaction
from discordbot.utils.usage_log import UsageRecorder
from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.utils.llm_errors import extract_friendly_error
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.media_delivery import MediaDeliveryPlanner, build_media_delivery_planner
from discordbot.services.memory.facts import render_owner_identity
from discordbot.services.memory.store import flavor_of, read_owner, iter_scopes
from discordbot.cogs.gen_reply.surface import TurnSurface
from discordbot.cogs.gen_reply.toolkit import ReplyToolkit
from discordbot.cogs.gen_reply.pipeline import ReplyPipeline
from discordbot.services.memory.pipeline import safe_list_resumable, resume_memory_update
from discordbot.cogs.gen_reply.turn_state import dispatched_model, current_answer_streamer
from discordbot.cogs.gen_reply.ask_message import build_ask_message, interaction_channel
from discordbot.services.memory.git_history import memory_git
from discordbot.services.memory.consolidation import needs_consolidation, consolidate_if_needed
from discordbot.cogs.gen_reply.research_bridge import in_active_research_thread

if TYPE_CHECKING:
    from collections.abc import Coroutine


class _MessageLogFields(TypedDict):
    """Exact key set for `_message_log_fields`, so `**`-spreading it into a logfire call
    keeps statically known keys (none underscore-prefixed) and never collides with logfire's
    `_tags` / `_exc_info` keyword-only parameters.
    """

    user_id: int
    user_name: str
    display_name: str
    message_id: int
    channel_id: int
    guild_id: int | None
    guild_name: str | None


def _message_log_fields(*, surface: TurnSurface) -> _MessageLogFields:
    """Standard Discord identifying fields for correlating one reply's logs.

    The pipeline-entry log carries the full set; every downstream log carries only
    `message_id` as the correlation key, so a whole turn reconstructs by grepping it.
    `user_name` is the stable handle, `display_name` the per-guild nickname;
    `guild_id` / `guild_name` are None in a DM.

    Read off the surface rather than the message so a `/ask` turn in a server is not logged as
    a DM. `guild_name` still comes from the message and so stays None there: the bot is not a
    member of that server and does not know what it is called.
    """
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
        bot: The Discord bot instance that owns this cog.
        config: The LLM client configuration loaded for reply generation.
        usage_recorder: The per-reply usage-record writer read by `scripts/usage_report.py`.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the ReplyGeneratorCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()
        self.usage_recorder = UsageRecorder()
        # Tracked background tasks for the one-shot restart memory resume.
        self._tasks: set[asyncio.Task[None]] = set()
        self._resume_started = False

    def _spawn(self, coro: "Coroutine[Any, Any, None]") -> None:
        """Runs `coro` as a tracked background task so the gateway never blocks on it."""
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @cached_property
    def openai_client(self) -> AsyncOpenAI:
        """The cached AsyncOpenAI client for all LiteLLM-proxy Responses / audio / image calls.

        Returns:
            A configured AsyncOpenAI client reused across reply requests.
        """
        return AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    @cached_property
    def toolkit(self) -> ReplyToolkit:
        """The clients, generators and caches every turn composes its reply from.

        Built on first use and kept for the life of the process, which is what the caches
        inside it are for: the input builder holds the Files API uris a message's attachments
        were uploaded to, so a message that stays in the history window is uploaded once
        rather than once per reply.

        Returns:
            The process-wide reply toolkit.
        """
        return ReplyToolkit(
            bot=self.bot,
            openai_client=self.openai_client,
            gemini_api_key=self.config.gemini_api_key,
        )

    @cached_property
    def media_delivery(self) -> MediaDeliveryPlanner:
        """The cached media-delivery planner shared by the IMAGE / VIDEO routes and QA streamer.

        Returns:
            A planner that decides which media attach natively and which are hosted as a public
            URL (media too big for Discord's upload limit); its host self-disables when
            unconfigured, so every oversize item then degrades to the route's host-free path.
        """
        return build_media_delivery_planner()

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Resumes persisted memory work after a restart (runs once).

        `on_ready` fires on every gateway reconnect, so `_resume_started` guards it
        to a single sweep per process. The sweep is spawned, never awaited, so the
        gateway is not blocked while it digests in the background.
        """
        if self._resume_started:
            return
        # Bound to this loop, so it starts here rather than at import: an unstarted
        # service drops every commit request instead of binding its queue to whichever
        # loop happened to enqueue first.
        memory_git.start()
        self._resume_started = True
        self._spawn(self._resume_memory())

    async def _resume_memory(self) -> None:
        """Re-enqueues persisted review jobs and consolidates over-threshold scopes.

        Two paths, both riding the existing per-scope lock + global concurrency
        semaphore: persisted `pending`/`failed` jobs are re-run (transcript intact),
        and every scope whose raw backlog is over threshold is swept. The sweep
        covers scopes with a resumed job too: the per-scope lock plus the under-lock
        `_should_consolidate` re-check make the resumed review and the sweep
        idempotent, so a consolidation interrupted by the restart still finishes
        even when the resumed review early-returns (failed, no signal, or all
        duplicates) before it would reach the consolidation check.
        """
        jobs = await safe_list_resumable()
        for job in jobs:
            if job.transcript is None:
                continue
            writer = (
                self.toolkit.server_memory_writer
                if job.flavor == "server"
                else self.toolkit.memory_writer
            )
            resume_memory_update(
                scope=job.scope,
                subject=job.subject,
                transcript=job.transcript,
                writer=writer,
                identity=job.identity,
                token=job.token,
            )
        if jobs:
            logfire.info("resumed persisted memory jobs", count=len(jobs))
        swept = 0
        for scope in iter_scopes():
            if not needs_consolidation(scope=scope):
                continue
            writer = (
                self.toolkit.server_memory_writer
                if flavor_of(scope=scope) == "server"
                else self.toolkit.memory_writer
            )
            self._spawn(
                consolidate_if_needed(
                    scope=scope,
                    writer=writer,
                    identity=render_owner_identity(owner=read_owner(scope=scope)),
                )
            )
            swept += 1
        if swept:
            logfire.info("scheduled memory consolidation sweep", count=swept)

    async def _deliver_failure_notice(self, *, surface: TurnSurface, error_embed: Embed) -> None:
        """Shows the turn's failure, on the reply it was streaming into where there is one.

        Half the turns that fail here already painted something (23 of 46 in one 2026-08-21 log),
        and left beside that reply the embed reads as unrelated while the reply itself, carrying
        no usage footer, reads as an answer that merely stopped. So the streamer is asked first
        and takes the error onto its own message. Everything it turns down -- every failure
        before the answer, and a retry notice already withdrawn -- gets a fresh message here.

        Through the surface rather than `message.reply`, because on the `/ask` route a failure
        before the first content delta is the whole of what the user ever sees: there is no
        channel to post into, and Discord would otherwise leave the deferred response hanging
        until it expired with nothing in it.
        """
        streamer = current_answer_streamer.get()
        if streamer is not None and await streamer.land_failure(embed=error_embed):
            return
        message = surface.message
        spacer = embed_spacer_payload(embeds=[error_embed], is_edit=False, target=message)
        try:
            await surface.send(embed=error_embed, **spacer)
        except HTTPException as send_error:
            # Source deleted before the error landed (50035): send it unparented. Rebuild
            # the spacer; the failed reply already consumed the single-use spacer file.
            if send_error.code != 50035 and not isinstance(send_error, NotFound):
                raise
            fresh_spacer = embed_spacer_payload(
                embeds=[error_embed], is_edit=False, target=message
            )
            await surface.send_unparented(embed=error_embed, **fresh_spacer)

    async def _run_turn(self, *, surface: TurnSurface, user_prompt: str) -> None:
        """Runs one turn and reports whatever it failed on, whichever entry point started it.

        Shared by `on_message` and `/ask` because everything that differs between them is
        already inside the surface: where the answer lands, what history it may read, and
        whether the source message can be reacted to at all.
        """
        message = surface.message
        reactions = ReactionStatusChain(
            message=message, bot_user=self.bot.user, enabled=surface.interaction is None
        )
        try:
            await ReplyPipeline(
                client=self.openai_client,
                bot=self.bot,
                config=self.config,
                media_delivery=self.media_delivery,
                usage_recorder=self.usage_recorder,
                toolkit=self.toolkit,
                message=message,
                surface=surface,
                user_prompt=user_prompt,
                reactions=reactions,
            ).run()
        except Exception as e:
            logfire.error(
                "gen_reply failed",
                **_message_log_fields(surface=surface),
                model=dispatched_model.get(),
                error_type=type(e).__name__,
                _exc_info=True,
            )
            try:
                reactions.advance(emoji="<:redcross:1517565100838355016>")
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
            await reactions.flush()

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

        The response is deferred first and always: Discord invalidates the token after three
        seconds, and every phase of the turn is slower than that. It is deliberately not
        ephemeral — in a group DM the answer is the conversation, and the only place the choice
        would matter is one where Discord may force ephemeral on us regardless.

        Args:
            interaction: The invocation to answer through.
            question: What the user wants to say.
            attachment: An optional file to send with it; the only way to give the bot one here,
                since there is no message of theirs for it to hang off.
        """
        await interaction.response.defer()
        channel = interaction_channel(interaction=interaction)
        message = build_ask_message(interaction=interaction, question=question, channel=channel)
        surface = TurnSurface.for_interaction(message=message, interaction=interaction)
        user_prompt = await self.toolkit.input_builder.get_user_prompt(content=question)
        if not user_prompt and attachment is None:
            logfire.debug(
                "gen_reply empty prompt; replied with ?", **_message_log_fields(surface=surface)
            )
            await surface.send(content="?")
            return
        logfire.info(
            "gen_reply received",
            **_message_log_fields(surface=surface),
            prompt_chars=len(user_prompt),
            has_attachment=attachment is not None,
            attachment_count=len(message.attachments),
            sticker_count=0,
            is_dm=surface.guild_id is None,
        )
        await self._run_turn(surface=surface, user_prompt=user_prompt)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Listens for messages and handles AI reply generation.

        Args:
            message: The message that was sent.
        """
        # Ignore messages from bots.
        if message.author.bot:
            return

        # Match <@ID> in content, not message.mentions: reply notifications add
        # the bot to mentions and would trigger on replies to functional bot
        # posts (e.g. Threads embeds, video downloads).
        is_dm = message.guild is None
        if not is_dm and not has_bot_mention(content=message.content, bot_user=self.bot.user):
            return

        # Skip a (mentioned) message typed inside a research thread the ResearchCogs cog is
        # actively driving: the thread is its workspace until the report lands, so QA must not
        # answer over the live status edits. The skip lifts the moment the run finishes.
        surface = TurnSurface.for_message(message=message)
        if in_active_research_thread(bot=self.bot, channel_id=message.channel.id):
            logfire.debug(
                "gen_reply skipped: the research cog is still writing into this thread",
                **_message_log_fields(surface=surface),
            )
            return

        user_prompt = await self.toolkit.input_builder.get_user_prompt(content=message.content)
        has_attachment = bool(message.attachments or message.stickers)
        # A forward leaves content/attachments/stickers empty and puts the payload in
        # `message.snapshots`, so it must not be gated out as an empty message here, or the
        # snapshot text/media render in `input.py` never runs.
        is_forward = bool(message.snapshots)
        # A forward puts its request in `message.snapshots`, not content, so merge the forwarded
        # text into the prompt (after the forwarder's own comment, if any). A guild forward can
        # only trigger via a `<@bot>` comment, so the comment is usually non-empty: merging (not
        # just an empty fallback) is what lets an IMAGE/VIDEO route render the forwarded "draw a
        # cat" even when the trigger comment ("@bot please") survives mention-stripping.
        if is_forward and (
            forwarded := self.toolkit.input_builder.forwarded_request_text(message=message)
        ):
            user_prompt = f"{user_prompt}\n{forwarded}".strip() if user_prompt else forwarded

        if not user_prompt and not has_attachment and not is_forward:
            logfire.debug(
                "gen_reply empty prompt; replied with ?", **_message_log_fields(surface=surface)
            )
            await update_reaction(message=message, bot_user=self.bot.user, emoji="❓")
            await message.reply(content="?")
            return

        logfire.info(
            "gen_reply received",
            **_message_log_fields(surface=surface),
            prompt_chars=len(user_prompt),
            has_attachment=has_attachment,
            attachment_count=len(message.attachments),
            sticker_count=len(message.stickers),
            is_dm=is_dm,
        )
        await self._run_turn(surface=surface, user_prompt=user_prompt)


def setup(bot: commands.Bot) -> None:
    """Adds the ReplyGeneratorCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
