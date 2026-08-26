"""Cog that routes Discord messages through the AI reply pipeline.

What is left here is what belongs to the PROCESS rather than to one turn: the gateway listeners,
the clients and per-key toolkits every reply leases, the restart memory resume, and the last-resort
failure notice. One turn's own work lives in `pipeline.py` and the phase modules beside it.
"""

from typing import TYPE_CHECKING, Any, TypedDict
import asyncio
from functools import cached_property

from openai import AsyncOpenAI
import logfire
from nextcord import Embed, Message, NotFound, HTTPException
from nextcord.ext import commands

from discordbot.typings.llm import LLMConfig
from discordbot.typings.colors import DISCORD_RED
from discordbot.utils.mentions import has_bot_mention
from discordbot.utils.reactions import ReactionStatusChain, update_reaction
from discordbot.utils.usage_log import UsageRecorder
from discordbot.utils.llm_errors import extract_friendly_error
from discordbot.utils.discord_embeds import embed_spacer_payload
from discordbot.utils.media_delivery import MediaDeliveryPlanner, build_media_delivery_planner
from discordbot.services.memory.facts import render_owner_identity
from discordbot.services.memory.store import read_owner, iter_scopes
from discordbot.cogs.gen_reply.toolkit import GeminiKeyToolkit
from discordbot.cogs.gen_reply.pipeline import ReplyPipeline
from discordbot.services.memory.pipeline import (
    flavor_of,
    needs_consolidation,
    safe_list_resumable,
    resume_memory_update,
    consolidate_if_needed,
)
from discordbot.cogs.gen_reply.turn_state import dispatched_model, current_answer_streamer
from discordbot.services.memory.git_history import memory_git
from discordbot.services.gemini_keys.balancer import pick_gemini_key
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


def _message_log_fields(message: Message) -> _MessageLogFields:
    """Standard Discord identifying fields for correlating one reply's logs.

    The pipeline-entry log carries the full set; every downstream log carries only
    `message_id` as the correlation key, so a whole turn reconstructs by grepping it.
    `user_name` is the stable handle, `display_name` the per-guild nickname;
    `guild_id` / `guild_name` are None in a DM.
    """
    guild = message.guild
    return {
        "user_id": message.author.id,
        "user_name": message.author.name,
        "display_name": message.author.display_name,
        "message_id": message.id,
        "channel_id": message.channel.id,
        "guild_id": guild.id if guild else None,
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
        # One toolkit per Gemini key, built on first use and kept for the life of the process.
        # Keyed by the key number, with None for the unconfigured deployment. Long-lived on
        # purpose: the caches inside hold Files API uris only that key can read, so rebuilding
        # per reply would re-upload the whole history window every time.
        self._toolkits: dict[int | None, GeminiKeyToolkit] = {}
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

    async def lease_toolkit(self) -> GeminiKeyToolkit:
        """Leases the least-used Gemini key and returns the toolkit bound to it.

        One call per unit of work that must not cross keys: a reply, or one background job.
        Everything downstream then reads its clients and models off the returned toolkit
        rather than off this cog, which is what keeps a reply's Files API uploads and the
        request naming them on one Google project.

        Returns:
            The toolkit for the leased key, or the unpinned one when no key is configured.
        """
        slot = await pick_gemini_key(config=self.config)
        index = slot.index if slot is not None else None
        cached = self._toolkits.get(index)
        if cached is not None:
            return cached
        toolkit = GeminiKeyToolkit(bot=self.bot, openai_client=self.openai_client, slot=slot)
        self._toolkits[index] = toolkit
        return toolkit

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
            # Leased per job rather than once for the sweep: this is the burstiest moment in
            # the process's life, and one lease would land all of it on a single key. Nothing
            # here is bound to a key (the memory review reaches no Files API), so the lease is
            # only about the count.
            toolkit = await self.lease_toolkit()
            writer = (
                toolkit.server_memory_writer if job.flavor == "server" else toolkit.memory_writer
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
            swept_toolkit = await self.lease_toolkit()
            writer = (
                swept_toolkit.server_memory_writer
                if flavor_of(scope=scope) == "server"
                else swept_toolkit.memory_writer
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

    async def _deliver_failure_notice(self, *, message: Message, error_embed: Embed) -> None:
        """Shows the turn's failure, on the reply it was streaming into where there is one.

        Half the turns that fail here already painted something (23 of 46 in one 2026-08-21 log),
        and left beside that reply the embed reads as unrelated while the reply itself, carrying
        no usage footer, reads as an answer that merely stopped. So the streamer is asked first
        and takes the error onto its own message. Everything it turns down -- every failure
        before the answer, and a retry notice already withdrawn -- gets a fresh message here.
        """
        streamer = current_answer_streamer.get()
        if streamer is not None and await streamer.land_failure(embed=error_embed):
            return
        spacer = embed_spacer_payload(embeds=[error_embed], is_edit=False, target=message)
        try:
            await message.reply(content=None, embed=error_embed, **spacer)
        except HTTPException as send_error:
            # Source deleted before the error landed (50035): send it unparented. Rebuild
            # the spacer; the failed reply already consumed the single-use spacer file.
            if send_error.code != 50035 and not isinstance(send_error, NotFound):
                raise
            fresh_spacer = embed_spacer_payload(
                embeds=[error_embed], is_edit=False, target=message
            )
            await message.channel.send(content=None, embed=error_embed, **fresh_spacer)

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
        if in_active_research_thread(bot=self.bot, channel_id=message.channel.id):
            logfire.debug(
                "gen_reply skipped: the research cog is still writing into this thread",
                **_message_log_fields(message=message),
            )
            return

        toolkit = await self.lease_toolkit()
        user_prompt = await toolkit.input_builder.get_user_prompt(content=message.content)
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
            forwarded := toolkit.input_builder.forwarded_request_text(message=message)
        ):
            user_prompt = f"{user_prompt}\n{forwarded}".strip() if user_prompt else forwarded

        if not user_prompt and not has_attachment and not is_forward:
            logfire.debug(
                "gen_reply empty prompt; replied with ?", **_message_log_fields(message=message)
            )
            await update_reaction(message=message, bot_user=self.bot.user, emoji="❓")
            await message.reply(content="?")
            return

        logfire.info(
            "gen_reply received",
            **_message_log_fields(message=message),
            prompt_chars=len(user_prompt),
            has_attachment=has_attachment,
            attachment_count=len(message.attachments),
            sticker_count=len(message.stickers),
            is_dm=is_dm,
        )

        reactions = ReactionStatusChain(message=message, bot_user=self.bot.user)
        try:
            await ReplyPipeline(
                client=self.openai_client,
                bot=self.bot,
                config=self.config,
                media_delivery=self.media_delivery,
                usage_recorder=self.usage_recorder,
                toolkit=toolkit,
                message=message,
                user_prompt=user_prompt,
                reactions=reactions,
            ).run()
        except Exception as e:
            logfire.error(
                "gen_reply failed",
                **_message_log_fields(message=message),
                model=dispatched_model.get(),
                key_index=toolkit.key_index,
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
                await self._deliver_failure_notice(message=message, error_embed=error_embed)
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


def setup(bot: commands.Bot) -> None:
    """Adds the ReplyGeneratorCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(ReplyGeneratorCogs(bot), override=True)
