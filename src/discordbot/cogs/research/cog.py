"""Deep-research cog: long-running Gemini managed-agent research delivered in a Discord thread.

A user asks for deep research (the QA answer model emits a `<deep-research>` marker, handed
here by `gen_reply`, or they run `/deep_research`). The bot opens a thread, runs the
`antigravity-preview-05-2026` agent, and posts the cited report there, pinging the user. That
one report is the whole feature: there is no tier to upgrade into and no button under it.

Everything talks DIRECT to Google (`gemini_api_key`, no proxy), like every Interactions API path
in this project (see `agent.py`). Sessions persist in `reply.db` so a restart resumes an
in-flight research (`store=True` keeps the interaction alive server-side). The cog never blocks
the gateway: agent work runs in tracked background tasks.
"""

from typing import TYPE_CHECKING
import asyncio
from functools import cached_property
import contextlib

from google import genai
from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import (
    Embed,
    Locale,
    Object,
    Thread,
    Message,
    NotFound,
    Forbidden,
    Interaction,
    SlashOption,
    TextChannel,
    AllowedMentions,
)
from nextcord.ext import commands

from discordbot.utils.llm import create_text_or_none
from discordbot.typings.llm import LLMConfig
from discordbot.cogs.research import database as db
from discordbot.typings.colors import DISCORD_RED
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.timezone import database_now
from discordbot.utils.reactions import update_reaction
from discordbot.typings.timeouts import THREAD_TITLE_TIMEOUT_SECONDS
from discordbot.utils.llm_errors import extract_friendly_error
from discordbot.cogs.research.agent import (
    ResearchResult,
    stream_antigravity,
    resume_research_stream,
)
from discordbot.utils.asyncio_locks import KeyedLockManager
from discordbot.utils.model_pricing import get_token_rates
from discordbot.utils.media_delivery import build_media_delivery_planner
from discordbot.cogs.research.prompts import THREAD_TITLE_PROMPT, RESEARCH_SYSTEM_INSTRUCTION
from discordbot.cogs.research.delivery import deliver_report
from discordbot.cogs.research.streaming import ResearchProgressStreamer

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Coroutine

# The agent name shown in the thread's status line and in the streamer's live header. One agent
# runs every research, so the two must agree on one string rather than each spelling their own.
RESEARCH_LABEL = "Antigravity"
# Discord thread names cap at 100 chars; keep margin (a hard-limit safety trim, not length control).
THREAD_NAME_MAX = 90
# The bot's `dino` app emoji, reacted onto the source message when deep research is launched so
# the activation reads as distinct from the normal QA pipeline reactions.
DINO_EMOJI = "<:dino:1517560319281594570>"


def _fallback_thread_name(*, brief: str) -> str:
    """Thread-title fallback (the brief's first line) when LLM title generation is unavailable."""
    first_line = next((line.strip() for line in brief.splitlines() if line.strip()), "")
    title = first_line or "深度研究"
    return title[:THREAD_NAME_MAX]


def _terminal_phase(*, status: str) -> db.ResearchPhase:
    """Maps a terminal interaction status onto a stored phase."""
    if status == "completed":
        return "done"
    if status == "cancelled":
        return "cancelled"
    return "failed"


class ResearchCogs(commands.Cog):
    """Owns the deep-research thread lifecycle, slash command, and restart resume."""

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the research cog.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()
        self.runtime_models = RuntimeModelCatalog()
        self.media_delivery = build_media_delivery_planner()
        # One in-flight research per owner; the lock guards the check-then-create.
        self._owner_locks: KeyedLockManager[int] = KeyedLockManager()
        self._tasks: set[asyncio.Task[None]] = set()
        # Thread ids the cog is actively driving; `gen_reply` checks this so QA does not answer
        # inside a thread the cog is still writing its own status, reasoning and report into.
        self._active_threads: set[int] = set()
        self._resume_started = False

    @cached_property
    def interactions_client(self) -> genai.Client:
        """The Gemini Interactions client, built lazily on first use.

        DIRECT to Google (`gemini_api_key`, no base_url / proxy): a managed agent rides the native
        Interactions API, which this project always calls direct. Built inline like every other
        direct-to-Google path (the `create_*_client` factories are gone). `genai.Client` raises
        `ValueError` on a missing key rather than deferring it to the first call (measured), and
        both run loops read this property inside their own try, so that raise still lands as a
        thread failure notice instead of an unhandled background-task error.
        """
        return genai.Client(api_key=self.config.gemini_api_key)

    @cached_property
    def responses_client(self) -> AsyncOpenAI:
        """The LiteLLM-proxy Responses client for small side calls (the thread-title generator).

        Built inline like every other client here (there is no `utils/llm.py` client factory left);
        distinct from the direct `interactions_client` since a plain Responses call rides the proxy
        fine.
        """
        return AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def is_research_thread(self, *, channel_id: int) -> bool:
        """Whether a channel id is a research thread the cog is actively driving."""
        return channel_id in self._active_threads

    def _spawn(self, coro: "Coroutine[Any, Any, None]") -> None:
        """Runs `coro` as a tracked background task so the gateway never blocks on agent work."""
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _system_instruction(self) -> str:
        """The research agent system instruction with today's date appended for recency."""
        return f"{RESEARCH_SYSTEM_INSTRUCTION}\n\nToday's date: {database_now():%Y-%m-%d}."

    async def _generate_thread_name(self, *, brief: str) -> str:
        """Generates a short thread title from the brief via `triage_model`, best-effort.

        Brevity is steered by the prompt (not a token cap); on timeout or failure the brief's
        first line is used, and the result is trimmed to Discord's hard name limit as a safety net.
        """
        raw = await create_text_or_none(
            client=self.responses_client,
            model=self.runtime_models.triage_model,
            instructions=THREAD_TITLE_PROMPT,
            user_text=brief,
            end_user_id="deep-research",
            timeout_seconds=THREAD_TITLE_TIMEOUT_SECONDS,
        )
        title = next(
            (line.strip().strip('"') for line in (raw or "").splitlines() if line.strip()), ""
        )
        return (title or _fallback_thread_name(brief=brief))[:THREAD_NAME_MAX]

    # ----- entry points -------------------------------------------------------------------

    async def launch(
        self, *, message: "Message", brief: str, anchor: "Message | None" = None
    ) -> None:
        """QA-marker entry: opens a thread and starts the research.

        `message` identifies the owner; `anchor` is the message the thread hangs off. The bot's
        own reply reads more intuitively than the user's message, so the caller passes it; it
        falls back to the user's message when the reply is unavailable.
        """
        if not self.config.deep_research_available:
            return
        outcome, existing = await self._start_for(
            owner_id=message.author.id,
            owner_mention=message.author.mention,
            brief=brief,
            anchor=anchor or message,
        )
        if outcome == "exists" and existing is not None:
            with contextlib.suppress(Exception):
                await message.reply(content=f"你已經有一個深度研究在進行了:<#{existing}>")
        elif outcome == "unsupported":
            with contextlib.suppress(Exception):
                await message.reply(
                    content="深度研究只能在伺服器的一般文字頻道開(私訊或討論串裡開不了新的 thread)"
                )
        elif outcome == "error":
            with contextlib.suppress(Exception):
                await message.reply(content="開研究串失敗了,等等再試一次")

    @nextcord.slash_command(
        name="deep_research",
        description="Kick off a long, cited deep-research report in a thread.",
        name_localizations={Locale.zh_TW: "深度研究", Locale.ja: "ディープリサーチ"},
        description_localizations={
            Locale.zh_TW: "開一條 thread 進行帶引用的深度研究(耗時數分鐘,完成後標記你)",
            Locale.ja: "スレッドで引用付きのディープリサーチを実行します（数分かかり、完了時にメンションします）。",
        },
        nsfw=False,
    )
    async def deep_research(
        self,
        interaction: Interaction[commands.Bot],
        topic: str = SlashOption(
            name="topic",
            description="What to research (a clear, self-contained topic).",
            name_localizations={Locale.zh_TW: "主題", Locale.ja: "トピック"},
            description_localizations={
                Locale.zh_TW: "要研究的主題(清楚、可獨立理解的題目)",
                Locale.ja: "調査するトピック(明確で自己完結したテーマ)。",
            },
            required=True,
        ),
    ) -> None:
        """Opens a research thread for the given topic and starts the research.

        Args:
            interaction: The slash interaction.
            topic: The research topic / brief.
        """
        if not self.config.deep_research_available:
            await interaction.response.send_message(content="深度研究目前停用中", ephemeral=True)
            return
        if interaction.user is None or not isinstance(interaction.channel, TextChannel):
            await interaction.response.send_message(
                content="深度研究只能在伺服器的一般文字頻道開喔(私訊或討論串裡開不了 thread)",
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=True)
        # Anchor the thread on a bot message so the same message-based create_thread path is reused.
        # The topic is user-supplied: restrict mentions to the requester so an `@everyone` / role
        # mention embedded in it cannot turn a research request into a mass ping.
        anchor = await interaction.channel.send(
            content=f"{interaction.user.mention} 要研究:{topic[:200]}",
            allowed_mentions=AllowedMentions(
                everyone=False, roles=False, users=[interaction.user]
            ),
        )
        outcome, existing = await self._start_for(
            owner_id=interaction.user.id,
            owner_mention=interaction.user.mention,
            brief=topic,
            anchor=anchor,
        )
        if outcome == "started" and existing is not None:
            await interaction.edit_original_message(content=f"開好了:<#{existing}>")
        elif outcome == "exists" and existing is not None:
            with contextlib.suppress(Exception):
                await anchor.delete()
            await interaction.edit_original_message(content=f"你已經有一個在進行了:<#{existing}>")
        else:
            with contextlib.suppress(Exception):
                await anchor.delete()
            await interaction.edit_original_message(content="開研究串失敗了,等等再試一次")

    async def _start_for(
        self, *, owner_id: int, owner_mention: str, brief: str, anchor: "Message"
    ) -> tuple[str, int | None]:
        """Claims the owner's slot, opens the thread, and spawns the research.

        Returns `(outcome, thread_or_existing_id)` where outcome is one of
        `started` / `exists` / `unsupported` / `error`.
        """
        # A research thread can only hang off a message in a guild text channel; a DM, an existing
        # thread, or a forum post cannot host a nested thread, so refuse before promising research.
        if anchor.guild is None or not isinstance(anchor.channel, TextChannel):
            return "unsupported", None
        async with self._owner_locks.hold(key=owner_id):
            existing = await db.active_thread_for_owner(owner_id=owner_id)
            if existing is not None:
                return "exists", existing
            name = await self._generate_thread_name(brief=brief)
            try:
                thread = await anchor.create_thread(name=name, auto_archive_duration=1440)
            except Exception as exc:
                # Broad: create_thread can fail on permissions, an LLM-authored name Discord
                # rejects, or an outage; all of them end the launch the same way.
                logfire.error(
                    "failed to create research thread",
                    message_id=anchor.id,
                    owner_id=owner_id,
                    channel_id=anchor.channel.id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
                return "error", None
            agent = self.runtime_models.antigravity_model.name
            await db.upsert_session(
                thread_id=thread.id,
                owner_id=owner_id,
                channel_id=anchor.channel.id,
                guild_id=anchor.guild.id,
                source_message_id=anchor.id,
                agent=agent,
                interaction_id=None,
                brief=brief,
                phase="researching",
            )
            self._active_threads.add(thread.id)
        # Mark the source message so the deep-research activation is visually distinct from the
        # normal QA pipeline reactions (best-effort).
        with contextlib.suppress(Exception):
            await update_reaction(message=anchor, bot_user=self.bot.user, emoji=DINO_EMOJI)
        self._spawn(
            self._run_research(
                thread=thread, owner_mention=owner_mention, brief=brief, agent=agent
            )
        )
        return "started", thread.id

    # ----- research runs ------------------------------------------------------------------

    async def _run_research(
        self, *, thread: "Thread", owner_mention: str, brief: str, agent: str
    ) -> None:
        """Streams the Antigravity research and delivers the report into the thread."""
        status = await self._safe_send(
            thread=thread, content=f"-# Researching... ({RESEARCH_LABEL})"
        )
        streamer = ResearchProgressStreamer(status=status, label=RESEARCH_LABEL)

        async def _persist(interaction_id: str) -> None:
            await db.set_interaction(
                thread_id=thread.id,
                interaction_id=interaction_id,
                agent=agent,
                phase="researching",
            )

        # The agent run and the delivery are separate steps: both stay broad (a fire-and-forget task
        # has nobody to raise to) but each names what actually failed.
        try:
            result = await stream_antigravity(
                client=self.interactions_client,
                agent=agent,
                brief=brief,
                system_instruction=self._system_instruction(),
                streamer=streamer,
                on_created=_persist,
            )
        except Exception as exc:
            logfire.error(
                "research failed",
                thread_id=thread.id,
                agent=agent,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await self._fail_run(
                thread=thread, owner_mention=owner_mention, exc=exc, status=status
            )
            return
        try:
            await self._finish(
                thread=thread,
                owner_mention=owner_mention,
                result=result,
                agent=agent,
                status=status,
            )
        except Exception as exc:
            logfire.error(
                "research delivery failed",
                thread_id=thread.id,
                agent=agent,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await self._fail_run(
                thread=thread, owner_mention=owner_mention, exc=exc, status=status
            )

    async def _fail_run(
        self, *, thread: "Thread", owner_mention: str, exc: Exception, status: Message | None
    ) -> None:
        """Tells the owner a run died, finalizes its status message, and frees the owner's slot."""
        await self._post_failure(thread=thread, owner_mention=owner_mention, exc=exc)
        await self._finalize_status(
            status=status, thread=thread, content=f"-# Research failed ({RESEARCH_LABEL})"
        )
        await db.set_phase(thread_id=thread.id, phase="failed")
        self._active_threads.discard(thread.id)

    async def _finish(
        self,
        *,
        thread: "Thread",
        owner_mention: str,
        result: ResearchResult,
        agent: str,
        status: Message | None,
    ) -> None:
        """Delivers a terminal result, records its phase, and releases the thread.

        On a completed run the opening status message is spent by `deliver_report`, which edits the
        report's first chunk into it; any other terminal status finalizes it with a failure line
        instead.
        """
        if not result.ok:
            await self._post_failure(
                thread=thread,
                owner_mention=owner_mention,
                reason=_failure_text(status=result.status),
            )
            await self._finalize_status(
                status=status, thread=thread, content=f"-# Research failed ({RESEARCH_LABEL})"
            )
            await db.set_phase(thread_id=thread.id, phase=_terminal_phase(status=result.status))
            self._active_threads.discard(thread.id)
            return
        footer = _usage_footer(
            agent=agent, input_tokens=result.input_tokens, output_tokens=result.output_tokens
        )
        await deliver_report(
            thread=thread,
            status=status,
            owner_mention=owner_mention,
            result=result,
            footer=footer,
            allowed_mentions=_owner_allowed_mentions(
                owner_id=_owner_id_from_mention(mention=owner_mention)
            ),
            media_delivery=self.media_delivery,
        )
        await db.set_phase(thread_id=thread.id, phase="done")
        self._active_threads.discard(thread.id)

    async def _finalize_status(
        self, *, status: Message | None, thread: "Thread", content: str
    ) -> None:
        """Edits the opening status message to its terminal content.

        Falls back to a fresh send when there is no status message (a restart resume) or the edit
        fails (e.g. the opening message was deleted).
        """
        if status is not None:
            try:
                await status.edit(content=content, allowed_mentions=AllowedMentions.none())
                return
            except Exception as exc:
                # Broad: any Discord failure is recoverable by the fallback send below.
                logfire.warn(
                    "failed to edit research status message",
                    thread_id=thread.id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
        try:
            await thread.send(content=content, allowed_mentions=AllowedMentions.none())
        except Exception as exc:
            # Broad: callers record the terminal phase right after us and cannot handle a raise.
            logfire.warn(
                "failed to post terminal research status",
                thread_id=thread.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    async def _post_failure(
        self,
        *,
        thread: "Thread",
        owner_mention: str,
        exc: Exception | None = None,
        reason: str | None = None,
    ) -> None:
        """Posts the real failure reason as an error embed pinging the owner (mirrors gen_reply).

        Pass `exc` for an exception path (the friendly error + its type are shown so the cause is
        fixable) or `reason` for a non-completed terminal status.
        """
        if reason is None and exc is not None:
            reason = extract_friendly_error(exc=exc)
        embed = Embed(
            title="深度研究失敗",
            description=f"```\n{reason or '未知錯誤'}\n```",
            color=DISCORD_RED,
        )
        if exc is not None:
            embed.set_footer(text=type(exc).__name__)
        try:
            await thread.send(
                content=f"{owner_mention} ⚠️",
                embed=embed,
                allowed_mentions=_owner_allowed_mentions(
                    owner_id=_owner_id_from_mention(mention=owner_mention)
                ),
            )
        except Exception as send_exc:
            # Broad: every caller runs its cleanup (phase write, slot release) right after us, so
            # this last user-facing step must never raise.
            logfire.warn(
                "failed to post research failure notice",
                thread_id=thread.id,
                error_type=type(send_exc).__name__,
                _exc_info=send_exc,
            )

    # ----- restart resume -----------------------------------------------------------------

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Resumes in-flight research after a restart (runs once)."""
        if self._resume_started:
            return
        self._resume_started = True
        self._spawn(self._resume_all())

    async def _resume_all(self) -> None:
        """Resumes every session still `researching` when the process came back up.

        The kill-switch gates this exactly as it gates `launch` and `/deep_research`: it is
        flipped over a provider or a cost problem, and a run already open is still work with that
        provider, so an off switch means the bot re-attaches to nothing and delivers nothing.

        A skipped row is left `researching` rather than failed, which is the truth about it: the
        interaction runs server-side under `background=True` / `store=True` whether or not the
        bot is attached, so the next start with the switch on picks it up exactly as a plain
        restart does, and marking it failed would throw away a report the provider has already
        produced and billed for. Nothing is posted into the threads either, since this sweep runs
        on every start and a notice would repeat for as long as the switch stays off.
        """
        sessions = await db.list_resumable()
        if not sessions:
            return
        if not self.config.deep_research_available:
            logfire.info(
                "deep research is off; left in-flight sessions for a later start",
                count=len(sessions),
            )
            return
        for session in sessions:
            self._active_threads.add(session.thread_id)
            self._spawn(self._resume_one(session=session))
        logfire.info("resumed in-flight research sessions", count=len(sessions))

    async def _resume_one(self, *, session: db.PersistentResearchSession) -> None:
        """Resumes one research session, delivering when it settles."""
        thread = await self._fetch_thread(thread_id=session.thread_id)
        owner_mention = f"<@{session.owner_id}>"
        # No interaction id means the row was written but the bot restarted before the run id was
        # stored; there is nothing to resume. Tell the thread so the owner is not left staring at
        # the old `Researching...` message forever.
        if session.interaction_id is None:
            await db.set_phase(thread_id=session.thread_id, phase="failed")
            self._active_threads.discard(session.thread_id)
            await self._notify_resume_failed(thread=thread, owner_id=session.owner_id)
            return
        # Give the resumed run the same live reasoning view as a fresh one; a fetch miss leaves
        # status None so the streamer's editor no-ops but still drives the stream to a result.
        status = (
            await self._safe_send(thread=thread, content=f"-# Researching... ({RESEARCH_LABEL})")
            if thread is not None
            else None
        )
        streamer = ResearchProgressStreamer(status=status, label=RESEARCH_LABEL)
        try:
            result = await resume_research_stream(
                client=self.interactions_client,
                interaction_id=session.interaction_id,
                streamer=streamer,
            )
        except Exception:
            logfire.warn("research resume failed", thread_id=session.thread_id, _exc_info=True)
            await db.set_phase(thread_id=session.thread_id, phase="failed")
            self._active_threads.discard(session.thread_id)
            await self._notify_resume_failed(thread=thread, owner_id=session.owner_id)
            return
        if thread is None:
            await db.set_phase(
                thread_id=session.thread_id, phase=_terminal_phase(status=result.status)
            )
            self._active_threads.discard(session.thread_id)
            return
        await self._finish(
            thread=thread,
            owner_mention=owner_mention,
            result=result,
            agent=session.agent,
            status=status,
        )

    async def _notify_resume_failed(self, *, thread: "Thread | None", owner_id: int) -> None:
        """Tells a thread its interrupted research could not be resumed after a restart (best-effort)."""
        if thread is None:
            return
        await self._safe_send(
            thread=thread,
            content=f"<@{owner_id}> 重啟後沒辦法接回剛剛的研究,麻煩重新發起一次",
            allowed_mentions=_owner_allowed_mentions(owner_id=owner_id),
        )

    async def _fetch_thread(self, *, thread_id: int) -> "Thread | None":
        """Returns the thread by id from cache or a REST fetch, or None when gone."""
        cached = self.bot.get_channel(thread_id)
        if isinstance(cached, Thread):
            return cached
        try:
            fetched = await self.bot.fetch_channel(thread_id)
        except (NotFound, Forbidden):
            logfire.info("research thread is gone; skipping", thread_id=thread_id)
            return None
        # Broad on purpose: every caller treats None as "gone" and returns, so a transient REST
        # or transport failure must not raise into a resume sweep. It is logged apart from the
        # deleted case so the two stop looking the same in the log.
        except Exception as exc:
            logfire.warn(
                "could not fetch the research thread; treating it as gone",
                thread_id=thread_id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None
        return fetched if isinstance(fetched, Thread) else None

    # ----- helpers ------------------------------------------------------------------------

    async def _safe_send(
        self, *, thread: "Thread", content: str, allowed_mentions: "AllowedMentions | None" = None
    ) -> Message | None:
        """Best-effort `thread.send`, returning the message or None on failure.

        Mentions default to fully suppressed (`AllowedMentions.none()`); a caller that wants the
        owner pinged passes an owner-only policy, so agent-generated content can never mass-ping.
        """
        mentions = allowed_mentions if allowed_mentions is not None else AllowedMentions.none()
        try:
            return await thread.send(content=content, allowed_mentions=mentions)
        except Exception as exc:
            # Broad: every caller treats a missing message as a degraded outcome, never a failure.
            logfire.warn(
                "failed to send research thread message",
                thread_id=thread.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            return None


def _usage_footer(*, agent: str, input_tokens: int, output_tokens: int) -> str:
    """Builds the gen_reply-style usage footer (full model name, tokens, cost) for a result.

    No memory-lookup line: research never reads memory. The agent string is the full model name;
    rates come from the shared LiteLLM pricing table, so an unpriced preview agent shows $0.
    """
    input_rate, output_rate = get_token_rates(model_name=agent)
    cost = input_rate * input_tokens + output_rate * output_tokens
    return f"-# {agent} · ⬆ {input_tokens:,} ⬇ {output_tokens:,} · ${cost:.8f}"


def _failure_text(*, status: str) -> str:
    """Friendly Chinese message for a non-completed terminal status."""
    if status == "budget_exceeded":
        return "研究碰到成本上限了,先到這裡"
    if status == "cancelled":
        return "研究被取消了"
    return "研究沒有順利完成,等等再試試"


def _owner_id_from_mention(*, mention: str) -> int:
    """Parses a `<@id>` mention back into the user id (0 when it has no digits)."""
    digits = "".join(ch for ch in mention if ch.isdigit())
    return int(digits) if digits else 0


def _owner_allowed_mentions(*, owner_id: int) -> AllowedMentions:
    """Restricts a research-thread message to pinging only its owner.

    The report text is agent-generated, so any `@everyone` / role / other-user mention it
    contains must not resolve; only the deliberate owner ping is allowed through.
    """
    return AllowedMentions(everyone=False, roles=False, users=[Object(id=owner_id)])


def setup(bot: commands.Bot) -> None:
    """Adds the ResearchCogs to the bot."""
    bot.add_cog(ResearchCogs(bot), override=True)
