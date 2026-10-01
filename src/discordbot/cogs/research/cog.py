"""Deep-research cog: long-running Gemini managed-agent research delivered in a Discord thread.

A user asks for deep research (the QA answer model emits a `<deep-research>` marker, handed
here by `gen_reply`, or they run `/deep_research`). The bot opens a thread, runs the
`RuntimeModelCatalog.antigravity_model` agent, and posts the cited report there, pinging the
user. That one report is the whole feature: there is no tier to upgrade into and no button under
it.

Everything talks DIRECT to Google (`gemini_api_key`, no proxy), like every Interactions API path
in this project (see `agent.py`). Sessions persist in `reply.db` so a restart resumes an
in-flight research (`store=True` keeps the interaction alive server-side). The cog never blocks
the gateway: agent work runs in tracked background tasks.
"""

from typing import TYPE_CHECKING, Literal
from functools import cached_property
import contextlib

from google import genai
from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import (
    Embed,
    Locale,
    Thread,
    Message,
    NotFound,
    Forbidden,
    Interaction,
    SlashOption,
    TextChannel,
    HTTPException,
    AllowedMentions,
)
from nextcord.ext import commands

from discordbot.utils.llm import create_text_or_none
from discordbot.typings.llm import LLMConfig

# Imported as a module so a test that swaps one store call on it reaches this cog.
from discordbot.cogs.research import database as db
from discordbot.typings.colors import DISCORD_RED
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.timezone import database_now
from discordbot.utils.reactions import update_reaction
from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.typings.research import has_research_permissions
from discordbot.typings.timeouts import THREAD_TITLE_TIMEOUT_SECONDS
from discordbot.utils.llm_errors import extract_friendly_error
from discordbot.cogs.research.agent import (
    ResearchResult,
    stream_antigravity,
    resume_research_stream,
)
from discordbot.utils.asyncio_locks import KeyedLockManager, spawn_tracked
from discordbot.utils.model_pricing import get_token_rates
from discordbot.utils.media_delivery import build_media_delivery_planner
from discordbot.cogs.research.prompts import THREAD_TITLE_PROMPT, RESEARCH_SYSTEM_INSTRUCTION
from discordbot.cogs.research.delivery import deliver_report, owner_allowed_mentions
from discordbot.cogs.research.streaming import RESEARCHING_PREFIX, ResearchProgressStreamer

if TYPE_CHECKING:
    import asyncio

# The agent name shown in the thread's status line and in the streamer's live header. One agent
# runs every research, so the two must agree on one string rather than each spelling their own.
RESEARCH_LABEL = "Antigravity"
# Discord thread names cap at 100 chars; keep margin (a hard-limit safety trim, not length control).
THREAD_NAME_MAX = 90

# How a launch attempt ended. Both entry points branch on it, so it is a closed set rather than
# a word each of them spells for itself.
type StartOutcome = Literal["started", "exists", "unsupported", "forbidden", "error"]
# The opening status line a fresh or a resumed run posts before its live view takes over.
RESEARCHING_STATUS = f"{RESEARCHING_PREFIX} ({RESEARCH_LABEL})"
# What that status line ends as when the run it announced ends without a report.
RESEARCH_FAILED_STATUS = f"-# Research failed ({RESEARCH_LABEL})"


def _fallback_thread_name(*, brief: str) -> str:
    """Thread-title fallback (the brief's first line) when LLM title generation is unavailable."""
    first_line = next((line.strip() for line in brief.splitlines() if line.strip()), "")
    title = first_line or "深度研究"
    return title[:THREAD_NAME_MAX]


def _launch_reply(*, outcome: StartOutcome, thread_id: int | None) -> str:
    """What either entry point tells the requester about a launch that ended with `outcome`.

    `thread_id` is the run's thread for `started` and the owner's running one for `exists`.
    """
    if outcome == "started":
        return f"開好了:<#{thread_id}>"
    if outcome == "exists":
        return f"你已經有一個深度研究在進行了:<#{thread_id}>"
    if outcome == "unsupported":
        return "深度研究只能在伺服器的一般文字頻道開(私訊或討論串裡開不了新的 thread)"
    if outcome == "forbidden":
        return "我在這個頻道的權限不夠,開不了研究串"
    return "開研究串失敗了,等等再試一次"


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
        Interactions API, which this project always calls direct. `genai.Client` raises
        `ValueError` on a missing key rather than deferring it to the first call (measured), and
        both run loops read this property inside their own try, so that raise still lands as a
        thread failure notice instead of an unhandled background-task error.
        """
        return genai.Client(api_key=self.config.gemini_api_key)

    @cached_property
    def responses_client(self) -> AsyncOpenAI:
        """The LiteLLM-proxy Responses client for small side calls (the thread-title generator).

        Distinct from the direct `interactions_client`, since a plain Responses call rides the
        proxy fine.
        """
        return AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def is_research_thread(self, *, channel_id: int) -> bool:
        """Whether a channel id is a research thread the cog is actively driving."""
        return channel_id in self._active_threads

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
        outcome, thread_id = await self._start_for(
            owner_id=message.author.id, brief=brief, anchor=anchor or message
        )
        if outcome == "started":
            return
        try:
            await message.reply(content=_launch_reply(outcome=outcome, thread_id=thread_id))
        except Forbidden:
            # The server's overwrites decide who may post here; the ids are the whole finding.
            logfire.warn(
                "deep research cannot say why it did not start",
                message_id=message.id,
                channel_id=message.channel.id,
            )
        # Broad on purpose: the launch has already ended, and anything raised here would reach
        # `research_bridge`, which reports it as a dropped brief.
        except Exception as exc:
            # A reply to a message that is already gone comes back as 50035, not only as NotFound.
            if isinstance(exc, HTTPException) and (isinstance(exc, NotFound) or exc.code == 50035):
                logfire.info(
                    "deep research's request is gone before it could say why it did not start",
                    message_id=message.id,
                    channel_id=message.channel.id,
                )
                return
            logfire.warn(
                "failed to say why deep research did not start",
                message_id=message.id,
                channel_id=message.channel.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    @nextcord.slash_command(
        name="deep_research",
        description="Kick off a long, cited deep-research report in a thread.",
        name_localizations={Locale.zh_TW: "深度研究", Locale.ja: "ディープリサーチ"},
        description_localizations={
            Locale.zh_TW: "開一條 thread 進行帶引用的深度研究(耗時數分鐘,完成後標記你)",
            Locale.ja: "スレッドで引用付きのディープリサーチを実行します（数分かかり、完了時にメンションします）。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
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
                content=_launch_reply(outcome="unsupported", thread_id=None), ephemeral=True
            )
            return
        # Read for the bot's own member, whose token every later write uses, so a channel it cannot
        # run in is refused before a title call and an anchor ping are spent on it.
        if not has_research_permissions(
            channel=interaction.channel, member=interaction.channel.guild.me
        ):
            await interaction.response.send_message(
                content=_launch_reply(outcome="forbidden", thread_id=None), ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=True)
        # Anchor the thread on a bot message so the same message-based create_thread path is reused.
        # The topic is user-supplied: restrict mentions to the requester so an `@everyone` / role
        # mention embedded in it cannot turn a research request into a mass ping.
        try:
            anchor = await interaction.channel.send(
                content=f"{interaction.user.mention} 要研究:{topic[:200]}",
                allowed_mentions=owner_allowed_mentions(owner_id=interaction.user.id),
            )
        except Forbidden:
            # The command reached a channel the bot's own identity may not post in; the server's
            # overwrites decide that, so the type and the ids are the whole finding.
            logfire.warn(
                "deep research cannot post its anchor in this channel",
                channel_id=interaction.channel.id,
                owner_id=interaction.user.id,
            )
            await interaction.edit_original_message(
                content=_launch_reply(outcome="forbidden", thread_id=None)
            )
            return
        outcome, thread_id = await self._start_for(
            owner_id=interaction.user.id, brief=topic, anchor=anchor
        )
        if outcome != "started":
            # Inert cleanup: the anchor announced a run that is not happening.
            with contextlib.suppress(Exception):
                await anchor.delete()
        await interaction.edit_original_message(
            content=_launch_reply(outcome=outcome, thread_id=thread_id)
        )

    async def _start_for(  # noqa: PLR0911 -- one early outcome per way a launch stops short
        self, *, owner_id: int, brief: str, anchor: "Message"
    ) -> tuple[StartOutcome, int | None]:
        """Claims the owner's slot, opens the thread, and spawns the research.

        Returns `(outcome, thread_or_existing_id)`.
        """
        # A research thread can only hang off a message in a guild text channel; a DM, an existing
        # thread, or a forum post cannot host a nested thread, so refuse before promising research.
        if anchor.guild is None or not isinstance(anchor.channel, TextChannel):
            return "unsupported", None
        async with self._owner_locks.hold(key=owner_id):
            try:
                existing = await db.active_thread_for_owner(owner_id=owner_id)
            except Exception as exc:
                # Broad: every store failure ends the launch the same way, as an answered error.
                logfire.error(
                    "failed to read the owner's research slot",
                    message_id=anchor.id,
                    owner_id=owner_id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
                return "error", None
            if existing is not None:
                return "exists", existing
            name = await self._generate_thread_name(brief=brief)
            try:
                thread = await anchor.create_thread(name=name, auto_archive_duration=1440)
            except Forbidden:
                # Expected rather than diagnosable: the server's overwrites deny the bot threads
                # here, so the ids say everything a traceback would.
                logfire.warn(
                    "deep research cannot open a thread in this channel",
                    message_id=anchor.id,
                    owner_id=owner_id,
                    channel_id=anchor.channel.id,
                )
                return "forbidden", None
            except Exception as exc:
                # Broad: create_thread can fail on an LLM-authored name Discord rejects or an
                # outage; all of them end the launch the same way.
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
            try:
                await db.insert_session(
                    thread_id=thread.id,
                    owner_id=owner_id,
                    channel_id=anchor.channel.id,
                    guild_id=anchor.guild.id,
                    source_message_id=anchor.id,
                    agent=agent,
                    brief=brief,
                )
            except Exception as exc:
                # Broad, as above. A thread with no row is never run, so the one just opened is
                # withdrawn.
                logfire.error(
                    "failed to record research session",
                    message_id=anchor.id,
                    owner_id=owner_id,
                    thread_id=thread.id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
                try:
                    await thread.delete()
                except Forbidden:
                    # Deleting a thread takes Manage Threads, which a launch never requires, so
                    # the ids are the whole finding.
                    logfire.warn(
                        "research thread of a failed launch could not be deleted",
                        thread_id=thread.id,
                        owner_id=owner_id,
                    )
                except Exception as delete_error:
                    # Broad: the launch has already failed, and any failure leaves the thread.
                    logfire.warn(
                        "failed to delete the research thread of a failed launch",
                        thread_id=thread.id,
                        owner_id=owner_id,
                        error_type=type(delete_error).__name__,
                        _exc_info=delete_error,
                    )
                return "error", None
            self._active_threads.add(thread.id)
        # Mark the source message with the bot's `dino` app emoji so the deep-research activation
        # is visually distinct from the normal QA pipeline reactions (best-effort).
        await update_reaction(
            message=anchor, bot_user=self.bot.user, emoji="<:dino:1517560319281594570>"
        )
        spawn_tracked(
            coro=self._run_research(thread=thread, owner_id=owner_id, brief=brief, agent=agent),
            tasks=self._tasks,
            name=f"research-run-{thread.id}",
        )
        return "started", thread.id

    # ----- research runs ------------------------------------------------------------------

    async def _run_research(
        self, *, thread: "Thread", owner_id: int, brief: str, agent: str
    ) -> None:
        """Streams the Antigravity research and delivers the report into the thread."""
        status = await self._safe_send(thread=thread, content=RESEARCHING_STATUS)
        streamer = ResearchProgressStreamer(status=status, label=RESEARCH_LABEL)

        async def _persist(interaction_id: str) -> None:
            await db.set_interaction(thread_id=thread.id, interaction_id=interaction_id)

        # Broad: a fire-and-forget task has nobody to raise to. The delivery's own guard sits in
        # `_finish`, which a resumed run shares.
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
            await self._fail_run(thread=thread, owner_id=owner_id, status=status, failure=exc)
            return
        await self._finish(
            thread=thread, owner_id=owner_id, result=result, agent=agent, status=status
        )

    async def _fail_run(
        self, *, thread: "Thread", owner_id: int, status: Message | None, failure: Exception | str
    ) -> None:
        """Tells the owner a run ended without a report, finalizes its status, and releases it.

        `failure` is the exception that ended the run, or the non-completed terminal status the
        interaction settled with, which also decides the phase recorded.
        """
        if isinstance(failure, Exception):
            await self._post_failure(thread=thread, owner_id=owner_id, exc=failure)
            phase: db.ResearchPhase = "failed"
        else:
            await self._post_failure(
                thread=thread, owner_id=owner_id, reason=_failure_text(status=failure)
            )
            phase = _terminal_phase(status=failure)
        await self._finalize_status(status=status, thread=thread, content=RESEARCH_FAILED_STATUS)
        await self._release(thread_id=thread.id, phase=phase)

    async def _release(self, *, thread_id: int, phase: db.ResearchPhase) -> None:
        """Ends a run: lets QA answer in its thread again and records its terminal phase.

        The recorded phase is what frees the owner's one-research slot. Every caller has already
        decided how the run ended, so a failed write is logged, never raised: a raise would read
        as the run failing, or cut off the steps that tell the thread how it ended.
        """
        self._active_threads.discard(thread_id)
        try:
            await db.set_phase(thread_id=thread_id, phase=phase)
        # Broad: whatever the store raised, the row stays `researching` and nothing here can
        # end it, so every failure is the same finding.
        except Exception as exc:
            logfire.error(
                "failed to record how a research run ended",
                thread_id=thread_id,
                phase=phase,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    async def _finish(
        self,
        *,
        thread: "Thread",
        owner_id: int,
        result: ResearchResult,
        agent: str,
        status: Message | None,
    ) -> None:
        """Delivers a terminal result, records its phase, and releases the thread.

        On a completed run the opening status message is spent by `deliver_report`, which edits the
        report's first chunk into it. Any other terminal status, or a delivery that raises before
        any of the report lands, ends the run as a failure.
        """
        if not result.ok:
            await self._fail_run(
                thread=thread, owner_id=owner_id, status=status, failure=result.status
            )
            return
        try:
            footer = _usage_footer(
                agent=agent, input_tokens=result.input_tokens, output_tokens=result.output_tokens
            )
            await deliver_report(
                thread=thread,
                status=status,
                owner_id=owner_id,
                result=result,
                footer=footer,
                media_delivery=self.media_delivery,
            )
        # Broad: a run task has nobody to raise to. Every write `deliver_report` makes is guarded,
        # so whatever it raises came before any of the report landed, and the run ends failed.
        except Exception as exc:
            logfire.error(
                "research delivery failed",
                thread_id=thread.id,
                agent=agent,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await self._fail_run(thread=thread, owner_id=owner_id, status=status, failure=exc)
            return
        await self._release(thread_id=thread.id, phase="done")

    async def _finalize_status(
        self, *, status: Message | None, thread: "Thread", content: str
    ) -> None:
        """Edits the opening status message to its terminal content.

        Falls back to a fresh send when there is no status message (its post failed, or a resume
        found none from before the restart) or the edit fails (e.g. the opening message was
        deleted).
        """
        if status is not None:
            try:
                await status.edit(content=content, allowed_mentions=AllowedMentions.none())
                return
            except Forbidden:
                # The thread's overwrites changed under the run; the id is the whole finding.
                logfire.warn("research thread refused the status edit", thread_id=thread.id)
            except Exception as exc:
                # Broad: any Discord failure is recoverable by the fallback send below.
                logfire.warn(
                    "failed to edit research status message",
                    thread_id=thread.id,
                    error_type=type(exc).__name__,
                    _exc_info=exc,
                )
        await self._safe_send(
            thread=thread,
            content=content,
            refused="research thread refused the terminal status",
            failed="failed to post terminal research status",
        )

    async def _post_failure(
        self,
        *,
        thread: "Thread",
        owner_id: int,
        exc: Exception | None = None,
        reason: str | None = None,
    ) -> None:
        """Posts the real failure reason as an error embed pinging the owner.

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
        await self._safe_send(
            thread=thread,
            content=f"<@{owner_id}> ⚠️",
            embed=embed,
            allowed_mentions=owner_allowed_mentions(owner_id=owner_id),
            refused="research thread refused the failure notice",
            failed="failed to post research failure notice",
        )

    # ----- restart resume -----------------------------------------------------------------

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Resumes in-flight research after a restart (runs once)."""
        if self._resume_started:
            return
        self._resume_started = True
        spawn_tracked(coro=self._resume_all(), tasks=self._tasks, name="research-resume")

    async def _resume_all(self) -> None:
        """Resumes every session still `researching` when the process came back up.

        `deep_research_available` gates this exactly as it gates `launch` and `/deep_research`,
        so a missing key is refused here rather than at `genai.Client` inside the resume's own
        try. The switch is flipped over a provider or a cost problem, and a run already open is
        still work with that provider, so off means the bot re-attaches to nothing.

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
                "deep research is unavailable; left in-flight sessions for a later start",
                count=len(sessions),
            )
            return
        for session in sessions:
            self._active_threads.add(session.thread_id)
            spawn_tracked(
                coro=self._resume_one(session=session),
                tasks=self._tasks,
                name=f"research-resume-{session.thread_id}",
            )
        logfire.info("resumed in-flight research sessions", count=len(sessions))

    async def _resume_one(self, *, session: db.PersistentResearchSession) -> None:
        """Resumes one research session, delivering when it settles.

        The status line the run posted before the restart is taken over, so it ends with the
        resume instead of reading `Researching...` forever.
        """
        thread = await self._fetch_thread(thread_id=session.thread_id)
        status = await self._find_prior_status(thread=thread) if thread is not None else None
        # No interaction id means the row was written but the bot restarted before the run id was
        # stored; there is nothing to resume. End the old status line and tell the thread so the
        # owner is not left staring at `Researching...` forever.
        if session.interaction_id is None:
            await self._abandon_resume(session=session, thread=thread, status=status)
            return
        # Give the resumed run the same live reasoning view as a fresh one, on a line of its own
        # when none survived the restart; a fetch miss leaves status None so the streamer's editor
        # no-ops but still drives the stream to a result.
        if thread is not None and status is None:
            status = await self._safe_send(thread=thread, content=RESEARCHING_STATUS)
        streamer = ResearchProgressStreamer(status=status, label=RESEARCH_LABEL)
        try:
            result = await resume_research_stream(
                client=self.interactions_client,
                interaction_id=session.interaction_id,
                streamer=streamer,
            )
        except Exception as exc:
            logfire.error(
                "research resume failed",
                thread_id=session.thread_id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await self._abandon_resume(session=session, thread=thread, status=status)
            return
        if thread is None:
            await self._release(
                thread_id=session.thread_id, phase=_terminal_phase(status=result.status)
            )
            return
        await self._finish(
            thread=thread,
            owner_id=session.owner_id,
            result=result,
            agent=session.agent,
            status=status,
        )

    async def _abandon_resume(
        self,
        session: db.PersistentResearchSession,
        thread: "Thread | None",
        status: Message | None,
    ) -> None:
        """Records a run a restart could not re-attach to as failed, and tells its thread so."""
        await self._release(thread_id=session.thread_id, phase="failed")
        if thread is None:
            return
        await self._finalize_status(status=status, thread=thread, content=RESEARCH_FAILED_STATUS)
        await self._safe_send(
            thread=thread,
            content=f"<@{session.owner_id}> 重啟後沒辦法接回剛剛的研究,麻煩重新發起一次",
            allowed_mentions=owner_allowed_mentions(owner_id=session.owner_id),
        )

    async def _find_prior_status(self, *, thread: "Thread") -> Message | None:
        """Returns the bot's status line from before the restart, or None when none is found.

        Only the process that posted it held that message, so it is read back off the thread: the
        run's first post, still headed `-# Researching...`. Without Read Message History, which a
        launch never checks, Discord answers the read with no messages rather than an error, so
        that gap logs nothing; an empty or failed read only means the resume posts its own line.
        """
        try:
            # The thread's id predates every message in it, so the one default page read after it
            # holds the thread's oldest messages, the run's first post among them.
            async for message in thread.history(after=thread):
                if message.author == self.bot.user and message.content.startswith(
                    RESEARCHING_PREFIX
                ):
                    return message
        except Forbidden:
            # Missing Access, such as View Channel lost on the parent; the id is the whole finding.
            logfire.warn("research thread refused the history read", thread_id=thread.id)
        except Exception as exc:
            # Broad: a resume must not die on a best-effort lookup it can do without.
            logfire.warn(
                "failed to read research thread history",
                thread_id=thread.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        return None

    async def _fetch_thread(self, *, thread_id: int) -> "Thread | None":
        """Returns the thread by id from cache or a REST fetch, or None when gone."""
        cached = self.bot.get_channel(thread_id)
        if isinstance(cached, Thread):
            return cached
        try:
            fetched = await self.bot.fetch_channel(thread_id)
        except NotFound:
            logfire.info("research thread is gone; skipping", thread_id=thread_id)
            return None
        except Forbidden:
            # Missing Access, such as View Channel lost on the parent; the id is the whole finding.
            logfire.warn("research thread refused the fetch; skipping", thread_id=thread_id)
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

    async def _safe_send(  # noqa: PLR0913 -- one thread post plus the two log lines naming it
        self,
        *,
        thread: "Thread",
        content: str,
        embed: Embed | None = None,
        allowed_mentions: "AllowedMentions | None" = None,
        refused: str = "research thread refused a message",
        failed: str = "failed to send research thread message",
    ) -> Message | None:
        """Best-effort `thread.send`, returning the message or None on failure.

        Mentions default to fully suppressed (`AllowedMentions.none()`); a caller that wants the
        owner pinged passes an owner-only policy, so agent-generated content can never mass-ping.
        `refused` is logged, with the id alone, when Discord refuses the post, and `failed` with
        its traceback for any other failure, so each caller's post stays apart in the log.
        """
        mentions = allowed_mentions if allowed_mentions is not None else AllowedMentions.none()
        try:
            # Two calls: none of nextcord's `send` overloads accepts `embed=None`.
            if embed is None:
                return await thread.send(content=content, allowed_mentions=mentions)
            return await thread.send(content=content, embed=embed, allowed_mentions=mentions)
        except Forbidden:
            logfire.warn(refused, thread_id=thread.id)
            return None
        except Exception as exc:
            # Broad: every caller treats a missing message as a degraded outcome, never a failure,
            # and a failed run's cleanup must still run around this post.
            logfire.warn(failed, thread_id=thread.id, error_type=type(exc).__name__, _exc_info=exc)
            return None


def _usage_footer(*, agent: str, input_tokens: int, output_tokens: int) -> str:
    """Builds the usage footer (full model name, tokens, cost) for a result.

    Its `⬆` / `⬇` line is the shape `utils/llm_transcript.py::USAGE_FOOTER_RE` strips back out of
    the bot's own history.

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


def setup(bot: commands.Bot) -> None:
    """Adds the ResearchCogs to the bot."""
    bot.add_cog(ResearchCogs(bot), override=True)
