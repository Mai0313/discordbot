"""Deep-research cog: long-running Gemini managed-agent research delivered in a Discord thread.

A user asks for deep research (the QA answer model emits a `<deep-research>` marker, handed
here by `gen_reply`, or they run `/deep_research`). The bot opens a thread, runs the default
`antigravity-preview-05-2026` agent (fast, cheap, one-shot), and posts the cited report there,
pinging the user. Under the report it offers escalation buttons to `deep-research-preview` /
`deep-research-max`: those enter a plan discussion (Deep Research native collaborative planning,
refine by typing in the thread) before spending the pricier run.

Everything talks DIRECT to Google (`gemini_api_key`, no proxy): the proxy drops `agent_config`,
so `collaborative_planning` only works direct (see `_research/agent.py`). Sessions persist in
`reply.db` so a restart resumes an in-flight research (`store=True` keeps the interaction alive
server-side). The cog never blocks the gateway: agent work runs in tracked background tasks.
"""

from typing import TYPE_CHECKING
import asyncio
from functools import cached_property
import contextlib

from google import genai
from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import Embed, Locale, Interaction, SlashOption
from nextcord.ext import commands

from discordbot.utils.llm import create_text_or_none
from discordbot.typings.llm import LLMConfig
from discordbot.cogs._research import database as db
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.utils.timezone import database_now
from discordbot.utils.reactions import update_reaction
from discordbot.utils.asyncio_locks import KeyedLockManager
from discordbot.utils.model_pricing import get_token_rates
from discordbot.cogs._research.agent import (
    ResearchPlan,
    ResearchResult,
    start_plan,
    refine_plan,
    resume_research,
    start_antigravity,
    start_deep_research,
)
from discordbot.cogs._research.views import PlanApprovalView, ResultEscalationView
from discordbot.cogs._research.prompts import THREAD_TITLE_PROMPT, RESEARCH_SYSTEM_INSTRUCTION
from discordbot.cogs._research.delivery import split_report, deliver_report
from discordbot.cogs._gen_reply.exceptions import extract_friendly_error

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable, Awaitable, Coroutine

    from nextcord import Thread, Message

# How long the modify flow waits for the owner to type their changes in the thread.
MODIFY_WAIT_TIMEOUT_SECONDS = 600.0
# Discord thread names cap at 100 chars; keep margin (a hard-limit safety trim, not length control).
THREAD_NAME_MAX = 90
# Bound the small title-generation side call; on timeout/failure the brief's first line is used.
THREAD_TITLE_TIMEOUT_SECONDS = 15.0
# The bot's `dino` app emoji, reacted onto the source message when deep research is launched so
# the activation reads as distinct from the normal QA pipeline reactions.
DINO_EMOJI = "<:dino:1517560319281594570>"


def _fallback_thread_name(*, brief: str) -> str:
    """Thread-title fallback (the brief's first line) when LLM title generation is unavailable."""
    first_line = next((line.strip() for line in brief.splitlines() if line.strip()), "")
    title = first_line or "深度研究"
    return title[:THREAD_NAME_MAX]


def _tier_label(*, agent: str) -> str:
    """Human label for an agent string."""
    if "max" in agent:
        return "Deep Research Max"
    if "deep-research" in agent:
        return "Deep Research"
    return "Antigravity"


def _terminal_phase(*, status: str) -> db.ResearchPhase:
    """Maps a terminal interaction status onto a stored phase."""
    if status == "completed":
        return "done"
    if status == "cancelled":
        return "cancelled"
    return "failed"


class ResearchCogs(commands.Cog):
    """Owns the deep-research thread lifecycle, slash command, escalation, and restart resume."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config = LLMConfig()
        self.runtime_models = RuntimeModelCatalog()
        # One in-flight research per owner; the lock guards the check-then-create.
        self._owner_locks: KeyedLockManager[int] = KeyedLockManager()
        self._tasks: set[asyncio.Task[None]] = set()
        # Thread ids the cog is actively driving; `gen_reply` checks this so QA does not
        # double-handle a message typed inside a research thread.
        self._active_threads: set[int] = set()
        self._resume_started = False

    @cached_property
    def interactions_client(self) -> genai.Client:
        """The Gemini Interactions client, built lazily on first use.

        DIRECT to Google (`gemini_api_key`, no base_url / proxy): the LiteLLM proxy drops
        `agent_config`, so `collaborative_planning` only works direct. Built inline here (not via
        a `utils/llm.py` factory) so no new factory caller is added. A missing key does not fail
        construction; it surfaces at the first interaction call, which the run loop catches.
        """
        return genai.Client(api_key=self.config.gemini_api_key)

    @cached_property
    def responses_client(self) -> AsyncOpenAI:
        """The LiteLLM-proxy Responses client for small side calls (the thread-title generator).

        Built inline (no `utils/llm.py` factory) per the no-new-factory convention; distinct from
        the direct `interactions_client` since a plain Responses call rides the proxy fine.
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
        """Generates a short thread title from the brief via `fast_model`, best-effort.

        Brevity is steered by the prompt (not a token cap); on timeout or failure the brief's
        first line is used, and the result is trimmed to Discord's hard name limit as a safety net.
        """
        raw = await create_text_or_none(
            client=self.responses_client,
            model=self.runtime_models.fast_model,
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
        """QA-marker entry: opens a thread and starts the default research.

        `message` identifies the owner; `anchor` is the message the thread hangs off. The bot's
        own reply reads more intuitively than the user's message, so the caller passes it; it
        falls back to the user's message when the reply is unavailable.
        """
        if not self.config.deep_research_enabled:
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
        elif outcome == "dm":
            with contextlib.suppress(Exception):
                await message.reply(
                    content="深度研究目前只在伺服器頻道支援(私訊沒有 thread 可以開)"
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
        interaction: Interaction,
        topic: str = SlashOption(
            description="What to research (a clear, self-contained topic).", required=True
        ),
    ) -> None:
        """Opens a research thread for the given topic and starts the default research.

        Args:
            interaction: The slash interaction.
            topic: The research topic / brief.
        """
        if not self.config.deep_research_enabled:
            await interaction.response.send_message(content="深度研究目前停用中", ephemeral=True)
            return
        if interaction.guild is None or interaction.user is None:
            await interaction.response.send_message(
                content="深度研究只在伺服器頻道支援喔", ephemeral=True
            )
            return
        await interaction.response.defer(ephemeral=True)
        # Anchor the thread on a bot message so the same message-based create_thread path is reused.
        anchor = await interaction.channel.send(
            content=f"{interaction.user.mention} 要研究:{topic[:200]}"
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
        """Claims the owner's slot, opens the thread, and spawns the default research.

        Returns `(outcome, thread_or_existing_id)` where outcome is one of
        `started` / `exists` / `dm` / `error`.
        """
        if anchor.guild is None:
            return "dm", None
        async with self._owner_locks.hold(key=owner_id):
            existing = await db.active_thread_for_owner(owner_id=owner_id)
            if existing is not None:
                return "exists", existing
            name = await self._generate_thread_name(brief=brief)
            try:
                thread = await anchor.create_thread(name=name, auto_archive_duration=1440)
            except Exception:
                logfire.warn("failed to create research thread", message_id=anchor.id)
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
            self._run_default_research(
                thread=thread, owner_mention=owner_mention, brief=brief, agent=agent
            )
        )
        return "started", thread.id

    # ----- research runs ------------------------------------------------------------------

    async def _run_default_research(
        self, *, thread: "Thread", owner_mention: str, brief: str, agent: str
    ) -> None:
        """Runs the default Antigravity research and delivers it, offering escalation after."""
        status = await self._safe_send(thread=thread, content="-# Researching... (Antigravity)")
        try:
            interaction_id = await start_antigravity(
                client=self.interactions_client,
                agent=agent,
                brief=brief,
                system_instruction=self._system_instruction(),
            )
            await db.set_interaction(
                thread_id=thread.id,
                interaction_id=interaction_id,
                agent=agent,
                phase="researching",
            )
            result = await resume_research(
                client=self.interactions_client,
                interaction_id=interaction_id,
                on_progress=self._progress_editor(status=status, label="Antigravity"),
            )
            await self._finish(
                thread=thread,
                owner_mention=owner_mention,
                result=result,
                agent=agent,
                status=status,
                offer_escalation=True,
            )
        except Exception as exc:
            logfire.warn("default research failed", thread_id=thread.id, _exc_info=True)
            await self._post_failure(thread=thread, owner_mention=owner_mention, exc=exc)
            await self._finalize_status(
                status=status, thread=thread, content="-# Research failed (Antigravity)"
            )
            await db.set_phase(thread_id=thread.id, phase="failed")
            self._active_threads.discard(thread.id)

    async def _run_deep_research(
        self, *, thread: "Thread", owner_mention: str, agent: str, previous_interaction_id: str
    ) -> None:
        """Approves a planned interaction and runs the full Deep Research report."""
        status = await self._safe_send(
            thread=thread, content=f"-# Researching... ({_tier_label(agent=agent)})"
        )
        try:
            interaction_id = await start_deep_research(
                client=self.interactions_client,
                agent=agent,
                previous_interaction_id=previous_interaction_id,
                system_instruction=self._system_instruction(),
            )
            await db.set_interaction(
                thread_id=thread.id,
                interaction_id=interaction_id,
                agent=agent,
                phase="researching",
            )
            result = await resume_research(
                client=self.interactions_client,
                interaction_id=interaction_id,
                on_progress=self._progress_editor(status=status, label=_tier_label(agent=agent)),
            )
            await self._finish(
                thread=thread,
                owner_mention=owner_mention,
                result=result,
                agent=agent,
                status=status,
                offer_escalation=False,
            )
        except Exception as exc:
            logfire.warn("deep research failed", thread_id=thread.id, _exc_info=True)
            await self._post_failure(thread=thread, owner_mention=owner_mention, exc=exc)
            await self._finalize_status(
                status=status,
                thread=thread,
                content=f"-# Research failed ({_tier_label(agent=agent)})",
            )
            await db.set_phase(thread_id=thread.id, phase="failed")
            self._active_threads.discard(thread.id)

    async def _finish(  # noqa: PLR0913 -- terminal-result inputs plus the opening status message
        self,
        *,
        thread: "Thread",
        owner_mention: str,
        result: ResearchResult,
        agent: str,
        status: "nextcord.Message | None",
        offer_escalation: bool,
    ) -> None:
        """Delivers a terminal result, finalizes the opening status message, and records the phase."""
        tier = _tier_label(agent=agent)
        if not result.ok:
            await self._post_failure(
                thread=thread,
                owner_mention=owner_mention,
                reason=_failure_text(status=result.status),
            )
            await self._finalize_status(
                status=status, thread=thread, content=f"-# Research failed ({tier})"
            )
            await db.set_phase(thread_id=thread.id, phase=_terminal_phase(status=result.status))
            self._active_threads.discard(thread.id)
            return
        await deliver_report(thread=thread, result=result)
        view = (
            ResultEscalationView(
                cog=self,
                owner_id=_owner_id_from_mention(mention=owner_mention),
                max_enabled=self.config.deep_research_max_enabled,
            )
            if offer_escalation and self.config.deep_research_enabled
            else None
        )
        footer = _usage_footer(
            agent=agent, input_tokens=result.input_tokens, output_tokens=result.output_tokens
        )
        await self._finalize_status(
            status=status,
            thread=thread,
            content=f"{owner_mention} Research complete ({tier})\n{footer}",
            view=view,
        )
        await db.set_phase(thread_id=thread.id, phase="done")
        self._active_threads.discard(thread.id)

    async def _finalize_status(
        self,
        *,
        status: "nextcord.Message | None",
        thread: "Thread",
        content: str,
        view: "nextcord.ui.View | None" = None,
    ) -> None:
        """Edits the opening status message to its terminal content (with optional buttons).

        Falls back to a fresh send when there is no status message (a restart resume) or the edit
        fails (e.g. the opening message was deleted).
        """
        if status is not None:
            try:
                await status.edit(content=content, view=view)
                return
            except Exception:
                logfire.warn("failed to finalize research status message", thread_id=thread.id)
        with contextlib.suppress(Exception):
            await thread.send(content=content, view=view)

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
            title="深度研究失敗", description=f"```\n{reason or '未知錯誤'}\n```", color=0xED4245
        )
        if exc is not None:
            embed.set_footer(text=type(exc).__name__)
        with contextlib.suppress(Exception):
            await thread.send(content=f"{owner_mention} ⚠️", embed=embed)

    # ----- escalation (Deep Research) -----------------------------------------------------

    async def on_escalate(
        self, *, interaction: Interaction, view: ResultEscalationView, max_tier: bool
    ) -> None:
        """Escalation button: opens a Deep Research plan discussion."""
        with contextlib.suppress(Exception):
            await interaction.response.edit_message(view=None)
        thread = interaction.channel
        if thread is None:
            return
        existing = await db.active_thread_for_owner(owner_id=view.owner_id)
        if existing is not None and existing != thread.id:
            await self._safe_send(
                thread=thread, content=f"你已經有另一個研究在進行了:<#{existing}>"
            )
            return
        agent = (
            self.runtime_models.deep_research_max_model.name
            if max_tier
            else self.runtime_models.deep_research_model.name
        )
        self._active_threads.add(thread.id)
        await db.set_phase(thread_id=thread.id, phase="planning")
        owner_mention = f"<@{view.owner_id}>"
        self._spawn(self._run_planning(thread=thread, owner_mention=owner_mention, agent=agent))

    async def _run_planning(self, *, thread: "Thread", owner_mention: str, agent: str) -> None:
        """Asks Deep Research for a plan and posts it with approve / modify buttons."""
        session = await db.get_session(thread_id=thread.id)
        if session is None:
            return
        status = await self._safe_send(thread=thread, content="-# Planning... (Deep Research)")
        try:
            plan = await start_plan(
                client=self.interactions_client,
                agent=agent,
                brief=session.brief,
                system_instruction=self._system_instruction(),
            )
            await db.set_interaction(
                thread_id=thread.id,
                interaction_id=plan.interaction_id,
                agent=agent,
                phase="planning",
            )
            await self._post_plan(
                thread=thread, owner_mention=owner_mention, plan=plan, agent=agent, status=status
            )
        except Exception as exc:
            logfire.warn("research planning failed", thread_id=thread.id, _exc_info=True)
            await self._post_failure(thread=thread, owner_mention=owner_mention, exc=exc)
            await db.set_phase(thread_id=thread.id, phase="failed")
            self._active_threads.discard(thread.id)

    async def _post_plan(
        self,
        *,
        thread: "Thread",
        owner_mention: str,
        plan: ResearchPlan,
        agent: str,
        status: "nextcord.Message | None",
    ) -> None:
        """Posts the proposed plan text plus the approve / modify view."""
        if status is not None:
            with contextlib.suppress(Exception):
                await status.delete()
        for chunk in split_report(text=plan.plan_text.strip() or "(沒有收到計畫內容)"):
            await self._safe_send(thread=thread, content=chunk)
        owner_id = _owner_id_from_mention(mention=owner_mention)
        await self._safe_send(
            thread=thread,
            content=f"{owner_mention} 📋 接受就開始研究(會花時間與成本),或點「修改計畫」直接打字告訴我要調整什麼",
            view=PlanApprovalView(
                cog=self, owner_id=owner_id, plan_interaction_id=plan.interaction_id, agent=agent
            ),
        )

    async def on_accept_plan(self, *, interaction: Interaction, view: PlanApprovalView) -> None:
        """Approve button: runs the full Deep Research from the approved plan."""
        with contextlib.suppress(Exception):
            await interaction.response.edit_message(view=None)
        thread = interaction.channel
        if thread is None:
            return
        self._active_threads.add(thread.id)
        await db.set_phase(thread_id=thread.id, phase="researching")
        self._spawn(
            self._run_deep_research(
                thread=thread,
                owner_mention=f"<@{view.owner_id}>",
                agent=view.agent,
                previous_interaction_id=view.plan_interaction_id,
            )
        )

    async def on_modify_plan(self, *, interaction: Interaction, view: PlanApprovalView) -> None:
        """Modify button: waits for the owner to type changes, then re-plans."""
        await interaction.response.send_message(
            content="好,直接在這個 thread 打你想調整的地方,我會重新規劃(10 分鐘內回覆有效)"
        )
        with contextlib.suppress(Exception):
            await interaction.message.edit(view=None)
        thread = interaction.channel
        if thread is None:
            return

        def _is_owner_reply(candidate: "Message") -> bool:
            return (
                candidate.channel.id == thread.id
                and candidate.author.id == view.owner_id
                and not candidate.author.bot
            )

        try:
            reply = await self.bot.wait_for(
                "message", check=_is_owner_reply, timeout=MODIFY_WAIT_TIMEOUT_SECONDS
            )
        except TimeoutError:
            await self._safe_send(thread=thread, content="等太久了,要改再點一次「修改計畫」就好")
            return
        self._spawn(
            self._run_refine(
                thread=thread,
                owner_mention=f"<@{view.owner_id}>",
                agent=view.agent,
                previous_interaction_id=view.plan_interaction_id,
                feedback=reply.content,
            )
        )

    async def _run_refine(
        self,
        *,
        thread: "Thread",
        owner_mention: str,
        agent: str,
        previous_interaction_id: str,
        feedback: str,
    ) -> None:
        """Refines the plan with the owner's feedback and reposts it."""
        status = await self._safe_send(thread=thread, content="-# Re-planning...")
        try:
            plan = await refine_plan(
                client=self.interactions_client,
                agent=agent,
                previous_interaction_id=previous_interaction_id,
                feedback=feedback,
                system_instruction=self._system_instruction(),
            )
            await db.set_interaction(
                thread_id=thread.id,
                interaction_id=plan.interaction_id,
                agent=agent,
                phase="planning",
            )
            await self._post_plan(
                thread=thread, owner_mention=owner_mention, plan=plan, agent=agent, status=status
            )
        except Exception as exc:
            logfire.warn("research re-planning failed", thread_id=thread.id, _exc_info=True)
            await self._post_failure(thread=thread, owner_mention=owner_mention, exc=exc)
            await db.set_phase(thread_id=thread.id, phase="failed")
            self._active_threads.discard(thread.id)

    # ----- restart resume -----------------------------------------------------------------

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Resumes in-flight research after a restart (runs once)."""
        if self._resume_started:
            return
        self._resume_started = True
        self._spawn(self._resume_all())

    async def _resume_all(self) -> None:
        """Re-enters the poll loop for every session left `researching`."""
        sessions = await db.list_resumable()
        for session in sessions:
            self._active_threads.add(session.thread_id)
            self._spawn(self._resume_one(session=session))
        if sessions:
            logfire.info("resumed in-flight research sessions", count=len(sessions))

    async def _resume_one(self, *, session: db.PersistentResearchSession) -> None:
        """Resumes one research session, delivering when it settles."""
        if session.interaction_id is None:
            await db.set_phase(thread_id=session.thread_id, phase="failed")
            self._active_threads.discard(session.thread_id)
            return
        thread = await self._fetch_thread(thread_id=session.thread_id)
        owner_mention = f"<@{session.owner_id}>"
        try:
            result = await resume_research(
                client=self.interactions_client,
                interaction_id=session.interaction_id,
                on_progress=None,
            )
        except Exception:
            logfire.warn("research resume failed", thread_id=session.thread_id, _exc_info=True)
            await db.set_phase(thread_id=session.thread_id, phase="failed")
            self._active_threads.discard(session.thread_id)
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
            status=None,
            offer_escalation="deep-research" not in session.agent,
        )

    async def _fetch_thread(self, *, thread_id: int) -> "Thread | None":
        """Returns the thread by id from cache or a REST fetch, or None when gone."""
        cached = self.bot.get_channel(thread_id)
        if cached is not None:
            return cached
        try:
            fetched = await self.bot.fetch_channel(thread_id)
        except Exception:
            return None
        return fetched

    # ----- helpers ------------------------------------------------------------------------

    def _progress_editor(
        self, *, status: "nextcord.Message | None", label: str
    ) -> "Callable[[str | None, float], Awaitable[None]]":
        """Builds an on-progress callback that edits the status message with elapsed time."""

        async def _on_progress(thought: str | None, elapsed: float) -> None:
            if status is None:
                return
            mins, secs = divmod(int(elapsed), 60)
            line = f"-# Researching... ({label}, {mins}m{secs:02d}s)"
            if thought:
                line = f"{line}\n-# {thought[:200]}"
            with contextlib.suppress(Exception):
                await status.edit(content=line)

        return _on_progress

    async def _safe_send(
        self, *, thread: "Thread", content: str, view: "nextcord.ui.View | None" = None
    ) -> "nextcord.Message | None":
        """Best-effort `thread.send`, returning the message or None on failure."""
        try:
            if view is not None:
                return await thread.send(content=content, view=view)
            return await thread.send(content=content)
        except Exception:
            logfire.warn("failed to send research thread message", thread_id=thread.id)
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


def setup(bot: commands.Bot) -> None:
    """Adds the ResearchCogs to the bot."""
    bot.add_cog(ResearchCogs(bot), override=True)
