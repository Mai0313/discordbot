"""Discord surface for the bot releasing itself from a moderator timeout.

Registers two listeners and no commands. `on_message` remembers, per guild, the channel a human
last spoke in; `on_member_update` fires when the bot itself gains a future-dated timeout, clears
it with a silent PATCH, and posts one short model-written reply at whoever applied it. That reply
is the only thing the channel sees.

How much of the flow works depends on permissions, and none of them is required. `moderate_members`
is what lets the bot clear its own timeout; without it the failure is logged and the reply still
goes out. `view_audit_log` is what names the moderator; without it the model gripes at an anonymous
one instead, a branch `prompts.py::UNMUTE_PROMPT` keys off the wording built here. There is no
kill-switch: the reply rides `utils/llm.py::create_text_or_none` on `fast_model`, so a failed or
slow call posts nothing rather than degrading to a template line.
"""

from datetime import UTC, datetime
from functools import cached_property

from openai import AsyncOpenAI
import logfire
from nextcord import User, Guild, Member, Message, Forbidden, HTTPException, AuditLogAction
from nextcord.abc import Messageable
from nextcord.ext import commands

from discordbot.utils.llm import create_text_or_none
from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import RuntimeModelCatalog
from discordbot.cogs.auto_unmute.prompts import UNMUTE_PROMPT

# Auto-unmute replies are off the critical path; bound the call so a hung provider never
# leaves the best-effort post-timeout reply pending forever.
AUTO_UNMUTE_AI_TIMEOUT_SECONDS = 10.0


class AutoUnmuteCogs(commands.Cog):
    """Releases the bot from member timeouts and posts an AI reaction.

    Per-guild we remember the channel id where a human last spoke; that is where the post-timeout
    reply lands. There is no per-moderator "current channel" to use instead: Discord's audit log
    entry for a timeout carries no channel, and keying on the last active one keeps the dict
    O(guilds).

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: The LLM client configuration loaded for reply generation.
        runtime_models: The catalog supplying the model tier the reply is generated on.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Builds the cog with its own LLM config, model catalog, and empty per-guild channel map.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()
        self.runtime_models = RuntimeModelCatalog()
        self._last_active_channel: dict[int, int] = {}

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The proxy-backed Responses client for auto-unmute replies, built lazily on first use.

        Built inline rather than through a `utils/llm.py` factory, per the no-new-factory
        convention, and cached so one client serves every timeout instead of one per event.

        Returns:
            A client pointed at the LiteLLM proxy.
        """
        return AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Records the channel a human last spoke in for this guild.

        DMs have no guild to key on, and bot traffic is no sign of where the humans are, so both
        leave the map untouched.

        Args:
            message (Message): The message the gateway dispatched.
        """
        if message.guild is None or message.author.bot:
            return
        self._last_active_channel[message.guild.id] = message.channel.id

    @commands.Cog.listener()
    async def on_member_update(self, before: Member, after: Member) -> None:
        """Starts the release flow when the bot itself gains a future-dated timeout.

        Another member's update, or one that leaves `communication_disabled_until` unchanged, is
        ignored, so the flow runs once per timeout.

        Args:
            before (Member): The member snapshot before Discord applied the update.
            after (Member): The member snapshot after Discord applied the update.
        """
        if not self.bot.user or after.id != self.bot.user.id:
            return
        before_until = before.communication_disabled_until
        after_until = after.communication_disabled_until
        if before_until == after_until:
            return
        # Only react to transitions *into* a future-dated timeout. The PATCH
        # we issue below to clear the timeout will fire this listener again
        # with after_until=None, which falls through to the early return.
        if not after_until or after_until <= datetime.now(tz=UTC):
            return
        try:
            await self._handle_self_timeout(member=after, until=after_until)
        except Exception as exc:
            # Stays broad: nextcord's dispatcher swallows anything escaping a listener,
            # so this is the last place the residual failure can be reported.
            logfire.error(
                "auto-unmute flow failed",
                guild_id=after.guild.id,
                guild_name=after.guild.name,
                until=after_until.isoformat(),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    async def _handle_self_timeout(self, member: Member, until: datetime) -> None:
        """Looks up who timed us out, releases the timeout, and posts an AI reply.

        The audit lookup runs first and every step after it is best-effort, so a partial failure
        still delivers what it can. We still post a reply when that lookup fails (Forbidden,
        missing entry, or timed-out bots being denied this endpoint per discord-api-docs #6847),
        the AI just gripes at an anonymous moderator instead of pinging. A release that fails is
        logged and the reply goes out anyway; an empty model result or a guild with no postable
        channel ends the flow silently.

        Args:
            member (Member): The bot's own member object, already carrying the timeout.
            until (datetime): When the timeout would have expired; the reply quotes what is left
                of it.
        """
        moderator, reason = await self._lookup_audit(guild=member.guild)
        try:
            await member.edit(timeout=None, reason="auto-unmute")
        except HTTPException as exc:
            # Forbidden subclasses HTTPException, so the missing-moderate_members case lands here.
            logfire.error(
                "failed to clear self timeout (missing moderate_members?)",
                guild_id=member.guild.id,
                guild_name=member.guild.name,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        ai_reply = await self._generate_reply(
            guild_name=member.guild.name, moderator=moderator, reason=reason, until=until
        )
        if not ai_reply:
            return
        channel = self._resolve_channel(guild=member.guild)
        if channel is None:
            logfire.info("no sendable channel for auto-unmute reply", guild_id=member.guild.id)
            return
        try:
            await channel.send(content=ai_reply)
        except HTTPException as exc:
            # A timed-out bot's send is denied, which arrives as Forbidden (an HTTPException).
            logfire.warn(
                "failed to send auto-unmute reply",
                guild_id=member.guild.id,
                channel_id=getattr(channel, "id", None),
                error_type=type(exc).__name__,
                _exc_info=exc,
            )

    async def _lookup_audit(self, guild: Guild) -> tuple[Member | User | None, str | None]:
        """Walks recent member_update audit entries to find the timeout that hit us.

        We scan a small window because nextcord's `AuditLogAction.member_update`
        bucket also covers nickname / mute / deafen edits. Only the entry whose
        diff carries `communication_disabled_until` is the one we want. Every failure degrades to
        a pair of Nones rather than raising, so a missing `view_audit_log` costs the mention and
        nothing else.

        Args:
            guild (Guild): The guild whose audit log is read.

        Returns:
            The moderator who applied the timeout and the reason they gave; both are None when no
            matching entry could be read, and the reason alone is None when none was given.
        """
        bot_user = self.bot.user
        if bot_user is None:
            return None, None
        try:
            async for entry in guild.audit_logs(action=AuditLogAction.member_update, limit=5):
                if not entry.target or entry.target.id != bot_user.id:
                    continue
                if not hasattr(entry.changes.after, "communication_disabled_until"):
                    continue
                return entry.user, entry.reason
        except Forbidden as exc:
            logfire.warn("missing view_audit_log permission", guild_id=guild.id, _exc_info=exc)
        except HTTPException as exc:
            logfire.warn(
                "audit log lookup failed",
                guild_id=guild.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
        return None, None

    def _resolve_channel(self, guild: Guild) -> Messageable | None:
        """Picks a target channel: last active channel, then system channel.

        Both candidates are checked against `Messageable`, since `get_channel` also hands back
        categories and other channels nothing can be sent to; a tracked channel that fails the
        check falls through to the system channel.

        Args:
            guild (Guild): The guild the reply is posted in.

        Returns:
            A channel the reply can be sent to, or None when neither candidate resolves.
        """
        channel_id = self._last_active_channel.get(guild.id)
        if channel_id is not None:
            channel = guild.get_channel(channel_id)
            if isinstance(channel, Messageable):
                return channel
        if isinstance(guild.system_channel, Messageable):
            return guild.system_channel
        return None

    async def _generate_reply(
        self, guild_name: str, moderator: Member | User | None, reason: str | None, until: datetime
    ) -> str | None:
        """Builds a single user-role prompt and asks the model for one Discord reply.

        The moderator line carries the raw id because `UNMUTE_PROMPT` mentions them with it, and an
        unknown moderator is spelled out in the exact wording that prompt keys its no-mention
        branch off, so neither can be reworded on its own. Remaining minutes floor at zero, so a
        timeout that expired while the flow ran does not read as negative.

        Args:
            guild_name (str): Name of the guild, handed to the model as context.
            moderator (Member | User | None): Who applied the timeout, or None when the audit
                lookup found nothing.
            reason (str | None): The audit reason, or None when the moderator gave none.
            until (datetime): When the timeout would have expired; the quoted duration is derived
                from it.

        Returns:
            The model's reply line, "" when the turn produced no text, or None when the call
            failed or timed out; the caller posts nothing for the last two alike.
        """
        remaining = until - datetime.now(tz=UTC)
        minutes = max(int(remaining.total_seconds()) // 60, 0)
        readable_reason = reason if reason else "(no reason given)"
        if moderator is None:
            moderator_line = "Moderator: unknown (audit log unavailable)"
        else:
            moderator_line = (
                f"Moderator: {moderator.display_name} ({moderator.name}) [id: {moderator.id}]"
            )
        user_text = (
            f"Guild: {guild_name}\n"
            f"{moderator_line}\n"
            f"Timeout duration: {minutes} minute(s)\n"
            f"Reason: {readable_reason}"
        )
        return await create_text_or_none(
            client=self.client,
            model=self.runtime_models.fast_model,
            instructions=UNMUTE_PROMPT,
            user_text=user_text,
            end_user_id="auto-unmute",
            timeout_seconds=AUTO_UNMUTE_AI_TIMEOUT_SECONDS,
        )


def setup(bot: commands.Bot) -> None:
    """Adds the AutoUnmuteCogs to the bot.

    Args:
        bot (commands.Bot): The bot the cog is added to.
    """
    bot.add_cog(AutoUnmuteCogs(bot), override=True)
