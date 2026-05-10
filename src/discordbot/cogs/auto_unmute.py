from typing import cast
from datetime import UTC, datetime
from functools import cached_property

from openai import AsyncOpenAI
import logfire
import nextcord
from nextcord import User, Guild, Member, Message, AuditLogAction
from nextcord.ext import commands
from openai.types.responses.response_input_param import ResponseInputParam, EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs._auto_unmute.prompts import UNMUTE_PROMPT


class AutoUnmuteCogs(commands.Cog):
    """Releases the bot from member timeouts and posts an AI reaction.

    Per-guild we remember the channel ID where a human last spoke; that's
    where the AI's post-timeout reply lands. We do not track a per-moderator
    "current channel" — Discord's audit log entry for a timeout does not carry
    a channel, and using `last_active_channel` keeps the dict O(guilds).

    Attributes:
        bot: The Discord bot instance that owns this cog.
        config: The LLM client configuration loaded for reply generation.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initialises the AutoUnmuteCogs instance.

        Args:
            bot: The Discord bot instance.
        """
        self.bot = bot
        self.config = LLMConfig()
        self._last_active_channel: dict[int, int] = {}

    @cached_property
    def client(self) -> AsyncOpenAI:
        """The cached AsyncOpenAI client instance."""
        client = AsyncOpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return client

    @property
    def model(self) -> ModelSettings:
        """Model used to generate the post-timeout reaction.

        Reuses the fast model — this path runs off the user request path, but a
        heavier model adds cost without buying anything for one short message.
        """
        return ModelSettings(name="gemini-flash-latest", effort="none")

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        """Tracks the last channel a human spoke in, per guild."""
        if message.guild is None or message.author.bot:
            return
        self._last_active_channel[message.guild.id] = message.channel.id

    @commands.Cog.listener()
    async def on_member_update(self, before: Member, after: Member) -> None:
        """Detects a transition into timeout for the bot itself and self-unmutes."""
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
        except Exception:
            logfire.error("auto-unmute flow failed", _exc_info=True)

    async def _handle_self_timeout(self, member: Member, until: datetime) -> None:
        """Looks up who timed us out, releases the timeout, and posts an AI reply.

        We still post a reply when the audit lookup fails (Forbidden, missing
        entry, or timed-out bots being denied this endpoint per discord-api-docs
        #6847) — the AI just gripes at an anonymous moderator instead of pinging.
        """
        moderator, reason = await self._lookup_audit(guild=member.guild)
        try:
            await member.edit(timeout=None, reason="auto-unmute")
        except Exception:
            logfire.warn(
                f"failed to clear self timeout in {member.guild.name} (missing moderate_members?)",
                _exc_info=True,
            )
        ai_reply = await self._generate_reply(
            guild_name=member.guild.name, moderator=moderator, reason=reason, until=until
        )
        if not ai_reply:
            return
        channel = self._resolve_channel(guild=member.guild)
        if channel is None:
            logfire.warn(f"no sendable channel for auto-unmute reply in {member.guild.name}")
            return
        try:
            await channel.send(content=ai_reply)
        except Exception:
            logfire.warn(
                f"failed to send auto-unmute reply in {member.guild.name} "
                "(timed-out bot's send may be blocked)",
                _exc_info=True,
            )

    async def _lookup_audit(self, guild: Guild) -> tuple[Member | User | None, str | None]:
        """Walks recent member_update audit entries to find the timeout that hit us.

        We scan a small window because nextcord's `AuditLogAction.member_update`
        bucket also covers nickname / mute / deafen edits — only the entry whose
        diff carries `communication_disabled_until` is the one we want.
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
        except nextcord.Forbidden:
            logfire.warn(f"missing view_audit_log permission in {guild.name}")
        return None, None

    def _resolve_channel(self, guild: Guild) -> nextcord.abc.Messageable | None:
        """Picks a target channel: last active channel, then system channel."""
        channel_id = self._last_active_channel.get(guild.id)
        if channel_id is not None:
            channel = guild.get_channel(channel_id)
            if isinstance(channel, nextcord.abc.Messageable):
                return channel
        if isinstance(guild.system_channel, nextcord.abc.Messageable):
            return guild.system_channel
        return None

    async def _generate_reply(
        self, guild_name: str, moderator: Member | User | None, reason: str | None, until: datetime
    ) -> str:
        """Builds a single user-role prompt and asks the model for one Discord reply."""
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
        message_list: list[EasyInputMessageParam] = [
            EasyInputMessageParam(role="user", content=user_text)
        ]
        responses = await self.client.responses.create(
            model=self.model.name,
            instructions=UNMUTE_PROMPT,
            input=cast("ResponseInputParam", message_list),
            reasoning=self.model.reasoning,
            service_tier="auto",
            extra_headers={"x-litellm-end-user-id": "auto-unmute"},
            extra_body={"mock_testing_fallbacks": False},
        )
        return (responses.output_text or "").strip()


def setup(bot: commands.Bot) -> None:
    """Adds the AutoUnmuteCogs to the bot.

    Args:
        bot: The Discord bot instance.
    """
    bot.add_cog(AutoUnmuteCogs(bot), override=True)
