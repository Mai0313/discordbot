"""How `gen_reply` talks to the deep-research cog, without importing it.

Nothing under `cogs/<a>/` may import from `cogs/<b>/`, so every hop across to `ResearchCogs`
goes through the bot instance and duck-typing. Keeping the three hops here rather than inline
means the whole cross-cog surface is one short file: whether a research thread can be opened
from this message, whether the research cog is currently driving this channel, and handing it a
brief the answer model emitted.
"""

import logfire
from nextcord import Message, TextChannel
from nextcord.ext import commands

from discordbot.typings.research import has_research_permissions


def can_launch_research(message: Message) -> bool:
    """Returns whether a research thread can be opened from `message`.

    Requires a guild `TextChannel` where the bot has permissions to create and send in threads;
    otherwise `<deep-research>` is suppressed from the QA instructions.

    Args:
        message (Message): Triggering Discord message.
    """
    if message.guild is None or not isinstance(message.channel, TextChannel):
        return False
    return has_research_permissions(channel=message.channel, member=message.guild.me)


def _research_cog(bot: commands.Bot) -> commands.Cog | None:
    """The loaded ResearchCogs cog, or None when it is not loaded."""
    return bot.get_cog(name="ResearchCogs")


def in_active_research_thread(bot: commands.Bot, channel_id: int) -> bool:
    """Returns whether `channel_id` is an active research thread managed by `ResearchCogs`.

    Args:
        bot (commands.Bot): Discord bot instance holding loaded cogs.
        channel_id (int): Discord channel or thread id to check.
    """
    checker = getattr(_research_cog(bot=bot), "is_research_thread", None)
    return bool(checker(channel_id=channel_id)) if checker is not None else False


async def maybe_launch_research(
    bot: commands.Bot, message: Message, anchor: Message | None, brief: str
) -> None:
    """Forwards a QA-emitted research brief to `ResearchCogs.launch` when the cog is loaded.

    Args:
        bot (commands.Bot): Discord bot instance holding loaded cogs.
        message (Message): Triggering user message.
        anchor (Message | None): Bot reply message to attach the research thread to, or None to
            fall back to `message`.
        brief (str): Self-contained research brief extracted from `<deep-research>`.
    """
    launcher = getattr(_research_cog(bot=bot), "launch", None)
    if launcher is None:
        return
    # Best-effort boundary: the research launch must never break an already-delivered reply, so
    # the except stays broad. ResearchCogs.launch handles its expected outcomes by return value,
    # so anything raising here is unexpected and the emitted brief is lost — markers.py already
    # stripped it from the visible text.
    try:
        await launcher(message=message, anchor=anchor, brief=brief)
    except Exception as exc:
        logfire.warn(
            "deep research launch failed; the emitted brief was dropped",
            message_id=message.id,
            anchor_id=anchor.id if anchor is not None else None,
            error_type=type(exc).__name__,
            _exc_info=exc,
        )
