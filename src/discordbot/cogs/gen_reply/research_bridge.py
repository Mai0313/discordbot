"""Duck-typed bridge from `gen_reply` to `ResearchCogs` via `bot.get_cog`.

Covers permission checks, active-research-thread detection, and forwarding `<deep-research>`
briefs without importing across cog packages.
"""

import logfire
from nextcord import Message
from nextcord.ext import commands

from discordbot.utils.research_access import research_access_denial


def can_launch_research(message: Message) -> bool:
    """Returns whether a research thread can be opened from `message`.

    Requires a guild `TextChannel` where the bot has permissions to create and send in threads;
    otherwise `<deep-research>` is suppressed from the QA instructions.

    Args:
        message (Message): Triggering Discord message.
    """
    return research_access_denial(channel=message.channel, guild=message.guild) is None


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
    # Broad: best-effort launch must never break a delivered reply; ResearchCogs.launch handles
    # expected outcomes internally.
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
