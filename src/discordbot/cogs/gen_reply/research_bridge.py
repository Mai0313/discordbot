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


def can_launch_research(*, message: Message) -> bool:
    """Whether a research thread can be opened from this message.

    Only a guild text channel can host a nested thread; in a DM or inside an existing thread the
    `<deep-research>` marker is suppressed so the answer model never promises a run that cannot
    actually start (the launch would otherwise return the no-thread path and contradict itself).
    """
    return message.guild is not None and isinstance(message.channel, TextChannel)


def in_active_research_thread(*, bot: commands.Bot, channel_id: int) -> bool:
    """Whether a channel id is a research thread the ResearchCogs cog is actively driving."""
    get_cog = getattr(bot, "get_cog", None)
    cog = get_cog("ResearchCogs") if callable(get_cog) else None
    checker = getattr(cog, "is_research_thread", None)
    return bool(checker(channel_id=channel_id)) if checker is not None else False


async def maybe_launch_research(
    *, bot: commands.Bot, message: Message, anchor: Message | None, brief: str
) -> None:
    """Hands a QA-emitted research brief to the ResearchCogs cog when it is loaded and enabled.

    `anchor` is the bot's own reply message; the research thread hangs off it (more intuitive than
    the user's message), falling back to the user's message inside the cog when it is None.
    """
    get_cog = getattr(bot, "get_cog", None)
    cog = get_cog("ResearchCogs") if callable(get_cog) else None
    launcher = getattr(cog, "launch", None)
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
