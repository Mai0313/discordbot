"""The channel permissions a deep-research run needs from the bot.

Shared by the two entry points that can start one, which may not import each other: `gen_reply`'s
marker gate and `/deep_research` both read them off the cached channel for the bot's own member.
"""

from nextcord import Permissions

# See the channel, post the anchor, open a public thread from it, write into that thread, and
# attach the `research.md` the last report message always carries.
RESEARCH_THREAD_PERMISSIONS = Permissions(
    view_channel=True,
    send_messages=True,
    create_public_threads=True,
    send_messages_in_threads=True,
    attach_files=True,
)
