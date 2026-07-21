"""Whether a message is addressed to the bot, shared by the reply pipeline and the link cogs.

`gen_reply` answers a message only when it is a DM or explicitly mentions the bot. The link
expansion cogs (`parse_threads`, `parse_douyin`) use the same test in reverse: a message the
reply pipeline will answer is left alone, so a link is either expanded into the channel or
answered about, never both.

Lives in `utils/` rather than on `MessageInputBuilder` because the expansion cogs have no
input builder and must not import a peer cog to reach one.
"""

import re

from nextcord import Message, ClientUser


def has_bot_mention(*, content: str, bot_user: ClientUser | None) -> bool:
    """Whether the message body explicitly mentions the bot.

    Matches `<@id>` / `<@!id>` in the raw content rather than reading `message.mentions`:
    a reply notification adds the bot to `mentions`, so replying to one of the bot's own
    functional posts (a Threads embed, a downloaded video) would otherwise read as a mention.

    Args:
        content: The raw message content.
        bot_user: The bot's own user, or None before the gateway connects.

    Returns:
        True when the content mentions the bot.
    """
    if bot_user is None:
        return False
    bot_id = re.escape(str(bot_user.id))
    return re.search(rf"<@!?{bot_id}>", content) is not None


def is_addressed_to_bot(*, message: Message, bot_user: ClientUser | None) -> bool:
    """Whether the reply pipeline will treat this message as directed at the bot.

    A DM needs no mention (every DM reaches `gen_reply`), so it counts as addressed.

    Args:
        message: The incoming message.
        bot_user: The bot's own user, or None before the gateway connects.

    Returns:
        True for a DM, or for a guild message that mentions the bot.
    """
    if message.guild is None:
        return True
    return has_bot_mention(content=message.content, bot_user=bot_user)
