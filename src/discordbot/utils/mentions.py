"""Whether a message mentions the bot.

`gen_reply` answers a guild message only when it explicitly mentions the bot. Lives in
`utils/` rather than on `MessageInputBuilder` so anything outside the reply pipeline can ask
the same question without reaching through a peer cog.
"""

import re

from nextcord import ClientUser


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
