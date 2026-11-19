"""Bot-addressing predicates shared by the reply pipeline and link-expansion cogs.

A message is addressed to the bot when it is a DM or explicitly mentions the bot in raw text.
Expansion cogs use this predicate in reverse so a link is either expanded or answered, not both;
the check is intentionally coarser than `gen_reply`'s full routing guards.
"""

import re

from nextcord import Message, ClientUser


def bot_mention_pattern(bot_id: int) -> re.Pattern[str]:
    """Returns the compiled pattern matching `<@id>` and `<@!id>` mentions of `bot_id`.

    Args:
        bot_id (int): The bot's own user id.
    """
    return re.compile(pattern=rf"<@!?{re.escape(pattern=str(bot_id))}>")


def has_bot_mention(content: str, bot_user: ClientUser | None) -> bool:
    """Returns whether `content` explicitly mentions `bot_user` (`<@id>` or `<@!id>`).

    Inspects raw content rather than `message.mentions`, which includes reply notifications to
    the bot's own messages.

    Args:
        content (str): Raw message content.
        bot_user (ClientUser | None): Bot user, or None before gateway connect.
    """
    if bot_user is None:
        return False
    return bot_mention_pattern(bot_id=bot_user.id).search(string=content) is not None


def is_addressed_to_bot(message: Message, bot_user: ClientUser | None) -> bool:
    """Returns whether the reply pipeline will treat `message` as directed at `bot_user`.

    A DM needs no mention (every DM reaches `gen_reply`), so it counts as addressed.

    Args:
        message (Message): The incoming message.
        bot_user (ClientUser | None): The bot's own user, or None before the gateway connects.
    """
    if message.guild is None:
        return True
    return has_bot_mention(content=message.content, bot_user=bot_user)
