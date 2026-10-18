"""Whether a message is addressed to the bot, shared by the reply pipeline and the link cogs.

`gen_reply` answers a message only when it is a DM or explicitly mentions the bot. An expansion
cog uses the same test in reverse: a message the reply pipeline will answer is left alone, so a
link in one message is either expanded into the channel or answered about, never both. Keeping it
to one is what the rule is for — reading a post is the slow half, and on the path the reply takes
handing the model a URL instead saves nothing, since the proxy fetches it and inlines it.

Across two messages it can still be both: a source that opts into the replied-to scan reads a
link the user only replied to, so a link expanded when it was posted is read again when someone
later replies to that message and mentions the bot. That is a second ask, for what the expansion
deliberately does not show.

The predicate is deliberately coarser than `gen_reply`'s own guards, so a few addressed messages
get neither treatment — one typed inside an active research thread, and one the router sends to a
media route. Both are rare enough to accept rather than couple the two paths together.

Lives in `utils/` rather than on the reply pipeline's input builder because an expansion cog has
no input builder and must not import a peer cog to reach one.
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
    """Returns whether `content` explicitly mentions `bot_user`.

    Matches `<@id>` / `<@!id>` in the raw content rather than reading `message.mentions`: a reply
    notification adds the bot to `mentions`, so replying to one of the bot's own functional posts
    (an expansion card, a downloaded video) would otherwise read as a mention.

    Args:
        content (str): The raw message content.
        bot_user (ClientUser | None): The bot's own user, or None before the gateway connects.
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
