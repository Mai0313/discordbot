"""The installation and interaction contexts every root slash command declares.

Shared rather than spelled out per command because either list only means anything when all of
them carry the same one: a command left out of `user_install` is missing for everyone who added
the bot to their own account, and one left out of `private_channel` is missing from group DMs and
from DMs with anyone but the bot itself.

`contexts` has to be stated rather than left to a default. An unset one is stored as null, which
Discord reads back through the deprecated `dm_permission` boolean, and that flag can express
"DMs with the bot" and nothing else. Measured 2026-08-26 against the live app: with `contexts`
null and `dm_permission` true, a user install worked in servers and in DMs with the bot while
both private-channel surfaces came up empty, which is exactly the line that boolean can draw.

Tuples rather than lists so one command cannot mutate what the other fifteen declare; nextcord
takes any iterable and rebuilds its own list per command.
"""

from nextcord import IntegrationType, InteractionContextType

INSTALL_CONTEXTS = (IntegrationType.guild_install, IntegrationType.user_install)
INTERACTION_CONTEXTS = (
    InteractionContextType.guild,
    InteractionContextType.bot_dm,
    InteractionContextType.private_channel,
)
