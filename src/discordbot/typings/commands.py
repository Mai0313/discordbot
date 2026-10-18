"""Installation and interaction contexts declared by every root slash command.

Every root command shares both tuples: omitting `user_install` hides a command from user-installed
accounts, and omitting `private_channel` hides it from group DMs and DMs with other users.

`contexts` must be passed explicitly rather than left to default. Discord stores an unset `contexts`
as null and falls back to the deprecated `dm_permission` boolean, which only enables DMs with the
bot itself and leaves `private_channel` surfaces empty even on a user install.

Stored as tuples so no command can mutate the shared sequence; nextcord accepts any iterable and
builds its own list per command.
"""

from nextcord import IntegrationType, InteractionContextType

INSTALL_CONTEXTS = (IntegrationType.guild_install, IntegrationType.user_install)
INTERACTION_CONTEXTS = (
    InteractionContextType.guild,
    InteractionContextType.bot_dm,
    InteractionContextType.private_channel,
)
