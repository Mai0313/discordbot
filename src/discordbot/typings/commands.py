"""Installation and interaction contexts declared by every root slash command.

Pass both tuples explicitly on every root command: leaving `contexts` unset falls back to
Discord's deprecated `dm_permission`, which hides the command in `private_channel` surfaces
even on user installs.
"""

from nextcord import IntegrationType, InteractionContextType

INSTALL_CONTEXTS = (IntegrationType.guild_install, IntegrationType.user_install)
INTERACTION_CONTEXTS = (
    InteractionContextType.guild,
    InteractionContextType.bot_dm,
    InteractionContextType.private_channel,
)
