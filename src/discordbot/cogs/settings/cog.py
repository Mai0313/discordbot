"""Slash command for per-server bot feature settings."""

import logfire
import nextcord
from nextcord import Locale, Interaction
from nextcord.ext import commands

from discordbot.typings.commands import INSTALL_CONTEXTS, INTERACTION_CONTEXTS
from discordbot.cogs.settings.views import (
    SETTINGS_TITLE,
    ServerSettingsView,
    build_error_embed,
    build_refusal_embed,
    build_settings_embed,
)
from discordbot.utils.sqlite_config import ensure_database_upgraded
from discordbot.utils.server_settings import get_guild_auto_expand


class SettingsCogs(commands.Cog):
    """Provides the `/settings` per-server configuration panel for server administrators.

    Attributes:
        bot (commands.Bot): The Discord bot instance that owns this cog.
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initializes the settings cog.

        Args:
            bot (commands.Bot): The Discord bot instance.
        """
        self.bot = bot

    @nextcord.slash_command(
        name="settings",
        description="server admin only: configure bot features for this server.",
        name_localizations={Locale.zh_TW: "設定", Locale.ja: "設定"},
        description_localizations={
            Locale.zh_TW: "server admin 限定：調整這個伺服器的 bot 功能設定",
            Locale.ja: "server admin 専用：このサーバーのボット機能設定を変更します。",
        },
        nsfw=False,
        integration_types=INSTALL_CONTEXTS,
        contexts=INTERACTION_CONTEXTS,
    )
    async def settings(self, interaction: Interaction[commands.Bot]) -> None:
        """Opens the ephemeral per-server settings toggle panel for server administrators.

        Args:
            interaction (Interaction[commands.Bot]): Discord command interaction.
        """
        guild = interaction.guild
        if guild is None:
            # User-installed commands can fire in unjoined guilds (`guild_id` set, `guild` None).
            description = (
                "我沒有被加進這個伺服器，這個指令只能在我所在的伺服器裡使用。"
                if interaction.guild_id is not None
                else "`/settings` 只適用於伺服器，私訊對話一律使用預設設定。"
            )
            await interaction.response.send_message(
                embed=build_refusal_embed(title=SETTINGS_TITLE, description=description),
                ephemeral=True,
            )
            return
        if not interaction.permissions.administrator:
            await interaction.response.send_message(
                embed=build_refusal_embed(
                    title="權限不足",
                    description="只有伺服器管理員（Administrator）才能修改這個伺服器的設定。",
                    is_permission_error=True,
                ),
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)
        try:
            state = await get_guild_auto_expand(guild_id=guild.id)
        except Exception as exc:
            # Broad on purpose: command was already deferred, so report failure cleanly instead of
            # letting the generic error handler send a second reply or leak internals.
            logfire.error(
                "Failed to load server settings",
                guild_id=guild.id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            await interaction.edit_original_message(
                embed=build_error_embed(description="無法讀取伺服器設定，請稍後再試。")
            )
            return

        view = ServerSettingsView(guild_id=guild.id, guild_name=guild.name, state=state)
        await interaction.edit_original_message(
            embed=build_settings_embed(guild_name=guild.name, state=state), view=view
        )
        view.bind_origin(interaction=interaction)


def setup(bot: commands.Bot) -> None:
    """Upgrades `settings.db` and adds `SettingsCogs` to the bot.

    Args:
        bot (commands.Bot): The Discord bot instance.
    """
    ensure_database_upgraded(section="settings")
    bot.add_cog(SettingsCogs(bot), override=True)
