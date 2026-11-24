"""Ephemeral embed builders and interactive toggle view for `/settings`."""

from typing import Final
import contextlib

import logfire
from nextcord import Embed, ButtonStyle, Interaction, HTTPException
from nextcord.ui import Button
from nextcord.ext import commands

from discordbot.typings.colors import DISCORD_RED, NEUTRAL_BLUE, DISCORD_YELLOW
from discordbot.typings.config import AutoExpandPlatform, load_bot_config
from discordbot.utils.logged_ui import LoggedView
from discordbot.utils.server_settings import (
    AUTO_EXPAND_PLATFORMS,
    GuildAutoExpandState,
    set_auto_expand_override,
    reset_auto_expand_overrides,
)

SETTINGS_VIEW_TIMEOUT_SECONDS = 180

SETTINGS_TITLE: Final[str] = "⚙️ 伺服器設定 — 連結自動展開"

_PLATFORM_LABELS: Final[dict[AutoExpandPlatform, str]] = {
    "twitter": "Twitter / X",
    "threads": "Threads",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "douyin": "Douyin",
}

_BUTTON_LABELS: Final[dict[AutoExpandPlatform, str]] = {
    "twitter": "Twitter",
    "threads": "Threads",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "douyin": "Douyin",
}


def _platform_emoji(platform: AutoExpandPlatform) -> str:
    """Returns the configured display emoji for `platform` from `config.toml`."""
    return str(getattr(load_bot_config().global_config.emojis, platform))


def build_settings_embed(guild_name: str, state: GuildAutoExpandState) -> Embed:
    """Returns the ephemeral embed summarizing a guild's link auto-expansion switches.

    Args:
        guild_name (str): Display name of the guild being configured.
        state (GuildAutoExpandState): Effective defaults and overrides for the guild.
    """
    lines: list[str] = []
    for platform in AUTO_EXPAND_PLATFORMS:
        emoji = _platform_emoji(platform=platform)
        label = _PLATFORM_LABELS[platform]
        enabled = state.effective(platform=platform)
        status = "開啟" if enabled else "關閉"
        source = "已自訂" if state.is_overridden(platform=platform) else "預設"
        lines.append(f"{emoji} **{label}** — {status}（{source}）")
    embed = Embed(title=SETTINGS_TITLE, description="\n".join(lines), color=NEUTRAL_BLUE)
    footer = "點擊下方按鈕即時切換 · 未自訂項目會跟著預設設定走"
    if guild_name:
        footer = f"{guild_name} · {footer}"
    embed.set_footer(text=footer)
    return embed


def build_refusal_embed(title: str, description: str, is_permission_error: bool = False) -> Embed:
    """Returns an ephemeral refusal embed for `/settings` context or permission checks.

    Args:
        title (str): Embed heading.
        description (str): Reason the command or button action was refused.
        is_permission_error (bool): Uses red for permission refusals and yellow for context checks.
    """
    return Embed(
        title=title,
        description=description,
        color=DISCORD_RED if is_permission_error else DISCORD_YELLOW,
    )


def build_error_embed(description: str) -> Embed:
    """Returns an ephemeral red error embed when reading or saving settings fails.

    Args:
        description (str): User-facing failure message.
    """
    return Embed(title=SETTINGS_TITLE, description=description, color=DISCORD_RED)


class _PlatformToggleButton(Button["ServerSettingsView"]):
    """Toggle button for one platform's auto-expansion setting."""

    def __init__(self, platform: AutoExpandPlatform, enabled: bool) -> None:
        """Initializes the toggle button for `platform`.

        Args:
            platform (AutoExpandPlatform): Platform controlled by this button.
            enabled (bool): Current effective enabled state.
        """
        super().__init__(
            label=_BUTTON_LABELS[platform],
            emoji=_platform_emoji(platform=platform),
            style=ButtonStyle.success if enabled else ButtonStyle.secondary,
            row=0,
        )
        self.platform = platform

    async def callback(self, interaction: Interaction[commands.Bot]) -> None:
        """Forwards the click to the parent `ServerSettingsView`."""
        if self.view is None:
            return
        await self.view.handle_toggle(interaction=interaction, platform=self.platform)


class _ResetDefaultsButton(Button["ServerSettingsView"]):
    """Button clearing every auto-expand override for the guild."""

    def __init__(self, has_overrides: bool) -> None:
        """Initializes the reset button, disabled when every setting already matches defaults.

        Args:
            has_overrides (bool): Whether at least one override is stored for the guild.
        """
        super().__init__(
            label="恢復預設",
            emoji="↺",
            style=ButtonStyle.secondary,
            disabled=not has_overrides,
            row=1,
        )

    async def callback(self, interaction: Interaction[commands.Bot]) -> None:
        """Forwards the click to the parent `ServerSettingsView`."""
        if self.view is None:
            return
        await self.view.handle_reset(interaction=interaction)


class ServerSettingsView(LoggedView):
    """Ephemeral toggle panel for per-server link auto-expansion overrides.

    Attributes:
        guild_id (int): Discord guild id controlled by this view.
        guild_name (str): Guild display name shown in the embed footer.
        state (GuildAutoExpandState): Latest known defaults and overrides for the guild.
    """

    def __init__(self, guild_id: int, guild_name: str, state: GuildAutoExpandState) -> None:
        """Initializes the view with five platform toggles and a reset button.

        Args:
            guild_id (int): Discord guild id.
            guild_name (str): Guild display name.
            state (GuildAutoExpandState): Current defaults and overrides for the guild.
        """
        super().__init__(timeout=SETTINGS_VIEW_TIMEOUT_SECONDS)
        self.guild_id = guild_id
        self.guild_name = guild_name
        self.state = state
        self._origin: Interaction[commands.Bot] | None = None
        self._platform_buttons: dict[AutoExpandPlatform, _PlatformToggleButton] = {}
        for platform in AUTO_EXPAND_PLATFORMS:
            button = _PlatformToggleButton(
                platform=platform, enabled=state.effective(platform=platform)
            )
            self._platform_buttons[platform] = button
            self.add_item(item=button)
        self._reset_button = _ResetDefaultsButton(has_overrides=state.override_count() > 0)
        self.add_item(item=self._reset_button)

    def bind_origin(self, interaction: Interaction[commands.Bot]) -> None:
        """Records the newest interaction handle used by `on_timeout` to disable buttons.

        Args:
            interaction (Interaction[commands.Bot]): Opening command or newest button interaction.
        """
        self._origin = interaction

    def _sync_buttons(self) -> None:
        """Updates button colors and reset availability to match `self.state`."""
        for platform, button in self._platform_buttons.items():
            enabled = self.state.effective(platform=platform)
            button.style = ButtonStyle.success if enabled else ButtonStyle.secondary
        self._reset_button.disabled = self.state.override_count() == 0

    async def _require_admin(self, interaction: Interaction[commands.Bot]) -> bool:
        """Checks that the clicker still holds administrator permission in `self.guild_id`."""
        if interaction.guild_id == self.guild_id and interaction.permissions.administrator:
            return True
        await interaction.response.send_message(
            embed=build_refusal_embed(
                title="權限不足",
                description="只有伺服器管理員（Administrator）才能修改這個伺服器的設定。",
                is_permission_error=True,
            ),
            ephemeral=True,
        )
        return False

    async def handle_toggle(
        self, interaction: Interaction[commands.Bot], platform: AutoExpandPlatform
    ) -> None:
        """Persists a single platform toggle and refreshes the panel.

        Args:
            interaction (Interaction[commands.Bot]): Discord button interaction.
            platform (AutoExpandPlatform): Platform button that was pressed.
        """
        if not await self._require_admin(interaction=interaction):
            return
        await interaction.response.defer()
        self.bind_origin(interaction=interaction)
        next_enabled = not self.state.effective(platform=platform)
        try:
            self.state = await set_auto_expand_override(
                guild_id=self.guild_id, platform=platform, enabled=next_enabled
            )
        except Exception as exc:
            # Broad on purpose: button-callback boundary after defer; report failure ephemerally
            # instead of leaving the interaction silently stuck.
            logfire.error(
                "Failed to update server auto-expand setting",
                guild_id=self.guild_id,
                platform=platform,
                enabled=next_enabled,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            with contextlib.suppress(HTTPException):
                await interaction.followup.send(
                    embed=build_error_embed(description="更新伺服器設定失敗，請稍後再試。"),
                    ephemeral=True,
                )
            return
        self._sync_buttons()
        await interaction.edit_original_message(
            embed=build_settings_embed(guild_name=self.guild_name, state=self.state), view=self
        )

    async def handle_reset(self, interaction: Interaction[commands.Bot]) -> None:
        """Clears every auto-expand override for this guild and refreshes the panel.

        Args:
            interaction (Interaction[commands.Bot]): Discord button interaction.
        """
        if not await self._require_admin(interaction=interaction):
            return
        await interaction.response.defer()
        self.bind_origin(interaction=interaction)
        try:
            self.state = await reset_auto_expand_overrides(guild_id=self.guild_id)
        except Exception as exc:
            # Broad on purpose: button-callback boundary after defer; report failure ephemerally
            # instead of leaving the interaction silently stuck.
            logfire.error(
                "Failed to reset server auto-expand settings",
                guild_id=self.guild_id,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
            with contextlib.suppress(HTTPException):
                await interaction.followup.send(
                    embed=build_error_embed(description="恢復預設設定失敗，請稍後再試。"),
                    ephemeral=True,
                )
            return
        self._sync_buttons()
        await interaction.edit_original_message(
            embed=build_settings_embed(guild_name=self.guild_name, state=self.state), view=self
        )

    async def on_timeout(self) -> None:
        """Disables every button once the panel goes idle, ignoring cleanup errors."""
        if self._origin is None:
            return
        for child in self.children:
            if isinstance(child, Button):
                child.disabled = True
        # Broad on purpose: inert cleanup when the ephemeral panel may already be dismissed.
        with contextlib.suppress(Exception):
            await self._origin.edit_original_message(view=self)
