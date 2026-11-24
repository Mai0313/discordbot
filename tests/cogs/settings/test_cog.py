"""Tests for `/settings` slash command authorization, context checks, and setup."""

import pytest
from nextcord import Embed

from discordbot.cogs.settings import cog as settings_cog_module
from discordbot.typings.colors import DISCORD_RED, NEUTRAL_BLUE, DISCORD_YELLOW
from discordbot.cogs.settings.cog import SettingsCogs, setup
from discordbot.cogs.settings.views import ServerSettingsView
from discordbot.utils.server_settings import set_auto_expand_override

from tests.helpers.casting import as_interaction
from tests.helpers.discord_mocks import FakeInteraction, stub_bot
from tests.helpers.logfire_capture import capture_logs


async def test_settings_refuses_dm_with_ephemeral_notice() -> None:
    """In DMs (`guild` and `guild_id` both None), `/settings` refuses ephemerally."""
    cog = SettingsCogs(bot=stub_bot())
    interaction = FakeInteraction(in_guild=False, administrator=True)

    await cog.settings.callback(cog, as_interaction(fake=interaction))

    assert interaction.response.deferred is False
    assert len(interaction.response.sent) == 1
    sent = interaction.response.sent[0]
    assert sent.get("ephemeral") is True
    embed = sent.get("embed")
    assert isinstance(embed, Embed)
    assert embed.colour is not None
    assert embed.colour.value == DISCORD_YELLOW
    assert "私訊" in (embed.description or "")


async def test_settings_refuses_unjoined_guild_with_distinct_notice() -> None:
    """In a user-installed unjoined guild (`guild` None, `guild_id` set), `/settings` explains why."""
    cog = SettingsCogs(bot=stub_bot())
    interaction = FakeInteraction(in_guild=False, administrator=True)
    interaction.guild_id = 555

    await cog.settings.callback(cog, as_interaction(fake=interaction))

    assert len(interaction.response.sent) == 1
    sent = interaction.response.sent[0]
    assert sent.get("ephemeral") is True
    embed = sent.get("embed")
    assert isinstance(embed, Embed)
    assert embed.colour is not None
    assert embed.colour.value == DISCORD_YELLOW
    assert "沒有被加進這個伺服器" in (embed.description or "")


async def test_settings_refuses_non_admin_member() -> None:
    """Members without Discord's Administrator permission receive an ephemeral red refusal."""
    cog = SettingsCogs(bot=stub_bot())
    interaction = FakeInteraction(in_guild=True, guild_id=100, administrator=False)

    await cog.settings.callback(cog, as_interaction(fake=interaction))

    assert interaction.response.deferred is False
    assert len(interaction.response.sent) == 1
    sent = interaction.response.sent[0]
    assert sent.get("ephemeral") is True
    embed = sent.get("embed")
    assert isinstance(embed, Embed)
    assert embed.colour is not None
    assert embed.colour.value == DISCORD_RED
    assert "管理員" in (embed.description or "")


async def test_settings_opens_ephemeral_panel_for_server_admin() -> None:
    """Server admins get an ephemeral panel showing current effective states and overrides."""
    await set_auto_expand_override(guild_id=100, platform="twitter", enabled=False)
    cog = SettingsCogs(bot=stub_bot())
    interaction = FakeInteraction(
        in_guild=True, guild_id=100, guild_name="破貓測試群", administrator=True
    )

    await cog.settings.callback(cog, as_interaction(fake=interaction))

    assert interaction.response.deferred is True
    assert interaction.response.deferred_ephemeral is True
    assert len(interaction.edits) == 1
    edit = interaction.edits[0]
    embed = edit.get("embed")
    assert isinstance(embed, Embed)
    assert embed.colour is not None
    assert embed.colour.value == NEUTRAL_BLUE
    assert "Twitter / X" in (embed.description or "")
    assert "關閉（已自訂）" in (embed.description or "")
    assert "破貓測試群" in (embed.footer.text or "")
    view = edit.get("view")
    assert isinstance(view, ServerSettingsView)
    assert view._origin is interaction


async def test_settings_reports_load_error_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Database failures during initial load log at error and edit with a red error embed."""
    errors = capture_logs(monkeypatch=monkeypatch, level="error")

    async def broken_load(guild_id: int) -> object:
        del guild_id
        raise RuntimeError("settings.db unavailable")

    monkeypatch.setattr(
        target=settings_cog_module, name="get_guild_auto_expand", value=broken_load
    )
    cog = SettingsCogs(bot=stub_bot())
    interaction = FakeInteraction(in_guild=True, guild_id=100, administrator=True)

    await cog.settings.callback(cog, as_interaction(fake=interaction))

    assert interaction.response.deferred is True
    assert len(interaction.edits) == 1
    embed = interaction.edits[0].get("embed")
    assert isinstance(embed, Embed)
    assert embed.colour is not None
    assert embed.colour.value == DISCORD_RED
    assert interaction.edits[0].get("view") is None
    assert len(errors) == 1


def test_setup_upgrades_settings_db_and_registers_cog(monkeypatch: pytest.MonkeyPatch) -> None:
    """`setup` upgrades the `settings` Alembic section and attaches `SettingsCogs`."""
    upgraded: list[str] = []
    added: list[object] = []

    def record_upgrade(section: str) -> None:
        upgraded.append(section)

    monkeypatch.setattr(
        target=settings_cog_module, name="ensure_database_upgraded", value=record_upgrade
    )
    bot = stub_bot()
    bot.add_cog = lambda cog, **_kwargs: added.append(cog)

    setup(bot=bot)

    assert upgraded == ["settings"]
    assert len(added) == 1
    assert isinstance(added[0], SettingsCogs)
