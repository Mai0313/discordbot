"""Tests for `ServerSettingsView` button toggles, reset, permission checks, and timeout."""

import asyncio
from pathlib import Path

import pytest
from nextcord import Embed, ButtonStyle
from nextcord.ui import Button

from discordbot.cogs.settings import views as views_module
from discordbot.typings.colors import DISCORD_RED
from discordbot.typings.config import clear_bot_config_cache
from discordbot.cogs.settings.views import ServerSettingsView
from discordbot.utils.server_settings import get_guild_auto_expand, is_auto_expand_enabled

from tests.helpers.casting import as_interaction
from tests.helpers.discord_mocks import FakeInteraction
from tests.helpers.logfire_capture import capture_logs


def _write_local_config(path: Path, body: str) -> None:
    """Writes a throwaway local `config.toml` and clears the process config cache."""
    path.write_text(data=body, encoding="utf-8")
    clear_bot_config_cache()


async def test_toggle_persists_override_updates_styles_and_reset_restores_defaults() -> None:
    """Pressing a platform button toggles its state in `settings.db`, and Reset clears all overrides."""
    initial = await get_guild_auto_expand(guild_id=77)
    view = ServerSettingsView(guild_id=77, guild_name="測試群", state=initial)
    assert view._platform_buttons["twitter"].style is ButtonStyle.success
    assert view._reset_button.disabled is True

    # Toggle Twitter OFF.
    toggle_off = FakeInteraction(in_guild=True, guild_id=77, administrator=True)
    await view._platform_buttons["twitter"].callback(as_interaction(fake=toggle_off))

    assert toggle_off.response.deferred is True
    assert await is_auto_expand_enabled(guild_id=77, platform="twitter") is False
    assert view._platform_buttons["twitter"].style is ButtonStyle.secondary
    assert view._reset_button.disabled is False
    assert len(toggle_off.edits) == 1
    off_embed = toggle_off.edits[0].get("embed")
    assert isinstance(off_embed, Embed)
    assert "關閉（已自訂）" in (off_embed.description or "")

    # Reset restores all defaults and disables the reset button again.
    reset_click = FakeInteraction(in_guild=True, guild_id=77, administrator=True)
    await view._reset_button.callback(as_interaction(fake=reset_click))

    assert reset_click.response.deferred is True
    assert await is_auto_expand_enabled(guild_id=77, platform="twitter") is True
    assert view._platform_buttons["twitter"].style is ButtonStyle.success
    assert view._reset_button.disabled is True
    assert len(reset_click.edits) == 1
    reset_embed = reset_click.edits[0].get("embed")
    assert isinstance(reset_embed, Embed)
    assert "開啟（預設）" in (reset_embed.description or "")


async def test_button_press_rechecks_admin_permission_and_guild() -> None:
    """Revoked admin permissions or mismatched guild ids are refused without touching `settings.db`."""
    initial = await get_guild_auto_expand(guild_id=88)
    view = ServerSettingsView(guild_id=88, guild_name="測試群", state=initial)

    non_admin = FakeInteraction(in_guild=True, guild_id=88, administrator=False)
    await view._platform_buttons["threads"].callback(as_interaction(fake=non_admin))

    assert non_admin.response.deferred is False
    assert len(non_admin.response.sent) == 1
    assert non_admin.response.sent[0].get("ephemeral") is True
    assert await is_auto_expand_enabled(guild_id=88, platform="threads") is True

    wrong_guild = FakeInteraction(in_guild=True, guild_id=99, administrator=True)
    await view._reset_button.callback(as_interaction(fake=wrong_guild))

    assert wrong_guild.response.deferred is False
    assert len(wrong_guild.response.sent) == 1
    assert wrong_guild.response.sent[0].get("ephemeral") is True


async def test_button_write_error_logs_and_sends_ephemeral_followup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database failure during toggle reports an ephemeral error without mutating view state."""
    errors = capture_logs(monkeypatch=monkeypatch, level="error")

    async def broken_set(**_kwargs: object) -> object:
        raise RuntimeError("disk full")

    monkeypatch.setattr(target=views_module, name="set_auto_expand_override", value=broken_set)
    initial = await get_guild_auto_expand(guild_id=66)
    view = ServerSettingsView(guild_id=66, guild_name="測試群", state=initial)

    click = FakeInteraction(in_guild=True, guild_id=66, administrator=True)
    await view._platform_buttons["douyin"].callback(as_interaction(fake=click))

    assert click.response.deferred is True
    assert click.edits == []
    assert len(click.followup.sent) == 1
    followup = click.followup.sent[0]
    assert followup.get("ephemeral") is True
    embed = followup.get("embed")
    assert isinstance(embed, Embed)
    assert embed.colour is not None
    assert embed.colour.value == DISCORD_RED
    assert view._platform_buttons["douyin"].style is ButtonStyle.success
    assert len(errors) == 1


async def test_custom_application_emoji_from_config_toml_renders_on_buttons(
    bot_config_isolated: Path,
) -> None:
    """Custom Discord Application Emojis configured in `[global.emojis]` attach to toggle buttons."""
    _write_local_config(
        path=bot_config_isolated,
        body='[global.emojis]\nfacebook = "<:facebook:1546179601288396810>"\n',
    )
    initial = await get_guild_auto_expand(guild_id=55)
    view = ServerSettingsView(guild_id=55, guild_name="測試群", state=initial)

    fb_emoji = view._platform_buttons["facebook"].emoji
    assert fb_emoji is not None
    assert fb_emoji.name == "facebook"
    assert fb_emoji.id == 1546179601288396810


async def test_timeout_disables_all_buttons_via_bound_origin() -> None:
    """`on_timeout` disables every button and edits the panel through the bound interaction."""
    initial = await get_guild_auto_expand(guild_id=44)
    view = ServerSettingsView(guild_id=44, guild_name="測試群", state=initial)
    origin = FakeInteraction(in_guild=True, guild_id=44, administrator=True)
    view.bind_origin(interaction=as_interaction(fake=origin))

    await view.on_timeout()

    assert all(child.disabled for child in view.children if isinstance(child, Button))
    assert len(origin.edits) == 1
    assert origin.edits[0].get("view") is view


async def test_reset_button_keeps_symbol_in_label_rather_than_emoji() -> None:
    """Discord rejects non-emoji Unicode arrows like `↺` in `Button.emoji`; keep it in `label`."""
    initial = await get_guild_auto_expand(guild_id=33)
    view = ServerSettingsView(guild_id=33, guild_name="測試群", state=initial)

    assert view._reset_button.emoji is None
    assert view._reset_button.label == "↺ 恢復預設"


async def test_reset_write_error_logs_and_sends_ephemeral_followup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database failure during reset reports an ephemeral error without mutating view state."""
    errors = capture_logs(monkeypatch=monkeypatch, level="error")
    await views_module.set_auto_expand_override(guild_id=67, platform="twitter", enabled=False)
    initial = await get_guild_auto_expand(guild_id=67)
    view = ServerSettingsView(guild_id=67, guild_name="測試群", state=initial)

    async def broken_reset(guild_id: int) -> object:
        del guild_id
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        target=views_module, name="reset_auto_expand_overrides", value=broken_reset
    )
    click = FakeInteraction(in_guild=True, guild_id=67, administrator=True)
    await view._reset_button.callback(as_interaction(fake=click))

    assert click.response.deferred is True
    assert click.edits == []
    assert len(click.followup.sent) == 1
    assert click.followup.sent[0].get("ephemeral") is True
    assert view._reset_button.disabled is False
    assert len(errors) == 1


async def test_toggle_arriving_after_timeout_does_not_reenable_buttons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialized locking ensures `on_timeout` leaves buttons disabled even across in-flight presses."""
    initial = await get_guild_auto_expand(guild_id=45)
    view = ServerSettingsView(guild_id=45, guild_name="測試群", state=initial)
    origin = FakeInteraction(in_guild=True, guild_id=45, administrator=True)
    view.bind_origin(interaction=as_interaction(fake=origin))

    entered_write = asyncio.Event()
    release_write = asyncio.Event()
    real_set = views_module.set_auto_expand_override

    async def slow_set(guild_id: int, platform: str, enabled: bool) -> object:
        entered_write.set()
        await release_write.wait()
        return await real_set(guild_id=guild_id, platform=platform, enabled=enabled)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(target=views_module, name="set_auto_expand_override", value=slow_set)

    click = FakeInteraction(in_guild=True, guild_id=45, administrator=True)
    toggle_task = asyncio.create_task(
        view._platform_buttons["twitter"].callback(as_interaction(fake=click))
    )
    await entered_write.wait()
    timeout_task = asyncio.create_task(view.on_timeout())
    await asyncio.sleep(0)
    release_write.set()
    await asyncio.gather(toggle_task, timeout_task)

    assert all(child.disabled for child in view.children if isinstance(child, Button))

    # Any late click after timeout is ignored without editing or writing.
    late_click = FakeInteraction(in_guild=True, guild_id=45, administrator=True)
    await view._platform_buttons["threads"].callback(as_interaction(fake=late_click))
    assert late_click.edits == []
    assert await is_auto_expand_enabled(guild_id=45, platform="threads") is True
