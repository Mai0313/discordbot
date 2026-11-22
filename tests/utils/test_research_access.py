"""Tests for shared deep-research channel eligibility and permission checks."""

from unittest.mock import MagicMock

import pytest
from nextcord import Thread, Permissions, PartialMessageable

from discordbot.utils.research_access import research_access_denial, has_research_permissions

from tests.helpers.discord_mocks import text_channel_granting

_REQUIRED_PERMISSIONS = (
    "view_channel",
    "send_messages",
    "create_public_threads",
    "send_messages_in_threads",
    "attach_files",
)


def _permissions_without(missing: str) -> Permissions:
    """Returns the required deep-research permissions with `missing` omitted."""
    return Permissions(**{name: True for name in _REQUIRED_PERMISSIONS if name != missing})


def test_has_research_permissions_accepts_full_grant() -> None:
    """Holding every required permission passes the predicate."""
    channel = text_channel_granting(
        permissions=Permissions(**dict.fromkeys(_REQUIRED_PERMISSIONS, True))
    )
    assert has_research_permissions(channel=channel, member=channel.guild.me) is True


@pytest.mark.parametrize("missing", _REQUIRED_PERMISSIONS)
def test_has_research_permissions_rejects_any_missing_permission(missing: str) -> None:
    """Dropping any single required permission denies the run."""
    channel = text_channel_granting(permissions=_permissions_without(missing=missing))
    assert has_research_permissions(channel=channel, member=channel.guild.me) is False


def test_research_access_denial_allows_permitted_guild_text_channel() -> None:
    """A guild TextChannel where the bot holds every required permission returns None."""
    channel = text_channel_granting()
    assert research_access_denial(channel=channel, guild=channel.guild) is None
    channel.permissions_for.assert_called_once_with(channel.guild.me)


def test_research_access_denial_returns_unsupported_outside_guild_text_channels() -> None:
    """A missing guild or a non-TextChannel returns unsupported without reading permissions."""
    channel = text_channel_granting()
    assert research_access_denial(channel=channel, guild=None) == "unsupported"
    channel.permissions_for.assert_not_called()

    for other in (MagicMock(spec=Thread), MagicMock(spec=PartialMessageable), None):
        assert research_access_denial(channel=other, guild=channel.guild) == "unsupported"


def test_research_access_denial_returns_forbidden_when_permissions_are_missing() -> None:
    """A guild TextChannel missing a required bot permission returns forbidden."""
    channel = text_channel_granting(permissions=_permissions_without(missing="attach_files"))
    assert research_access_denial(channel=channel, guild=channel.guild) == "forbidden"
    channel.permissions_for.assert_called_once_with(channel.guild.me)
