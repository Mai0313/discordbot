"""Tests for the economy admin management script."""

import pytest
from scripts import manage_admin as manage_admin_script

from discordbot.cogs._economy import database

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


def test_parse_args_accepts_grant() -> None:
    """The CLI parses a grant request."""
    args = manage_admin_script._parse_args(argv=["grant", "42", "--name", "alice"])

    assert args.command == "grant"
    assert args.user_id == 42
    assert args.name == "alice"


def test_parse_args_accepts_list() -> None:
    """The CLI parses the list command."""
    args = manage_admin_script._parse_args(argv=["list"])

    assert args.command == "list"


async def test_grant_admin_creates_admin_row() -> None:
    """Granting admin creates a zero-balance admin account."""
    result = await manage_admin_script.grant_admin(user_id=42, name="alice")

    assert result.applied is True
    assert result.is_admin is True
    assert await database.get_admin(user_id=42) is True
    assert await database.get_balance(user_id=42) == 0


async def test_revoke_admin_clears_existing_flag() -> None:
    """Revoking admin clears the flag on an existing account."""
    await manage_admin_script.grant_admin(user_id=42, name="alice")

    result = await manage_admin_script.revoke_admin(user_id=42, name="alice")

    assert result.applied is True
    assert result.is_admin is False
    assert await database.get_admin(user_id=42) is False


async def test_revoke_admin_missing_user_noops() -> None:
    """Revoking a missing user does not create a row."""
    result = await manage_admin_script.revoke_admin(user_id=42)

    assert result.applied is False
    assert result.is_admin is False
    assert await database.get_account(user_id=42) is None


async def test_list_admin_accounts_filters_non_admins() -> None:
    """The script lists only current admins."""
    await manage_admin_script.grant_admin(user_id=42, name="alice")
    await database.adjust_balance(user_id=43, name="bob", delta=100)

    assert await manage_admin_script.list_admin_accounts() == [(42, "alice")]
