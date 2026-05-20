"""Tests for the central banker management script."""

import pytest
from scripts import manage_central_banker as manage_central_banker_script

from discordbot.typings.economy import CentralBankerAccount
from discordbot.cogs._economy.database import (
    get_account,
    get_balance,
    adjust_balance,
    get_central_banker,
)

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


def test_parse_args_accepts_grant() -> None:
    """The CLI parses a grant request."""
    args = manage_central_banker_script._parse_args(argv=["grant", "42", "--name", "alice"])

    assert args.command == "grant"
    assert args.user_id == 42
    assert args.name == "alice"


def test_parse_args_accepts_list() -> None:
    """The CLI parses the list command."""
    args = manage_central_banker_script._parse_args(argv=["list"])

    assert args.command == "list"


async def test_grant_central_banker_creates_account_row() -> None:
    """Granting central banker creates a zero-balance account."""
    result = await manage_central_banker_script.grant_central_banker(user_id=42, name="alice")

    assert result.applied is True
    assert result.is_central_banker is True
    assert await get_central_banker(user_id=42) is True
    assert await get_balance(user_id=42) == 0


async def test_revoke_central_banker_clears_existing_flag() -> None:
    """Revoking central banker clears the flag on an existing account."""
    await manage_central_banker_script.grant_central_banker(user_id=42, name="alice")

    result = await manage_central_banker_script.revoke_central_banker(user_id=42, name="alice")

    assert result.applied is True
    assert result.is_central_banker is False
    assert await get_central_banker(user_id=42) is False


async def test_revoke_central_banker_missing_user_noops() -> None:
    """Revoking a missing user does not create a row."""
    result = await manage_central_banker_script.revoke_central_banker(user_id=42)

    assert result.applied is False
    assert result.is_central_banker is False
    assert await get_account(user_id=42) is None


async def test_list_central_banker_accounts_filters_non_bankers() -> None:
    """The script lists only current central bankers."""
    await manage_central_banker_script.grant_central_banker(user_id=42, name="alice")
    await adjust_balance(user_id=43, name="bob", delta=100)

    assert await manage_central_banker_script.list_central_banker_accounts() == [
        CentralBankerAccount(user_id=42, name="alice")
    ]
