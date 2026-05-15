"""Tests for the manual balance adjustment script."""

import pytest
from scripts import modify_balance as modify_balance_script

from discordbot.cogs._economy import database

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


async def _add_balance(user_id: int, name: str, amount: int) -> int:
    """Seeds a balance through the manual adjustment API."""
    result = await database.adjust_balance(user_id=user_id, name=name, delta=amount)
    return result.new_balance


def test_parse_args_accepts_all_target() -> None:
    """The CLI accepts `all` instead of a numeric Discord user ID."""
    args = modify_balance_script._parse_args(argv=["all", "50000"])

    assert args.target == "all"
    assert args.delta == 50_000


async def test_modify_all_balances_updates_existing_accounts_only() -> None:
    """Bulk adjustment updates only accounts already present in the DB."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await _add_balance(user_id=2, name="bob", amount=200)

    result = await modify_balance_script.modify_all_balances(delta=50_000)

    assert len(result.changes) == 2
    assert result.applied_delta == 100_000
    assert all(not change.created for change in result.changes)
    assert await database.get_account(user_id=3) is None

    alice = await database.get_account(user_id=1)
    bob = await database.get_account(user_id=2)
    assert alice is not None
    assert bob is not None
    assert alice[1] == 50_100
    assert bob[1] == 50_200


async def test_modify_balance_reports_actual_adjustment_after_stale_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI summary uses the adjustment result, not the pre-read projection."""
    call: dict[str, object] = {}

    async def fake_get_account(user_id: int) -> tuple[str, int, int, int]:
        """Returns a stale pre-adjustment account snapshot."""
        assert user_id == 1
        return ("alice", 100, 100, 0)

    async def fake_adjust_balance(
        user_id: int, name: str, delta: int, allow_negative: bool
    ) -> database.BalanceAdjustmentResult:
        """Records the requested adjustment and returns the true DB result."""
        call.update({
            "user_id": user_id,
            "name": name,
            "delta": delta,
            "allow_negative": allow_negative,
        })
        return database.BalanceAdjustmentResult(new_balance=0, applied_delta=-25)

    monkeypatch.setattr(
        target=modify_balance_script.database, name="get_account", value=fake_get_account
    )
    monkeypatch.setattr(
        target=modify_balance_script.database, name="adjust_balance", value=fake_adjust_balance
    )

    result = await modify_balance_script.modify_balance(user_id=1, name="", delta=-80)

    assert call == {"user_id": 1, "name": "alice", "delta": -80, "allow_negative": False}
    assert result.before == 25
    assert result.requested_delta == -80
    assert result.applied_delta == -25
    assert result.after == 0


async def test_modify_balance_missing_user_negative_noops_without_creating() -> None:
    """A clamped negative adjustment to a missing account remains a no-op."""
    result = await modify_balance_script.modify_balance(user_id=3, name="", delta=-100)

    assert result.before == 0
    assert result.applied_delta == 0
    assert result.after == 0
    assert result.created is False
    assert await database.get_account(user_id=3) is None


async def test_modify_balance_missing_user_negative_delegates_to_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing-user negative writes still go through the transactional API."""
    call: dict[str, object] = {}

    async def fake_get_account(user_id: int) -> None:
        """Returns no account for the requested user."""
        assert user_id == 3

    async def fake_adjust_balance(
        user_id: int, name: str, delta: int, allow_negative: bool
    ) -> database.BalanceAdjustmentResult:
        """Records that the missing-user write still reaches the DB facade."""
        call.update({
            "user_id": user_id,
            "name": name,
            "delta": delta,
            "allow_negative": allow_negative,
        })
        return database.BalanceAdjustmentResult(new_balance=20, applied_delta=-80)

    monkeypatch.setattr(
        target=modify_balance_script.database, name="get_account", value=fake_get_account
    )
    monkeypatch.setattr(
        target=modify_balance_script.database, name="adjust_balance", value=fake_adjust_balance
    )

    result = await modify_balance_script.modify_balance(user_id=3, name="", delta=-100)

    assert call == {"user_id": 3, "name": "3", "delta": -100, "allow_negative": False}
    assert result.before == 100
    assert result.applied_delta == -80
    assert result.after == 20
    assert result.created is False
