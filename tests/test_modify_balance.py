"""Tests for the manual balance adjustment script."""

import pytest
from scripts import modify_balance as modify_balance_script

from discordbot.typings.economy import AccountSnapshot, BalanceAdjustmentResult
from discordbot.services.economy.database import get_account, adjust_balance

from tests.helpers.economy import seed_balance


def test_parse_args_accepts_all_target() -> None:
    """The CLI accepts `all` instead of a numeric Discord user ID."""
    args = modify_balance_script._parse_args(argv=["all", "50000"])

    assert args.target == "all"
    assert args.delta == 50_000


async def test_modify_all_balances_updates_existing_accounts_only() -> None:
    """Bulk adjustment updates only accounts already present in the DB."""
    await seed_balance(user_id=1, name="alice", amount=100)
    await seed_balance(user_id=2, name="bob", amount=200)

    result = await modify_balance_script.modify_all_balances(delta=50_000)

    assert len(result.changes) == 2
    assert result.applied_delta == 100_000
    assert all(not change.created for change in result.changes)
    assert await get_account(user_id=3) is None

    alice = await get_account(user_id=1)
    bob = await get_account(user_id=2)
    assert alice is not None
    assert bob is not None
    assert alice.balance == 50_100
    assert bob.balance == 50_200


@pytest.mark.parametrize(
    argnames=("account", "adjustment", "expected_name", "expected"),
    argvalues=[
        (
            AccountSnapshot(name="alice", balance=100, total_earned=100, total_spent=0),
            BalanceAdjustmentResult(new_balance=0, applied_delta=-25),
            "alice",
            (25, -25, 0),
        ),
        (None, BalanceAdjustmentResult(new_balance=20, applied_delta=-80), "1", (100, -80, 20)),
    ],
    ids=["stale-account", "missing-account"],
)
async def test_modify_balance_reports_the_adjustment_not_the_stale_read(
    monkeypatch: pytest.MonkeyPatch,
    account: AccountSnapshot | None,
    adjustment: BalanceAdjustmentResult,
    expected_name: str,
    expected: tuple[int, int, int],
) -> None:
    """The CLI summary uses the adjustment result, not the pre-read projection.

    A missing account still goes through the transactional write, since the read may be stale.
    """
    call: dict[str, int | str | bool] = {}

    async def fake_get_account(user_id: int) -> AccountSnapshot | None:
        """Returns the pre-adjustment read, which the write has already overtaken."""
        assert user_id == 1
        return account

    async def fake_adjust_balance(
        user_id: int, name: str, delta: int, allow_negative: bool
    ) -> BalanceAdjustmentResult:
        """Records the requested adjustment and returns the true DB result."""
        call.update({
            "user_id": user_id,
            "name": name,
            "delta": delta,
            "allow_negative": allow_negative,
        })
        return adjustment

    monkeypatch.setattr(target=modify_balance_script, name="get_account", value=fake_get_account)
    monkeypatch.setattr(
        target=modify_balance_script, name="adjust_balance", value=fake_adjust_balance
    )

    result = await modify_balance_script.modify_balance(user_id=1, name="", delta=-100)

    assert call == {"user_id": 1, "name": expected_name, "delta": -100, "allow_negative": False}
    assert (result.before, result.applied_delta, result.after) == expected
    assert result.requested_delta == -100
    assert result.created is False


async def test_modify_balance_missing_user_negative_noops_without_creating() -> None:
    """A clamped negative adjustment to a missing account remains a no-op."""
    result = await modify_balance_script.modify_balance(user_id=3, name="", delta=-100)

    assert result.before == 0
    assert result.applied_delta == 0
    assert result.after == 0
    assert result.created is False
    assert await get_account(user_id=3) is None


@pytest.mark.parametrize(
    argnames=("start", "delta", "allow_negative", "expected"),
    argvalues=[
        (100, 50, False, (100, 50, 150)),
        (30, -100, False, (30, -30, 0)),
        (30, -100, True, (30, -100, -70)),
        (0, 100, False, (0, 100, 100)),
        (0, -100, False, (0, 0, 0)),
        (0, -100, True, (0, -100, -100)),
        (-100, -50, False, (-100, 0, -100)),
        (-100, 30, False, (-100, 30, -70)),
    ],
    ids=[
        "credit",
        "clamped-debit",
        "allow-negative",
        "missing-account",
        "missing-account-debit",
        "missing-account-allow-negative-debit",
        "debit-below-zero",
        "credit-below-zero",
    ],
)
async def test_dry_run_projects_the_change_without_writing(
    start: int, delta: int, allow_negative: bool, expected: tuple[int, int, int]
) -> None:
    """A dry run reports the `(before, applied_delta, after, created)` the real run applies, and writes nothing."""
    await adjust_balance(user_id=1, name="alice", delta=start, allow_negative=True)
    account = await get_account(user_id=1)

    change = await modify_balance_script.modify_balance(
        user_id=1, name="", delta=delta, allow_negative=allow_negative, dry_run=True
    )

    assert (change.before, change.applied_delta, change.after) == expected
    assert change.dry_run is True
    assert await get_account(user_id=1) == account

    applied = await modify_balance_script.modify_balance(
        user_id=1, name="", delta=delta, allow_negative=allow_negative
    )
    assert (applied.before, applied.applied_delta, applied.after) == expected
    assert change.created == applied.created
