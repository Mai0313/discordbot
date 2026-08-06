"""Pins `scripts/modify_balance.py`, the offline CLI that edits balances with no Discord in front.

An operator running it sees nothing but the summary it prints: no ephemeral embed to re-read, no
confirmation view, no second person watching. So most of what is pinned here is that the summary
tells the truth about what the ledger actually did.

- `all` survives argument parsing. A target is normally a Discord user ID and `_parse_target` is
  wired in as argparse's `type=`, so the bulk target is one `int()` away from being rejected.
- The bulk run touches stored accounts only. It enumerates the ledger itself, so a user the ledger
  has never seen has no row to update and never gets one minted.
- The reported figures come from the write, not from the read before it. `modify_balance` reads
  the account and then writes in a separate `adjust_balance` transaction, so it recovers `before`
  as `new_balance - applied_delta` instead of trusting its own projection. A stale read is exactly
  what a concurrent write produces, and reporting the projection would hide it.
- Clamping a negative delta and refusing to mint a zero-balance row stay the ledger's decision
  (`_apply_clamped_delta_in_session`), so the script must not grow a second copy of them: one test
  pins the no-op end to end, the other pins that the call is still made, carrying the fallback
  name the script derived, rather than short-circuited locally.

`pytestmark` puts the module on `economy_isolated_db`, so the unstubbed tests write to a throwaway
`economy.db`. The stubbed tests monkeypatch `get_account` / `adjust_balance` on the script module
rather than on the service, because the script binds both names at import.
"""

import pytest
from scripts import modify_balance as modify_balance_script

from discordbot.typings.economy import AccountSnapshot
from discordbot.services.economy.database import (
    BalanceAdjustmentResult,
    get_account,
    adjust_balance,
)

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


async def _add_balance(user_id: int, name: str, amount: int) -> int:
    """Seeds an account balance through the ledger's own maintenance write.

    `adjust_balance` neither repays loans nor moves the daily casino counters, so a seeded account
    starts in the state a manual top-up would leave it in.

    Returns:
        The balance after the credit.
    """
    result = await adjust_balance(user_id=user_id, name=name, delta=amount)
    return result.new_balance


def test_parse_args_accepts_all_target() -> None:
    """The CLI accepts `all` instead of a numeric Discord user ID."""
    args = modify_balance_script._parse_args(argv=["all", "50000"])

    assert args.target == "all"
    assert args.delta == 50_000


async def test_modify_all_balances_updates_existing_accounts_only() -> None:
    """Bulk adjustment credits every stored account and mints none for a user with no row."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await _add_balance(user_id=2, name="bob", amount=200)

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


async def test_modify_balance_reports_actual_adjustment_after_stale_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported change comes from the write's own result, not the pre-read projection."""
    call: dict[str, int | str | bool] = {}

    async def fake_get_account(user_id: int) -> AccountSnapshot:
        """Answers the pre-write read with the balance a concurrent write has already moved.

        Returns:
            A snapshot carrying 100, which the adjustment below contradicts.
        """
        assert user_id == 1
        return AccountSnapshot(name="alice", balance=100, total_earned=100, total_spent=0)

    async def fake_adjust_balance(
        user_id: int, name: str, delta: int, allow_negative: bool
    ) -> BalanceAdjustmentResult:
        """Records the requested adjustment and answers with what the ledger really did.

        Returns:
            A result implying a starting balance of 25, not the 100 that was read.
        """
        call.update({
            "user_id": user_id,
            "name": name,
            "delta": delta,
            "allow_negative": allow_negative,
        })
        return BalanceAdjustmentResult(new_balance=0, applied_delta=-25)

    monkeypatch.setattr(target=modify_balance_script, name="get_account", value=fake_get_account)
    monkeypatch.setattr(
        target=modify_balance_script, name="adjust_balance", value=fake_adjust_balance
    )

    result = await modify_balance_script.modify_balance(user_id=1, name="", delta=-80)

    assert call == {"user_id": 1, "name": "alice", "delta": -80, "allow_negative": False}
    assert result.before == 25
    assert result.requested_delta == -80
    assert result.applied_delta == -25
    assert result.after == 0


async def test_modify_balance_missing_user_negative_noops_without_creating() -> None:
    """A negative adjustment against a missing account writes nothing and creates no row."""
    result = await modify_balance_script.modify_balance(user_id=3, name="", delta=-100)

    assert result.before == 0
    assert result.applied_delta == 0
    assert result.after == 0
    assert result.created is False
    assert await get_account(user_id=3) is None


async def test_modify_balance_missing_user_negative_delegates_to_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing-user negative delta still reaches `adjust_balance` instead of stopping early."""
    call: dict[str, int | str | bool] = {}

    async def fake_get_account(user_id: int) -> AccountSnapshot | None:
        """Answers the pre-write read with nothing, so the script takes the missing-user path.

        Returns:
            None, the ledger's answer for a user it has never seen.
        """
        assert user_id == 3
        return None

    async def fake_adjust_balance(
        user_id: int, name: str, delta: int, allow_negative: bool
    ) -> BalanceAdjustmentResult:
        """Records the write the script would have skipped had it clamped on its own.

        Returns:
            A result implying a starting balance of 100, which the missing-account read denied.
        """
        call.update({
            "user_id": user_id,
            "name": name,
            "delta": delta,
            "allow_negative": allow_negative,
        })
        return BalanceAdjustmentResult(new_balance=20, applied_delta=-80)

    monkeypatch.setattr(target=modify_balance_script, name="get_account", value=fake_get_account)
    monkeypatch.setattr(
        target=modify_balance_script, name="adjust_balance", value=fake_adjust_balance
    )

    result = await modify_balance_script.modify_balance(user_id=3, name="", delta=-100)

    assert call == {"user_id": 3, "name": "3", "delta": -100, "allow_negative": False}
    assert result.before == 100
    assert result.applied_delta == -80
    assert result.after == 20
    assert result.created is False
