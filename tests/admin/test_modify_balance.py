"""Tests for the manual balance adjustment script."""

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from discordbot.admin import modify_balance as modify_balance_script
from discordbot.typings.economy import AccountSnapshot, BalanceAdjustmentResult
from discordbot.services.economy.database import get_account, adjust_balance

from tests.helpers.economy import seed_balance, hide_from_leaderboard


def test_parse_args_accepts_all_target() -> None:
    """The CLI accepts `all` instead of a numeric Discord user ID."""
    args = modify_balance_script._parse_args(argv=["all", "50000"])

    assert args.target == "all"
    assert args.delta == 50_000


async def test_modify_all_balances_updates_existing_accounts_only() -> None:
    """Bulk adjustment updates every account already in the DB, one hidden from the boards too."""
    await seed_balance(user_id=1, name="alice", amount=100)
    await seed_balance(user_id=2, name="bob", amount=200)
    await hide_from_leaderboard(user_id=2)

    result = await modify_balance_script.modify_all_balances(delta=50_000)

    assert len(result.changes) == 2
    assert result.applied_delta == 100_000
    assert all(not change.created for change in result.changes)

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


async def test_modify_all_balances_avoids_per_account_get_account_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk adjustment uses `top_n` snapshots directly without N+1 `get_account` queries."""
    await seed_balance(user_id=1, name="alice", amount=100)
    await seed_balance(user_id=2, name="bob", amount=200)

    async def fail_get_account(user_id: int) -> None:
        raise AssertionError(f"unexpected N+1 get_account({user_id})")

    monkeypatch.setattr(target=modify_balance_script, name="get_account", value=fail_get_account)

    result = await modify_balance_script.modify_all_balances(delta=50, dry_run=True)

    assert len(result.changes) == 2
    assert result.applied_delta == 100


def test_main_skips_db_setup_on_help_invalid_args_and_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`main` parses CLI args before `mkdir` + Alembic upgrade and handles missing DB on `--dry-run`."""
    monkeypatch.chdir(path=tmp_path)
    upgrades: list[str] = []
    monkeypatch.setattr(
        target=modify_balance_script.command,
        name="upgrade",
        value=lambda **kwargs: upgrades.append(str(kwargs)),
    )

    with pytest.raises(SystemExit):
        modify_balance_script.main(argv=["--help"])
    with pytest.raises(SystemExit):
        modify_balance_script.main(argv=["not-a-user", "10"])
    with pytest.raises(SystemExit, match="no economy database"):
        modify_balance_script.main(argv=["1", "10", "--dry-run"])

    assert upgrades == []
    assert not (tmp_path / "data").exists()

    # Once `economy.db` exists on disk, `--dry-run` runs cleanly without invoking Alembic upgrade.
    db_file = tmp_path / "data" / "database" / "economy.db"
    db_file.parent.mkdir(parents=True)
    db_file.touch()
    buf = StringIO()
    monkeypatch.setattr(
        target=modify_balance_script,
        name="console",
        value=Console(file=buf, force_terminal=False, color_system=None),
    )
    modify_balance_script.main(argv=["1", "10", "--dry-run"])

    assert upgrades == []
    assert "Dry run" in buf.getvalue()


def test_print_helpers_escape_rich_markup_in_usernames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Usernames containing Rich markup brackets render literally without `MarkupError`."""
    buf = StringIO()
    monkeypatch.setattr(
        target=modify_balance_script,
        name="console",
        value=Console(file=buf, width=200, force_terminal=False, color_system=None),
    )
    change = modify_balance_script.BalanceChange(
        user_id=1,
        name="[Owner] [/b]Nick[/]",
        before=100,
        requested_delta=50,
        applied_delta=50,
        after=150,
        created=False,
        dry_run=True,
    )
    bulk = modify_balance_script.BulkBalanceChange(
        changes=(change,), requested_delta=50, applied_delta=50, dry_run=True
    )

    modify_balance_script._print_change(change=change)
    modify_balance_script._print_bulk_change(change=bulk)

    rendered = buf.getvalue()
    assert rendered.count("[Owner] [/b]Nick[/]") == 2
