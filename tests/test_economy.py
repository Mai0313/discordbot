"""Tests for the economy persistence layer."""

from random import Random, SystemRandom
from typing import Any, cast
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text, select, update
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.typings.games import (
    GameParticipant,
    BlackjackPlayerResult,
    BlackjackDealerDecision,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
)
from discordbot.utils.timezone import TAIWAN_TIMEZONE
from discordbot.cogs._games.blackjack import Card, BlackjackRound, BlackjackHandState
from discordbot.cogs._economy.database import (
    VIP_PURCHASE_COST,
    CHECKIN_STREAK_CYCLE,
    BASE_CHECKIN_REWARD_AMOUNT,
    UserWallet,
    JackpotPool,
    UserAccount,
    AdminAccount,
    CasinoAccount,
    TransferResult,
    AccountSnapshot,
    JackpotSnapshot,
    LeaderboardEntry,
    LossLeaderboardEntry,
    BalanceAdjustmentResult,
    JackpotSettlementRequest,
    top_n,
    buy_vip,
    checkin,
    get_vip,
    transfer,
    get_admin,
    set_admin,
    _as_taipei,
    top_losers,
    get_account,
    get_balance,
    list_admins,
    open_session,
    _database_now,
    _ensure_schema,
    add_bot_status,
    adjust_balance,
    checkin_reward,
    _taipei_midnight,
    get_bot_statuses,
    get_jackpot_pool,
    get_casino_ledger,
    remove_bot_status,
    _stored_int_to_int,
    get_jackpot_snapshot,
    list_bot_status_rows,
    credit_with_repayment,
    apply_round_settlement,
    get_casino_daily_stats,
    apply_jackpot_settlement,
    open_global_state_session,
    apply_jackpot_settlement_batch,
    _apply_jackpot_delta_in_session,
    _apply_daily_casino_delta_in_session,
    invalidate_economy_leaderboard_cache,
)
from discordbot.cogs._games.settlement import settle_wager, settle_blackjack_player
from discordbot.cogs._games.blackjack_views import BlackjackView, build_final_embeds

pytestmark = pytest.mark.usefixtures("economy_isolated_db")


class _DealerStub:
    """Minimal dealer stub for BlackjackView settlement tests."""

    def __init__(self) -> None:
        """Initializes call counters for dealer interactions."""
        self.settle_calls = 0
        self.hint_calls = 0
        self.decision_calls = 0
        self.decisions: list[BlackjackDealerDecision] = []
        self.hints: list[dict[str, Any]] = []

    async def settle(self, **_kwargs: Any) -> str:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Returns deterministic banter and tracks settlement calls."""
        self.settle_calls += 1
        await asyncio.sleep(delay=0)
        return "settled"

    async def hint(self, **_kwargs: Any) -> str:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Returns deterministic in-progress banter and tracks hint calls."""
        self.hint_calls += 1
        self.hints.append(_kwargs)
        await asyncio.sleep(delay=0)
        return "hint"

    async def decide_blackjack_action(self, **_kwargs: Any) -> BlackjackDealerDecision:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Returns deterministic dealer decisions and tracks calls."""
        self.decision_calls += 1
        await asyncio.sleep(delay=0)
        if self.decisions:
            return self.decisions.pop(0)
        return BlackjackDealerDecision(action="stand", reason="stub stand")


class _SlowSettleDealerStub(_DealerStub):
    """Dealer stub whose settlement banter blocks until released."""

    def __init__(self) -> None:
        """Initializes gate events around the settlement line."""
        super().__init__()
        self.settle_started = asyncio.Event()
        self.release_settle = asyncio.Event()

    async def settle(self, **_kwargs: Any) -> str:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Blocks settlement banter so tests can inspect the immediate final edit."""
        self.settle_calls += 1
        self.settle_started.set()
        await self.release_settle.wait()
        return "settled"


def test_blackjack_player_settlement_hands_default_is_isolated() -> None:
    """Default Blackjack hand settlement lists are isolated per model instance."""
    first = BlackjackPlayerSettlement(
        delta=0, payout=0, new_balance=100, casino_balance=0, outcome="push", detail="first"
    )
    second = BlackjackPlayerSettlement(
        delta=0, payout=0, new_balance=100, casino_balance=0, outcome="push", detail="second"
    )

    first.hands.append(BlackjackHandSettlement(cards=[], bet=10, outcome="push", delta=0))

    assert second.hands == []


class _MessageStub:
    """Minimal message stub that records edit calls."""

    def __init__(self) -> None:
        """Initializes the message edit counter."""
        self.edit_calls = 0
        self.edits: list[dict[str, Any]] = []

    async def edit(self, **_kwargs: Any) -> None:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Records a Discord message edit."""
        self.edit_calls += 1
        self.edits.append(_kwargs)


class _ResponseStub:
    """Minimal interaction response stub for button callback tests."""

    def __init__(self) -> None:
        """Initializes the deferred flag."""
        self.deferred = False

    async def defer(self) -> None:
        """Records that the button interaction was deferred."""
        self.deferred = True

    def is_done(self) -> bool:
        """Returns whether the interaction response was already used."""
        return self.deferred


class _FollowupStub:
    """Minimal followup stub for private button notices."""

    def __init__(self) -> None:
        """Initializes recorded followup sends."""
        self.sent: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> None:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Records a followup send payload."""
        self.sent.append(kwargs)


class _UserStub:
    """Minimal interaction user stub."""

    def __init__(self, user_id: int = 1) -> None:
        """Initializes a Discord-like user identity."""
        self.id = user_id


class _InteractionStub:
    """Minimal button interaction stub."""

    def __init__(self, message: _MessageStub, user_id: int = 1) -> None:
        """Initializes an interaction with a message and response stub."""
        self.message = message
        self.response = _ResponseStub()
        self.followup = _FollowupStub()
        self.user = _UserStub(user_id=user_id)


def _participant(
    user_id: int = 1,
    account_name: str = "alice",
    display_name: str = "Alice",
    bet: int = 50,
    balance_at_start: int = 100,
) -> GameParticipant:
    """Builds a prepared Blackjack participant for view tests."""
    return GameParticipant(
        user_id=user_id,
        account_name=account_name,
        display_name=display_name,
        bet=bet,
        balance_at_start=balance_at_start,
        is_allin=False,
    )


def _round_from_cards(
    player_cards: list[Card],
    dealer_cards: list[Card],
    participant: GameParticipant,
    finished: bool,
) -> BlackjackRound:
    """Builds the production Blackjack round shape used by views and settlement."""
    round_state = BlackjackRound.from_participants(rng=SystemRandom(), participants=[participant])
    round_state.players[0].hands[0].cards = list(player_cards)
    round_state.players[0].hands[0].finished = finished
    round_state.dealer = list(dealer_cards)
    round_state.finished = finished
    round_state.dealer_played = finished
    round_state.phase = "settled" if finished else "player_actions"
    return round_state


async def _stored_avatar_url(user_id: int) -> str:
    """Reads the cached avatar URL for one account."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.avatar_url).where(UserAccount.user_id == user_id)
        )
        return result.scalar_one()


async def _stored_wallet_name(user_id: int) -> str:
    """Reads the denormalized wallet name for one account."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserWallet.name).where(UserWallet.user_id == user_id)
        )
        return result.scalar_one()


async def _daily_casino_stats(user_id: int) -> tuple[int, int, int, datetime | None]:
    """Reads daily casino `(loss, win, net, day_started_at)` counters."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(
                CasinoAccount.daily_loss,
                CasinoAccount.daily_win,
                CasinoAccount.daily_net,
                CasinoAccount.day_started_at,
            ).where(CasinoAccount.user_id == user_id)
        )
        row = result.one_or_none()
    if row is None:
        return 0, 0, 0, None
    return (
        _stored_int_to_int(value=row[0]),
        _stored_int_to_int(value=row[1]),
        _stored_int_to_int(value=row[2]),
        row[3],
    )


async def _economy_schema_details() -> tuple[
    set[str], set[str], set[str], dict[str, set[str]], dict[str, dict[str, str]]
]:
    """Reads current economy schema metadata."""
    async with open_session() as session:
        result = await session.execute(
            statement=text(
                text="SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        economy_tables = {row[0] for row in result.all()}
        result = await session.execute(statement=text(text="PRAGMA index_list(user_wallet)"))
        wallet_index_names = {row[1] for row in result.all()}
        result = await session.execute(statement=text(text="PRAGMA index_list(casino_account)"))
        casino_index_names = {row[1] for row in result.all()}
        column_queries = {
            "user_account": "PRAGMA table_info(user_account)",
            "user_wallet": "PRAGMA table_info(user_wallet)",
            "loan_proposal": "PRAGMA table_info(loan_proposal)",
            "loan_contract": "PRAGMA table_info(loan_contract)",
            "casino_account": "PRAGMA table_info(casino_account)",
        }
        table_columns: dict[str, set[str]] = {}
        table_column_types: dict[str, dict[str, str]] = {}
        for table_name, query in column_queries.items():
            result = await session.execute(statement=text(text=query))
            table_info = result.all()
            table_columns[table_name] = {row[1] for row in table_info}
            table_column_types[table_name] = {row[1]: row[2] for row in table_info}
    return (
        economy_tables,
        wallet_index_names,
        casino_index_names,
        table_columns,
        table_column_types,
    )


async def _global_state_schema_details() -> tuple[
    set[str], tuple[int, int, int, int, int], dict[str, str]
]:
    """Reads current global-state schema metadata."""
    async with open_global_state_session() as session:
        result = await session.execute(
            statement=text(
                text="SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        global_state_tables = {row[0] for row in result.all()}
        result = await session.execute(
            statement=select(
                JackpotPool.pool_balance,
                JackpotPool.total_contributed,
                JackpotPool.total_claimed,
                JackpotPool.seeded_amount,
                JackpotPool.generation,
            ).where(JackpotPool.game_id == "dragon_gate")
        )
        jackpot_row = result.one()
        result = await session.execute(statement=text(text="PRAGMA table_info(jackpot_pool)"))
        jackpot_column_types = {row[1]: row[2] for row in result.all()}
    return (
        global_state_tables,
        cast("tuple[int, int, int, int, int]", tuple(jackpot_row)),
        jackpot_column_types,
    )


def _assert_money_columns_are_text(
    table_column_types: dict[str, dict[str, str]], jackpot_column_types: dict[str, str]
) -> None:
    """Checks all decimal-string money columns use SQLite TEXT affinity."""
    economy_money_columns = {
        "user_wallet": ("balance", "total_earned", "total_spent"),
        "loan_proposal": ("amount", "escrow_amount"),
        "loan_contract": (
            "original_principal",
            "principal_remaining",
            "interest_due",
            "total_interest_paid",
            "total_principal_paid",
        ),
        "casino_account": ("daily_loss", "daily_win", "daily_net"),
    }
    for table_name, column_names in economy_money_columns.items():
        for column_name in column_names:
            assert table_column_types[table_name][column_name] == "TEXT"
    for column_name in ("pool_balance", "total_contributed", "total_claimed", "seeded_amount"):
        assert jackpot_column_types[column_name] == "TEXT"


async def _add_balance(user_id: int, name: str, amount: int, avatar_url: str = "") -> int:
    """Seeds a positive balance without loan or casino side effects."""
    if amount <= 0:
        return await get_balance(user_id=user_id)
    result = await adjust_balance(user_id=user_id, name=name, delta=amount, avatar_url=avatar_url)
    return result.new_balance


async def test_adjust_balance_creates_user() -> None:
    """First manual adjustment upserts the row and returns the new balance."""
    result = await adjust_balance(user_id=42, name="alice", delta=100)
    assert result == BalanceAdjustmentResult(new_balance=100, applied_delta=100)
    assert await get_balance(user_id=42) == 100


async def test_adjust_balance_accumulates() -> None:
    """Repeated manual adjustments increment the running balance."""
    await adjust_balance(user_id=42, name="alice", delta=100)
    result = await adjust_balance(user_id=42, name="alice", delta=50)
    assert result == BalanceAdjustmentResult(new_balance=150, applied_delta=50)


async def test_adjust_balance_zero_is_noop() -> None:
    """Zero deltas do not change balance or lifetime totals."""
    await _add_balance(user_id=42, name="alice", amount=100)
    result = await adjust_balance(user_id=42, name="alice", delta=0)
    assert result == BalanceAdjustmentResult(new_balance=100, applied_delta=0)
    account = await get_account(user_id=42)
    assert account == AccountSnapshot(name="alice", balance=100, total_earned=100, total_spent=0)


async def test_adjust_balance_positive_updates_total_earned() -> None:
    """Positive manual adjustments are counted as earned points."""
    result = await adjust_balance(user_id=42, name="alice", delta=100)
    assert result == BalanceAdjustmentResult(new_balance=100, applied_delta=100)
    account = await get_account(user_id=42)
    assert account == AccountSnapshot(name="alice", balance=100, total_earned=100, total_spent=0)


async def test_adjust_balance_clamps_at_zero() -> None:
    """Negative manual adjustment clamps at zero by default."""
    await _add_balance(user_id=42, name="alice", amount=10)
    result = await adjust_balance(user_id=42, name="alice", delta=-1_000)
    assert result == BalanceAdjustmentResult(new_balance=0, applied_delta=-10)


async def test_adjust_balance_negative_missing_user_does_not_create_row() -> None:
    """Clamped negative adjustments to absent users stay no-op reads."""
    result = await adjust_balance(user_id=42, name="alice", delta=-1_000)

    assert result == BalanceAdjustmentResult(new_balance=0, applied_delta=0)
    assert await get_account(user_id=42) is None


async def test_adjust_balance_allows_negative_when_requested() -> None:
    """Manual tooling can explicitly allow a negative resulting balance."""
    await _add_balance(user_id=42, name="alice", amount=10)
    result = await adjust_balance(user_id=42, name="alice", delta=-500, allow_negative=True)
    assert result == BalanceAdjustmentResult(new_balance=-490, applied_delta=-500)


async def test_adjust_balance_refreshes_name() -> None:
    """Subsequent writes refresh the cached display name."""
    await _add_balance(user_id=42, name="alice", amount=10)
    await _add_balance(user_id=42, name="alice_renamed", amount=10)
    rows = await top_n(limit=1)
    assert rows[0].name == "alice_renamed"
    assert rows[0].avatar_url == ""
    assert await _stored_wallet_name(user_id=42) == "alice_renamed"


async def test_adjust_balance_stores_and_refreshes_avatar_url() -> None:
    """Subsequent writes refresh the cached avatar URL."""
    await _add_balance(user_id=42, name="alice", amount=10, avatar_url="https://cdn.example/a.png")
    assert await _stored_avatar_url(user_id=42) == "https://cdn.example/a.png"

    await _add_balance(user_id=42, name="alice", amount=10, avatar_url="https://cdn.example/b.png")
    assert await _stored_avatar_url(user_id=42) == "https://cdn.example/b.png"


async def test_admin_flag_defaults_to_false() -> None:
    """Unknown users and normal accounts are not economy admins."""
    assert await get_admin(user_id=42) is False
    await _add_balance(user_id=42, name="alice", amount=10)
    assert await get_admin(user_id=42) is False


async def test_leaderboard_hidden_flag_defaults_to_false() -> None:
    """New accounts are visible on public leaderboards by default."""
    await _add_balance(user_id=42, name="alice", amount=10)
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.hide_from_leaderboard).where(UserAccount.user_id == 42)
        )
    assert result.scalar_one() is False


async def test_set_admin_creates_user() -> None:
    """Granting admin creates a zero-balance account row."""
    applied = await set_admin(user_id=42, name="alice", is_admin=True)
    assert applied is True
    assert await get_admin(user_id=42) is True
    assert await get_balance(user_id=42) == 0


async def test_set_admin_revokes_existing_user() -> None:
    """Revoking admin clears the flag on an existing account."""
    await set_admin(user_id=42, name="alice", is_admin=True)
    applied = await set_admin(user_id=42, name="alice", is_admin=False)
    assert applied is True
    assert await get_admin(user_id=42) is False


async def test_set_admin_revoke_missing_user_noops() -> None:
    """Revoking a missing user does not create an account row."""
    applied = await set_admin(user_id=42, name="alice", is_admin=False)
    assert applied is False
    assert await get_account(user_id=42) is None


async def test_list_admins_returns_only_admin_accounts() -> None:
    """Admin listing filters out normal economy users."""
    await set_admin(user_id=42, name="alice", is_admin=True)
    await set_admin(user_id=43, name="bob", is_admin=True)
    await _add_balance(user_id=44, name="carol", amount=10)
    await set_admin(user_id=43, name="bob", is_admin=False)
    assert await list_admins() == [AdminAccount(user_id=42, name="alice")]


async def test_write_timestamps_use_taiwan_local_time() -> None:
    """Account timestamps are persisted as Taiwan-local wall time."""
    before = datetime.now(tz=TAIWAN_TIMEZONE).replace(tzinfo=None)
    await credit_with_repayment(user_id=42, name="alice", amount=10)
    after = datetime.now(tz=TAIWAN_TIMEZONE).replace(tzinfo=None)

    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.updated_at).where(UserAccount.user_id == 42)
        )
        updated_at = result.scalar_one()

    assert before <= updated_at <= after


async def test_ensure_schema_bootstraps_current_databases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean startup creates only the current economy and global-state tables."""
    db_path = tmp_path / "current-economy.db"
    global_state_db_path = tmp_path / "current-global-state.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    global_state_engine = create_async_engine(url=f"sqlite+aiosqlite:///{global_state_db_path}")
    monkeypatch.setattr("discordbot.cogs._economy.database._engine", engine)
    monkeypatch.setattr(
        "discordbot.cogs._economy.database._global_state_engine", global_state_engine
    )
    monkeypatch.setattr("discordbot.cogs._economy.database._schema_ready_for", None)
    monkeypatch.setattr("discordbot.cogs._economy.database._global_state_schema_ready_for", None)

    await _ensure_schema()

    (
        economy_tables,
        wallet_index_names,
        casino_index_names,
        table_columns,
        table_column_types,
    ) = await _economy_schema_details()
    global_state_tables, jackpot_row, jackpot_column_types = await _global_state_schema_details()
    assert economy_tables == {
        "user_account",
        "user_wallet",
        "loan_proposal",
        "loan_contract",
        "casino_account",
    }
    assert global_state_tables == {"jackpot_pool", "casino_ledger", "bot_status"}
    assert {"user_id", "name", "is_central_banker"} <= table_columns["user_account"]
    assert {"user_id", "name", "balance", "total_earned", "total_spent"} <= table_columns[
        "user_wallet"
    ]
    assert {"balance", "total_earned", "total_spent"}.isdisjoint(table_columns["user_account"])
    assert {"borrower_id", "borrower_name", "lender_id", "lender_name"} <= table_columns[
        "loan_proposal"
    ]
    assert {"borrower_id", "borrower_name", "lender_type"} <= table_columns["loan_contract"]
    _assert_money_columns_are_text(
        table_column_types=table_column_types, jackpot_column_types=jackpot_column_types
    )
    assert "ix_user_wallet_balance" in wallet_index_names
    assert "ix_casino_account_day_loss" in casino_index_names
    assert jackpot_row == (1_000, 0, 0, 1_000, 0)

    await _add_balance(
        user_id=42, name="alice", amount=5, avatar_url="https://cdn.example/avatar.png"
    )
    assert await _stored_avatar_url(user_id=42) == "https://cdn.example/avatar.png"
    assert await _stored_wallet_name(user_id=42) == "alice"
    account = await get_account(user_id=42)
    assert account == AccountSnapshot(name="alice", balance=5, total_earned=5, total_spent=0)
    await engine.dispose()
    await global_state_engine.dispose()


async def test_ensure_schema_serializes_concurrent_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent first-use schema bootstrap does not race SQLite CREATE TABLE."""
    db_path = tmp_path / "concurrent-economy.db"
    global_state_db_path = tmp_path / "concurrent-global-state.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    global_state_engine = create_async_engine(url=f"sqlite+aiosqlite:///{global_state_db_path}")
    monkeypatch.setattr("discordbot.cogs._economy.database._engine", engine)
    monkeypatch.setattr(
        "discordbot.cogs._economy.database._global_state_engine", global_state_engine
    )
    monkeypatch.setattr("discordbot.cogs._economy.database._schema_ready_for", None)
    monkeypatch.setattr("discordbot.cogs._economy.database._global_state_schema_ready_for", None)

    await asyncio.gather(*(_ensure_schema() for _ in range(20)))

    async with open_session() as session:
        result = await session.execute(
            statement=text(
                text="SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'loan_proposal'"
            )
        )
        assert result.scalar_one_or_none() == "loan_proposal"
    async with open_global_state_session() as session:
        result = await session.execute(
            statement=select(JackpotPool.pool_balance).where(JackpotPool.game_id == "dragon_gate")
        )
        assert result.scalar_one() == 1_000
    await engine.dispose()
    await global_state_engine.dispose()


async def test_get_balance_unknown_user_returns_zero() -> None:
    """Reading a never-seen user returns zero, not an error."""
    assert await get_balance(user_id=999) == 0


async def test_transfer_moves_currency_between_users() -> None:
    """Successful transfer debits sender the full amount and credits the taxed net."""
    await _add_balance(user_id=1, name="alice", amount=200)
    result = await transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=80
    )
    # 80 transferred, 5% (4) burned, receiver nets 76.
    assert result == TransferResult(
        sender_balance=120, receiver_balance=76, received_amount=76, tax_amount=4
    )
    assert await get_balance(user_id=1) == 120
    assert await get_balance(user_id=2) == 76


async def test_transfer_burns_tax_and_preserves_invariant() -> None:
    """The transfer tax removes points from circulation and keeps each side's invariant."""
    await _add_balance(user_id=1, name="alice", amount=1_000)
    await _add_balance(user_id=2, name="bob", amount=0)

    result = await transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=1_000
    )

    assert result is not None
    # 5% of 1000 is burned; receiver nets 950.
    assert result.tax_amount == 50
    assert result.received_amount == 950

    sender = await get_account(user_id=1)
    receiver = await get_account(user_id=2)
    assert sender is not None
    assert receiver is not None
    # Sender spent the full amount; receiver only earned the net. The 50 burned
    # is gone from total circulation (1000 in, 950 + 0 left).
    assert sender == AccountSnapshot(
        name="alice", balance=0, total_earned=1_000, total_spent=1_000
    )
    assert receiver == AccountSnapshot(name="bob", balance=950, total_earned=950, total_spent=0)
    assert sender.balance == sender.total_earned - sender.total_spent
    assert receiver.balance == receiver.total_earned - receiver.total_spent


async def test_transfer_rejects_self() -> None:
    """Transfers to oneself must be rejected."""
    await _add_balance(user_id=1, name="alice", amount=100)
    result = await transfer(
        sender_id=1, sender_name="alice", receiver_id=1, receiver_name="alice", amount=10
    )
    assert result is None
    assert await get_balance(user_id=1) == 100


async def test_transfer_rejects_insufficient_balance() -> None:
    """Transfers exceeding the sender's balance must be rejected."""
    await _add_balance(user_id=1, name="alice", amount=10)
    result = await transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=100
    )
    assert result is None
    assert await get_balance(user_id=1) == 10
    assert await get_balance(user_id=2) == 0


async def test_transfer_prevents_concurrent_double_spend() -> None:
    """Concurrent transfers from one sender cannot reuse the same points."""
    await _add_balance(user_id=1, name="alice", amount=100)
    results = await asyncio.gather(
        transfer(sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=80),
        transfer(
            sender_id=1, sender_name="alice", receiver_id=3, receiver_name="carol", amount=80
        ),
    )
    assert sum(result is not None for result in results) == 1
    assert results.count(None) == 1
    assert await get_balance(user_id=1) == 20
    # Whichever transfer won, the receiver nets 80 minus the 5% (4) tax burn.
    assert await get_balance(user_id=2) + await get_balance(user_id=3) == 76


async def test_transfer_concurrent_credits_accumulate() -> None:
    """Concurrent transfers into one receiver must not lose either credit."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await _add_balance(user_id=2, name="bob", amount=100)
    results = await asyncio.gather(
        transfer(
            sender_id=1, sender_name="alice", receiver_id=3, receiver_name="carol", amount=80
        ),
        transfer(sender_id=2, sender_name="bob", receiver_id=3, receiver_name="carol", amount=70),
    )
    assert all(result is not None for result in results)
    assert {result.sender_balance for result in results if result is not None} == {20, 30}
    # 80 nets 76 and 70 nets 67 after the 5% burn; both credits accumulate to 143.
    assert max(result.receiver_balance for result in results if result is not None) == 143
    assert await get_balance(user_id=3) == 143


@pytest.mark.parametrize(argnames="amount", argvalues=[0, -1, -1000])
async def test_transfer_rejects_non_positive(amount: int) -> None:
    """Transfers with non-positive amounts must be rejected."""
    await _add_balance(user_id=1, name="alice", amount=100)
    result = await transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=amount
    )
    assert result is None


async def test_top_n_orders_by_balance_descending() -> None:
    """Leaderboard returns the top accounts ordered by balance."""
    await _add_balance(user_id=1, name="alice", amount=100, avatar_url="https://cdn/a.png")
    await _add_balance(user_id=2, name="bob", amount=300, avatar_url="https://cdn/b.png")
    await _add_balance(user_id=3, name="carol", amount=50)
    rows = await top_n(limit=2)
    assert rows == [
        LeaderboardEntry(user_id=2, name="bob", balance=300, avatar_url="https://cdn/b.png"),
        LeaderboardEntry(user_id=1, name="alice", balance=100, avatar_url="https://cdn/a.png"),
    ]


async def test_top_n_excludes_specified_users() -> None:
    """Excluded user IDs (e.g. the bot's house ledger) must not appear in the result."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await _add_balance(user_id=2, name="bob", amount=300)
    await _add_balance(user_id=99, name="house", amount=999)
    rows = await top_n(limit=10, exclude_user_ids=(99,))
    assert all(row.user_id != 99 for row in rows)
    assert rows[0] == LeaderboardEntry(user_id=2, name="bob", balance=300, avatar_url="")


async def test_top_n_excludes_leaderboard_hidden_accounts_by_default() -> None:
    """Accounts marked hidden do not appear on the public balance leaderboard."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await _add_balance(user_id=2, name="bob", amount=300)
    await _add_balance(user_id=3, name="carol", amount=200)
    async with open_session() as session:
        await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == 2)
            .values(hide_from_leaderboard=True)
        )
        await session.commit()

    rows = await top_n(limit=2)
    assert rows == [
        LeaderboardEntry(user_id=3, name="carol", balance=200, avatar_url=""),
        LeaderboardEntry(user_id=1, name="alice", balance=100, avatar_url=""),
    ]


async def test_top_n_can_include_leaderboard_hidden_accounts() -> None:
    """Maintenance callers can still enumerate hidden accounts when needed."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await _add_balance(user_id=2, name="bob", amount=300)
    async with open_session() as session:
        await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == 2)
            .values(hide_from_leaderboard=True)
        )
        await session.commit()

    rows = await top_n(limit=2, include_hidden=True)
    assert rows[0] == LeaderboardEntry(user_id=2, name="bob", balance=300, avatar_url="")


async def test_top_n_none_limit_returns_all_matching_accounts() -> None:
    """Maintenance callers can request every matching account without a sentinel limit."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await _add_balance(user_id=2, name="bob", amount=300)
    await _add_balance(user_id=3, name="carol", amount=200)

    rows = await top_n(limit=None)
    assert [row.user_id for row in rows] == [2, 3, 1]


async def test_top_n_db_order_handles_large_zero_and_negative_balances() -> None:
    """DB-side ordering keeps decimal-text balances in numeric order."""
    await adjust_balance(user_id=1, name="huge", delta=10**30)
    await adjust_balance(user_id=2, name="small", delta=999)
    await adjust_balance(user_id=3, name="zero", delta=0)
    await adjust_balance(user_id=4, name="minus_one", delta=-1, allow_negative=True)
    await adjust_balance(user_id=5, name="minus_ten", delta=-10, allow_negative=True)
    await adjust_balance(user_id=6, name="minus_two", delta=-2, allow_negative=True)

    rows = await top_n(limit=None)

    assert [(row.user_id, row.balance) for row in rows] == [
        (1, 10**30),
        (2, 999),
        (4, -1),
        (6, -2),
        (5, -10),
    ]


async def test_top_n_short_cache_hit_and_manual_invalidation() -> None:
    """Repeated leaderboard reads use cached rows until explicitly invalidated."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await _add_balance(user_id=2, name="bob", amount=50)

    assert [row.user_id for row in await top_n(limit=1)] == [1]
    async with open_session() as session:
        await session.execute(
            statement=update(UserWallet)
            .where(UserWallet.user_id == 2)
            .values(balance=1_000, total_earned=1_000, total_spent=0)
        )
        await session.commit()

    assert [row.user_id for row in await top_n(limit=1)] == [1]
    invalidate_economy_leaderboard_cache()
    assert [row.user_id for row in await top_n(limit=1)] == [2]


async def test_top_n_write_path_invalidates_cache() -> None:
    """Balance writes clear cached leaderboard rows."""
    await _add_balance(user_id=1, name="alice", amount=100)
    await _add_balance(user_id=2, name="bob", amount=50)

    assert [row.user_id for row in await top_n(limit=1)] == [1]
    await credit_with_repayment(user_id=2, name="bob", amount=200)

    assert [row.user_id for row in await top_n(limit=1)] == [2]


async def test_apply_round_settlement_allows_negative_casino_balance() -> None:
    """Casino ledger keeps a true running net even when the casino is down."""
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=500, casino_delta=-500
    )
    ledger = await get_casino_ledger()
    assert ledger.balance == -500


async def test_apply_round_settlement_casino_accumulates_gross_flows() -> None:
    """Wins and losses both accumulate gross totals, not just the net balance."""
    await _add_balance(user_id=1, name="alice", amount=200)
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-200, casino_delta=200
    )
    await apply_round_settlement(
        player_id=2, player_account_name="bob", player_delta=300, casino_delta=-300
    )
    ledger = await get_casino_ledger()
    assert ledger.balance == -100
    assert ledger.total_earned == 200
    assert ledger.total_spent == 300


async def test_settle_wager_updates_player_and_casino() -> None:
    """Shared wager settlement applies net delta and mirrors casino P&L."""
    await _add_balance(user_id=1, name="alice", amount=100)

    settlement = await settle_wager(player_id=1, player_account_name="alice", delta=40)
    assert settlement.payout == 40
    assert settlement.new_balance == 140
    assert settlement.casino_balance == -40
    ledger = await get_casino_ledger()
    assert ledger.balance == -40


async def test_get_account_returns_none_for_unseen_user() -> None:
    """Unknown users return None instead of a synthetic zero row."""
    assert await get_account(user_id=12345) is None


async def test_settle_blackjack_player_updates_player_and_casino() -> None:
    """Shared Blackjack settlement applies net delta and mirrors casino P&L."""
    await _add_balance(user_id=1, name="alice", amount=100)

    participant = _participant()
    round_state = _round_from_cards(
        player_cards=[Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")],
        dealer_cards=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
        participant=participant,
        finished=True,
    )

    settlement = await settle_blackjack_player(
        round_state=round_state,
        player=round_state.players[0],
        player_id=1,
        player_account_name="alice",
    )
    assert settlement.delta == 50
    assert settlement.payout == 50
    assert settlement.new_balance == 150
    assert settlement.casino_balance == -50
    ledger = await get_casino_ledger()
    assert ledger.balance == -50


async def test_blackjack_view_finalizes_once_when_called_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent finalization attempts must not pay out one Blackjack hand twice."""
    cleanup_messages: list[_MessageStub] = []

    def fake_schedule_public_message_delete(
        message: _MessageStub, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records the final message scheduled for cleanup."""
        cleanup_messages.append(message)

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.schedule_public_message_delete",
        fake_schedule_public_message_delete,
    )
    await _add_balance(user_id=1, name="alice", amount=100)

    dealer = _DealerStub()
    message = _MessageStub()
    participant = _participant()
    view = BlackjackView(
        narrator=dealer,
        round_state=_round_from_cards(
            player_cards=[Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")],
            dealer_cards=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
            participant=participant,
            finished=True,
        ),
        starter_id=1,
        author_name="alice",
    )

    await asyncio.gather(view.finalize(message=message), view.finalize(message=message))

    assert await get_balance(user_id=1) == 150
    _ledger = await get_casino_ledger()
    assert _ledger.balance == -50
    assert "embeds" not in message.edits[0]
    await view.wait_for_background_tasks()
    assert dealer.settle_calls == 1
    assert message.edit_calls == 3
    assert message.edits[1]["view"] is None
    assert cleanup_messages == [message]


async def test_blackjack_view_timeout_auto_stands_and_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A player who walks away is treated as standing and the wager resolves."""
    cleanup_messages: list[_MessageStub] = []

    def fake_schedule_public_message_delete(
        message: _MessageStub, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records the final message scheduled for cleanup."""
        cleanup_messages.append(message)

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.schedule_public_message_delete",
        fake_schedule_public_message_delete,
    )
    await _add_balance(user_id=1, name="alice", amount=100)

    dealer = _DealerStub()
    message = _MessageStub()
    participant = _participant()
    view = BlackjackView(
        narrator=dealer,
        round_state=_round_from_cards(
            player_cards=[Card(rank="10", suit="♠"), Card(rank="8", suit="♥")],
            dealer_cards=[Card(rank="10", suit="♣"), Card(rank="Q", suit="♦")],
            participant=participant,
            finished=False,
        ),
        starter_id=1,
        author_name="alice",
    )
    view.message = message

    await view.on_timeout()

    assert view.round_state.finished is True
    assert await get_balance(user_id=1) == 50
    _ledger = await get_casino_ledger()
    assert _ledger.balance == 50
    assert "embeds" not in message.edits[0]
    await view.wait_for_background_tasks()
    assert dealer.settle_calls == 1
    assert message.edit_calls == 3
    assert message.edits[1]["view"] is None
    assert cleanup_messages == [message]


async def test_blackjack_view_final_edit_does_not_wait_for_settlement_banter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final results are visible before slow DealerAI settlement banter returns."""
    cleanup_messages: list[_MessageStub] = []

    def fake_schedule_public_message_delete(
        message: _MessageStub, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records the final message scheduled for cleanup."""
        cleanup_messages.append(message)

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.schedule_public_message_delete",
        fake_schedule_public_message_delete,
    )
    await _add_balance(user_id=1, name="alice", amount=100)

    dealer = _SlowSettleDealerStub()
    message = _MessageStub()
    participant = _participant()
    view = BlackjackView(
        narrator=dealer,
        round_state=_round_from_cards(
            player_cards=[Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")],
            dealer_cards=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
            participant=participant,
            finished=True,
        ),
        starter_id=1,
        author_name="alice",
    )

    await view.finalize(message=message)

    assert message.edit_calls == 2
    assert message.edits[1]["view"] is None
    await asyncio.wait_for(fut=dealer.settle_started.wait(), timeout=1)
    assert message.edit_calls == 2

    dealer.release_settle.set()
    await view.wait_for_background_tasks()

    assert dealer.settle_calls == 1
    assert message.edit_calls == 3
    assert cleanup_messages == [message]


async def test_blackjack_view_dealer_plays_h17_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dealer plays deterministically under H17 (hits below 17, stands on hard 17+)."""
    cleanup_messages: list[_MessageStub] = []

    def fake_schedule_public_message_delete(
        message: _MessageStub, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records the final message scheduled for cleanup."""
        cleanup_messages.append(message)

    def draw_fixed_card(rng: Random) -> Card:
        """Returns a deterministic dealer draw."""
        return Card(rank="5", suit="♣")

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.schedule_public_message_delete",
        fake_schedule_public_message_delete,
    )
    monkeypatch.setattr("discordbot.cogs._games.blackjack.draw_card", draw_fixed_card)
    await _add_balance(user_id=1, name="alice", amount=100)

    participant = _participant()
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[participant], auto_play_dealer=False
    )
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    round_state.dealer = [Card(rank="10", suit="♣"), Card(rank="3", suit="♦")]
    round_state.phase = "player_actions"
    round_state.shoe = []

    dealer = _DealerStub()
    message = _MessageStub()
    view = BlackjackView(
        narrator=dealer, round_state=round_state, starter_id=1, author_name="alice"
    )

    await view.finalize(message=message)

    assert [str(card) for card in view.round_state.dealer] == ["10♣", "3♦", "5♣"]
    assert view.round_state.dealer_played is True
    assert await get_balance(user_id=1) == 50
    _ledger = await get_casino_ledger()
    assert _ledger.balance == 50
    assert "embeds" not in message.edits[0]
    final_embeds = cast("list[Any]", message.edits[1]["embeds"])
    description = cast("str", final_embeds[1].description)
    assert "規則: 13 hit 抽 5♣ → 18" in description
    await view.wait_for_background_tasks()
    assert dealer.settle_calls == 1
    assert cleanup_messages == [message]


async def test_blackjack_view_insurance_buttons_only_during_insurance_phase() -> None:
    """Insurance controls should be hidden outside the insurance decision phase."""
    participant = _participant()
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[participant], auto_play_dealer=False
    )
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    round_state.dealer = [Card(rank="5", suit="♣"), Card(rank="A", suit="♦")]
    round_state.phase = "player_actions"
    view = BlackjackView(
        narrator=_DealerStub(), round_state=round_state, starter_id=1, author_name="alice"
    )

    view.sync_buttons()
    custom_ids = {child.custom_id for child in view.children if hasattr(child, "custom_id")}
    assert "bj:insure_yes" not in custom_ids
    assert "bj:insure_no" not in custom_ids

    round_state.phase = "insurance"
    view.sync_buttons()
    custom_ids = {child.custom_id for child in view.children if hasattr(child, "custom_id")}
    assert "bj:insure_yes" in custom_ids
    assert "bj:insure_no" in custom_ids

    round_state.phase = "player_actions"
    view.sync_buttons()
    custom_ids = {child.custom_id for child in view.children if hasattr(child, "custom_id")}
    assert "bj:insure_yes" not in custom_ids
    assert "bj:insure_no" not in custom_ids


async def test_blackjack_view_dealer_hits_soft_17(monkeypatch: pytest.MonkeyPatch) -> None:
    """Soft 17 forces a hit under the H17 rule."""
    cleanup_messages: list[_MessageStub] = []

    def fake_schedule_public_message_delete(
        message: _MessageStub, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records the final message scheduled for cleanup."""
        cleanup_messages.append(message)

    def draw_fixed_card(rng: Random) -> Card:
        """Returns a deterministic dealer draw of K (10 value) to land on 17."""
        return Card(rank="K", suit="♠")

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.schedule_public_message_delete",
        fake_schedule_public_message_delete,
    )
    monkeypatch.setattr("discordbot.cogs._games.blackjack.draw_card", draw_fixed_card)
    await _add_balance(user_id=1, name="alice", amount=100)

    participant = _participant()
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[participant], auto_play_dealer=False
    )
    round_state.players[0].hands[0].cards = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    round_state.dealer = [Card(rank="A", suit="♣"), Card(rank="6", suit="♦")]
    round_state.phase = "player_actions"
    round_state.shoe = []

    dealer = _DealerStub()
    message = _MessageStub()
    view = BlackjackView(
        narrator=dealer, round_state=round_state, starter_id=1, author_name="alice"
    )

    await view.finalize(message=message)

    # Soft 17 must trigger at least one draw, landing the dealer above 17.
    assert len(view.round_state.dealer) >= 3
    assert view.round_state.dealer_played is True
    assert "embeds" not in message.edits[0]
    final_embeds = cast("list[Any]", message.edits[1]["embeds"])
    description = cast("str", final_embeds[1].description)
    assert "規則: 17 hit" in description
    await view.wait_for_background_tasks()
    assert dealer.settle_calls == 1
    assert cleanup_messages == [message]


async def test_blackjack_view_locks_actions_while_finalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late Hit cannot mutate a hand that is already finalizing from Stand."""
    cleanup_messages: list[_MessageStub] = []
    settlement_started = asyncio.Event()
    continue_settlement = asyncio.Event()

    def fake_schedule_public_message_delete(
        message: _MessageStub, delay: float = 180, user_name: str | None = None
    ) -> None:
        """Records the final message scheduled for cleanup."""
        cleanup_messages.append(message)

    async def delayed_settle_blackjack_player(**_kwargs: Any) -> BlackjackPlayerSettlement:  # noqa: ANN401 -- test double accepts heterogeneous kwargs
        """Blocks settlement until the test releases the finalization lock."""
        settlement_started.set()
        await continue_settlement.wait()
        return BlackjackPlayerSettlement(
            outcome="win",
            delta=50,
            payout=50,
            new_balance=150,
            casino_balance=-50,
            detail="win",
            hands=[
                BlackjackHandSettlement(
                    cards=[Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")],
                    bet=50,
                    outcome="win",
                    delta=50,
                )
            ],
        )

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.schedule_public_message_delete",
        fake_schedule_public_message_delete,
    )
    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.settle_blackjack_player",
        delayed_settle_blackjack_player,
    )

    dealer = _DealerStub()
    message = _MessageStub()
    participant = _participant(balance_at_start=50)
    view = BlackjackView(
        narrator=dealer,
        round_state=_round_from_cards(
            player_cards=[Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")],
            dealer_cards=[Card(rank="10", suit="♣"), Card(rank="8", suit="♦")],
            participant=participant,
            finished=False,
        ),
        starter_id=1,
        author_name="alice",
    )

    hit_button = next(child for child in view.children if child.custom_id == "bj:hit")
    stand_button = next(child for child in view.children if child.custom_id == "bj:stand")
    stand_task = asyncio.create_task(coro=stand_button.callback(_InteractionStub(message=message)))
    await settlement_started.wait()

    assert message.edit_calls == 1
    in_flight_view = cast("BlackjackView", message.edits[0]["view"])
    assert all(child.disabled for child in in_flight_view.children)

    hit_task = asyncio.create_task(coro=hit_button.callback(_InteractionStub(message=message)))
    await asyncio.sleep(delay=0)

    assert len(view.round_state.players[0].hands[0].cards) == 2
    continue_settlement.set()
    await asyncio.gather(stand_task, hit_task)

    assert len(view.round_state.players[0].hands[0].cards) == 2
    assert dealer.hint_calls == 0
    assert "embeds" not in message.edits[0]
    await view.wait_for_background_tasks()
    assert dealer.settle_calls == 1
    assert message.edit_calls == 3
    assert message.edits[1]["view"] is None
    assert cleanup_messages == [message]


async def test_blackjack_view_rejects_stale_double_without_mutating_next_player() -> None:
    """A stale Double interaction cannot double the next active player's hand."""
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(),
        participants=[
            _participant(user_id=1, account_name="alice", display_name="Alice"),
            _participant(user_id=2, account_name="bob", display_name="Bob"),
        ],
        auto_play_dealer=False,
    )
    alice = round_state.players[0].hands[0]
    bob = round_state.players[1].hands[0]
    alice.cards = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    alice.finished = True
    bob.cards = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    round_state.dealer = [Card(rank="9", suit="♣"), Card(rank="7", suit="♦")]
    round_state.current_player_index = 1

    dealer = _DealerStub()
    message = _MessageStub()
    view = BlackjackView(
        narrator=dealer, round_state=round_state, starter_id=1, author_name="alice"
    )

    double_button = next(child for child in view.children if child.custom_id == "bj:double")
    interaction = _InteractionStub(message=message, user_id=1)
    await double_button.callback(interaction)

    assert bob.bet == 50
    assert [str(card) for card in bob.cards] == ["5♣", "6♦"]
    assert interaction.followup.sent[0]["content"] == "這個操作已經失效，請看最新牌桌"
    assert interaction.followup.sent[0]["ephemeral"] is True
    assert message.edit_calls == 1
    assert dealer.hint_calls == 0


async def test_blackjack_view_rejects_stale_hit_without_drawing_for_next_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale Hit interaction cannot draw a card for the next active player."""

    def fail_draw(rng: Random) -> Card:
        """Fails the test if stale Hit reaches card draw."""
        raise AssertionError("stale hit should not draw")

    monkeypatch.setattr("discordbot.cogs._games.blackjack.draw_card", fail_draw)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(),
        participants=[
            _participant(user_id=1, account_name="alice", display_name="Alice"),
            _participant(user_id=2, account_name="bob", display_name="Bob"),
        ],
        auto_play_dealer=False,
    )
    alice = round_state.players[0].hands[0]
    bob = round_state.players[1].hands[0]
    alice.cards = [Card(rank="10", suit="♠"), Card(rank="7", suit="♥")]
    alice.finished = True
    bob.cards = [Card(rank="5", suit="♣"), Card(rank="6", suit="♦")]
    round_state.dealer = [Card(rank="9", suit="♣"), Card(rank="7", suit="♦")]
    round_state.current_player_index = 1
    round_state.shoe = []

    message = _MessageStub()
    view = BlackjackView(
        narrator=_DealerStub(), round_state=round_state, starter_id=1, author_name="alice"
    )

    hit_button = next(child for child in view.children if child.custom_id == "bj:hit")
    interaction = _InteractionStub(message=message, user_id=1)
    await hit_button.callback(interaction)

    assert [str(card) for card in bob.cards] == ["5♣", "6♦"]
    assert interaction.followup.sent[0]["content"] == "這個操作已經失效，請看最新牌桌"
    assert message.edit_calls == 1


async def test_blackjack_view_hit_hint_uses_active_split_hand_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hit hint should describe the active split hand, not the first hand."""

    def draw_five(rng: Random) -> Card:
        """Returns a deterministic card for the active split hand."""
        return Card(rank="5", suit="♣")

    monkeypatch.setattr("discordbot.cogs._games.blackjack.draw_card", draw_five)
    participant = _participant(user_id=1, account_name="alice", display_name="Alice")
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[participant], auto_play_dealer=False
    )
    player = round_state.players[0]
    player.hands = [
        BlackjackHandState(
            cards=[Card(rank="10", suit="♠"), Card(rank="2", suit="♥")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
            finished=True,
        ),
        BlackjackHandState(
            cards=[Card(rank="9", suit="♣"), Card(rank="2", suit="♦")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
        ),
    ]
    round_state.dealer = [Card(rank="9", suit="♥"), Card(rank="7", suit="♦")]
    round_state.current_hand_index = 1
    round_state.shoe = []

    dealer = _DealerStub()
    message = _MessageStub()
    view = BlackjackView(
        narrator=dealer, round_state=round_state, starter_id=1, author_name="alice"
    )

    hit_button = next(child for child in view.children if child.custom_id == "bj:hit")
    await hit_button.callback(_InteractionStub(message=message, user_id=1))

    assert [str(card) for card in player.hands[1].cards] == ["9♣", "2♦", "5♣"]
    assert dealer.hint_calls == 0
    assert message.edit_calls == 1

    await view.wait_for_background_tasks()

    assert dealer.hint_calls == 1
    assert dealer.hints[0]["player_total"] == 16
    assert message.edit_calls == 2


async def test_add_balance_concurrent_credits_accumulate() -> None:
    """Verifies that concurrent credits on the same user do not lose updates."""
    await _add_balance(user_id=42, name="alice", amount=100)
    await asyncio.gather(*[_add_balance(user_id=42, name="alice", amount=10) for _ in range(20)])
    assert await get_balance(user_id=42) == 300


async def test_add_balance_concurrent_first_sight_does_not_raise() -> None:
    """Verifies that concurrent first-sight credits merge instead of racing."""
    results = await asyncio.gather(*[
        _add_balance(user_id=42, name="alice", amount=10) for _ in range(8)
    ])
    assert all(isinstance(value, int) for value in results)
    assert await get_balance(user_id=42) == 80


async def test_apply_round_settlement_concurrent_credits_accumulate() -> None:
    """Concurrent positive settlements on the same user must not lose updates."""
    await _add_balance(user_id=42, name="alice", amount=100)
    await asyncio.gather(*[
        apply_round_settlement(
            player_id=42, player_account_name="alice", player_delta=10, casino_delta=-10
        )
        for _ in range(10)
    ])
    assert await get_balance(user_id=42) == 200


async def test_apply_round_settlement_concurrent_casino_updates_accumulate() -> None:
    """Verifies that concurrent casino ledger settlements accumulate."""
    for user_id in range(10):
        await _add_balance(user_id=user_id, name=f"player{user_id}", amount=10)
    await asyncio.gather(*[
        apply_round_settlement(
            player_id=user_id,
            player_account_name=f"player{user_id}",
            player_delta=-10,
            casino_delta=10,
        )
        for user_id in range(10)
    ])
    ledger = await get_casino_ledger()
    assert ledger.balance == 100
    assert ledger.total_earned == 100
    assert ledger.total_spent == 0


async def test_apply_round_settlement_is_atomic() -> None:
    """Player delta and casino mirror share one transaction and one return."""
    await _add_balance(user_id=1, name="alice", amount=100)

    result = await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=40, casino_delta=-40
    )
    assert result.player_balance == 140
    assert result.casino_balance == -40
    assert await get_balance(user_id=1) == 140
    ledger = await get_casino_ledger()
    assert ledger.balance == -40


async def test_apply_round_settlement_loss_debits_player_and_casino() -> None:
    """A loss debits the player and credits the casino."""
    await _add_balance(user_id=1, name="alice", amount=100)

    result = await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-40, casino_delta=40
    )
    assert result.player_balance == 60
    assert result.casino_balance == 40
    account = await get_account(user_id=1)
    assert account is not None
    assert account.total_earned == 100
    assert account.total_spent == 40


async def test_apply_round_settlement_loss_clamps_player_and_casino_to_available_balance() -> None:
    """Deferred settlement stops at zero and only credits the casino with actual debit."""
    await _add_balance(user_id=1, name="alice", amount=25)

    result = await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-40, casino_delta=40
    )

    assert result.player_balance == 0
    assert result.casino_balance == 25
    account = await get_account(user_id=1)
    assert account is not None
    assert account.total_spent == 25


async def test_apply_round_settlement_updates_daily_casino_counters() -> None:
    """Blackjack-style player settlements persist gross loss, gross win, and net."""
    await _add_balance(user_id=1, name="alice", amount=1_000)

    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-300, casino_delta=300
    )
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=500, casino_delta=-500
    )

    loss, win, net, day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (300, 500, 200)
    assert day_started_at is not None
    assert _as_taipei(dt=day_started_at) == _taipei_midnight(now=_database_now())
    stats = await get_casino_daily_stats(user_id=1)
    assert stats.daily_loss == 300
    assert stats.daily_win == 500
    assert stats.daily_net == 200


async def test_daily_casino_counters_store_large_values_as_text() -> None:
    """Casino counters can exceed SQLite's INTEGER range without becoming REAL."""
    await _add_balance(user_id=1, name="alice", amount=1)
    large_loss = 10**20

    async with open_session() as session:
        now = _database_now()
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=1, name="alice", delta=-large_loss, now=now
        )
        await _apply_daily_casino_delta_in_session(
            session=session, user_id=1, name="alice", delta=-7, now=now
        )
        await session.commit()

    async with open_session() as session:
        result = await session.execute(
            statement=text(
                text="""
                SELECT daily_loss, typeof(daily_loss), daily_win, typeof(daily_win), daily_net, typeof(daily_net)
                  FROM casino_account
                 WHERE user_id = 1
                """
            )
        )
        counter_row = result.one()

    rows = await top_losers(limit=10)

    assert counter_row == (
        str(large_loss + 7),
        "text",
        "0",
        "text",
        str(-(large_loss + 7)),
        "text",
    )
    assert rows == [
        LossLeaderboardEntry(user_id=1, name="alice", loss_amount=large_loss + 7, avatar_url="")
    ]


async def test_wallet_and_jackpot_store_large_values_as_text() -> None:
    """Core wallet and jackpot money columns can exceed SQLite's INTEGER range."""
    large_amount = 10**20

    await adjust_balance(user_id=1, name="alice", delta=large_amount)
    result = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=-large_amount, game_id="dragon_gate"
    )

    async with open_session() as session:
        wallet_result = await session.execute(
            statement=text(
                text="""
                SELECT balance, typeof(balance), total_earned, typeof(total_earned), total_spent, typeof(total_spent)
                  FROM user_wallet
                 WHERE user_id = 1
                """
            )
        )
        wallet_row = wallet_result.one()
    async with open_global_state_session() as session:
        jackpot_result = await session.execute(
            statement=text(
                text="""
                SELECT pool_balance, typeof(pool_balance), total_contributed, typeof(total_contributed)
                  FROM jackpot_pool
                 WHERE game_id = 'dragon_gate'
                """
            )
        )
        jackpot_row = jackpot_result.one()

    assert result.player_balance == 0
    assert result.applied_player_delta == -large_amount
    assert wallet_row == ("0", "text", str(large_amount), "text", str(large_amount), "text")
    assert jackpot_row == (str(1_000 + large_amount), "text", str(large_amount), "text")


async def test_daily_casino_counters_skip_push_and_house_ledger() -> None:
    """Zero deltas and dealer ledger mirrors do not enter player loss counters."""
    await _add_balance(user_id=1, name="alice", amount=100)

    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=0, casino_delta=0
    )
    assert await _daily_casino_stats(user_id=1) == (0, 0, 0, None)

    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-40, casino_delta=40
    )
    assert await _daily_casino_stats(user_id=99) == (0, 0, 0, None)


# Daily check-in ------------------------------------------------------------


async def test_checkin_first_time_credits_base_reward() -> None:
    """A first check-in pays the base reward and persists a streak of 1."""
    result = await checkin(user_id=1, name="alice")
    assert result is not None
    assert result.amount == BASE_CHECKIN_REWARD_AMOUNT
    assert result.streak == 1
    assert result.is_vip is False
    assert result.new_balance == BASE_CHECKIN_REWARD_AMOUNT


async def test_checkin_same_day_is_rejected() -> None:
    """A second check-in within the same Taipei day must return None."""
    first = await checkin(user_id=1, name="alice")
    assert first is not None
    second = await checkin(user_id=1, name="alice")
    assert second is None
    assert await get_balance(user_id=1) == first.new_balance


async def test_checkin_consecutive_day_advances_streak() -> None:
    """A check-in on the next calendar day bumps the streak by 1."""
    first = await checkin(user_id=1, name="alice")
    assert first is not None
    # Backdate the previous check-in to yesterday Taipei
    yesterday = datetime.now(tz=TAIWAN_TIMEZONE) - timedelta(days=1)
    async with open_session() as session:
        await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == 1)
            .values(last_checkin_at=yesterday)
        )
        await session.commit()
    second = await checkin(user_id=1, name="alice")
    assert second is not None
    assert second.streak == 2
    assert second.amount > first.amount


async def test_checkin_streak_cycles_back_to_one_after_seven() -> None:
    """Day 8 in a row resets back to streak 1."""
    await checkin(user_id=1, name="alice")
    async with open_session() as session:
        await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == 1)
            .values(
                last_checkin_at=datetime.now(tz=TAIWAN_TIMEZONE) - timedelta(days=1),
                checkin_streak=CHECKIN_STREAK_CYCLE,
            )
        )
        await session.commit()
    result = await checkin(user_id=1, name="alice")
    assert result is not None
    assert result.streak == 1


async def test_checkin_missed_day_resets_streak_to_one() -> None:
    """Skipping a day resets the streak back to 1."""
    await checkin(user_id=1, name="alice")
    async with open_session() as session:
        await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == 1)
            .values(
                last_checkin_at=datetime.now(tz=TAIWAN_TIMEZONE) - timedelta(days=3),
                checkin_streak=4,
            )
        )
        await session.commit()
    result = await checkin(user_id=1, name="alice")
    assert result is not None
    assert result.streak == 1


async def test_checkin_vip_gets_double_base() -> None:
    """A VIP account starts at 2x base before the streak multiplier."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None
    result = await checkin(user_id=1, name="alice")
    assert result is not None
    assert result.is_vip is True
    assert result.amount == 2 * BASE_CHECKIN_REWARD_AMOUNT


@pytest.mark.parametrize(
    argnames=("streak", "is_vip", "expected"),
    argvalues=[
        (1, False, 500),
        (2, False, 750),
        (7, False, 2_000),
        (1, True, 1_000),
        (7, True, 4_000),
    ],
)
def test_checkin_reward_formula(streak: int, is_vip: bool, expected: int) -> None:
    """Streak + VIP combinations compute to the expected reward."""
    assert checkin_reward(streak=streak, is_vip=is_vip) == expected


async def test_checkin_updates_lifetime_totals() -> None:
    """A successful check-in counts as earned points."""
    result = await checkin(user_id=1, name="alice")
    assert result is not None
    account = await get_account(user_id=1)
    assert account == AccountSnapshot(
        name="alice", balance=result.amount, total_earned=result.amount, total_spent=0
    )


# VIP purchase --------------------------------------------------------------


async def test_buy_vip_sets_flag_and_debits_balance() -> None:
    """A successful purchase costs `VIP_PURCHASE_COST` and flips `is_vip`."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST + 100)
    result = await buy_vip(user_id=1, name="alice")
    assert result is not None
    assert result.new_balance == 100
    assert result.cost == VIP_PURCHASE_COST
    assert await get_vip(user_id=1) is True


async def test_buy_vip_rejects_insufficient_balance() -> None:
    """Users without enough points cannot purchase VIP."""
    await _add_balance(user_id=1, name="alice", amount=100)
    result = await buy_vip(user_id=1, name="alice")
    assert result is None
    assert await get_vip(user_id=1) is False


async def test_buy_vip_rejects_existing_vip() -> None:
    """A second purchase by an existing VIP returns None and does not re-debit."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST * 2)
    first = await buy_vip(user_id=1, name="alice")
    assert first is not None
    second = await buy_vip(user_id=1, name="alice")
    assert second is None
    assert await get_balance(user_id=1) == VIP_PURCHASE_COST


async def test_buy_vip_rejects_unseen_user() -> None:
    """A user without a row cannot purchase (no balance to debit)."""
    assert await buy_vip(user_id=999, name="ghost") is None


async def test_buy_vip_updates_lifetime_spent() -> None:
    """A successful purchase counts as spent points."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    await buy_vip(user_id=1, name="alice")
    account = await get_account(user_id=1)
    assert account == AccountSnapshot(
        name="alice", balance=0, total_earned=VIP_PURCHASE_COST, total_spent=VIP_PURCHASE_COST
    )


async def test_get_vip_unknown_user_returns_false() -> None:
    """Unknown users report no VIP perk rather than raising."""
    assert await get_vip(user_id=12345) is False


# Loss leaderboard ----------------------------------------------------------


async def test_top_losers_uses_gross_loss_not_net() -> None:
    """Winning later does not erase a player's gross loss leaderboard amount."""
    await _add_balance(user_id=1, name="alice", amount=1_000)
    await _add_balance(user_id=2, name="bob", amount=1_000)
    await _add_balance(user_id=3, name="carol", amount=1_000)
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-300, casino_delta=300
    )
    await apply_round_settlement(
        player_id=2, player_account_name="bob", player_delta=200, casino_delta=-200
    )
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=500, casino_delta=-500
    )
    await apply_round_settlement(
        player_id=3, player_account_name="carol", player_delta=-200, casino_delta=200
    )
    rows = await top_losers(limit=10, exclude_user_ids=(99,))
    assert rows == [
        LossLeaderboardEntry(user_id=1, name="alice", loss_amount=300, avatar_url=""),
        LossLeaderboardEntry(user_id=3, name="carol", loss_amount=200, avatar_url=""),
    ]


async def test_top_losers_orders_by_loss_magnitude() -> None:
    """The leaderboard sorts from biggest loss to smallest."""
    for user_id, name, loss in [(1, "alice", 100), (2, "bob", 500), (3, "carol", 250)]:
        await _add_balance(user_id=user_id, name=name, amount=loss)
        await apply_round_settlement(
            player_id=user_id, player_account_name=name, player_delta=-loss, casino_delta=loss
        )
    rows = await top_losers(limit=10, exclude_user_ids=(99,))
    assert [(row.user_id, row.loss_amount) for row in rows] == [(2, 500), (3, 250), (1, 100)]


async def test_top_losers_excludes_specified_users() -> None:
    """`exclude_user_ids` filters the house ledger out of the report."""
    await _add_balance(user_id=1, name="alice", amount=500)
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-500, casino_delta=500
    )
    rows = await top_losers(limit=10, exclude_user_ids=(99,))
    assert all(row.user_id != 99 for row in rows)


async def test_top_losers_excludes_leaderboard_hidden_accounts_by_default() -> None:
    """Hidden accounts do not appear on the public daily loss leaderboard."""
    await _add_balance(user_id=1, name="alice", amount=500)
    await _add_balance(user_id=2, name="bob", amount=400)
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-500, casino_delta=500
    )
    await apply_round_settlement(
        player_id=2, player_account_name="bob", player_delta=-400, casino_delta=400
    )
    async with open_session() as session:
        await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == 1)
            .values(hide_from_leaderboard=True)
        )
        await session.commit()

    rows = await top_losers(limit=10, exclude_user_ids=(99,))
    assert rows == [LossLeaderboardEntry(user_id=2, name="bob", loss_amount=400, avatar_url="")]


async def test_top_losers_can_include_leaderboard_hidden_accounts() -> None:
    """Maintenance callers can include hidden accounts in daily loss queries."""
    await _add_balance(user_id=1, name="alice", amount=500)
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-500, casino_delta=500
    )
    async with open_session() as session:
        await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == 1)
            .values(hide_from_leaderboard=True)
        )
        await session.commit()

    rows = await top_losers(limit=10, exclude_user_ids=(99,), include_hidden=True)
    assert rows == [LossLeaderboardEntry(user_id=1, name="alice", loss_amount=500, avatar_url="")]


async def test_top_losers_ignores_counters_before_today() -> None:
    """Stale account counters from an older Taipei day do not count."""
    await _add_balance(user_id=1, name="alice", amount=500)
    await apply_round_settlement(
        player_id=1, player_account_name="alice", player_delta=-500, casino_delta=500
    )
    past = datetime.now(tz=TAIWAN_TIMEZONE) - timedelta(days=2)
    async with open_session() as session:
        await session.execute(
            statement=update(CasinoAccount)
            .where(CasinoAccount.user_id == 1)
            .values(day_started_at=_taipei_midnight(now=past))
        )
        await session.commit()
    assert await top_losers(limit=10, exclude_user_ids=(99,)) == []


async def test_top_losers_empty_when_no_casino_activity() -> None:
    """Without any daily casino loss counters the leaderboard is empty."""
    await _add_balance(user_id=1, name="alice", amount=100)
    assert await top_losers(limit=10, exclude_user_ids=(99,)) == []


async def test_top_losers_ignores_manual_adjustments() -> None:
    """Manual admin debits do not count as casino losses."""
    await adjust_balance(user_id=1, name="alice", delta=-100, allow_negative=True)
    assert await top_losers(limit=10, exclude_user_ids=(99,)) == []


# VIP blackjack settlement -------------------------------------------------


async def test_settle_wager_applies_vip_bonus_on_win() -> None:
    """A VIP player wins 1.2x of the base delta; house mirrors the boosted amount."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None
    settlement = await settle_wager(player_id=1, player_account_name="alice", delta=100)
    assert settlement.delta == 120
    assert settlement.base_delta == 100
    assert settlement.vip_bonus == 20
    assert settlement.is_vip is True
    assert settlement.casino_balance == -120


async def test_settle_wager_keeps_loss_unchanged_for_vip() -> None:
    """The VIP perk does not soften losses."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST + 1_000)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None
    settlement = await settle_wager(player_id=1, player_account_name="alice", delta=-100)
    assert settlement.delta == -100
    assert settlement.base_delta == -100
    assert settlement.vip_bonus == 0
    assert settlement.is_vip is True
    assert settlement.casino_balance == 100


# Multi-hand Blackjack settlement -----------------------------------------


async def _settle_player(round_state: BlackjackRound) -> BlackjackPlayerSettlement:
    """Helper that runs settle_blackjack_player against the only player."""
    player = round_state.players[0]
    return await settle_blackjack_player(
        round_state=round_state,
        player=player,
        player_id=player.participant.user_id,
        player_account_name=player.participant.account_name,
    )


async def test_settle_blackjack_player_surrender_returns_half_bet() -> None:
    """Surrender refunds half the original bet and writes the audit row."""
    await _add_balance(user_id=1, name="alice", amount=100)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=50)]
    )
    hand = round_state.players[0].hands[0]
    hand.cards = [Card(rank="10", suit="♠"), Card(rank="6", suit="♥")]
    hand.surrendered = True
    hand.finished = True
    round_state.dealer = [Card(rank="10", suit="♣"), Card(rank="8", suit="♦")]
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    assert settlement.outcome == "surrender"
    assert settlement.delta == -25
    assert settlement.new_balance == 75
    assert settlement.casino_balance == 25


async def test_settle_blackjack_player_double_doubles_loss_when_dealer_higher() -> None:
    """Doubled hands lose 2x the original bet on settlement."""
    await _add_balance(user_id=1, name="alice", amount=200)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=50)]
    )
    hand = round_state.players[0].hands[0]
    hand.cards = [Card(rank="5", suit="♠"), Card(rank="6", suit="♥"), Card(rank="2", suit="♣")]
    hand.bet = 100
    hand.doubled = True
    hand.finished = True
    round_state.dealer = [Card(rank="10", suit="♣"), Card(rank="9", suit="♦")]
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    assert settlement.delta == -100
    assert settlement.new_balance == 100


async def test_settle_blackjack_player_split_both_wins_aggregates_delta() -> None:
    """Split hands aggregate into a single ledger write."""
    await _add_balance(user_id=1, name="alice", amount=200)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=50)]
    )
    player = round_state.players[0]
    player.hands = [
        BlackjackHandState(
            cards=[Card(rank="8", suit="♠"), Card(rank="K", suit="♥")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
            finished=True,
        ),
        BlackjackHandState(
            cards=[Card(rank="8", suit="♣"), Card(rank="9", suit="♦")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
            finished=True,
        ),
    ]
    round_state.dealer = [Card(rank="10", suit="♣"), Card(rank="6", suit="♦")]
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    assert settlement.delta == 100
    assert len(settlement.hands) == 2
    assert settlement.hands[0].outcome == "win"
    assert settlement.hands[1].outcome == "win"
    assert settlement.new_balance == 300


async def test_settle_blackjack_player_split_offset_skips_vip_bonus() -> None:
    """A split that nets to zero does not trigger the VIP bonus."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST + 200)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=50)]
    )
    player = round_state.players[0]
    player.hands = [
        BlackjackHandState(
            cards=[Card(rank="8", suit="♠"), Card(rank="K", suit="♥")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
            finished=True,
        ),
        BlackjackHandState(
            cards=[Card(rank="8", suit="♣"), Card(rank="2", suit="♦")],
            bet=50,
            base_bet=50,
            is_split_hand=True,
            finished=True,
        ),
    ]
    round_state.dealer = [Card(rank="10", suit="♣"), Card(rank="7", suit="♦")]
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    # hand1 win 50, hand2 lose 50 → net 0; VIP perk is suppressed on non-positive.
    assert settlement.base_delta == 0
    assert settlement.delta == 0
    assert settlement.vip_bonus == 0


async def test_settle_blackjack_player_five_card_non_21_wins_without_system_bonus() -> None:
    """Five-card non-bust hands win normally without the five-card 21 bonus."""
    await _add_balance(user_id=1, name="alice", amount=100_000)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=10_000, balance_at_start=100_000)]
    )
    player = round_state.players[0]
    hand = player.hands[0]
    hand.cards = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
        Card(rank="6", suit="♠"),
    ]
    hand.finished = True
    round_state.dealer = [
        Card(rank="7", suit="♣"),
        Card(rank="7", suit="♦"),
        Card(rank="7", suit="♥"),
    ]
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    assert settlement.outcome == "five_card_win"
    assert settlement.hands[0].five_card_twenty_one is False
    assert settlement.hands[0].five_card_bonus == 0
    assert settlement.base_delta == 10_000
    assert settlement.five_card_bonus == 0
    assert settlement.delta == 10_000
    assert settlement.new_balance == 110_000
    assert settlement.casino_balance == -10_000
    _ledger = await get_casino_ledger()
    assert _ledger.balance == -10_000
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (0, 10_000, 10_000)


async def test_settle_blackjack_player_five_card_bonus_excludes_house_ledger() -> None:
    """Five-card 21 pays the system bonus without moving the house ledger for it."""
    await _add_balance(user_id=1, name="alice", amount=100_000)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=10_000, balance_at_start=100_000)]
    )
    player = round_state.players[0]
    hand = player.hands[0]
    hand.cards = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
        Card(rank="7", suit="♠"),
    ]
    hand.finished = True
    round_state.dealer = [Card(rank="10", suit="♣"), Card(rank="9", suit="♦")]
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    assert settlement.outcome == "five_card_twenty_one"
    assert settlement.hands[0].five_card_twenty_one is True
    assert settlement.hands[0].five_card_bonus == 10_000
    assert settlement.base_delta == 10_000
    assert settlement.five_card_bonus == 10_000
    assert settlement.delta == 20_000
    assert settlement.new_balance == 120_000
    assert settlement.casino_balance == -10_000
    _ledger = await get_casino_ledger()
    assert _ledger.balance == -10_000
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (0, 20_000, 20_000)
    account = await get_account(user_id=1)
    assert account == AccountSnapshot(
        name="alice", balance=120_000, total_earned=120_000, total_spent=0
    )


async def test_settle_blackjack_player_five_card_vip_keeps_system_bonus_out_of_house() -> None:
    """VIP still boosts the regular win, while five-card bonus stays system-funded."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST + 100_000)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=10_000, balance_at_start=100_000)]
    )
    hand = round_state.players[0].hands[0]
    hand.cards = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
        Card(rank="7", suit="♠"),
    ]
    hand.finished = True
    round_state.dealer = [Card(rank="10", suit="♣"), Card(rank="9", suit="♦")]
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    assert settlement.base_delta == 10_000
    assert settlement.vip_bonus == 2_000
    assert settlement.five_card_bonus == 10_000
    assert settlement.delta == 22_000
    assert settlement.new_balance == 122_000
    assert settlement.casino_balance == -12_000
    _ledger = await get_casino_ledger()
    assert _ledger.balance == -12_000
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (0, 22_000, 22_000)


async def test_settle_blackjack_player_five_card_push_still_pays_bonus() -> None:
    """Dealer 21 pushes the main hand, but the five-card bonus still pays."""
    await _add_balance(user_id=1, name="alice", amount=100_000)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=10_000, balance_at_start=100_000)]
    )
    hand = round_state.players[0].hands[0]
    hand.cards = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
        Card(rank="7", suit="♠"),
    ]
    hand.finished = True
    round_state.dealer = [
        Card(rank="7", suit="♣"),
        Card(rank="7", suit="♦"),
        Card(rank="7", suit="♥"),
    ]
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    assert settlement.base_delta == 0
    assert settlement.five_card_bonus == 10_000
    assert settlement.delta == 10_000
    assert settlement.new_balance == 110_000
    assert settlement.casino_balance == 0
    _ledger = await get_casino_ledger()
    assert _ledger.balance == 0
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (0, 10_000, 10_000)


async def test_settle_blackjack_player_five_card_push_pays_vip_bonus_without_house() -> None:
    """VIP gets its five-card bonus even when the dealer also has 21."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST + 100_000)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=10_000, balance_at_start=100_000)]
    )
    hand = round_state.players[0].hands[0]
    hand.cards = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
        Card(rank="7", suit="♠"),
    ]
    hand.finished = True
    round_state.dealer = [
        Card(rank="7", suit="♣"),
        Card(rank="7", suit="♦"),
        Card(rank="7", suit="♥"),
    ]
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    assert settlement.base_delta == 0
    assert settlement.vip_bonus == 2_000
    assert settlement.five_card_bonus == 10_000
    assert settlement.delta == 12_000
    assert settlement.new_balance == 112_000
    assert settlement.casino_balance == 0
    _ledger = await get_casino_ledger()
    assert _ledger.balance == 0
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (0, 12_000, 12_000)


async def test_blackjack_final_embed_shows_five_card_bonus_metadata() -> None:
    """Final Blackjack embeds display five-card outcome and bonus metadata."""
    await _add_balance(user_id=1, name="alice", amount=100_000)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=10_000, balance_at_start=100_000)]
    )
    player = round_state.players[0]
    hand = player.hands[0]
    hand.cards = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
        Card(rank="7", suit="♠"),
    ]
    hand.finished = True
    round_state.dealer = [Card(rank="10", suit="♣"), Card(rank="9", suit="♦")]
    round_state.finished = True
    round_state.phase = "settled"
    settlement = await _settle_player(round_state=round_state)

    embeds = build_final_embeds(
        round_state=round_state,
        results=[BlackjackPlayerResult(participant=player.participant, settlement=settlement)],
    )
    player_embed = embeds[1]

    description = cast("str", player_embed.description)
    assert "## ✨ 過五關 · 21" in description
    assert "過五關 bonus `+1萬`" in description


async def test_blackjack_final_embed_shows_five_card_win_without_bonus_metadata() -> None:
    """Final Blackjack embeds display non-21 five-card wins without bonus metadata."""
    await _add_balance(user_id=1, name="alice", amount=100_000)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=10_000, balance_at_start=100_000)]
    )
    player = round_state.players[0]
    hand = player.hands[0]
    hand.cards = [
        Card(rank="2", suit="♠"),
        Card(rank="3", suit="♥"),
        Card(rank="4", suit="♣"),
        Card(rank="5", suit="♦"),
        Card(rank="6", suit="♠"),
    ]
    hand.finished = True
    round_state.dealer = [
        Card(rank="7", suit="♣"),
        Card(rank="7", suit="♦"),
        Card(rank="7", suit="♥"),
    ]
    round_state.finished = True
    round_state.phase = "settled"
    settlement = await _settle_player(round_state=round_state)

    embeds = build_final_embeds(
        round_state=round_state,
        results=[BlackjackPlayerResult(participant=player.participant, settlement=settlement)],
    )
    player_embed = embeds[1]

    description = cast("str", player_embed.description)
    assert "## 🎉 過五關 · 20" in description
    assert "過五關 bonus" not in description


async def test_settle_blackjack_player_insurance_won_with_dealer_blackjack() -> None:
    """Insurance pays 2:1 when peek confirms dealer Blackjack."""
    await _add_balance(user_id=1, name="alice", amount=300)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=100)]
    )
    player = round_state.players[0]
    hand = player.hands[0]
    hand.cards = [Card(rank="9", suit="♠"), Card(rank="8", suit="♥")]
    hand.finished = True
    player.insurance_bet = 50
    player.insurance_resolved = True
    round_state.dealer = [Card(rank="K", suit="♣"), Card(rank="A", suit="♦")]
    round_state.peeked_blackjack = True
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    assert settlement.insurance is not None
    assert settlement.insurance.won is True
    assert settlement.insurance.delta == 100
    assert settlement.base_delta == 0  # -100 main bet + +100 insurance
    assert settlement.delta == 0
    assert settlement.outcome == "push"


async def test_blackjack_final_embed_uses_aggregate_insurance_push_title() -> None:
    """Insurance break-even should present as aggregate push in the final title."""
    await _add_balance(user_id=1, name="alice", amount=300)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=100)]
    )
    player = round_state.players[0]
    hand = player.hands[0]
    hand.cards = [Card(rank="9", suit="♠"), Card(rank="8", suit="♥")]
    hand.finished = True
    player.insurance_bet = 50
    player.insurance_resolved = True
    round_state.dealer = [Card(rank="K", suit="♣"), Card(rank="A", suit="♦")]
    round_state.peeked_blackjack = True
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)
    embeds = build_final_embeds(
        round_state=round_state,
        results=[BlackjackPlayerResult(participant=player.participant, settlement=settlement)],
    )
    player_embed = embeds[1]

    description = cast("str", player_embed.description)
    assert "## 😢 你輸了 · 17 < 21" in description
    assert "保險 `50` → 中獎 `+100`" in description


async def test_settle_blackjack_player_insurance_lost_when_no_dealer_blackjack() -> None:
    """Insurance loses when the peek shows no Blackjack."""
    await _add_balance(user_id=1, name="alice", amount=300)
    round_state = BlackjackRound.from_participants(
        rng=SystemRandom(), participants=[_participant(bet=100)]
    )
    player = round_state.players[0]
    hand = player.hands[0]
    hand.cards = [Card(rank="K", suit="♠"), Card(rank="Q", suit="♥")]
    hand.finished = True
    player.insurance_bet = 50
    player.insurance_resolved = True
    round_state.dealer = [
        Card(rank="9", suit="♣"),
        Card(rank="A", suit="♦"),
        Card(rank="9", suit="♥"),
    ]
    round_state.peeked_blackjack = False
    round_state.dealer_played = True
    round_state.finished = True
    round_state.phase = "settled"

    settlement = await _settle_player(round_state=round_state)

    assert settlement.insurance is not None
    assert settlement.insurance.won is False
    assert settlement.insurance.delta == -50
    # main win 100 - insurance 50 = +50
    assert settlement.base_delta == 50
    assert settlement.outcome == "win"


async def test_apply_jackpot_settlement_credits_player_and_drains_pool() -> None:
    """Player wins pull points out of the jackpot row in one atomic step."""
    await _add_balance(user_id=1, name="alice", amount=10_000)
    # _ensure_schema already seeded the dragon_gate pool at 1_000.
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000

    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=200, game_id="dragon_gate"
    )

    assert settlement.player_balance == 10_200
    assert settlement.jackpot_balance == 800
    assert settlement.applied_player_delta == 200
    assert settlement.jackpot_depleted is False
    assert await get_jackpot_pool(game_id="dragon_gate") == 800
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (0, 200, 200)


async def test_apply_jackpot_settlement_replenishes_drained_seed_pool() -> None:
    """A seeded jackpot restores itself after a player wins the whole pool."""
    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=1_000, game_id="dragon_gate"
    )

    assert settlement.player_balance == 1_000
    assert settlement.jackpot_balance == 1_000
    assert settlement.applied_player_delta == 1_000
    assert settlement.jackpot_depleted is True
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000
    async with open_global_state_session() as session:
        result = await session.execute(
            statement=select(JackpotPool.seeded_amount, JackpotPool.total_claimed).where(
                JackpotPool.game_id == "dragon_gate"
            )
        )
        seeded_amount, total_claimed = result.one()
    assert seeded_amount == 2_000
    assert total_claimed == 1_000


async def test_apply_jackpot_settlement_clamps_loss_and_grows_pool_by_actual_debit() -> None:
    """Player losses stop at zero and feed the jackpot with the actual debit."""
    await _add_balance(user_id=1, name="alice", amount=15_000)

    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=-25_000, game_id="dragon_gate"
    )

    assert settlement.player_balance == 0
    assert settlement.jackpot_balance == 16_000
    assert settlement.applied_player_delta == -15_000
    account = await get_account(user_id=1)
    assert account == AccountSnapshot(
        name="alice", balance=0, total_earned=15_000, total_spent=15_000
    )
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (15_000, 0, -15_000)


async def test_apply_jackpot_settlement_concurrent_clamped_losses_count_actual_debit() -> None:
    """Concurrent clamped jackpot losses cannot over-credit the pool."""
    await _add_balance(user_id=1, name="alice", amount=100)

    first, second = await asyncio.gather(
        apply_jackpot_settlement(
            player_id=1, player_account_name="alice", player_delta=-80, game_id="dragon_gate"
        ),
        apply_jackpot_settlement(
            player_id=1, player_account_name="alice", player_delta=-80, game_id="dragon_gate"
        ),
    )

    applied_total = first.applied_player_delta + second.applied_player_delta
    assert applied_total == -100
    assert await get_balance(user_id=1) == 0
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_100
    account = await get_account(user_id=1)
    assert account == AccountSnapshot(name="alice", balance=0, total_earned=100, total_spent=100)
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (100, 0, -100)


async def test_apply_jackpot_settlement_caps_win_to_live_pool() -> None:
    """A stale oversized jackpot win only pays the live pool amount."""
    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=150_000, game_id="dragon_gate"
    )

    assert settlement.player_balance == 1_000
    assert settlement.applied_player_delta == 1_000
    assert settlement.jackpot_balance == 1_000
    assert settlement.jackpot_depleted is True


async def test_apply_jackpot_settlement_concurrent_wins_do_not_double_claim_pool() -> None:
    """Concurrent whole-pool wins cannot both claim the same jackpot generation."""
    snapshot = await get_jackpot_snapshot(game_id="dragon_gate")
    first, second = await asyncio.gather(
        apply_jackpot_settlement(
            player_id=1,
            player_account_name="alice",
            player_delta=1_000,
            game_id="dragon_gate",
            expected_jackpot_generation=snapshot.generation,
        ),
        apply_jackpot_settlement(
            player_id=2,
            player_account_name="bob",
            player_delta=1_000,
            game_id="dragon_gate",
            expected_jackpot_generation=snapshot.generation,
        ),
    )

    applied_total = first.applied_player_delta + second.applied_player_delta
    assert applied_total == 1_000
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000


async def test_apply_jackpot_settlement_batch_charges_multiple_players_atomically() -> None:
    """Batch jackpot settlements share one transaction and one final snapshot."""
    await _add_balance(user_id=1, name="alice", amount=10_000)
    await _add_balance(user_id=2, name="bob", amount=10_000)

    result = await apply_jackpot_settlement_batch(
        game_id="dragon_gate",
        settlements=(
            JackpotSettlementRequest(
                player_id=1, player_account_name="alice", player_delta=-5_000
            ),
            JackpotSettlementRequest(player_id=2, player_account_name="bob", player_delta=-7_000),
        ),
    )

    assert result.player_balances == {1: 5_000, 2: 3_000}
    assert result.applied_player_deltas == {1: -5_000, 2: -7_000}
    assert result.jackpot_balance == 13_000
    assert await get_jackpot_pool(game_id="dragon_gate") == 13_000


async def test_apply_jackpot_settlement_batch_rejects_required_full_debit() -> None:
    """Ante-style full-debit batches reject without partially charging anyone."""
    await _add_balance(user_id=1, name="alice", amount=10_000)
    await _add_balance(user_id=2, name="bob", amount=3_000)

    result = await apply_jackpot_settlement_batch(
        game_id="dragon_gate",
        settlements=(
            JackpotSettlementRequest(
                player_id=1,
                player_account_name="alice",
                player_delta=-5_000,
                require_full_debit=True,
            ),
            JackpotSettlementRequest(
                player_id=2,
                player_account_name="bob",
                player_delta=-5_000,
                require_full_debit=True,
            ),
        ),
    )

    assert result.rejected_player_ids == (2,)
    assert result.player_balances == {}
    assert result.applied_player_deltas == {}
    assert await get_balance(user_id=1) == 10_000
    assert await get_balance(user_id=2) == 3_000
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000


async def test_apply_jackpot_settlement_batch_rolls_back_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed batch ante settlement cannot partially charge players."""
    await _add_balance(user_id=1, name="alice", amount=10_000)
    await _add_balance(user_id=2, name="bob", amount=10_000)
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000

    calls = 0
    original_apply = _apply_jackpot_delta_in_session

    async def flaky_apply_jackpot_delta_in_session(
        **kwargs: Any,  # noqa: ANN401 -- test double accepts heterogeneous kwargs
    ) -> tuple[JackpotSnapshot, bool]:
        """Fails on the second jackpot write to test batch rollback."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced batch failure")
        return await original_apply(**kwargs)

    monkeypatch.setattr(
        "discordbot.cogs._economy.database._apply_jackpot_delta_in_session",
        flaky_apply_jackpot_delta_in_session,
    )

    with pytest.raises(expected_exception=RuntimeError, match="forced batch failure"):
        await apply_jackpot_settlement_batch(
            game_id="dragon_gate",
            settlements=(
                JackpotSettlementRequest(
                    player_id=1, player_account_name="alice", player_delta=-5_000
                ),
                JackpotSettlementRequest(
                    player_id=2, player_account_name="bob", player_delta=-7_000
                ),
            ),
        )

    assert await get_balance(user_id=1) == 10_000
    assert await get_balance(user_id=2) == 10_000
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000


async def test_apply_jackpot_settlement_skips_vip_blackjack_bonus() -> None:
    """射龍門 winnings stay at face value even for VIP accounts."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None

    player_balance_before = await get_balance(user_id=1)
    pool_before = await get_jackpot_pool(game_id="dragon_gate")
    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=100, game_id="dragon_gate"
    )

    assert settlement.player_balance == player_balance_before + 100
    assert settlement.jackpot_balance == pool_before - 100
    assert settlement.applied_player_delta == 100


async def test_get_jackpot_pool_returns_zero_for_missing_game() -> None:
    """Unseeded game ids surface as 0 instead of raising."""
    assert await get_jackpot_pool(game_id="never_registered") == 0


async def test_get_jackpot_pool_replenishes_drained_seed_pool() -> None:
    """Reading a seeded jackpot replenishes a zero-balance row."""
    await _ensure_schema()
    async with open_global_state_session() as session:
        await session.execute(
            statement=update(JackpotPool)
            .where(JackpotPool.game_id == "dragon_gate")
            .values(pool_balance=0)
        )
        await session.commit()

    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000
    snapshot = await get_jackpot_snapshot(game_id="dragon_gate")
    assert snapshot.generation == 1


async def test_ensure_schema_seeds_dragon_gate_jackpot_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ensure_schema seeds the dragon_gate pool exactly once across calls."""
    db_path = tmp_path / "seed-economy.db"
    global_state_db_path = tmp_path / "seed-global-state.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    global_state_engine = create_async_engine(url=f"sqlite+aiosqlite:///{global_state_db_path}")
    monkeypatch.setattr("discordbot.cogs._economy.database._engine", engine)
    monkeypatch.setattr(
        "discordbot.cogs._economy.database._global_state_engine", global_state_engine
    )
    monkeypatch.setattr("discordbot.cogs._economy.database._schema_ready_for", None)
    monkeypatch.setattr("discordbot.cogs._economy.database._global_state_schema_ready_for", None)

    await _ensure_schema()
    first_balance = await get_jackpot_pool(game_id="dragon_gate")
    assert first_balance == 1_000

    # Calling again is idempotent: the seed must not pile on top of itself.
    monkeypatch.setattr("discordbot.cogs._economy.database._schema_ready_for", None)
    monkeypatch.setattr("discordbot.cogs._economy.database._global_state_schema_ready_for", None)
    await _ensure_schema()
    assert await get_jackpot_pool(game_id="dragon_gate") == 1_000
    async with open_session() as session:
        result = await session.execute(
            statement=text(
                text="SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'jackpot_pool'"
            )
        )
        assert result.scalar_one_or_none() is None

    await engine.dispose()
    await global_state_engine.dispose()


async def test_get_bot_statuses_empty_returns_empty_list() -> None:
    """A fresh database has no presence lines, so callers fall back to a default."""
    assert await get_bot_statuses() == []


async def test_bot_status_rotation_orders_enabled_lines() -> None:
    """Enabled lines are returned ordered by order_index then id; disabled excluded."""
    second_id = await add_bot_status(status_text="second", order_index=2)
    await add_bot_status(status_text="first", order_index=1)
    await add_bot_status(status_text="hidden", order_index=0, enabled=False)

    assert await get_bot_statuses() == ["first", "second"]

    rows = await list_bot_status_rows()
    assert [(row.status_text, row.enabled) for row in rows] == [
        ("hidden", False),
        ("first", True),
        ("second", True),
    ]

    assert await remove_bot_status(status_id=second_id) is True
    assert await get_bot_statuses() == ["first"]
    assert await remove_bot_status(status_id=second_id) is False
