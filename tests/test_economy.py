"""Tests for the economy persistence layer."""

from random import Random, SystemRandom
from typing import Any, cast
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, text, select, update
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.typings.games import (
    GameParticipant,
    BlackjackPlayerResult,
    BlackjackDealerDecision,
    BlackjackHandSettlement,
    BlackjackPlayerSettlement,
)
from discordbot.cogs._games.blackjack import (
    Card,
    BlackjackHand,
    BlackjackRound,
    BlackjackHandState,
)
from discordbot.cogs._economy.database import (
    TAIWAN_TIMEZONE,
    VIP_PURCHASE_COST,
    CHECKIN_STREAK_CYCLE,
    BASE_CHECKIN_REWARD_AMOUNT,
    JackpotPool,
    UserAccount,
    AdminAccount,
    TransferResult,
    AccountSnapshot,
    JackpotSnapshot,
    TransactionKind,
    LeaderboardEntry,
    PointTransaction,
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
    adjust_balance,
    checkin_reward,
    _taipei_midnight,
    get_jackpot_pool,
    _build_credit_upsert,
    get_jackpot_snapshot,
    credit_with_repayment,
    apply_round_settlement,
    apply_jackpot_settlement,
    apply_jackpot_settlement_batch,
    _apply_jackpot_delta_in_session,
)
from discordbot.cogs._games.settlement import (
    settle_wager,
    settle_blackjack_round,
    settle_blackjack_player,
)
from discordbot.cogs._games.blackjack_views import BlackjackView, build_final_embed

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


def test_blackjack_player_settlement_hands_default_is_isolated() -> None:
    """Default Blackjack hand settlement lists are isolated per model instance."""
    first = BlackjackPlayerSettlement(
        delta=0, payout=0, new_balance=100, house_balance=0, outcome="push", detail="first"
    )
    second = BlackjackPlayerSettlement(
        delta=0, payout=0, new_balance=100, house_balance=0, outcome="push", detail="second"
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


def _round_from_hand(hand: BlackjackHand, participant: GameParticipant) -> BlackjackRound:
    """Adapts a single-player hand into the multiplayer round shape."""
    round_state = BlackjackRound.from_participants(rng=hand.rng, participants=[participant])
    round_state.players[0].hands[0].cards = list(hand.player)
    round_state.players[0].hands[0].finished = hand.finished
    round_state.dealer = list(hand.dealer)
    round_state.finished = hand.finished
    round_state.phase = "settled" if hand.finished else "player_actions"
    return round_state


async def _stored_avatar_url(user_id: int) -> str:
    """Reads the cached avatar URL for one account."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.avatar_url).where(UserAccount.user_id == user_id)
        )
        return result.scalar_one()


async def _daily_casino_stats(user_id: int) -> tuple[int, int, int, datetime | None]:
    """Reads daily casino ``(loss, win, net, day_started_at)`` counters."""
    async with open_session() as session:
        result = await session.execute(
            statement=select(
                UserAccount.daily_casino_loss,
                UserAccount.daily_casino_win,
                UserAccount.daily_casino_net,
                UserAccount.casino_day_started_at,
            ).where(UserAccount.user_id == user_id)
        )
        row = result.one_or_none()
    if row is None:
        return 0, 0, 0, None
    return row[0], row[1], row[2], row[3]


async def _add_balance(user_id: int, name: str, amount: int, avatar_url: str = "") -> int:
    """Seeds a positive balance without writing audit rows."""
    await _ensure_schema()
    if amount <= 0:
        return await get_balance(user_id=user_id)
    now = _database_now()
    async with open_session() as session:
        result = await session.execute(
            statement=_build_credit_upsert(
                user_id=user_id, name=name, amount=amount, avatar_url=avatar_url, now=now
            )
        )
        await session.commit()
        return result.scalar_one()


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
    """Zero deltas do not change the balance or write an audit row."""
    await _add_balance(user_id=42, name="alice", amount=100)
    result = await adjust_balance(user_id=42, name="alice", delta=0)
    assert result == BalanceAdjustmentResult(new_balance=100, applied_delta=0)
    async with open_session() as session:
        count = await session.scalar(
            statement=select(func.count()).where(
                PointTransaction.user_id == 42,
                PointTransaction.kind == TransactionKind.MANUAL_ADJUSTMENT.value,
            )
        )
    assert count == 0


async def test_adjust_balance_logs_manual_adjustment() -> None:
    """Manual adjustments write explicit MANUAL_ADJUSTMENT audit rows."""
    result = await adjust_balance(user_id=42, name="alice", delta=100)
    assert result == BalanceAdjustmentResult(new_balance=100, applied_delta=100)
    async with open_session() as session:
        rows = (
            await session.execute(
                statement=select(
                    PointTransaction.kind, PointTransaction.delta, PointTransaction.balance_after
                ).where(PointTransaction.user_id == 42)
            )
        ).all()
    assert rows == [(TransactionKind.MANUAL_ADJUSTMENT.value, 100, 100)]


async def test_adjust_balance_logs_note() -> None:
    """Manual adjustments can annotate the audit row."""
    await adjust_balance(user_id=42, name="alice", delta=100, note="refund_tax by 1")
    async with open_session() as session:
        note = await session.scalar(
            statement=select(PointTransaction.note).where(PointTransaction.user_id == 42)
        )
    assert note == "refund_tax by 1"


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
    """Account and audit timestamps are persisted as Taiwan-local wall time."""
    before = datetime.now(tz=TAIWAN_TIMEZONE).replace(tzinfo=None)
    await credit_with_repayment(
        user_id=42, name="alice", amount=10, kind=TransactionKind.CHAT_REWARD
    )
    after = datetime.now(tz=TAIWAN_TIMEZONE).replace(tzinfo=None)

    async with open_session() as session:
        result = await session.execute(
            statement=select(UserAccount.updated_at, PointTransaction.occurred_at)
            .join(PointTransaction, PointTransaction.user_id == UserAccount.user_id)
            .where(UserAccount.user_id == 42)
        )
        updated_at, occurred_at = result.one()

    assert before <= updated_at <= after
    assert before <= occurred_at <= after


async def test_existing_economy_db_gets_schema_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-feature economy DB picks up new columns and drops dead legacy ones."""
    db_path = tmp_path / "legacy-economy.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.execute(
            statement=text(
                text="""
                CREATE TABLE user_account (
                    user_id INTEGER NOT NULL PRIMARY KEY,
                    name VARCHAR(128),
                    balance INTEGER NOT NULL,
                    total_earned INTEGER NOT NULL,
                    total_spent INTEGER NOT NULL,
                    updated_at DATETIME,
                    loan_principal INTEGER NOT NULL,
                    loan_interest INTEGER NOT NULL,
                    loan_total_borrowed INTEGER NOT NULL,
                    loan_total_repaid INTEGER NOT NULL,
                    loan_last_accrual_at DATETIME,
                    loan_opened_at DATETIME
                )
                """
            )
        )
        await conn.execute(
            statement=text(
                text="""
                INSERT INTO user_account (
                    user_id, name, balance, total_earned, total_spent, updated_at,
                    loan_principal, loan_interest, loan_total_borrowed, loan_total_repaid,
                    loan_last_accrual_at, loan_opened_at
                )
                VALUES (42, 'alice', 10, 10, 0, CURRENT_TIMESTAMP, 0, 0, 0, 0, NULL, NULL)
                """
            )
        )
        await conn.execute(
            statement=text(
                text="""
                CREATE TABLE point_transaction (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    kind VARCHAR(32) NOT NULL,
                    delta INTEGER NOT NULL,
                    balance_after INTEGER NOT NULL,
                    note VARCHAR(256),
                    occurred_at DATETIME NOT NULL
                )
                """
            )
        )
    monkeypatch.setattr("discordbot.cogs._economy.database._engine", engine)

    await _add_balance(
        user_id=42, name="alice", amount=5, avatar_url="https://cdn.example/avatar.png"
    )

    assert await _stored_avatar_url(user_id=42) == "https://cdn.example/avatar.png"
    async with open_session() as session:
        result = await session.execute(statement=text(text="PRAGMA table_info(user_account)"))
        columns = {row[1] for row in result.all()}
        result = await session.execute(statement=text(text="PRAGMA table_info(point_transaction)"))
        transaction_columns = {row[1] for row in result.all()}
        result = await session.execute(statement=text(text="PRAGMA index_list(user_account)"))
        index_names = {row[1] for row in result.all()}
    assert {
        "is_vip",
        "last_checkin_at",
        "checkin_streak",
        "is_admin",
        "casino_day_started_at",
        "daily_casino_loss",
        "daily_casino_win",
        "daily_casino_net",
    } <= columns
    assert "debt_after" in transaction_columns
    assert "ix_user_account_casino_day_loss" in index_names
    assert "loan_interest" not in columns
    assert "loan_last_accrual_at" not in columns

    # A brand-new user must be insertable after the migration even when the
    # legacy schema had NOT NULL columns without DEFAULT.
    await _add_balance(user_id=43, name="bob", amount=7)
    assert await get_balance(user_id=43) == 7
    await engine.dispose()


async def test_get_balance_unknown_user_returns_zero() -> None:
    """Reading a never-seen user returns zero, not an error."""
    assert await get_balance(user_id=999) == 0


async def test_transfer_moves_currency_between_users() -> None:
    """Successful transfer debits sender and credits receiver atomically."""
    await _add_balance(user_id=1, name="alice", amount=200)
    result = await transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=80
    )
    assert result == TransferResult(sender_balance=120, receiver_balance=80)
    assert await get_balance(user_id=1) == 120
    assert await get_balance(user_id=2) == 80


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
    assert await get_balance(user_id=2) + await get_balance(user_id=3) == 80


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
    assert max(result.receiver_balance for result in results if result is not None) == 150
    assert await get_balance(user_id=3) == 150


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


async def test_apply_round_settlement_allows_negative_house_balance() -> None:
    """House ledger keeps a true running net even when the dealer is down."""
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=500,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=-500,
    )
    assert await get_balance(user_id=99) == -500


async def test_apply_round_settlement_house_accumulates_gross_flows() -> None:
    """Wins and losses both accumulate gross totals, not just the net balance."""
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-200,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=200,
    )
    await apply_round_settlement(
        player_id=2,
        player_account_name="bob",
        player_delta=300,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=-300,
    )
    account = await get_account(user_id=99)
    assert account == AccountSnapshot(
        name="house", balance=-100, total_earned=200, total_spent=300
    )


async def test_settle_wager_updates_player_and_house() -> None:
    """Shared wager settlement applies net delta and mirrors house P&L."""
    await _add_balance(user_id=1, name="alice", amount=100)

    settlement = await settle_wager(
        player_id=1, player_account_name="alice", dealer_id=99, dealer_name="house", delta=40
    )
    assert settlement.payout == 40
    assert settlement.new_balance == 140
    assert settlement.house_balance == -40


async def test_get_account_returns_none_for_unseen_user() -> None:
    """Unknown users return None instead of a synthetic zero row."""
    assert await get_account(user_id=12345) is None


async def test_settle_blackjack_round_updates_player_and_house() -> None:
    """Shared Blackjack settlement applies net delta and mirrors house P&L."""
    await _add_balance(user_id=1, name="alice", amount=100)

    hand = BlackjackHand(rng=SystemRandom(), bet=50)
    hand.player = [Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")]
    hand.dealer = [Card(rank="10", suit="♣"), Card(rank="8", suit="♦")]
    hand.finished = True

    settlement = await settle_blackjack_round(
        hand=hand, player_id=1, player_account_name="alice", dealer_id=99, dealer_name="house"
    )
    assert settlement.delta == 50
    assert settlement.payout == 50
    assert settlement.new_balance == 150
    assert settlement.house_balance == -50
    assert await get_balance(user_id=99) == -50


async def test_blackjack_view_finalizes_once_when_called_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent finalization attempts must not pay out one Blackjack hand twice."""
    cleanup_messages: list[_MessageStub] = []

    def fake_schedule_game_message_delete(message: _MessageStub, delay: float = 180) -> None:
        """Records the final message scheduled for cleanup."""
        cleanup_messages.append(message)

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.schedule_game_message_delete",
        fake_schedule_game_message_delete,
    )
    await _add_balance(user_id=1, name="alice", amount=100)

    hand = BlackjackHand(rng=SystemRandom(), bet=50)
    hand.player = [Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")]
    hand.dealer = [Card(rank="10", suit="♣"), Card(rank="8", suit="♦")]
    hand.finished = True

    dealer = _DealerStub()
    message = _MessageStub()
    participant = _participant()
    view = BlackjackView(
        dealer=dealer,
        round_state=_round_from_hand(hand=hand, participant=participant),
        starter_id=1,
        author_name="alice",
        dealer_id=99,
        dealer_name="house",
    )

    await asyncio.gather(view.finalize(message=message), view.finalize(message=message))

    assert await get_balance(user_id=1) == 150
    assert await get_balance(user_id=99) == -50
    assert dealer.settle_calls == 1
    assert message.edit_calls == 3
    assert "embeds" not in message.edits[0]
    thinking_embeds = cast("list[Any]", message.edits[1]["embeds"])
    thinking_line = cast("str", thinking_embeds[0].description)
    assert "莊家正在思考 hit / stand" in thinking_line
    assert cleanup_messages == [message]


async def test_blackjack_view_timeout_auto_stands_and_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A player who walks away is treated as standing and the wager resolves."""
    cleanup_messages: list[_MessageStub] = []

    def fake_schedule_game_message_delete(message: _MessageStub, delay: float = 180) -> None:
        """Records the final message scheduled for cleanup."""
        cleanup_messages.append(message)

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.schedule_game_message_delete",
        fake_schedule_game_message_delete,
    )
    await _add_balance(user_id=1, name="alice", amount=100)

    hand = BlackjackHand(rng=SystemRandom(), bet=50)
    hand.player = [Card(rank="10", suit="♠"), Card(rank="8", suit="♥")]
    hand.dealer = [Card(rank="10", suit="♣"), Card(rank="Q", suit="♦")]

    dealer = _DealerStub()
    message = _MessageStub()
    participant = _participant()
    view = BlackjackView(
        dealer=dealer,
        round_state=_round_from_hand(hand=hand, participant=participant),
        starter_id=1,
        author_name="alice",
        dealer_id=99,
        dealer_name="house",
    )
    view.message = message

    await view.on_timeout()

    assert view.round_state.finished is True
    assert await get_balance(user_id=1) == 50
    assert await get_balance(user_id=99) == 50
    assert dealer.settle_calls == 1
    assert message.edit_calls == 3
    assert "embeds" not in message.edits[0]
    thinking_embeds = cast("list[Any]", message.edits[1]["embeds"])
    thinking_line = cast("str", thinking_embeds[0].description)
    assert "莊家正在思考 hit / stand" in thinking_line
    assert cleanup_messages == [message]


async def test_blackjack_view_uses_ai_dealer_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dealer AI can hit once, stand, and settle with the updated dealer hand."""
    cleanup_messages: list[_MessageStub] = []

    def fake_schedule_game_message_delete(message: _MessageStub, delay: float = 180) -> None:
        """Records the final message scheduled for cleanup."""
        cleanup_messages.append(message)

    def draw_fixed_card(rng: Random) -> Card:
        """Returns a deterministic dealer draw."""
        return Card(rank="5", suit="♣")

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.schedule_game_message_delete",
        fake_schedule_game_message_delete,
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

    dealer = _DealerStub()
    dealer.decisions = [
        BlackjackDealerDecision(action="hit", reason="追過玩家"),
        BlackjackDealerDecision(action="stand", reason="18 點夠了"),
    ]
    message = _MessageStub()
    view = BlackjackView(
        dealer=dealer,
        round_state=round_state,
        starter_id=1,
        author_name="alice",
        dealer_id=99,
        dealer_name="house",
    )

    await view.finalize(message=message)

    assert [str(card) for card in view.round_state.dealer] == ["10♣", "3♦", "5♣"]
    assert view.round_state.dealer_played is True
    assert dealer.decision_calls == 1
    assert await get_balance(user_id=1) == 50
    assert await get_balance(user_id=99) == 50
    assert "embeds" not in message.edits[0]
    thinking_embeds = cast("list[Any]", message.edits[1]["embeds"])
    thinking_line = cast("str", thinking_embeds[0].description)
    assert "莊家正在思考 hit / stand" in thinking_line
    final_embeds = cast("list[Any]", message.edits[2]["embeds"])
    description = cast("str", final_embeds[1].description)
    assert "13 hit 抽 5♣ → 18 (guard)" in description
    assert "18 stand" in description
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
        dealer=_DealerStub(),
        round_state=round_state,
        starter_id=1,
        author_name="alice",
        dealer_id=99,
        dealer_name="house",
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


async def test_blackjack_view_basic_rule_fallback_finishes_dealer_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DealerAI fallback decision switches the rest of the dealer phase to basic rule."""
    cleanup_messages: list[_MessageStub] = []
    draws = [Card(rank="5", suit="♣")]

    def fake_schedule_game_message_delete(message: _MessageStub, delay: float = 180) -> None:
        """Records the final message scheduled for cleanup."""
        cleanup_messages.append(message)

    def draw_fixed_card(rng: Random) -> Card:
        """Returns deterministic dealer draws."""
        return draws.pop(0)

    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.schedule_game_message_delete",
        fake_schedule_game_message_delete,
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

    dealer = _DealerStub()
    dealer.decisions = [BlackjackDealerDecision(action="hit", reason="basic rule: 未滿 17 點")]
    message = _MessageStub()
    view = BlackjackView(
        dealer=dealer,
        round_state=round_state,
        starter_id=1,
        author_name="alice",
        dealer_id=99,
        dealer_name="house",
    )

    await view.finalize(message=message)

    assert dealer.decision_calls == 1
    assert view.round_state.dealer_total() == 18
    assert len(view.round_state.dealer) == 3
    assert "embeds" not in message.edits[0]
    thinking_embeds = cast("list[Any]", message.edits[1]["embeds"])
    thinking_line = cast("str", thinking_embeds[0].description)
    assert "莊家正在思考 hit / stand" in thinking_line
    final_embeds = cast("list[Any]", message.edits[2]["embeds"])
    description = cast("str", final_embeds[1].description)
    assert "fallback basic rule" in description or "override" in description
    assert cleanup_messages == [message]


async def test_blackjack_view_locks_actions_while_finalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late Hit cannot mutate a hand that is already finalizing from Stand."""
    cleanup_messages: list[_MessageStub] = []
    settlement_started = asyncio.Event()
    continue_settlement = asyncio.Event()

    def fake_schedule_game_message_delete(message: _MessageStub, delay: float = 180) -> None:
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
            house_balance=-50,
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
        "discordbot.cogs._games.blackjack_views.schedule_game_message_delete",
        fake_schedule_game_message_delete,
    )
    monkeypatch.setattr(
        "discordbot.cogs._games.blackjack_views.settle_blackjack_player",
        delayed_settle_blackjack_player,
    )

    hand = BlackjackHand(rng=SystemRandom(), bet=50)
    hand.player = [Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")]
    hand.dealer = [Card(rank="10", suit="♣"), Card(rank="8", suit="♦")]

    dealer = _DealerStub()
    message = _MessageStub()
    participant = _participant(balance_at_start=50)
    view = BlackjackView(
        dealer=dealer,
        round_state=_round_from_hand(hand=hand, participant=participant),
        starter_id=1,
        author_name="alice",
        dealer_id=99,
        dealer_name="house",
    )

    hit_button = next(child for child in view.children if child.custom_id == "bj:hit")
    stand_button = next(child for child in view.children if child.custom_id == "bj:stand")
    stand_task = asyncio.create_task(coro=stand_button.callback(_InteractionStub(message=message)))
    await settlement_started.wait()

    hit_task = asyncio.create_task(coro=hit_button.callback(_InteractionStub(message=message)))
    await asyncio.sleep(delay=0)

    assert len(view.round_state.players[0].hands[0].cards) == 2
    continue_settlement.set()
    await asyncio.gather(stand_task, hit_task)

    assert len(view.round_state.players[0].hands[0].cards) == 2
    assert dealer.settle_calls == 1
    assert dealer.hint_calls == 0
    assert message.edit_calls == 3
    assert "embeds" not in message.edits[0]
    thinking_embeds = cast("list[Any]", message.edits[1]["embeds"])
    thinking_line = cast("str", thinking_embeds[0].description)
    assert "莊家正在思考 hit / stand" in thinking_line
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
        dealer=dealer,
        round_state=round_state,
        starter_id=1,
        author_name="alice",
        dealer_id=99,
        dealer_name="house",
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

    message = _MessageStub()
    view = BlackjackView(
        dealer=_DealerStub(),
        round_state=round_state,
        starter_id=1,
        author_name="alice",
        dealer_id=99,
        dealer_name="house",
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

    dealer = _DealerStub()
    message = _MessageStub()
    view = BlackjackView(
        dealer=dealer,
        round_state=round_state,
        starter_id=1,
        author_name="alice",
        dealer_id=99,
        dealer_name="house",
    )

    hit_button = next(child for child in view.children if child.custom_id == "bj:hit")
    await hit_button.callback(_InteractionStub(message=message, user_id=1))

    assert [str(card) for card in player.hands[1].cards] == ["9♣", "2♦", "5♣"]
    assert dealer.hint_calls == 1
    assert dealer.hints[0]["player_total"] == 16
    assert message.edit_calls == 1


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
            player_id=42,
            player_account_name="alice",
            player_delta=10,
            dealer_id=99,
            dealer_name="house",
            dealer_delta=-10,
        )
        for _ in range(10)
    ])
    assert await get_balance(user_id=42) == 200


async def test_apply_round_settlement_concurrent_house_updates_accumulate() -> None:
    """Verifies that concurrent dealer ledger settlements accumulate."""
    await asyncio.gather(*[
        apply_round_settlement(
            player_id=user_id,
            player_account_name=f"player{user_id}",
            player_delta=-10,
            dealer_id=99,
            dealer_name="house",
            dealer_delta=10,
        )
        for user_id in range(10)
    ])
    account = await get_account(user_id=99)
    assert account is not None
    assert account.balance == 100
    assert account.total_earned == 100
    assert account.total_spent == 0


async def test_apply_round_settlement_is_atomic() -> None:
    """Player delta and house mirror share one transaction and one return."""
    await _add_balance(user_id=1, name="alice", amount=100)

    player_balance, house_balance = await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=40,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=-40,
    )
    assert player_balance == 140
    assert house_balance == -40
    assert await get_balance(user_id=1) == 140
    assert await get_balance(user_id=99) == -40


async def test_apply_round_settlement_loss_debits_player_and_house() -> None:
    """A loss debits the player and credits the house."""
    await _add_balance(user_id=1, name="alice", amount=100)

    player_balance, house_balance = await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-40,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=40,
    )
    assert player_balance == 60
    assert house_balance == 40
    account = await get_account(user_id=1)
    assert account is not None
    assert account.total_earned == 100
    assert account.total_spent == 40


async def test_apply_round_settlement_loss_can_make_player_negative() -> None:
    """Deferred settlement still collects a loss after the balance was spent elsewhere."""
    await _add_balance(user_id=1, name="alice", amount=25)

    player_balance, house_balance = await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-40,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=40,
    )

    assert player_balance == -15
    assert house_balance == 40


async def test_apply_round_settlement_updates_daily_casino_counters() -> None:
    """Blackjack-style player settlements persist gross loss, gross win, and net."""
    await _add_balance(user_id=1, name="alice", amount=1_000)

    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-300,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=300,
    )
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=500,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=-500,
    )

    loss, win, net, day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (300, 500, 200)
    assert day_started_at is not None
    assert _as_taipei(dt=day_started_at) == _taipei_midnight(now=_database_now())


async def test_daily_casino_counters_skip_push_and_house_ledger() -> None:
    """Zero deltas and dealer ledger mirrors do not enter player loss counters."""
    await _add_balance(user_id=1, name="alice", amount=100)

    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=0,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=0,
    )
    assert await _daily_casino_stats(user_id=1) == (0, 0, 0, None)

    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-40,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=40,
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
        (1, False, 100_000),
        (2, False, 150_000),
        (7, False, 400_000),
        (1, True, 200_000),
        (7, True, 800_000),
    ],
)
def test_checkin_reward_formula(streak: int, is_vip: bool, expected: int) -> None:
    """Streak + VIP combinations compute to the expected reward."""
    assert checkin_reward(streak=streak, is_vip=is_vip) == expected


async def test_checkin_logs_audit_row() -> None:
    """A successful check-in writes one CHECKIN_REWARD row tagged with the streak."""
    result = await checkin(user_id=1, name="alice")
    assert result is not None
    async with open_session() as session:
        rows = (
            await session.execute(
                statement=select(
                    PointTransaction.kind, PointTransaction.delta, PointTransaction.note
                ).where(PointTransaction.user_id == 1)
            )
        ).all()
    assert rows == [(TransactionKind.CHECKIN_REWARD.value, result.amount, "streak 1")]


# VIP purchase --------------------------------------------------------------


async def test_buy_vip_sets_flag_and_debits_balance() -> None:
    """A successful purchase costs ``VIP_PURCHASE_COST`` and flips ``is_vip``."""
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


async def test_buy_vip_logs_audit_row() -> None:
    """A successful purchase records one VIP_PURCHASE audit row."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    await buy_vip(user_id=1, name="alice")
    async with open_session() as session:
        rows = (
            await session.execute(
                statement=select(PointTransaction.kind, PointTransaction.delta).where(
                    PointTransaction.user_id == 1
                )
            )
        ).all()
    assert rows == [(TransactionKind.VIP_PURCHASE.value, -VIP_PURCHASE_COST)]


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
        player_id=1,
        player_account_name="alice",
        player_delta=-300,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=300,
    )
    await apply_round_settlement(
        player_id=2,
        player_account_name="bob",
        player_delta=200,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=-200,
    )
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=500,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=-500,
    )
    await apply_round_settlement(
        player_id=3,
        player_account_name="carol",
        player_delta=-200,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=200,
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
            player_id=user_id,
            player_account_name=name,
            player_delta=-loss,
            dealer_id=99,
            dealer_name="house",
            dealer_delta=loss,
        )
    rows = await top_losers(limit=10, exclude_user_ids=(99,))
    assert [(row.user_id, row.loss_amount) for row in rows] == [(2, 500), (3, 250), (1, 100)]


async def test_top_losers_excludes_specified_users() -> None:
    """``exclude_user_ids`` filters the house ledger out of the report."""
    await _add_balance(user_id=1, name="alice", amount=500)
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-500,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=500,
    )
    rows = await top_losers(limit=10, exclude_user_ids=(99,))
    assert all(row.user_id != 99 for row in rows)


async def test_top_losers_ignores_counters_before_today() -> None:
    """Stale account counters from an older Taipei day do not count."""
    await _add_balance(user_id=1, name="alice", amount=500)
    await apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-500,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=500,
    )
    past = datetime.now(tz=TAIWAN_TIMEZONE) - timedelta(days=2)
    async with open_session() as session:
        await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == 1)
            .values(casino_day_started_at=_taipei_midnight(now=past))
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
    """A VIP player wins 1.5x of the base delta; house mirrors the boosted amount."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None
    settlement = await settle_wager(
        player_id=1, player_account_name="alice", dealer_id=99, dealer_name="house", delta=100
    )
    assert settlement.delta == 150
    assert settlement.base_delta == 100
    assert settlement.vip_bonus == 50
    assert settlement.is_vip is True
    assert settlement.house_balance == -150


async def test_settle_wager_keeps_loss_unchanged_for_vip() -> None:
    """The VIP perk does not soften losses."""
    await _add_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST + 1_000)
    purchase = await buy_vip(user_id=1, name="alice")
    assert purchase is not None
    settlement = await settle_wager(
        player_id=1, player_account_name="alice", dealer_id=99, dealer_name="house", delta=-100
    )
    assert settlement.delta == -100
    assert settlement.base_delta == -100
    assert settlement.vip_bonus == 0
    assert settlement.is_vip is True
    assert settlement.house_balance == 100


# Multi-hand Blackjack settlement -----------------------------------------


async def _settle_player(round_state: BlackjackRound) -> BlackjackPlayerSettlement:
    """Helper that runs settle_blackjack_player against the only player."""
    player = round_state.players[0]
    return await settle_blackjack_player(
        round_state=round_state,
        player=player,
        player_id=player.participant.user_id,
        player_account_name=player.participant.account_name,
        dealer_id=99,
        dealer_name="house",
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
    assert settlement.house_balance == 25


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
    embed = build_final_embed(
        dealer_name="house",
        round_state=round_state,
        results=[BlackjackPlayerResult(participant=player.participant, settlement=settlement)],
    )

    assert embed.title == "♠️ 二十一點 · 1 平"
    description = cast("str", embed.description)
    assert "## 😢 你輸了 · 17 < 21" in description
    assert "保險 `50` → 中獎 (+100)" in description
    assert "17 = 21" not in embed.title


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
    # _ensure_schema already seeded the dragon_gate pool at 100_000.
    assert await get_jackpot_pool(game_id="dragon_gate") == 100_000

    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=20_000, game_id="dragon_gate"
    )

    assert settlement.player_balance == 30_000
    assert settlement.jackpot_balance == 80_000
    assert settlement.applied_player_delta == 20_000
    assert settlement.jackpot_depleted is False
    assert await get_jackpot_pool(game_id="dragon_gate") == 80_000
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (0, 20_000, 20_000)


async def test_apply_jackpot_settlement_replenishes_drained_seed_pool() -> None:
    """A seeded jackpot restores itself after a player wins the whole pool."""
    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=100_000, game_id="dragon_gate"
    )

    assert settlement.player_balance == 100_000
    assert settlement.jackpot_balance == 100_000
    assert settlement.applied_player_delta == 100_000
    assert settlement.jackpot_depleted is True
    assert await get_jackpot_pool(game_id="dragon_gate") == 100_000
    async with open_session() as session:
        result = await session.execute(
            statement=select(JackpotPool.seeded_amount, JackpotPool.total_claimed).where(
                JackpotPool.game_id == "dragon_gate"
            )
        )
        seeded_amount, total_claimed = result.one()
    assert seeded_amount == 200_000
    assert total_claimed == 100_000


async def test_apply_jackpot_settlement_clamps_loss_and_grows_pool_by_actual_debit() -> None:
    """Player losses stop at zero and feed the jackpot with the actual debit."""
    await _add_balance(user_id=1, name="alice", amount=15_000)

    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=-25_000, game_id="dragon_gate"
    )

    assert settlement.player_balance == 0
    assert settlement.jackpot_balance == 115_000
    assert settlement.applied_player_delta == -15_000
    async with open_session() as session:
        result = await session.execute(
            statement=select(
                PointTransaction.kind, PointTransaction.delta, PointTransaction.balance_after
            ).where(PointTransaction.user_id == 1)
        )
        kind, delta, balance_after = result.one()
    assert kind == TransactionKind.CASINO_BET.value
    assert delta == -15_000
    assert balance_after == 0
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (15_000, 0, -15_000)


async def test_apply_jackpot_settlement_concurrent_clamped_losses_log_actual_debit() -> None:
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
    assert await get_jackpot_pool(game_id="dragon_gate") == 100_100
    async with open_session() as session:
        result = await session.execute(
            statement=select(func.sum(PointTransaction.delta)).where(
                PointTransaction.user_id == 1,
                PointTransaction.kind == TransactionKind.CASINO_BET.value,
            )
        )
        assert result.scalar_one() == -100
    loss, win, net, _day_started_at = await _daily_casino_stats(user_id=1)
    assert (loss, win, net) == (100, 0, -100)


async def test_apply_jackpot_settlement_caps_win_to_live_pool() -> None:
    """A stale oversized jackpot win only pays the live pool amount."""
    settlement = await apply_jackpot_settlement(
        player_id=1, player_account_name="alice", player_delta=150_000, game_id="dragon_gate"
    )

    assert settlement.player_balance == 100_000
    assert settlement.applied_player_delta == 100_000
    assert settlement.jackpot_balance == 100_000
    assert settlement.jackpot_depleted is True


async def test_apply_jackpot_settlement_concurrent_wins_do_not_double_claim_pool() -> None:
    """Concurrent whole-pool wins cannot both claim the same jackpot generation."""
    snapshot = await get_jackpot_snapshot(game_id="dragon_gate")
    first, second = await asyncio.gather(
        apply_jackpot_settlement(
            player_id=1,
            player_account_name="alice",
            player_delta=100_000,
            game_id="dragon_gate",
            expected_jackpot_generation=snapshot.generation,
        ),
        apply_jackpot_settlement(
            player_id=2,
            player_account_name="bob",
            player_delta=100_000,
            game_id="dragon_gate",
            expected_jackpot_generation=snapshot.generation,
        ),
    )

    applied_total = first.applied_player_delta + second.applied_player_delta
    assert applied_total == 100_000
    assert await get_jackpot_pool(game_id="dragon_gate") == 100_000


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
    assert result.jackpot_balance == 112_000
    assert await get_jackpot_pool(game_id="dragon_gate") == 112_000


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
    assert await get_jackpot_pool(game_id="dragon_gate") == 100_000


async def test_apply_jackpot_settlement_batch_rolls_back_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed batch ante settlement cannot partially charge players."""
    await _add_balance(user_id=1, name="alice", amount=10_000)
    await _add_balance(user_id=2, name="bob", amount=10_000)
    assert await get_jackpot_pool(game_id="dragon_gate") == 100_000

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
    assert await get_jackpot_pool(game_id="dragon_gate") == 100_000


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


async def test_get_jackpot_pool_replenishes_legacy_drained_seed_pool() -> None:
    """Reading a seeded jackpot repairs an older zero-balance row."""
    await _ensure_schema()
    async with open_session() as session:
        await session.execute(
            statement=update(JackpotPool)
            .where(JackpotPool.game_id == "dragon_gate")
            .values(pool_balance=0)
        )
        await session.commit()

    assert await get_jackpot_pool(game_id="dragon_gate") == 100_000
    snapshot = await get_jackpot_snapshot(game_id="dragon_gate")
    assert snapshot.generation == 1


async def test_ensure_schema_seeds_dragon_gate_jackpot_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ensure_schema seeds the dragon_gate pool exactly once across calls."""
    db_path = tmp_path / "seed-economy.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setattr("discordbot.cogs._economy.database._engine", engine)
    monkeypatch.setattr("discordbot.cogs._economy.database._schema_ready_for", None)

    await _ensure_schema()
    first_balance = await get_jackpot_pool(game_id="dragon_gate")
    assert first_balance == 100_000

    # Calling again is idempotent: the seed must not pile on top of itself.
    monkeypatch.setattr("discordbot.cogs._economy.database._schema_ready_for", None)
    await _ensure_schema()
    assert await get_jackpot_pool(game_id="dragon_gate") == 100_000

    await engine.dispose()
