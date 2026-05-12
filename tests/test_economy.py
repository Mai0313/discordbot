"""Tests for the economy persistence layer."""

from random import SystemRandom
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs._games import views
from discordbot.cogs._economy import database
from discordbot.typings.games import GameParticipant, BlackjackSettlement
from discordbot.cogs._games.views import BlackjackView
from discordbot.cogs._games.blackjack import Card, BlackjackHand, BlackjackRound
from discordbot.cogs._games.settlement import settle_wager, settle_blackjack_round


class _DealerStub:
    """Minimal dealer stub for BlackjackView settlement tests."""

    def __init__(self) -> None:
        self.settle_calls = 0
        self.hint_calls = 0

    async def settle(self, **_kwargs: object) -> str:
        """Returns deterministic banter and tracks settlement calls."""
        self.settle_calls += 1
        await asyncio.sleep(delay=0)
        return "settled"

    async def hint(self, **_kwargs: object) -> str:
        """Returns deterministic in-progress banter and tracks hint calls."""
        self.hint_calls += 1
        await asyncio.sleep(delay=0)
        return "hint"


class _MessageStub:
    """Minimal message stub that records edit calls."""

    def __init__(self) -> None:
        self.edit_calls = 0

    async def edit(self, **_kwargs: object) -> None:
        """Records a Discord message edit."""
        self.edit_calls += 1


class _ResponseStub:
    """Minimal interaction response stub for button callback tests."""

    def __init__(self) -> None:
        self.deferred = False

    async def defer(self) -> None:
        """Records that the button interaction was deferred."""
        self.deferred = True


class _InteractionStub:
    """Minimal button interaction stub."""

    def __init__(self, *, message: _MessageStub) -> None:
        self.message = message
        self.response = _ResponseStub()


def _participant(
    *,
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


def _round_from_hand(*, hand: BlackjackHand, participant: GameParticipant) -> BlackjackRound:
    """Adapts a single-player hand into the multiplayer round shape."""
    round_state = BlackjackRound.from_participants(rng=hand.rng, participants=[participant])
    round_state.players[0].cards = list(hand.player)
    round_state.players[0].finished = hand.finished
    round_state.dealer = list(hand.dealer)
    round_state.finished = hand.finished
    return round_state


@pytest.fixture(autouse=True)
async def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Replaces the module-level engine with a per-test SQLite file.

    Each test gets a fresh DB so writes never leak between tests, and the
    real ``data/economy.db`` is left alone.
    """
    db_path = tmp_path / "economy.db"
    engine = create_async_engine(url=f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(database.Base.metadata.create_all)
    monkeypatch.setattr(target=database, name="_engine", value=engine)
    yield
    await engine.dispose()


async def _stored_avatar_url(*, user_id: int) -> str:
    """Reads the cached avatar URL for one account."""
    async with database.open_session() as session:
        result = await session.execute(
            statement=select(database.UserAccount.avatar_url).where(
                database.UserAccount.user_id == user_id
            )
        )
        return result.scalar_one()


async def test_add_balance_creates_user() -> None:
    """First write upserts the row and returns the new balance."""
    new = await database.add_balance(user_id=42, name="alice", amount=100)
    assert new == 100
    assert await database.get_balance(user_id=42) == 100


async def test_add_balance_accumulates() -> None:
    """Repeated adds increment the running balance."""
    await database.add_balance(user_id=42, name="alice", amount=100)
    new = await database.add_balance(user_id=42, name="alice", amount=50)
    assert new == 150


async def test_add_balance_zero_is_noop() -> None:
    """Zero or negative amounts must not change the balance."""
    await database.add_balance(user_id=42, name="alice", amount=100)
    assert await database.add_balance(user_id=42, name="alice", amount=0) == 100
    assert await database.add_balance(user_id=42, name="alice", amount=-5) == 100


async def test_add_balance_refreshes_name() -> None:
    """Subsequent writes refresh the cached display name."""
    await database.add_balance(user_id=42, name="alice", amount=10)
    await database.add_balance(user_id=42, name="alice_renamed", amount=10)
    rows = await database.top_n(limit=1)
    assert rows[0][1] == "alice_renamed"
    assert rows[0][3] == ""


async def test_add_balance_stores_and_refreshes_avatar_url() -> None:
    """Subsequent writes refresh the cached avatar URL."""
    await database.add_balance(
        user_id=42, name="alice", amount=10, avatar_url="https://cdn.example/a.png"
    )
    assert await _stored_avatar_url(user_id=42) == "https://cdn.example/a.png"

    await database.add_balance(
        user_id=42, name="alice", amount=10, avatar_url="https://cdn.example/b.png"
    )
    assert await _stored_avatar_url(user_id=42) == "https://cdn.example/b.png"


async def test_write_timestamps_use_taiwan_local_time() -> None:
    """Account and audit timestamps are persisted as Taiwan-local wall time."""
    before = datetime.now(tz=database.TAIWAN_TIMEZONE).replace(tzinfo=None)
    await database.credit_with_repayment(
        user_id=42, name="alice", amount=10, kind=database.TransactionKind.CHAT_REWARD
    )
    after = datetime.now(tz=database.TAIWAN_TIMEZONE).replace(tzinfo=None)

    async with database.open_session() as session:
        result = await session.execute(
            statement=select(
                database.UserAccount.updated_at, database.PointTransaction.occurred_at
            )
            .join(
                database.PointTransaction,
                database.PointTransaction.user_id == database.UserAccount.user_id,
            )
            .where(database.UserAccount.user_id == 42)
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
    monkeypatch.setattr(target=database, name="_engine", value=engine)

    await database.add_balance(
        user_id=42, name="alice", amount=5, avatar_url="https://cdn.example/avatar.png"
    )

    assert await _stored_avatar_url(user_id=42) == "https://cdn.example/avatar.png"
    async with database.open_session() as session:
        result = await session.execute(statement=text(text="PRAGMA table_info(user_account)"))
        columns = {row[1] for row in result.all()}
    assert {"is_vip", "last_checkin_at", "checkin_streak"} <= columns
    assert "loan_interest" not in columns
    assert "loan_last_accrual_at" not in columns

    # A brand-new user must be insertable after the migration even when the
    # legacy schema had NOT NULL columns without DEFAULT.
    await database.add_balance(user_id=43, name="bob", amount=7)
    assert await database.get_balance(user_id=43) == 7
    await engine.dispose()


async def test_place_bet_withdraws_requested_amount() -> None:
    """A valid wager is deducted before the game starts."""
    await database.add_balance(user_id=42, name="alice", amount=100)
    placed = await database.place_bet(user_id=42, name="alice", requested_bet=40)
    assert placed == database.PlacedBet(amount=40, balance_after=60, is_allin=False)
    assert await database.get_balance(user_id=42) == 60


async def test_place_bet_clamps_to_available_balance() -> None:
    """Over-betting turns into an all-in for the remaining balance."""
    await database.add_balance(user_id=42, name="alice", amount=25)
    placed = await database.place_bet(user_id=42, name="alice", requested_bet=100)
    assert placed == database.PlacedBet(amount=25, balance_after=0, is_allin=True)


async def test_place_bet_rejects_empty_or_invalid_wager() -> None:
    """Users with no points, or invalid wager amounts, cannot start a bet."""
    assert await database.place_bet(user_id=404, name="nobody", requested_bet=10) is None
    await database.add_balance(user_id=42, name="alice", amount=10)
    assert await database.place_bet(user_id=42, name="alice", requested_bet=0) is None
    assert await database.get_balance(user_id=42) == 10


async def test_place_bet_prevents_concurrent_double_spend() -> None:
    """Two simultaneous all-ins must not spend the same balance twice."""
    await database.add_balance(user_id=42, name="alice", amount=100)
    results = await asyncio.gather(
        database.place_bet(user_id=42, name="alice", requested_bet=100),
        database.place_bet(user_id=42, name="alice", requested_bet=100),
    )
    placed = [result for result in results if result is not None]
    rejected = [result for result in results if result is None]
    assert placed == [database.PlacedBet(amount=100, balance_after=0, is_allin=False)]
    assert rejected == [None]
    assert await database.get_balance(user_id=42) == 0
    account = await database.get_account(user_id=42)
    assert account is not None
    _, _, _, total_spent = account
    assert total_spent == 100


async def test_settle_game_clamps_at_zero() -> None:
    """A loss larger than the balance must clamp the balance at zero."""
    await database.add_balance(user_id=42, name="alice", amount=10)
    new = await database.settle_game(user_id=42, name="alice", delta=-1000)
    assert new == 0


async def test_settle_game_positive_pays_out() -> None:
    """Positive delta credits the account and increments total_earned."""
    await database.add_balance(user_id=42, name="alice", amount=10)
    new = await database.settle_game(user_id=42, name="alice", delta=50)
    assert new == 60


async def test_get_balance_unknown_user_returns_zero() -> None:
    """Reading a never-seen user returns zero, not an error."""
    assert await database.get_balance(user_id=999) == 0


async def test_transfer_moves_currency_between_users() -> None:
    """Successful transfer debits sender and credits receiver atomically."""
    await database.add_balance(user_id=1, name="alice", amount=200)
    result = await database.transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=80
    )
    assert result == database.TransferResult(sender_balance=120, receiver_balance=80)
    assert await database.get_balance(user_id=1) == 120
    assert await database.get_balance(user_id=2) == 80


async def test_transfer_rejects_self() -> None:
    """Transfers to oneself must be rejected."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    result = await database.transfer(
        sender_id=1, sender_name="alice", receiver_id=1, receiver_name="alice", amount=10
    )
    assert result is None
    assert await database.get_balance(user_id=1) == 100


async def test_transfer_rejects_insufficient_balance() -> None:
    """Transfers exceeding the sender's balance must be rejected."""
    await database.add_balance(user_id=1, name="alice", amount=10)
    result = await database.transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=100
    )
    assert result is None
    assert await database.get_balance(user_id=1) == 10
    assert await database.get_balance(user_id=2) == 0


async def test_transfer_prevents_concurrent_double_spend() -> None:
    """Concurrent transfers from one sender cannot reuse the same points."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    results = await asyncio.gather(
        database.transfer(
            sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=80
        ),
        database.transfer(
            sender_id=1, sender_name="alice", receiver_id=3, receiver_name="carol", amount=80
        ),
    )
    assert sum(result is not None for result in results) == 1
    assert results.count(None) == 1
    assert await database.get_balance(user_id=1) == 20
    assert await database.get_balance(user_id=2) + await database.get_balance(user_id=3) == 80


async def test_transfer_concurrent_credits_accumulate() -> None:
    """Concurrent transfers into one receiver must not lose either credit."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    await database.add_balance(user_id=2, name="bob", amount=100)
    results = await asyncio.gather(
        database.transfer(
            sender_id=1, sender_name="alice", receiver_id=3, receiver_name="carol", amount=80
        ),
        database.transfer(
            sender_id=2, sender_name="bob", receiver_id=3, receiver_name="carol", amount=70
        ),
    )
    assert all(result is not None for result in results)
    assert {result.sender_balance for result in results if result is not None} == {20, 30}
    assert max(result.receiver_balance for result in results if result is not None) == 150
    assert await database.get_balance(user_id=3) == 150


@pytest.mark.parametrize(argnames="amount", argvalues=[0, -1, -1000])
async def test_transfer_rejects_non_positive(amount: int) -> None:
    """Transfers with non-positive amounts must be rejected."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    result = await database.transfer(
        sender_id=1, sender_name="alice", receiver_id=2, receiver_name="bob", amount=amount
    )
    assert result is None


async def test_top_n_orders_by_balance_descending() -> None:
    """Leaderboard returns the top accounts ordered by balance."""
    await database.add_balance(user_id=1, name="alice", amount=100, avatar_url="https://cdn/a.png")
    await database.add_balance(user_id=2, name="bob", amount=300, avatar_url="https://cdn/b.png")
    await database.add_balance(user_id=3, name="carol", amount=50)
    rows = await database.top_n(limit=2)
    assert rows == [(2, "bob", 300, "https://cdn/b.png"), (1, "alice", 100, "https://cdn/a.png")]


async def test_top_n_excludes_specified_users() -> None:
    """Excluded user IDs (e.g. the bot's house ledger) must not appear in the result."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    await database.add_balance(user_id=2, name="bob", amount=300)
    await database.add_balance(user_id=99, name="house", amount=999)
    rows = await database.top_n(limit=10, exclude_user_ids=(99,))
    assert all(row[0] != 99 for row in rows)
    assert rows[0][:3] == (2, "bob", 300)


async def test_house_settle_allows_negative_balance() -> None:
    """House ledger keeps a true running net even when the dealer is down."""
    await database.house_settle(user_id=99, name="house", delta=-500)
    assert await database.get_balance(user_id=99) == -500


async def test_house_settle_accumulates_gross_flows() -> None:
    """Wins and losses both accumulate gross totals, not just the net balance."""
    await database.house_settle(user_id=99, name="house", delta=200)
    await database.house_settle(user_id=99, name="house", delta=-300)
    account = await database.get_account(user_id=99)
    assert account is not None
    name, balance, total_earned, total_spent = account
    assert name == "house"
    assert balance == -100
    assert total_earned == 200
    assert total_spent == 300


async def test_settle_wager_updates_player_and_house() -> None:
    """Shared wager settlement applies net delta and mirrors house P&L."""
    await database.add_balance(user_id=1, name="alice", amount=100)

    settlement = await settle_wager(
        player_id=1,
        player_account_name="alice",
        dealer_id=99,
        dealer_name="house",
        bet=40,
        delta=40,
    )
    assert settlement.payout == 40
    assert settlement.new_balance == 140
    assert settlement.house_balance == -40


async def test_get_account_returns_none_for_unseen_user() -> None:
    """Unknown users return None instead of a synthetic zero row."""
    assert await database.get_account(user_id=12345) is None


async def test_settle_blackjack_round_updates_player_and_house() -> None:
    """Shared Blackjack settlement applies net delta and mirrors house P&L."""
    await database.add_balance(user_id=1, name="alice", amount=100)

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
    assert await database.get_balance(user_id=99) == -50


async def test_blackjack_view_finalizes_once_when_called_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent finalization attempts must not pay out one Blackjack hand twice."""
    cleanup_messages: list[object] = []

    def fake_schedule_game_message_delete(*, message: object, delay: float = 180) -> None:
        cleanup_messages.append(message)

    monkeypatch.setattr(
        target=views, name="schedule_game_message_delete", value=fake_schedule_game_message_delete
    )
    await database.add_balance(user_id=1, name="alice", amount=100)

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

    assert await database.get_balance(user_id=1) == 150
    assert await database.get_balance(user_id=99) == -50
    assert dealer.settle_calls == 1
    assert message.edit_calls == 1
    assert cleanup_messages == [message]


async def test_blackjack_view_timeout_auto_stands_and_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A player who walks away is treated as standing and the wager resolves."""
    cleanup_messages: list[object] = []

    def fake_schedule_game_message_delete(*, message: object, delay: float = 180) -> None:
        cleanup_messages.append(message)

    monkeypatch.setattr(
        target=views, name="schedule_game_message_delete", value=fake_schedule_game_message_delete
    )
    await database.add_balance(user_id=1, name="alice", amount=100)

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
    assert await database.get_balance(user_id=1) == 50
    assert await database.get_balance(user_id=99) == 50
    assert dealer.settle_calls == 1
    assert message.edit_calls == 1
    assert cleanup_messages == [message]


async def test_blackjack_view_locks_actions_while_finalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late Hit cannot mutate a hand that is already finalizing from Stand."""
    cleanup_messages: list[object] = []
    settlement_started = asyncio.Event()
    continue_settlement = asyncio.Event()

    def fake_schedule_game_message_delete(*, message: object, delay: float = 180) -> None:
        cleanup_messages.append(message)

    async def delayed_settle_blackjack_round(**_kwargs: object) -> BlackjackSettlement:
        settlement_started.set()
        await continue_settlement.wait()
        return BlackjackSettlement(
            outcome="win", delta=50, payout=50, new_balance=150, house_balance=-50, detail="win"
        )

    monkeypatch.setattr(
        target=views, name="schedule_game_message_delete", value=fake_schedule_game_message_delete
    )
    monkeypatch.setattr(
        target=views, name="settle_blackjack_round", value=delayed_settle_blackjack_round
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

    hit_button, stand_button = view.children
    stand_task = asyncio.create_task(coro=stand_button.callback(_InteractionStub(message=message)))
    await settlement_started.wait()

    hit_task = asyncio.create_task(coro=hit_button.callback(_InteractionStub(message=message)))
    await asyncio.sleep(delay=0)

    assert len(view.round_state.players[0].cards) == 2
    continue_settlement.set()
    await asyncio.gather(stand_task, hit_task)

    assert len(view.round_state.players[0].cards) == 2
    assert dealer.settle_calls == 1
    assert dealer.hint_calls == 0
    assert message.edit_calls == 1
    assert cleanup_messages == [message]


async def test_add_balance_concurrent_credits_accumulate() -> None:
    """Concurrent credits on the same user must not lose updates.

    The old read-modify-write path would race: two coroutines read 100,
    both compute 110, last commit wins, the first +10 silently vanishes.
    UPSERT serializes the writes inside SQLite so both increments land.
    """
    await database.add_balance(user_id=42, name="alice", amount=100)
    await asyncio.gather(*[
        database.add_balance(user_id=42, name="alice", amount=10) for _ in range(20)
    ])
    assert await database.get_balance(user_id=42) == 300


async def test_add_balance_concurrent_first_sight_does_not_raise() -> None:
    """Two concurrent first-sight credits on the same user must not raise.

    The old `session.get()`-then-`session.add()` path would see both
    coroutines find ``None``, both INSERT, and one would raise
    `IntegrityError`. UPSERT collapses the race into a deterministic merge.
    """
    results = await asyncio.gather(*[
        database.add_balance(user_id=42, name="alice", amount=10) for _ in range(8)
    ])
    assert all(isinstance(value, int) for value in results)
    assert await database.get_balance(user_id=42) == 80


async def test_settle_game_concurrent_credits_accumulate() -> None:
    """Concurrent positive settlements on the same user must not lose updates."""
    await database.add_balance(user_id=42, name="alice", amount=100)
    await asyncio.gather(*[
        database.settle_game(user_id=42, name="alice", delta=10) for _ in range(10)
    ])
    assert await database.get_balance(user_id=42) == 200


async def test_house_settle_concurrent_updates_accumulate() -> None:
    """The dealer's hot row mustn't lose updates under concurrent settlements.

    Every player wager mirrors into the dealer's ledger row, so this is the
    single hottest row in the schema. The old read-modify-write path could
    silently drop one of two simultaneous house settlements.
    """
    await asyncio.gather(*[
        database.house_settle(user_id=99, name="house", delta=10) for _ in range(10)
    ])
    account = await database.get_account(user_id=99)
    assert account is not None
    _, balance, total_earned, total_spent = account
    assert balance == 100
    assert total_earned == 100
    assert total_spent == 0


async def test_apply_round_settlement_is_atomic() -> None:
    """Player delta and house mirror share one transaction and one return."""
    await database.add_balance(user_id=1, name="alice", amount=100)

    player_balance, house_balance = await database.apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=40,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=-40,
    )
    assert player_balance == 140
    assert house_balance == -40
    assert await database.get_balance(user_id=1) == 140
    assert await database.get_balance(user_id=99) == -40


async def test_apply_round_settlement_loss_debits_player_and_house() -> None:
    """A loss debits the player and credits the house."""
    await database.add_balance(user_id=1, name="alice", amount=100)

    player_balance, house_balance = await database.apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-40,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=40,
    )
    assert player_balance == 60
    assert house_balance == 40
    account = await database.get_account(user_id=1)
    assert account is not None
    _, _, total_earned, total_spent = account
    assert total_earned == 100
    assert total_spent == 40


async def test_apply_round_settlement_loss_can_make_player_negative() -> None:
    """Deferred settlement still collects a loss after the balance was spent elsewhere."""
    await database.add_balance(user_id=1, name="alice", amount=25)

    player_balance, house_balance = await database.apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-40,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=40,
    )

    assert player_balance == -15
    assert house_balance == 40


# Daily check-in ------------------------------------------------------------


async def test_checkin_first_time_credits_base_reward() -> None:
    """A first check-in pays the base reward and persists a streak of 1."""
    result = await database.checkin(user_id=1, name="alice")
    assert result is not None
    assert result.amount == database.BASE_CHECKIN_REWARD_AMOUNT
    assert result.streak == 1
    assert result.is_vip is False
    assert result.new_balance == database.BASE_CHECKIN_REWARD_AMOUNT


async def test_checkin_same_day_is_rejected() -> None:
    """A second check-in within the same Taipei day must return None."""
    first = await database.checkin(user_id=1, name="alice")
    assert first is not None
    second = await database.checkin(user_id=1, name="alice")
    assert second is None
    assert await database.get_balance(user_id=1) == first.new_balance


async def test_checkin_consecutive_day_advances_streak() -> None:
    """A check-in on the next calendar day bumps the streak by 1."""
    first = await database.checkin(user_id=1, name="alice")
    assert first is not None
    # Backdate the previous check-in to yesterday Taipei
    yesterday = datetime.now(tz=database.TAIWAN_TIMEZONE) - timedelta(days=1)
    async with database.open_session() as session:
        await session.execute(
            statement=database
            .update(database.UserAccount)
            .where(database.UserAccount.user_id == 1)
            .values(last_checkin_at=yesterday)
        )
        await session.commit()
    second = await database.checkin(user_id=1, name="alice")
    assert second is not None
    assert second.streak == 2
    assert second.amount > first.amount


async def test_checkin_streak_cycles_back_to_one_after_seven() -> None:
    """Day 8 in a row resets back to streak 1."""
    await database.checkin(user_id=1, name="alice")
    async with database.open_session() as session:
        await session.execute(
            statement=database
            .update(database.UserAccount)
            .where(database.UserAccount.user_id == 1)
            .values(
                last_checkin_at=datetime.now(tz=database.TAIWAN_TIMEZONE) - timedelta(days=1),
                checkin_streak=database.CHECKIN_STREAK_CYCLE,
            )
        )
        await session.commit()
    result = await database.checkin(user_id=1, name="alice")
    assert result is not None
    assert result.streak == 1


async def test_checkin_missed_day_resets_streak_to_one() -> None:
    """Skipping a day resets the streak back to 1."""
    await database.checkin(user_id=1, name="alice")
    async with database.open_session() as session:
        await session.execute(
            statement=database
            .update(database.UserAccount)
            .where(database.UserAccount.user_id == 1)
            .values(
                last_checkin_at=datetime.now(tz=database.TAIWAN_TIMEZONE) - timedelta(days=3),
                checkin_streak=4,
            )
        )
        await session.commit()
    result = await database.checkin(user_id=1, name="alice")
    assert result is not None
    assert result.streak == 1


async def test_checkin_vip_gets_double_base() -> None:
    """A VIP account starts at 2x base before the streak multiplier."""
    await database.add_balance(user_id=1, name="alice", amount=database.VIP_PURCHASE_COST)
    purchase = await database.buy_vip(user_id=1, name="alice")
    assert purchase is not None
    result = await database.checkin(user_id=1, name="alice")
    assert result is not None
    assert result.is_vip is True
    assert result.amount == 2 * database.BASE_CHECKIN_REWARD_AMOUNT


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
    assert database.checkin_reward(streak=streak, is_vip=is_vip) == expected


async def test_checkin_logs_audit_row() -> None:
    """A successful check-in writes one CHECKIN_REWARD row tagged with the streak."""
    result = await database.checkin(user_id=1, name="alice")
    assert result is not None
    async with database.open_session() as session:
        rows = (
            await session.execute(
                statement=select(
                    database.PointTransaction.kind,
                    database.PointTransaction.delta,
                    database.PointTransaction.note,
                ).where(database.PointTransaction.user_id == 1)
            )
        ).all()
    assert rows == [(database.TransactionKind.CHECKIN_REWARD.value, result.amount, "streak 1")]


# VIP purchase --------------------------------------------------------------


async def test_buy_vip_sets_flag_and_debits_balance() -> None:
    """A successful purchase costs ``VIP_PURCHASE_COST`` and flips ``is_vip``."""
    await database.add_balance(user_id=1, name="alice", amount=database.VIP_PURCHASE_COST + 100)
    result = await database.buy_vip(user_id=1, name="alice")
    assert result is not None
    assert result.new_balance == 100
    assert result.cost == database.VIP_PURCHASE_COST
    assert await database.get_vip(user_id=1) is True


async def test_buy_vip_rejects_insufficient_balance() -> None:
    """Users without enough points cannot purchase VIP."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    result = await database.buy_vip(user_id=1, name="alice")
    assert result is None
    assert await database.get_vip(user_id=1) is False


async def test_buy_vip_rejects_existing_vip() -> None:
    """A second purchase by an existing VIP returns None and does not re-debit."""
    await database.add_balance(user_id=1, name="alice", amount=database.VIP_PURCHASE_COST * 2)
    first = await database.buy_vip(user_id=1, name="alice")
    assert first is not None
    second = await database.buy_vip(user_id=1, name="alice")
    assert second is None
    assert await database.get_balance(user_id=1) == database.VIP_PURCHASE_COST


async def test_buy_vip_rejects_unseen_user() -> None:
    """A user without a row cannot purchase (no balance to debit)."""
    assert await database.buy_vip(user_id=999, name="ghost") is None


async def test_buy_vip_logs_audit_row() -> None:
    """A successful purchase records one VIP_PURCHASE audit row."""
    await database.add_balance(user_id=1, name="alice", amount=database.VIP_PURCHASE_COST)
    await database.buy_vip(user_id=1, name="alice")
    async with database.open_session() as session:
        rows = (
            await session.execute(
                statement=select(
                    database.PointTransaction.kind, database.PointTransaction.delta
                ).where(database.PointTransaction.user_id == 1)
            )
        ).all()
    assert rows == [(database.TransactionKind.VIP_PURCHASE.value, -database.VIP_PURCHASE_COST)]


async def test_get_vip_unknown_user_returns_false() -> None:
    """Unknown users report no VIP perk rather than raising."""
    assert await database.get_vip(user_id=12345) is False


# Loss leaderboard ----------------------------------------------------------


async def test_top_losers_only_lists_net_negative_players() -> None:
    """A player with a positive casino net does not appear on the loss board."""
    await database.add_balance(user_id=1, name="alice", amount=1_000)
    await database.add_balance(user_id=2, name="bob", amount=1_000)
    await database.apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-300,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=300,
    )
    await database.apply_round_settlement(
        player_id=2,
        player_account_name="bob",
        player_delta=200,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=-200,
    )
    rows = await database.top_losers(limit=10, exclude_user_ids=(99,))
    assert [(row[0], row[1], row[2]) for row in rows] == [(1, "alice", 300)]


async def test_top_losers_orders_by_loss_magnitude() -> None:
    """The leaderboard sorts from biggest loss to smallest."""
    for user_id, name, loss in [(1, "alice", 100), (2, "bob", 500), (3, "carol", 250)]:
        await database.add_balance(user_id=user_id, name=name, amount=loss)
        await database.apply_round_settlement(
            player_id=user_id,
            player_account_name=name,
            player_delta=-loss,
            dealer_id=99,
            dealer_name="house",
            dealer_delta=loss,
        )
    rows = await database.top_losers(limit=10, exclude_user_ids=(99,))
    assert [(row[0], row[2]) for row in rows] == [(2, 500), (3, 250), (1, 100)]


async def test_top_losers_excludes_specified_users() -> None:
    """``exclude_user_ids`` filters the house ledger out of the report."""
    await database.add_balance(user_id=1, name="alice", amount=500)
    await database.apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-500,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=500,
    )
    rows = await database.top_losers(limit=10, exclude_user_ids=(99,))
    assert all(row[0] != 99 for row in rows)


async def test_top_losers_ignores_events_before_today() -> None:
    """Audit rows older than today's Taipei midnight do not count."""
    await database.add_balance(user_id=1, name="alice", amount=500)
    await database.apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        player_delta=-500,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=500,
    )
    past = datetime.now(tz=database.TAIWAN_TIMEZONE) - timedelta(days=2)
    async with database.open_session() as session:
        await session.execute(
            statement=database.update(database.PointTransaction).values(occurred_at=past)
        )
        await session.commit()
    assert await database.top_losers(limit=10, exclude_user_ids=(99,)) == []


async def test_top_losers_empty_when_no_casino_activity() -> None:
    """Without any CASINO_BET / CASINO_PAYOUT rows the leaderboard is empty."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    assert await database.top_losers(limit=10, exclude_user_ids=(99,)) == []


# VIP blackjack settlement -------------------------------------------------


async def test_settle_wager_applies_vip_bonus_on_win() -> None:
    """A VIP player wins 1.5x of the base delta; house mirrors the boosted amount."""
    await database.add_balance(user_id=1, name="alice", amount=database.VIP_PURCHASE_COST)
    purchase = await database.buy_vip(user_id=1, name="alice")
    assert purchase is not None
    settlement = await settle_wager(
        player_id=1,
        player_account_name="alice",
        dealer_id=99,
        dealer_name="house",
        bet=100,
        delta=100,
    )
    assert settlement.delta == 150
    assert settlement.house_balance == -150


async def test_settle_wager_keeps_loss_unchanged_for_vip() -> None:
    """The VIP perk does not soften losses."""
    await database.add_balance(user_id=1, name="alice", amount=database.VIP_PURCHASE_COST + 1_000)
    purchase = await database.buy_vip(user_id=1, name="alice")
    assert purchase is not None
    settlement = await settle_wager(
        player_id=1,
        player_account_name="alice",
        dealer_id=99,
        dealer_name="house",
        bet=100,
        delta=-100,
    )
    assert settlement.delta == -100
    assert settlement.house_balance == 100
