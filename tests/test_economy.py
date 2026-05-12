"""Tests for the economy persistence layer."""

from random import SystemRandom
import asyncio
from pathlib import Path
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import create_async_engine

from discordbot.cogs._games import views
from discordbot.cogs._economy import database
from discordbot.cogs._games.views import BlackjackView
from discordbot.cogs._games.blackjack import Card, BlackjackHand
from discordbot.cogs._games.settlement import settle_wager, settle_blackjack_round


class _DealerStub:
    """Minimal dealer stub for BlackjackView settlement tests."""

    def __init__(self) -> None:
        self.settle_calls = 0

    async def settle(self, **_kwargs: object) -> str:
        """Returns deterministic banter and tracks settlement calls."""
        self.settle_calls += 1
        await asyncio.sleep(delay=0)
        return "settled"


class _MessageStub:
    """Minimal message stub that records edit calls."""

    def __init__(self) -> None:
        self.edit_calls = 0

    async def edit(self, **_kwargs: object) -> None:
        """Records a Discord message edit."""
        self.edit_calls += 1


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


async def test_existing_economy_db_gets_avatar_url_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-avatar economy DB is migrated before avatar-aware writes run."""
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
    """Shared wager settlement credits payout and mirrors house P&L."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    placed = await database.place_bet(user_id=1, name="alice", requested_bet=40)
    assert placed is not None

    settlement = await settle_wager(
        player_id=1,
        player_account_name="alice",
        dealer_id=99,
        dealer_name="house",
        bet=placed.amount,
        delta=40,
    )
    assert settlement.payout == 80
    assert settlement.new_balance == 140
    assert settlement.house_balance == -40


async def test_get_account_returns_none_for_unseen_user() -> None:
    """Unknown users return None instead of a synthetic zero row."""
    assert await database.get_account(user_id=12345) is None


async def test_settle_blackjack_round_updates_player_and_house() -> None:
    """Shared Blackjack settlement credits the player and mirrors house P&L."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    placed = await database.place_bet(user_id=1, name="alice", requested_bet=50)
    assert placed is not None

    hand = BlackjackHand(rng=SystemRandom(), bet=placed.amount)
    hand.player = [Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")]
    hand.dealer = [Card(rank="10", suit="♣"), Card(rank="8", suit="♦")]
    hand.finished = True

    settlement = await settle_blackjack_round(
        hand=hand, player_id=1, player_account_name="alice", dealer_id=99, dealer_name="house"
    )
    assert settlement.delta == 50
    assert settlement.payout == 100
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
    placed = await database.place_bet(user_id=1, name="alice", requested_bet=50)
    assert placed is not None

    hand = BlackjackHand(rng=SystemRandom(), bet=placed.amount)
    hand.player = [Card(rank="10", suit="♠"), Card(rank="Q", suit="♥")]
    hand.dealer = [Card(rank="10", suit="♣"), Card(rank="8", suit="♦")]
    hand.finished = True

    dealer = _DealerStub()
    message = _MessageStub()
    view = BlackjackView(
        dealer=dealer,
        hand=hand,
        owner_id=1,
        author_name="alice",
        player_name="Alice",
        dealer_id=99,
        dealer_name="house",
        balance_after_bet=placed.balance_after,
    )

    await asyncio.gather(view._finalize(message=message), view._finalize(message=message))

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
    placed = await database.place_bet(user_id=1, name="alice", requested_bet=50)
    assert placed is not None

    hand = BlackjackHand(rng=SystemRandom(), bet=placed.amount)
    hand.player = [Card(rank="10", suit="♠"), Card(rank="8", suit="♥")]
    hand.dealer = [Card(rank="10", suit="♣"), Card(rank="Q", suit="♦")]

    dealer = _DealerStub()
    message = _MessageStub()
    view = BlackjackView(
        dealer=dealer,
        hand=hand,
        owner_id=1,
        author_name="alice",
        player_name="Alice",
        dealer_id=99,
        dealer_name="house",
        balance_after_bet=placed.balance_after,
    )
    view.message = message

    await view.on_timeout()

    assert hand.finished is True
    assert await database.get_balance(user_id=1) == 50
    assert await database.get_balance(user_id=99) == 50
    assert dealer.settle_calls == 1
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
    """Player credit and house mirror share one transaction and one return."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    placed = await database.place_bet(user_id=1, name="alice", requested_bet=40)
    assert placed is not None

    player_balance, house_balance = await database.apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        payout=80,
        dealer_id=99,
        dealer_name="house",
        dealer_delta=-40,
    )
    assert player_balance == 140
    assert house_balance == -40
    assert await database.get_balance(user_id=1) == 140
    assert await database.get_balance(user_id=99) == -40


async def test_apply_round_settlement_zero_payout_only_touches_house() -> None:
    """A pure loss reads the player balance without re-crediting it."""
    await database.add_balance(user_id=1, name="alice", amount=100)
    placed = await database.place_bet(user_id=1, name="alice", requested_bet=40)
    assert placed is not None

    player_balance, house_balance = await database.apply_round_settlement(
        player_id=1,
        player_account_name="alice",
        payout=0,
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
