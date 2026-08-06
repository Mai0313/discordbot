"""Persistent Blackjack round history for the games cog.

Every settled Blackjack round writes one row per seated player (the bot player included) into
`data/database/games.db`. The query side reads the most recent rounds for a single player so
`/games blackjack_history` can show someone's recent hands, bets, dealer hands, and results.
It is a separate file from the table itself because the two halves run at different times: the
rules and the money settle inside `blackjack_views.py`, and this store is written afterwards
from a background task, the one thing that round deliberately does off its critical path. A
failure here therefore costs a history row and nothing else, so nothing in this module may be
awaited from inside a live round.

The engine is a module-level `AsyncEngine` singleton, mirroring the economy and stock stores.
Each operation opens an `AsyncSession` bound to the current `_engine`, so tests can monkeypatch
`_engine` per-test and every subsequent call sees the swap. `games.db` is shared with fishing
and the message-cleanup table, each of which owns its own engine and its own tables over the
same file; the PRAGMA setup all three agree on comes from `utils/sqlite_config.py`.

Money and bet columns use `StoredInteger` decimal text so large wagers do not inherit SQLite's
64-bit integer ceiling. The rich per-hand card detail (player hands, dealer hand, insurance) is
serialized into one typed `BlackjackHistoryPayload` JSON column; the flat `user_id` /
`created_at` / `outcome` / `delta` columns drive filtering, ordering, and summaries. That split
is also how the shape evolves: `_ensure_schema` only ever runs `create_all`, so an existing file
never gains a column, while every payload field carries a default and a row written before a
field existed still reads back.
"""

from typing import Any, cast
import asyncio
from datetime import datetime
from collections.abc import Sequence

from sqlalchemy import Text, Index, String, Boolean, Integer, DateTime, event, select
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.typings.games import (
    Card,
    SettleOutcome,
    BlackjackHistoryHand,
    BlackjackPlayerResult,
    BlackjackHistoryRecord,
    BlackjackHandSettlement,
    BlackjackHistoryPayload,
    BlackjackHistoryInsurance,
)
from discordbot.utils.timezone import as_taipei as _as_taipei
from discordbot.utils.timezone import database_now as _database_now
from discordbot.utils.asyncio_locks import LoopLocalLock
from discordbot.utils.sqlite_config import ensure_sqlite_hooks, configure_sqlite_connection
from discordbot.cogs.games.blackjack import hand_value
from discordbot.utils.stored_integer import StoredInteger

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/database/games.db")
_schema_ready_for: AsyncEngine | None = None
_schema_lock = LoopLocalLock()


def _configure_sqlite_connection(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Configures a newly opened games-history SQLite connection.

    The one place this store's PRAGMA choices are made, and both are the shared helper's
    defaults: no `PRAGMA foreign_keys`, since nothing here references another table, and the
    `StoredInteger` UDFs registered for the decimal-text `bet` / `delta` columns.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
    """
    configure_sqlite_connection(dbapi_connection=dbapi_connection)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Configures a newly opened SQLite connection.

    Bound to the import-time `_engine` by the decorator, and handed to `ensure_sqlite_hooks` by
    every session open so an engine a test swapped in gets the same listener; it stays a named
    module-level function so that second registration can be deduplicated.

    Args:
        dbapi_connection (Any): The freshly opened DBAPI connection.
        _connection_record (Any): SQLAlchemy's pool record for the connection, unused.
    """
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _configure_sqlite_on_checkout(
    dbapi_connection: object, _connection_record: object, _connection_proxy: object
) -> None:
    """Configures pooled connections from test-swapped engines.

    `connect` fires only when the pool opens a new connection, so a fixture that created the
    schema before monkeypatching `_engine` leaves one pooled connection the connect listener
    never reaches; this catches it on its way out of the pool instead.

    Args:
        dbapi_connection (object): The connection being handed out of the pool.
        _connection_record (object): SQLAlchemy's pool record for the connection, unused.
        _connection_proxy (object): SQLAlchemy's proxy wrapping the checked-out connection,
            unused.
    """
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


class Base(DeclarativeBase):
    """Base class for games-history ORM models."""

    pass


class BlackjackRoundResult(Base):
    """One seated player's settled result for a single Blackjack round.

    A round writes one row per seat, all sharing a `round_id` and one `created_at`. The
    composite `(user_id, created_at)` index is the read path: one player's rows, newest first.
    Nothing selects on `round_id` today — it is written so a round's seats can be reassembled,
    and its index is there for that read whenever one appears.
    """

    __tablename__ = "blackjack_round_result"
    __table_args__ = (
        Index("ix_blackjack_round_result_user_created", "user_id", "created_at"),
        Index("ix_blackjack_round_result_round", "round_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[str] = mapped_column(String(length=36), nullable=False)
    channel_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    guild_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    user_name: Mapped[str] = mapped_column(String(length=128), default="", nullable=False)
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_vip: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    bet: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    outcome: Mapped[str] = mapped_column(String(length=32), nullable=False)
    delta: Mapped[int] = mapped_column(StoredInteger(), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_database_now, nullable=False
    )


def _current_schema_lock() -> asyncio.Lock:
    """Returns the schema bootstrap lock bound to the current event loop.

    Returns:
        The lock for the running loop, rebuilt whenever the loop changed, so a per-test loop
        never waits on a primitive bound to a dead one.
    """
    return _schema_lock.get()


async def _ensure_schema() -> None:
    """Bootstraps the games-history schema once per engine.

    The listeners are re-installed on every call, because an engine a test monkeypatched in was
    never reached by the import-time `@event.listens_for`. `_schema_ready_for` holds engine
    identity rather than a flag for the same reason, and the check is repeated under the lock so
    two rounds settling together still run `create_all` once.
    """
    global _schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    if _schema_ready_for is _engine:
        return
    async with _current_schema_lock():
        if _schema_ready_for is _engine:
            return
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _schema_ready_for = _engine


def open_session() -> AsyncSession:
    """Creates an async session bound to the current games-history engine.

    `_engine` is read at call time rather than through a cached sessionmaker, so a test swap
    takes effect on the next call; the listeners are installed here for the same reason
    `_ensure_schema` installs them. `expire_on_commit=False` keeps a written row readable after
    its commit.

    Returns:
        A new session the caller owns closing, usually as an async context manager.
    """
    ensure_sqlite_hooks(
        engine=_engine,
        on_connect_fn=_configure_sqlite,
        on_checkout_fn=_configure_sqlite_on_checkout,
    )
    return AsyncSession(bind=_engine, expire_on_commit=False)


def _history_hand(hand: BlackjackHandSettlement) -> BlackjackHistoryHand:
    """Projects a settled sub-hand into its persisted history snapshot.

    `total` is evaluated once here, so displaying a stored hand never re-runs the rules engine's
    ace demotion over cards that can no longer change.

    Args:
        hand (BlackjackHandSettlement): One settled sub-hand of a player's round.

    Returns:
        The snapshot stored inside the row's payload.
    """
    return BlackjackHistoryHand(
        cards=list(hand.cards),
        total=hand_value(cards=hand.cards),
        bet=hand.bet,
        outcome=hand.outcome,
        delta=hand.delta,
        five_card_bonus=hand.five_card_bonus,
        five_card_twenty_one=hand.five_card_twenty_one,
        doubled=hand.doubled,
        surrendered=hand.surrendered,
        is_split_hand=hand.is_split_hand,
    )


def _history_payload(
    *, result: BlackjackPlayerResult, dealer_cards: Sequence[Card], dealer_total: int
) -> BlackjackHistoryPayload:
    """Builds the full per-player snapshot stored in the history row.

    The dealer's hand is copied into every seat's payload rather than stored once for the round,
    so a single row renders on its own without reading the other seats back.

    Args:
        result (BlackjackPlayerResult): The seat and the settlement it is being paired with.
        dealer_cards (Sequence[Card]): The dealer's final hand for the round.
        dealer_total (int): The dealer's final hand value.

    Returns:
        The payload serialized into the row's JSON column.
    """
    settlement = result.settlement
    insurance = (
        BlackjackHistoryInsurance(
            bet=settlement.insurance.bet,
            won=settlement.insurance.won,
            delta=settlement.insurance.delta,
        )
        if settlement.insurance is not None
        else None
    )
    return BlackjackHistoryPayload(
        hands=[_history_hand(hand=hand) for hand in settlement.hands],
        dealer_cards=list(dealer_cards),
        dealer_total=dealer_total,
        insurance=insurance,
        vip_bonus=settlement.vip_bonus,
        five_card_bonus=settlement.five_card_bonus,
        balance_at_start=result.participant.balance_at_start,
        new_balance=settlement.new_balance,
    )


async def record_blackjack_history(  # noqa: PLR0913 -- round persistence needs full table context
    *,
    round_id: str,
    channel_id: int,
    guild_id: int,
    message_id: int,
    bot_user_id: int | None,
    results: Sequence[BlackjackPlayerResult],
    dealer_cards: Sequence[Card],
    dealer_total: int,
) -> None:
    """Persists one Blackjack round's per-player results in a single commit.

    One timestamp is taken before the loop and stamped on every row, so a round's seats carry
    the same ordering key and only their insertion `id` separates them. A seat is marked
    `is_bot` only when `bot_user_id` was given and matches it, so a table the bot never joined
    marks nothing. An empty `results` returns before the schema bootstrap, leaving a round with
    no seats touching no file at all.

    Args:
        round_id (str): Identifier shared by every row this call writes.
        channel_id (int): Discord channel the round was played in.
        guild_id (int): Discord guild the round was played in, or 0 for a DM.
        message_id (int): Discord message id of the settled table.
        bot_user_id (int | None): The bot's own user id, or None when it did not sit at this
            table.
        results (Sequence[BlackjackPlayerResult]): One entry per seated player, in seat order.
        dealer_cards (Sequence[Card]): The dealer's final hand.
        dealer_total (int): The dealer's final hand value.
    """
    if not results:
        return
    await _ensure_schema()
    now = _database_now()
    async with open_session() as session:
        for result in results:
            participant = result.participant
            settlement = result.settlement
            session.add(
                instance=BlackjackRoundResult(
                    round_id=round_id,
                    channel_id=channel_id,
                    guild_id=guild_id,
                    message_id=message_id,
                    user_id=participant.user_id,
                    user_name=participant.account_name,
                    is_bot=bot_user_id is not None and participant.user_id == bot_user_id,
                    is_vip=settlement.is_vip,
                    bet=participant.bet,
                    outcome=settlement.outcome,
                    delta=settlement.delta,
                    payload_json=_history_payload(
                        result=result, dealer_cards=dealer_cards, dealer_total=dealer_total
                    ).model_dump_json(),
                    created_at=now,
                )
            )
        await session.commit()


def _history_record(row: BlackjackRoundResult) -> BlackjackHistoryRecord:
    """Projects a stored row into the typed read model used for display.

    `outcome` is cast rather than validated: the column is plain text and every value in it was
    written from a `SettleOutcome`. `created_at` comes back naive from SQLite, so `as_taipei`
    re-attaches the zone it was stamped in instead of letting the container's local time shift
    it.

    Args:
        row (BlackjackRoundResult): One persisted round-result row.

    Returns:
        The record the history renderer reads.
    """
    return BlackjackHistoryRecord(
        round_id=row.round_id,
        channel_id=row.channel_id,
        guild_id=row.guild_id,
        message_id=row.message_id,
        user_id=row.user_id,
        user_name=row.user_name,
        is_bot=row.is_bot,
        is_vip=row.is_vip,
        bet=row.bet,
        outcome=cast("SettleOutcome", row.outcome),
        delta=row.delta,
        payload=BlackjackHistoryPayload.model_validate_json(row.payload_json),
        created_at=_as_taipei(dt=row.created_at),
    )


async def fetch_recent_blackjack_rounds(
    *, user_id: int, limit: int
) -> tuple[BlackjackHistoryRecord, ...]:
    """Returns the most recent settled rounds for one player, newest first.

    `created_at` has a per-round granularity, so two rounds settled in the same instant tie on
    it; the descending `id` is what keeps the order total instead of leaving it to whatever
    SQLite returns.

    Args:
        user_id (int): Discord user id whose rows are read; the bot's own id is a valid target.
        limit (int): Maximum number of rows to return.

    Returns:
        The player's rows newest first, empty when they have never sat at a settled table.
    """
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(BlackjackRoundResult)
            .where(BlackjackRoundResult.user_id == user_id)
            .order_by(BlackjackRoundResult.created_at.desc(), BlackjackRoundResult.id.desc())
            .limit(limit)
        )
        return tuple(_history_record(row=row) for row in result.scalars())
