"""Persistent Blackjack round history for the games cog.

Every settled Blackjack round writes one row per seated player (the bot
player included) into `data/games.db`. The query side reads the most recent
rounds for a single player so `/games blackjack_history` can show someone's
recent hands, bets, dealer hands, and results.

The engine is a module-level `AsyncEngine` singleton, mirroring the economy
and stock stores. Each operation opens an `AsyncSession` bound to the current
`_engine`, so tests can monkeypatch `_engine` per-test and every subsequent
call sees the swap. Money and bet columns use `StoredInteger` decimal text so
large wagers do not inherit SQLite's 64-bit integer ceiling. The rich per-hand
card detail (player hands, dealer hand, insurance) is serialized into one typed
`BlackjackHistoryPayload` JSON column; the flat `user_id` / `created_at` /
`outcome` / `delta` columns drive filtering, ordering, and summaries.
"""

from typing import Any, cast
import asyncio
from datetime import datetime
from collections.abc import Sequence

from sqlalchemy import Text, Index, String, Boolean, Integer, DateTime, event, select
from sqlalchemy.orm import Mapped, DeclarativeBase, mapped_column
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from discordbot.utils.timezone import as_taipei as _as_taipei
from discordbot.utils.timezone import database_now as _database_now
from discordbot.typings.games import (
    Card,
    SettleOutcome,
    BlackjackPlayerResult,
    BlackjackHistoryHand,
    BlackjackHistoryRecord,
    BlackjackHistoryPayload,
    BlackjackHandSettlement,
    BlackjackHistoryInsurance,
)
from discordbot.utils.sqlite_config import configure_sqlite_connection
from discordbot.cogs._games.blackjack import hand_value
from discordbot.utils.stored_integer import StoredInteger

_engine: AsyncEngine = create_async_engine(url="sqlite+aiosqlite:///data/games.db")
_schema_ready_for: AsyncEngine | None = None
_schema_lock: asyncio.Lock | None = None
_schema_lock_loop: asyncio.AbstractEventLoop | None = None


def _configure_sqlite_connection(dbapi_connection: Any) -> None:  # noqa: ANN401 -- SQLAlchemy connection type depends on the driver
    """Configures a newly opened games-history SQLite connection."""
    configure_sqlite_connection(dbapi_connection=dbapi_connection)


@event.listens_for(_engine.sync_engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:  # noqa: ANN401 -- SQLAlchemy event signature is dynamically typed
    """Configures a newly opened SQLite connection."""
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _configure_sqlite_on_checkout(
    dbapi_connection: object, _connection_record: object, _connection_proxy: object
) -> None:
    """Configures pooled connections from test-swapped engines."""
    _configure_sqlite_connection(dbapi_connection=dbapi_connection)


def _ensure_sqlite_hooks(engine: AsyncEngine) -> None:
    """Installs SQLite connection hooks on the active engine."""
    if not event.contains(target=engine.sync_engine, identifier="connect", fn=_configure_sqlite):
        event.listen(target=engine.sync_engine, identifier="connect", fn=_configure_sqlite)
    if not event.contains(
        target=engine.sync_engine, identifier="checkout", fn=_configure_sqlite_on_checkout
    ):
        event.listen(
            target=engine.sync_engine, identifier="checkout", fn=_configure_sqlite_on_checkout
        )


class Base(DeclarativeBase):
    """Base class for games-history ORM models."""

    pass


class BlackjackRoundResult(Base):
    """One seated player's settled result for a single Blackjack round."""

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
    """Returns a schema bootstrap lock bound to the current event loop."""
    global _schema_lock, _schema_lock_loop  # noqa: PLW0603 -- module-level loop-local lock
    loop = asyncio.get_running_loop()
    if _schema_lock is None or _schema_lock_loop is not loop:
        _schema_lock = asyncio.Lock()
        _schema_lock_loop = loop
    return _schema_lock


async def _ensure_schema() -> None:
    """Bootstraps the games-history schema once per engine."""
    global _schema_ready_for  # noqa: PLW0603 -- module-level cache by engine identity
    _ensure_sqlite_hooks(engine=_engine)
    if _schema_ready_for is _engine:
        return
    async with _current_schema_lock():
        if _schema_ready_for is _engine:
            return
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _schema_ready_for = _engine


def open_session() -> AsyncSession:
    """Creates an async session bound to the current games-history engine."""
    _ensure_sqlite_hooks(engine=_engine)
    return AsyncSession(bind=_engine, expire_on_commit=False)


def _history_hand(hand: BlackjackHandSettlement) -> BlackjackHistoryHand:
    """Projects a settled sub-hand into its persisted history snapshot."""
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
    """Builds the full per-player snapshot stored in the history row."""
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


async def record_blackjack_history(
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
    """Persists one Blackjack round's per-player results in a single commit."""
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
    """Projects a stored row into the typed read model used for display."""
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
    """Returns the most recent settled rounds for one player, newest first."""
    await _ensure_schema()
    async with open_session() as session:
        result = await session.execute(
            statement=select(BlackjackRoundResult)
            .where(BlackjackRoundResult.user_id == user_id)
            .order_by(BlackjackRoundResult.created_at.desc(), BlackjackRoundResult.id.desc())
            .limit(limit)
        )
        return tuple(_history_record(row=row) for row in result.scalars())
