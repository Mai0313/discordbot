"""Seeding and read helpers for tests that start from, or check, a known economy state."""

from pydantic import Field, BaseModel, ConfigDict
from sqlalchemy import select, update

from discordbot.utils.timezone import as_taipei, database_now
from discordbot.services.economy.database import (
    UserAccount,
    CasinoAccount,
    open_session,
    adjust_balance,
    _taipei_midnight,
    get_jackpot_snapshot,
)


class CasinoDailyStats(BaseModel):
    """Per-user current-day casino loss/win/net totals.

    All zero when no row exists or the stored counters belong to a previous
    Taipei day.
    """

    model_config = ConfigDict(frozen=True)

    daily_loss: int = Field(..., description="Gross current-day casino loss total.")
    daily_win: int = Field(..., description="Gross current-day casino win total.")
    daily_net: int = Field(..., description="Net current-day casino result (win minus loss).")


async def seed_balance(user_id: int, name: str, amount: int, avatar_url: str = "") -> int:
    """Credits `amount` through the manual adjustment path and returns the new balance.

    That path touches neither loan contracts nor the daily casino counters, so the seed
    leaves no trace a test could mistake for the behaviour under test. A zero amount
    writes nothing and creates no account.
    """
    result = await adjust_balance(user_id=user_id, name=name, delta=amount, avatar_url=avatar_url)
    return result.new_balance


async def hide_from_leaderboard(user_id: int) -> None:
    """Marks an existing account hidden from the public leaderboards.

    No runtime path sets the flag, so this writes the row directly. It leaves the
    leaderboard caches alone: hide before the first leaderboard read.
    """
    async with open_session() as session:
        await session.execute(
            statement=update(UserAccount)
            .where(UserAccount.user_id == user_id)
            .values(hide_from_leaderboard=True)
        )
        await session.commit()


async def get_jackpot_pool(game_id: str) -> int:
    """Returns a game's jackpot balance, replenishing a drained seeded pool first.

    A game with no pool row reads as 0.
    """
    snapshot = await get_jackpot_snapshot(game_id=game_id)
    return snapshot.balance


async def get_casino_daily_stats(user_id: int) -> CasinoDailyStats:
    """Returns the current-day casino loss/win/net for one user.

    Returns all-zero when no row exists or when the stored counters are from a
    previous Taipei day (the next casino settlement will reset them anyway).
    """
    today_midnight = _taipei_midnight(now=database_now())
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
        return CasinoDailyStats(daily_loss=0, daily_win=0, daily_net=0)
    daily_loss, daily_win, daily_net, day_started_at = row
    if day_started_at is None or as_taipei(dt=day_started_at) != today_midnight:
        return CasinoDailyStats(daily_loss=0, daily_win=0, daily_net=0)
    return CasinoDailyStats(daily_loss=daily_loss, daily_win=daily_win, daily_net=daily_net)
