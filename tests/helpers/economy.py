"""Seeding helpers for tests that start from a known economy state."""

from sqlalchemy import update

from discordbot.services.economy.database import UserAccount, open_session, adjust_balance


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
