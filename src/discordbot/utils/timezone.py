"""Shared Asia/Taipei timezone helpers for persisted timestamps.

Tables stamped via SQLite's `CURRENT_TIMESTAMP` or Discord snowflakes use UTC instead and do not
pass through these helpers.
"""

from typing import Final
from datetime import datetime, timezone, timedelta

TAIWAN_TIMEZONE: Final[timezone] = timezone(offset=timedelta(hours=8), name="Asia/Taipei")


def database_now() -> datetime:
    """Returns the Asia/Taipei wall-clock timestamp used for persisted rows."""
    return datetime.now(tz=TAIWAN_TIMEZONE)


def as_taipei(dt: datetime) -> datetime:
    """Returns `dt` re-interpreted in Asia/Taipei (treating naive as Taipei).

    Args:
        dt (datetime): Datetime to convert or attach `TAIWAN_TIMEZONE` to.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TAIWAN_TIMEZONE)
    return dt.astimezone(tz=TAIWAN_TIMEZONE)
