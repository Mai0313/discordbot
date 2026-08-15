"""Shared Asia/Taipei timezone helpers for persisted timestamps.

The storage layers that reach for these stamp their rows in Asia/Taipei and
re-interpret stored datetimes in that zone; this module is the single source so
no layer grows its own copy. Not every table is in that zone — anything stamped
by SQLite's own `CURRENT_TIMESTAMP` or by a Discord snowflake time is UTC and
never comes through here.
"""

from typing import Final
from datetime import datetime, timezone, timedelta

TAIWAN_TIMEZONE: Final[timezone] = timezone(offset=timedelta(hours=8), name="Asia/Taipei")


def database_now() -> datetime:
    """Returns the Asia/Taipei wall-clock timestamp used for persisted rows."""
    return datetime.now(tz=TAIWAN_TIMEZONE)


def as_taipei(dt: datetime) -> datetime:
    """Returns `dt` re-interpreted in Asia/Taipei (treating naive as Taipei)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TAIWAN_TIMEZONE)
    return dt.astimezone(tz=TAIWAN_TIMEZONE)
