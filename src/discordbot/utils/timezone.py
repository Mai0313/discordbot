"""Shared Asia/Taipei timezone helpers for persisted timestamps.

Economy and stock storage both stamp rows in Asia/Taipei and re-interpret
stored datetimes in that zone. This module is the single source for those
helpers so the two database layers stay aligned.
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
