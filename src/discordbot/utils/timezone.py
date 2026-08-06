"""Asia/Taipei timestamp helpers, shared by every layer that has to agree on the zone.

Every layer that persists a datetime stamps it through `database_now`, and the layers that
compare stored values re-read them through `as_taipei`; a few callers take `TAIWAN_TIMEZONE`
directly for a timestamp this bot never stored, such as a Discord message's `created_at` on
its way into a prompt. They are deliberately not enumerated here; a hand-written census rots
the moment the next caller lands. The helpers sit below all of them because the zone is a
cross-layer agreement rather than one storage layer's detail: the stamping side and the
reading side have to pick the same offset, and a mismatch does not fail, it silently moves a
day boundary. Check-in streaks, the daily casino counters and the Taiwan-style stock price
limit are all keyed on Taipei midnight.

`TAIWAN_TIMEZONE` is a fixed +08:00 offset rather than `ZoneInfo("Asia/Taipei")`: Taiwan has
kept UTC+8 with no DST since 1980 (1979 was its last DST year), so the offset is exact for
anything this bot stamps and costs no tz database at runtime. The `name=` is only what
`tzname()` reports; nothing ever looks it up.

The module stops there on purpose. It formats nothing for display, converts nothing back to
UTC, and does not decide where a day starts — `_taipei_midnight` in economy storage and
`tick_boundary` in the market helpers own that question in the layers that ask it.
"""

from typing import Final
from datetime import datetime, timezone, timedelta

TAIWAN_TIMEZONE: Final[timezone] = timezone(offset=timedelta(hours=8), name="Asia/Taipei")


def database_now() -> datetime:
    """Returns the current Asia/Taipei timestamp that persisted rows are stamped with.

    Handed straight to SQLAlchemy as a `default=` / `onupdate=` callable, so it has to stay
    zero-argument. SQLite keeps no offset, so the aware value written here comes back naive on
    read and means the right instant again only once `as_taipei` re-attaches the zone.

    Returns:
        The current time as an aware Asia/Taipei datetime.
    """
    return datetime.now(tz=TAIWAN_TIMEZONE)


def as_taipei(dt: datetime) -> datetime:
    """Returns `dt` in Asia/Taipei, reading a naive value as Taipei wall time.

    That naive branch is the whole point of the helper: a column declared
    `DateTime(timezone=True)` still reads back naive from SQLite, so `datetime.astimezone`
    would take every stored row for the container's local time and shift it. Assuming Taipei
    is what makes the write-then-read round trip exact.

    Args:
        dt (datetime): An aware timestamp, or a naive one already in Taipei wall time.

    Returns:
        An aware Asia/Taipei datetime: the same instant for an aware input, the same wall
        clock for a naive one.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TAIWAN_TIMEZONE)
    return dt.astimezone(tz=TAIWAN_TIMEZONE)
