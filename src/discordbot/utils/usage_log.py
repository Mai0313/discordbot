"""Append-only usage records, kept out of the runtime log on purpose.

`data/logs/<start time>.log` is debug-level, grows by roughly 2MB a day and gets cleaned
out by hand, so a usage history living inside it dies with it; it is also gated on
`LOG_LEVEL`, so raising the level on a deployment would silently stop recording. These
records therefore get their own directory, their own kill-switch, and a shape an agent
can read a month of in one pass.

What a record answers is "someone used this", nothing more: there is no success/failure
field, because how well a use went is what the runtime log is for, and the slash record
is written at the one point that sees an invocation before its outcome exists (see
`cogs/usage/cog.py`). Only numeric ids are stored — no names, no message content, no
command arguments — since every question this file exists to answer needs a stable
identifier and nothing else, names drift, and nothing prunes these files.

Held here rather than inside a cog because the two writers are different cogs — the slash
listener in `cogs/usage/cog.py` and the reply turn in `gen_reply/cog.py` — and one cog may
not import from another. `UsageLogConfig` is the kill-switch and destination, `UsageRecord`
the line shape, `UsageRecorder` the writer; the append runs off the event loop and `record`
swallows its failures, so recording can never cost the feature being recorded. Nothing here
reads the records back: they are written for an operator or an agent to sweep later.
"""

from typing import Literal
import asyncio
from pathlib import Path
from datetime import datetime
import threading

import logfire
from pydantic import Field, BaseModel, AliasChoices
from pydantic_settings import BaseSettings

from discordbot.utils.timezone import database_now

# What produced the record. `slash` is one application-command invocation; `reply` is one
# AI reply turn, named after the route it took.
UsageKind = Literal["slash", "reply"]

# Serialised writes: `record` hands the append to a worker thread, so two concurrent
# recorders would otherwise interleave inside one line. Deliberately a threading lock
# rather than one of `utils/asyncio_locks.py` — it is taken off the event loop and so
# never binds to one.
_WRITE_LOCK = threading.Lock()


class UsageLogConfig(BaseSettings):
    """Usage-recording settings, read from environment variables.

    Attributes:
        enabled: Kill-switch; when false nothing is recorded and no file is created.
        directory: Directory the monthly record files are written into.
    """

    enabled: bool = Field(
        default=True,
        description="Whether feature usage is recorded at all.",
        examples=[True],
        validation_alias=AliasChoices("USAGE_LOG_ENABLED"),
    )
    directory: str = Field(
        default="./data/usage",
        description="Directory the monthly usage record files are written into.",
        examples=["./data/usage"],
        validation_alias=AliasChoices("USAGE_LOG_DIR"),
    )


class UsageRecord(BaseModel):
    """One recorded use of one feature.

    Attributes:
        at: When it was used, stamped in Asia/Taipei so grouping by day is a string slice.
        kind: Whether this was a slash command or an AI reply.
        name: The command path (`memory server show`) or the reply's route (`QA`).
        user_id: Discord user ID that used it.
        guild_id: Discord guild ID, or None in a DM.
        channel_id: Discord channel ID, or None when the interaction carries none.
    """

    at: datetime = Field(..., description="When the feature was used, in Asia/Taipei.")
    kind: UsageKind = Field(..., description="Whether this was a slash command or an AI reply.")
    name: str = Field(
        ...,
        description="The full command path, or the reply's route.",
        examples=["memory server show", "QA"],
    )
    user_id: int = Field(..., description="Discord user ID that used the feature.")
    guild_id: int | None = Field(..., description="Discord guild ID, or None in a DM.")
    channel_id: int | None = Field(..., description="Discord channel ID, when there is one.")


def _append_sync(directory: Path, record: UsageRecord) -> None:
    """Appends one JSON line to the record's own month file.

    The handle is opened and closed per record instead of being held: a month rollover
    then needs no bookkeeping, and a file moved out from under a long-running bot cannot
    take further writes into an unlinked inode. Runs on a worker thread, so it takes
    `_WRITE_LOCK` for the whole mkdir-open-append-close. Nothing is swallowed here: an
    `OSError` propagates, and `UsageRecorder.record` is what turns it into a log line.

    Args:
        directory (Path): Directory the monthly files live in.
        record (UsageRecord): The use to append.
    """
    with _WRITE_LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{record.at:%Y-%m}.jsonl"
        with path.open(mode="a", encoding="utf-8") as handle:
            handle.write(f"{record.model_dump_json()}\n")


class UsageRecorder(BaseModel):
    """Appends usage records to a monthly JSONL file.

    One JSON object per line so a partial write can never cost the records before it, and
    so reading a month is `for line in file: json.loads(line)`.

    Attributes:
        config: The usage-recording configuration backing this recorder.
    """

    config: UsageLogConfig = Field(
        default_factory=UsageLogConfig,
        description="The usage-recording configuration backing this recorder.",
    )

    async def record(
        self,
        kind: UsageKind,
        name: str,
        user_id: int,
        guild_id: int | None,
        channel_id: int | None,
    ) -> None:
        """Records one use, off the event loop and best-effort.

        Recording must never cost the thing it records, so a failure to write is logged
        and swallowed rather than raised into the command or the reply pipeline. A
        disabled recorder returns before the timestamp is taken and creates no directory.

        Args:
            kind (UsageKind): Whether this is a slash invocation or an AI reply turn.
            name (str): The full command path, or the route the reply took.
            user_id (int): Discord user ID that used the feature.
            guild_id (int | None): Discord guild ID, or None in a DM.
            channel_id (int | None): Discord channel ID, or None when the interaction carries none.
        """
        if not self.config.enabled:
            return
        record = UsageRecord(
            at=database_now(),
            kind=kind,
            name=name,
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id,
        )
        try:
            await asyncio.to_thread(
                _append_sync, directory=Path(self.config.directory), record=record
            )
        except Exception as exc:
            # Broad on purpose: every failure mode here (unwritable dir, full disk, a
            # revoked mount) is one the recorded feature must not notice.
            logfire.warn(
                "Failed to record feature usage",
                kind=kind,
                name=name,
                error_type=type(exc).__name__,
                _exc_info=exc,
            )
