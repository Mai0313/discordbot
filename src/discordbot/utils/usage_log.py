"""Append-only usage records, kept out of the runtime log on purpose.

`data/logs/<start time>.log` is debug-level, grows by roughly 2MB a day and gets cleaned
out by hand, so a usage history living inside it dies with it; it is also gated on
`LOG_LEVEL`, so raising the level on a deployment would silently stop recording. These
records therefore get their own directory, their own kill-switch, and a shape an agent
can read a month of in one pass.

What a record answers is "someone used this", nothing more: there is no success/failure
field, because how well a use went is what the runtime log is for, and the slash record
is written at the one point that sees an invocation before its outcome exists (see
`cogs/usage/cog.py`). No message content and no command arguments are stored, since no
question this file exists to answer needs them and nothing prunes these files.

Who is stored twice over: `user_id` is the identifier every read groups by, and
`user_name` is the Discord username as it read at write time, kept only so an operator
does not have to resolve a hundred ids by hand. It is a snapshot and it drifts — someone
who renames leaves both names behind in the same file — so it is a label, never a key.
The per-guild display name is deliberately not what is stored: it differs per server and
changes far more often, so it would drift without even identifying anyone.
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
        user_name: That user's Discord username when the record was written.
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
    # Defaulted rather than required, for the records written before this field existed:
    # a month file is append-only and never rewritten, so a reader has to expect both
    # shapes forever. Every new record carries it, since `record` takes it as an argument.
    user_name: str = Field(
        default="",
        description="The user's Discord username as it read at write time; a label, not a key.",
        examples=["weichenglee"],
    )
    guild_id: int | None = Field(..., description="Discord guild ID, or None in a DM.")
    channel_id: int | None = Field(..., description="Discord channel ID, when there is one.")


def _append_sync(directory: Path, record: UsageRecord) -> None:
    """Appends one JSON line to the record's own month file.

    The handle is opened and closed per record instead of being held: a month rollover
    then needs no bookkeeping, and a file moved out from under a long-running bot cannot
    take further writes into an unlinked inode.
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

    async def record(  # noqa: PLR0913 -- one record's fields are all per-call inputs
        self,
        kind: UsageKind,
        name: str,
        user_id: int,
        user_name: str,
        guild_id: int | None,
        channel_id: int | None,
    ) -> None:
        """Records one use, off the event loop and best-effort.

        Recording must never cost the thing it records, so a failure to write is logged
        and swallowed rather than raised into the command or the reply pipeline.
        """
        if not self.config.enabled:
            return
        record = UsageRecord(
            at=database_now(),
            kind=kind,
            name=name,
            user_id=user_id,
            user_name=user_name,
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
