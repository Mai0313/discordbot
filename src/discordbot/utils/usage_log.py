"""Append-only monthly JSONL usage records, separate from `./data/logs`.

Records only feature invocations (no outcomes, message content, or command arguments).
`user_id` is the grouping key; `user_name` is a write-time username snapshot for readability.
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

# `slash`: application-command invocation; `reply`: AI reply turn (named by route).
UsageKind = Literal["slash", "reply"]

# Serializes worker-thread appends; threading.Lock because writes run off the event loop.
_WRITE_LOCK = threading.Lock()


class UsageLogConfig(BaseSettings):
    """Usage-recording settings, read from environment variables."""

    enabled: bool = Field(
        default=True,
        description="Whether feature usage is recorded at all; when false no file is created.",
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
    """One recorded use of one feature."""

    at: datetime = Field(
        ..., description="When it was used, in Asia/Taipei so grouping by day is a string slice."
    )
    kind: UsageKind = Field(..., description="Whether this was a slash command or an AI reply.")
    name: str = Field(
        ...,
        description="The full command path, or the reply's route.",
        examples=["memory server show", "QA"],
    )
    user_id: int = Field(..., description="Discord user ID that used the feature.")
    # Defaulted rather than required so older monthly JSONL rows without this field still parse.
    user_name: str = Field(
        default="",
        description="The user's Discord username as it read at write time; a label, not a key.",
        examples=["weichenglee"],
    )
    guild_id: int | None = Field(..., description="Discord guild ID, or None in a DM.")
    channel_id: int | None = Field(..., description="Discord channel ID, when there is one.")


def _append_sync(directory: Path, record: UsageRecord) -> None:
    """Appends `record` as one JSON line to its monthly JSONL file under `_WRITE_LOCK`."""
    with _WRITE_LOCK:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{record.at:%Y-%m}.jsonl"
        # Reopen per append to handle month rollovers and externally moved files.
        with path.open(mode="a", encoding="utf-8") as handle:
            handle.write(f"{record.model_dump_json()}\n")


class UsageRecorder(BaseModel):
    """Appends `UsageRecord` entries to monthly JSONL files."""

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
        """Records one feature use off the event loop, logging and swallowing write failures.

        Args:
            kind (UsageKind): Whether this was a slash command or an AI reply.
            name (str): Full command path or reply route name.
            user_id (int): Discord user id that used the feature.
            user_name (str): Discord username snapshot at write time.
            guild_id (int | None): Discord guild id, or None in a DM.
            channel_id (int | None): Discord channel id, when there is one.
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
