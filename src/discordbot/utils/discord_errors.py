"""Discord error classifiers and structured failure reporting."""

from typing import Any, Literal, cast

import logfire
from nextcord import NotFound, Forbidden, HTTPException

DiscordLogLevel = Literal["debug", "info", "warn", "error"]


def is_reply_target_gone(error: HTTPException) -> bool:
    """Returns whether a reply send failed with `NotFound` or Discord's `50035` code.

    `50035` is Discord's generic `Invalid Form Body` code (used for both a deleted reply target
    and any rejected send or edit payload), so callers should use this only as a heuristic on
    sends carrying a message reference and fall back to a non-reply send rather than dropping
    the message.

    Args:
        error (HTTPException): Exception raised by the reply send.
    """
    return isinstance(error, NotFound) or error.code == 50035


def _emit_log(
    level: DiscordLogLevel,
    message: str,
    fields: dict[str, object],
    exc_info: Exception | None = None,
) -> None:
    """Emits one logfire record at `level`, attaching `exc_info` only when provided."""
    # `logfire.<level>` types `_tags`/`_exc_info` alongside `**attributes: Any`, so spreading
    # `dict[str, object]` fails type checking even when none of those reserved keys are present.
    attributes = cast("dict[str, Any]", fields)
    if level == "debug":
        if exc_info is not None:
            logfire.debug(message, _exc_info=exc_info, **attributes)
        else:
            logfire.debug(message, **attributes)
    elif level == "info":
        if exc_info is not None:
            logfire.info(message, _exc_info=exc_info, **attributes)
        else:
            logfire.info(message, **attributes)
    elif level == "warn":
        if exc_info is not None:
            logfire.warn(message, _exc_info=exc_info, **attributes)
        else:
            logfire.warn(message, **attributes)
    elif exc_info is not None:
        logfire.error(message, _exc_info=exc_info, **attributes)
    else:
        logfire.error(message, **attributes)


def report_discord_failure(  # noqa: PLR0913 -- shared failure router carries per-branch messages and levels
    error: Exception,
    forbidden_message: str,
    other_message: str,
    not_found_message: str | None = None,
    not_found_level: DiscordLogLevel = "info",
    not_found_exc_info: bool = False,
    other_level: DiscordLogLevel = "warn",
    **fields: object,
) -> None:
    """Logs a Discord API failure with consistent `code`, `error_type`, and traceback policy.

    `Forbidden` always logs at `warn` with `code` and `error_type` and no traceback (expected
    when channel overwrites exclude the bot). `NotFound` logs at `not_found_level` with `code`
    and `error_type` (or stays silent when `not_found_message` is None), attaching a traceback
    only when `not_found_exc_info` is True. Any other exception logs at `other_level` with
    `error_type`, its traceback, and `code` when it is an `HTTPException`.

    Args:
        error (Exception): Exception raised by the Discord call.
        forbidden_message (str): Log message for `Forbidden` refusals.
        other_message (str): Log message for 5xx, timeouts, and other unexpected failures.
        not_found_message (str | None): Log message for `NotFound`, or None to stay silent.
        not_found_level (DiscordLogLevel): Log level for `NotFound` records.
        not_found_exc_info (bool): Whether to attach the traceback to `NotFound` records.
        other_level (DiscordLogLevel): Log level for all other exceptions.
        **fields (object): Structured subject fields attached to the emitted log record.
    """
    payload: dict[str, object] = {**fields, "error_type": type(error).__name__}
    if isinstance(error, NotFound):
        if not_found_message is None:
            return
        payload["code"] = error.code
        _emit_log(
            level=not_found_level,
            message=not_found_message,
            fields=payload,
            exc_info=error if not_found_exc_info else None,
        )
        return
    if isinstance(error, Forbidden):
        payload["code"] = error.code
        _emit_log(level="warn", message=forbidden_message, fields=payload)
        return
    if isinstance(error, HTTPException):
        payload["code"] = error.code
    _emit_log(level=other_level, message=other_message, fields=payload, exc_info=error)
