"""Reads back what code under test logged, in place of logfire."""

from typing import Literal
from collections.abc import Callable

import pytest
import logfire

LogLevel = Literal["debug", "info", "warn", "error"]


def capture_logs(
    monkeypatch: pytest.MonkeyPatch, level: LogLevel
) -> list[tuple[str, dict[str, object]]]:
    """Records every `logfire.<level>` call for the rest of the test as `(message, fields)`.

    Every module logs through the one imported `logfire` module, so this catches a call from any
    of them, not only the module under test.
    """
    records: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        target=logfire,
        name=level,
        value=lambda message, **fields: records.append((message, fields)),
    )
    return records


def capture_levels(
    monkeypatch: pytest.MonkeyPatch, levels: tuple[LogLevel, ...]
) -> list[tuple[LogLevel, str, dict[str, object]]]:
    """Records every call at any of `levels` as `(level, message, fields)`, in call order.

    For a test whose finding is which level a failure was reported at; `capture_logs` reads one.
    """
    records: list[tuple[LogLevel, str, dict[str, object]]] = []

    def recorder(level: LogLevel) -> Callable[..., None]:
        return lambda message, **fields: records.append((level, message, fields))

    for level in levels:
        monkeypatch.setattr(target=logfire, name=level, value=recorder(level=level))
    return records
