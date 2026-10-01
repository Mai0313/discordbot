"""Reads back what code under test logged, in place of logfire."""

from typing import Literal

import pytest
import logfire


def capture_logs(
    monkeypatch: pytest.MonkeyPatch, level: Literal["debug", "info", "warn", "error"]
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
