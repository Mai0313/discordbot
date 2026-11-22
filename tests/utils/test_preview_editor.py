"""Unit tests for the shared fixed-cadence preview editor."""

import asyncio

import pytest

from discordbot.utils.preview_editor import CadencePreviewEditor

from tests.helpers.casting import make_forbidden, make_not_found, make_server_error
from tests.helpers.logfire_capture import capture_logs


async def test_preview_editor_runs_ticks_and_waits_out_inflight_write_on_stop() -> None:
    """`stop` sets the stop event rather than cancelling, so an in-flight Discord edit completes."""
    started = asyncio.Event()
    release = asyncio.Event()
    completed = 0

    async def write_snapshot() -> None:
        nonlocal completed
        started.set()
        await release.wait()
        completed += 1

    editor = CadencePreviewEditor(
        write_snapshot=write_snapshot, interval_seconds=0.005, message_id=42
    )
    editor.start()
    assert editor.is_running
    # Starting an already-running editor keeps the same task.
    editor.start()
    await asyncio.wait_for(started.wait(), timeout=5)

    stop_task = asyncio.create_task(coro=editor.stop())
    await asyncio.sleep(0.01)
    assert not stop_task.done()

    release.set()
    await asyncio.wait_for(stop_task, timeout=5)
    assert completed == 1
    assert not editor.is_running
    # Calling stop on an already-stopped editor is a safe no-op.
    await editor.stop()


@pytest.mark.parametrize(
    ("error", "level", "channel_id", "expected_fields"),
    [
        (make_not_found(), "info", 99, {"message_id": 42}),
        (make_forbidden(), "warn", None, {"message_id": 42}),
        (make_forbidden(), "warn", 99, {"channel_id": 99, "message_id": 42}),
    ],
    ids=["not_found", "forbidden_without_channel", "forbidden_with_channel"],
)
async def test_preview_editor_stops_immediately_on_terminal_discord_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    level: str,
    channel_id: int | None,
    expected_fields: dict[str, int],
) -> None:
    """`NotFound` and `Forbidden` never recover on retry, so the editor logs untraced and exits."""
    attempts = 0

    async def write_snapshot() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    records = {
        name: capture_logs(monkeypatch=monkeypatch, level=name) for name in ("info", "warn")
    }
    editor = CadencePreviewEditor(
        write_snapshot=write_snapshot, interval_seconds=0.005, message_id=42, channel_id=channel_id
    )

    await asyncio.wait_for(editor.run(), timeout=5)

    assert attempts == 1
    assert [(name, fields) for name, found in records.items() for _, fields in found] == [
        (level, expected_fields)
    ]


async def test_preview_editor_logs_transient_errors_once_across_ticks_and_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected write errors keep ticking and stay suppressed across retries via `reset`."""
    attempts = 0
    seen_three = asyncio.Event()

    async def write_snapshot() -> None:
        nonlocal attempts
        attempts += 1
        if attempts >= 3:
            seen_three.set()
        raise make_server_error()

    warns = capture_logs(monkeypatch=monkeypatch, level="warn")
    editor = CadencePreviewEditor(
        write_snapshot=write_snapshot, interval_seconds=0.005, message_id=42
    )

    editor.start()
    await asyncio.wait_for(seen_three.wait(), timeout=5)
    await editor.stop()

    # After reset(), a second cycle runs ticks again without emitting a duplicate warning.
    seen_four = asyncio.Event()
    original_attempts = attempts

    async def write_again() -> None:
        nonlocal attempts
        attempts += 1
        seen_four.set()
        raise make_server_error()

    editor.write_snapshot = write_again
    editor.reset()
    editor.start()
    await asyncio.wait_for(seen_four.wait(), timeout=5)
    await editor.stop()

    assert attempts > original_attempts
    assert [fields["error_type"] for _, fields in warns] == ["HTTPException"]
    assert warns[0][1]["_exc_info"] is not None


async def test_preview_editor_stop_catches_crashed_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crashed background task is logged on `stop` without propagating into the caller."""
    warns = capture_logs(monkeypatch=monkeypatch, level="warn")

    async def noop() -> None:
        return None

    async def boom() -> None:
        raise RuntimeError("task exploded")

    editor = CadencePreviewEditor(write_snapshot=noop, interval_seconds=0.005, message_id=42)
    editor._task = asyncio.create_task(coro=boom())
    await asyncio.sleep(0)

    await editor.stop()

    assert not editor.is_running
    assert [fields["error_type"] for _, fields in warns] == ["RuntimeError"]
    assert warns[0][1]["message_id"] == 42
    assert warns[0][1]["_exc_info"] is not None


async def test_preview_editor_reports_not_running_after_terminal_exit_and_clears_on_cancel() -> (
    None
):
    """`is_running` reflects task completion before `stop()`, and `stop()` clears `_task` on cancel."""
    exited = asyncio.Event()

    async def fail_not_found() -> None:
        exited.set()
        raise make_not_found()

    editor = CadencePreviewEditor(
        write_snapshot=fail_not_found, interval_seconds=0.005, message_id=42
    )
    editor.start()
    await asyncio.wait_for(exited.wait(), timeout=5)
    assert editor._task is not None
    await asyncio.wait_for(editor._task, timeout=5)
    assert not editor.is_running

    started = asyncio.Event()
    gate = asyncio.Event()

    async def slow_write() -> None:
        started.set()
        await gate.wait()

    cancelling_editor = CadencePreviewEditor(
        write_snapshot=slow_write, interval_seconds=0.005, message_id=42
    )
    cancelling_editor.start()
    bg_task = cancelling_editor._task
    assert bg_task is not None
    await asyncio.wait_for(started.wait(), timeout=5)
    stop_task = asyncio.create_task(coro=cancelling_editor.stop())
    await asyncio.sleep(0.01)
    stop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stop_task
    assert cancelling_editor._task is None
    assert not cancelling_editor.is_running
    gate.set()
    await asyncio.gather(bg_task, return_exceptions=True)
