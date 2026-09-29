"""Tests for `utils/asyncio_locks.py::spawn_tracked`, the one fire-and-forget spawner."""

import asyncio

import pytest

from discordbot.utils import asyncio_locks
from discordbot.utils.asyncio_locks import spawn_tracked


def _recorded_errors(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object]]]:
    """Captures what the spawner reports."""
    errors: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        target=asyncio_locks.logfire,
        name="error",
        value=lambda message, **fields: errors.append((message, fields)),
    )
    return errors


async def test_a_task_is_held_until_it_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The owner's set is the strong reference, and a clean finish reports nothing."""
    errors = _recorded_errors(monkeypatch=monkeypatch)
    release = asyncio.Event()
    tasks: set[asyncio.Task[None]] = set()

    async def wait_for_release() -> None:
        await release.wait()

    spawn_tracked(coro=wait_for_release(), tasks=tasks, name="held")

    assert len(tasks) == 1
    release.set()
    await asyncio.gather(*tasks)
    assert tasks == set()
    assert errors == []


async def test_an_escaped_failure_is_logged_with_the_task_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure the coroutine did not handle lands in the log instead of `sys.stderr`."""
    errors = _recorded_errors(monkeypatch=monkeypatch)
    tasks: set[asyncio.Task[None]] = set()

    async def fail() -> None:
        raise RuntimeError("boom")

    spawn_tracked(coro=fail(), tasks=tasks, name="failing")
    await asyncio.gather(*tasks, return_exceptions=True)

    assert tasks == set()
    assert len(errors) == 1
    _message, fields = errors[0]
    assert fields["task"] == "failing"
    assert fields["error_type"] == "RuntimeError"


async def test_a_cancelled_task_is_released_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancellation is not a failure, so it leaves the set without a report.

    Reading a cancelled task's exception raises inside the done callback, which asyncio hands to
    the loop's exception handler rather than to anything the test awaits, so that handler is
    what has to stay silent.
    """
    errors = _recorded_errors(monkeypatch=monkeypatch)
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(handler=lambda _loop, context: loop_errors.append(context))
    tasks: set[asyncio.Task[None]] = set()

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    try:
        spawn_tracked(coro=wait_forever(), tasks=tasks, name="cancelled")
        (task,) = tasks
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    finally:
        loop.set_exception_handler(handler=previous_handler)

    assert tasks == set()
    assert errors == []
    assert loop_errors == []
