"""Lifecycle helpers for the tasks a reply starts before it knows whether it needs them.

A turn speculates: the reply context and the attachment uploads start while the route call is
still in flight, and a turn that fails before consuming them has to throw them away. These
helpers are how such a task is awaited, bounded or drained without ever orphaning it or losing
the exception it raised off-route.
"""

import asyncio
from collections.abc import Awaitable

import logfire


def _report_discarded_failure(exc: Exception, label: str, message_id: int) -> None:
    """Records a speculative task that failed after the turn stopped needing its result."""
    logfire.warn(
        "Discarded speculative task failed",
        task_label=label,
        error_type=type(exc).__name__,
        message_id=message_id,
        _exc_info=exc,
    )


async def discard_task[TaskResultT](
    task: asyncio.Task[TaskResultT], label: str, message_id: int
) -> None:
    """Cancels and drains a speculative task, logging any non-cancellation failure under `label`.

    Link-context tasks that own an internal deadline use `drain_deadline_bound_task` instead.

    Args:
        task (asyncio.Task[TaskResultT]): Speculative task to cancel and await.
        label (str): Subsystem name for failure logs.
        message_id (int): Triggering Discord message id for log correlation.
    """
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        _report_discarded_failure(exc=exc, label=label, message_id=message_id)


async def await_deadline_bound_task[DeadlineT](
    task: asyncio.Task[DeadlineT], deadline: float, label: str, message_id: int
) -> DeadlineT:
    """Returns the result of `task`, draining it cleanly if the caller is cancelled first.

    Shields `task` so an outer cancellation before `deadline` cancels and drains `task` via
    `drain_deadline_bound_task` without interrupting its post-deadline cleanup.

    Args:
        task (asyncio.Task[DeadlineT]): Deadline-bound task to await.
        deadline (float): Monotonic event-loop deadline owned by `task`.
        label (str): Source name for failure logs if drained.
        message_id (int): Triggering Discord message id for log correlation.
    """
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await drain_deadline_bound_task(
            task=task, deadline=deadline, label=label, message_id=message_id
        )
        raise


async def drain_deadline_bound_task[DeadlineT](
    task: asyncio.Task[DeadlineT], deadline: float, label: str, message_id: int
) -> None:
    """Cancels `task` if `deadline` has not passed, then waits for its cleanup to finish.

    Args:
        task (asyncio.Task[DeadlineT]): Deadline-bound task to drain.
        deadline (float): Monotonic event-loop deadline owned by `task`.
        label (str): Source name for failure logs.
        message_id (int): Triggering Discord message id for log correlation.
    """
    if not task.done() and asyncio.get_running_loop().time() < deadline:
        task.cancel()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                break
        except Exception:
            break
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        _report_discarded_failure(exc=exc, label=label, message_id=message_id)


async def discard_link_tasks[DeadlineT](
    link_tasks: dict[str, asyncio.Task[DeadlineT]], deadline: float, message_id: int
) -> None:
    """Drains all remaining `link_tasks` against `deadline` and clears the mapping.

    Args:
        link_tasks (dict[str, asyncio.Task[DeadlineT]]): Active link-context tasks keyed by source
            name.
        deadline (float): Shared monotonic event-loop deadline for the link builds.
        message_id (int): Triggering Discord message id for log correlation.
    """
    for name, task in link_tasks.items():
        await drain_deadline_bound_task(
            task=task, deadline=deadline, label=name, message_id=message_id
        )
    link_tasks.clear()


async def run_until_deadline[DeadlineT](
    awaitable: Awaitable[DeadlineT], deadline: float
) -> DeadlineT:
    """Returns the result of `awaitable` bounded by the remaining time until `deadline`.

    Relies on `asyncio.wait_for` alone rather than a post-await clock check so a result completed
    before `deadline` is not rejected if a busy event loop resumes this coroutine late.

    Args:
        awaitable (Awaitable[DeadlineT]): Cancellation-propagating coroutine or future to run.
        deadline (float): Monotonic event-loop deadline (`loop.time()`).
    """
    event_loop = asyncio.get_running_loop()
    remaining_seconds = max(0.0, deadline - event_loop.time())
    return await asyncio.wait_for(fut=awaitable, timeout=remaining_seconds)
