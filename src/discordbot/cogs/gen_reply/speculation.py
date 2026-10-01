"""Lifecycle helpers for the tasks a reply starts before it knows whether it needs them.

A turn speculates: the reply context and the attachment uploads start while the route call is
still in flight, and a turn that fails before consuming them has to throw them away. These
helpers are how such a task is awaited, bounded or drained without ever orphaning it or losing
the exception it raised off-route.
"""

import asyncio
from collections.abc import Awaitable

import logfire


def _report_discarded_failure(*, exc: Exception, label: str, message_id: int) -> None:
    """Records a speculative task that failed after the turn stopped needing its result."""
    logfire.warn(
        "Discarded speculative task failed",
        task_label=label,
        error_type=type(exc).__name__,
        message_id=message_id,
        _exc_info=exc,
    )


async def discard_task[TaskResultT](
    *, task: asyncio.Task[TaskResultT], label: str, message_id: int
) -> None:
    """Cancels and drains a speculative task so its exception is retrieved.

    The except is deliberately broad: this drains unrelated subsystems, so anything they can
    raise must be swallowed here rather than surfacing on a route that already decided it does
    not need the result. `label` names which one failed, since the tasks are otherwise
    indistinguishable at this point. A link-context build is drained by
    `drain_deadline_bound_task` instead, which must not steal its own deadline's cancellation.
    """
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        _report_discarded_failure(exc=exc, label=label, message_id=message_id)


async def await_deadline_bound_task[DeadlineT](
    *, task: asyncio.Task[DeadlineT], deadline: float, label: str, message_id: int
) -> DeadlineT:
    """Awaits a self-deadline-bound task while preserving its cancellation cleanup ownership."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await drain_deadline_bound_task(
            task=task, deadline=deadline, label=label, message_id=message_id
        )
        raise


async def drain_deadline_bound_task[DeadlineT](
    *, task: asyncio.Task[DeadlineT], deadline: float, label: str, message_id: int
) -> None:
    """Cancels before a task's deadline or preserves its in-progress deadline cleanup."""
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
    *, link_tasks: dict[str, asyncio.Task[DeadlineT]], deadline: float, message_id: int
) -> None:
    """Drains link builds without stealing cancellation from their shared deadline."""
    for name, task in link_tasks.items():
        await drain_deadline_bound_task(
            task=task, deadline=deadline, label=name, message_id=message_id
        )
    link_tasks.clear()


async def run_until_deadline[DeadlineT](
    *, awaitable: Awaitable[DeadlineT], deadline: float
) -> DeadlineT:
    """Runs a cancellation-propagating builder until its fixed event-loop deadline.

    Registered builders all propagate `CancelledError`, so `wait_for` alone owns the boundary.
    A clock check after this await would reject a pre-deadline result when a busy event loop only
    resumes this wrapper after the deadline.
    """
    event_loop = asyncio.get_running_loop()
    remaining_seconds = max(0.0, deadline - event_loop.time())
    return await asyncio.wait_for(fut=awaitable, timeout=remaining_seconds)
