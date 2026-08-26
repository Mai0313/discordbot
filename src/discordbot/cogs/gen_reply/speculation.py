"""Lifecycle helpers for the tasks a reply starts before it knows whether it needs them.

A turn speculates: the reply context, the effort grade and the optional memory selection all
start while the route call is still in flight, and a non-QA route then throws most of it away.
These helpers are how such a task is awaited, bounded or drained without ever orphaning it or
losing the exception it raised off-route.
"""

import asyncio
import contextlib
from collections.abc import Awaitable

import logfire


async def discard_task[TaskResultT](
    *, task: asyncio.Task[TaskResultT], label: str = "speculative", message_id: int | None = None
) -> None:
    """Cancels and drains a speculative task so its exception is retrieved.

    The except is deliberately broad: this drains unrelated subsystems (prep, effort, parts,
    memory selection), so anything they can raise must be swallowed here rather than surfacing on
    a route that already decided it does not need the result. `label` names which one failed,
    since the tasks are otherwise indistinguishable at this point. A link-context build is drained
    by `drain_deadline_bound_task` instead, which must not steal its own deadline's cancellation.
    """
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logfire.warn(
            "Speculative reply context build failed off-route",
            task_label=label,
            error_type=type(exc).__name__,
            message_id=message_id,
            _exc_info=exc,
        )


async def await_gated[GatedT](
    *, task: asyncio.Task[GatedT], label: str, route_done: asyncio.Event, grace_seconds: float
) -> GatedT:
    """Awaits a side task with a grace period beginning when routing finishes.

    A task started before routing completes overlaps it without consuming the grace. A task
    started after routing receives the grace immediately. The task is always cancelled on exit
    so it never orphans.
    """
    route_wait = asyncio.create_task(coro=route_done.wait())
    try:
        await asyncio.wait({task, route_wait}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return task.result()
        return await asyncio.wait_for(fut=task, timeout=grace_seconds)
    finally:
        route_wait.cancel()
        # `route_done.wait()` has no other terminal state, so nothing else is worth catching.
        with contextlib.suppress(asyncio.CancelledError):
            await route_wait
        if not task.done():
            await discard_task(task=task, label=label)


async def await_deadline_bound_task[DeadlineT](
    *, task: asyncio.Task[DeadlineT], deadline: float, label: str
) -> DeadlineT:
    """Awaits a self-deadline-bound task while preserving its cancellation cleanup ownership."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await drain_deadline_bound_task(task=task, deadline=deadline, label=label)
        raise


async def drain_deadline_bound_task[DeadlineT](
    *, task: asyncio.Task[DeadlineT], deadline: float, label: str, message_id: int | None = None
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
        logfire.warn(
            "Speculative reply context build failed off-route",
            task_label=label,
            error_type=type(exc).__name__,
            message_id=message_id,
            _exc_info=exc,
        )


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
