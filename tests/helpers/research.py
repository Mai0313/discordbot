"""Doubles for the deep-research tests: scripted Interactions streams and a logging thread."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from tests.helpers.discord_mocks import FakeDiscordMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from google.genai.interactions import InteractionSSEEvent


class _FakeStream:
    """Async iterator over scripted SSE events; can raise after a prefix to simulate a drop."""

    def __init__(self, events: list[object], raise_after: int | None = None) -> None:
        self._events = list(events)
        self._raise_after = raise_after
        self._yielded = 0

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> object:
        if self._raise_after is not None and self._yielded >= self._raise_after:
            raise RuntimeError("stream dropped")
        if not self._events:
            raise StopAsyncIteration
        self._yielded += 1
        return self._events.pop(0)


class _FakeInteractions:
    """Fakes `client.aio.interactions`: `create`/`get(stream=True)` yield scripted streams.

    A non-stream `get(id=...)` returns the terminal interaction (the authoritative final read).
    """

    def __init__(self, streams: list[_FakeStream], terminal: object) -> None:
        self._streams = list(streams)
        self._terminal = terminal
        self.create_kwargs: dict[str, object] = {}
        self.stream_get_calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> _FakeStream:
        self.create_kwargs = kwargs
        return self._streams.pop(0)

    async def get(self, **kwargs: object) -> object:
        if kwargs.get("stream"):
            self.stream_get_calls.append(kwargs)
            return self._streams.pop(0)
        return self._terminal


def _fake_client(streams: list[_FakeStream], terminal: object) -> SimpleNamespace:
    return SimpleNamespace(
        aio=SimpleNamespace(interactions=_FakeInteractions(streams=streams, terminal=terminal))
    )


def _as_event(fake: object) -> "InteractionSSEEvent":
    """Views a fabricated SSE event double as the real SDK union a production signature expects.

    Production discriminates on `.event_type`, not isinstance, so a SimpleNamespace event is safe.
    """
    return cast("InteractionSSEEvent", fake)


def _created_event(interaction_id: str = "int_9", event_id: str = "e1") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="interaction.created",
        event_id=event_id,
        interaction=SimpleNamespace(id=interaction_id, model="m"),
    )


def _thought_event(text: str, event_id: str = "e2") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="step.delta",
        event_id=event_id,
        delta=SimpleNamespace(type="thought_summary", content=SimpleNamespace(text=text)),
    )


def _completed_event(event_id: str = "e9") -> SimpleNamespace:
    return SimpleNamespace(
        event_type="interaction.completed", event_id=event_id, interaction=SimpleNamespace()
    )


def _terminal_interaction(
    status: str = "completed", input_tokens: int = 10, output_tokens: int = 5
) -> SimpleNamespace:
    return SimpleNamespace(
        id="int_9",
        status=status,
        output_text="# Report\nbody",
        usage=SimpleNamespace(total_input_tokens=input_tokens, total_output_tokens=output_tokens),
        steps=[],
    )


# The `_Anchor` author owns every run below, and each run lands in this thread.
_OWNER_ID = 300
_THREAD_ID = 50


class _RunStatus:
    """A run's opening status message, whose edits land in its thread's write log."""

    id = 2

    def __init__(self, thread: "_RunThread") -> None:
        """Binds the status to the thread whose write log and failure it shares."""
        self.thread = thread

    async def edit(self, **kwargs: object) -> None:
        """Records the edit, or fails it the way its thread fails every later write."""
        if self.thread.error is not None:
            raise self.thread.error
        self.thread.writes.append(kwargs)


class _RunThread:
    """A research thread logging every write of a run in order, status edits included.

    With `error` set, every write fails with it except the opening status post while
    `status_posts` holds.
    """

    id = _THREAD_ID

    def __init__(
        self,
        error: Exception | None = None,
        status_posts: bool = True,
        earlier: list[FakeDiscordMessage] | None = None,
    ) -> None:
        """Initializes the write log, the guild upload limit, how writes fail, and the history."""
        self.guild = SimpleNamespace(filesize_limit=10 * 1024 * 1024)
        self.writes: list[dict[str, object]] = []
        self.error = error
        self.status_posts = status_posts
        self.sends = 0
        self.deleted = False
        self.earlier = earlier or []

    async def history(self, **kwargs: object) -> "AsyncIterator[FakeDiscordMessage]":
        """Hands back what the thread held before the restart, oldest first."""
        del kwargs
        for message in self.earlier:
            yield message

    async def send(self, **kwargs: object) -> _RunStatus:
        """Records a post and answers with the message it created, as Discord does."""
        self.sends += 1
        if self.error is not None and not (self.sends == 1 and self.status_posts):
            raise self.error
        self.writes.append(kwargs)
        return _RunStatus(thread=self)

    async def delete(self) -> None:
        """Records the thread's deletion, or fails it the way every later write fails."""
        if self.error is not None:
            raise self.error
        self.deleted = True
