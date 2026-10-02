"""Shared doubles for the reply pipeline's direct-to-Google paths.

The Files API upload and the Interactions answer stream are each driven from more than one test
module, so their fakes live here once.
"""

from types import SimpleNamespace
from typing import Any
from datetime import UTC, datetime
from collections.abc import AsyncIterator

import pytest
from google.genai.types import File, FileState

from discordbot.cogs.gen_reply.files_api import poll_while_processing

# Every module that runs the Files API activation poll, each through its own name for it.
_POLL_CALLERS = (
    "discordbot.cogs.gen_reply.files_api",
    "discordbot.cogs.gen_reply.attachment.gemini_file_api",
)


class FakeGeminiFiles:
    """Fake async Gemini Files resource that records uploads and drives the activation poll.

    A file is named after its display name. `processing_rounds` makes `upload` return a
    PROCESSING file that flips to `final_state` after that many `get` polls, so the poll loop is
    exercised; a non-ACTIVE `final_state` (FAILED, say) drives the failed-processing branch.
    `download` answers any uri with fake MP4 bytes, the clip a finished video render names.
    """

    def __init__(
        self, processing_rounds: int = 0, final_state: FileState = FileState.ACTIVE
    ) -> None:
        """Initializes the upload records and the processing-to-final schedule."""
        self.upload_calls: list[tuple[str, str]] = []
        self.uploaded_sources: list[object] = []
        self.get_calls = 0
        self.processing_rounds = processing_rounds
        self.final_state = final_state
        self._remaining = 0

    def _file(self, name: str, state: FileState) -> SimpleNamespace:
        """Builds a fake uploaded-file object carrying the uri the answer references."""
        return SimpleNamespace(
            name=name,
            uri=f"https://files.test/{name}",
            state=state,
            error=None,
            expiration_time=datetime(2099, 1, 1, tzinfo=UTC),
        )

    async def upload(self, file: object, config: dict[str, str]) -> SimpleNamespace:
        """Records the upload as `(display_name, mime_type)` plus its source, and returns it."""
        display_name = config["display_name"]
        self.upload_calls.append((display_name, config["mime_type"]))
        self.uploaded_sources.append(file)
        self._remaining = self.processing_rounds
        state = FileState.PROCESSING if self.processing_rounds else self.final_state
        return self._file(name=display_name, state=state)

    async def get(self, name: str) -> SimpleNamespace:
        """Returns the polled file, flipping to the final state once the rounds elapse."""
        self.get_calls += 1
        self._remaining -= 1
        state = FileState.PROCESSING if self._remaining > 0 else self.final_state
        return self._file(name=name, state=state)

    async def download(self, file: object) -> bytes:
        """Returns fake MP4 bytes whatever the uri."""
        del file
        return b"mp4"


class FakeGeminiClient:
    """Fake Gemini client with its Files and Interactions resources under `aio`, as the real one.

    Files defaults to a fresh `FakeGeminiFiles`. Interactions has no default, so a path that
    reaches it in a test that staged none fails on the call instead of reading a canned answer.
    """

    def __init__(self, files: object | None = None, interactions: object | None = None) -> None:
        """Wires the resources under `aio.files` and `aio.interactions`."""
        self.aio = SimpleNamespace(files=files or FakeGeminiFiles(), interactions=interactions)


def skip_files_api_poll_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes every Files API activation poll re-read at once instead of waiting its interval.

    The interval is zeroed rather than `asyncio.sleep` patched, which would stop every other
    `asyncio.sleep` in the test from yielding too.
    """

    async def poll_at_once(**kwargs: Any) -> File:  # noqa: ANN401 -- forwarded untouched to the real poll
        """Polls exactly as the real loop does, with no wait between two reads."""
        kwargs["poll_interval_seconds"] = 0.0
        return await poll_while_processing(**kwargs)

    for module in _POLL_CALLERS:
        monkeypatch.setattr(f"{module}.poll_while_processing", poll_at_once)


async def event_stream(events: list[SimpleNamespace]) -> AsyncIterator[SimpleNamespace]:
    """Yields fake stream events in order."""
    for event in events:
        yield event


def interactions_turn_events() -> list[SimpleNamespace]:
    """A minimal Interactions stream: created, a thought, two text deltas, completed with usage."""
    return [
        SimpleNamespace(
            event_type="interaction.created",
            interaction=SimpleNamespace(model="gemini-3.1-pro-preview"),
        ),
        SimpleNamespace(
            event_type="step.delta",
            metadata=None,
            delta=SimpleNamespace(type="thought_summary", content=SimpleNamespace(text="hmm")),
        ),
        SimpleNamespace(
            event_type="step.delta",
            metadata=None,
            delta=SimpleNamespace(type="text", text="Hello"),
        ),
        SimpleNamespace(
            event_type="step.delta",
            metadata=None,
            delta=SimpleNamespace(type="text", text=" world"),
        ),
        SimpleNamespace(
            event_type="interaction.completed",
            interaction=SimpleNamespace(
                model="gemini-3.1-pro-preview",
                usage=SimpleNamespace(total_input_tokens=12, total_output_tokens=34),
            ),
        ),
    ]
