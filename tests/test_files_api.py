"""Tests for the shared Gemini Files API upload of media the bot fetched or generated itself."""

import io
from types import SimpleNamespace
import asyncio
from pathlib import Path

from google import genai
import pytest
from google.genai.types import FileState

from discordbot.cogs.gen_reply.files_api import INPUT_FILE_UPLOAD_CONCURRENCY, upload_as_input_file

from tests.helpers.casting import as_client
from tests.helpers.gen_reply import FakeGeminiFiles, FakeGeminiClient


def _client(files: FakeGeminiFiles) -> genai.Client:
    """Wraps a fake Files resource in the client shape the helper reaches through."""
    return as_client(fake=FakeGeminiClient(files=files))


async def _uploaded_uri(
    files: FakeGeminiFiles,
    filename: str = "clip.mp4",
    source: Path | bytes = b"data",
    timeout_seconds: float = 5.0,
) -> str | None:
    """Uploads through the helper and returns the uri its part references, or None."""
    part = await upload_as_input_file(
        client=_client(files=files),
        source=source,
        mime_type="video/mp4",
        filename=filename,
        timeout_seconds=timeout_seconds,
    )
    return None if part is None else part.get("file_id")


async def test_upload_returns_the_active_uri() -> None:
    """A file that is ACTIVE straight away yields its full uri."""
    files = FakeGeminiFiles()
    uri = await _uploaded_uri(files=files)
    assert uri == "https://files.test/clip.mp4"
    assert files.upload_calls == [("clip.mp4", "video/mp4")]
    # Bytes are wrapped in a stream because the SDK's `file` parameter takes no raw bytes.
    (uploaded_source,) = files.uploaded_sources
    assert isinstance(uploaded_source, io.BytesIO)
    assert uploaded_source.getvalue() == b"data"


async def test_upload_streams_from_a_path_without_reading_it(tmp_path: Path) -> None:
    """A path source is handed to the SDK as-is, so a large clip is never read into memory."""
    files = FakeGeminiFiles()
    path = tmp_path / "clip.mp4"
    await _uploaded_uri(files=files, source=path)
    assert files.uploaded_sources[0] is path


async def test_upload_polls_until_active() -> None:
    """A file still PROCESSING is polled until it flips to ACTIVE."""
    files = FakeGeminiFiles(processing_rounds=2)
    uri = await _uploaded_uri(files=files, timeout_seconds=30.0)
    assert uri == "https://files.test/clip.mp4"
    assert files.get_calls == 2


async def test_upload_gives_up_when_activation_exceeds_the_bound() -> None:
    """A file that never leaves PROCESSING degrades to None rather than hanging or raising."""
    files = FakeGeminiFiles(processing_rounds=10_000)
    uri = await _uploaded_uri(files=files, timeout_seconds=0.0)
    assert uri is None


async def test_a_hung_upload_frees_its_slot_for_the_next_caller() -> None:
    """The slot is shared, so a hung upload must not starve everything behind it.

    google-genai disables the transport timeout by default, so an upload into a black-holed
    connection never returns on its own; only a bound covering the transfer, not just the
    PROCESSING poll, gives its slot back.
    """

    class _Hangs(FakeGeminiFiles):
        async def upload(self, file: object, config: dict[str, str]) -> SimpleNamespace:
            """Never returns, the way a black-holed connection behaves."""
            await asyncio.sleep(30)
            raise AssertionError("should have been abandoned")

    hung = [
        _uploaded_uri(files=_Hangs(), filename=f"hung{index}.mp4", timeout_seconds=0.05)
        for index in range(INPUT_FILE_UPLOAD_CONCURRENCY)
    ]
    healthy = _uploaded_uri(files=FakeGeminiFiles())
    results = await asyncio.wait_for(asyncio.gather(*hung, healthy), timeout=10.0)

    assert results[:-1] == [None] * INPUT_FILE_UPLOAD_CONCURRENCY
    assert results[-1] == "https://files.test/clip.mp4"


async def test_a_queued_upload_is_timed_from_its_slot_not_from_its_call() -> None:
    """Waiting for a slot spends none of an upload's own bound.

    The bound is on the transfer. A queue behind long but healthy uploads says nothing about this
    one's connection, so timing it from the call would give up on an upload that works.
    """
    release = asyncio.Event()

    class _Slow(FakeGeminiFiles):
        async def upload(self, file: object, config: dict[str, str]) -> SimpleNamespace:
            """Holds its slot until released, then succeeds."""
            await release.wait()
            return await super().upload(file=file, config=config)

    holders = [
        asyncio.create_task(_uploaded_uri(files=_Slow(), filename=f"slow{index}.mp4"))
        for index in range(INPUT_FILE_UPLOAD_CONCURRENCY)
    ]
    queued = asyncio.create_task(
        _uploaded_uri(files=FakeGeminiFiles(), filename="queued.mp4", timeout_seconds=0.05)
    )
    # Longer than the queued upload's own bound, all of it spent waiting for a slot.
    await asyncio.sleep(0.2)
    release.set()
    results = await asyncio.wait_for(asyncio.gather(*holders, queued), timeout=5.0)

    assert results == [
        *(f"https://files.test/slow{index}.mp4" for index in range(INPUT_FILE_UPLOAD_CONCURRENCY)),
        "https://files.test/queued.mp4",
    ]


async def test_upload_degrades_on_a_failed_file() -> None:
    """A terminal non-ACTIVE state degrades to None."""
    files = FakeGeminiFiles(final_state=FileState.FAILED)
    uri = await _uploaded_uri(files=files)
    assert uri is None


async def test_upload_degrades_when_the_sdk_raises() -> None:
    """The helper is best-effort: an SDK failure returns None instead of raising."""

    class _Boom(FakeGeminiFiles):
        async def upload(self, file: object, config: dict[str, str]) -> SimpleNamespace:
            """Fails the upload the way a transport error would."""
            raise RuntimeError("network down")

    uri = await _uploaded_uri(files=_Boom())
    assert uri is None


async def test_input_file_part_carries_the_uri_and_a_real_extension() -> None:
    """The part references the Files uri via file_id and keeps the extension-bearing filename.

    Never `file_url`: the proxy rewrites an http-bearing url into base64 inline data, and the
    native Interactions path classifies the part by the filename's extension.
    """
    part = await upload_as_input_file(
        client=_client(FakeGeminiFiles()),
        source=b"data",
        mime_type="video/mp4",
        filename="douyin_123.mp4",
        timeout_seconds=5.0,
    )
    assert part == {
        "type": "input_file",
        "file_id": "https://files.test/douyin_123.mp4",
        "filename": "douyin_123.mp4",
    }


async def test_the_kill_switch_skips_the_upload_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switched off, the transfer never starts, so an outage costs no upload either."""
    monkeypatch.setenv(name="FILE_API_ENABLED", value="false")
    files = FakeGeminiFiles()
    uri = await _uploaded_uri(files=files)
    assert uri is None
    assert files.upload_calls == []
