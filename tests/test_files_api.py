"""Tests for the shared Gemini Files API upload used by link-media ingestion."""

import io
from types import SimpleNamespace
from typing import cast
import asyncio
from pathlib import Path

from google import genai
from google.genai.types import FileState

from discordbot.cogs._gen_reply.files_api import (
    LINK_MEDIA_UPLOAD_CONCURRENCY,
    upload_to_files_api,
    upload_as_input_file,
)


class _Files:
    """Fake async Files resource recording uploads and driving the PROCESSING poll."""

    def __init__(
        self, processing_rounds: int = 0, final_state: FileState = FileState.ACTIVE
    ) -> None:
        """Initializes the upload record and the processing-to-final schedule."""
        self.uploads: list[tuple[object, str, str]] = []
        self.get_calls = 0
        self.processing_rounds = processing_rounds
        self.final_state = final_state
        self._remaining = 0

    def _file(self, state: FileState) -> SimpleNamespace:
        """Builds a fake file object carrying the uri the answer would reference."""
        return SimpleNamespace(name="files/abc", uri="https://files.test/abc", state=state)

    async def upload(self, file: object, config: dict[str, str]) -> SimpleNamespace:
        """Records the upload source and returns a PROCESSING or final file."""
        self.uploads.append((file, config["mime_type"], config["display_name"]))
        self._remaining = self.processing_rounds
        return self._file(FileState.PROCESSING if self.processing_rounds else self.final_state)

    async def get(self, name: str) -> SimpleNamespace:
        """Polls the file, flipping to the final state once the rounds elapse."""
        del name
        self.get_calls += 1
        self._remaining -= 1
        return self._file(FileState.PROCESSING if self._remaining > 0 else self.final_state)


def _client(files: _Files) -> genai.Client:
    """Wraps a fake Files resource in the client shape the helper reaches through."""
    return cast("genai.Client", SimpleNamespace(aio=SimpleNamespace(files=files)))


async def test_upload_returns_the_active_uri() -> None:
    """A file that is ACTIVE straight away yields its full uri."""
    files = _Files()
    uri = await upload_to_files_api(
        client=_client(files),
        source=b"data",
        mime_type="video/mp4",
        display_name="clip.mp4",
        timeout_seconds=5.0,
    )
    assert uri == "https://files.test/abc"
    uploaded_source, mime_type, display_name = files.uploads[0]
    assert (mime_type, display_name) == ("video/mp4", "clip.mp4")
    # Bytes are wrapped in a stream because the SDK's `file` parameter takes no raw bytes.
    assert isinstance(uploaded_source, io.BytesIO)
    assert uploaded_source.getvalue() == b"data"


async def test_upload_streams_from_a_path_without_reading_it(tmp_path: Path) -> None:
    """A path source is handed to the SDK as-is, so a large clip is never read into memory."""
    files = _Files()
    path = tmp_path / "clip.mp4"
    await upload_to_files_api(
        client=_client(files),
        source=path,
        mime_type="video/mp4",
        display_name="clip.mp4",
        timeout_seconds=5.0,
    )
    assert files.uploads[0][0] is path


async def test_upload_polls_until_active() -> None:
    """A file still PROCESSING is polled until it flips to ACTIVE."""
    files = _Files(processing_rounds=2)
    uri = await upload_to_files_api(
        client=_client(files),
        source=b"data",
        mime_type="video/mp4",
        display_name="clip.mp4",
        timeout_seconds=30.0,
    )
    assert uri == "https://files.test/abc"
    assert files.get_calls == 2


async def test_upload_gives_up_when_activation_exceeds_the_bound() -> None:
    """A file that never leaves PROCESSING degrades to None rather than hanging or raising."""
    files = _Files(processing_rounds=10_000)
    uri = await upload_to_files_api(
        client=_client(files),
        source=b"data",
        mime_type="video/mp4",
        display_name="clip.mp4",
        timeout_seconds=0.0,
    )
    assert uri is None


async def test_the_bound_covers_the_transfer_not_only_the_poll() -> None:
    """A hung upload must give up and free its slot, not wedge it for the process lifetime.

    google-genai disables the transport timeout by default, so an upload into a black-holed
    connection never returns on its own. Bounding only the PROCESSING poll would let two such
    uploads occupy both concurrency slots forever, after which every link-media build burns its
    whole budget queueing here and silently degrades to text.
    """

    class _Hangs(_Files):
        async def upload(self, file: object, config: dict[str, str]) -> SimpleNamespace:
            """Never returns, the way a black-holed connection behaves."""
            await asyncio.sleep(30)
            raise AssertionError("should have been abandoned")

    uri = await asyncio.wait_for(
        upload_to_files_api(
            client=_client(_Hangs()),
            source=b"data",
            mime_type="video/mp4",
            display_name="clip.mp4",
            timeout_seconds=0.05,
        ),
        timeout=5.0,
    )
    assert uri is None


async def test_a_hung_upload_frees_its_slot_for_the_next_caller() -> None:
    """The slot is shared, so a hung upload must not starve everything behind it."""

    class _Hangs(_Files):
        async def upload(self, file: object, config: dict[str, str]) -> SimpleNamespace:
            """Never returns, the way a black-holed connection behaves."""
            await asyncio.sleep(30)
            raise AssertionError("should have been abandoned")

    hung = [
        upload_to_files_api(
            client=_client(_Hangs()),
            source=b"data",
            mime_type="video/mp4",
            display_name=f"hung{index}.mp4",
            timeout_seconds=0.05,
        )
        for index in range(LINK_MEDIA_UPLOAD_CONCURRENCY)
    ]
    healthy = upload_to_files_api(
        client=_client(_Files()),
        source=b"data",
        mime_type="video/mp4",
        display_name="clip.mp4",
        timeout_seconds=5.0,
    )
    results = await asyncio.wait_for(asyncio.gather(*hung, healthy), timeout=10.0)

    assert results[:-1] == [None] * LINK_MEDIA_UPLOAD_CONCURRENCY
    assert results[-1] == "https://files.test/abc"


async def test_upload_degrades_on_a_failed_file() -> None:
    """A terminal non-ACTIVE state degrades to None."""
    files = _Files(final_state=FileState.FAILED)
    uri = await upload_to_files_api(
        client=_client(files),
        source=b"data",
        mime_type="video/mp4",
        display_name="clip.mp4",
        timeout_seconds=5.0,
    )
    assert uri is None


async def test_upload_degrades_when_the_sdk_raises() -> None:
    """The helper is best-effort: an SDK failure returns None instead of raising."""

    class _Boom(_Files):
        async def upload(self, file: object, config: dict[str, str]) -> SimpleNamespace:
            """Fails the upload the way a transport error would."""
            raise RuntimeError("network down")

    uri = await upload_to_files_api(
        client=_client(_Boom()),
        source=b"data",
        mime_type="video/mp4",
        display_name="clip.mp4",
        timeout_seconds=5.0,
    )
    assert uri is None


async def test_input_file_part_carries_the_uri_and_a_real_extension() -> None:
    """The part references the Files uri via file_id and keeps the extension-bearing filename.

    Never `file_url`: the proxy rewrites an http-bearing url into base64 inline data, and the
    native Interactions path classifies the part by the filename's extension.
    """
    part = await upload_as_input_file(
        client=_client(_Files()),
        source=b"data",
        mime_type="video/mp4",
        filename="douyin_123.mp4",
        timeout_seconds=5.0,
    )
    assert part == {
        "type": "input_file",
        "file_id": "https://files.test/abc",
        "filename": "douyin_123.mp4",
    }


async def test_input_file_part_is_none_when_the_upload_fails() -> None:
    """A failed upload produces no part, so the caller degrades to text instead of a bad ref."""
    part = await upload_as_input_file(
        client=_client(_Files(final_state=FileState.FAILED)),
        source=b"data",
        mime_type="video/mp4",
        filename="douyin_123.mp4",
        timeout_seconds=5.0,
    )
    assert part is None
