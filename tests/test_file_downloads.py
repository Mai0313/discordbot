"""Tests for the shared primitive that streams one remote file to disk.

Every downloading platform hands its transfers to `stream_to_file`, so what it promises is pinned
here once: the size cap, the partial file it removes, and the directory it never re-creates. Each
platform's own tests keep only what it decides on top: what is retried, and what a failure is
called.
"""

import shutil
from typing import IO, Self
from pathlib import Path
from collections.abc import Iterator

import pytest
import requests

from discordbot.services.platforms import file_downloads
from discordbot.services.platforms.file_downloads import DownloadTooLargeError, stream_to_file

_URL = "https://cdn.test/clip.mp4"


class _Response:
    """A streamed response carrying a canned body, optionally dying part-way through it."""

    def __init__(
        self, body: bytes = b"", headers: dict[str, str] | None = None, stall: bool = False
    ) -> None:
        """Stores the body, its headers, and whether the transfer dies after the first chunk."""
        self.body = body
        self.headers = headers or {}
        self.stall = stall
        self.body_reads = 0

    def raise_for_status(self) -> None:
        """Accepts the transfer."""

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Yields the body, counting the read, then fails if this transfer stalls."""
        del chunk_size
        self.body_reads += 1
        yield self.body
        if self.stall:
            raise requests.ConnectionError("read timed out")

    def close(self) -> None:
        """Releases the connection, as a refusal on the header does."""


def _serve(monkeypatch: pytest.MonkeyPatch, response: _Response) -> list[str]:
    """Answers every GET with `response`, returning each URL asked for."""
    requested: list[str] = []

    class _Session:
        """A session answering with the canned response."""

        def __enter__(self) -> Self:
            """Enters the session context."""
            return self

        def __exit__(self, *_: object) -> None:
            """Leaves the session context."""

        def get(self, url: str, **kwargs: object) -> _Response:
            """Records the URL and answers."""
            del kwargs
            requested.append(url)
            return response

    monkeypatch.setattr(target=file_downloads.requests, name="Session", value=_Session)
    return requested


def _stream(filepath: Path, max_bytes: int | None = None) -> Path:
    """Streams `_URL` into `filepath` under the given cap."""
    return stream_to_file(
        url=_URL, filepath=filepath, headers={}, timeout=1.0, max_bytes=max_bytes
    )


def test_a_file_under_the_cap_is_written_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file inside the cap downloads exactly as it does with no cap at all."""
    _serve(
        monkeypatch=monkeypatch,
        response=_Response(body=b"video-bytes", headers={"Content-Length": "11"}),
    )

    written = _stream(filepath=tmp_path / "clip.mp4", max_bytes=1024)

    assert written.read_bytes() == b"video-bytes"


def test_an_oversize_content_length_is_refused_before_the_body_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guard exists to spend a couple of seconds instead of a whole time budget.

    So the raise alone proves nothing: the body must never have been read, and no file opened.
    """
    response = _Response(body=b"x" * 100, headers={"Content-Length": "100"})
    _serve(monkeypatch=monkeypatch, response=response)

    with pytest.raises(DownloadTooLargeError):
        _stream(filepath=tmp_path / "clip.mp4", max_bytes=10)

    assert response.body_reads == 0
    assert list(tmp_path.iterdir()) == []


def test_an_oversize_stream_without_a_content_length_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing or lying Content-Length is caught mid-stream, and the partial file is removed."""
    _serve(monkeypatch=monkeypatch, response=_Response(body=b"x" * 100))

    with pytest.raises(DownloadTooLargeError):
        _stream(filepath=tmp_path / "clip.mp4", max_bytes=10)

    assert list(tmp_path.iterdir()) == []


def test_a_transfer_dying_mid_body_leaves_no_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A later `stat()` would report the truncated file as a finished download.

    The transfer dies after a chunk was written, so a partial file really was on disk; one that
    failed before the first chunk would pass whether or not the cleanup exists.
    """
    _serve(monkeypatch=monkeypatch, response=_Response(body=b"half-a-video", stall=True))

    with pytest.raises(requests.ConnectionError):
        _stream(filepath=tmp_path / "clip.mp4")

    assert list(tmp_path.iterdir()) == []


def test_a_local_write_failure_leaves_no_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A full disk fails the write after some bytes landed, and the partial file goes with it."""
    _serve(monkeypatch=monkeypatch, response=_Response(body=b"video-bytes"))
    real_open = Path.open

    def failing_open(self: Path, mode: str = "r") -> IO[bytes]:
        """Writes a partial file and then fails, as a full disk would.

        The primitive only ever opens with a positional mode, so the stub mirrors that shape.
        """
        handle: IO[bytes] = real_open(self, mode)
        original_write = handle.write

        def write(data: bytes) -> int:
            original_write(data)
            raise OSError(28, "No space left on device")

        # Simulate a mid-write disk failure by shadowing the handle's bound write.
        monkeypatch.setattr(target=handle, name="write", value=write)
        return handle

    monkeypatch.setattr(target=Path, name="open", value=failing_open)

    with pytest.raises(OSError, match="No space left"):
        _stream(filepath=tmp_path / "clip.mp4")

    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_a_removed_folder_is_never_recreated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller that gave up removes the folder, and that removal is its only stop signal.

    It has to hold in the gap BETWEEN two files as well, or a post carrying a second clip would
    rebuild the directory it was just stopped with and download straight through the give-up.
    """
    _serve(monkeypatch=monkeypatch, response=_Response(body=b"clip"))
    scratch = tmp_path / "gone"  # the scratch dir a caller that gave up has already removed

    for name in ("clip.mp4", "clip2.mp4"):
        with pytest.raises(FileNotFoundError):
            _stream(filepath=scratch / name)

        assert not scratch.exists()


def test_a_folder_removed_mid_transfer_stops_the_file_being_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An open handle keeps taking writes after the removal, so the next chunk has to stop it.

    Failing the next open stops only the next file, and a lone clip has none: without this the
    abandoned worker would pull the whole clip into a deleted file.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pulled: list[int] = []

    class _RemovedMidStream(_Response):
        """Removes the scratch dir after the first chunk, the way a caller giving up would."""

        def iter_content(self, chunk_size: int) -> Iterator[bytes]:
            """Yields five chunks, recording each one pulled."""
            for index in range(5):
                if index == 1:
                    shutil.rmtree(path=scratch)
                pulled.append(index)
                yield b"x" * chunk_size

    _serve(monkeypatch=monkeypatch, response=_RemovedMidStream())

    with pytest.raises(FileNotFoundError):
        _stream(filepath=scratch / "clip.mp4")

    assert pulled == [0, 1]  # nothing read past the chunk that arrived after the removal
