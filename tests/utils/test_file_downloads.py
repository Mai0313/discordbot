"""Tests for the shared primitive that streams one remote file to disk.

What `stream_to_file` promises a caller is pinned here once: the size cap, empty-body handling,
connection closure, the partial file it removes, and the directory it never re-creates. A
caller's own tests keep only what it decides on top: what is retried, and what a failure is
called.
"""

import shutil
from typing import IO
from pathlib import Path
from collections.abc import Iterator

import pytest
import requests

from discordbot.utils import file_downloads
from discordbot.utils.file_downloads import DownloadTooLargeError, stream_to_file

from tests.helpers.http import StreamedResponse, make_response, install_session

_URL = "https://cdn.test/clip.mp4"


def _serve(monkeypatch: pytest.MonkeyPatch, response: StreamedResponse) -> list[dict[str, object]]:
    """Answers every GET with `response`, returning each captured call."""
    return install_session(
        monkeypatch=monkeypatch, target=file_downloads.requests, response=response
    )


def _stream(filepath: Path, max_bytes: int | None = None, allow_empty: bool = True) -> Path:
    """Streams `_URL` into `filepath` under the given cap and empty-body policy."""
    return stream_to_file(
        url=_URL,
        filepath=filepath,
        headers={},
        timeout=1.0,
        max_bytes=max_bytes,
        allow_empty=allow_empty,
    )


def test_a_file_under_the_cap_is_written_whole_and_closes_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file inside the cap downloads whole and releases its streamed HTTP response."""
    response = make_response(body=b"video-bytes", headers={"Content-Length": "11"})
    _serve(monkeypatch=monkeypatch, response=response)

    written = _stream(filepath=tmp_path / "clip.mp4", max_bytes=1024)

    assert written.read_bytes() == b"video-bytes"
    assert response.closed


def test_an_oversize_content_length_is_refused_before_the_body_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guard exists to spend a couple of seconds instead of a whole time budget.

    So the raise alone proves nothing: the body must never have been read, no file opened, and
    the response closed.
    """
    response = make_response(body=b"x" * 100, headers={"Content-Length": "100"})
    _serve(monkeypatch=monkeypatch, response=response)

    with pytest.raises(DownloadTooLargeError):
        _stream(filepath=tmp_path / "clip.mp4", max_bytes=10)

    assert response.body_reads == 0
    assert response.closed
    assert list(tmp_path.iterdir()) == []


def test_non_decimal_content_length_falls_through_to_stream_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unicode digits like `'²'` pass `.isdigit()` but not `int()`; `.isdecimal()` skips them."""
    _serve(
        monkeypatch=monkeypatch,
        response=make_response(body=b"ok", headers={"Content-Length": "\u00b2"}),
    )

    written = _stream(filepath=tmp_path / "clip.mp4", max_bytes=10)

    assert written.read_bytes() == b"ok"


def test_an_oversize_stream_without_a_content_length_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing or lying Content-Length is caught mid-stream, and the partial file is removed."""
    response = make_response(body=b"x" * 100)
    _serve(monkeypatch=monkeypatch, response=response)

    with pytest.raises(DownloadTooLargeError):
        _stream(filepath=tmp_path / "clip.mp4", max_bytes=10)

    assert response.closed
    assert list(tmp_path.iterdir()) == []


def test_empty_body_behavior_follows_allow_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty bodies succeed when `allow_empty=True` and raise `ValueError` with cleanup when False."""
    _serve(monkeypatch=monkeypatch, response=make_response(body=b""))

    allowed = _stream(filepath=tmp_path / "allowed.mp4", max_bytes=100, allow_empty=True)
    assert allowed.read_bytes() == b""

    refused = tmp_path / "refused.mp4"
    with pytest.raises(ValueError, match="empty"):
        _stream(filepath=refused, max_bytes=100, allow_empty=False)
    assert not refused.exists()


def test_a_transfer_dying_mid_body_leaves_no_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A later `stat()` would report the truncated file as a finished download.

    The transfer dies after a chunk was written, so a partial file really was on disk; one that
    failed before the first chunk would pass whether or not the cleanup exists.
    """
    response = make_response(body=b"half-a-video", stall=True)
    _serve(monkeypatch=monkeypatch, response=response)

    with pytest.raises(requests.ConnectionError):
        _stream(filepath=tmp_path / "clip.mp4")

    assert response.closed
    assert list(tmp_path.iterdir()) == []


def test_a_local_write_failure_leaves_no_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A full disk fails the write after some bytes landed, and the partial file goes with it."""
    _serve(monkeypatch=monkeypatch, response=make_response(body=b"video-bytes"))
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
    _serve(monkeypatch=monkeypatch, response=make_response(body=b"clip"))
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

    def removed_mid_stream(chunk_size: int) -> Iterator[bytes]:
        """Yields five chunks, removing the scratch dir after the first."""
        for index in range(5):
            if index == 1:
                shutil.rmtree(path=scratch)
            pulled.append(index)
            yield b"x" * chunk_size

    _serve(monkeypatch=monkeypatch, response=make_response(chunks=removed_mid_stream))

    with pytest.raises(FileNotFoundError):
        _stream(filepath=scratch / "clip.mp4")

    assert pulled == [0, 1]  # nothing read past the chunk that arrived after the removal
