"""Shared primitive for pulling a file down over HTTP.

`stream_to_file` writes one remote file and leaves the failure POLICY to its caller, since what
a failed transfer means (worth retrying, or worth reporting) differs per source.
"""

from pathlib import Path

import requests

# How much is read off the socket at a time. A buffer size, not a policy.
_CHUNK_BYTES = 1 << 16


class DownloadTooLargeError(RuntimeError):
    """The media exceeds the caller's cap. Deterministic, so it is never worth retrying."""


def _reject_oversize_header(response: requests.Response, url: str, max_bytes: int | None) -> None:
    """Refuses an oversize transfer from its `Content-Length`, before a byte is written.

    Closing the response here is the whole point of the guard: the body is never read, so no
    file is opened and nothing lands on disk.

    Raises:
        DownloadTooLargeError: If the declared length exceeds `max_bytes`.
    """
    if max_bytes is None:
        return
    declared = response.headers.get("Content-Length")
    if declared is None or not declared.isdigit() or int(declared) <= max_bytes:
        return
    response.close()
    raise DownloadTooLargeError(
        f"Media at {url} declares {declared} bytes, over the {max_bytes} byte cap"
    )


def stream_to_file(
    url: str, filepath: Path, headers: dict[str, str], timeout: float, max_bytes: int | None = None
) -> Path:
    """Streams a remote file into `filepath`, refusing anything past `max_bytes`.

    Deliberately does NOT create the parent directory: whoever hands one over makes it once, up
    front. A caller that gives up mid-download cannot stop the worker thread (`asyncio.to_thread`
    abandons it), so it removes the scratch dir instead and the open below fails. A `mkdir` here
    would undo that between two files and quietly rebuild a directory nobody will clean up.

    `max_bytes` is a fail-fast guard, not a policy: it exists so a caller whose downstream would
    reject the file anyway (the Files API caps a single upload at 2 GB) finds out from the
    `Content-Length` in a couple of seconds instead of spending its whole time budget fetching
    bytes nobody can use. The streamed re-check backs it up, since `Content-Length` can be absent
    or wrong.

    A partial file is removed on any failure, since leaving it would let a later `stat()` report
    a truncated download as a finished one.

    Args:
        url: The media URL.
        filepath: Where to write the file; its parent directory must already exist.
        headers: Request headers the source expects.
        timeout: Per-read timeout in seconds.
        max_bytes: Refuse media larger than this; None accepts any size.

    Returns:
        The path written, which is `filepath`.

    Raises:
        DownloadTooLargeError: If the media exceeds `max_bytes`.
        requests.RequestException: If the transfer fails; whether that is worth retrying is the
            caller's call.
        OSError: If the file cannot be written, `FileNotFoundError` for a removed directory
            included.
    """
    try:
        with requests.Session() as session:
            response = session.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            _reject_oversize_header(response=response, url=url, max_bytes=max_bytes)
            written = 0
            with filepath.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        raise DownloadTooLargeError(f"Media at {url} exceeds {max_bytes} bytes")
                    handle.write(chunk)
        return filepath
    except Exception:
        filepath.unlink(missing_ok=True)
        raise
