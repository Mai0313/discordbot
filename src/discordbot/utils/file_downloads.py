"""HTTP file download helper; failure classification is left to callers."""

import os
from pathlib import Path

import requests

_CHUNK_BYTES = 1 << 16


class DownloadTooLargeError(RuntimeError):
    """The media exceeds the caller's cap; deterministic and not retryable."""


def _reject_oversize_header(response: requests.Response, url: str, max_bytes: int | None) -> None:
    """Refuses an oversize transfer from its `Content-Length` before reading the body."""
    if max_bytes is None:
        return
    declared = response.headers.get("Content-Length")
    if declared is None or not declared.isdecimal() or int(declared) <= max_bytes:
        return
    raise DownloadTooLargeError(
        f"Media at {url} declares {declared} bytes, over the {max_bytes} byte cap"
    )


def _write_chunks(
    response: requests.Response, filepath: Path, url: str, max_bytes: int | None
) -> int:
    """Streams `response` chunks into `filepath`, returning bytes written under `max_bytes`."""
    written = 0
    with filepath.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
            if not chunk:
                continue
            # A removed scratch dir unlinks this file without failing its writes.
            if os.fstat(fd=handle.fileno()).st_nlink == 0:
                raise FileNotFoundError(f"{filepath} was removed mid-download")
            written += len(chunk)
            if max_bytes is not None and written > max_bytes:
                raise DownloadTooLargeError(f"Media at {url} exceeds {max_bytes} bytes")
            handle.write(chunk)
    return written


def stream_to_file(  # noqa: PLR0913 -- url, path, headers, timeout, plus optional size/empty guards
    url: str,
    filepath: Path,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int | None = None,
    allow_empty: bool = True,
) -> Path:
    """Streams `url` into `filepath` and returns `filepath`, enforcing `max_bytes`.

    Does not create the parent directory: callers remove the scratch directory to abort an
    abandoned `asyncio.to_thread` download. Checks `max_bytes` against `Content-Length` before
    reading and again while streaming, unlinking any partial file on failure.

    Args:
        url (str): The media URL.
        filepath (Path): Destination file path; its parent directory must already exist.
        headers (dict[str, str]): Request headers the source expects.
        timeout (float): Per-read timeout in seconds.
        max_bytes (int | None): Maximum allowed size in bytes; `None` disables the cap.
        allow_empty (bool): Whether a zero-byte body is accepted; when False, raises `ValueError`.

    Raises:
        DownloadTooLargeError: If the media exceeds `max_bytes`.
        ValueError: If `allow_empty` is False and the server returns an empty body.
        requests.RequestException: If the transfer fails.
        OSError: If the file cannot be written, including `FileNotFoundError` when the parent
            directory was removed mid-download.
    """
    try:
        with (
            requests.Session() as session,
            session.get(url=url, headers=headers, timeout=timeout, stream=True) as response,
        ):
            response.raise_for_status()
            _reject_oversize_header(response=response, url=url, max_bytes=max_bytes)
            written = _write_chunks(
                response=response, filepath=filepath, url=url, max_bytes=max_bytes
            )
        if not allow_empty and written == 0:
            raise ValueError(f"Media at {url} returned an empty body")
        return filepath
    except Exception:
        filepath.unlink(missing_ok=True)
        raise
