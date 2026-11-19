"""HTTP file download helper; failure classification is left to callers."""

import os
from pathlib import Path

import requests

_CHUNK_BYTES = 1 << 16


class DownloadTooLargeError(RuntimeError):
    """The media exceeds the caller's cap; deterministic and not retryable."""


def _reject_oversize_header(response: requests.Response, url: str, max_bytes: int | None) -> None:
    """Refuses an oversize transfer from its `Content-Length`, closing `response` before reading."""
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

    Raises:
        DownloadTooLargeError: If the media exceeds `max_bytes`.
        requests.RequestException: If the transfer fails.
        OSError: If the file cannot be written, including `FileNotFoundError` when the parent
            directory was removed mid-download.
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
                    # A removed scratch dir unlinks this file without failing its writes.
                    if os.fstat(fd=handle.fileno()).st_nlink == 0:
                        raise FileNotFoundError(f"{filepath} was removed mid-download")
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        raise DownloadTooLargeError(f"Media at {url} exceeds {max_bytes} bytes")
                    handle.write(chunk)
        return filepath
    except Exception:
        filepath.unlink(missing_ok=True)
        raise
