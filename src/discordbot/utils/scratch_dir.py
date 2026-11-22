"""Per-invocation scratch directory whose teardown logs `OSError` instead of raising.

Used around blocking download workers run in `asyncio.to_thread`, which cannot be cancelled
directly and may keep writing past a caller timeout. Isolating each invocation prevents
concurrent overwrites and system temp leaks: `utils/file_downloads.py::stream_to_file`
stops when its folder is removed, whereas `services/platforms/ytdlp.py::download_with_stop_signal`
stops via a `threading.Event` and bounded join because yt-dlp recreates output directories.

Teardown via `TemporaryDirectory.cleanup()` can race a still-running worker and raise `ENOTEMPTY`.
Propagating that error from `__exit__` would mask a timeout or `CancelledError` (breaking
`asyncio.wait_for` timeout reporting) or discard a completed return value, so teardown logs at
`error` and swallows `OSError`.
"""

from typing import TYPE_CHECKING
import tempfile
import contextlib

import logfire

if TYPE_CHECKING:
    from collections.abc import Generator


@contextlib.contextmanager
def scratch_directory(prefix: str) -> "Generator[str]":
    """Yields a private temp directory path, removing the directory on exit.

    Args:
        prefix (str): Unique per-call-site prefix so leaked temp directories stay attributable.
    """
    holder = tempfile.TemporaryDirectory(prefix=prefix)
    try:
        yield holder.name
    finally:
        try:
            holder.cleanup()
        except OSError as error:
            # Swallow to avoid masking caller errors/results; log at error to surface leaks.
            logfire.error(
                "Could not remove a scratch directory",
                directory=holder.name,
                error_type=type(error).__name__,
                _exc_info=error,
            )
